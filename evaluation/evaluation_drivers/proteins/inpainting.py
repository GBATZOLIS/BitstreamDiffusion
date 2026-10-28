"""Complete partially observed protein sequences and structures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# Make the repository root importable so the module works both as a
# package (import evaluation.evaluation_drivers.proteins.inpainting) and
# as a direct script
# (python evaluation/evaluation_drivers/proteins/inpainting.py). Idempotent.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.protein_multimodal import CANONICAL_AA
from data.protein_structure_codec import (
    DEFAULT_STRUCT_CODEC,
    PATCH_BITS_PER_RESIDUE,
    SEQ_BITS_PER_RESIDUE,
    STRUCT_BITS_PER_RESIDUE,
)
from data.protein_tasks import PARTIAL_PAIR
from evaluation.evaluation_drivers.proteins.io import (
    atomic_evaluation_json,
    input_identities,
    repository_commit,
)
from evaluation.evaluation_drivers.proteins.sharding import (
    ShardWriter,
    add_shard_args,
    finalise,
    partition,
    resolve_shard,
    unit_keys,
    unit_seed,
)

__all__ = [
    "DeletionRegime",
    "DELETION_REGIMES",
    "SDEDIT_REGIME",
    "InpaintingProblem",
    "InpaintingResult",
    "NATIVE_CA_CA_DISTANCE",
    "SEAM_CLASH_RADIUS",
    "build_deletion_mask",
    "build_problems",
    "evaluate_problem",
    "aggregate_records",
    "run_inpainting",
    "shard_problems",
    "load_completed_keys",
    "main",
]


# Index of the alpha carbon in the [L, 4, 3] backbone layout (N, CA, C,
# O).
_ATOM_CA = 1

# Native trans-peptide CA-CA distance in Angstroms. Junction continuity
# is reported as the raw distance and as its deviation from this value,
# so a broken seam (a completion that does not connect to the context)
# is visible directly.
NATIVE_CA_CA_DISTANCE = 3.80

# Heavy-atom clash radius in Angstroms, matching the CA-CA clash
# threshold used by structure_metrics.backbone_geometry.
SEAM_CLASH_RADIUS = 2.0

# Residue separations at or below this are chemically bonded across a
# seam (the peptide C-N bond is 1.329 A and N-CA is 1.458 A), so
# counting them as clashes would report a nonzero clash count for a
# flawless completion.
_CLASH_MIN_SEPARATION = 2

# lDDT inclusion radius and tolerance thresholds, matching
# structure_metrics.lddt so the region-restricted variant is comparable.
_LDDT_CUTOFF = 15.0
_LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class DeletionRegime:
    """One way of choosing which residues the model has to reconstruct."""

    name: str
    kind: str
    fraction: float
    sdedit: bool = False

    def task_for(self, modality: str) -> str:
        """Generation task implementing this regime for one modality."""
        if self.sdedit:
            # structure_denoise is deliberately not a training task, and
            # is not in generate()'s KNOWN_TASKS. data/protein_tasks.py
            # records the reason: its state grid would be identical to
            # structure_marginal (sequence ABSENT, structure NOISY), and
            # SDEdit refinement needs no training of its own because
            # ordinary diffusion training already supervises every noise
            # level. So the regime is exactly structure_marginal driven
            # from a corrupted native payload.
            return "structure_marginal"
        return (
            "sequence_inpaint"
            if modality == "sequence"
            else "structure_inpaint"
        )


# The deletion regimes, as a table rather than as numbers scattered
# through the driver: the contiguous ladder measures how completion
# degrades as the context shrinks, the loop regime is the realistic case
# (loops are what is missing from real experimental models), and the
# terminus regime is the one-sided case where the completion has a seam
# on one side only.
DELETION_REGIMES: tuple[DeletionRegime, ...] = (
    DeletionRegime("contiguous_10", "contiguous", 0.10),
    DeletionRegime("contiguous_25", "contiguous", 0.25),
    DeletionRegime("contiguous_50", "contiguous", 0.50),
    DeletionRegime("contiguous_75", "contiguous", 0.75),
    DeletionRegime("loop", "loop", float("nan")),
    DeletionRegime("terminus_25", "terminus", 0.25),
)

# Only reachable with --sigma-start; appended to the regime list there.
# The whole chain is the "deleted" region, so there is no observed
# context: the junction and seam-clash metrics are undefined (reported
# as null) and the region superposition degenerates to a global one. Its
# input contains the native at noise level sigma_start, so its numbers
# are NOT comparable to the deletion regimes above -- it measures
# refinement fidelity, not completion.
SDEDIT_REGIME = DeletionRegime("structure_denoise", "whole", 1.0, sdedit=True)

REGIMES_BY_NAME: dict[str, DeletionRegime] = {
    regime.name: regime for regime in (*DELETION_REGIMES, SDEDIT_REGIME)
}

COMPLETION_TASKS: tuple[str, ...] = (
    "sequence_inpaint",
    "structure_inpaint",
    "sequence_redesign",
    "structure_redesign",
    PARTIAL_PAIR,
)


def _task_modality(task: str) -> str:
    if task.startswith("sequence_"):
        return "sequence"
    if task.startswith("structure_"):
        return "structure"
    if task == PARTIAL_PAIR:
        return "pair"
    raise ValueError(f"unsupported completion task {task!r}")


def _task_needs_structure(task: str) -> bool:
    return task != "sequence_inpaint"


# -----------------------------------------------------------------------------
# Deletion masks
# -----------------------------------------------------------------------------


def build_deletion_mask(
    regime: DeletionRegime,
    length: int,
    *,
    coords: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    structure_metrics=None,
) -> np.ndarray:
    """[L] bool mask, True where the model must generate the residue."""
    L = int(length)
    mask = np.zeros(L, dtype=bool)
    if L <= 0:
        return mask
    rng = rng if rng is not None else np.random.default_rng(0)

    if regime.kind == "whole":
        mask[:] = True
        return mask

    if regime.kind == "loop":
        if coords is None:
            raise ValueError(
                "the loop deletion regime needs backbone coordinates to "
                "assign secondary structure"
            )
        sm = (
            structure_metrics
            if structure_metrics is not None
            else _load_structure_metrics()
        )
        labels = sm.secondary_structure_labels(np.asarray(coords))
        mask = np.asarray([label == "coil" for label in labels], dtype=bool)
        if mask.all():
            # A chain the approximate assignment calls all-coil would
            # leave no context at all, which is a different task; drop
            # it rather than silently evaluating an unconditional
            # generation.
            raise ValueError(
                "loop regime found no non-coil context on this chain"
            )
        return mask

    span = int(round(float(regime.fraction) * L))
    span = max(1, min(L - 1, span))
    if regime.kind == "terminus":
        if bool(rng.integers(0, 2)):
            mask[L - span :] = True
        else:
            mask[:span] = True
        return mask
    if regime.kind == "contiguous":
        start = int(rng.integers(0, L - span + 1))
        mask[start : start + span] = True
        return mask
    raise ValueError(f"unknown deletion regime kind {regime.kind!r}")


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of True in a boolean mask."""
    flags = np.asarray(mask, dtype=bool)
    if flags.size == 0:
        return []
    padded = np.concatenate([[False], flags, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)
    ]


# -----------------------------------------------------------------------------
# Lazy dependency loaders (kept out of import time)
# -----------------------------------------------------------------------------


def _load_generate():
    """Import the clamped multimodal generation driver, guarded and lazily."""
    try:
        from evaluation.evaluation_drivers.proteins import generate_multimodal
    # re-raised with an actionable hint
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Inpainting needs "
            "evaluation.evaluation_drivers.proteins."
            "generate_multimodal.generate to sample "
            "clamped completions. Ensure the protein evaluation package "
            "is importable; see ./setup.sh eval."
        ) from exc
    return generate_multimodal.generate


def _load_structure_metrics():
    """Import the TM-score, RMSD, and secondary-structure judge, lazily."""
    try:
        from evaluation.evaluation_drivers.proteins import structure_metrics
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Inpainting needs evaluation.evaluation_drivers.proteins."
            "structure_metrics for "
            "tm_score, rmsd, and secondary_structure_labels. Ensure the "
            "protein evaluation package is importable; see "
            "./setup.sh eval."
        ) from exc
    return structure_metrics


def _load_tokenizer(tokenizer_path, device: str):
    """Load the frozen DPLM-2 structure tokenizer for decoding, lazily."""
    if tokenizer_path is None:
        raise RuntimeError(
            "Structure inpainting decodes generated structure tokens to "
            "backbone coordinates with the DPLM-2 tokenizer, which needs"
            " a checkpoint path. Pass --struct-tokenizer (download it "
            "with scripts/proteins/setup/download_dplm_tokenizer.py) and"
            " install the DPLM-inference environment from the eval extra"
            " (uv sync --extra eval)."
        )
    from evaluation.evaluation_drivers.proteins.dplm_struct_tokenizer import (
        load_struct_tokenizer,
    )

    return load_struct_tokenizer(tokenizer_path, device=device)


def _load_model(config_path, checkpoint_path, device, num_steps):
    """
    Build the diffusion model and config with EMA weights applied,
    lazily.
    """
    from evaluation.evaluation_drivers.proteins.io import (
        load_binary_protein_model,
    )

    return load_binary_protein_model(
        config_path,
        checkpoint_path,
        device,
        num_steps=num_steps,
        context="structure inpainting",
    )


def _load_backbone_corpus():
    """Import the shared held-out backbone loader, guarded and lazily."""
    try:
        from evaluation.evaluation_drivers.proteins.inverse_folding import (
            _StructureTokenResolver,
            load_backbone_records,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Inpainting reads held-out chains with "
            "evaluation.evaluation_drivers.proteins.inverse_folding."
            "load_backbone_records so the "
            "two evaluations run on the identical corpus. Ensure the protein "
            "evaluation package is importable."
        ) from exc
    return load_backbone_records, _StructureTokenResolver


# -----------------------------------------------------------------------------
# Geometry: context superposition, junctions, seam clashes, region lDDT
# -----------------------------------------------------------------------------


def _ca_track(coords: np.ndarray) -> np.ndarray:
    """[L, 3] alpha-carbon track from an [L, 4, 3] backbone array."""
    array = np.asarray(coords, dtype=np.float64)
    if array.ndim == 3 and array.shape[1] >= 2 and array.shape[2] == 3:
        return np.ascontiguousarray(array[:, _ATOM_CA, :])
    if array.ndim == 2 and array.shape[1] == 3:
        return np.ascontiguousarray(array)
    raise ValueError(
        f"coordinates must be [L, 4, 3] or [L, 3], got {tuple(array.shape)}"
    )


def _kabsch_transform(
    mobile: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation taking mobile onto reference."""
    a = np.ascontiguousarray(np.asarray(mobile, dtype=np.float64))
    b = np.ascontiguousarray(np.asarray(reference, dtype=np.float64))
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(
            f"_kabsch_transform needs matching [N, 3] arrays, got "
            f"{a.shape} and {b.shape}"
        )
    a_centre = a.mean(axis=0)
    b_centre = b.mean(axis=0)
    covariance = (a - a_centre).T @ (b - b_centre)
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rotation, b_centre - a_centre @ rotation.T


def _apply(points: np.ndarray, transform: tuple[np.ndarray, np.ndarray]):
    """Apply a (R, t) transform to an array of points [..., 3]."""
    rotation, translation = transform
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    moved = flat @ rotation.T + translation
    return moved.reshape(np.asarray(points).shape)


def context_superposed_region_rmsd(
    generated: np.ndarray,
    native: np.ndarray,
    deleted: np.ndarray,
    *,
    backbone: bool = False,
) -> tuple[float, str]:
    """Region RMSD after superposing on the OBSERVED context only."""
    deleted = np.asarray(deleted, dtype=bool)
    gen = np.asarray(generated, dtype=np.float64)
    nat = np.asarray(native, dtype=np.float64)
    region = np.flatnonzero(deleted)
    context = np.flatnonzero(~deleted)
    if region.size == 0:
        return float("nan"), "empty region"

    gen_ca, nat_ca = _ca_track(gen), _ca_track(nat)
    if context.size >= 3:
        transform = _kabsch_transform(gen_ca[context], nat_ca[context])
        superposed_on = "observed context CA"
    else:
        transform = _kabsch_transform(gen_ca[region], nat_ca[region])
        superposed_on = "region CA (no observed context)"

    if backbone:
        gen_sel = gen[region].reshape(-1, 3)
        nat_sel = nat[region].reshape(-1, 3)
    else:
        gen_sel = gen_ca[region]
        nat_sel = nat_ca[region]
    difference = _apply(gen_sel, transform) - nat_sel
    value = float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))
    return value, superposed_on


def region_lddt(
    generated: np.ndarray,
    native: np.ndarray,
    deleted: np.ndarray,
    *,
    cutoff: float = _LDDT_CUTOFF,
    thresholds: Sequence[float] = _LDDT_THRESHOLDS,
) -> float:
    """lDDT restricted to pairs with one end inside the deleted region."""
    deleted = np.asarray(deleted, dtype=bool)
    pred = _ca_track(generated)
    true = _ca_track(native)
    if pred.shape != true.shape:
        raise ValueError(
            f"region_lddt needs matching chains, got {pred.shape} and "
            f"{true.shape}"
        )
    length = true.shape[0]
    if length < 2 or not deleted.any():
        return float("nan")
    dist_pred = np.linalg.norm(pred[:, None, :] - pred[None, :, :], axis=-1)
    dist_true = np.linalg.norm(true[:, None, :] - true[None, :, :], axis=-1)
    included = (
        (dist_true < float(cutoff))
        & ~np.eye(length, dtype=bool)
        & deleted[:, None]
    )
    if not included.any():
        return float("nan")
    delta = np.abs(dist_pred - dist_true)
    preserved = np.zeros_like(delta)
    for threshold in thresholds:
        preserved += (delta < float(threshold)).astype(np.float64)
    preserved /= float(len(thresholds))
    return float(preserved[included].mean())


def junction_continuity(
    coords: np.ndarray, deleted: np.ndarray
) -> dict[str, object]:
    """
    CA-CA distances across every seam between context and generated
    residues.
    """
    ca = _ca_track(coords)
    deleted = np.asarray(deleted, dtype=bool)
    length = ca.shape[0]
    seams: list[dict[str, object]] = []
    for start, end in _segments(deleted):
        if start > 0:
            seams.append(
                {
                    "segment": [start, end],
                    "side": "n_terminal",
                    "distance": float(
                        np.linalg.norm(ca[start] - ca[start - 1])
                    ),
                }
            )
        if end < length:
            seams.append(
                {
                    "segment": [start, end],
                    "side": "c_terminal",
                    "distance": float(np.linalg.norm(ca[end] - ca[end - 1])),
                }
            )
    distances = [float(seam["distance"]) for seam in seams]
    deviations = [abs(d - NATIVE_CA_CA_DISTANCE) for d in distances]
    return {
        "seams": seams,
        "num_seams": len(seams),
        "mean_distance": float(np.mean(distances)) if distances else None,
        "max_distance": float(np.max(distances)) if distances else None,
        "mean_abs_deviation": (
            float(np.mean(deviations)) if deviations else None
        ),
        "max_abs_deviation": float(np.max(deviations)) if deviations else None,
        "native_ca_ca": NATIVE_CA_CA_DISTANCE,
    }


def seam_clash_count(
    coords: np.ndarray,
    deleted: np.ndarray,
    *,
    radius: float = SEAM_CLASH_RADIUS,
) -> int | None:
    """Backbone atoms of generated residues too close to observed residues."""
    backbone = np.asarray(coords, dtype=np.float64)
    if backbone.ndim != 3 or backbone.shape[2] != 3:
        raise ValueError(
            f"seam_clash_count needs [L, 4, 3] coordinates, got "
            f"{tuple(backbone.shape)}"
        )
    deleted = np.asarray(deleted, dtype=bool)
    region = np.flatnonzero(deleted)
    context = np.flatnonzero(~deleted)
    if region.size == 0 or context.size == 0:
        return None
    diff = (
        backbone[region][:, None, :, None, :]
        - backbone[context][None, :, None, :, :]
    )  # [R, C, atoms, atoms, 3]
    distance = np.linalg.norm(diff, axis=-1)
    separation = np.abs(region[:, None] - context[None, :])
    close = (distance < float(radius)) & (separation >= _CLASH_MIN_SEPARATION)[
        :, :, None, None
    ]
    return int(close.sum())


# -----------------------------------------------------------------------------
# Bit-level and sequence-level metrics
# -----------------------------------------------------------------------------


def _struct_bits_from_payload(bits: np.ndarray, length: int) -> np.ndarray:
    """[B, L, 13] structure bits from the sampler's [B, S] payload."""
    array = np.asarray(bits).reshape(-1, int(length), PATCH_BITS_PER_RESIDUE)
    return array[:, :, SEQ_BITS_PER_RESIDUE:].astype(np.uint8)


def structure_bit_metrics(
    generated_bits: np.ndarray,
    native_index: np.ndarray,
    deleted: np.ndarray,
) -> dict[str, object]:
    """
    LFQ bit accuracy on the deleted region, plus the clamp fidelity
    check.
    """
    native_bits = DEFAULT_STRUCT_CODEC.index_to_bits_np(
        np.asarray(native_index).reshape(-1)
    ).astype(np.uint8)  # [L, 13]
    gen = np.asarray(generated_bits, dtype=np.uint8)
    if gen.shape[-1] != STRUCT_BITS_PER_RESIDUE:
        raise ValueError(
            f"expected {STRUCT_BITS_PER_RESIDUE} structure bits per "
            f"residue, got {gen.shape[-1]}"
        )
    deleted = np.asarray(deleted, dtype=bool)
    correct = gen == native_bits
    region = float(correct[deleted].mean()) if deleted.any() else float("nan")
    context = float(correct[~deleted].mean()) if (~deleted).any() else None
    exact = correct.all(axis=-1)
    return {
        "region_bit_accuracy": region,
        "region_exact_token_match": (
            float(exact[deleted].mean()) if deleted.any() else float("nan")
        ),
        "context_bit_accuracy": context,
    }


def masked_recovery(
    generated_seq: str, native_seq: str, deleted: np.ndarray
) -> float:
    """Amino-acid recovery over the deleted positions only."""
    deleted = np.asarray(deleted, dtype=bool)
    if len(generated_seq) != len(native_seq):
        raise ValueError(
            f"sequence lengths differ: {len(generated_seq)} and "
            f"{len(native_seq)}"
        )
    positions = np.flatnonzero(deleted)
    if positions.size == 0:
        return float("nan")
    matches = sum(1 for i in positions if generated_seq[i] == native_seq[i])
    return float(matches) / float(positions.size)


def shuffled_target_recovery(
    generated_seq: str,
    native_seq: str,
    deleted: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Recovery against a permuted target: the background rate control."""
    deleted = np.asarray(deleted, dtype=bool)
    positions = np.flatnonzero(deleted)
    if positions.size == 0:
        return float("nan")
    target = [native_seq[i] for i in positions]
    permuted = [target[i] for i in rng.permutation(len(target))]
    matches = sum(
        1
        for position, residue in zip(positions.tolist(), permuted, strict=True)
        if generated_seq[position] == residue
    )
    return float(matches) / float(positions.size)


# -----------------------------------------------------------------------------
# Problems and results
# -----------------------------------------------------------------------------


@dataclass
class InpaintingProblem:
    """One held-out chain under one deletion regime for one modality."""

    stable_id: str
    regime: str
    modality: str
    coords: np.ndarray
    native_seq: str
    struct_index: np.ndarray
    deleted: np.ndarray
    source: str = ""
    generation_task: str = ""
    seq_deleted: np.ndarray | None = None
    struct_deleted: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=np.float64)
        self.struct_index = np.asarray(self.struct_index).reshape(-1)
        self.deleted = np.asarray(self.deleted, dtype=bool).reshape(-1)
        length = int(self.coords.shape[0])
        if (
            len(self.native_seq) != length
            or self.struct_index.shape[0] != length
            or self.deleted.shape[0] != length
        ):
            raise ValueError(
                f"problem {self.key!r} is inconsistent: coords {length}, "
                f"sequence {len(self.native_seq)}, tokens "
                f"{self.struct_index.shape[0]}, mask "
                f"{self.deleted.shape[0]}"
            )
        if not self.generation_task:
            self.generation_task = (
                "sequence_inpaint"
                if self.modality == "sequence"
                else "structure_inpaint"
            )
        if self.generation_task not in (
            *COMPLETION_TASKS,
            "structure_denoise",
        ):
            raise ValueError(
                f"unsupported generation_task {self.generation_task!r}"
            )
        if self.modality not in ("structure", "sequence", "pair"):
            raise ValueError(
                f"modality must be 'structure', 'sequence', or 'pair', got"
                f" {self.modality!r}"
            )
        if self.seq_deleted is None:
            self.seq_deleted = (
                self.deleted.copy()
                if self.modality == "sequence"
                else np.zeros(length, dtype=bool)
            )
        else:
            self.seq_deleted = np.asarray(
                self.seq_deleted, dtype=bool
            ).reshape(-1)
        if self.struct_deleted is None:
            self.struct_deleted = (
                self.deleted.copy()
                if self.modality in ("structure", "pair")
                else np.zeros(length, dtype=bool)
            )
        else:
            self.struct_deleted = np.asarray(
                self.struct_deleted, dtype=bool
            ).reshape(-1)
        if (
            self.seq_deleted.shape[0] != length
            or self.struct_deleted.shape[0] != length
        ):
            raise ValueError(
                "sequence/structure deletion masks must match chain length"
            )

    @property
    def key(self) -> str:
        """Work-unit identity used for sharding and for resume."""
        return f"{self.stable_id}|{self.generation_task}|{self.regime}"

    @property
    def length(self) -> int:
        return int(self.coords.shape[0])

    @property
    def num_deleted(self) -> int:
        return int(self.deleted.sum())


@dataclass
class InpaintingResult:
    """Per-sample metrics for one problem, plus its per-problem context."""

    key: str
    stable_id: str
    regime: str
    modality: str
    generation_task: str
    length: int
    num_deleted: int
    num_segments: int
    samples: list[dict[str, object]] = field(default_factory=list)
    tokenizer_ceiling: dict[str, object] = field(default_factory=dict)
    notes: dict[str, object] = field(default_factory=dict)

    def to_record(self) -> dict[str, object]:
        """
        JSON-serialisable record, one per work unit, written to the
        shard.
        """
        return {
            "key": self.key,
            "stable_id": self.stable_id,
            "regime": self.regime,
            "modality": self.modality,
            "generation_task": self.generation_task,
            "length": self.length,
            "num_deleted": self.num_deleted,
            "deleted_fraction": (
                self.num_deleted / self.length if self.length else 0.0
            ),
            "num_segments": self.num_segments,
            "tokenizer_ceiling": self.tokenizer_ceiling,
            "notes": self.notes,
            "samples": self.samples,
        }


def _stable_seed(seed: int, *parts: str) -> int:
    """
    Process-independent seed from a run seed and some identifying
    strings.
    """
    digest = hashlib.blake2b(
        "\x1f".join((str(int(seed)), *parts)).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % (2**32)


def build_problems(
    records: Sequence[object],
    regimes: Sequence[DeletionRegime],
    modalities: Sequence[str],
    *,
    struct_index_resolver,
    seed: int = 0,
    structure_metrics=None,
    resolve_structure: bool = True,
    tasks: Sequence[str] | None = None,
) -> list[InpaintingProblem]:
    """Expand held-out chains into (chain, modality, regime) work units."""
    selected_tasks = list(tasks or ())
    if not selected_tasks:
        selected_tasks = [
            "sequence_inpaint"
            if modality == "sequence"
            else "structure_inpaint"
            for modality in modalities
        ]
    unknown_tasks = sorted(set(selected_tasks).difference(COMPLETION_TASKS))
    if unknown_tasks:
        raise ValueError(
            f"unsupported completion task(s) {unknown_tasks}; choose from "
            f"{COMPLETION_TASKS}"
        )
    needs_structure = any(
        _task_needs_structure(task) for task in selected_tasks
    )
    problems: list[InpaintingProblem] = []
    for record in records:
        # The 5-bit sequence codec and the PDB writer both cover the
        # canonical 20 only, and a chain carrying X or U survives corpus
        # loading. Rejecting it here costs nothing; rejecting it later
        # costs a completed sampling run, since the failure would land
        # inside evaluate_problem.
        unknown = sorted(set(record.native_seq) - set(CANONICAL_AA))
        if unknown:
            print(
                f"[inpainting] skipping {record.stable_id}: non-canonical "
                f"residue(s) {unknown}"
            )
            continue
        # A sequence-only run must not require the structure tokenizer,
        # so the tokens are resolved only when a structure regime will
        # read them.
        struct_index = (
            np.asarray(struct_index_resolver(record), dtype=np.uint16)
            if needs_structure and resolve_structure
            else np.zeros(record.length, dtype=np.uint16)
        )
        for task in selected_tasks:
            modality = _task_modality(task)
            for regime in regimes:
                if regime.sdedit and task != "structure_inpaint":
                    continue  # the refinement regime is structure-only
                stream = np.random.default_rng(
                    _stable_seed(seed, record.stable_id, regime.name)
                )
                try:
                    deleted = build_deletion_mask(
                        regime,
                        record.length,
                        coords=record.coords,
                        rng=stream,
                        structure_metrics=structure_metrics,
                    )
                except ValueError as exc:
                    print(
                        f"[inpainting] skipping {record.stable_id} regime "
                        f"{regime.name}: {exc}"
                    )
                    continue
                seq_deleted = (
                    deleted.copy()
                    if modality == "sequence"
                    else np.zeros(record.length, dtype=bool)
                )
                struct_deleted = (
                    deleted.copy()
                    if modality == "structure"
                    else np.zeros(record.length, dtype=bool)
                )
                if task == PARTIAL_PAIR:
                    seq_deleted = build_deletion_mask(
                        regime,
                        record.length,
                        coords=record.coords,
                        rng=np.random.default_rng(
                            _stable_seed(
                                seed, record.stable_id, regime.name, "sequence"
                            )
                        ),
                        structure_metrics=structure_metrics,
                    )
                    struct_deleted = build_deletion_mask(
                        regime,
                        record.length,
                        coords=record.coords,
                        rng=np.random.default_rng(
                            _stable_seed(
                                seed,
                                record.stable_id,
                                regime.name,
                                "structure",
                            )
                        ),
                        structure_metrics=structure_metrics,
                    )
                    deleted = np.logical_or(seq_deleted, struct_deleted)
                problems.append(
                    InpaintingProblem(
                        stable_id=record.stable_id,
                        regime=regime.name,
                        modality=modality,
                        coords=record.coords,
                        native_seq=record.native_seq,
                        struct_index=struct_index,
                        deleted=deleted,
                        source=getattr(record, "source", ""),
                        generation_task=(
                            "structure_denoise" if regime.sdedit else task
                        ),
                        seq_deleted=seq_deleted,
                        struct_deleted=struct_deleted,
                    )
                )
    return problems


# -----------------------------------------------------------------------------
# Core evaluation
# -----------------------------------------------------------------------------


def _seq_to_ids(sequence: str) -> np.ndarray:
    """Canonical-20 ids for an amino-acid string, rejecting unknown letters."""
    ids = []
    for index, character in enumerate(sequence):
        position = CANONICAL_AA.find(character)
        if position < 0:
            raise ValueError(
                f"residue {index} is {character!r}, which is not one of "
                f"the canonical 20 ({CANONICAL_AA}); the 5-bit sequence "
                f"codec cannot represent it"
            )
        ids.append(position)
    return np.asarray(ids, dtype=np.int64)


def _observed_for(problem: InpaintingProblem, regime: DeletionRegime) -> dict:
    """Conditioning payload handed to generate for one problem."""
    task = problem.generation_task
    if task == "structure_denoise":
        return {"struct_index": problem.struct_index}
    if task == "sequence_inpaint":
        return {
            "seq_ids": _seq_to_ids(problem.native_seq),
            "seq_observed_mask": ~problem.seq_deleted,
        }
    if task == "sequence_redesign":
        return {
            "seq_ids": _seq_to_ids(problem.native_seq),
            "seq_observed_mask": ~problem.seq_deleted,
            "struct_index": problem.struct_index,
        }
    if task == "structure_inpaint":
        return {
            "struct_index": problem.struct_index,
            "struct_observed_mask": ~problem.struct_deleted,
        }
    if task == "structure_redesign":
        return {
            "seq_ids": _seq_to_ids(problem.native_seq),
            "struct_index": problem.struct_index,
            "struct_observed_mask": ~problem.struct_deleted,
        }
    if task == PARTIAL_PAIR:
        return {
            "seq_ids": _seq_to_ids(problem.native_seq),
            "struct_index": problem.struct_index,
            "seq_observed_mask": ~problem.seq_deleted,
            "struct_observed_mask": ~problem.struct_deleted,
        }
    raise ValueError(f"unsupported completion task {task!r}")


def evaluate_problem(
    problem: InpaintingProblem,
    *,
    generate_fn,
    tokenizer,
    judge,
    model,
    cfg,
    device,
    num_samples: int,
    seed: int = 0,
    guidance_scale: float | None = None,
    sigma_start: float | None = None,
    num_steps: int | None = None,
    pdb_dir: Path | None = None,
) -> InpaintingResult:
    """Complete one problem through the sampler and score every sample."""
    regime = REGIMES_BY_NAME[problem.regime]
    task = (
        "structure_marginal"
        if problem.generation_task == "structure_denoise"
        else problem.generation_task
    )
    if regime.sdedit and sigma_start is None:
        raise ValueError(
            f"regime {regime.name!r} is SDEdit refinement and needs "
            f"--sigma-start"
        )
    generated = generate_fn(
        model,
        cfg,
        task,
        problem.length,
        int(num_samples),
        device,
        observed=_observed_for(problem, regime),
        guidance_scale=guidance_scale,
        sigma_start=sigma_start if regime.sdedit else None,
        num_steps=num_steps,
    )

    result = InpaintingResult(
        key=problem.key,
        stable_id=problem.stable_id,
        regime=problem.regime,
        modality=problem.modality,
        generation_task=problem.generation_task,
        length=problem.length,
        num_deleted=problem.num_deleted,
        num_segments=len(_segments(problem.deleted)),
    )
    if regime.sdedit:
        result.notes["sdedit"] = (
            "input is the native structure corrupted to sigma_start, so "
            "these numbers measure refinement fidelity and are not "
            "comparable to the deletion regimes"
        )
        result.notes["sigma_start"] = float(sigma_start)

    rng = np.random.default_rng(int(seed))
    if problem.modality in ("sequence", "pair"):
        for index, sequence in enumerate(generated["seq_strings"]):
            result.samples.append(
                {
                    "sample": index,
                    "region_recovery": masked_recovery(
                        sequence, problem.native_seq, problem.seq_deleted
                    ),
                    "region_recovery_shuffled_target": (
                        shuffled_target_recovery(
                            sequence,
                            problem.native_seq,
                            problem.seq_deleted,
                            rng,
                        )
                    ),
                    "context_recovery": masked_recovery(
                        sequence, problem.native_seq, ~problem.seq_deleted
                    ),
                }
            )
        result.notes["num_sequence_deleted"] = int(problem.seq_deleted.sum())
        if problem.modality == "sequence":
            return result

    native_coords = np.asarray(problem.coords, dtype=np.float64)
    # The tokenizer ceiling: the same region metrics for a chain whose
    # tokens are the NATIVE ones. The generated backbone is only
    # reachable through this decoder, so this is the floor the
    # completion numbers are read against.
    ceiling_coords = np.asarray(
        tokenizer.decode(problem.struct_index.astype(np.uint16)),
        dtype=np.float64,
    )
    ceiling_rmsd, superposed_on = context_superposed_region_rmsd(
        ceiling_coords, native_coords, problem.struct_deleted
    )
    result.tokenizer_ceiling = {
        "region_ca_rmsd": ceiling_rmsd,
        "region_lddt_vs_context": region_lddt(
            ceiling_coords, native_coords, problem.struct_deleted
        ),
        "tm_score": float(
            judge.tm_score(_ca_track(ceiling_coords), _ca_track(native_coords))
        ),
        "superposed_on": superposed_on,
    }

    struct_bits = _struct_bits_from_payload(
        np.asarray(generated["bits"]), problem.length
    )
    struct_index = np.asarray(generated["struct_index"])
    for index in range(struct_bits.shape[0]):
        design_coords = np.asarray(
            tokenizer.decode(struct_index[index].astype(np.uint16)),
            dtype=np.float64,
        )
        region_rmsd, region_frame = context_superposed_region_rmsd(
            design_coords, native_coords, problem.struct_deleted
        )
        region_bb_rmsd, _ = context_superposed_region_rmsd(
            design_coords, native_coords, problem.struct_deleted, backbone=True
        )
        sample: dict[str, object] = {
            "sample": index,
            "region_ca_rmsd": region_rmsd,
            "region_backbone_rmsd": region_bb_rmsd,
            "superposed_on": region_frame,
            "region_lddt_vs_context": region_lddt(
                design_coords, native_coords, problem.struct_deleted
            ),
            "junctions": junction_continuity(
                design_coords, problem.struct_deleted
            ),
            "seam_clashes": seam_clash_count(
                design_coords, problem.struct_deleted
            ),
            "tm_score": float(
                judge.tm_score(
                    _ca_track(design_coords), _ca_track(native_coords)
                )
            ),
        }
        sample.update(
            structure_bit_metrics(
                struct_bits[index],
                problem.struct_index,
                problem.struct_deleted,
            )
        )
        if problem.modality == "pair":
            result.samples[index].update(sample)
        else:
            result.samples.append(sample)
        if pdb_dir is not None and index == 0:
            _write_completion_pdbs(
                pdb_dir,
                problem,
                design_coords,
                native_coords,
                deleted=problem.struct_deleted,
            )
    result.notes["num_structure_deleted"] = int(problem.struct_deleted.sum())
    return result


def _write_completion_pdbs(
    pdb_dir: Path,
    problem: InpaintingProblem,
    design_coords: np.ndarray,
    native_coords: np.ndarray,
    *,
    deleted: np.ndarray | None = None,
) -> None:
    """Write the native and the completion for one problem, for inspection."""
    from evaluation.evaluation_drivers.proteins.pdb_io import (
        write_backbone_pdb,
    )

    pdb_dir = Path(pdb_dir)
    stem = problem.key.replace("|", "__").replace("/", "_")
    flags = np.asarray(
        problem.deleted if deleted is None else deleted, dtype=np.float64
    )
    write_backbone_pdb(
        pdb_dir / f"{stem}.completion.pdb",
        design_coords,
        problem.native_seq,
        bfactors=flags,
        remarks=[
            (
                f"ProteBit structure inpainting completion, regime "
                f"{problem.regime}"
            ),
            "B-factor 1.00 marks residues generated by the sampler",
        ],
    )
    write_backbone_pdb(
        pdb_dir / f"{stem}.native.pdb",
        native_coords,
        problem.native_seq,
        bfactors=flags,
        remarks=[f"native chain {problem.stable_id}"],
    )


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def _distribution_summary(
    values: Sequence[float | None],
    *,
    include_values: bool = True,
    bootstrap: bool = False,
) -> dict[str, object]:
    """Summarise a metric distribution and publish its raw values."""
    raw: list[float | None] = []
    finite: list[float] = []
    for value in values:
        if value is None:
            raw.append(None)
            continue
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            raw.append(None)
        else:
            raw.append(numeric)
            finite.append(numeric)
    if finite:
        array = np.asarray(finite, dtype=np.float64)
        summary: dict[str, object] = {
            "count": int(array.size),
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
            "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)),
            "p25": float(np.percentile(array, 25)),
            "p75": float(np.percentile(array, 75)),
            "p90": float(np.percentile(array, 90)),
        }
    else:
        summary = {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    if bootstrap:
        from evaluation.evaluation_drivers.proteins.statistics import (
            bootstrap_mean_ci,
        )

        summary["mean_ci95"] = bootstrap_mean_ci(finite)
    if include_values:
        summary["values"] = raw
    return summary


_STRUCTURE_FIELDS = (
    "region_ca_rmsd",
    "region_backbone_rmsd",
    "region_lddt_vs_context",
    "tm_score",
    "region_bit_accuracy",
    "region_exact_token_match",
    "context_bit_accuracy",
    "seam_clashes",
)
_SEQUENCE_FIELDS = (
    "region_recovery",
    "region_recovery_shuffled_target",
    "context_recovery",
)

_METRIC_DIRECTIONS = {
    "region_ca_rmsd": "min",
    "region_backbone_rmsd": "min",
    "region_lddt_vs_context": "max",
    "tm_score": "max",
    "region_bit_accuracy": "max",
    "region_exact_token_match": "max",
    "context_bit_accuracy": "max",
    "seam_clashes": "min",
    "region_recovery": "max",
    "region_recovery_shuffled_target": "max",
    "context_recovery": "max",
}


def _target_metric(
    record: dict[str, object],
    metric: str,
    *,
    budget: int | None = None,
    reduction: str,
) -> float | None:
    """
    Reduce one chain's ordered candidates without pooling target
    identity.
    """
    samples = list(record.get("samples", []))
    if budget is not None:
        samples = samples[: int(budget)]
    if reduction == "first":
        if not samples:
            return None
        value = samples[0].get(metric)
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    values = []
    for sample in samples:
        value = sample.get(metric)
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    if not values:
        return None
    if reduction == "mean":
        return float(np.mean(values))
    if reduction == "best":
        return float(
            max(values) if _METRIC_DIRECTIONS[metric] == "max" else min(values)
        )
    raise ValueError(f"unknown target-level reduction: {reduction}")


def aggregate_records(
    records: Sequence[dict[str, object]],
) -> dict[str, object]:
    """
    Group per-work-unit records by modality and regime and summarise
    them.
    """
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for record in records:
        modality = str(record["modality"])
        task = str(
            record.get(
                "generation_task",
                "sequence_inpaint"
                if modality == "sequence"
                else "structure_inpaint",
            )
        )
        label = (
            modality
            if task in ("sequence_inpaint", "structure_inpaint")
            else task
        )
        key = (label, modality, task, str(record["regime"]))
        grouped.setdefault(key, []).append(record)

    by_regime: dict[str, dict[str, object]] = {}
    for (label, modality, task, regime), group in sorted(grouped.items()):
        samples: list[dict[str, object]] = []
        for record in group:
            samples.extend(record.get("samples", []))
        if modality == "sequence":
            fields = _SEQUENCE_FIELDS
        elif modality == "structure":
            fields = _STRUCTURE_FIELDS
        else:
            fields = _SEQUENCE_FIELDS + _STRUCTURE_FIELDS
        distributions = {
            name: _distribution_summary(
                [sample.get(name) for sample in samples]
            )
            for name in fields
        }
        from evaluation.evaluation_drivers.proteins.statistics import (
            power_of_two_prefix_budgets,
        )

        candidate_count = max(
            (len(record.get("samples", [])) for record in group), default=0
        )
        reporting_budgets = power_of_two_prefix_budgets(candidate_count)
        target_level = {
            "statistical_unit": "held-out chain",
            "single_output": {
                name: _distribution_summary(
                    [
                        _target_metric(
                            record, name, budget=1, reduction="first"
                        )
                        for record in group
                    ],
                    include_values=False,
                    bootstrap=True,
                )
                for name in fields
            },
            "candidate_mean": {
                name: _distribution_summary(
                    [
                        _target_metric(record, name, reduction="mean")
                        for record in group
                    ],
                    include_values=False,
                    bootstrap=True,
                )
                for name in fields
            },
            "best_of_n": {
                str(budget): {
                    name: _distribution_summary(
                        [
                            _target_metric(
                                record,
                                name,
                                budget=budget,
                                reduction="best",
                            )
                            for record in group
                        ],
                        include_values=False,
                        bootstrap=True,
                    )
                    for name in fields
                }
                for budget in reporting_budgets
            },
            "reporting_budgets": reporting_budgets,
            "best_of_n_selection": (
                "metric-specific native-reference oracle over a deterministic "
                "generation-order prefix; selection-assisted, not a direct "
                "single-output comparison"
            ),
        }
        entry: dict[str, object] = {
            "modality": modality,
            "generation_task": task,
            "regime": regime,
            "num_problems": len(group),
            "num_samples": len(samples),
            "mean_deleted_fraction": (
                float(np.mean([r["deleted_fraction"] for r in group]))
                if group
                else None
            ),
            "mean_num_segments": (
                float(np.mean([r["num_segments"] for r in group]))
                if group
                else None
            ),
            "distributions": distributions,
            "target_level": target_level,
        }
        if modality in ("structure", "pair"):
            seam_distances = [
                float(seam["distance"])
                for sample in samples
                for seam in sample.get("junctions", {}).get("seams", [])
            ]
            entry["junction_ca_ca"] = _distribution_summary(seam_distances)
            entry["junction_abs_deviation"] = _distribution_summary(
                [
                    abs(value - NATIVE_CA_CA_DISTANCE)
                    for value in seam_distances
                ]
            )
            entry["tokenizer_ceiling"] = {
                "region_ca_rmsd": _distribution_summary(
                    [
                        r.get("tokenizer_ceiling", {}).get("region_ca_rmsd")
                        for r in group
                    ]
                ),
                "region_lddt_vs_context": _distribution_summary(
                    [
                        r.get("tokenizer_ceiling", {}).get(
                            "region_lddt_vs_context"
                        )
                        for r in group
                    ]
                ),
                "tm_score": _distribution_summary(
                    [
                        r.get("tokenizer_ceiling", {}).get("tm_score")
                        for r in group
                    ]
                ),
            }
            # The clamp is what makes this a conditional evaluation at
            # all, so a single record with imperfect context bits
            # invalidates its group.
            context_values = [
                sample.get("context_bit_accuracy")
                for sample in samples
                if sample.get("context_bit_accuracy") is not None
            ]
            entry["clamp_held"] = (
                bool(all(value >= 1.0 for value in context_values))
                if context_values
                else None
            )
        by_regime[f"{label}/{regime}"] = entry

    return {
        "num_records": len(records),
        "by_regime": by_regime,
        "regime_table": {
            regime.name: {
                "kind": regime.kind,
                "fraction": (
                    None
                    if math.isnan(regime.fraction)
                    else float(regime.fraction)
                ),
                "sdedit": bool(regime.sdedit),
            }
            for regime in REGIMES_BY_NAME.values()
        },
    }


# -----------------------------------------------------------------------------
# Sharding and resume
# -----------------------------------------------------------------------------


def shard_problems(
    problems: Sequence[InpaintingProblem], shard_id: int, num_shards: int
) -> list[InpaintingProblem]:
    """Deterministic strided slice of the work units for one shard."""
    if int(num_shards) <= 1:
        return list(problems)
    if not 0 <= int(shard_id) < int(num_shards):
        raise ValueError(
            f"shard_id {shard_id} out of range for {num_shards} shards"
        )
    return list(problems)[int(shard_id) :: int(num_shards)]


def shard_path(out_dir, shard_id: int, num_shards: int) -> Path:
    """Path of the JSONL file this shard appends its finished work units to."""
    return (
        Path(out_dir)
        / "shards"
        / f"shard_{int(shard_id):05d}-of-{int(num_shards):05d}.jsonl"
    )


def load_completed_keys(path) -> set[str]:
    """Work-unit keys already present in a shard file, for --resume."""
    path = Path(path)
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(str(json.loads(line)["key"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return keys


def _read_shard_records(out_dir) -> tuple[list[dict[str, object]], list[str]]:
    """Every record written by every shard present, with the file list."""
    shards_dir = Path(out_dir) / "shards"
    files = sorted(shards_dir.glob("shard_*.jsonl"))
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(record.get("key", ""))
                if key in seen:
                    continue  # a re-run without --resume rewrote the unit
                seen.add(key)
                records.append(record)
    return records, [str(path) for path in files]


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    """
    Append one record and flush, so a killed shard loses at most one
    unit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# -----------------------------------------------------------------------------
# Library entry point
# -----------------------------------------------------------------------------


def run_inpainting(
    model,
    cfg,
    problems: Sequence[InpaintingProblem],
    device,
    *,
    tokenizer_path=None,
    num_samples: int = 4,
    seed: int = 0,
    guidance_scale: float | None = None,
    sigma_start: float | None = None,
    num_steps: int | None = None,
    jsonl_path: Path | None = None,
    completed: set[str] | None = None,
    pdb_dir: Path | None = None,
    generate_fn=None,
    tokenizer=None,
    judge=None,
) -> dict[str, object]:
    """Complete every problem and return the aggregation over what it ran."""
    generate_fn = generate_fn if generate_fn is not None else _load_generate()
    judge = judge if judge is not None else _load_structure_metrics()
    completed = set(completed or ())

    needs_tokenizer = any(
        problem.modality in ("structure", "pair") for problem in problems
    )
    if tokenizer is None and needs_tokenizer:
        tokenizer = _load_tokenizer(
            tokenizer_path, str(getattr(device, "type", device))
        )

    records: list[dict[str, object]] = []
    for problem in problems:
        if problem.key in completed:
            continue
        result = evaluate_problem(
            problem,
            generate_fn=generate_fn,
            tokenizer=tokenizer,
            judge=judge,
            model=model,
            cfg=cfg,
            device=device,
            num_samples=num_samples,
            seed=seed,
            guidance_scale=guidance_scale,
            sigma_start=sigma_start,
            num_steps=num_steps,
            pdb_dir=pdb_dir,
        )
        record = result.to_record()
        records.append(record)
        if jsonl_path is not None:
            _append_jsonl(Path(jsonl_path), record)
    return aggregate_records(records)


# -----------------------------------------------------------------------------
# Command-line entry point
# -----------------------------------------------------------------------------


def _jsonable(obj):
    """Coerce numpy scalars, arrays, and non-finite floats for JSON."""
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Write JSON to a temporary sibling and atomically replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _resolve_regimes(names: str | None, sigma_start) -> list[DeletionRegime]:
    """
    Regime list for this run, adding the SDEdit regime only if it can
    run.
    """
    available = list(DELETION_REGIMES)
    if sigma_start is not None:
        available.append(SDEDIT_REGIME)
    if not names:
        return available
    wanted = [name.strip() for name in str(names).split(",") if name.strip()]
    known = {regime.name: regime for regime in available}
    missing = [name for name in wanted if name not in known]
    if missing:
        raise SystemExit(
            f"unknown regime(s) {missing}; available: {sorted(known)}"
            + (
                " (structure_denoise additionally needs --sigma-start)"
                if sigma_start is None
                else ""
            )
        )
    return [known[name] for name in wanted]


def _resolve_completion_tasks(names: str | None, modality: str) -> list[str]:
    """
    Resolve explicit model task modes, preserving the legacy modality
    CLI.
    """
    if not names:
        return (
            ["sequence_inpaint", "structure_inpaint"]
            if modality == "both"
            else [f"{modality}_inpaint"]
        )
    tasks = [name.strip() for name in str(names).split(",") if name.strip()]
    unknown = sorted(set(tasks).difference(COMPLETION_TASKS))
    if unknown:
        raise SystemExit(
            f"unknown completion task(s) {unknown}; available: "
            f"{list(COMPLETION_TASKS)}"
        )
    if len(set(tasks)) != len(tasks):
        raise SystemExit("--tasks contains a duplicate task")
    return tasks


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the inpainting evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Structure and sequence completion (inpainting) through the "
            "clamped bitstream diffusion sampler."
        )
    )
    parser.add_argument(
        "--config", required=True, help="Path to the binary model config"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--backbones",
        required=True,
        help=(
            "Held-out chains: an .npz file or a directory of them, in the "
            "layout evaluation.evaluation_drivers.proteins.inverse_folding."
            "load_backbone_records "
            "reads (coords [L, 4, 3] plus a native sequence)"
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        help=(
            "Output report JSON path; resumable shard files are placed in the "
            "shared shard directory derived from this path"
        ),
    )
    parser.add_argument(
        "--struct-tokenizer",
        default=None,
        help=(
            "Local DPLM-2 structure tokenizer checkpoint, used to encode "
            "chains without cached tokens and to decode completions"
        ),
    )
    parser.add_argument(
        "--modality",
        default="structure",
        choices=("structure", "sequence", "both"),
        help="Which completion task to evaluate (default: structure)",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help=(
            f"Comma-separated explicit task modes. Overrides --modality; "
            f"choose from {', '.join(COMPLETION_TASKS)}."
        ),
    )
    parser.add_argument(
        "--regimes",
        default=None,
        help=(
            f"Comma-separated deletion regimes; default is every regime ("
            f"{', '.join(r.name for r in DELETION_REGIMES)})"
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Completions sampled per chain and regime",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--min-len", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument(
        "--splits",
        default=None,
        help="Comma-separated split labels to keep (e.g. test)",
    )
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument(
        "--sigma-start",
        type=float,
        default=None,
        help=(
            "Enable the structure_denoise regime: start the sampler from the "
            "native structure corrupted to this sigma and denoise the rest of "
            "the way (SDEdit-style refinement)"
        ),
    )
    parser.add_argument(
        "--pdb-dir",
        default=None,
        help="Optional directory for one native/completion PDB pair per unit",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    add_shard_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI driver using the shared 8-GPU shard/resume/merge contract."""
    args = build_parser().parse_args(argv)
    plan = resolve_shard(args)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    regimes = _resolve_regimes(args.regimes, args.sigma_start)
    completion_tasks = _resolve_completion_tasks(args.tasks, args.modality)
    modalities = (
        ("structure", "sequence")
        if args.modality == "both"
        else (args.modality,)
    )

    load_backbone_records, resolver_cls = _load_backbone_corpus()
    records = load_backbone_records(
        args.backbones,
        splits=(
            [s.strip() for s in args.splits.split(",")]
            if args.splits
            else None
        ),
        min_len=args.min_len,
        max_len=args.max_len,
        max_examples=None,
    )
    if not records:
        raise SystemExit(f"no held-out chains found under {args.backbones}")
    # A publication cap is a deterministic length-stratified subset, not
    # the first N rows of an archive whose storage order may itself be
    # length- or source-sorted.  The selected stable ids are written
    # into the report.
    if args.max_examples is not None and len(records) > int(args.max_examples):
        cap = int(args.max_examples)
        if cap <= 0:
            raise SystemExit("--max-examples must be positive when supplied")
        ordered = sorted(records, key=lambda row: (row.length, row.stable_id))
        positions = np.linspace(0, len(ordered) - 1, cap).round().astype(int)
        records = [ordered[int(position)] for position in positions]

    resolver = resolver_cls(
        args.struct_tokenizer, device=str(getattr(device, "type", device))
    )
    problems = build_problems(
        records,
        regimes,
        modalities,
        struct_index_resolver=resolver.resolve,
        seed=args.seed,
        # Enumerate the full deterministic key universe without making
        # every GPU encode every backbone. Tokens are resolved below
        # only for this shard's structure work, once per chain and
        # reused across regimes.
        resolve_structure=False,
        tasks=completion_tasks,
    )
    all_keys = unit_keys(problems, key=lambda problem: problem.key)
    payload: dict[str, object] = {
        "schema_version": 1,
        "task": "inpainting",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "backbones": str(args.backbones),
        "selected_backbone_ids": [record.stable_id for record in records],
        "parameters": {
            "modality": args.modality,
            "tasks": completion_tasks,
            "regimes": [regime.name for regime in regimes],
            "num_samples": args.num_samples,
            "num_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "sigma_start": args.sigma_start,
            "seed": args.seed,
            "device": str(device),
            "struct_tokenizer": args.struct_tokenizer,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "git_commit": repository_commit(_REPO_ROOT),
        },
    }
    writer = ShardWriter(
        plan,
        task="inpainting",
        meta={
            "payload": payload,
            "input_identities": input_identities(
                [
                    args.config,
                    args.checkpoint,
                    args.backbones,
                    args.struct_tokenizer,
                ]
            ),
        },
    )
    writer.write_manifest(all_keys)
    if not args.resume:
        writer.reset()
    done = writer.completed_keys() if args.resume else set()
    mine = partition(
        problems,
        plan.shard_id,
        plan.num_shards,
        key=lambda problem: problem.key,
    )
    todo = [problem for problem in mine if problem.key not in done]

    model = cfg = tokenizer = judge = None
    if todo:
        records_by_id = {record.stable_id: record for record in records}
        token_cache: dict[str, np.ndarray] = {}
        for problem in todo:
            if not _task_needs_structure(problem.generation_task):
                continue
            if problem.stable_id not in token_cache:
                token_cache[problem.stable_id] = np.asarray(
                    resolver.resolve(records_by_id[problem.stable_id]),
                    dtype=np.uint16,
                )
            problem.struct_index = token_cache[problem.stable_id]
        model, cfg = _load_model(
            args.config, args.checkpoint, device, args.num_steps
        )
        judge = _load_structure_metrics()
        needs_tokenizer = any(
            problem.modality in ("structure", "pair") for problem in todo
        )
        if needs_tokenizer:
            tokenizer = getattr(resolver, "_tokenizer", None)
            if tokenizer is None:
                tokenizer = _load_tokenizer(
                    args.struct_tokenizer, str(device.type)
                )

    for problem in todo:
        random_seed = unit_seed(args.seed, problem.key, namespace="inpainting")
        torch.manual_seed(random_seed)
        np.random.seed(random_seed % (2**32))
        result = evaluate_problem(
            problem,
            generate_fn=_load_generate(),
            tokenizer=tokenizer,
            judge=judge,
            model=model,
            cfg=cfg,
            device=device,
            num_samples=args.num_samples,
            seed=random_seed,
            guidance_scale=args.guidance_scale,
            sigma_start=args.sigma_start,
            num_steps=args.num_steps,
            pdb_dir=Path(args.pdb_dir) if args.pdb_dir else None,
        )
        record = result.to_record()
        record["random_seed"] = int(random_seed)
        writer.write(problem.key, record)

    pairs = finalise(plan, writer, all_keys)
    if pairs is None:
        return 0
    all_records = [record for _key, record in pairs]
    payload["results"] = aggregate_records(all_records)
    atomic_evaluation_json(Path(args.out), payload, task="inpainting")
    print(
        f"Saved inpainting report over {len(all_records)} records to "
        f"{args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
