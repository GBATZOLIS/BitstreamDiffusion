"""Forward-folding evaluation for the 18-bit multimodal bitstream model.

This is plan section 10.6: the forward-folding conditional p(z | s). The protocol
runs on a held-out temporal benchmark (CAMEO 2022 or an RCSB PDB deposition-date
split) that shares no cluster with training. For every target the five sequence
bits per residue are clamped as OBSERVED and the thirteen LFQ structure bits are
sampled by ``evaluation.proteins.generate_multimodal.generate`` with
``task='forward_folding'``. The sampled structure token ids are decoded to a
backbone by the frozen DPLM-2 structure tokenizer, and the decoded backbone is
scored against the native backbone with the independent geometric judges in
``evaluation.proteins.structure_metrics`` (RMSD, TM-score, lDDT). Two token-level
diagnostics accompany the geometry: the LFQ per-bit accuracy and the exact token
index accuracy of the sampled structure code against the native code.

Three reference conditions frame the model number. The tokenizer reconstruction
ceiling encodes and decodes the native backbone through the same frozen tokenizer
and measures agreement with the native; no bit model can beat it, so it is the
upper bound on achievable geometry. The shuffled-sequence control clamps a
permuted copy of the native sequence, which should collapse folding quality if
the model truly uses the sequence. The unconditional control samples structure
with nothing clamped (the structure marginal), which measures how much the
sequence conditioning contributes. Performance is additionally broken down by
chain length and by the target's maximum sequence identity to the training set.

The module imports with only numpy and torch present. The diffusion model, the
DPLM-2 tokenizer, the sampler, and tmtools are all imported lazily inside the
functions that need them, so importing this module never pulls in a heavy or
external dependency. The command-line entry point wires a checkpoint, a config,
a target manifest, and a tokenizer checkpoint into a folding metrics JSON.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Make the repository root importable whether this file is imported as a module
# or executed directly as a script, so the absolute ``data`` / ``evaluation``
# imports below resolve in both cases.
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from data.protein_multimodal import CANONICAL_AA
from data.protein_structure_codec import DEFAULT_STRUCT_CODEC
from evaluation.proteins import structure_metrics
from evaluation.proteins.io import atomic_json

__all__ = [
    "FoldingTarget",
    "sequence_to_ids",
    "ids_to_sequence",
    "load_targets",
    "backbone_agreement",
    "structure_token_accuracy",
    "tokenizer_reconstruction_ceiling",
    "fold_target",
    "evaluate_folding",
    "load_folding_model",
    "main",
]


# Amino-acid id lookup for the canonical-20 alphabet used by the sequence codec.
_AA_TO_ID = {aa: i for i, aa in enumerate(CANONICAL_AA)}
SEQ_VOCAB_SIZE = len(CANONICAL_AA)  # 20

# Per-metric optimization direction. ``max`` metrics improve as they rise (TM,
# lDDT, token accuracies); ``min`` metrics improve as they fall (RMSD). The
# best-of-N reduction over samples uses this direction.
METRIC_DIRECTIONS: Dict[str, str] = {
    "tm_score": "max",
    "lddt": "max",
    "rmsd": "min",
    "lfq_bit_accuracy": "max",
    "exact_index_accuracy": "max",
}

# Task-summary keys reported per target, one section per condition evaluated.
TASK_KEYS = (
    "forward_folding",
    "shuffled_control",
    "unconditional_control",
    "tokenizer_ceiling",
)

# Keys that describe a summary rather than a metric and are excluded when
# aggregating summaries across targets.
_NON_AGGREGATED_KEYS = frozenset({"n_samples"})

# Chain-length buckets (half-open, low inclusive) for the by-length breakdown.
_LENGTH_BUCKETS = (
    (0, 100, "len_0_99"),
    (100, 200, "len_100_199"),
    (200, 300, "len_200_299"),
    (300, 400, "len_300_399"),
    (400, 512, "len_400_511"),
    (512, 10**9, "len_512_plus"),
)

# Training sequence-identity buckets (half-open, low inclusive) for the
# by-identity breakdown. A target with no identity annotation is labelled below.
_IDENTITY_BUCKETS = (
    (0.0, 0.3, "id_0.0_0.3"),
    (0.3, 0.5, "id_0.3_0.5"),
    (0.5, 0.7, "id_0.5_0.7"),
    (0.7, 0.9, "id_0.7_0.9"),
    (0.9, 1.0001, "id_0.9_1.0"),
)
_IDENTITY_UNKNOWN = "id_unknown"


@dataclass
class FoldingTarget:
    """One forward-folding target: an observed sequence and its native backbone.

    ``seq_ids`` holds the canonical-20 amino-acid ids that are clamped as the
    OBSERVED conditioning. ``native_coords`` is the ground-truth backbone of shape
    ``[L, 4, 3]`` (atoms N, CA, C, O) used as the reference for geometry, and it
    may be absent when only the token-level diagnostics are wanted.
    ``native_struct_index`` is the native LFQ token id per residue; when it is
    absent it is recovered by encoding ``native_coords`` through the tokenizer.
    ``train_seq_identity`` is the maximum sequence identity to the training set
    and drives the by-identity breakdown; it is NaN when not annotated.
    """

    target_id: str
    seq_ids: np.ndarray
    native_coords: Optional[np.ndarray] = None
    native_struct_index: Optional[np.ndarray] = None
    source: str = "unknown"
    release: str = ""
    train_seq_identity: float = float("nan")

    @property
    def length(self) -> int:
        """Number of residues, taken from the observed sequence."""
        return int(np.asarray(self.seq_ids).shape[0])


def sequence_to_ids(sequence: str) -> np.ndarray:
    """Map a canonical-20 amino-acid string to an int64 id array.

    The mapping matches the sequence codec used everywhere else in the pipeline.
    A residue outside the canonical twenty raises a clear error, since folding
    can only clamp sequence bits for residues that have a valid codeword.
    """
    text = str(sequence).strip().upper()
    ids = np.empty(len(text), dtype=np.int64)
    unknown = sorted({ch for ch in text if ch not in _AA_TO_ID})
    if unknown:
        raise ValueError(
            "sequence contains non-canonical residues "
            f"{unknown}; the folding target must use only {CANONICAL_AA}"
        )
    for i, ch in enumerate(text):
        ids[i] = _AA_TO_ID[ch]
    return ids


def ids_to_sequence(seq_ids) -> str:
    """Map canonical-20 amino-acid ids back to a string, the inverse of the map."""
    ids = np.asarray(seq_ids, dtype=np.int64).reshape(-1)
    if ids.size and (ids.min() < 0 or ids.max() >= SEQ_VOCAB_SIZE):
        raise ValueError(f"seq_ids out of range [0, {SEQ_VOCAB_SIZE - 1}]")
    return "".join(CANONICAL_AA[i] for i in ids)


def _load_coords_from_npz(path: Path, key: Optional[str]) -> np.ndarray:
    """Load a backbone coordinate array from an ``.npz`` file.

    When ``key`` is given that array is read; otherwise the first of a small set
    of conventional coordinate keys present in the archive is used.
    """
    if not path.exists():
        raise FileNotFoundError(f"native coordinate file not found: {path}")
    data = np.load(path)
    if key is not None:
        return np.asarray(data[key], dtype=np.float32)
    for candidate in ("coords", "backbone", "atom_positions", "bb_coords"):
        if candidate in data:
            return np.asarray(data[candidate], dtype=np.float32)
    raise KeyError(
        f"{path} contains no recognised coordinate array; available keys are "
        f"{list(data.keys())}. Pass 'coords_key' in the target entry."
    )


def _load_native_coords(
    entry: Mapping[str, object], base: Path
) -> Optional[np.ndarray]:
    """Resolve the native backbone for one target entry, or None when absent."""
    if entry.get("coords") is not None:
        arr = np.asarray(entry["coords"], dtype=np.float32)
    else:
        location = (
            entry.get("coords_file")
            or entry.get("coords_npz")
            or entry.get("coords_path")
        )
        if not location:
            return None
        path = Path(str(location))
        if not path.is_absolute():
            path = base / path
        arr = _load_coords_from_npz(path, entry.get("coords_key"))  # type: ignore[arg-type]
    if arr.ndim != 3 or arr.shape[1] < 4 or arr.shape[2] != 3:
        raise ValueError(
            "native coords must have shape [L, 4, 3] with atoms N, CA, C, O; "
            f"got shape {tuple(arr.shape)}"
        )
    return arr[:, :4, :].astype(np.float32)


def _target_from_entry(
    entry: Mapping[str, object], base: Path, index: int
) -> FoldingTarget:
    """Build one :class:`FoldingTarget` from a manifest entry."""
    target_id = str(
        entry.get("target_id")
        or entry.get("id")
        or entry.get("name")
        or f"target_{index}"
    )
    if entry.get("seq_ids") is not None:
        seq_ids = np.asarray(entry["seq_ids"], dtype=np.int64)
        if seq_ids.size and (
            seq_ids.min() < 0 or seq_ids.max() >= SEQ_VOCAB_SIZE
        ):
            raise ValueError(
                f"target {target_id}: seq_ids out of canonical-20 range"
            )
    elif entry.get("sequence"):
        seq_ids = sequence_to_ids(str(entry["sequence"]))
    else:
        raise ValueError(
            f"target {target_id} has neither 'sequence' nor 'seq_ids'"
        )

    coords = _load_native_coords(entry, base)
    struct_index = None
    if entry.get("struct_index") is not None:
        struct_index = np.asarray(entry["struct_index"], dtype=np.int64)

    identity_raw = (
        entry.get("train_seq_identity")
        if entry.get("train_seq_identity") is not None
        else entry.get("max_train_identity", entry.get("train_identity"))
    )
    identity = (
        float(identity_raw) if identity_raw is not None else float("nan")
    )

    return FoldingTarget(
        target_id=target_id,
        seq_ids=seq_ids,
        native_coords=coords,
        native_struct_index=struct_index,
        source=str(entry.get("source", "unknown")),
        release=str(entry.get("release", "")),
        train_seq_identity=identity,
    )


def load_targets(manifest_path) -> List[FoldingTarget]:
    """Load forward-folding targets from a JSON manifest.

    The manifest is either a list of target objects or an object with a
    ``targets`` (or ``rows``) list. Each target object supplies the observed
    sequence (``sequence`` or ``seq_ids``), an optional native backbone
    (``coords`` inline, or ``coords_file`` / ``coords_npz`` pointing at an
    ``.npz`` with a coordinate array), an optional native ``struct_index``, and
    optional ``source``, ``release``, and ``train_seq_identity`` fields.
    Relative coordinate paths resolve against the manifest's directory.
    """
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        entries = raw.get("targets", raw.get("rows"))
        if entries is None:
            raise ValueError(
                "targets manifest object must contain a 'targets' or 'rows' list"
            )
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError("targets manifest must be a JSON list or object")
    base = manifest_path.resolve().parent
    return [
        _target_from_entry(entry, base, i) for i, entry in enumerate(entries)
    ]


def _ca_track(coords: np.ndarray) -> np.ndarray:
    """Return the CA track ``[L, 3]`` from backbone ``[L, 4, 3]`` or CA input."""
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1] >= 2 and arr.shape[2] == 3:
        return np.ascontiguousarray(arr[:, 1, :])
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    raise ValueError(
        "coordinates must be backbone [L, 4, 3] or CA [L, 3]; "
        f"got shape {tuple(arr.shape)}"
    )


def backbone_agreement(pred_coords, native_coords) -> Dict[str, float]:
    """Score a predicted backbone against a native one with the geometry judges.

    Returns TM-score, superposed RMSD, and lDDT between the two backbones,
    computed with ``evaluation.proteins.structure_metrics`` (the exact TM-score
    needs tmtools; without it structure_metrics falls back to its documented
    approximation). The two structures are truncated to their common length so a
    length mismatch degrades gracefully rather than raising.
    """
    pred = np.asarray(pred_coords, dtype=np.float64)
    native = np.asarray(native_coords, dtype=np.float64)
    pred_ca = _ca_track(pred)
    native_ca = _ca_track(native)
    length = int(min(pred_ca.shape[0], native_ca.shape[0]))
    if length == 0:
        return {
            "tm_score": float("nan"),
            "rmsd": float("nan"),
            "lddt": float("nan"),
        }
    pred_ca = pred_ca[:length]
    native_ca = native_ca[:length]
    pred_bb = pred[:length]
    native_bb = native[:length]
    return {
        "tm_score": float(structure_metrics.tm_score(pred_ca, native_ca)),
        "rmsd": float(
            structure_metrics.rmsd(pred_ca, native_ca, superpose=True)
        ),
        "lddt": float(structure_metrics.lddt(pred_bb, native_bb)),
    }


def structure_token_accuracy(pred_index, native_index) -> Dict[str, float]:
    """Token-level diagnostics between a sampled and a native LFQ structure code.

    Returns the exact index accuracy (fraction of residues whose sampled token id
    equals the native id) and the LFQ per-bit accuracy (fraction of the thirteen
    structure bits per residue that match), the latter computed with the frozen
    codec's id-to-bits map. The two arrays are truncated to their common length.
    """
    pred = np.asarray(pred_index, dtype=np.int64).reshape(-1)
    native = np.asarray(native_index, dtype=np.int64).reshape(-1)
    length = int(min(pred.shape[0], native.shape[0]))
    if length == 0:
        return {
            "exact_index_accuracy": float("nan"),
            "lfq_bit_accuracy": float("nan"),
        }
    pred = pred[:length]
    native = native[:length]
    exact = float(np.mean(pred == native))
    pred_bits = DEFAULT_STRUCT_CODEC.index_to_bits_np(pred)
    native_bits = DEFAULT_STRUCT_CODEC.index_to_bits_np(native)
    bit_acc = float(np.mean(pred_bits == native_bits))
    return {"exact_index_accuracy": exact, "lfq_bit_accuracy": bit_acc}


def _summarize_samples(
    samples: Sequence[Mapping[str, float]],
) -> Dict[str, float]:
    """Reduce a list of per-sample metric dicts to mean and best-of-N values.

    For every metric present on any sample this reports the mean over samples
    and the best value according to the metric direction (max for TM, lDDT, and
    the token accuracies; min for RMSD). Non-finite entries are ignored.
    """
    summary: Dict[str, float] = {"n_samples": int(len(samples))}
    for metric, direction in METRIC_DIRECTIONS.items():
        values = [
            float(s[metric])
            for s in samples
            if metric in s
            and s[metric] is not None
            and not math.isnan(float(s[metric]))
        ]
        if not values:
            continue
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_best"] = float(
            max(values) if direction == "max" else min(values)
        )
    return summary


def _run_generate(
    model, cfg, task: str, length: int, num_samples: int, device, observed
):
    """Sample structure token ids for one target, importing generate lazily."""
    from evaluation.proteins.generate_multimodal import generate

    out = generate(
        model,
        cfg,
        task,
        int(length),
        int(num_samples),
        device,
        observed=observed,
    )
    return np.asarray(out["struct_index"])  # [num_samples, L]


def fold_target(
    model,
    cfg,
    target: FoldingTarget,
    device,
    *,
    tokenizer=None,
    num_samples: int = 8,
    task: str = "forward_folding",
    observed_seq_ids=None,
    native_struct_index=None,
) -> Dict[str, float]:
    """Sample structure for one target and summarize the agreement metrics.

    For ``forward_folding`` the observed sequence (``observed_seq_ids`` when given,
    else the target's own sequence) is clamped and the structure is sampled; for
    ``structure_marginal`` nothing is clamped (the unconditional control). Every
    sampled structure code is scored for token accuracy against
    ``native_struct_index`` when available, and, when a tokenizer and native
    backbone are present, decoded to a backbone and scored geometrically. The
    return value is the mean and best-of-N summary over ``num_samples`` samples.
    """
    length = target.length
    if task == "forward_folding":
        seq_ids = (
            target.seq_ids if observed_seq_ids is None else observed_seq_ids
        )
        observed: Optional[Dict[str, object]] = {
            "seq_ids": np.asarray(seq_ids, dtype=np.int64)
        }
    elif task == "structure_marginal":
        observed = None
    else:
        raise ValueError(
            f"fold_target supports 'forward_folding' or 'structure_marginal'; got {task!r}"
        )

    pred_index = _run_generate(
        model, cfg, task, length, num_samples, device, observed
    )

    samples: List[Dict[str, float]] = []
    for row in range(pred_index.shape[0]):
        idx = pred_index[row]
        record: Dict[str, float] = {}
        if native_struct_index is not None:
            record.update(structure_token_accuracy(idx, native_struct_index))
        if tokenizer is not None and target.native_coords is not None:
            pred_coords = tokenizer.decode(np.asarray(idx, dtype=np.uint16))
            record.update(
                backbone_agreement(pred_coords, target.native_coords)
            )
        samples.append(record)
    return _summarize_samples(samples)


def tokenizer_reconstruction_ceiling(target: FoldingTarget, tokenizer):
    """Encode and decode the native backbone to measure the reconstruction ceiling.

    Runs the native backbone through the frozen tokenizer (encode then decode) and
    scores the reconstruction against the native backbone. Returns a
    ``(metrics, native_index)`` pair, where ``metrics`` is the geometry agreement
    dict and ``native_index`` is the native LFQ token code produced by the encoder
    (reusable as the reference for token accuracy). Both are None when the
    tokenizer or the native backbone is unavailable.
    """
    if tokenizer is None or target.native_coords is None:
        return None, None
    native_index = np.asarray(
        tokenizer.encode(np.asarray(target.native_coords, dtype=np.float32))
    )
    recon = tokenizer.decode(np.asarray(native_index, dtype=np.uint16))
    metrics = backbone_agreement(recon, target.native_coords)
    return metrics, native_index


def _shuffle_sequence(seq_ids, rng: np.random.Generator) -> np.ndarray:
    """Return a randomly permuted copy of the sequence ids (shuffled control)."""
    ids = np.asarray(seq_ids, dtype=np.int64).copy()
    rng.shuffle(ids)
    return ids


def _bucket_length(length: int) -> str:
    """Return the length-bucket label for a chain length."""
    for low, high, label in _LENGTH_BUCKETS:
        if low <= length < high:
            return label
    return _LENGTH_BUCKETS[-1][2]


def _bucket_identity(identity: float) -> str:
    """Return the identity-bucket label, or the unknown label for NaN identity."""
    if identity is None or (
        isinstance(identity, float) and math.isnan(identity)
    ):
        return _IDENTITY_UNKNOWN
    for low, high, label in _IDENTITY_BUCKETS:
        if low <= identity < high:
            return label
    return _IDENTITY_BUCKETS[-1][2]


def _summ(values: Sequence[Optional[float]]) -> Dict[str, object]:
    """Summarize a list of scalar values as count, mean, median, std, min, max."""
    finite = np.asarray(
        [
            float(v)
            for v in values
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _aggregate_summaries(
    summaries: Sequence[Mapping[str, object]],
) -> Dict[str, Dict[str, object]]:
    """Aggregate a list of per-target task summaries into per-metric statistics."""
    keys: List[str] = []
    for summary in summaries:
        for key, value in summary.items():
            if key in _NON_AGGREGATED_KEYS:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and key not in keys:
                keys.append(key)
    return {
        key: _summ([summary.get(key) for summary in summaries]) for key in keys
    }


def _aggregate_over_targets(
    per_target: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Aggregate every task section across a group of per-target records."""
    aggregated: Dict[str, object] = {}
    for task_key in TASK_KEYS:
        summaries = [
            rec[task_key]
            for rec in per_target
            if isinstance(rec.get(task_key), dict)
        ]
        if summaries:
            aggregated[task_key] = _aggregate_summaries(summaries)
    return aggregated


def _aggregate_by_bucket(
    per_target: Sequence[Mapping[str, object]], bucket_key: str
) -> Dict[str, object]:
    """Group per-target records by a bucket label and aggregate each group."""
    groups: Dict[str, List[Mapping[str, object]]] = {}
    for rec in per_target:
        groups.setdefault(str(rec.get(bucket_key)), []).append(rec)
    out: Dict[str, object] = {}
    for bucket, records in sorted(groups.items()):
        out[bucket] = {
            "num_targets": len(records),
            "metrics": _aggregate_over_targets(records),
        }
    return out


def evaluate_folding(
    model,
    cfg,
    targets: Sequence[FoldingTarget],
    device,
    *,
    tokenizer=None,
    num_samples: int = 8,
    include_controls: bool = True,
    include_ceiling: bool = True,
    seed: int = 0,
) -> Dict[str, object]:
    """Run the full forward-folding protocol over a set of targets.

    For each target this samples structure under forward folding, and optionally
    the shuffled-sequence control, the unconditional control, and the tokenizer
    reconstruction ceiling. Every condition is summarized per target, then
    aggregated overall and by both length and training sequence-identity buckets.
    The native token code used for the token-accuracy diagnostics is the target's
    own ``native_struct_index`` when present, otherwise the code produced by
    encoding the native backbone during the ceiling computation. Returns a nested
    metrics dictionary that includes the per-target breakdown.
    """
    rng = np.random.default_rng(int(seed))
    per_target: List[Dict[str, object]] = []

    for target in targets:
        native_index = target.native_struct_index
        ceiling_metrics = None
        if include_ceiling:
            ceiling_metrics, encoded_index = tokenizer_reconstruction_ceiling(
                target, tokenizer
            )
            if native_index is None and encoded_index is not None:
                native_index = encoded_index

        record: Dict[str, object] = {
            "target_id": target.target_id,
            "length": target.length,
            "source": target.source,
            "release": target.release,
            "train_seq_identity": (
                None
                if math.isnan(target.train_seq_identity)
                else float(target.train_seq_identity)
            ),
            "length_bucket": _bucket_length(target.length),
            "identity_bucket": _bucket_identity(target.train_seq_identity),
            "forward_folding": fold_target(
                model,
                cfg,
                target,
                device,
                tokenizer=tokenizer,
                num_samples=num_samples,
                task="forward_folding",
                native_struct_index=native_index,
            ),
        }
        if ceiling_metrics is not None:
            record["tokenizer_ceiling"] = ceiling_metrics
        if include_controls:
            record["shuffled_control"] = fold_target(
                model,
                cfg,
                target,
                device,
                tokenizer=tokenizer,
                num_samples=num_samples,
                task="forward_folding",
                observed_seq_ids=_shuffle_sequence(target.seq_ids, rng),
                native_struct_index=native_index,
            )
            record["unconditional_control"] = fold_target(
                model,
                cfg,
                target,
                device,
                tokenizer=tokenizer,
                num_samples=num_samples,
                task="structure_marginal",
                native_struct_index=native_index,
            )
        per_target.append(record)

    return {
        "protocol": {
            "name": "forward_folding_p_z_given_s",
            "plan_section": "10.6",
            "benchmark": "CAMEO 2022 / PDB deposition-date split",
            "observed_modality": "sequence (5 bits per residue clamped)",
            "sampled_modality": "structure (13 LFQ bits per residue)",
            "geometry_judge": "evaluation.proteins.structure_metrics",
            "controls": ["shuffled_sequence", "unconditional"],
            "ceiling": "tokenizer_reconstruction",
        },
        "num_targets": len(per_target),
        "num_samples_per_target": int(num_samples),
        "overall": _aggregate_over_targets(per_target),
        "by_length": _aggregate_by_bucket(per_target, "length_bucket"),
        "by_sequence_identity": _aggregate_by_bucket(
            per_target, "identity_bucket"
        ),
        "per_target": per_target,
    }


def load_folding_model(
    config_path, checkpoint_path, device, *, num_steps: Optional[int] = None
):
    """Load the multimodal bitstream model and its config for folding.

    Thin wrapper over ``evaluation.proteins.io.load_binary_protein_model``: the
    config must select the binary 18-bit patch representation and the sampling
    step count is overridden when ``num_steps`` is provided. Returns
    ``(model, cfg)``.
    """
    from evaluation.proteins.io import load_binary_protein_model

    return load_binary_protein_model(
        config_path,
        checkpoint_path,
        device,
        num_steps=num_steps,
        context="forward folding",
    )


def _load_tokenizer(tokenizer_path, device):
    """Load the frozen DPLM-2 structure tokenizer, importing it lazily."""
    from evaluation.proteins.dplm_struct_tokenizer import load_struct_tokenizer

    return load_struct_tokenizer(tokenizer_path, device=str(device))


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser for the folding driver."""
    parser = argparse.ArgumentParser(
        description=(
            "Forward-folding evaluation (plan 10.6): clamp the sequence bits, "
            "sample structure, decode to a backbone, and score against native."
        )
    )
    parser.add_argument(
        "--config", required=True, help="Path to the model config module"
    )
    parser.add_argument(
        "--checkpoint", required=True, type=Path, help="Model checkpoint"
    )
    parser.add_argument(
        "--targets",
        required=True,
        type=Path,
        help="CAMEO 2022 / PDB date-split folding target manifest (JSON)",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Output metrics JSON path"
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Local DPLM-2 structure tokenizer checkpoint directory or HF repo id",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Structures sampled per target",
    )
    parser.add_argument(
        "--num-steps", type=int, default=None, help="Sampler step override"
    )
    parser.add_argument(
        "--max-targets", type=int, default=0, help="Cap on targets (0 = all)"
    )
    parser.add_argument(
        "--device", default=None, help="Torch device (default: auto)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help="Skip the shuffled-sequence and unconditional controls",
    )
    parser.add_argument(
        "--skip-ceiling",
        action="store_true",
        help="Skip the tokenizer reconstruction ceiling",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help=(
            "Skip tokenizer decoding and report only the token-accuracy "
            "diagnostics; no coordinate geometry is computed"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point: run folding on a manifest and write a JSON report.

    Loads the target manifest, the model, and (unless decoding is skipped) the
    frozen tokenizer; runs :func:`evaluate_folding`; and writes the metrics JSON.
    The heavy model and tokenizer are loaded only after arguments parse, so the
    argparse help path never imports them.
    """
    args = _build_arg_parser().parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    targets = load_targets(args.targets)
    if args.max_targets and args.max_targets > 0:
        targets = targets[: args.max_targets]
    if not targets:
        raise ValueError(f"no folding targets loaded from {args.targets}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_folding_model(
        args.config, args.checkpoint, device, num_steps=args.num_steps
    )

    tokenizer = None
    if not args.skip_decode:
        tokenizer = _load_tokenizer(args.tokenizer_path, device)

    result = evaluate_folding(
        model,
        cfg,
        targets,
        device,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        include_controls=not args.skip_controls,
        include_ceiling=(not args.skip_ceiling) and tokenizer is not None,
        seed=args.seed,
    )
    result["config"] = str(args.config)
    result["checkpoint"] = str(args.checkpoint)
    result["targets_manifest"] = str(args.targets)
    result["device"] = str(device)

    atomic_json(Path(args.out), result)
    print(json.dumps(result.get("overall", {}), indent=2, sort_keys=True))
    print(f"Saved forward-folding metrics to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
