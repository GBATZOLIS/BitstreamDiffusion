"""Motif scaffolding evaluation for the clamped bitstream diffusion model.

This module implements plan section 10.10: three-dimensional to
three-dimensional completion of a protein around a fixed functional motif. A
motif problem supplies one or more contiguous motif segments, each with a known
amino-acid sequence, a known structure (as DPLM-2 LFQ token ids and, where
available, reference backbone coordinates), and a scaffold length range. For
every problem the evaluator samples candidate placements of the motif inside a
scaffold, generates completions with ``generate(task='motif', observed=...)``,
and scores each completion.

The generation clamps both the motif sequence identities and the motif structure
bits at the placed motif positions and diffuses the rest of the design. The clamp
is built by ``evaluation.proteins.generate_multimodal.build_task_conditioning``
(driven inside ``generate``) from the motif spans, so the observed sequence and
structure bits are pinned together while their position mapping to the reference
motif residues is preserved explicitly by this module.

Reported metrics follow the DPLM motif-scaffolding convention:
  - motif RMSD, the backbone root-mean-square deviation between the generated
    motif region and the reference motif after superposition;
  - motif sequence preservation, the fraction of motif residues whose generated
    amino acid matches the reference;
  - whole-design self-consistency TM-score, from the ProteinMPNN plus ESMFold
    designability loop over the full generated backbone;
  - the number and fraction of solved targets, where a target counts as solved
    when at least one of its candidates has motif RMSD below one Angstrom and
    overall self-consistency TM-score above 0.8;
  - the diversity among the successful scaffolds;
  - the failures broken down by motif problem and by scaffold length.

The continuous metric distributions are always published alongside the pass or
fail summaries so callers can rank rather than only threshold.

Only numpy and torch are imported eagerly, so the module imports cleanly with
just those two packages present. The diffusion model, the generation driver, the
DPLM structure tokenizer, the ProteinMPNN plus ESMFold self-consistency loop, and
the structure-metrics judge are all imported lazily inside the functions that use
them; a missing dependency raises a clear RuntimeError pointing at
the protein-eval extra (uv sync --extra protein-eval), the dplm-inference extra (uv sync --extra dplm-inference), and
scripts/proteins/setup/setup_evaluation.sh.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Make the repository root importable so the module works both as a package
# (import evaluation.proteins.scaffolding) and as a direct script
# (python evaluation/proteins/scaffolding.py). The insertion is idempotent and a
# no-op when the root is already on the path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The 20 canonical amino acids and their fixed ordering, reused from the data
# package so this module never reimplements the alphabet.
from data.protein_multimodal import CANONICAL_AA

__all__ = [
    "MotifSegment",
    "MotifProblem",
    "DPLM_MOTIF_RMSD_THRESHOLD",
    "DPLM_SCTM_THRESHOLD",
    "DEFAULT_NUM_CANDIDATES",
    "load_motif_problems",
    "sample_placement",
    "evaluate_problem",
    "run_scaffolding",
    "main",
]


# DPLM motif-scaffolding success thresholds (plan section 10.10): a candidate is
# a success when the motif backbone RMSD is below one Angstrom and the overall
# self-consistency TM-score is above 0.8.
DPLM_MOTIF_RMSD_THRESHOLD = 1.0
DPLM_SCTM_THRESHOLD = 0.8

# Number of scaffold candidates generated per motif problem by default.
DEFAULT_NUM_CANDIDATES = 100

# Index of the alpha-carbon atom inside the [L, 4, 3] backbone layout (N, CA, C, O).
_ATOM_CA = 1


# -----------------------------------------------------------------------------
# Motif problem representation and loading
# -----------------------------------------------------------------------------


@dataclass
class MotifSegment:
    """One contiguous motif segment with its known sequence and structure.

    ``seq_ids`` are canonical-20 amino-acid ids of shape ``[li]``. ``struct_index``
    are the DPLM-2 LFQ structure token ids of shape ``[li]`` (uint16 range). The
    optional ``coords`` are reference backbone coordinates of shape ``[li, 4, 3]``
    in atom order N, CA, C, O; when absent the reference structure is recovered by
    decoding ``struct_index`` with the DPLM tokenizer, which is the tokenizer
    reconstruction ceiling rather than the true experimental coordinates.
    """

    seq_ids: np.ndarray
    struct_index: np.ndarray
    coords: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.seq_ids = np.asarray(self.seq_ids, dtype=np.int64).reshape(-1)
        self.struct_index = np.asarray(
            self.struct_index, dtype=np.int64
        ).reshape(-1)
        if self.seq_ids.shape[0] != self.struct_index.shape[0]:
            raise ValueError(
                "MotifSegment seq_ids and struct_index must have equal length, got "
                f"{self.seq_ids.shape[0]} and {self.struct_index.shape[0]}"
            )
        if self.seq_ids.size and (
            self.seq_ids.min() < 0 or self.seq_ids.max() >= len(CANONICAL_AA)
        ):
            raise ValueError(
                f"seq_ids out of range [0, {len(CANONICAL_AA) - 1}]"
            )
        if self.coords is not None:
            self.coords = np.asarray(self.coords, dtype=np.float64)
            if self.coords.shape != (self.seq_ids.shape[0], 4, 3):
                raise ValueError(
                    "MotifSegment coords must have shape [li, 4, 3] with atoms "
                    f"N, CA, C, O; got {tuple(self.coords.shape)}"
                )

    @property
    def size(self) -> int:
        """Number of residues in this segment."""
        return int(self.seq_ids.shape[0])


@dataclass
class MotifProblem:
    """A motif scaffolding problem: motif segments plus a scaffold length range.

    ``segments`` are the ordered motif pieces to place, ``min_length`` and
    ``max_length`` bound the total design length (motif plus scaffold). The motif
    is placed contiguously segment by segment; scaffold residues fill the gaps
    before, between, and after the segments up to the sampled total length.
    """

    name: str
    segments: List[MotifSegment]
    min_length: int
    max_length: int
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError(f"motif problem {self.name!r} has no segments")
        self.min_length = int(self.min_length)
        self.max_length = int(self.max_length)
        if self.min_length <= 0 or self.max_length < self.min_length:
            raise ValueError(
                f"motif problem {self.name!r} has an invalid length range "
                f"[{self.min_length}, {self.max_length}]"
            )
        if self.max_length < self.motif_size:
            raise ValueError(
                f"motif problem {self.name!r} max_length {self.max_length} is smaller "
                f"than the motif size {self.motif_size}"
            )

    @property
    def motif_size(self) -> int:
        """Total number of motif residues across all segments."""
        return int(sum(segment.size for segment in self.segments))


def _seq_to_ids(seq: str) -> np.ndarray:
    """Map an amino-acid string to canonical-20 ids, rejecting unknown letters."""
    ids = []
    for character in seq:
        position = CANONICAL_AA.find(character)
        if position < 0:
            raise ValueError(
                f"amino acid {character!r} is not one of the canonical 20 "
                f"({CANONICAL_AA})"
            )
        ids.append(position)
    return np.asarray(ids, dtype=np.int64)


def load_motif_problems(path) -> List[MotifProblem]:
    """Load motif problems from a JSON specification file.

    The file holds an object with a ``problems`` list (a bare list is also
    accepted). Each problem has a ``name``, integer ``min_length`` and
    ``max_length``, and a ``segments`` list. Each segment provides either a
    ``seq`` amino-acid string or explicit ``seq_ids``, a ``struct_index`` list of
    LFQ token ids of the same length, and optionally ``coords`` of shape
    ``[li, 4, 3]`` giving the reference backbone. The specification is produced by
    a preparation step that tokenizes reference structures; see
    scripts/proteins/setup/tokenize_structures.py.
    """
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload["problems"] if isinstance(payload, dict) else payload
    problems: List[MotifProblem] = []
    for entry in entries:
        segments: List[MotifSegment] = []
        for segment in entry["segments"]:
            if "seq_ids" in segment:
                seq_ids = np.asarray(segment["seq_ids"], dtype=np.int64)
            else:
                seq_ids = _seq_to_ids(str(segment["seq"]))
            struct_index = np.asarray(segment["struct_index"], dtype=np.int64)
            coords = segment.get("coords")
            coords = (
                np.asarray(coords, dtype=np.float64)
                if coords is not None
                else None
            )
            segments.append(
                MotifSegment(
                    seq_ids=seq_ids, struct_index=struct_index, coords=coords
                )
            )
        problems.append(
            MotifProblem(
                name=str(entry["name"]),
                segments=segments,
                min_length=int(entry["min_length"]),
                max_length=int(entry["max_length"]),
                metadata=dict(entry.get("metadata", {})),
            )
        )
    return problems


# -----------------------------------------------------------------------------
# Motif placement
# -----------------------------------------------------------------------------


def _split_into_gaps(
    total: int, num_gaps: int, rng: np.random.Generator
) -> List[int]:
    """Distribute ``total`` scaffold residues into ``num_gaps`` non-negative bins."""
    if num_gaps <= 0:
        return []
    if total <= 0:
        return [0] * num_gaps
    counts = rng.multinomial(int(total), [1.0 / num_gaps] * num_gaps)
    return [int(value) for value in counts]


def sample_placement(
    problem: MotifProblem, rng: np.random.Generator
) -> Dict[str, object]:
    """Sample one placement of the motif inside a scaffold of a chosen length.

    A total design length is drawn uniformly from the problem length range (and
    raised to the motif size if necessary), and the scaffold residues are split
    into the gaps around the ordered motif segments. Returns a dictionary with the
    chosen length, the per-segment motif spans, the concatenated design positions
    of the clamped motif residues, and the reference sequence ids, structure ids,
    and backbone coordinates aligned to those positions. The design positions and
    the reference arrays share the same order, which is the position mapping that
    ties every generated motif residue back to its reference residue.
    """
    seg_sizes = [segment.size for segment in problem.segments]
    motif_total = int(sum(seg_sizes))
    length = int(rng.integers(problem.min_length, problem.max_length + 1))
    length = max(length, motif_total)
    gaps = _split_into_gaps(
        length - motif_total, len(problem.segments) + 1, rng
    )

    spans: List[Tuple[int, int]] = []
    positions: List[int] = []
    cursor = 0
    for index, segment in enumerate(problem.segments):
        cursor += gaps[index]
        start = cursor
        end = cursor + segment.size
        spans.append((start, end))
        positions.extend(range(start, end))
        cursor = end
    cursor += gaps[-1]
    # The gap sizes sum exactly to length minus the motif, so the cursor lands on
    # the chosen length; the max is a defensive guard that never shortens a span.
    length = max(length, cursor)

    design_positions = np.asarray(positions, dtype=np.int64)
    ref_seq_ids = np.concatenate(
        [segment.seq_ids for segment in problem.segments]
    )
    ref_struct_index = np.concatenate(
        [segment.struct_index for segment in problem.segments]
    ).astype(np.int64)

    if all(segment.coords is not None for segment in problem.segments):
        ref_coords: Optional[np.ndarray] = np.concatenate(
            [segment.coords for segment in problem.segments], axis=0
        )
    else:
        ref_coords = None

    seq_ids_full = np.zeros(length, dtype=np.int64)
    struct_index_full = np.zeros(length, dtype=np.int64)
    seq_ids_full[design_positions] = ref_seq_ids
    struct_index_full[design_positions] = ref_struct_index

    return {
        "length": int(length),
        "motif_spans": spans,
        "design_positions": design_positions,
        "ref_seq_ids": ref_seq_ids,
        "ref_struct_index": ref_struct_index,
        "ref_coords": ref_coords,
        "seq_ids_full": seq_ids_full,
        "struct_index_full": struct_index_full,
    }


# -----------------------------------------------------------------------------
# Lazy dependency loaders (kept out of import time)
# -----------------------------------------------------------------------------


def _load_generate():
    """Import the clamped multimodal generation driver, guarded and lazily."""
    try:
        from evaluation.proteins.generate_multimodal import generate
    except Exception as exc:  # noqa: BLE001 - re-raised with an actionable hint
        raise RuntimeError(
            "Motif scaffolding needs evaluation.proteins.generate_multimodal.generate "
            "to sample clamped completions. Ensure the protein evaluation package is "
            "importable; see scripts/proteins/setup/setup_evaluation.sh."
        ) from exc
    return generate


def _load_structure_metrics():
    """Import the independent TM-score and RMSD judge, guarded and lazily."""
    try:
        from evaluation.proteins import structure_metrics
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Motif scaffolding needs evaluation.proteins.structure_metrics for "
            "tm_score and rmsd. Ensure the protein evaluation package is importable; "
            "see scripts/proteins/setup/setup_evaluation.sh."
        ) from exc
    return structure_metrics


def _load_self_consistency(n_seqs: int):
    """Build a ProteinMPNN plus ESMFold self-consistency runner, guarded and lazily."""
    try:
        from evaluation.proteins.self_consistency import SelfConsistency
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Motif scaffolding needs evaluation.proteins.self_consistency for the "
            "whole-design self-consistency TM-score (ProteinMPNN plus ESMFold). "
            "Install the protein inference stack with "
            "scripts/proteins/setup/setup_evaluation.sh; see the protein-eval extra (uv sync --extra protein-eval) "
            "and the dplm-inference extra (uv sync --extra dplm-inference)."
        ) from exc
    return SelfConsistency()


def _load_tokenizer(tokenizer_path, device: str):
    """Load the frozen DPLM-2 structure tokenizer for decoding, guarded and lazily."""
    if tokenizer_path is None:
        raise RuntimeError(
            "Motif scaffolding decodes generated structure tokens to backbone "
            "coordinates with the DPLM-2 tokenizer, which needs a checkpoint path. "
            "Pass tokenizer_path (download it with "
            "scripts/proteins/setup/download_dplm_tokenizer.py) and install the "
            "DPLM-inference environment from the dplm-inference extra (uv sync --extra dplm-inference)."
        )
    from evaluation.proteins.dplm_struct_tokenizer import load_struct_tokenizer

    return load_struct_tokenizer(tokenizer_path, device=device)


def _load_model(
    config_path: str,
    checkpoint_path,
    device: torch.device,
    num_steps: Optional[int],
):
    """Build the diffusion model and config, applying the EMA weights, lazily.

    Thin wrapper over ``evaluation.proteins.io.load_binary_protein_model``, shared
    with the other protein generation drivers. Returns the ready-to-sample model
    and its config.
    """
    from evaluation.proteins.io import load_binary_protein_model

    return load_binary_protein_model(
        config_path,
        checkpoint_path,
        device,
        num_steps=num_steps,
        context="motif scaffolding",
    )


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------


def _ca_track(coords: np.ndarray) -> np.ndarray:
    """Return the [L, 3] alpha-carbon track from an [L, 4, 3] backbone array."""
    array = np.asarray(coords, dtype=np.float64)
    if array.ndim == 3 and array.shape[1] >= 2 and array.shape[2] == 3:
        return np.ascontiguousarray(array[:, _ATOM_CA, :])
    if array.ndim == 2 and array.shape[1] == 3:
        return np.ascontiguousarray(array)
    raise ValueError(
        "coordinates must have shape [L, 4, 3] or [L, 3], got "
        f"{tuple(array.shape)}"
    )


def _motif_backbone_rmsd(
    gen_motif: np.ndarray, ref_motif: np.ndarray, judge
) -> float:
    """Superposed backbone RMSD between two motif regions of shape [M, 4, 3].

    Every backbone atom (N, CA, C, O) of the motif participates: the two regions
    are flattened to [4M, 3] and superposed once before the deviation is measured,
    which is the DPLM motif-RMSD convention over backbone atoms.
    """
    gen = np.asarray(gen_motif, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(ref_motif, dtype=np.float64).reshape(-1, 3)
    return float(judge.rmsd(gen, ref, superpose=True))


def _motif_sequence_preservation(
    seq_string: str, design_positions: np.ndarray, ref_seq_ids: np.ndarray
) -> float:
    """Fraction of motif positions whose generated amino acid matches the reference."""
    if design_positions.size == 0:
        return float("nan")
    matches = 0
    for position, reference_id in zip(
        design_positions.tolist(), ref_seq_ids.tolist()
    ):
        if (
            position < len(seq_string)
            and seq_string[position] == CANONICAL_AA[int(reference_id)]
        ):
            matches += 1
    return float(matches) / float(design_positions.size)


def _greedy_tm_clusters(
    ca_list: Sequence[np.ndarray], judge, threshold: float
) -> int:
    """Count clusters of CA structures by greedy TM-score assignment at ``threshold``."""
    representatives: List[np.ndarray] = []
    for candidate in ca_list:
        assigned = False
        for representative in representatives:
            usable = min(len(candidate), len(representative))
            if usable == 0:
                continue
            if (
                float(
                    judge.tm_score(candidate[:usable], representative[:usable])
                )
                >= threshold
            ):
                assigned = True
                break
        if not assigned:
            representatives.append(candidate)
    return len(representatives)


def _diversity_summary(
    ca_list: Sequence[np.ndarray], judge, cluster_threshold: float = 0.6
) -> Dict[str, object]:
    """Summarise structural diversity within a set of CA backbones.

    Reports the number of structures, the mean pairwise TM-score, the derived
    diversity (one minus the mean pairwise TM-score), and the number and fraction
    of distinct structural clusters at the TM-score cluster threshold. Pairs of
    unequal length are compared over their shared prefix, which is a documented
    approximation used only for the diversity summary.
    """
    count = len(ca_list)
    if count < 2:
        return {
            "n": count,
            "mean_pairwise_tm": None,
            "diversity": None,
            "num_clusters": count,
            "cluster_fraction": float("nan") if count == 0 else 1.0,
            "cluster_threshold": float(cluster_threshold),
        }
    pairwise: List[float] = []
    for i in range(count):
        for j in range(i + 1, count):
            first = ca_list[i]
            second = ca_list[j]
            usable = min(len(first), len(second))
            if usable == 0:
                continue
            pairwise.append(
                float(judge.tm_score(first[:usable], second[:usable]))
            )
    mean_tm = float(np.mean(pairwise)) if pairwise else float("nan")
    num_clusters = _greedy_tm_clusters(ca_list, judge, cluster_threshold)
    return {
        "n": count,
        "mean_pairwise_tm": mean_tm,
        "diversity": (1.0 - mean_tm) if pairwise else float("nan"),
        "num_clusters": num_clusters,
        "cluster_fraction": num_clusters / float(count),
        "cluster_threshold": float(cluster_threshold),
    }


def _distribution_summary(
    values: Sequence[Optional[float]],
) -> Dict[str, object]:
    """Summarise a metric distribution and publish its raw per-candidate values.

    Returns the finite-value count, mean, standard deviation, minimum, maximum,
    median, and the 10th, 25th, 75th, and 90th percentiles, together with the full
    list of raw values (non-finite entries rendered as null) so downstream reports
    can rank on the continuous distribution rather than only on threshold counts.
    """
    raw: List[Optional[float]] = []
    finite: List[float] = []
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
        summary = {
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
    summary["values"] = raw
    return summary


def _length_bin(length: int, width: int) -> str:
    """Return the half-open scaffold-length bin label for a design length."""
    width = max(1, int(width))
    low = (int(length) // width) * width
    return f"{low}-{low + width}"


# -----------------------------------------------------------------------------
# Core evaluation
# -----------------------------------------------------------------------------


def evaluate_problem(
    problem: MotifProblem,
    *,
    generate_fn,
    tokenizer,
    self_consistency,
    judge,
    model,
    cfg,
    device,
    num_candidates: int,
    micro_batch_size: int,
    n_seqs: int,
    motif_rmsd_threshold: float,
    sc_tm_threshold: float,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, object]], List[np.ndarray]]:
    """Generate and score all candidates for one motif problem.

    Candidates are produced in micro-batches; every micro-batch draws a fresh
    motif placement (length and gaps), so the ``num_candidates`` completions span
    a range of scaffold lengths and motif positions. Each micro-batch is generated
    by a single clamped ``generate(task='motif', ...)`` call whose observed
    sequence ids, structure ids, and motif spans pin both modalities at the placed
    motif positions. Returns the per-candidate metric records and the list of
    alpha-carbon backbones for the candidates that were scored as successes.
    """
    records: List[Dict[str, object]] = []
    successful_ca: List[np.ndarray] = []
    produced = 0
    while produced < num_candidates:
        batch = min(int(micro_batch_size), num_candidates - produced)
        placement = sample_placement(problem, rng)
        design_positions = placement["design_positions"]
        length = int(placement["length"])

        observed = {
            "seq_ids": placement["seq_ids_full"],
            "struct_index": placement["struct_index_full"],
            "motif_spans": placement["motif_spans"],
        }
        generated = generate_fn(
            model, cfg, "motif", length, batch, device, observed=observed
        )
        seq_strings = generated["seq_strings"]
        struct_index = np.asarray(generated["struct_index"])

        # Reference motif backbone for this placement: use supplied coordinates
        # when present, otherwise decode the reference structure tokens once.
        if placement["ref_coords"] is not None:
            ref_motif_coords = np.asarray(
                placement["ref_coords"], dtype=np.float64
            )
        else:
            ref_motif_coords = np.asarray(
                tokenizer.decode(
                    placement["ref_struct_index"].astype(np.uint16)
                ),
                dtype=np.float64,
            )

        for candidate in range(batch):
            design_coords = np.asarray(
                tokenizer.decode(struct_index[candidate].astype(np.uint16)),
                dtype=np.float64,
            )
            gen_motif = design_coords[design_positions]
            motif_rmsd = _motif_backbone_rmsd(
                gen_motif, ref_motif_coords, judge
            )
            motif_ca_rmsd = float(
                judge.rmsd(
                    _ca_track(gen_motif),
                    _ca_track(ref_motif_coords),
                    superpose=True,
                )
            )
            preservation = _motif_sequence_preservation(
                seq_strings[candidate],
                design_positions,
                placement["ref_seq_ids"],
            )
            sc_result = self_consistency.run(design_coords, n_seqs=int(n_seqs))
            sc_tm = float(sc_result["sc_tm"])
            sc_rmsd = float(sc_result["sc_rmsd"])
            plddt = float(sc_result["plddt"])
            solved = bool(
                motif_rmsd < float(motif_rmsd_threshold)
                and sc_tm > float(sc_tm_threshold)
            )
            records.append(
                {
                    "problem": problem.name,
                    "length": length,
                    "motif_size": int(problem.motif_size),
                    "length_bin": None,  # filled in during aggregation
                    "motif_rmsd": motif_rmsd,
                    "motif_ca_rmsd": motif_ca_rmsd,
                    "motif_seq_preservation": preservation,
                    "sc_tm": sc_tm,
                    "sc_rmsd": sc_rmsd,
                    "plddt": plddt,
                    "solved": solved,
                }
            )
            if solved:
                successful_ca.append(_ca_track(design_coords))
        produced += batch
    return records, successful_ca


def run_scaffolding(
    model,
    cfg,
    problems: Sequence[MotifProblem],
    device,
    *,
    tokenizer_path=None,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
    micro_batch_size: int = 10,
    n_seqs: int = 8,
    motif_rmsd_threshold: float = DPLM_MOTIF_RMSD_THRESHOLD,
    sc_tm_threshold: float = DPLM_SCTM_THRESHOLD,
    length_bin_width: int = 25,
    seed: int = 0,
    generate_fn=None,
    tokenizer=None,
    self_consistency=None,
    judge=None,
) -> Dict[str, object]:
    """Run the full motif scaffolding evaluation over a set of motif problems.

    For every problem this samples ``num_candidates`` clamped completions, scores
    motif RMSD, motif sequence preservation, and whole-design self-consistency
    TM-score, and aggregates the DPLM solved-target statistics, the diversity among
    successful scaffolds, and the failures grouped by motif problem and by scaffold
    length. The continuous metric distributions are always included. The heavy
    dependencies are loaded lazily; they can also be injected for testing through
    ``generate_fn``, ``tokenizer``, ``self_consistency``, and ``judge``.
    """
    generate_fn = generate_fn if generate_fn is not None else _load_generate()
    judge = judge if judge is not None else _load_structure_metrics()
    self_consistency = (
        self_consistency
        if self_consistency is not None
        else _load_self_consistency(n_seqs)
    )
    device_string = str(getattr(device, "type", device))
    tokenizer = (
        tokenizer
        if tokenizer is not None
        else _load_tokenizer(tokenizer_path, device_string)
    )
    rng = np.random.default_rng(int(seed))

    all_records: List[Dict[str, object]] = []
    per_problem: Dict[str, object] = {}
    failures_by_motif: Dict[str, object] = {}
    overall_successful_ca: List[np.ndarray] = []

    for problem in problems:
        records, successful_ca = evaluate_problem(
            problem,
            generate_fn=generate_fn,
            tokenizer=tokenizer,
            self_consistency=self_consistency,
            judge=judge,
            model=model,
            cfg=cfg,
            device=device,
            num_candidates=num_candidates,
            micro_batch_size=micro_batch_size,
            n_seqs=n_seqs,
            motif_rmsd_threshold=motif_rmsd_threshold,
            sc_tm_threshold=sc_tm_threshold,
            rng=rng,
        )
        for record in records:
            record["length_bin"] = _length_bin(
                int(record["length"]), length_bin_width
            )
        all_records.extend(records)
        overall_successful_ca.extend(successful_ca)

        n_success = sum(1 for record in records if record["solved"])
        n_candidates = len(records)
        solved = n_success > 0
        per_problem[problem.name] = {
            "n_candidates": n_candidates,
            "n_success": n_success,
            "success_rate": (n_success / n_candidates)
            if n_candidates
            else 0.0,
            "solved": solved,
            "motif_size": int(problem.motif_size),
            "length_range": [int(problem.min_length), int(problem.max_length)],
            "diversity": _diversity_summary(successful_ca, judge),
            "distributions": {
                "motif_rmsd": _distribution_summary(
                    [r["motif_rmsd"] for r in records]
                ),
                "motif_seq_preservation": _distribution_summary(
                    [r["motif_seq_preservation"] for r in records]
                ),
                "sc_tm": _distribution_summary([r["sc_tm"] for r in records]),
            },
        }
        failures_by_motif[problem.name] = {
            "n_candidates": n_candidates,
            "n_failed": n_candidates - n_success,
            "fail_fraction": ((n_candidates - n_success) / n_candidates)
            if n_candidates
            else 0.0,
            "target_failed": not solved,
        }

    # Failures broken down by scaffold-length bin across every candidate.
    failures_by_length: Dict[str, Dict[str, object]] = {}
    for record in all_records:
        label = str(record["length_bin"])
        bucket = failures_by_length.setdefault(
            label, {"n_candidates": 0, "n_failed": 0, "fail_fraction": 0.0}
        )
        bucket["n_candidates"] += 1
        if not record["solved"]:
            bucket["n_failed"] += 1
    for bucket in failures_by_length.values():
        total = bucket["n_candidates"]
        bucket["fail_fraction"] = (
            (bucket["n_failed"] / total) if total else 0.0
        )

    num_targets = len(problems)
    num_solved_targets = sum(
        1 for value in per_problem.values() if value["solved"]
    )
    total_candidates = len(all_records)
    total_success = sum(1 for record in all_records if record["solved"])

    distributions = {
        "motif_rmsd": _distribution_summary(
            [r["motif_rmsd"] for r in all_records]
        ),
        "motif_ca_rmsd": _distribution_summary(
            [r["motif_ca_rmsd"] for r in all_records]
        ),
        "motif_seq_preservation": _distribution_summary(
            [r["motif_seq_preservation"] for r in all_records]
        ),
        "sc_tm": _distribution_summary([r["sc_tm"] for r in all_records]),
        "sc_rmsd": _distribution_summary([r["sc_rmsd"] for r in all_records]),
        "plddt": _distribution_summary([r["plddt"] for r in all_records]),
    }

    return {
        "thresholds": {
            "motif_rmsd": float(motif_rmsd_threshold),
            "sc_tm": float(sc_tm_threshold),
            "convention": "DPLM: motif RMSD < 1A and overall scTM > 0.8",
        },
        "targets": {
            "num_targets": num_targets,
            "num_solved_targets": num_solved_targets,
            "fraction_solved_targets": (num_solved_targets / num_targets)
            if num_targets
            else 0.0,
        },
        "candidates": {
            "num_candidates": total_candidates,
            "num_success": total_success,
            "success_rate": (total_success / total_candidates)
            if total_candidates
            else 0.0,
        },
        "diversity_successful_scaffolds": _diversity_summary(
            overall_successful_ca, judge
        ),
        "failures_by_motif": failures_by_motif,
        "failures_by_length": failures_by_length,
        "per_problem": per_problem,
        "distributions": distributions,
        "candidates_detail": all_records,
    }


# -----------------------------------------------------------------------------
# Command-line entry point
# -----------------------------------------------------------------------------


def _git(*args: str) -> Optional[str]:
    """Return the trimmed output of a git command, or None if git fails."""
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def _jsonable(obj):
    """Recursively coerce numpy scalars and arrays into JSON-serialisable values."""
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    """Write JSON to a temporary sibling file and atomically replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main() -> None:
    """Command-line driver: evaluate motif scaffolding and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to the binary model config"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--problems",
        required=True,
        help="Path to the motif problem JSON specification (see load_motif_problems)",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Local DPLM-2 structure tokenizer checkpoint for decoding backbones",
    )
    parser.add_argument(
        "--out", required=True, help="Path to the output JSON report"
    )
    parser.add_argument(
        "--num-candidates", type=int, default=DEFAULT_NUM_CANDIDATES
    )
    parser.add_argument("--micro-batch-size", type=int, default=10)
    parser.add_argument("--n-seqs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument(
        "--motif-rmsd-threshold", type=float, default=DPLM_MOTIF_RMSD_THRESHOLD
    )
    parser.add_argument(
        "--sc-tm-threshold", type=float, default=DPLM_SCTM_THRESHOLD
    )
    parser.add_argument("--length-bin-width", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    problems = load_motif_problems(args.problems)
    model, cfg = _load_model(
        args.config, args.checkpoint, device, args.num_steps
    )

    results = run_scaffolding(
        model,
        cfg,
        problems,
        device,
        tokenizer_path=args.tokenizer_path,
        num_candidates=args.num_candidates,
        micro_batch_size=args.micro_batch_size,
        n_seqs=args.n_seqs,
        motif_rmsd_threshold=args.motif_rmsd_threshold,
        sc_tm_threshold=args.sc_tm_threshold,
        length_bin_width=args.length_bin_width,
        seed=args.seed,
    )

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "motif_scaffolding",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "problems_path": str(args.problems),
        "parameters": {
            "num_candidates": args.num_candidates,
            "micro_batch_size": args.micro_batch_size,
            "n_seqs": args.n_seqs,
            "num_steps": args.num_steps,
            "motif_rmsd_threshold": args.motif_rmsd_threshold,
            "sc_tm_threshold": args.sc_tm_threshold,
            "length_bin_width": args.length_bin_width,
            "seed": args.seed,
            "device": str(device),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
        },
        "results": results,
    }
    out_path = Path(args.out)
    _atomic_write_json(out_path, payload)

    targets = results["targets"]
    print(
        "Solved {solved}/{total} targets ({fraction:.3f}); wrote {path}".format(
            solved=targets["num_solved_targets"],
            total=targets["num_targets"],
            fraction=targets["fraction_solved_targets"],
            path=out_path,
        )
    )


if __name__ == "__main__":
    main()
