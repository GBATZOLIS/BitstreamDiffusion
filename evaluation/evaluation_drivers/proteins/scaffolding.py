"""Generate protein scaffolds around fixed motifs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

# Make the repository root importable so the module works both as a
# package (import evaluation.evaluation_drivers.proteins.scaffolding) and
# as a direct script
# (python evaluation/evaluation_drivers/proteins/scaffolding.py).
# The insertion is
# idempotent and a no-op when the root is already on the path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The 20 canonical amino acids and their fixed ordering, reused from the
# data package so this module never reimplements the alphabet.
from data.protein_multimodal import CANONICAL_AA
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
    "MotifSegment",
    "MotifProblem",
    "ScaffoldWorkUnit",
    "DPLM_MOTIF_RMSD_THRESHOLD",
    "DPLM_SCTM_THRESHOLD",
    "DEFAULT_NUM_CANDIDATES",
    "load_motif_problems",
    "sample_placement",
    "evaluate_problem",
    "build_scaffold_work_units",
    "problem_unit",
    "write_scaffold_info_files",
    "aggregate_scaffolding",
    "run_scaffolding",
    "main",
]


# DPLM motif-scaffolding success thresholds: a candidate is a success
# when the motif backbone RMSD is below one Angstrom and the overall
# self-consistency TM-score is above 0.8.
DPLM_MOTIF_RMSD_THRESHOLD = 1.0
DPLM_SCTM_THRESHOLD = 0.8

# Number of scaffold candidates generated per motif problem by default.
DEFAULT_NUM_CANDIDATES = 100

# Index of the alpha-carbon atom inside the [L, 4, 3] backbone layout
# (N, CA, C, O).
_ATOM_CA = 1


@dataclass(frozen=True)
class ScaffoldWorkUnit:
    """A restart-safe candidate block for one MotifBench problem."""

    problem: MotifProblem
    start: int
    count: int

    @property
    def key(self) -> str:
        stop = self.start + self.count
        return (
            f"{self.problem.name}/candidates-{self.start:03d}-{stop - 1:03d}"
        )


def build_scaffold_work_units(
    problems: Sequence[MotifProblem],
    num_candidates: int,
    unit_size: int,
) -> list[ScaffoldWorkUnit]:
    """Split each problem into stable candidate blocks for four-hour resume."""
    num_candidates = int(num_candidates)
    unit_size = int(unit_size)
    if num_candidates < 1:
        raise ValueError("num_candidates must be at least one")
    if unit_size < 1:
        raise ValueError("scaffold resume unit size must be at least one")
    return [
        ScaffoldWorkUnit(
            problem=problem,
            start=start,
            count=min(unit_size, num_candidates - start),
        )
        for problem in problems
        for start in range(0, num_candidates, unit_size)
    ]


# -----------------------------------------------------------------------------
# Motif problem representation and loading
# -----------------------------------------------------------------------------


@dataclass
class MotifSegment:
    """One contiguous motif segment with its known sequence and structure."""

    seq_ids: np.ndarray
    struct_index: np.ndarray | None = None
    coords: np.ndarray | None = None
    seq_observed_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.seq_ids = np.asarray(self.seq_ids, dtype=np.int64).reshape(-1)
        if self.struct_index is not None:
            self.struct_index = np.asarray(
                self.struct_index, dtype=np.int64
            ).reshape(-1)
        if (
            self.struct_index is not None
            and self.seq_ids.shape[0] != self.struct_index.shape[0]
        ):
            raise ValueError(
                f"MotifSegment seq_ids and struct_index must have equal "
                f"length, got {self.seq_ids.shape[0]} and "
                f"{self.struct_index.shape[0]}"
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
                    f"MotifSegment coords must have shape [li, 4, 3] with "
                    f"atoms N, CA, C, O; got {tuple(self.coords.shape)}"
                )
        if self.struct_index is None and self.coords is None:
            raise ValueError(
                "MotifSegment needs either struct_index or reference "
                "backbone coords"
            )
        if self.seq_observed_mask is None:
            self.seq_observed_mask = np.ones(self.seq_ids.shape[0], dtype=bool)
        else:
            self.seq_observed_mask = np.asarray(
                self.seq_observed_mask, dtype=bool
            ).reshape(-1)
            if self.seq_observed_mask.shape[0] != self.seq_ids.shape[0]:
                raise ValueError(
                    "seq_observed_mask must have one value per motif residue"
                )

    @property
    def size(self) -> int:
        """Number of residues in this segment."""
        return int(self.seq_ids.shape[0])


@dataclass
class MotifProblem:
    """
    A motif scaffolding problem: motif segments plus a scaffold length
    range.
    """

    name: str
    segments: list[MotifSegment]
    min_length: int
    max_length: int
    metadata: dict[str, object] = field(default_factory=dict)
    # Per-gap ranges from the benchmark's contig string, in order: the
    # gap before the first segment, between each pair, and after the
    # last, so there are len(segments) + 1 of them. When present these
    # drive placement, because they are the benchmark's actual
    # specification; min_length / max_length then act as the additional
    # total-length constraint that RFdiffusion applies with
    # --contigmap.length.
    #
    # When absent, placement falls back to a uniform total length plus a
    # uniform composition into gaps, which is the best available
    # approximation and is flagged in the report so a reader knows the
    # problem was not run under its own contig.
    gap_ranges: list[tuple[int, int]] | None = None
    contig: str = ""

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError(f"motif problem {self.name!r} has no segments")
        self.min_length = int(self.min_length)
        self.max_length = int(self.max_length)
        if self.gap_ranges is not None:
            self.gap_ranges = [(int(a), int(b)) for a, b in self.gap_ranges]
            if len(self.gap_ranges) != len(self.segments) + 1:
                raise ValueError(
                    f"motif problem {self.name!r} has {len(self.segments)}"
                    f" segments so it needs {len(self.segments) + 1} gap "
                    f"ranges, got {len(self.gap_ranges)}"
                )
            for a, b in self.gap_ranges:
                if a < 0 or b < a:
                    raise ValueError(
                        f"motif problem {self.name!r} has an invalid gap "
                        f"range ({a}, {b})"
                    )
        if self.min_length <= 0 or self.max_length < self.min_length:
            raise ValueError(
                f"motif problem {self.name!r} has an invalid length range "
                f"[{self.min_length}, {self.max_length}]"
            )
        if self.max_length < self.motif_size:
            raise ValueError(
                f"motif problem {self.name!r} max_length {self.max_length}"
                f" is smaller than the motif size {self.motif_size}"
            )

    @property
    def motif_size(self) -> int:
        """Total number of motif residues across all segments."""
        return int(sum(segment.size for segment in self.segments))


def _seq_to_ids(seq: str) -> np.ndarray:
    """
    Map an amino-acid string to canonical-20 ids, rejecting unknown
    letters.
    """
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


def load_motif_problems(path) -> list[MotifProblem]:
    """Load motif problems from a JSON specification file."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload["problems"] if isinstance(payload, dict) else payload
    problems: list[MotifProblem] = []
    for entry in entries:
        segments: list[MotifSegment] = []
        for segment in entry["segments"]:
            if "seq_ids" in segment:
                seq_ids = np.asarray(segment["seq_ids"], dtype=np.int64)
            else:
                seq_ids = _seq_to_ids(str(segment["seq"]))
            struct_index_raw = segment.get("struct_index")
            struct_index = (
                np.asarray(struct_index_raw, dtype=np.int64)
                if struct_index_raw is not None
                else None
            )
            coords = segment.get("coords")
            coords = (
                np.asarray(coords, dtype=np.float64)
                if coords is not None
                else None
            )
            segments.append(
                MotifSegment(
                    seq_ids=seq_ids,
                    struct_index=struct_index,
                    coords=coords,
                    seq_observed_mask=segment.get("seq_observed_mask"),
                )
            )
        gap_ranges = entry.get("gap_ranges")
        if gap_ranges is not None:
            gap_ranges = [(int(a), int(b)) for a, b in gap_ranges]
        problems.append(
            MotifProblem(
                name=str(entry["name"]),
                segments=segments,
                min_length=int(entry["min_length"]),
                max_length=int(entry["max_length"]),
                metadata=dict(entry.get("metadata", {})),
                gap_ranges=gap_ranges,
                contig=str(entry.get("contig", "")),
            )
        )
    return problems


def parse_contig(contig: str) -> tuple[list[tuple[int, int]], list[str]]:
    """Split an RFdiffusion contig string into gap ranges and motif tokens."""
    gaps: list[tuple[int, int]] = []
    motifs: list[str] = []
    expect_gap = True
    for raw in str(contig).split(","):
        token = raw.strip()
        if not token:
            continue
        if token[0].isalpha():
            if expect_gap:
                gaps.append((0, 0))  # no scaffold before this segment
            motifs.append(token)
            expect_gap = True
        else:
            lo, _, hi = token.partition("-")
            span = (int(lo), int(hi or lo))
            gaps.append(span)
            expect_gap = False
    if expect_gap:
        gaps.append((0, 0))  # no scaffold after the last segment
    if len(gaps) != len(motifs) + 1:
        raise ValueError(
            f"contig {contig!r} parsed to {len(gaps)} gaps for "
            f"{len(motifs)} segments; expected {len(motifs) + 1}"
        )
    return gaps, motifs


# -----------------------------------------------------------------------------
# Motif placement
# -----------------------------------------------------------------------------


def _split_into_gaps(
    total: int, num_gaps: int, rng: np.random.Generator
) -> list[int]:
    """Distribute total scaffold residues into num_gaps bins, UNIFORMLY."""
    if num_gaps <= 0:
        return []
    total = int(total)
    if total <= 0:
        return [0] * num_gaps
    if num_gaps == 1:
        return [total]
    cuts = np.sort(
        rng.choice(total + num_gaps - 1, size=num_gaps - 1, replace=False)
    )
    bounds = np.concatenate([[-1], cuts, [total + num_gaps - 1]])
    return [
        int(b - a - 1) for a, b in zip(bounds[:-1], bounds[1:], strict=True)
    ]


def _sample_gaps_from_ranges(
    gap_ranges: Sequence[tuple[int, int]],
    rng: np.random.Generator,
    *,
    min_total: int | None = None,
    max_total: int | None = None,
    motif_total: int = 0,
    max_tries: int = 200,
) -> list[int]:
    """Sample each gap independently from its own range (contig semantics)."""
    lo = [int(a) for a, _ in gap_ranges]
    hi = [int(b) for _, b in gap_ranges]

    constrained = min_total is not None or max_total is not None
    if not constrained:
        return [
            int(rng.integers(a, b + 1)) for a, b in zip(lo, hi, strict=True)
        ]

    # With a total-length constraint, drawing each gap independently and
    # rejecting is badly inefficient: for 7MRX (two gaps of 0-38, total
    # pinned at 60) only ~1.4% of independent draws hit, so a bounded
    # retry budget silently returns off-length designs a few percent of
    # the time -- and an off-length design is scored against the wrong
    # problem.
    #
    # Sample the total FIRST, uniformly over the feasible range, then
    # draw a uniform composition of the remaining residues and reject
    # only on the per-gap bounds. That moves the same 7MRX case from
    # ~1.4% to ~39% per try.
    lo_total = (
        int(min_total) if min_total is not None else int(motif_total + sum(lo))
    )
    hi_total = (
        int(max_total) if max_total is not None else int(motif_total + sum(hi))
    )
    lo_total = max(lo_total, int(motif_total + sum(lo)))
    hi_total = min(hi_total, int(motif_total + sum(hi)))
    if hi_total < lo_total:
        # The contig and the length bounds are inconsistent. Honour the
        # contig, which is the benchmark's actual specification, and let
        # the reported length show the disagreement rather than silently
        # picking one.
        return [
            int(rng.integers(a, b + 1)) for a, b in zip(lo, hi, strict=True)
        ]

    best: list[int] | None = None
    best_penalty: int | None = None
    for _ in range(max_tries):
        total = int(rng.integers(lo_total, hi_total + 1))
        draw = _split_into_gaps(total - int(motif_total), len(lo), rng)
        penalty = sum(
            max(0, a - g) + max(0, g - b)
            for g, a, b in zip(draw, lo, hi, strict=True)
        )
        if penalty == 0:
            return draw
        if best_penalty is None or penalty < best_penalty:
            best, best_penalty = draw, penalty
    # Fall back to clipping into range. The total then drifts, which is
    # exactly what the caller sees in the reported design length.
    if best is None:
        return list(lo)
    return [min(max(g, a), b) for g, a, b in zip(best, lo, hi, strict=True)]


def sample_placement(
    problem: MotifProblem, rng: np.random.Generator
) -> dict[str, object]:
    """
    Sample one placement of the motif inside a scaffold of a chosen
    length.
    """
    seg_sizes = [segment.size for segment in problem.segments]
    motif_total = int(sum(seg_sizes))

    if problem.gap_ranges is not None:
        # Contig semantics: each gap is drawn from its own range and the
        # total design length falls out of that draw. This is the
        # benchmark's actual specification, so it takes precedence over
        # the length bounds.
        gaps = _sample_gaps_from_ranges(
            problem.gap_ranges,
            rng,
            min_total=problem.min_length,
            max_total=problem.max_length,
            motif_total=motif_total,
        )
        length = motif_total + int(sum(gaps))
    else:
        length = int(rng.integers(problem.min_length, problem.max_length + 1))
        length = max(length, motif_total)
        gaps = _split_into_gaps(
            length - motif_total, len(problem.segments) + 1, rng
        )

    spans: list[tuple[int, int]] = []
    positions: list[int] = []
    cursor = 0
    for index, segment in enumerate(problem.segments):
        cursor += gaps[index]
        start = cursor
        end = cursor + segment.size
        spans.append((start, end))
        positions.extend(range(start, end))
        cursor = end
    cursor += gaps[-1]
    # The gap sizes sum exactly to length minus the motif, so the cursor
    # lands on the chosen length; the max is a defensive guard that
    # never shortens a span.
    length = max(length, cursor)

    design_positions = np.asarray(positions, dtype=np.int64)
    ref_seq_ids = np.concatenate(
        [segment.seq_ids for segment in problem.segments]
    )
    if any(segment.struct_index is None for segment in problem.segments):
        raise ValueError(
            f"motif problem {problem.name!r} has untokenized coordinates; "
            f"resolve them with the structure tokenizer before sampling "
            f"placement"
        )
    ref_struct_index = np.concatenate(
        [segment.struct_index for segment in problem.segments]
    ).astype(np.int64)
    ref_seq_observed = np.concatenate(
        [segment.seq_observed_mask for segment in problem.segments]
    ).astype(bool)

    if all(segment.coords is not None for segment in problem.segments):
        ref_coords: np.ndarray | None = np.concatenate(
            [segment.coords for segment in problem.segments], axis=0
        )
    else:
        ref_coords = None

    seq_ids_full = np.zeros(length, dtype=np.int64)
    struct_index_full = np.zeros(length, dtype=np.int64)
    seq_observed_mask = np.zeros(length, dtype=bool)
    struct_observed_mask = np.zeros(length, dtype=bool)
    seq_ids_full[design_positions] = ref_seq_ids
    struct_index_full[design_positions] = ref_struct_index
    seq_observed_mask[design_positions] = ref_seq_observed
    struct_observed_mask[design_positions] = True

    return {
        "length": int(length),
        "motif_spans": spans,
        "gaps": [int(gap) for gap in gaps],
        "design_positions": design_positions,
        "ref_seq_ids": ref_seq_ids,
        "ref_seq_observed": ref_seq_observed,
        "ref_struct_index": ref_struct_index,
        "ref_coords": ref_coords,
        "seq_ids_full": seq_ids_full,
        "struct_index_full": struct_index_full,
        "seq_observed_mask": seq_observed_mask,
        "struct_observed_mask": struct_observed_mask,
    }


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
            "Motif scaffolding needs "
            "evaluation.evaluation_drivers.proteins."
            "generate_multimodal.generate to sample "
            "clamped completions. Ensure the protein evaluation package "
            "is importable; see ./setup.sh eval."
        ) from exc
    return generate_multimodal.generate


def _load_structure_metrics():
    """Import the independent TM-score and RMSD judge, guarded and lazily."""
    try:
        from evaluation.evaluation_drivers.proteins import structure_metrics
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Motif scaffolding needs "
            "evaluation.evaluation_drivers.proteins.structure_metrics "
            "for tm_score and "
            "rmsd. Ensure the protein evaluation package is importable; "
            "see ./setup.sh eval."
        ) from exc
    return structure_metrics


def _load_self_consistency(n_seqs: int):
    """
    Build a ProteinMPNN plus ESMFold self-consistency runner, guarded
    and lazily.
    """
    try:
        from evaluation.evaluation_drivers.proteins.self_consistency import (
            SelfConsistency,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Motif scaffolding needs "
            "evaluation.evaluation_drivers.proteins.self_consistency "
            "for the whole-design "
            "self-consistency TM-score (ProteinMPNN plus ESMFold). "
            "Install the protein inference stack with ./setup.sh eval; "
            "see the eval extra (uv sync --extra eval) and the eval "
            "extra (uv sync --extra eval)."
        ) from exc
    return SelfConsistency()


def _score_generated_pair(sequence: str, backbone: np.ndarray, judge) -> dict:
    """
    Independently refold the generated sequence against its paired
    structure.
    """
    from evaluation.evaluation_drivers.proteins.self_consistency import (
        esmfold_plddt,
    )

    folded, plddt = esmfold_plddt(sequence)
    folded_ca = _ca_track(folded)
    reference_ca = _ca_track(backbone)
    usable = min(len(folded_ca), len(reference_ca))
    if usable == 0:
        raise RuntimeError("independent refold produced an empty structure")
    folded_ca = folded_ca[:usable]
    reference_ca = reference_ca[:usable]
    return {
        "pair_sc_tm": float(judge.tm_score(folded_ca, reference_ca)),
        "pair_sc_rmsd": float(
            judge.rmsd(folded_ca, reference_ca, superpose=True)
        ),
        "pair_sc_plddt": float(np.mean(plddt)),
    }


def _load_tokenizer(tokenizer_path, device: str):
    """
    Load the frozen DPLM-2 structure tokenizer for decoding, guarded and
    lazily.
    """
    if tokenizer_path is None:
        raise RuntimeError(
            "Motif scaffolding decodes generated structure tokens to "
            "backbone coordinates with the DPLM-2 tokenizer, which needs"
            " a checkpoint path. Pass tokenizer_path (download it with "
            "scripts/proteins/setup/download_dplm_tokenizer.py) and "
            "install the DPLM-inference environment from the eval extra "
            "(uv sync --extra eval)."
        )
    from evaluation.evaluation_drivers.proteins.dplm_struct_tokenizer import (
        load_struct_tokenizer,
    )

    return load_struct_tokenizer(tokenizer_path, device=device)


def _load_model(
    config_path: str,
    checkpoint_path,
    device: torch.device,
    num_steps: int | None,
):
    """
    Build the diffusion model and config, applying the EMA weights,
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
        context="motif scaffolding",
    )


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------


def _ca_track(coords: np.ndarray) -> np.ndarray:
    """
    Return the [L, 3] alpha-carbon track from an [L, 4, 3] backbone
    array.
    """
    array = np.asarray(coords, dtype=np.float64)
    if array.ndim == 3 and array.shape[1] >= 2 and array.shape[2] == 3:
        return np.ascontiguousarray(array[:, _ATOM_CA, :])
    if array.ndim == 2 and array.shape[1] == 3:
        return np.ascontiguousarray(array)
    raise ValueError(
        f"coordinates must have shape [L, 4, 3] or [L, 3], got "
        f"{tuple(array.shape)}"
    )


def _motif_backbone_rmsd(
    gen_motif: np.ndarray, ref_motif: np.ndarray, judge
) -> float:
    """
    Superposed backbone RMSD between two motif regions of shape [M, 4,
    3].
    """
    gen = np.asarray(gen_motif, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(ref_motif, dtype=np.float64).reshape(-1, 3)
    return float(judge.rmsd(gen, ref, superpose=True))


def _motif_sequence_preservation(
    seq_string: str,
    design_positions: np.ndarray,
    ref_seq_ids: np.ndarray,
    seq_observed_mask: np.ndarray | None = None,
) -> float:
    """Recovery on sequence-constrained motif positions only."""
    observed = (
        np.ones(design_positions.shape[0], dtype=bool)
        if seq_observed_mask is None
        else np.asarray(seq_observed_mask, dtype=bool).reshape(-1)
    )
    if observed.shape[0] != design_positions.shape[0]:
        raise ValueError(
            "motif sequence mask length does not match motif positions"
        )
    if not bool(observed.any()):
        return float("nan")
    matches = 0
    for position, reference_id in zip(
        design_positions[observed].tolist(),
        ref_seq_ids[observed].tolist(),
        strict=True,
    ):
        if (
            position < len(seq_string)
            and seq_string[position] == CANONICAL_AA[int(reference_id)]
        ):
            matches += 1
    return float(matches) / float(observed.sum())


def _resolve_problem_structure_tokens(
    problem: MotifProblem, tokenizer
) -> None:
    """
    Encode official coordinate-only motif segments exactly once per
    problem.
    """
    for segment in problem.segments:
        if segment.struct_index is None:
            if segment.coords is None:
                raise ValueError(
                    f"problem {problem.name!r} has no coordinates to tokenize"
                )
            encoded = np.asarray(
                tokenizer.encode(np.asarray(segment.coords, dtype=np.float32))
            ).reshape(-1)
            if encoded.shape[0] != segment.size:
                raise ValueError(
                    f"tokenizer returned {encoded.shape[0]} residues for a"
                    f" {segment.size}-residue motif segment"
                )
            segment.struct_index = encoded.astype(np.int64)


def _motifbench_placement(
    gaps: Sequence[int], segment_labels: Sequence[str]
) -> str:
    """Return MotifBench's 12/A/15/B/29 placement notation."""
    labels = [str(label) for label in segment_labels]
    parts: list[str] = []
    for index, label in enumerate(labels):
        if int(gaps[index]) > 0:
            parts.append(str(int(gaps[index])))
        parts.append(label)
    if int(gaps[-1]) > 0:
        parts.append(str(int(gaps[-1])))
    return "/".join(parts)


def _greedy_tm_clusters(
    ca_list: Sequence[np.ndarray], judge, threshold: float
) -> int:
    """
    Count clusters of CA structures by greedy TM-score assignment at
    threshold.
    """
    representatives: list[np.ndarray] = []
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
) -> dict[str, object]:
    """Summarise structural diversity within a set of CA backbones."""
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
    pairwise: list[float] = []
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
    values: Sequence[float | None],
) -> dict[str, object]:
    """
    Summarise a metric distribution and publish its raw per-candidate
    values.
    """
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
        from evaluation.evaluation_drivers.proteins.statistics import (
            bootstrap_mean_ci,
        )

        summary = {
            "count": int(array.size),
            "mean": float(array.mean()),
            "mean_ci95": bootstrap_mean_ci(array),
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
    structures_dir: Path | None = None,
    sample_offset: int = 0,
    write_scaffold_info: bool = True,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    """Generate and score all candidates for one motif problem."""
    records: list[dict[str, object]] = []
    successful_ca: list[np.ndarray] = []
    _resolve_problem_structure_tokens(problem, tokenizer)
    case_dir = None
    scaffold_rows: list[dict[str, object]] = []
    if structures_dir is not None:
        case_dir = Path(structures_dir) / problem.name
        case_dir.mkdir(parents=True, exist_ok=True)
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
            "seq_observed_mask": placement["seq_observed_mask"],
            "struct_observed_mask": placement["struct_observed_mask"],
        }
        generated = generate_fn(
            model, cfg, "motif", length, batch, device, observed=observed
        )
        seq_strings = generated["seq_strings"]
        struct_index = np.asarray(generated["struct_index"])

        # Reference motif backbone for this placement: use supplied
        # coordinates when present, otherwise decode the reference
        # structure tokens once.
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
            generated_motif_ca = _ca_track(gen_motif)
            reference_motif_ca = _ca_track(ref_motif_coords)
            generated_distances = np.linalg.norm(
                generated_motif_ca[:, None, :]
                - generated_motif_ca[None, :, :],
                axis=-1,
            )
            reference_distances = np.linalg.norm(
                reference_motif_ca[:, None, :]
                - reference_motif_ca[None, :, :],
                axis=-1,
            )
            upper = np.triu_indices(len(generated_motif_ca), k=1)
            motif_pair_distance_mae = (
                float(
                    np.mean(
                        np.abs(
                            generated_distances[upper]
                            - reference_distances[upper]
                        )
                    )
                )
                if upper[0].size
                else 0.0
            )
            geometry = judge.backbone_geometry(design_coords)
            preservation = _motif_sequence_preservation(
                seq_strings[candidate],
                design_positions,
                placement["ref_seq_ids"],
                placement["ref_seq_observed"],
            )
            sc_result = self_consistency.run(design_coords, n_seqs=int(n_seqs))
            sc_tm = float(sc_result["sc_tm"])
            sc_rmsd = float(sc_result["sc_rmsd"])
            plddt = float(sc_result["plddt"])
            pair_sc = _score_generated_pair(
                seq_strings[candidate], design_coords, judge
            )
            solved = bool(
                motif_rmsd < float(motif_rmsd_threshold)
                and sc_tm > float(sc_tm_threshold)
            )
            sample_num = int(sample_offset) + produced + candidate
            segment_order = list(problem.metadata.get("segment_order", []))
            segment_labels = (
                [str(token)[0] for token in segment_order]
                if segment_order
                else [chr(ord("A") + i) for i in range(len(problem.segments))]
            )
            placement_text = _motifbench_placement(
                placement["gaps"], segment_labels
            )
            structure_path = None
            if case_dir is not None:
                from evaluation.evaluation_drivers.proteins.pdb_io import (
                    write_backbone_pdb,
                )

                structure_path = case_dir / f"{problem.name}_{sample_num}.pdb"
                write_backbone_pdb(
                    structure_path,
                    design_coords,
                    sequence=seq_strings[candidate],
                )
                scaffold_rows.append(
                    {
                        "sample_num": sample_num,
                        "motif_placements": placement_text,
                    }
                )
            records.append(
                {
                    "problem": problem.name,
                    "sample_num": sample_num,
                    "length": length,
                    "motif_size": int(problem.motif_size),
                    "length_bin": None,  # filled in during aggregation
                    "motif_rmsd": motif_rmsd,
                    "motif_ca_rmsd": motif_ca_rmsd,
                    "motif_pair_distance_mae": motif_pair_distance_mae,
                    "motif_seq_preservation": preservation,
                    "num_sequence_constrained_motif_residues": int(
                        np.asarray(
                            placement["ref_seq_observed"], dtype=bool
                        ).sum()
                    ),
                    "motif_placements": placement_text,
                    "structure_path": (
                        str(structure_path)
                        if structure_path is not None
                        else None
                    ),
                    "sc_tm": sc_tm,
                    "sc_rmsd": sc_rmsd,
                    "plddt": plddt,
                    "backbone_clash_count": int(geometry["clash_count"]),
                    "backbone_chain_break_count": int(
                        geometry["chain_break_count"]
                    ),
                    **pair_sc,
                    "solved": solved,
                }
            )
            if solved:
                successful_ca.append(_ca_track(design_coords))
        produced += batch
    if case_dir is not None and write_scaffold_info:
        csv_path = case_dir / "scaffold_info.csv"
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["sample_num", "motif_placements"]
            )
            writer.writeheader()
            writer.writerows(scaffold_rows)
        os.replace(temporary, csv_path)
    return records, successful_ca


def problem_unit(
    problem: MotifProblem,
    records: Sequence[dict[str, object]],
    successful_ca: Sequence[np.ndarray],
    length_bin_width: int,
    candidate_start: int = 0,
) -> dict[str, object]:
    """
    Package one motif problem's candidates as a shardable, serialisable
    unit.
    """
    for record in records:
        record["length_bin"] = _length_bin(
            int(record["length"]), length_bin_width
        )
    return {
        "problem": problem.name,
        "motif_size": int(problem.motif_size),
        "length_range": [int(problem.min_length), int(problem.max_length)],
        "contig": problem.contig,
        "segment_order": list(problem.metadata.get("segment_order", [])),
        "placement_policy": problem.metadata.get("placement_policy"),
        "candidate_start": int(candidate_start),
        "candidates": list(records),
        "successful_ca": [
            np.asarray(ca, dtype=np.float64).tolist() for ca in successful_ca
        ],
    }


def _consolidate_scaffolding_units(
    units: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Reassemble restart blocks into one ordered unit per motif problem."""
    combined: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for source in units:
        name = str(source["problem"])
        if name not in combined:
            order.append(name)
            combined[name] = {
                key: value
                for key, value in source.items()
                if key
                not in {"candidates", "successful_ca", "candidate_start"}
            }
            combined[name]["candidates"] = []
            combined[name]["successful_ca"] = []
        target = combined[name]
        target["candidates"].extend(list(source["candidates"]))
        target["successful_ca"].extend(list(source["successful_ca"]))

    for name in order:
        records = combined[name]["candidates"]
        if records and all("sample_num" in record for record in records):
            records.sort(key=lambda record: int(record["sample_num"]))
            sample_numbers = [int(record["sample_num"]) for record in records]
            if len(sample_numbers) != len(set(sample_numbers)):
                raise ValueError(
                    f"duplicate MotifBench candidate indices for {name}"
                )
    return [combined[name] for name in order]


def write_scaffold_info_files(
    units: Sequence[dict[str, object]], structures_dir: Path | None
) -> None:
    """Write complete official MotifBench CSVs once, after all blocks merge."""
    if structures_dir is None:
        return
    for unit in _consolidate_scaffolding_units(units):
        case_dir = Path(structures_dir) / str(unit["problem"])
        case_dir.mkdir(parents=True, exist_ok=True)
        csv_path = case_dir / "scaffold_info.csv"
        temporary = csv_path.with_suffix(".csv.tmp")
        records = list(unit["candidates"])
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["sample_num", "motif_placements"]
            )
            writer.writeheader()
            writer.writerows(
                {
                    "sample_num": int(record["sample_num"]),
                    "motif_placements": record["motif_placements"],
                }
                for record in records
            )
        os.replace(temporary, csv_path)


def aggregate_scaffolding(
    units: Sequence[dict[str, object]],
    judge,
    *,
    motif_rmsd_threshold: float = DPLM_MOTIF_RMSD_THRESHOLD,
    sc_tm_threshold: float = DPLM_SCTM_THRESHOLD,
) -> dict[str, object]:
    """Reduce per-problem scaffolding units to the report body."""
    units = _consolidate_scaffolding_units(units)
    all_records: list[dict[str, object]] = []
    per_problem: dict[str, object] = {}
    failures_by_motif: dict[str, object] = {}
    overall_successful_ca: list[np.ndarray] = []

    for unit in units:
        records = list(unit["candidates"])
        successful_ca = [
            np.asarray(ca, dtype=np.float64) for ca in unit["successful_ca"]
        ]
        all_records.extend(records)
        overall_successful_ca.extend(successful_ca)

        name = str(unit["problem"])
        n_success = sum(1 for record in records if record["solved"])
        n_candidates = len(records)
        solved = n_success > 0
        per_problem[name] = {
            "n_candidates": n_candidates,
            "n_success": n_success,
            "success_rate": (n_success / n_candidates)
            if n_candidates
            else 0.0,
            "solved": solved,
            "motif_size": int(unit["motif_size"]),
            "length_range": [int(value) for value in unit["length_range"]],
            "contig": unit.get("contig", ""),
            "segment_order": list(unit.get("segment_order", [])),
            "placement_policy": unit.get("placement_policy"),
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
        for metric in ("pair_sc_tm", "pair_sc_rmsd", "pair_sc_plddt"):
            values = [r[metric] for r in records if metric in r]
            if values:
                per_problem[name]["distributions"][metric] = (
                    _distribution_summary(values)
                )
        for metric in (
            "motif_pair_distance_mae",
            "backbone_clash_count",
            "backbone_chain_break_count",
        ):
            values = [r[metric] for r in records if metric in r]
            if values:
                per_problem[name]["distributions"][metric] = (
                    _distribution_summary(values)
                )
        failures_by_motif[name] = {
            "n_candidates": n_candidates,
            "n_failed": n_candidates - n_success,
            "fail_fraction": ((n_candidates - n_success) / n_candidates)
            if n_candidates
            else 0.0,
            "target_failed": not solved,
        }

    # Failures broken down by scaffold-length bin across every
    # candidate.
    failures_by_length: dict[str, dict[str, object]] = {}
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

    num_targets = len(units)
    num_solved_targets = sum(
        1 for value in per_problem.values() if value["solved"]
    )
    total_candidates = len(all_records)
    total_success = sum(1 for record in all_records if record["solved"])
    from evaluation.evaluation_drivers.proteins.statistics import (
        power_of_two_prefix_budgets,
        wilson_interval,
    )

    max_candidates = max(
        (len(unit["candidates"]) for unit in units), default=0
    )
    solved_at_n = {}
    for budget in power_of_two_prefix_budgets(max_candidates):
        solved_flags = [
            any(record["solved"] for record in unit["candidates"][:budget])
            for unit in units
        ]
        solved_count = sum(solved_flags)
        solved_at_n[str(budget)] = {
            "num_solved_targets": solved_count,
            "fraction_solved_targets": (
                solved_count / num_targets if num_targets else 0.0
            ),
            "fraction_solved_targets_ci95": wilson_interval(
                solved_count, num_targets
            ),
        }

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
    for metric in ("pair_sc_tm", "pair_sc_rmsd", "pair_sc_plddt"):
        values = [record[metric] for record in all_records if metric in record]
        if values:
            distributions[metric] = _distribution_summary(values)
    for metric in (
        "motif_pair_distance_mae",
        "backbone_clash_count",
        "backbone_chain_break_count",
    ):
        values = [record[metric] for record in all_records if metric in record]
        if values:
            distributions[metric] = _distribution_summary(values)

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
            "fraction_solved_targets_ci95": wilson_interval(
                num_solved_targets, num_targets
            ),
            "solved_at_n": solved_at_n,
            "selection_rule": (
                "success in the deterministic generation-order prefix; "
                "candidate budget is part of the metric"
            ),
        },
        "candidates": {
            "num_candidates": total_candidates,
            "num_success": total_success,
            "success_rate": (total_success / total_candidates)
            if total_candidates
            else 0.0,
            "success_rate_ci95": wilson_interval(
                total_success, total_candidates
            ),
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
    structures_dir: Path | None = None,
) -> dict[str, object]:
    """
    Run the full motif scaffolding evaluation over a set of motif
    problems.
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

    units = []
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
            structures_dir=structures_dir,
        )
        units.append(
            problem_unit(problem, records, successful_ca, length_bin_width)
        )

    return aggregate_scaffolding(
        units,
        judge,
        motif_rmsd_threshold=motif_rmsd_threshold,
        sc_tm_threshold=sc_tm_threshold,
    )


# -----------------------------------------------------------------------------
# Command-line entry point
# -----------------------------------------------------------------------------


def _jsonable(obj):
    """
    Recursively coerce numpy scalars and arrays into JSON-serialisable
    values.
    """
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


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """
    Write JSON to a temporary sibling file and atomically replace the
    target.
    """
    from evaluation.evaluation_drivers.proteins.io import (
        atomic_evaluation_json,
    )

    atomic_evaluation_json(
        path,
        _jsonable(payload),
        task="motif_scaffolding",
    )


def _provenance_payload(args, device) -> dict[str, object]:
    """Build the report shell (everything except results) for one run."""
    from evaluation.evaluation_drivers.proteins.io import repository_commit

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "task": "motif_scaffolding",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "problems_path": str(args.problems),
        "motifbench_scaffolds_dir": (
            str(args.structures_dir)
            if args.structures_dir is not None
            else None
        ),
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
            "git_commit": repository_commit(_REPO_ROOT),
            "git_dirty": None,
            ("git_dirty_note"): (
                "not scanned on shared Lustre; commit identity only"
            ),
        },
    }


def main() -> None:
    """
    Command-line driver: evaluate motif scaffolding and write a JSON
    report.
    """
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
        help=(
            "Path to the motif problem JSON specification (see "
            "load_motif_problems)"
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help=(
            "Local DPLM-2 structure tokenizer checkpoint for decoding "
            "backbones"
        ),
    )
    parser.add_argument(
        "--out", required=True, help="Path to the output JSON report"
    )
    parser.add_argument(
        "--structures-dir",
        type=Path,
        default=None,
        help=(
            "Export 1-indexed PDBs and scaffold_info.csv files in the "
            "official MotifBench directory layout"
        ),
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
    add_shard_args(parser)
    args = parser.parse_args()
    plan = resolve_shard(args)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    problems = load_motif_problems(args.problems)
    work_units = build_scaffold_work_units(
        problems, args.num_candidates, args.micro_batch_size
    )
    all_keys = unit_keys(work_units, key=lambda unit: unit.key)
    from evaluation.evaluation_drivers.proteins.io import input_identities

    writer = ShardWriter(
        plan,
        task="motif_scaffolding",
        meta={
            "motif_rmsd_threshold": float(args.motif_rmsd_threshold),
            "sc_tm_threshold": float(args.sc_tm_threshold),
            "payload": _provenance_payload(args, device),
            "input_identities": input_identities(
                [
                    args.config,
                    args.checkpoint,
                    args.problems,
                    args.tokenizer_path,
                    os.environ.get("PROTEINMPNN_WEIGHTS"),
                ]
            ),
        },
    )
    writer.write_manifest(all_keys)
    if not args.resume:
        writer.reset()
    done = writer.completed_keys() if args.resume else set()

    mine = partition(
        work_units,
        plan.shard_id,
        plan.num_shards,
        key=lambda unit: unit.key,
    )
    todo = [unit for unit in mine if unit.key not in done]

    judge = _load_structure_metrics()
    if todo:
        model, cfg = _load_model(
            args.config, args.checkpoint, device, args.num_steps
        )
        generate_fn = _load_generate()
        self_consistency = _load_self_consistency(args.n_seqs)
        tokenizer = _load_tokenizer(
            args.tokenizer_path, str(getattr(device, "type", device))
        )
        for work_unit in todo:
            problem = work_unit.problem
            random_seed = unit_seed(
                args.seed, work_unit.key, namespace="motif_scaffolding"
            )
            torch.manual_seed(random_seed)
            np.random.seed(random_seed % (2**32))
            records, successful_ca = evaluate_problem(
                problem,
                generate_fn=generate_fn,
                tokenizer=tokenizer,
                self_consistency=self_consistency,
                judge=judge,
                model=model,
                cfg=cfg,
                device=device,
                num_candidates=work_unit.count,
                micro_batch_size=args.micro_batch_size,
                n_seqs=args.n_seqs,
                motif_rmsd_threshold=args.motif_rmsd_threshold,
                sc_tm_threshold=args.sc_tm_threshold,
                rng=np.random.default_rng(random_seed),
                structures_dir=args.structures_dir,
                sample_offset=work_unit.start,
                write_scaffold_info=False,
            )
            unit = problem_unit(
                problem,
                records,
                successful_ca,
                args.length_bin_width,
                candidate_start=work_unit.start,
            )
            unit["random_seed"] = int(random_seed)
            unit["work_unit_key"] = work_unit.key
            writer.write(work_unit.key, unit)

    pairs = finalise(plan, writer, all_keys)
    if pairs is None:
        return

    results = aggregate_scaffolding(
        [unit for _key, unit in pairs],
        judge,
        motif_rmsd_threshold=args.motif_rmsd_threshold,
        sc_tm_threshold=args.sc_tm_threshold,
    )
    write_scaffold_info_files(
        [unit for _key, unit in pairs], args.structures_dir
    )

    payload = dict(writer.meta["payload"])
    payload["results"] = results
    out_path = Path(args.out)
    _atomic_write_json(out_path, payload)

    targets = results["targets"]
    print(
        (
            "Solved {solved}/{total} targets ({fraction:.3f}); wrote {path}"
        ).format(
            solved=targets["num_solved_targets"],
            total=targets["num_targets"],
            fraction=targets["fraction_solved_targets"],
            path=out_path,
        )
    )


if __name__ == "__main__":
    main()
