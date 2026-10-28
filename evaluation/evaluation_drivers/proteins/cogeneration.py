"""Generate protein sequences and structures and measure backbone quality."""

from __future__ import annotations

import argparse
import math
import os

# Make the repository root importable so this module works both as a
# package (import evaluation.evaluation_drivers.proteins.cogeneration)
# and as a direct script
# (python evaluation/evaluation_drivers/proteins/cogeneration.py).
# Every sibling
# driver does this; cogeneration was the one that did not, so running it
# as a script died on "No module named 'evaluation'" before argparse
# ever saw --help. Idempotent, and a no-op when the root is already on
# the path.
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    "DEFAULT_LENGTH_GRID",
    "CogenerationUnit",
    "build_cogeneration_units",
    "decode_structure_to_backbone",
    "sequence_identity",
    "sequence_recovery",
    "secondary_structure_composition",
    "geometry_metrics",
    "success_rate_curve",
    "structure_diversity_novelty",
    "sequence_diversity_novelty",
    "evaluate_length",
    "finalize_cogeneration_set_metrics",
    "aggregate_cogeneration",
    "build_reference_index",
    "run_cogeneration",
    "main",
]


# The reproducible length grid shared across the protein benchmark (plan
# 10.9).
DEFAULT_LENGTH_GRID: tuple[int, ...] = (100, 200, 300, 400, 500)
DEFAULT_SAMPLES_PER_LENGTH = 50
DEFAULT_N_DESIGN_SEQS = 8


@dataclass(frozen=True)
class CogenerationUnit:
    """A restart-safe contiguous candidate block at one sequence length."""

    length: int
    start: int
    count: int

    @property
    def key(self) -> str:
        stop = self.start + self.count
        return f"L{self.length}/samples-{self.start:04d}-{stop - 1:04d}"


def build_cogeneration_units(
    length_grid: Sequence[int], samples_per_length: int, unit_size: int
) -> list[CogenerationUnit]:
    """Split each length stratum into deterministic restart-safe blocks."""
    samples_per_length = int(samples_per_length)
    unit_size = int(unit_size)
    if samples_per_length < 1:
        raise ValueError("samples_per_length must be at least one")
    if unit_size < 1:
        raise ValueError("resume unit size must be at least one")
    return [
        CogenerationUnit(
            length=int(length),
            start=start,
            count=min(unit_size, samples_per_length - start),
        )
        for length in length_grid
        for start in range(0, samples_per_length, unit_size)
    ]


# Headline thresholds, matching
# evaluation.evaluation_drivers.proteins.report. A sample is
# designable when its self-consistency RMSD is below DESIGNABLE_SC_RMSD
# or its self-consistency TM rises above DESIGNABLE_SC_TM; it is novel
# when its maximum structural TM to the reference set stays below
# NOVEL_MAX_TRAIN_TM; it is diverse when its nearest-neighbour TM within
# the generated set stays below DIVERSE_CLUSTER_TM.
DESIGNABLE_SC_TM = 0.5
DESIGNABLE_SC_RMSD = 2.0
NOVEL_MAX_TRAIN_TM = 0.5
DIVERSE_CLUSTER_TM = 0.5

# Threshold grids for the success-rate curves.
SC_TM_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SC_RMSD_GRID: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
PLDDT_GRID: tuple[float, ...] = (50.0, 60.0, 70.0, 80.0, 90.0)
NOVELTY_TM_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

# Per-bond backbone bond-length deviation tolerated when calling a
# residue's geometry valid, in Angstroms.
_BOND_DEVIATION_TOLERANCE = 0.1


def _is_missing(value) -> bool:
    """
    Return True when a value is None or a NaN float and cannot be
    summarised.
    """
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def _mean(values: Sequence) -> float | None:
    """
    Return the mean of the finite, non-None entries, or None when there
    are none.
    """
    finite = [float(v) for v in values if not _is_missing(v)]
    return (sum(finite) / len(finite)) if finite else None


def _fraction(flags: Sequence) -> float | None:
    """
    Return the fraction of defined flags that are truthy, or None when
    none are defined.
    """
    defined = [flag for flag in flags if flag is not None]
    if not defined:
        return None
    return sum(1 for flag in defined if flag) / len(defined)


def _all_true_or_none(*flags):
    """
    Return the conjunction of the flags, or None when any flag is
    undefined.
    """
    if any(flag is None for flag in flags):
        return None
    return bool(all(flags))


def _to_ca(coords) -> np.ndarray:
    """
    Return a contiguous alpha-carbon array of shape [L, 3] from backbone
    or CA input.
    """
    arr = np.ascontiguousarray(np.asarray(coords, dtype=np.float64))
    if arr.ndim == 3 and arr.shape[1] >= 2 and arr.shape[2] == 3:
        return np.ascontiguousarray(arr[:, 1, :])
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    raise ValueError(
        f"Coordinates must have shape [L, 4, 3] (backbone atoms N, CA, C, "
        f"O) or [L, 3] (CA only), received array of shape {arr.shape}."
    )


def decode_structure_to_backbone(struct_index, struct_tokenizer) -> np.ndarray:
    """Decode a per-residue LFQ structure-token array to a backbone."""
    index = np.asarray(struct_index).reshape(-1)
    coords = struct_tokenizer.decode(index)
    return np.asarray(coords, dtype=np.float64)


def sequence_identity(a: str, b: str) -> float:
    """
    Return the fraction of aligned positions at which two sequences
    agree.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for x, y in zip(a[:n], b[:n], strict=True) if x == y)
    return matches / n


def sequence_recovery(
    designed_sequences: Sequence[str], reference_sequence: str
) -> dict:
    """
    Score how well re-designed sequences reproduce the co-generated
    sequence.
    """
    per_design = [
        sequence_identity(design, reference_sequence)
        for design in designed_sequences
    ]
    if not per_design:
        return {"best": None, "mean": None, "per_design": []}
    return {
        "best": max(per_design),
        "mean": sum(per_design) / len(per_design),
        "per_design": per_design,
    }


def secondary_structure_composition(backbone, structure_metrics=None) -> dict:
    """Report secondary-structure fractions and the alpha/beta/coil bias."""
    sm = structure_metrics
    if sm is None:
        from evaluation.evaluation_drivers.proteins import (
            structure_metrics as sm,
        )
    fractions = sm.secondary_structure_fractions(backbone)
    helix = float(fractions.get("helix", 0.0))
    sheet = float(fractions.get("sheet", 0.0))
    coil = float(fractions.get("coil", 0.0))
    classes = {"alpha": helix, "beta": sheet, "coil": coil}
    dominant = max(classes, key=classes.get)
    return {
        "helix": helix,
        "sheet": sheet,
        "coil": coil,
        "structured_fraction": helix + sheet,
        "alpha_bias": helix - sheet,
        "dominant_class": dominant,
    }


def geometry_metrics(backbone, structure_metrics=None) -> dict:
    """
    Report backbone geometry, clash, and a scalar geometry-validity
    score.
    """
    sm = structure_metrics
    if sm is None:
        from evaluation.evaluation_drivers.proteins import (
            structure_metrics as sm,
        )
    geom = sm.backbone_geometry(backbone)
    rg = sm.radius_of_gyration(_to_ca(backbone))

    checks: list[float] = []
    bond_deviation = {}
    for key in ("n_ca", "ca_c", "c_n"):
        deviation = geom.get(key, {}).get("deviation")
        bond_deviation[key] = deviation
        if _is_missing(deviation):
            continue
        checks.append(
            1.0 if float(deviation) <= _BOND_DEVIATION_TOLERANCE else 0.0
        )
    checks.append(1.0 if int(geom.get("clash_count", 0)) == 0 else 0.0)
    checks.append(1.0 if int(geom.get("chain_break_count", 0)) == 0 else 0.0)
    validity = (sum(checks) / len(checks)) if checks else None

    return {
        "clash_count": int(geom.get("clash_count", 0)),
        "chain_break_count": int(geom.get("chain_break_count", 0)),
        "radius_of_gyration": float(rg),
        "bond_deviation": bond_deviation,
        "chirality_consistency": geom.get("chirality_consistency"),
        "geometry_validity": validity,
    }


def success_rate_curve(
    values: Sequence,
    thresholds: Sequence[float],
    higher_is_better: bool = True,
) -> list[dict]:
    """Return the success rate at each threshold, not only the average."""
    finite = [float(v) for v in values if not _is_missing(v)]
    curve: list[dict] = []
    for threshold in thresholds:
        if not finite:
            rate: float | None = None
        elif higher_is_better:
            rate = sum(1 for v in finite if v >= threshold) / len(finite)
        else:
            rate = sum(1 for v in finite if v <= threshold) / len(finite)
        curve.append({"threshold": float(threshold), "success_rate": rate})
    return curve


def _safe_tm(
    structure_metrics, ca_a: np.ndarray, ca_b: np.ndarray
) -> float | None:
    """
    Return the TM-score between two CA arrays, or None if it cannot be
    computed.
    """
    try:
        return float(structure_metrics.tm_score(ca_a, ca_b))
    # an incomparable pair contributes nothing
    except Exception:  # noqa: BLE001
        return None


def structure_diversity_novelty(
    ca_list: Sequence[np.ndarray],
    lengths: Sequence[int],
    reference_ca_by_length: dict[int, list[np.ndarray]] | None,
    structure_metrics,
    diverse_tm: float = DIVERSE_CLUSTER_TM,
    novel_tm: float = NOVEL_MAX_TRAIN_TM,
) -> list[dict]:
    """Return per-sample structural diversity and novelty scores."""
    lengths = [int(length) for length in lengths]
    groups: dict[int, list[int]] = {}
    for position, length in enumerate(lengths):
        groups.setdefault(length, []).append(position)

    results: list[dict] = [
        {
            "structure_nn_tm": None,
            "structure_novelty_tm": None,
            "diverse": None,
            "novel": None,
        }
        for _ in ca_list
    ]
    for length, indices in groups.items():
        references = (reference_ca_by_length or {}).get(length)
        for a in indices:
            nearest = None
            for b in indices:
                if a == b:
                    continue
                tm = _safe_tm(structure_metrics, ca_list[a], ca_list[b])
                if tm is None:
                    continue
                nearest = tm if nearest is None else max(nearest, tm)
            novelty = None
            if references:
                for reference in references:
                    tm = _safe_tm(structure_metrics, ca_list[a], reference)
                    if tm is None:
                        continue
                    novelty = tm if novelty is None else max(novelty, tm)
            results[a] = {
                "structure_nn_tm": nearest,
                "structure_novelty_tm": novelty,
                "diverse": None
                if nearest is None
                else bool(nearest < diverse_tm),
                "novel": None if novelty is None else bool(novelty < novel_tm),
            }
    return results


def sequence_diversity_novelty(
    seq_list: Sequence[str],
    lengths: Sequence[int],
    reference_seq_by_length: dict[int, list[str]] | None,
) -> list[dict]:
    """Return per-sample sequence diversity and novelty by identity."""
    lengths = [int(length) for length in lengths]
    groups: dict[int, list[int]] = {}
    for position, length in enumerate(lengths):
        groups.setdefault(length, []).append(position)

    results: list[dict] = [
        {"sequence_nn_identity": None, "sequence_novelty_identity": None}
        for _ in seq_list
    ]
    for length, indices in groups.items():
        references = (reference_seq_by_length or {}).get(length)
        for a in indices:
            nearest = None
            for b in indices:
                if a == b:
                    continue
                identity = sequence_identity(seq_list[a], seq_list[b])
                nearest = (
                    identity if nearest is None else max(nearest, identity)
                )
            novelty = None
            if references:
                for reference in references:
                    identity = sequence_identity(seq_list[a], reference)
                    novelty = (
                        identity if novelty is None else max(novelty, identity)
                    )
            results[a] = {
                "sequence_nn_identity": nearest,
                "sequence_novelty_identity": novelty,
            }
    return results


def _generate_joint(
    generate_fn,
    model,
    cfg,
    length: int,
    num_samples: int,
    device,
    micro_batch_size,
) -> tuple[list[str], np.ndarray]:
    """
    Draw num_samples joint samples at one length, optionally in
    micro-batches.
    """
    sequences: list[str] = []
    indices: list[np.ndarray] = []
    remaining = int(num_samples)
    while remaining > 0:
        count = (
            remaining
            if micro_batch_size is None
            else min(remaining, int(micro_batch_size))
        )
        out = generate_fn(
            model, cfg, "joint", int(length), count, device, observed=None
        )
        sequences.extend(out["seq_strings"])
        indices.append(np.asarray(out["struct_index"]))
        remaining -= count
    struct_index = np.concatenate(indices, axis=0)
    return sequences, struct_index


def _evaluate_sample(
    backbone: np.ndarray,
    sequence: str,
    self_consistency,
    structure_metrics,
    n_design_seqs: int,
) -> dict:
    """
    Score one co-generated sample along the self-consistency and
    geometry axes.
    """
    sc = self_consistency.run(backbone, n_seqs=int(n_design_seqs))
    per_sequence = list(sc.get("per_sequence", []))
    designed = [entry["sequence"] for entry in per_sequence]
    recovery = sequence_recovery(designed, sequence)
    from evaluation.evaluation_drivers.proteins.statistics import (
        power_of_two_prefix_budgets,
    )

    best_of_n = {}
    for budget in power_of_two_prefix_budgets(len(per_sequence)):
        prefix = per_sequence[:budget]
        prefix_tm = max(float(entry["sc_tm"]) for entry in prefix)
        prefix_rmsd = min(float(entry["sc_rmsd"]) for entry in prefix)
        best_of_n[str(budget)] = {
            "sc_tm": prefix_tm,
            "sc_rmsd": prefix_rmsd,
            "plddt": max(float(entry["plddt"]) for entry in prefix),
            "designable": bool(
                prefix_rmsd < DESIGNABLE_SC_RMSD
                or prefix_tm > DESIGNABLE_SC_TM
            ),
        }
    return {
        "sc_tm": float(sc["sc_tm"]),
        "sc_rmsd": float(sc["sc_rmsd"]),
        "plddt": float(sc["plddt"]),
        "designable": bool(sc["designable"]),
        "seq_recovery_best": recovery["best"],
        "seq_recovery_mean": recovery["mean"],
        "self_consistency_best_of_n": best_of_n,
        "secondary_structure": secondary_structure_composition(
            backbone, structure_metrics
        ),
        "geometry": geometry_metrics(backbone, structure_metrics),
    }


def _secondary_structure_bias(records: Sequence[dict]) -> dict:
    """
    Return the fraction of samples dominated by each secondary-structure
    class.
    """
    dominant = [
        record["secondary_structure"]["dominant_class"] for record in records
    ]
    total = len(dominant) or 1
    return {
        label: dominant.count(label) / total
        for label in ("alpha", "beta", "coil")
    }


def _summarize(records: Sequence[dict]) -> dict:
    """
    Aggregate per-sample records into means, fractions, and success-rate
    curves.
    """
    sc_tm = [r.get("sc_tm") for r in records]
    sc_rmsd = [r.get("sc_rmsd") for r in records]
    plddt = [r.get("plddt") for r in records]
    novelty_tm = [r.get("structure_novelty_tm") for r in records]
    nn_tm = [r.get("structure_nn_tm") for r in records]

    means = {
        "sc_tm": _mean(sc_tm),
        "sc_rmsd": _mean(sc_rmsd),
        "plddt": _mean(plddt),
        "seq_recovery_best": _mean(
            [r.get("seq_recovery_best") for r in records]
        ),
        "seq_recovery_mean": _mean(
            [r.get("seq_recovery_mean") for r in records]
        ),
        "geometry_validity": _mean(
            [r["geometry"].get("geometry_validity") for r in records]
        ),
        "radius_of_gyration": _mean(
            [r["geometry"].get("radius_of_gyration") for r in records]
        ),
        "clash_count": _mean(
            [r["geometry"].get("clash_count") for r in records]
        ),
        "chain_break_count": _mean(
            [r["geometry"].get("chain_break_count") for r in records]
        ),
        "helix": _mean(
            [r["secondary_structure"].get("helix") for r in records]
        ),
        "sheet": _mean(
            [r["secondary_structure"].get("sheet") for r in records]
        ),
        "coil": _mean([r["secondary_structure"].get("coil") for r in records]),
        "alpha_bias": _mean(
            [r["secondary_structure"].get("alpha_bias") for r in records]
        ),
        "structure_nn_tm": _mean(nn_tm),
        "structure_novelty_tm": _mean(novelty_tm),
        "sequence_nn_identity": _mean(
            [r.get("sequence_nn_identity") for r in records]
        ),
        "sequence_novelty_identity": _mean(
            [r.get("sequence_novelty_identity") for r in records]
        ),
        "pdb_foldseek_query_tm": _mean(
            [r.get("pdb_foldseek_query_tm") for r in records]
        ),
    }
    metric_values = {
        "sc_tm": sc_tm,
        "sc_rmsd": sc_rmsd,
        "plddt": plddt,
        "seq_recovery_best": [r.get("seq_recovery_best") for r in records],
        "seq_recovery_mean": [r.get("seq_recovery_mean") for r in records],
        "structure_nn_tm": nn_tm,
        "structure_novelty_tm": novelty_tm,
        "pdb_foldseek_query_tm": [
            r.get("pdb_foldseek_query_tm") for r in records
        ],
    }
    from evaluation.evaluation_drivers.proteins.statistics import (
        bootstrap_mean_ci,
        fraction_with_wilson,
    )

    mean_ci95 = {
        metric: bootstrap_mean_ci(values)
        for metric, values in metric_values.items()
    }
    flag_values = {
        "designable": [r.get("designable") for r in records],
        "diverse": [r.get("diverse") for r in records],
        "novel": [r.get("novel") for r in records],
        "designable_novel_diverse": [
            r.get("designable_novel_diverse") for r in records
        ],
    }
    fractions = {}
    fraction_ci95 = {}
    for name, flags in flag_values.items():
        value, interval = fraction_with_wilson(flags)
        fractions[name] = value
        fraction_ci95[name] = interval
    candidate_budgets = sorted(
        {
            int(budget)
            for record in records
            for budget in record.get("self_consistency_best_of_n", {})
        }
    )
    self_consistency_best_of_n = {}
    for budget in candidate_budgets:
        values = [
            record.get("self_consistency_best_of_n", {}).get(str(budget))
            for record in records
        ]
        values = [value for value in values if value is not None]
        designable_fraction, designable_interval = fraction_with_wilson(
            [value["designable"] for value in values]
        )
        self_consistency_best_of_n[str(budget)] = {
            "sc_tm_mean": _mean([value["sc_tm"] for value in values]),
            "sc_rmsd_mean": _mean([value["sc_rmsd"] for value in values]),
            "plddt_mean": _mean([value["plddt"] for value in values]),
            "designable_fraction": designable_fraction,
            "designable_fraction_ci95": designable_interval,
        }
    curves = {
        "sc_tm": success_rate_curve(sc_tm, SC_TM_GRID, higher_is_better=True),
        "sc_rmsd": success_rate_curve(
            sc_rmsd, SC_RMSD_GRID, higher_is_better=False
        ),
        "plddt": success_rate_curve(plddt, PLDDT_GRID, higher_is_better=True),
        "structure_novelty": success_rate_curve(
            novelty_tm, NOVELTY_TM_GRID, higher_is_better=False
        ),
        "structure_diversity": success_rate_curve(
            nn_tm, NOVELTY_TM_GRID, higher_is_better=False
        ),
    }
    return {
        "num_samples": len(records),
        "means": means,
        "mean_ci95": mean_ci95,
        "fractions": fractions,
        "fraction_ci95": fraction_ci95,
        "self_consistency_best_of_n": self_consistency_best_of_n,
        "success_rate_curves": curves,
        "secondary_structure_bias": _secondary_structure_bias(records),
    }


def _headline_composite(
    records: Sequence[dict],
    by_length: dict[int, list[dict]],
    thresholds: dict,
    has_references: bool,
) -> dict:
    """
    Compute the headline fraction of designable, novel, and diverse
    samples.
    """
    designable = [r.get("designable") for r in records]
    diverse = [r.get("diverse") for r in records]
    novel = [r.get("novel") for r in records]
    composite = [r.get("designable_novel_diverse") for r in records]

    counts = {
        "num_samples": len(records),
        "num_designable": sum(1 for f in designable if f),
        "num_diverse": sum(1 for f in diverse if f),
        "num_novel": sum(1 for f in novel if f),
        "num_designable_novel_diverse": sum(1 for f in composite if f),
        "num_novelty_defined": sum(1 for f in novel if f is not None),
        "num_diversity_defined": sum(1 for f in diverse if f is not None),
        "num_composite_defined": sum(1 for f in composite if f is not None),
    }
    return {
        "metric": "fraction_designable_novel_diverse",
        "definition": (
            "Fraction of joint samples that are simultaneously "
            "designable, novel, and diverse."
        ),
        "thresholds": thresholds,
        "value": _fraction(composite),
        "component_fractions": {
            "designable": _fraction(designable),
            "novel": _fraction(novel),
            "diverse": _fraction(diverse),
        },
        "counts": counts,
        "by_length": {
            str(length): _fraction(
                [r.get("designable_novel_diverse") for r in recs]
            )
            for length, recs in sorted(by_length.items())
        },
        "requires_reference_set": not has_references,
    }


def evaluate_length(
    model,
    cfg,
    device,
    struct_tokenizer,
    length: int,
    *,
    self_consistency,
    structure_metrics,
    samples_per_length: int,
    n_design_seqs: int,
    micro_batch_size: int | None = None,
    reference_ca_by_length: dict[int, list[np.ndarray]] | None = None,
    reference_seq_by_length: dict[int, list[str]] | None = None,
    progress: bool = False,
    structures_dir: Path | None = None,
    sample_offset: int = 0,
    defer_set_metrics: bool = False,
) -> list[dict]:
    """Draw and score joint samples at one length."""
    from evaluation.evaluation_drivers.proteins.generate_multimodal import (
        generate,
    )

    length = int(length)
    sequences, struct_index = _generate_joint(
        generate,
        model,
        cfg,
        length,
        int(samples_per_length),
        device,
        micro_batch_size,
    )

    records: list[dict] = []
    ca_list: list[np.ndarray] = []
    seq_list: list[str] = []
    for local_index, sequence in enumerate(sequences):
        i = int(sample_offset) + local_index
        backbone = decode_structure_to_backbone(
            struct_index[local_index], struct_tokenizer
        )
        sample_id = f"L{length}_{i:03d}"
        structure_path = None
        if structures_dir is not None:
            from evaluation.evaluation_drivers.proteins.pdb_io import (
                write_backbone_pdb,
            )

            structure_path = Path(structures_dir) / f"{sample_id}.pdb"
            write_backbone_pdb(structure_path, backbone, sequence)
        record = {
            "sample_id": sample_id,
            "length": length,
            "index": i,
            "sequence": sequence,
            "structure_path": (
                str(structure_path) if structure_path is not None else None
            ),
        }
        record.update(
            _evaluate_sample(
                backbone,
                sequence,
                self_consistency,
                structure_metrics,
                n_design_seqs,
            )
        )
        records.append(record)
        ca_list.append(_to_ca(backbone))
        seq_list.append(sequence)
        if progress:
            print(
                f"length {length} sample {i + 1}: scTM="
                f"{record['sc_tm']:.3f} scRMSD={record['sc_rmsd']:.3f} "
                f"designable={record['designable']}"
            )

    if defer_set_metrics:
        for record, ca in zip(records, ca_list, strict=True):
            record["_backbone_ca"] = np.asarray(ca, dtype=np.float32).tolist()
            record["structure_nn_tm"] = None
            record["structure_novelty_tm"] = None
            record["diverse"] = None
            record["novel"] = None
            record["sequence_nn_identity"] = None
            record["sequence_novelty_identity"] = None
            record["designable_novel_diverse"] = None
        return records

    lengths = [length] * len(records)
    structure_dn = structure_diversity_novelty(
        ca_list, lengths, reference_ca_by_length, structure_metrics
    )
    sequence_dn = sequence_diversity_novelty(
        seq_list, lengths, reference_seq_by_length
    )
    for record, structure_scores, sequence_scores in zip(
        records, structure_dn, sequence_dn, strict=True
    ):
        record["structure_nn_tm"] = structure_scores["structure_nn_tm"]
        record["structure_novelty_tm"] = structure_scores[
            "structure_novelty_tm"
        ]
        record["diverse"] = structure_scores["diverse"]
        record["novel"] = structure_scores["novel"]
        record["sequence_nn_identity"] = sequence_scores[
            "sequence_nn_identity"
        ]
        record["sequence_novelty_identity"] = sequence_scores[
            "sequence_novelty_identity"
        ]
        record["designable_novel_diverse"] = _all_true_or_none(
            record["designable"], record["diverse"], record["novel"]
        )
    return records


def finalize_cogeneration_set_metrics(
    records: Sequence[dict],
    *,
    reference_ca_by_length: dict[int, list[np.ndarray]] | None,
    reference_seq_by_length: dict[int, list[str]] | None,
    structure_metrics,
) -> list[dict]:
    """Compute full-set metrics after restart units have been merged."""
    records = list(records)
    missing = [
        record.get("sample_id")
        for record in records
        if "_backbone_ca" not in record
    ]
    if missing:
        raise ValueError(
            f"co-generation restart records are missing deferred CA "
            f"traces; first={missing[0]}"
        )
    ca_list = [
        np.asarray(record["_backbone_ca"], dtype=np.float64)
        for record in records
    ]
    lengths = [int(record["length"]) for record in records]
    sequences = [str(record["sequence"]) for record in records]
    structure_dn = structure_diversity_novelty(
        ca_list, lengths, reference_ca_by_length, structure_metrics
    )
    sequence_dn = sequence_diversity_novelty(
        sequences, lengths, reference_seq_by_length
    )
    for record, structure_scores, sequence_scores in zip(
        records, structure_dn, sequence_dn, strict=True
    ):
        record["structure_nn_tm"] = structure_scores["structure_nn_tm"]
        record["structure_novelty_tm"] = structure_scores[
            "structure_novelty_tm"
        ]
        record["diverse"] = structure_scores["diverse"]
        record["novel"] = structure_scores["novel"]
        record["sequence_nn_identity"] = sequence_scores[
            "sequence_nn_identity"
        ]
        record["sequence_novelty_identity"] = sequence_scores[
            "sequence_novelty_identity"
        ]
        record["designable_novel_diverse"] = _all_true_or_none(
            record["designable"], record["diverse"], record["novel"]
        )

    # The first 100 draws at each length are the prespecified
    # DPLM-2-sized comparison set. Recompute generated-set neighbours
    # within that subset; copying the full-200 diversity fields would
    # give the comparison prefix an undeclared extra 100 possible
    # neighbours.
    prefix_indices = [
        i
        for i, record in enumerate(records)
        if int(record.get("index", 10**9)) < 100
    ]
    if prefix_indices:
        prefix_ca = [ca_list[i] for i in prefix_indices]
        prefix_lengths = [lengths[i] for i in prefix_indices]
        prefix_sequences = [sequences[i] for i in prefix_indices]
        prefix_structure = structure_diversity_novelty(
            prefix_ca,
            prefix_lengths,
            reference_ca_by_length,
            structure_metrics,
        )
        prefix_sequence = sequence_diversity_novelty(
            prefix_sequences, prefix_lengths, reference_seq_by_length
        )
        for i, structure_scores, sequence_scores in zip(
            prefix_indices, prefix_structure, prefix_sequence, strict=True
        ):
            record = records[i]
            record["paper100_structure_nn_tm"] = structure_scores[
                "structure_nn_tm"
            ]
            record["paper100_structure_novelty_tm"] = structure_scores[
                "structure_novelty_tm"
            ]
            record["paper100_diverse"] = structure_scores["diverse"]
            record["paper100_novel"] = structure_scores["novel"]
            record["paper100_sequence_nn_identity"] = sequence_scores[
                "sequence_nn_identity"
            ]
            record["paper100_sequence_novelty_identity"] = sequence_scores[
                "sequence_novelty_identity"
            ]
            record["paper100_designable_novel_diverse"] = _all_true_or_none(
                record["designable"],
                record["paper100_diverse"],
                record["paper100_novel"],
            )
    for record in records:
        record.pop("_backbone_ca", None)
    return records


def aggregate_cogeneration(
    records: Sequence[dict],
    *,
    length_grid: Sequence[int],
    samples_per_length: int,
    n_design_seqs: int,
    sc_temperature: float,
    seed: int,
    has_references: bool,
    foldseek_summary: dict | None = None,
    mmseqs_summary: dict | None = None,
) -> dict:
    """Reduce per-sample co-generation records to the report body."""
    records = list(records)
    by_length: dict[int, list[dict]] = {}
    for record in records:
        by_length.setdefault(record["length"], []).append(record)

    thresholds = {
        "designable_sc_tm": DESIGNABLE_SC_TM,
        "designable_sc_rmsd": DESIGNABLE_SC_RMSD,
        "novel_max_train_tm": NOVEL_MAX_TRAIN_TM,
        "diverse_cluster_tm": DIVERSE_CLUSTER_TM,
    }

    paper_records = []
    for record in records:
        if int(record.get("index", 10**9)) >= 100:
            continue
        comparable = dict(record)
        for field in (
            "structure_nn_tm",
            "structure_novelty_tm",
            "diverse",
            "novel",
            "sequence_nn_identity",
            "sequence_novelty_identity",
            "designable_novel_diverse",
        ):
            paper_field = f"paper100_{field}"
            if paper_field in record:
                comparable[field] = record[paper_field]
        paper_records.append(comparable)
    paper_by_length: dict[int, list[dict]] = {}
    for record in paper_records:
        paper_by_length.setdefault(int(record["length"]), []).append(record)

    return {
        "task": "joint",
        "plan_section": "10.9",
        "length_grid": [int(length) for length in length_grid],
        "samples_per_length": int(samples_per_length),
        "n_design_seqs": int(n_design_seqs),
        "sc_temperature": float(sc_temperature),
        "seed": int(seed),
        "thresholds": thresholds,
        "has_reference_set": has_references,
        "foldseek": foldseek_summary,
        "mmseqs": mmseqs_summary,
        "overall": _summarize(records),
        "by_length": {
            str(length): _summarize(recs)
            for length, recs in sorted(by_length.items())
        },
        "headline_composite": _headline_composite(
            records, by_length, thresholds, has_references
        ),
        "paper_comparable_prefix_100": {
            "selection": "prespecified sample index < 100 at every length",
            "num_samples": len(paper_records),
            "overall": _summarize(paper_records),
            "by_length": {
                str(length): _summarize(recs)
                for length, recs in sorted(paper_by_length.items())
            },
            "headline_composite": _headline_composite(
                paper_records, paper_by_length, thresholds, has_references
            ),
        },
        "samples": records,
    }


def build_reference_index(
    reference_ca: Sequence | None = None,
    reference_sequences: Sequence[str] | None = None,
) -> tuple[dict[int, list[np.ndarray]], dict[int, list[str]]]:
    """
    Group the novelty reference sets by length, as the comparisons need
    them.
    """
    reference_ca_by_length: dict[int, list[np.ndarray]] = {}
    for array in reference_ca or []:
        ca = _to_ca(array)
        reference_ca_by_length.setdefault(int(ca.shape[0]), []).append(ca)
    reference_seq_by_length: dict[int, list[str]] = {}
    for text in reference_sequences or []:
        reference_seq_by_length.setdefault(len(text), []).append(str(text))
    return reference_ca_by_length, reference_seq_by_length


def run_cogeneration(
    model,
    cfg,
    device,
    struct_tokenizer,
    *,
    length_grid: Sequence[int] = DEFAULT_LENGTH_GRID,
    samples_per_length: int = DEFAULT_SAMPLES_PER_LENGTH,
    n_design_seqs: int = DEFAULT_N_DESIGN_SEQS,
    micro_batch_size: int | None = None,
    reference_ca: Sequence | None = None,
    reference_sequences: Sequence[str] | None = None,
    sc_temperature: float = 0.1,
    seed: int = 0,
    progress: bool = False,
    structures_dir: Path | None = None,
) -> dict:
    """
    Run the full joint co-generation benchmark and return a report
    dictionary.
    """
    from evaluation.evaluation_drivers.proteins import structure_metrics
    from evaluation.evaluation_drivers.proteins.self_consistency import (
        SelfConsistency,
    )

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    self_consistency = SelfConsistency(temperature=float(sc_temperature))
    reference_ca_by_length, reference_seq_by_length = build_reference_index(
        reference_ca, reference_sequences
    )

    records: list[dict] = []
    for length in length_grid:
        records.extend(
            evaluate_length(
                model,
                cfg,
                device,
                struct_tokenizer,
                int(length),
                self_consistency=self_consistency,
                structure_metrics=structure_metrics,
                samples_per_length=samples_per_length,
                n_design_seqs=n_design_seqs,
                micro_batch_size=micro_batch_size,
                reference_ca_by_length=reference_ca_by_length,
                reference_seq_by_length=reference_seq_by_length,
                progress=progress,
                structures_dir=structures_dir,
            )
        )

    return aggregate_cogeneration(
        records,
        length_grid=length_grid,
        samples_per_length=samples_per_length,
        n_design_seqs=n_design_seqs,
        sc_temperature=sc_temperature,
        seed=seed,
        has_references=bool(reference_ca_by_length),
    )


def _load_reference_structures(
    path: Path | None,
) -> list[np.ndarray] | None:
    """Load reference backbone or CA structures for novelty scoring."""
    if path is None:
        return None
    data = np.load(path, allow_pickle=True)
    if hasattr(data, "files"):
        return [np.asarray(data[name]) for name in data.files]
    array = np.asarray(data)
    if array.dtype == object:
        return [np.asarray(item) for item in list(array)]
    raise ValueError(
        "Reference structures must be a .npz archive of per-structure"
        " arrays or a .npy object array of arrays."
    )


def _load_reference_sequences(
    path: Path | None, lengths: Sequence[int] | None = None
) -> list[str] | None:
    """Stream only relevant-length reference sequences from a FASTA file."""
    if path is None:
        return None
    allowed = None if lengths is None else {int(length) for length in lengths}
    sequences: list[str] = []
    pieces: list[str] = []

    def flush() -> None:
        if not pieces:
            return
        sequence = "".join(pieces).upper()
        if allowed is None or len(sequence) in allowed:
            sequences.append(sequence)
        pieces.clear()

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
            else:
                pieces.append(line)
    flush()
    return sequences


def _json_default(obj):
    """Coerce numpy scalars, numpy arrays, and paths for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def main() -> None:
    """
    Command-line entry point that runs the benchmark and writes a JSON
    report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Python config module path"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--struct-tokenizer-path",
        required=True,
        help=(
            "Local DPLM-2 structure tokenizer checkpoint directory or HF "
            "repo id"
        ),
    )
    parser.add_argument(
        "--lengths", nargs="+", type=int, default=list(DEFAULT_LENGTH_GRID)
    )
    parser.add_argument(
        "--samples-per-length", type=int, default=DEFAULT_SAMPLES_PER_LENGTH
    )
    parser.add_argument(
        "--n-design-seqs", type=int, default=DEFAULT_N_DESIGN_SEQS
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=0,
        help=(
            "Override cfg.evaluation.num_sampling_steps when greater than zero"
        ),
    )
    parser.add_argument("--sc-temperature", type=float, default=0.1)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=16,
        help=(
            "Split generation into chunks of this size; 0 generates each "
            "length at once"
        ),
    )
    parser.add_argument(
        "--resume-unit-size",
        type=int,
        default=4,
        help=(
            "Number of same-length samples committed in one durable work"
            " unit. Set-level diversity and novelty are computed only "
            "after all units merge."
        ),
    )
    parser.add_argument("--reference-structures", type=Path, default=None)
    parser.add_argument("--reference-sequences", type=Path, default=None)
    parser.add_argument(
        "--structures-dir",
        type=Path,
        default=None,
        help="Directory for generated PDBs (required with --foldseek-db)",
    )
    parser.add_argument("--foldseek-bin", type=Path, default=None)
    parser.add_argument(
        "--foldseek-db",
        type=Path,
        default=None,
        help=(
            "Frozen Foldseek PDB/PDB100 database prefix for structural novelty"
        ),
    )
    parser.add_argument(
        "--foldseek-work-dir",
        type=Path,
        default=None,
        help="Directory retaining Foldseek hit and cluster TSVs",
    )
    parser.add_argument("--foldseek-threads", type=int, default=16)
    parser.add_argument("--mmseqs-bin", type=Path, default=None)
    parser.add_argument("--mmseqs-reference", type=Path, default=None)
    parser.add_argument("--mmseqs-work-dir", type=Path, default=None)
    parser.add_argument("--mmseqs-threads", type=int, default=16)
    parser.add_argument("--mmseqs-gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--progress", action="store_true")
    add_shard_args(parser)
    args = parser.parse_args()
    plan = resolve_shard(args)
    if (args.foldseek_bin is None) != (args.foldseek_db is None):
        parser.error(
            "--foldseek-bin and --foldseek-db must be provided together"
        )
    if args.foldseek_db is not None and args.structures_dir is None:
        parser.error(
            "--structures-dir is required when Foldseek novelty is enabled"
        )
    if (args.mmseqs_bin is None) != (args.mmseqs_reference is None):
        parser.error(
            "--mmseqs-bin and --mmseqs-reference must be provided together"
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    length_grid = [int(length) for length in args.lengths]
    reference_ca = _load_reference_structures(args.reference_structures)
    reference_sequences = _load_reference_sequences(
        args.reference_sequences, lengths=length_grid
    )
    reference_ca_by_length, reference_seq_by_length = build_reference_index(
        reference_ca, reference_sequences
    )
    has_references = (
        bool(reference_ca_by_length) or args.foldseek_db is not None
    )

    units = build_cogeneration_units(
        length_grid, args.samples_per_length, args.resume_unit_size
    )
    all_keys = unit_keys(units, key=lambda unit: unit.key)

    from evaluation.evaluation_drivers.proteins.io import input_identities

    writer = ShardWriter(
        plan,
        task="cogeneration",
        meta={
            "length_grid": length_grid,
            "samples_per_length": int(args.samples_per_length),
            "resume_unit_size": int(args.resume_unit_size),
            "n_design_seqs": int(args.n_design_seqs),
            "sc_temperature": float(args.sc_temperature),
            "seed": int(args.seed),
            "has_references": has_references,
            "foldseek_bin": (
                str(args.foldseek_bin)
                if args.foldseek_bin is not None
                else None
            ),
            "foldseek_db": (
                str(args.foldseek_db) if args.foldseek_db is not None else None
            ),
            "foldseek_work_dir": str(
                args.foldseek_work_dir
                or (args.out.with_suffix("") / "foldseek")
            ),
            "foldseek_threads": int(args.foldseek_threads),
            "mmseqs_bin": str(args.mmseqs_bin) if args.mmseqs_bin else None,
            "mmseqs_reference": str(args.mmseqs_reference)
            if args.mmseqs_reference
            else None,
            "mmseqs_work_dir": str(
                args.mmseqs_work_dir or (args.out.with_suffix("") / "mmseqs")
            ),
            "mmseqs_threads": int(args.mmseqs_threads),
            "mmseqs_gpu": bool(args.mmseqs_gpu),
            "structures_dir": (
                str(args.structures_dir)
                if args.structures_dir is not None
                else None
            ),
            "arguments": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
                if key not in {"shard_id", "num_shards", "resume"}
            },
            "input_identities": input_identities(
                [
                    args.config,
                    args.checkpoint,
                    args.struct_tokenizer_path,
                    args.reference_sequences,
                    args.reference_structures,
                    args.foldseek_bin,
                    args.foldseek_db.parent if args.foldseek_db else None,
                    args.mmseqs_bin,
                    args.mmseqs_reference,
                    os.environ.get("PROTEINMPNN_WEIGHTS"),
                ]
            ),
            "payload": {
                "schema_version": 1,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "config": str(args.config),
                "checkpoint": str(args.checkpoint),
                "struct_tokenizer_path": str(args.struct_tokenizer_path),
                "device": str(device),
                "torch_version": torch.__version__,
                "seed": int(args.seed),
                "structural_novelty_reference": (
                    str(args.foldseek_db)
                    if args.foldseek_db is not None
                    else str(args.reference_structures)
                    if args.reference_structures is not None
                    else None
                ),
            },
        },
    )
    writer.write_manifest(all_keys)
    if not args.resume:
        writer.reset()
    done = writer.completed_keys() if args.resume else set()

    mine = partition(
        units,
        plan.shard_id,
        plan.num_shards,
        key=lambda unit: unit.key,
    )
    todo = [unit for unit in mine if unit.key not in done]

    if todo:
        from evaluation.evaluation_drivers.proteins import (
            dplm_struct_tokenizer,
            structure_metrics,
        )
        from evaluation.evaluation_drivers.proteins.io import (
            load_binary_protein_model,
        )
        from evaluation.evaluation_drivers.proteins.self_consistency import (
            SelfConsistency,
        )

        load_struct_tokenizer = dplm_struct_tokenizer.load_struct_tokenizer
        model, cfg = load_binary_protein_model(
            args.config,
            args.checkpoint,
            device,
            num_steps=(args.num_steps if args.num_steps > 0 else None),
            context="co-generation",
        )
        struct_tokenizer = load_struct_tokenizer(
            str(args.struct_tokenizer_path), device=str(device)
        )
        self_consistency = SelfConsistency(
            temperature=float(args.sc_temperature)
        )
        micro_batch_size = (
            args.micro_batch_size
            if args.micro_batch_size and args.micro_batch_size > 0
            else None
        )
        for unit in todo:
            random_seed = unit_seed(
                args.seed, unit.key, namespace="joint_cogeneration"
            )
            torch.manual_seed(random_seed)
            np.random.seed(random_seed % (2**32))
            samples = evaluate_length(
                model,
                cfg,
                device,
                struct_tokenizer,
                unit.length,
                self_consistency=self_consistency,
                structure_metrics=structure_metrics,
                samples_per_length=unit.count,
                n_design_seqs=args.n_design_seqs,
                micro_batch_size=micro_batch_size,
                reference_ca_by_length=reference_ca_by_length,
                reference_seq_by_length=reference_seq_by_length,
                progress=args.progress,
                structures_dir=args.structures_dir,
                sample_offset=unit.start,
                defer_set_metrics=True,
            )
            for sample in samples:
                sample["work_unit_random_seed"] = int(random_seed)
                sample["work_unit_key"] = unit.key
            writer.write(unit.key, samples)

    pairs = finalise(plan, writer, all_keys)
    if pairs is None:
        return

    records = [record for _key, samples in pairs for record in samples]
    records.sort(
        key=lambda record: (int(record["length"]), int(record["index"]))
    )
    from evaluation.evaluation_drivers.proteins import (
        structure_metrics as full_set_structure_metrics,
    )

    records = finalize_cogeneration_set_metrics(
        records,
        reference_ca_by_length=reference_ca_by_length,
        reference_seq_by_length=reference_seq_by_length,
        structure_metrics=full_set_structure_metrics,
    )
    foldseek_summary = None
    if args.foldseek_db is not None:
        from evaluation.evaluation_drivers.proteins.foldseek import (
            annotate_with_foldseek,
        )

        foldseek_summary = annotate_with_foldseek(
            records,
            foldseek_bin=args.foldseek_bin,
            database=args.foldseek_db,
            structures_dir=args.structures_dir,
            work_dir=(
                args.foldseek_work_dir
                or (args.out.with_suffix("") / "foldseek")
            ),
            threads=args.foldseek_threads,
        )
        missing_foldseek = [
            record.get("sample_id")
            for record in records
            if record.get("pdb_foldseek_query_tm") is None
            or record.get("novel") is None
            or record.get("designable_novel_diverse") is None
        ]
        if missing_foldseek:
            raise RuntimeError(
                f"Foldseek post-processing did not define structural "
                f"novelty for {len(missing_foldseek)} sample(s); first="
                f"{missing_foldseek[0]}"
            )
    mmseqs_summary = None
    if args.mmseqs_reference is not None:
        from evaluation.evaluation_drivers.proteins.mmseqs_novelty import (
            annotate_with_mmseqs,
        )

        mmseqs_summary = annotate_with_mmseqs(
            records,
            mmseqs_bin=args.mmseqs_bin,
            reference_fasta=args.mmseqs_reference,
            work_dir=args.mmseqs_work_dir
            or (args.out.with_suffix("") / "mmseqs"),
            threads=args.mmseqs_threads,
            gpu=args.mmseqs_gpu,
        )
    results = aggregate_cogeneration(
        records,
        length_grid=length_grid,
        samples_per_length=args.samples_per_length,
        n_design_seqs=args.n_design_seqs,
        sc_temperature=args.sc_temperature,
        seed=args.seed,
        has_references=has_references,
        foldseek_summary=foldseek_summary,
        mmseqs_summary=mmseqs_summary,
    )

    payload = dict(writer.meta["payload"])
    payload["results"] = results
    from evaluation.evaluation_drivers.proteins.io import (
        atomic_evaluation_json,
    )

    atomic_evaluation_json(
        args.out,
        payload,
        task="cogeneration",
        default=_json_default,
    )
    print(
        f"Wrote co-generation report with {len(results['samples'])} "
        f"samples to {args.out}"
    )


if __name__ == "__main__":
    main()
