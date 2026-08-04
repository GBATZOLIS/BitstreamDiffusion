"""Joint sequence-and-structure co-generation benchmark (plan section 10.9).

This module evaluates unconditional co-generation from the multimodal 18-bit
model: the joint distribution p(s, z) over an amino-acid sequence s and its
DPLM-2 LFQ structure code z. It draws a fixed number of joint samples at each
length on the reproducible grid 100/200/300/400/500 by clamping nothing and
diffusing both modalities, decodes each structure code back to a backbone, and
scores every sample along several independent axes.

For every generated sample the module reports:
  - self-consistency: the TM-score and RMSD between the generated structure and
    an independent ESMFold fold of the co-generated sequence, together with the
    pLDDT of that independent fold and a designable flag;
  - sequence recovery: how well ProteinMPNN, re-designing the generated
    backbone, reproduces the co-generated sequence;
  - diversity and novelty: nearest-neighbour agreement within the generated set
    and maximum agreement to a reference set, for both sequence identity and
    structural TM-score;
  - secondary-structure composition and the alpha/beta/coil bias of the set;
  - backbone geometry and clash diagnostics;
  - success-rate curves over a grid of thresholds, not only the averages.

It also computes the single headline composite: the fraction of generated
samples that are simultaneously designable, novel, and diverse (plan section
10.9). The success-rate curves and per-length breakdowns are kept alongside the
scalar averages so that a single hard threshold never hides the shape of the
distribution.

The module imports with only numpy and torch present. The generator, the
self-consistency loop (ProteinMPNN plus ESMFold), the structure-metrics judge,
the DPLM-2 structure tokenizer, and the model/checkpoint loaders are all heavy
or external, so each is imported lazily inside the function that needs it and
raises a clear, actionable error when it is missing.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "DEFAULT_LENGTH_GRID",
    "decode_structure_to_backbone",
    "sequence_identity",
    "sequence_recovery",
    "secondary_structure_composition",
    "geometry_metrics",
    "success_rate_curve",
    "structure_diversity_novelty",
    "sequence_diversity_novelty",
    "run_cogeneration",
    "main",
]


# The reproducible length grid shared across the protein benchmark (plan 10.9).
DEFAULT_LENGTH_GRID: Tuple[int, ...] = (100, 200, 300, 400, 500)
DEFAULT_SAMPLES_PER_LENGTH = 50
DEFAULT_N_DESIGN_SEQS = 8

# Headline thresholds, matching evaluation.proteins.report. A sample is
# designable when its self-consistency RMSD is below DESIGNABLE_SC_RMSD or its
# self-consistency TM rises above DESIGNABLE_SC_TM; it is novel when its maximum
# structural TM to the reference set stays below NOVEL_MAX_TRAIN_TM; it is
# diverse when its nearest-neighbour TM within the generated set stays below
# DIVERSE_CLUSTER_TM.
DESIGNABLE_SC_TM = 0.5
DESIGNABLE_SC_RMSD = 2.0
NOVEL_MAX_TRAIN_TM = 0.5
DIVERSE_CLUSTER_TM = 0.5

# Threshold grids for the success-rate curves.
SC_TM_GRID: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SC_RMSD_GRID: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
PLDDT_GRID: Tuple[float, ...] = (50.0, 60.0, 70.0, 80.0, 90.0)
NOVELTY_TM_GRID: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

# Per-bond backbone bond-length deviation tolerated when calling a residue's
# geometry valid, in Angstroms.
_BOND_DEVIATION_TOLERANCE = 0.1


def _is_missing(value) -> bool:
    """Return True when a value is None or a NaN float and cannot be summarised."""
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def _mean(values: Sequence) -> Optional[float]:
    """Return the mean of the finite, non-None entries, or None when there are none."""
    finite = [float(v) for v in values if not _is_missing(v)]
    return (sum(finite) / len(finite)) if finite else None


def _fraction(flags: Sequence) -> Optional[float]:
    """Return the fraction of defined flags that are truthy, or None when none are defined."""
    defined = [flag for flag in flags if flag is not None]
    if not defined:
        return None
    return sum(1 for flag in defined if flag) / len(defined)


def _all_true_or_none(*flags):
    """Return the conjunction of the flags, or None when any flag is undefined."""
    if any(flag is None for flag in flags):
        return None
    return bool(all(flags))


def _to_ca(coords) -> np.ndarray:
    """Return a contiguous alpha-carbon array of shape [L, 3] from backbone or CA input.

    Accepts a full backbone array of shape [L, 4, 3] in atom order N, CA, C, O,
    from which the CA atom at index one is taken, or an array of shape [L, 3]
    that is already the CA track.
    """
    arr = np.ascontiguousarray(np.asarray(coords, dtype=np.float64))
    if arr.ndim == 3 and arr.shape[1] >= 2 and arr.shape[2] == 3:
        return np.ascontiguousarray(arr[:, 1, :])
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    raise ValueError(
        "Coordinates must have shape [L, 4, 3] (backbone atoms N, CA, C, O) or "
        f"[L, 3] (CA only), received array of shape {arr.shape}."
    )


def decode_structure_to_backbone(struct_index, struct_tokenizer) -> np.ndarray:
    """Decode a per-residue LFQ structure-token array to a backbone.

    ``struct_index`` is a one-dimensional array of uint16 structure ids in
    ``[0, 8191]``; ``struct_tokenizer`` is the object returned by
    ``evaluation.proteins.dplm_struct_tokenizer.load_struct_tokenizer``. Returns
    the backbone coordinates of shape ``[L, 4, 3]`` with atom order N, CA, C, O.
    """
    index = np.asarray(struct_index).reshape(-1)
    coords = struct_tokenizer.decode(index)
    return np.asarray(coords, dtype=np.float64)


def sequence_identity(a: str, b: str) -> float:
    """Return the fraction of aligned positions at which two sequences agree.

    The two strings are compared position by position over the shorter length.
    Returns zero when either string is empty.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for x, y in zip(a[:n], b[:n]) if x == y)
    return matches / n


def sequence_recovery(
    designed_sequences: Sequence[str], reference_sequence: str
) -> dict:
    """Score how well re-designed sequences reproduce the co-generated sequence.

    Each ProteinMPNN design of the generated backbone is compared against the
    co-generated reference sequence by per-position identity. Returns a dict with
    the best and mean recovery over the designs and the per-design values. When
    no designs are supplied every field is None or empty.
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
    """Report secondary-structure fractions and the alpha/beta/coil bias.

    ``backbone`` is a backbone array of shape ``[L, 4, 3]`` (or a CA array of
    shape ``[L, 3]``). The helix, sheet, and coil fractions come from the
    approximate CA-geometry assignment in
    ``evaluation.proteins.structure_metrics``. The returned dict also carries the
    structured fraction (helix plus sheet), a signed alpha bias (helix minus
    sheet, positive when the fold leans alpha and negative when it leans beta),
    and the dominant class label ('alpha', 'beta', or 'coil').
    """
    sm = structure_metrics
    if sm is None:
        from evaluation.proteins import structure_metrics as sm
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
    """Report backbone geometry, clash, and a scalar geometry-validity score.

    ``backbone`` is a backbone array of shape ``[L, 4, 3]`` in atom order
    N, CA, C, O. The bond-length deviations, CA-CA clash count, chain-break
    count, and chirality consistency come from
    ``evaluation.proteins.structure_metrics.backbone_geometry``; the radius of
    gyration comes from the CA track. The scalar ``geometry_validity`` in
    ``[0, 1]`` is the fraction of the five checks that pass: each of the three
    backbone bonds deviating by at most ``_BOND_DEVIATION_TOLERANCE`` Angstroms,
    no CA-CA clashes, and no chain breaks.
    """
    sm = structure_metrics
    if sm is None:
        from evaluation.proteins import structure_metrics as sm
    geom = sm.backbone_geometry(backbone)
    rg = sm.radius_of_gyration(_to_ca(backbone))

    checks: List[float] = []
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
) -> List[dict]:
    """Return the success rate at each threshold, not only the average.

    For a higher-is-better metric the success rate at threshold ``t`` is the
    fraction of finite values that are at least ``t``; for a lower-is-better
    metric it is the fraction that are at most ``t``. None and NaN values are
    dropped, and the success rate is None when nothing remains. Returns a list of
    ``{"threshold": t, "success_rate": r}`` dicts in threshold order.
    """
    finite = [float(v) for v in values if not _is_missing(v)]
    curve: List[dict] = []
    for threshold in thresholds:
        if not finite:
            rate: Optional[float] = None
        elif higher_is_better:
            rate = sum(1 for v in finite if v >= threshold) / len(finite)
        else:
            rate = sum(1 for v in finite if v <= threshold) / len(finite)
        curve.append({"threshold": float(threshold), "success_rate": rate})
    return curve


def _safe_tm(
    structure_metrics, ca_a: np.ndarray, ca_b: np.ndarray
) -> Optional[float]:
    """Return the TM-score between two CA arrays, or None if it cannot be computed.

    Unequal-length arrays are only scorable when the exact tmtools backend is
    installed; the pure-numpy fallback requires equal lengths and raises, which
    is caught here so a single incomparable pair never aborts the sweep.
    """
    try:
        return float(structure_metrics.tm_score(ca_a, ca_b))
    except Exception:  # noqa: BLE001 - an incomparable pair contributes nothing
        return None


def structure_diversity_novelty(
    ca_list: Sequence[np.ndarray],
    lengths: Sequence[int],
    reference_ca_by_length: Optional[Dict[int, List[np.ndarray]]],
    structure_metrics,
    diverse_tm: float = DIVERSE_CLUSTER_TM,
    novel_tm: float = NOVEL_MAX_TRAIN_TM,
) -> List[dict]:
    """Return per-sample structural diversity and novelty scores.

    For every generated backbone the nearest-neighbour TM-score to the other
    generated backbones of the same length measures diversity: a low value means
    the sample is unlike the rest of the set. The maximum TM-score to the
    reference structures of the same length measures novelty: a low value means
    the sample is unlike anything in the reference set. Comparisons are grouped
    by length so the pure-numpy TM fallback (which requires equal lengths) always
    applies. Each returned dict carries the raw ``structure_nn_tm`` and
    ``structure_novelty_tm`` scores and the boolean ``diverse`` and ``novel``
    flags; a score and its flag are None when there is no comparable partner.
    """
    lengths = [int(length) for length in lengths]
    groups: Dict[int, List[int]] = {}
    for position, length in enumerate(lengths):
        groups.setdefault(length, []).append(position)

    results: List[dict] = [
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
    reference_seq_by_length: Optional[Dict[int, List[str]]],
) -> List[dict]:
    """Return per-sample sequence diversity and novelty by identity.

    For every generated sequence the maximum identity to the other generated
    sequences of the same length measures diversity, and the maximum identity to
    the reference sequences of the same length measures novelty. Both are None
    when there is no comparable partner. Returns a list of dicts with
    ``sequence_nn_identity`` and ``sequence_novelty_identity``.
    """
    lengths = [int(length) for length in lengths]
    groups: Dict[int, List[int]] = {}
    for position, length in enumerate(lengths):
        groups.setdefault(length, []).append(position)

    results: List[dict] = [
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
) -> Tuple[List[str], np.ndarray]:
    """Draw ``num_samples`` joint samples at one length, optionally in micro-batches.

    Repeatedly calls the joint generator so that a large request is split into
    chunks of at most ``micro_batch_size`` samples (or one chunk when it is
    None), then concatenates the amino-acid strings and the structure-index
    arrays. Returns ``(seq_strings, struct_index)`` with ``struct_index`` of
    shape ``[num_samples, length]``.
    """
    sequences: List[str] = []
    indices: List[np.ndarray] = []
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
    """Score one co-generated sample along the self-consistency and geometry axes.

    Runs the ProteinMPNN plus ESMFold self-consistency loop on the generated
    backbone, measures how well the re-designed sequences recover the
    co-generated sequence, and computes the secondary-structure composition and
    backbone geometry. Returns a per-sample record without the diversity and
    novelty fields, which need the whole set and are attached later.
    """
    sc = self_consistency.run(backbone, n_seqs=int(n_design_seqs))
    designed = [entry["sequence"] for entry in sc.get("per_sequence", [])]
    recovery = sequence_recovery(designed, sequence)
    return {
        "sc_tm": float(sc["sc_tm"]),
        "sc_rmsd": float(sc["sc_rmsd"]),
        "plddt": float(sc["plddt"]),
        "designable": bool(sc["designable"]),
        "seq_recovery_best": recovery["best"],
        "seq_recovery_mean": recovery["mean"],
        "secondary_structure": secondary_structure_composition(
            backbone, structure_metrics
        ),
        "geometry": geometry_metrics(backbone, structure_metrics),
    }


def _secondary_structure_bias(records: Sequence[dict]) -> dict:
    """Return the fraction of samples dominated by each secondary-structure class."""
    dominant = [
        record["secondary_structure"]["dominant_class"] for record in records
    ]
    total = len(dominant) or 1
    return {
        label: dominant.count(label) / total
        for label in ("alpha", "beta", "coil")
    }


def _summarize(records: Sequence[dict]) -> dict:
    """Aggregate per-sample records into means, fractions, and success-rate curves.

    Every mean drops None and NaN entries, every fraction is taken over the
    samples where the flag is defined, and the success-rate curves report the
    full threshold sweep for the self-consistency, confidence, novelty, and
    diversity metrics so that a single cutoff never hides the distribution.
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
    }
    fractions = {
        "designable": _fraction([r.get("designable") for r in records]),
        "diverse": _fraction([r.get("diverse") for r in records]),
        "novel": _fraction([r.get("novel") for r in records]),
        "designable_novel_diverse": _fraction(
            [r.get("designable_novel_diverse") for r in records]
        ),
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
        "fractions": fractions,
        "success_rate_curves": curves,
        "secondary_structure_bias": _secondary_structure_bias(records),
    }


def _headline_composite(
    records: Sequence[dict],
    by_length: Dict[int, List[dict]],
    thresholds: dict,
    has_references: bool,
) -> dict:
    """Compute the headline fraction of designable, novel, and diverse samples.

    The value is the fraction of samples that pass all three criteria, taken over
    the samples where the composite is defined (a sample is undefined when its
    novelty could not be scored for want of a reference set, or its diversity for
    want of a same-length partner). The component fractions and per-length values
    are reported alongside so the headline can be audited.
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
            "Fraction of joint samples that are simultaneously designable, novel, "
            "and diverse."
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


def run_cogeneration(
    model,
    cfg,
    device,
    struct_tokenizer,
    *,
    length_grid: Sequence[int] = DEFAULT_LENGTH_GRID,
    samples_per_length: int = DEFAULT_SAMPLES_PER_LENGTH,
    n_design_seqs: int = DEFAULT_N_DESIGN_SEQS,
    micro_batch_size: Optional[int] = None,
    reference_ca: Optional[Sequence] = None,
    reference_sequences: Optional[Sequence[str]] = None,
    sc_temperature: float = 0.1,
    seed: int = 0,
    progress: bool = False,
) -> dict:
    """Run the full joint co-generation benchmark and return a report dictionary.

    Draws ``samples_per_length`` joint samples at each length in ``length_grid``
    with ``generate(task='joint', ...)``, decodes each structure code to a
    backbone, and scores every sample with self-consistency, sequence recovery,
    secondary-structure composition, geometry, and diversity/novelty. Optional
    ``reference_ca`` (backbone or CA arrays) and ``reference_sequences`` supply
    the sets that novelty is measured against; without them novelty and the
    strict composite are left undefined. Returns a dict with the per-sample
    records, the overall and per-length summaries with success-rate curves, and
    the headline composite fraction.

    The generator, the self-consistency loop, and the structure-metrics judge are
    imported lazily here so the module stays importable with numpy and torch
    only.
    """
    from evaluation.proteins import structure_metrics
    from evaluation.proteins.generate_multimodal import generate
    from evaluation.proteins.self_consistency import SelfConsistency

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    self_consistency = SelfConsistency(temperature=float(sc_temperature))

    reference_ca_by_length: Dict[int, List[np.ndarray]] = {}
    for array in reference_ca or []:
        ca = _to_ca(array)
        reference_ca_by_length.setdefault(int(ca.shape[0]), []).append(ca)
    reference_seq_by_length: Dict[int, List[str]] = {}
    for text in reference_sequences or []:
        reference_seq_by_length.setdefault(len(text), []).append(str(text))
    has_references = bool(reference_ca_by_length)

    records: List[dict] = []
    ca_list: List[np.ndarray] = []
    seq_list: List[str] = []
    length_list: List[int] = []

    for length in length_grid:
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
        for i, sequence in enumerate(sequences):
            backbone = decode_structure_to_backbone(
                struct_index[i], struct_tokenizer
            )
            record = {"length": length, "index": i, "sequence": sequence}
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
            length_list.append(length)
            if progress:
                print(
                    f"length {length} sample {i + 1}/{len(sequences)}: "
                    f"scTM={record['sc_tm']:.3f} scRMSD={record['sc_rmsd']:.3f} "
                    f"designable={record['designable']}"
                )

    structure_dn = structure_diversity_novelty(
        ca_list, length_list, reference_ca_by_length, structure_metrics
    )
    sequence_dn = sequence_diversity_novelty(
        seq_list, length_list, reference_seq_by_length
    )
    for record, structure_scores, sequence_scores in zip(
        records, structure_dn, sequence_dn
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

    by_length: Dict[int, List[dict]] = {}
    for record in records:
        by_length.setdefault(record["length"], []).append(record)

    thresholds = {
        "designable_sc_tm": DESIGNABLE_SC_TM,
        "designable_sc_rmsd": DESIGNABLE_SC_RMSD,
        "novel_max_train_tm": NOVEL_MAX_TRAIN_TM,
        "diverse_cluster_tm": DIVERSE_CLUSTER_TM,
    }

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
        "overall": _summarize(records),
        "by_length": {
            str(length): _summarize(recs)
            for length, recs in sorted(by_length.items())
        },
        "headline_composite": _headline_composite(
            records, by_length, thresholds, has_references
        ),
        "samples": records,
    }


def _load_reference_structures(
    path: Optional[Path],
) -> Optional[List[np.ndarray]]:
    """Load reference backbone or CA structures for novelty scoring.

    ``path`` points to a ``.npz`` archive whose members are individual structures
    of shape ``[L, 4, 3]`` or ``[L, 3]``, or to a ``.npy`` object array of such
    structures. Returns the list of arrays, or None when ``path`` is None.
    """
    if path is None:
        return None
    data = np.load(path, allow_pickle=True)
    if hasattr(data, "files"):
        return [np.asarray(data[name]) for name in data.files]
    array = np.asarray(data)
    if array.dtype == object:
        return [np.asarray(item) for item in list(array)]
    raise ValueError(
        "Reference structures must be a .npz archive of per-structure arrays or a "
        ".npy object array of arrays."
    )


def _load_reference_sequences(path: Optional[Path]) -> Optional[List[str]]:
    """Load reference sequences from a FASTA file, or None when ``path`` is None."""
    if path is None:
        return None
    from evaluation.proteins.io import read_fasta

    return list(read_fasta(path))


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
    """Command-line entry point that runs the benchmark and writes a JSON report.

    Loads the multimodal checkpoint and the DPLM-2 structure tokenizer, runs the
    joint co-generation benchmark over the requested length grid, and writes the
    full report (per-sample records, per-length and overall summaries with
    success-rate curves, and the headline composite) to ``--out`` as JSON. All
    heavy loaders are imported lazily so ``--help`` works with only numpy and
    torch present.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Python config module path"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--struct-tokenizer-path",
        required=True,
        help="Local DPLM-2 structure tokenizer checkpoint directory or HF repo id",
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
        help="Override cfg.evaluation.num_sampling_steps when greater than zero",
    )
    parser.add_argument("--sc-temperature", type=float, default=0.1)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=16,
        help="Split generation into chunks of this size; 0 generates each length at once",
    )
    parser.add_argument("--reference-structures", type=Path, default=None)
    parser.add_argument("--reference-sequences", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    from evaluation.proteins.dplm_struct_tokenizer import load_struct_tokenizer
    from evaluation.proteins.io import load_binary_protein_model

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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

    reference_ca = _load_reference_structures(args.reference_structures)
    reference_sequences = _load_reference_sequences(args.reference_sequences)

    micro_batch_size = (
        args.micro_batch_size
        if args.micro_batch_size and args.micro_batch_size > 0
        else None
    )
    results = run_cogeneration(
        model,
        cfg,
        device,
        struct_tokenizer,
        length_grid=tuple(args.lengths),
        samples_per_length=args.samples_per_length,
        n_design_seqs=args.n_design_seqs,
        micro_batch_size=micro_batch_size,
        reference_ca=reference_ca,
        reference_sequences=reference_sequences,
        sc_temperature=args.sc_temperature,
        seed=args.seed,
        progress=args.progress,
    )

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "struct_tokenizer_path": str(args.struct_tokenizer_path),
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": int(args.seed),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
        handle.write("\n")
    print(
        f"Wrote co-generation report with {len(results['samples'])} samples to {args.out}"
    )


if __name__ == "__main__":
    main()
