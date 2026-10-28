"""
Generate sequences from backbones and measure recovery and self
consistency.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

# The repository root has to be importable before the sibling-package
# import below, because this module is also run as a bare script.
if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.evaluation_drivers.proteins.sharding import (  # noqa: E402
    ShardWriter,
    add_shard_args,
    finalise,
    partition,
    resolve_shard,
    unit_keys,
    unit_seed,
)

__all__ = [
    "BackboneRecord",
    "amino_acid_recovery",
    "sequence_identity",
    "sequence_diversity",
    "nearest_training_identity",
    "novelty_report",
    "shuffle_structure_index",
    "fold_and_score",
    "load_backbone_records",
    "make_designer",
    "evaluate_backbone",
    "aggregate_backbones",
    "aggregate_inverse_folding",
    "run_inverse_folding",
    "main",
]

# Designability thresholds shared with
# evaluation.evaluation_drivers.proteins.self_consistency
#: a fold counts as designable when its self-consistency
# RMSD is below two Angstroms or its self-consistency TM-score is above
# 0.5.
DESIGNABLE_SC_RMSD = 2.0
DESIGNABLE_SC_TM = 0.5

# Backbone atom order in the [L, 4, 3] layout used throughout the
# protein stack.
_ATOM_CA = 1


def _repo_root_on_path() -> None:
    """Ensure the repository root is importable when run as a bare script."""
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)


# -----------------------------------------------------------------------------
# Sequence metrics. These are pure string operations and need only the
# standard library, so they stay importable and testable with no heavy
# dependency.
# -----------------------------------------------------------------------------


def amino_acid_recovery(prediction: str, native: str) -> float:
    """
    Fraction of positions whose predicted residue matches the native
    residue.
    """
    if len(prediction) != len(native):
        raise ValueError(
            f"amino-acid recovery requires equal-length sequences, "
            f"received {len(prediction)} and {len(native)}."
        )
    if not native:
        return 0.0
    matches = sum(
        1
        for predicted, target in zip(prediction, native, strict=True)
        if predicted == target
    )
    return matches / len(native)


def sequence_identity(a: str, b: str) -> float:
    """Positional identity over the aligned prefix of two sequences."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / n


def sequence_diversity(sequences: Sequence[str]) -> dict[str, float]:
    """Diversity of a set of designs sampled for one fixed backbone."""
    seqs = [s for s in sequences if s]
    count = len(seqs)
    unique = float(len(set(seqs)))
    if count < 2:
        return {
            "mean_pairwise_identity": float("nan"),
            "diversity": float("nan"),
            "unique": unique,
            "count": float(count),
        }
    total = 0.0
    pairs = 0
    for i in range(count):
        for j in range(i + 1, count):
            total += sequence_identity(seqs[i], seqs[j])
            pairs += 1
    mean_identity = total / pairs
    return {
        "mean_pairwise_identity": mean_identity,
        "diversity": 1.0 - mean_identity,
        "unique": unique,
        "count": float(count),
    }


def _gapless_identity(query: str, reference: str) -> float:
    """Best gapless positional identity of query against reference."""
    if not query or not reference:
        return 0.0
    short, long = (
        (query, reference)
        if len(query) <= len(reference)
        else (reference, query)
    )
    short_len = len(short)
    long_len = len(long)
    best = 0
    for offset in range(long_len - short_len + 1):
        matches = sum(
            1 for i in range(short_len) if short[i] == long[offset + i]
        )
        if matches > best:
            best = matches
        if best == short_len:
            break
    return best / short_len


def nearest_training_identity(
    sequence: str, training_sequences: Sequence[str]
) -> float:
    """Maximum gapless identity of a design to any training sequence."""
    if not training_sequences:
        return float("nan")
    return max(
        _gapless_identity(sequence, reference)
        for reference in training_sequences
    )


def novelty_report(
    sequence: str,
    training_sequences: Sequence[str],
    novelty_threshold: float = 0.5,
) -> dict[str, float]:
    """Identity and novelty of one design relative to the training set."""
    nearest = nearest_training_identity(sequence, training_sequences)
    if np.isnan(nearest):
        return {
            "nearest_training_identity": float("nan"),
            "novelty": float("nan"),
            "novel": 0.0,
        }
    return {
        "nearest_training_identity": float(nearest),
        "novelty": float(1.0 - nearest),
        "novel": 1.0 if nearest < float(novelty_threshold) else 0.0,
    }


def shuffle_structure_index(
    struct_index: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Return a residue-permuted copy of the observed LFQ structure token
    ids.
    """
    index = np.asarray(struct_index)
    if index.ndim != 1:
        raise ValueError(
            "struct_index must be a one-dimensional array of LFQ token ids."
        )
    permutation = rng.permutation(index.shape[0])
    return index[permutation]


# -----------------------------------------------------------------------------
# Coordinate helpers and the independent-fold self-consistency score.
# -----------------------------------------------------------------------------


def _as_backbone(array) -> np.ndarray:
    """
    Coerce coordinates to [L, 4, 3] float32 in backbone atom order N,
    CA, C, O.
    """
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] >= 4 and arr.shape[2] == 3:
        return arr[:, :4, :]
    if arr.ndim == 2 and arr.shape[1] == 12:
        return arr.reshape(-1, 4, 3)
    raise ValueError(
        f"coordinates must have shape [L, 4, 3] (backbone atoms N, CA, C, "
        f"O) or [L, 12]; received array of shape {arr.shape}."
    )


def _ca_track(coords) -> np.ndarray:
    """
    Return the [L, 3] alpha-carbon track from backbone or CA-only
    coordinates.
    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1] >= 2 and arr.shape[2] == 3:
        return arr[:, _ATOM_CA, :]
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    raise ValueError(
        f"coordinates must have shape [L, 4, 3] or [L, 3]; received array "
        f"of shape {arr.shape}."
    )


def fold_and_score(sequence: str, native_backbone) -> dict[str, float]:
    """
    Fold a design with the independent model and score it against the
    native.
    """
    _repo_root_on_path()
    try:
        from evaluation.evaluation_drivers.proteins import structure_metrics
        from evaluation.evaluation_drivers.proteins.self_consistency import (
            esmfold_plddt,
        )
    # re-raised with an actionable hint
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Self-consistency scoring needs "
            "evaluation.evaluation_drivers.proteins.self_consistency and "
            "evaluation.evaluation_drivers.proteins.structure_metrics. "
            "Ensure the protein "
            "evaluation package is importable; see ./setup.sh eval."
        ) from exc

    coords, plddt = esmfold_plddt(sequence)
    predicted_ca = _ca_track(coords)
    native_ca = _ca_track(native_backbone)
    usable = min(len(predicted_ca), len(native_ca))
    if usable == 0:
        raise RuntimeError(
            "The independent fold produced no residues to score."
        )
    predicted = predicted_ca[:usable]
    native = native_ca[:usable]
    return {
        "sc_tm": float(structure_metrics.tm_score(predicted, native)),
        "sc_rmsd": float(
            structure_metrics.rmsd(predicted, native, superpose=True)
        ),
        "plddt": float(np.mean(plddt)),
        "folded_length": int(len(predicted_ca)),
    }


# -----------------------------------------------------------------------------
# Backbone corpus loading.
# -----------------------------------------------------------------------------


@dataclass
class BackboneRecord:
    """
    One held-out backbone with its native sequence and optional
    structure ids.
    """

    stable_id: str
    coords: np.ndarray
    native_seq: str
    split: str = "test"
    source: str = ""
    struct_index: np.ndarray | None = None
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Number of residues in the backbone."""
        return int(self.coords.shape[0])


def _canonical_alphabet() -> str:
    """
    Return the twenty-letter canonical amino-acid alphabet from the data
    package.
    """
    _repo_root_on_path()
    from data.protein_multimodal import CANONICAL_AA

    return CANONICAL_AA


def _sequence_from_npz(npz: dict[str, object], length: int) -> str:
    """
    Resolve a native amino-acid string of the given length from an npz
    payload.
    """
    for key in ("sequence", "seq", "native_seq"):
        if key in npz:
            value = npz[key]
            text = (
                value.item()
                if isinstance(value, np.ndarray) and value.ndim == 0
                else value
            )
            text = str(text)
            if len(text) != length:
                raise ValueError(
                    f"{key} length {len(text)} does not match backbone "
                    f"length {length}."
                )
            return text
    for key in ("seq_ids", "aatype"):
        if key in npz:
            ids = np.asarray(npz[key]).reshape(-1).astype(np.int64)
            if ids.shape[0] != length:
                raise ValueError(
                    f"{key} length {ids.shape[0]} does not match backbone "
                    f"length {length}."
                )
            alphabet = _canonical_alphabet()
            if ids.size and (ids.min() < 0 or ids.max() >= len(alphabet)):
                raise ValueError(
                    f"{key} contains ids outside the canonical range [0, "
                    f"{len(alphabet) - 1}]."
                )
            return "".join(alphabet[i] for i in ids)
    raise ValueError(
        "backbone record has no native sequence; provide 'sequence' "
        "or 'seq_ids' so amino-acid recovery can be measured."
    )


def _struct_index_from_npz(
    npz: dict[str, object], length: int
) -> np.ndarray | None:
    """
    Return precomputed LFQ structure token ids [L] uint16 if the npz has
    them.
    """
    for key in ("struct_index", "struct_tokens", "tokens"):
        if key in npz:
            index = np.asarray(npz[key]).reshape(-1)
            if index.shape[0] != length:
                raise ValueError(
                    f"{key} length {index.shape[0]} does not match "
                    f"backbone length {length}."
                )
            return index.astype(np.uint16)
    return None


def _scalar_str(value: object, default: str) -> str:
    """Read a possibly zero-dimensional numpy scalar as a plain string."""
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def _row_str(value: object, row: int, default: str) -> str:
    """Read a row label from a scalar or per-record numpy string array."""
    if value is None:
        return default
    array = np.asarray(value)
    if array.ndim == 0:
        return str(array.item())
    flat = array.reshape(-1)
    if row >= flat.size:
        raise ValueError(
            f"row label array has {flat.size} entries but row {row} was "
            f"requested"
        )
    return str(flat[row])


def _records_from_npz(
    path: Path, *, default_split: str = "test"
) -> list[BackboneRecord]:
    """Read every backbone chain stored in one npz file."""
    with np.load(path, allow_pickle=True) as loaded:
        npz = {key: loaded[key] for key in loaded.files}
    if "coords" not in npz:
        raise ValueError(f"{path}: missing required 'coords' array.")

    records: list[BackboneRecord] = []
    if "offsets" in npz:
        coords_flat = _as_backbone(npz["coords"])
        offsets = np.asarray(npz["offsets"]).astype(np.int64).reshape(-1)
        for row in range(int(offsets.shape[0] - 1)):
            start, end = int(offsets[row]), int(offsets[row + 1])
            coords = coords_flat[start:end]
            length = int(coords.shape[0])
            sub = {
                k: (
                    np.asarray(v)[start:end]
                    if k
                    in (
                        "seq_ids",
                        "aatype",
                        "struct_index",
                        "struct_tokens",
                        "tokens",
                    )
                    else v
                )
                for k, v in npz.items()
                if k not in ("coords", "offsets")
            }
            records.append(
                BackboneRecord(
                    stable_id=_row_str(
                        npz.get("stable_ids", npz.get("ids")),
                        row,
                        f"{path.stem}:{row}",
                    ),
                    coords=np.ascontiguousarray(coords),
                    native_seq=_sequence_from_npz(sub, length),
                    split=_row_str(
                        npz.get("splits", npz.get("split")),
                        row,
                        default_split,
                    ),
                    source=_row_str(
                        npz.get("sources", npz.get("source")), row, path.stem
                    ),
                    struct_index=_struct_index_from_npz(sub, length),
                )
            )
        return records

    coords = _as_backbone(npz["coords"])
    length = int(coords.shape[0])
    records.append(
        BackboneRecord(
            stable_id=_scalar_str(
                npz.get("stable_id") or npz.get("id"), path.stem
            ),
            coords=np.ascontiguousarray(coords),
            native_seq=_sequence_from_npz(npz, length),
            split=_scalar_str(npz.get("split"), default_split),
            source=_scalar_str(npz.get("source"), path.stem),
            struct_index=_struct_index_from_npz(npz, length),
        )
    )
    return records


def load_backbone_records(
    source,
    *,
    splits: Sequence[str] | None = None,
    min_len: int | None = None,
    max_len: int | None = None,
    max_examples: int | None = None,
    default_split: str = "test",
) -> list[BackboneRecord]:
    """Load held-out backbones from an npz file or a directory of npz files."""
    path = Path(source)
    if path.is_dir():
        files = sorted(path.rglob("*.npz"))
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(
            f"backbone source {path} is not a file or directory."
        )
    if not files:
        raise FileNotFoundError(
            f"no .npz backbone files found under {path}. Each file must "
            f"contain a 'coords' array of shape [L, 4, 3] (backbone N, CA,"
            f" C, O) and a native sequence via 'sequence' or 'seq_ids'."
        )

    wanted = set(splits) if splits else None
    records: list[BackboneRecord] = []
    for file_path in files:
        for record in _records_from_npz(
            file_path, default_split=str(default_split)
        ):
            if wanted is not None and record.split not in wanted:
                continue
            if min_len is not None and record.length < int(min_len):
                continue
            if max_len is not None and record.length > int(max_len):
                continue
            if not np.isfinite(record.coords).all():
                continue
            records.append(record)
            if max_examples is not None and len(records) >= int(max_examples):
                return records
    return records


# -----------------------------------------------------------------------------
# Model loading and structure-token resolution.
# -----------------------------------------------------------------------------


def _load_model(
    config_path: str,
    checkpoint: Path,
    device,
    num_steps: int | None,
    *,
    enforce_core_sampling: bool = True,
):
    """
    Build the multimodal model from a config and checkpoint, EMA weights
    applied.
    """
    _repo_root_on_path()
    from evaluation.evaluation_drivers.proteins.io import (
        load_binary_protein_model,
    )

    return load_binary_protein_model(
        config_path,
        checkpoint,
        device,
        num_steps=num_steps,
        context="inverse folding",
        enforce_core_sampling=enforce_core_sampling,
    )


class _StructureTokenResolver:
    """
    Resolve per-backbone LFQ structure token ids, caching the heavy
    tokenizer.
    """

    def __init__(self, tokenizer_path: str | None, device: str = "cpu"):
        self._tokenizer_path = tokenizer_path
        self._device = device
        self._tokenizer = None

    def _tokenizer_instance(self):
        if self._tokenizer is None:
            if not self._tokenizer_path:
                raise RuntimeError(
                    "A backbone record has no precomputed structure "
                    "tokens and no structure tokenizer was provided. "
                    "Pass --struct-tokenizer with the DPLM-2 tokenizer "
                    "checkpoint (see "
                    "scripts/proteins/setup/download_dplm_tokenizer.py) "
                    "or supply records that already carry a "
                    "'struct_index' array."
                )
            _repo_root_on_path()
            from evaluation.evaluation_drivers.proteins import (
                dplm_struct_tokenizer,
            )

            self._tokenizer = dplm_struct_tokenizer.load_struct_tokenizer(
                self._tokenizer_path, device=self._device
            )
        return self._tokenizer

    def resolve(self, record: BackboneRecord) -> np.ndarray:
        """Return the uint16 LFQ token ids [L] for one backbone record."""
        if record.struct_index is not None:
            return np.asarray(record.struct_index, dtype=np.uint16)
        tokenizer = self._tokenizer_instance()
        index = tokenizer.encode(np.asarray(record.coords, dtype=np.float32))
        return np.asarray(index).reshape(-1).astype(np.uint16)


# -----------------------------------------------------------------------------
# Design generators for the model, the shuffled control, and the
# ProteinMPNN baseline.
# -----------------------------------------------------------------------------


def _design_with_model(
    model,
    cfg,
    device,
    record: BackboneRecord,
    struct_index: np.ndarray,
    num_samples: int,
    *,
    guidance_scale: float | None = None,
    num_steps: int | None = None,
    schedule: str | None = None,
    entropic_blend_alpha: float | None = None,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    struct_sigma_scale: float | None = None,
) -> list[str]:
    """Sample sequences for a backbone by clamped inverse-folding diffusion."""
    _repo_root_on_path()
    from evaluation.evaluation_drivers.proteins.generate_multimodal import (
        generate,
    )

    result = generate(
        model,
        cfg,
        task="inverse_folding",
        length=record.length,
        num_samples=int(num_samples),
        device=device,
        observed={"struct_index": np.asarray(struct_index).reshape(-1)},
        guidance_scale=guidance_scale,
        num_steps=num_steps,
        schedule=schedule,
        entropic_blend_alpha=entropic_blend_alpha,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        struct_sigma_scale=struct_sigma_scale,
    )
    return list(result["seq_strings"])


def _design_with_proteinmpnn(
    record: BackboneRecord, num_samples: int, temperature: float
) -> list[str]:
    """
    Design sequences for a backbone with the ProteinMPNN baseline
    decoder.
    """
    _repo_root_on_path()
    from evaluation.evaluation_drivers.proteins.self_consistency import (
        proteinmpnn_sequences,
    )

    return list(
        proteinmpnn_sequences(
            np.asarray(record.coords, dtype=np.float32),
            n_seqs=int(num_samples),
            temperature=float(temperature),
        )
    )


# -----------------------------------------------------------------------------
# Per-method evaluation and aggregation.
# -----------------------------------------------------------------------------


def _summ(values: Sequence[float]) -> dict[str, object]:
    """
    Mean, median, minimum, and maximum of a list of finite floats, or
    NaNs.
    """
    finite = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not finite:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "count": 0,
        }
    array = np.asarray(finite, dtype=np.float64)
    from evaluation.evaluation_drivers.proteins.statistics import (
        bootstrap_mean_ci,
    )

    return {
        "mean": float(array.mean()),
        "mean_ci95": bootstrap_mean_ci(array),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "count": int(array.size),
    }


def evaluate_backbone(
    record: BackboneRecord,
    struct_index,
    designer: Callable[[BackboneRecord, np.ndarray], list[str]],
    *,
    training_sequences: Sequence[str],
    sc_designs: int,
    novelty_threshold: float,
) -> dict[str, object]:
    """Design, measure, and summarize one method on one backbone."""
    designs = [str(seq) for seq in designer(record, struct_index)]
    recoveries = [
        amino_acid_recovery(seq, record.native_seq) for seq in designs
    ]
    from evaluation.evaluation_drivers.proteins.statistics import (
        power_of_two_prefix_budgets,
    )

    aar_budgets = power_of_two_prefix_budgets(len(recoveries))
    entry: dict[str, object] = {
        "stable_id": record.stable_id,
        "split": record.split,
        "source": record.source,
        "length": record.length,
        "num_designs": len(designs),
        "aar_mean": float(np.mean(recoveries)) if recoveries else float("nan"),
        "aar_best": float(np.max(recoveries)) if recoveries else float("nan"),
        "aar_per_design": [float(value) for value in recoveries],
        "aar_best_of_n": {
            str(budget): float(np.max(recoveries[:budget]))
            for budget in aar_budgets
        },
        "diversity": sequence_diversity(designs),
    }

    first_index = 0 if designs else None
    if first_index is not None and training_sequences:
        entry["novelty"] = novelty_report(
            designs[first_index], training_sequences, novelty_threshold
        )

    if sc_designs > 0 and designs:
        folded = []
        for index in range(min(int(sc_designs), len(designs))):
            scored = fold_and_score(designs[index], record.coords)
            scored["design_index"] = int(index)
            scored["aar"] = float(recoveries[index])
            folded.append(scored)
        entry["self_consistency"] = folded
        entry["self_consistency_selection"] = (
            "generation_order_prefix_native_independent"
        )
        record_sc_tm = max(item["sc_tm"] for item in folded)
        record_sc_rmsd = min(item["sc_rmsd"] for item in folded)
        record_plddt = max(item["plddt"] for item in folded)
        entry["sc_tm_best"] = record_sc_tm
        entry["sc_rmsd_best"] = record_sc_rmsd
        entry["plddt_best"] = record_plddt
        entry["designable"] = bool(
            record_sc_rmsd < DESIGNABLE_SC_RMSD
            or record_sc_tm > DESIGNABLE_SC_TM
        )
        entry["self_consistency_best_of_n"] = {}
        for budget in power_of_two_prefix_budgets(len(folded)):
            prefix = folded[:budget]
            prefix_tm = max(float(item["sc_tm"]) for item in prefix)
            prefix_rmsd = min(float(item["sc_rmsd"]) for item in prefix)
            entry["self_consistency_best_of_n"][str(budget)] = {
                "sc_tm": prefix_tm,
                "sc_rmsd": prefix_rmsd,
                "plddt": max(float(item["plddt"]) for item in prefix),
                "designable": bool(
                    prefix_rmsd < DESIGNABLE_SC_RMSD
                    or prefix_tm > DESIGNABLE_SC_TM
                ),
            }

    return entry


def _evaluate_method(
    name: str,
    records: Sequence[BackboneRecord],
    struct_indices: Sequence[np.ndarray],
    designer: Callable[[BackboneRecord, np.ndarray], list[str]],
    *,
    training_sequences: Sequence[str],
    sc_designs: int,
    novelty_threshold: float,
) -> dict[str, object]:
    """
    Run one method over every backbone and pool the per-backbone
    summaries.
    """
    per_backbone = [
        evaluate_backbone(
            record,
            struct_index,
            designer,
            training_sequences=training_sequences,
            sc_designs=sc_designs,
            novelty_threshold=novelty_threshold,
        )
        for record, struct_index in zip(records, struct_indices, strict=True)
    ]
    return {
        "method": name,
        "aggregate": aggregate_backbones(per_backbone),
        "per_backbone": per_backbone,
    }


def aggregate_backbones(
    per_backbone: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Pool per-backbone entries for one method into the method aggregate."""
    best_aar: list[float] = []
    mean_aar: list[float] = []
    diversity: list[float] = []
    sc_tm: list[float] = []
    sc_rmsd: list[float] = []
    plddt: list[float] = []
    designable: list[float] = []
    novelty: list[float] = []
    novel_flags: list[float] = []
    aar_by_budget: dict[str, list[float]] = {}
    sc_by_budget: dict[str, dict[str, list[object]]] = {}

    for entry in per_backbone:
        if int(entry.get("num_designs", 0)) > 0:
            best_aar.append(entry["aar_best"])
            mean_aar.append(entry["aar_mean"])
            for budget, value in entry.get("aar_best_of_n", {}).items():
                aar_by_budget.setdefault(str(budget), []).append(float(value))
        entry_diversity = entry.get("diversity", {}).get("diversity")
        if entry_diversity is not None and not np.isnan(entry_diversity):
            diversity.append(entry_diversity)

        report = entry.get("novelty")
        if report is not None and not np.isnan(report["novelty"]):
            novelty.append(report["novelty"])
            novel_flags.append(report["novel"])

        if "sc_tm_best" in entry:
            sc_tm.append(entry["sc_tm_best"])
            sc_rmsd.append(entry["sc_rmsd_best"])
            plddt.append(entry["plddt_best"])
            designable.append(1.0 if entry["designable"] else 0.0)
            for budget, values in entry.get(
                "self_consistency_best_of_n", {}
            ).items():
                accumulators = sc_by_budget.setdefault(
                    str(budget),
                    {
                        "sc_tm": [],
                        "sc_rmsd": [],
                        "plddt": [],
                        "designable": [],
                    },
                )
                accumulators["sc_tm"].append(float(values["sc_tm"]))
                accumulators["sc_rmsd"].append(float(values["sc_rmsd"]))
                accumulators["plddt"].append(float(values["plddt"]))
                accumulators["designable"].append(bool(values["designable"]))

    failures = [entry for entry in per_backbone if entry.get("failure")]
    eligible_count = len(per_backbone) - len(failures)
    aggregate: dict[str, object] = {
        "num_backbones": len(per_backbone),
        "num_eligible_backbones": eligible_count,
        "eligible_coverage": (
            eligible_count / len(per_backbone)
            if per_backbone
            else float("nan")
        ),
        "num_explicit_failures": len(failures),
        "failure_reasons": {
            reason: sum(
                1
                for entry in failures
                if str(entry.get("failure", {}).get("reason")) == reason
            )
            for reason in sorted(
                {
                    str(entry.get("failure", {}).get("reason"))
                    for entry in failures
                }
            )
        },
        "aar_best": _summ(best_aar),
        "aar_mean": _summ(mean_aar),
        "aar_best_of_n": {
            budget: _summ(values)
            for budget, values in sorted(
                aar_by_budget.items(), key=lambda item: int(item[0])
            )
        },
        "diversity": _summ(diversity),
    }
    if novelty:
        aggregate["novelty"] = _summ(novelty)
        aggregate["fraction_novel"] = float(np.mean(novel_flags))
    if sc_tm:
        aggregate["sc_tm"] = _summ(sc_tm)
        aggregate["sc_rmsd"] = _summ(sc_rmsd)
        aggregate["plddt"] = _summ(plddt)
        aggregate["designable_fraction"] = float(np.mean(designable))
        from evaluation.evaluation_drivers.proteins.statistics import (
            fraction_with_wilson,
        )

        aggregate["self_consistency_best_of_n"] = {}
        for budget, values in sorted(
            sc_by_budget.items(), key=lambda item: int(item[0])
        ):
            fraction, interval = fraction_with_wilson(values["designable"])
            aggregate["self_consistency_best_of_n"][budget] = {
                "sc_tm": _summ(values["sc_tm"]),
                "sc_rmsd": _summ(values["sc_rmsd"]),
                "plddt": _summ(values["plddt"]),
                "designable_fraction": fraction,
                "designable_fraction_ci95": interval,
            }
    return aggregate


def unsupported_length_entry(
    record: BackboneRecord, *, maximum_length: int
) -> dict[str, object]:
    """
    Explicit null result for a target outside the configured model
    envelope.
    """
    return {
        "stable_id": record.stable_id,
        "split": record.split,
        "source": record.source,
        "length": record.length,
        "eligible": False,
        "num_designs": 0,
        "aar_mean": float("nan"),
        "aar_best": float("nan"),
        "aar_per_design": [],
        "aar_best_of_n": {},
        "diversity": sequence_diversity([]),
        "failure": {
            "reason": "unsupported_model_length",
            "maximum_supported_length": int(maximum_length),
            "observed_length": int(record.length),
            ("policy"): (
                "explicit null; target is retained in the benchmark matrix"
            ),
        },
    }


def aggregate_inverse_folding(
    entries_by_method: dict[str, Sequence[dict[str, object]]],
    methods: Sequence[str],
    *,
    num_samples: int,
    sc_designs: int,
) -> dict[str, object]:
    """
    Rebuild the inverse-folding results block from per-(method,
    backbone) records.
    """
    results: dict[str, object] = {}
    for method in methods:
        per_backbone = list(entries_by_method[method])
        results[method] = {
            "method": method,
            "aggregate": aggregate_backbones(per_backbone),
            "per_backbone": per_backbone,
        }

    # The summary mirrors run_inverse_folding's, but reads split and
    # length off the entries because the BackboneRecord objects do not
    # survive the JSONL. Every method covers the same backbones, so the
    # first one describes them all.
    probe = list(entries_by_method[methods[0]]) if methods else []
    results["summary"] = {
        "num_backbones": len(probe),
        "num_samples_per_backbone": int(num_samples),
        "sc_designs_folded": int(sc_designs),
        "methods": list(methods),
        "splits": sorted({entry["split"] for entry in probe}),
        "sources": sorted(
            {entry.get("source", "") for entry in probe if entry.get("source")}
        ),
        "mean_length": float(np.mean([entry["length"] for entry in probe])),
    }
    return results


def make_designer(
    method: str,
    *,
    model=None,
    cfg=None,
    device=None,
    num_samples: int = 8,
    guidance_scale: float | None = None,
    num_steps: int | None = None,
    proteinmpnn_temperature: float = 0.1,
    rng: np.random.Generator | None = None,
) -> Callable[[BackboneRecord, np.ndarray], list[str]]:
    """Build the sequence designer for one method."""
    if method == "model":

        def designer(record, struct_index):
            return _design_with_model(
                model,
                cfg,
                device,
                record,
                struct_index,
                num_samples,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
            )
    elif method == "shuffled":

        def designer(record, struct_index):
            shuffled = shuffle_structure_index(struct_index, rng)
            return _design_with_model(
                model,
                cfg,
                device,
                record,
                shuffled,
                num_samples,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
            )
    elif method == "proteinmpnn":

        def designer(record, struct_index):
            return _design_with_proteinmpnn(
                record, num_samples, proteinmpnn_temperature
            )
    else:
        raise ValueError(
            f"unknown method {method!r}; choose from 'model', 'shuffled', "
            f"'proteinmpnn'."
        )
    return designer


def _read_training_sequences(path: str | None, limit: int | None) -> list[str]:
    """
    Read training sequences for novelty from a FASTA file, optionally
    subsampled.
    """
    if not path:
        return []
    _repo_root_on_path()
    from evaluation.evaluation_drivers.proteins.io import read_fasta

    sequences = read_fasta(Path(path))
    if limit is not None and int(limit) < 0:
        raise ValueError("training sequence limit must be non-negative")
    # Across the setup scripts a limit of zero means "no cap".  Treat it
    # the same way here; the old slice [:0] silently disabled novelty
    # while the launcher claimed to compare against the complete
    # training FASTA.
    if limit and len(sequences) > int(limit):
        sequences = sequences[: int(limit)]
    return sequences


# -----------------------------------------------------------------------------
# Top-level protocol driver.
# -----------------------------------------------------------------------------


def run_inverse_folding(
    *,
    records: Sequence[BackboneRecord],
    methods: Sequence[str],
    training_sequences: Sequence[str],
    model=None,
    cfg=None,
    device=None,
    struct_resolver: _StructureTokenResolver | None = None,
    num_samples: int = 8,
    guidance_scale: float | None = None,
    num_steps: int | None = None,
    sc_designs: int = 1,
    novelty_threshold: float = 0.5,
    proteinmpnn_temperature: float = 0.1,
    seed: int = 0,
) -> dict[str, object]:
    """Run the inverse-folding protocol over a set of backbones and methods."""
    if not records:
        raise ValueError("no backbone records to evaluate.")
    rng = np.random.default_rng(int(seed))
    needs_model = any(method in ("model", "shuffled") for method in methods)
    if needs_model and (
        model is None or cfg is None or struct_resolver is None
    ):
        raise ValueError(
            "methods 'model' and 'shuffled' need a loaded model, cfg, "
            "and structure resolver; pass them or drop those methods."
        )

    struct_indices: list[np.ndarray | None] = [None] * len(records)
    if needs_model:
        struct_indices = [
            struct_resolver.resolve(record) for record in records
        ]

    results: dict[str, object] = {}
    for method in methods:
        results[method] = _evaluate_method(
            method,
            records,
            struct_indices,
            make_designer(
                method,
                model=model,
                cfg=cfg,
                device=device,
                num_samples=num_samples,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                proteinmpnn_temperature=proteinmpnn_temperature,
                rng=rng,
            ),
            training_sequences=training_sequences,
            sc_designs=int(sc_designs),
            novelty_threshold=float(novelty_threshold),
        )

    results["summary"] = {
        "num_backbones": len(records),
        "num_samples_per_backbone": int(num_samples),
        "sc_designs_folded": int(sc_designs),
        "methods": list(methods),
        "splits": sorted({record.split for record in records}),
        "mean_length": float(np.mean([record.length for record in records])),
    }
    return results


def _sha256(path: str | None) -> str | None:
    """Return the SHA-256 of a file, or None when the path is missing."""
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    pinned_checkpoint = os.environ.get("PROTEBIT_CHECKPOINT_SHA256")
    if pinned_checkpoint and candidate.suffix == ".pt":
        return pinned_checkpoint
    _repo_root_on_path()
    from evaluation.evaluation_drivers.proteins.io import sha256

    return sha256(candidate)


def _git(*args: str) -> str | None:
    """Compatibility provenance helper that never invokes git on Lustre."""
    if tuple(args) != ("rev-parse", "HEAD"):
        return None
    from evaluation.evaluation_drivers.proteins.io import repository_commit

    return repository_commit(Path(__file__).resolve().parents[3])


def _provenance_payload(
    args, device, records: Sequence[BackboneRecord]
) -> dict[str, object]:
    """Build the report shell (everything except results) for one run."""
    shard_flags = {"shard_id", "num_shards", "resume"}
    from evaluation.evaluation_drivers.proteins.io import repository_commit

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        ("protocol"): (
            "Inverse folding p(s | z) on frozen benchmark backbones (plan"
            " 10.7)."
        ),
        "benchmark": {
            "sources": sorted(
                {record.source for record in records if record.source}
            ),
            "splits": sorted({record.split for record in records}),
            "num_backbones": len(records),
        },
        "config": {"path": args.config, "sha256": _sha256(args.config)},
        "checkpoint": {
            "path": str(args.checkpoint) if args.checkpoint else None,
            "sha256": _sha256(
                str(args.checkpoint) if args.checkpoint else None
            ),
        },
        "data": {"path": str(args.data)},
        "arguments": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key not in shard_flags
        },
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "git_commit": repository_commit(
                Path(__file__).resolve().parents[3]
            ),
            "git_dirty": None,
            ("git_dirty_note"): (
                "not scanned on shared Lustre; commit identity only"
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the inverse-folding protocol."""
    parser = argparse.ArgumentParser(
        description="Inverse-folding evaluation p(s | z) for the multimodal "
        "bitstream diffusion model on a frozen backbone benchmark."
    )
    parser.add_argument(
        "--data",
        required=True,
        help=(
            "npz file or directory of backbones with coords and native "
            "sequences."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Path to write the JSON report.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["model", "proteinmpnn", "shuffled"],
        choices=["model", "proteinmpnn", "shuffled"],
        help="Which design methods to evaluate.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help=(
            "Restrict to these split labels (for example cath43_test "
            "pdb_heldout)."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Model config for the model-driven methods.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="Model checkpoint (.pt)."
    )
    parser.add_argument(
        "--struct-tokenizer",
        default=None,
        help=(
            "DPLM-2 structure tokenizer checkpoint, used when records "
            "lack struct_index."
        ),
    )
    parser.add_argument(
        "--training-fasta",
        default=None,
        help=(
            "FASTA of training sequences for the novelty and identity "
            "measurement."
        ),
    )
    parser.add_argument(
        "--training-limit",
        type=int,
        default=20000,
        help="Cap on training sequences read for novelty (0 means no cap).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Designs sampled per backbone.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override the number of sampling steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help=(
            "Classifier-free guidance scale for model-driven inverse folding. "
            "The config value is used when omitted."
        ),
    )
    parser.add_argument(
        "--sc-designs",
        type=int,
        default=1,
        help=(
            "Leading generated designs folded for self-consistency (0 "
            "disables)."
        ),
    )
    parser.add_argument(
        "--proteinmpnn-temperature",
        type=float,
        default=0.1,
        help="ProteinMPNN sampling temperature.",
    )
    parser.add_argument(
        "--novelty-threshold",
        type=float,
        default=0.5,
        help="Nearest-identity below this counts as novel.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=None,
        help="Skip backbones shorter than this.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Skip backbones longer than this.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Cap on the number of backbones.",
    )
    parser.add_argument(
        "--device", default="auto", help="Torch device: auto, cpu, or cuda."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling and sampling.",
    )
    add_shard_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the inverse-folding protocol from the command line and write a
    JSON report.
    """
    _repo_root_on_path()
    args = build_parser().parse_args(argv)
    plan = resolve_shard(args)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    records = load_backbone_records(
        args.data,
        splits=args.splits,
        min_len=args.min_len,
        max_len=args.max_len,
        max_examples=args.max_examples,
    )
    if not records:
        raise SystemExit("No backbone records matched the requested filters.")

    model_max_len = None
    if args.config is not None:
        from evaluation.utils import load_config

        loaded_config = load_config(str(args.config))
        configured_max_len = getattr(loaded_config.data, "max_len", None)
        if configured_max_len is not None:
            model_max_len = int(configured_max_len)

    by_id = {record.stable_id: record for record in records}
    units = [
        (method, record.stable_id)
        for method in args.methods
        for record in records
    ]
    all_keys = unit_keys(units, key=lambda unit: f"{unit[0]}/{unit[1]}")
    from evaluation.evaluation_drivers.proteins.io import input_identities

    writer = ShardWriter(
        plan,
        task="inverse_folding",
        meta={
            "methods": list(args.methods),
            "num_samples": int(args.num_samples),
            "sc_designs": int(args.sc_designs),
            "payload": _provenance_payload(args, device, records),
            "input_identities": input_identities(
                [
                    args.data,
                    args.config,
                    args.checkpoint,
                    args.struct_tokenizer,
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
        units,
        plan.shard_id,
        plan.num_shards,
        key=lambda unit: f"{unit[0]}/{unit[1]}",
    )
    todo = [unit for unit in mine if f"{unit[0]}/{unit[1]}" not in done]

    if todo:
        training_sequences = _read_training_sequences(
            args.training_fasta, args.training_limit
        )
        model = None
        cfg = None
        struct_resolver = None
        needs_model = any(
            method in ("model", "shuffled")
            and (
                model_max_len is None
                or by_id[stable_id].length <= model_max_len
            )
            for method, stable_id in todo
        )
        if needs_model:
            if args.config is None or args.checkpoint is None:
                raise SystemExit(
                    "methods 'model' and 'shuffled' require --config and"
                    " --checkpoint."
                )
            model, cfg = _load_model(
                args.config, args.checkpoint, device, args.num_steps
            )
            struct_resolver = _StructureTokenResolver(
                args.struct_tokenizer,
                device="cuda" if device.type == "cuda" else "cpu",
            )

        struct_cache: dict[str, object] = {}
        for method, stable_id in todo:
            key = f"{method}/{stable_id}"
            random_seed = unit_seed(
                args.seed, key, namespace="inverse_folding"
            )
            torch.manual_seed(random_seed)
            np.random.seed(random_seed % (2**32))
            record = by_id[stable_id]
            if model_max_len is not None and record.length > model_max_len:
                entry = unsupported_length_entry(
                    record, maximum_length=model_max_len
                )
                entry["random_seed"] = int(random_seed)
                writer.write(key, entry, kind=method)
                continue
            designer = make_designer(
                method,
                model=model,
                cfg=cfg,
                device=device,
                num_samples=args.num_samples,
                guidance_scale=args.guidance_scale,
                num_steps=args.num_steps,
                proteinmpnn_temperature=args.proteinmpnn_temperature,
                rng=np.random.default_rng(random_seed),
            )
            struct_index = None
            if struct_resolver is not None and method in ("model", "shuffled"):
                if stable_id not in struct_cache:
                    struct_cache[stable_id] = struct_resolver.resolve(record)
                struct_index = struct_cache[stable_id]
            entry = evaluate_backbone(
                record,
                struct_index,
                designer,
                training_sequences=training_sequences,
                sc_designs=int(args.sc_designs),
                novelty_threshold=float(args.novelty_threshold),
            )
            entry["random_seed"] = int(random_seed)
            writer.write(key, entry, kind=method)

    pairs = finalise(plan, writer, all_keys)
    if pairs is None:
        return 0

    entries_by_method: dict[str, list[dict[str, object]]] = {
        method: [] for method in args.methods
    }
    for (method, _stable_id), (_key, entry) in zip(units, pairs, strict=True):
        entries_by_method[method].append(entry)
    results = aggregate_inverse_folding(
        entries_by_method,
        args.methods,
        num_samples=args.num_samples,
        sc_designs=args.sc_designs,
    )

    payload = dict(_provenance_payload(args, device, records))
    payload["results"] = results

    from evaluation.evaluation_drivers.proteins.io import (
        atomic_evaluation_json,
    )

    atomic_evaluation_json(args.out, payload, task="inverse_folding")
    for method in args.methods:
        aggregate = results[method]["aggregate"]
        print(
            f"{method}: "
            f"{json.dumps(aggregate.get('aar_best', {}), sort_keys=True)}"
        )
    print(f"Saved inverse-folding report to {args.out}")
    return 0


if __name__ == "__main__":
    _repo_root_on_path()
    raise SystemExit(main())
