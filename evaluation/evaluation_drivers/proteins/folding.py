"""
Generate structures from sequences and compare them with native
backbones.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Make the repository root importable whether this file is imported as a
# module or executed directly as a script, so the absolute data /
# evaluation imports below resolve in both cases.
_REPO_ROOT = _Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data.protein_multimodal import CANONICAL_AA
from data.protein_structure_codec import DEFAULT_STRUCT_CODEC
from evaluation.evaluation_drivers.proteins import structure_metrics
from evaluation.evaluation_drivers.proteins.io import (
    atomic_evaluation_json,
    input_identities,
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
    "FoldingTarget",
    "sequence_to_ids",
    "ids_to_sequence",
    "load_targets",
    "backbone_agreement",
    "structure_token_accuracy",
    "tokenizer_reconstruction_ceiling",
    "fold_target",
    "evaluate_target",
    "aggregate_folding",
    "evaluate_folding",
    "load_folding_model",
    "main",
]


# Amino-acid id lookup for the canonical-20 alphabet used by the
# sequence codec.
_AA_TO_ID = {aa: i for i, aa in enumerate(CANONICAL_AA)}
SEQ_VOCAB_SIZE = len(CANONICAL_AA)  # 20

# Per-metric optimization direction. max metrics improve as they rise
# (TM, lDDT, token accuracies); min metrics improve as they fall (RMSD).
# The best-of-N reduction over samples uses this direction.
METRIC_DIRECTIONS: dict[str, str] = {
    "tm_score": "max",
    "lddt": "max",
    "rmsd": "min",
    "lfq_bit_accuracy": "max",
    "exact_index_accuracy": "max",
}

# Task-summary keys reported per target, one section per condition
# evaluated.
TASK_KEYS = (
    "forward_folding",
    "shuffled_control",
    "unconditional_control",
    "tokenizer_ceiling",
)

# Keys that describe a summary rather than a metric and are excluded
# when aggregating summaries across targets.
_NON_AGGREGATED_KEYS = frozenset({"n_samples"})

# Chain-length buckets (half-open, low inclusive) for the by-length
# breakdown.
_LENGTH_BUCKETS = (
    (0, 100, "len_0_99"),
    (100, 200, "len_100_199"),
    (200, 300, "len_200_299"),
    (300, 400, "len_300_399"),
    (400, 512, "len_400_511"),
    (512, 10**9, "len_512_plus"),
)

# Training sequence-identity buckets (half-open, low inclusive) for the
# by-identity breakdown. A target with no identity annotation is
# labelled below.
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
    """
    One forward-folding target: an observed sequence and its native
    backbone.
    """

    target_id: str
    seq_ids: np.ndarray
    native_coords: np.ndarray | None = None
    native_mask: np.ndarray | None = None
    native_struct_index: np.ndarray | None = None
    source: str = "unknown"
    release: str = ""
    train_seq_identity: float = float("nan")

    @property
    def length(self) -> int:
        """Number of residues, taken from the observed sequence."""
        return int(np.asarray(self.seq_ids).shape[0])


def sequence_to_ids(sequence: str) -> np.ndarray:
    """Map a canonical-20 amino-acid string to an int64 id array."""
    text = str(sequence).strip().upper()
    ids = np.empty(len(text), dtype=np.int64)
    unknown = sorted({ch for ch in text if ch not in _AA_TO_ID})
    if unknown:
        raise ValueError(
            f"sequence contains non-canonical residues {unknown}; the "
            f"folding target must use only {CANONICAL_AA}"
        )
    for i, ch in enumerate(text):
        ids[i] = _AA_TO_ID[ch]
    return ids


def ids_to_sequence(seq_ids) -> str:
    """
    Map canonical-20 amino-acid ids back to a string, the inverse of the
    map.
    """
    ids = np.asarray(seq_ids, dtype=np.int64).reshape(-1)
    if ids.size and (ids.min() < 0 or ids.max() >= SEQ_VOCAB_SIZE):
        raise ValueError(f"seq_ids out of range [0, {SEQ_VOCAB_SIZE - 1}]")
    return "".join(CANONICAL_AA[i] for i in ids)


def _load_coords_from_npz(path: Path, key: str | None) -> np.ndarray:
    """Load a backbone coordinate array from an .npz file."""
    if not path.exists():
        raise FileNotFoundError(f"native coordinate file not found: {path}")
    data = np.load(path)
    if key is not None:
        return np.asarray(data[key], dtype=np.float32)
    for candidate in ("coords", "backbone", "atom_positions", "bb_coords"):
        if candidate in data:
            return np.asarray(data[candidate], dtype=np.float32)
    raise KeyError(
        f"{path} contains no recognised coordinate array; available keys "
        f"are {list(data.keys())}. Pass 'coords_key' in the target entry."
    )


def _load_native_coords(
    entry: Mapping[str, object], base: Path
) -> np.ndarray | None:
    """
    Resolve the native backbone for one target entry, or None when
    absent.
    """
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
        coords_key = entry.get("coords_key")
        arr = _load_coords_from_npz(path, coords_key)  # type: ignore[arg-type]
    if arr.ndim != 3 or arr.shape[1] < 4 or arr.shape[2] != 3:
        raise ValueError(
            f"native coords must have shape [L, 4, 3] with atoms N, CA, C,"
            f" O; got shape {tuple(arr.shape)}"
        )
    return arr[:, :4, :].astype(np.float32)


def _load_native_mask(
    entry: Mapping[str, object], base: Path, length: int
) -> np.ndarray:
    """Resolve the native-residue mask used by date-split/CAMEO structures."""
    inline = entry.get("native_mask", entry.get("bb_mask"))
    if inline is not None:
        mask = np.asarray(inline, dtype=bool).reshape(-1)
    else:
        location = entry.get("native_mask_path") or entry.get("coords_path")
        mask = None
        if location:
            path = Path(str(location))
            if not path.is_absolute():
                path = base / path
            with np.load(path) as loaded:
                for key in ("native_mask", "bb_mask", "mask"):
                    if key in loaded:
                        mask = np.asarray(loaded[key], dtype=bool).reshape(-1)
                        break
        if mask is None:
            mask = np.ones(int(length), dtype=bool)
    if mask.shape[0] != int(length):
        raise ValueError(
            f"native mask length {mask.shape[0]} does not match sequence "
            f"length {length}"
        )
    if not bool(mask.any()):
        raise ValueError("native mask contains no resolved residues")
    return mask


def _target_from_entry(
    entry: Mapping[str, object], base: Path, index: int
) -> FoldingTarget:
    """Build one FoldingTarget from a manifest entry."""
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
    native_mask = _load_native_mask(entry, base, int(seq_ids.shape[0]))
    if coords is not None and coords.shape[0] != seq_ids.shape[0]:
        raise ValueError(
            f"target {target_id}: coordinate length {coords.shape[0]} does"
            f" not match sequence length {seq_ids.shape[0]}"
        )
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
        native_mask=native_mask,
        native_struct_index=struct_index,
        source=str(entry.get("source", "unknown")),
        release=str(entry.get("release", "")),
        train_seq_identity=identity,
    )


def load_targets(manifest_path) -> list[FoldingTarget]:
    """Load forward-folding targets from a JSON manifest."""
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        entries = raw.get("targets", raw.get("rows"))
        if entries is None:
            raise ValueError(
                "targets manifest object must contain a 'targets' or "
                "'rows' list"
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
    """Return the CA track [L, 3] from backbone [L, 4, 3] or CA input."""
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1] >= 2 and arr.shape[2] == 3:
        return np.ascontiguousarray(arr[:, 1, :])
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    raise ValueError(
        f"coordinates must be backbone [L, 4, 3] or CA [L, 3]; got shape "
        f"{tuple(arr.shape)}"
    )


def backbone_agreement(
    pred_coords, native_coords, native_mask=None
) -> dict[str, float]:
    """
    Score a predicted backbone against a native one with the geometry
    judges.
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
    if native_mask is not None:
        mask = np.asarray(native_mask, dtype=bool).reshape(-1)[:length]
        if not bool(mask.any()):
            return {
                "tm_score": float("nan"),
                "rmsd": float("nan"),
                "lddt": float("nan"),
            }
        pred_ca = pred_ca[mask]
        native_ca = native_ca[mask]
        pred_bb = pred_bb[mask]
        native_bb = native_bb[mask]
    return {
        "tm_score": float(structure_metrics.tm_score(pred_ca, native_ca)),
        "rmsd": float(
            structure_metrics.rmsd(pred_ca, native_ca, superpose=True)
        ),
        "lddt": float(structure_metrics.lddt(pred_bb, native_bb)),
    }


def structure_token_accuracy(pred_index, native_index) -> dict[str, float]:
    """
    Token-level diagnostics between a sampled and a native LFQ structure
    code.
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
) -> dict[str, float]:
    """Reduce samples to expected-quality and nested best-of-N values."""
    from evaluation.evaluation_drivers.proteins.statistics import (
        power_of_two_prefix_budgets,
    )

    summary: dict[str, float] = {"n_samples": int(len(samples))}
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
        for budget in power_of_two_prefix_budgets(len(samples)):
            prefix = [
                float(sample[metric])
                for sample in samples[:budget]
                if metric in sample
                and sample[metric] is not None
                and math.isfinite(float(sample[metric]))
            ]
            if prefix:
                summary[f"{metric}_best_at_{budget}"] = float(
                    max(prefix) if direction == "max" else min(prefix)
                )
    return summary


def _run_generate(
    model, cfg, task: str, length: int, num_samples: int, device, observed
):
    """Sample structure token ids for one target, importing generate lazily."""
    from evaluation.evaluation_drivers.proteins.generate_multimodal import (
        generate,
    )

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
) -> dict[str, float]:
    """Sample structure for one target and summarize the agreement metrics."""
    length = target.length
    if task == "forward_folding":
        seq_ids = (
            target.seq_ids if observed_seq_ids is None else observed_seq_ids
        )
        observed: dict[str, object] | None = {
            "seq_ids": np.asarray(seq_ids, dtype=np.int64)
        }
    elif task == "structure_marginal":
        observed = None
    else:
        raise ValueError(
            f"fold_target supports 'forward_folding' or "
            f"'structure_marginal'; got {task!r}"
        )

    pred_index = _run_generate(
        model, cfg, task, length, num_samples, device, observed
    )

    samples: list[dict[str, float]] = []
    for row in range(pred_index.shape[0]):
        idx = pred_index[row]
        record: dict[str, float] = {}
        if native_struct_index is not None:
            record.update(structure_token_accuracy(idx, native_struct_index))
        if tokenizer is not None and target.native_coords is not None:
            pred_coords = tokenizer.decode(np.asarray(idx, dtype=np.uint16))
            record.update(
                backbone_agreement(
                    pred_coords, target.native_coords, target.native_mask
                )
            )
        samples.append(record)
    return _summarize_samples(samples)


def tokenizer_reconstruction_ceiling(target: FoldingTarget, tokenizer):
    """
    Encode and decode the native backbone to measure the reconstruction
    ceiling.
    """
    if tokenizer is None or target.native_coords is None:
        return None, None
    if target.native_mask is not None and not bool(
        np.asarray(target.native_mask, dtype=bool).all()
    ):
        # Encoding a full chain with unresolved zero-coordinate residues
        # creates artificial geometry.  Date-split targets instead carry
        # the official precomputed structure tokens and are
        # decoded/masked by tokenizer_audit.
        return None, None
    native_index = np.asarray(
        tokenizer.encode(np.asarray(target.native_coords, dtype=np.float32))
    )
    recon = tokenizer.decode(np.asarray(native_index, dtype=np.uint16))
    metrics = backbone_agreement(recon, target.native_coords)
    return metrics, native_index


def _shuffle_sequence(seq_ids, rng: np.random.Generator) -> np.ndarray:
    """
    Return a randomly permuted copy of the sequence ids (shuffled
    control).
    """
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
    """
    Return the identity-bucket label, or the unknown label for NaN
    identity.
    """
    if identity is None or (
        isinstance(identity, float) and math.isnan(identity)
    ):
        return _IDENTITY_UNKNOWN
    for low, high, label in _IDENTITY_BUCKETS:
        if low <= identity < high:
            return label
    return _IDENTITY_BUCKETS[-1][2]


def _summ(values: Sequence[float | None]) -> dict[str, object]:
    """
    Summarize a list of scalar values as count, mean, median, std, min,
    max.
    """
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
    from evaluation.evaluation_drivers.proteins.statistics import (
        bootstrap_mean_ci,
    )

    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "mean_ci95": bootstrap_mean_ci(finite),
        "median": float(np.median(finite)),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _aggregate_summaries(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """
    Aggregate a list of per-target task summaries into per-metric
    statistics.
    """
    keys: list[str] = []
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
) -> dict[str, object]:
    """Aggregate every task section across a group of per-target records."""
    aggregated: dict[str, object] = {}
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
) -> dict[str, object]:
    """Group per-target records by a bucket label and aggregate each group."""
    groups: dict[str, list[Mapping[str, object]]] = {}
    for rec in per_target:
        groups.setdefault(str(rec.get(bucket_key)), []).append(rec)
    out: dict[str, object] = {}
    for bucket, records in sorted(groups.items()):
        out[bucket] = {
            "num_targets": len(records),
            "metrics": _aggregate_over_targets(records),
        }
    return out


def evaluate_target(
    model,
    cfg,
    target: FoldingTarget,
    device,
    *,
    tokenizer=None,
    num_samples: int = 8,
    include_controls: bool = True,
    include_ceiling: bool = True,
    rng: np.random.Generator | None = None,
) -> dict[str, object]:
    """Run every folding condition for one target and return its record."""
    native_index = target.native_struct_index
    ceiling_metrics = None
    if include_ceiling:
        ceiling_metrics, encoded_index = tokenizer_reconstruction_ceiling(
            target, tokenizer
        )
        if native_index is None and encoded_index is not None:
            native_index = encoded_index

    record: dict[str, object] = {
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
        if rng is None:
            rng = np.random.default_rng(0)
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
    return record


def aggregate_folding(
    per_target: Sequence[dict[str, object]], num_samples: int
) -> dict[str, object]:
    """Reduce per-target folding records to the report body."""
    per_target = list(per_target)
    return {
        "protocol": {
            "name": "forward_folding_p_z_given_s",
            "plan_section": "10.6",
            "benchmark": "CAMEO 2022 / PDB deposition-date split",
            "observed_modality": "sequence (5 bits per residue clamped)",
            "sampled_modality": "structure (13 LFQ bits per residue)",
            "geometry_judge": (
                "evaluation.evaluation_drivers.proteins.structure_metrics"
            ),
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
) -> dict[str, object]:
    """Run the full forward-folding protocol over a set of targets."""
    rng = np.random.default_rng(int(seed))
    per_target = [
        evaluate_target(
            model,
            cfg,
            target,
            device,
            tokenizer=tokenizer,
            num_samples=num_samples,
            include_controls=include_controls,
            include_ceiling=include_ceiling,
            rng=rng,
        )
        for target in targets
    ]
    return aggregate_folding(per_target, num_samples)


def load_folding_model(
    config_path, checkpoint_path, device, *, num_steps: int | None = None
):
    """Load the multimodal bitstream model and its config for folding."""
    from evaluation.evaluation_drivers.proteins.io import (
        load_binary_protein_model,
    )

    return load_binary_protein_model(
        config_path,
        checkpoint_path,
        device,
        num_steps=num_steps,
        context="forward folding",
    )


def _load_tokenizer(tokenizer_path, device):
    """Load the frozen DPLM-2 structure tokenizer, importing it lazily."""
    from evaluation.evaluation_drivers.proteins.dplm_struct_tokenizer import (
        load_struct_tokenizer,
    )

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
        help=(
            "Local DPLM-2 structure tokenizer checkpoint directory or HF "
            "repo id"
        ),
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
    add_shard_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Command-line entry point: fold a manifest of targets and write a
    JSON report.
    """
    args = _build_arg_parser().parse_args(argv)
    plan = resolve_shard(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    targets = load_targets(args.targets)
    if args.max_targets and args.max_targets > 0:
        targets = targets[: args.max_targets]
    if not targets:
        raise ValueError(f"no folding targets loaded from {args.targets}")

    all_keys = unit_keys(targets, key=lambda target: target.target_id)
    writer = ShardWriter(
        plan,
        task="forward_folding",
        meta={
            "num_samples": int(args.num_samples),
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "targets_manifest": str(args.targets),
            "device": str(
                args.device or ("cuda" if torch.cuda.is_available() else "cpu")
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
                    args.targets,
                    args.tokenizer_path,
                ]
            ),
        },
    )
    writer.write_manifest(all_keys)
    if not args.resume:
        writer.reset()
    done = writer.completed_keys() if args.resume else set()

    mine = partition(
        targets,
        plan.shard_id,
        plan.num_shards,
        key=lambda target: target.target_id,
    )
    todo = [target for target in mine if target.target_id not in done]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = cfg = tokenizer = None
    if todo:
        model, cfg = load_folding_model(
            args.config, args.checkpoint, device, num_steps=args.num_steps
        )
        if not args.skip_decode:
            tokenizer = _load_tokenizer(args.tokenizer_path, device)

    for target in todo:
        random_seed = unit_seed(
            args.seed, target.target_id, namespace="forward_folding"
        )
        torch.manual_seed(random_seed)
        np.random.seed(random_seed % (2**32))
        rng = np.random.default_rng(random_seed)
        record = evaluate_target(
            model,
            cfg,
            target,
            device,
            tokenizer=tokenizer,
            num_samples=args.num_samples,
            include_controls=not args.skip_controls,
            include_ceiling=(not args.skip_ceiling) and tokenizer is not None,
            rng=rng,
        )
        record["random_seed"] = int(random_seed)
        writer.write(target.target_id, record)

    pairs = finalise(plan, writer, all_keys)
    if pairs is None:
        return 0

    result = aggregate_folding(
        [record for _key, record in pairs], args.num_samples
    )
    result["config"] = str(args.config)
    result["checkpoint"] = str(args.checkpoint)
    result["targets_manifest"] = str(args.targets)
    result["device"] = str(device)

    atomic_evaluation_json(Path(args.out), result, task="forward_folding")
    print(json.dumps(result.get("overall", {}), indent=2, sort_keys=True))
    print(f"Saved forward-folding metrics to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
