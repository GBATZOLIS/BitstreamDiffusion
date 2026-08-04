"""Frozen-tokenizer audit run before any protein generator training.

This module implements the tokenizer audit called for by the protein plan
(Phase 1, sections 8.2 and 10.4). Before a single generator is trained the
frozen DPLM-2 LFQ structure tokenizer and the pinned 13-bit code have to be
proven correct and well behaved, otherwise every downstream metric is measured
against a silently corrupt target. The audit answers three questions.

First, is the id-to-bits mapping exact and are the cached sequence and structure
arrays consistent? :func:`exact_roundtrip_check` proves that expanding every
stored structure id to bits and reading it back recovers the same id, reusing
the exhaustive codebook fixture from ``data.protein_structure_codec``, and it
asserts that no protein has a sequence array whose residue count disagrees with
its structure array.

Second, is the structure code well balanced? :func:`bit_balance_stats` reports
per-bit marginal frequencies and entropies, pairwise bit correlations, token
usage, codebook perplexity, and per-protein token diversity, all in pure numpy.

Third, does the tokenizer reconstruct real backbones? :func:`reconstruction_report`
encodes and decodes coordinates through the frozen tokenizer and scores RMSD,
TM-score, and lDDT against the input, broken down by length and by source, and
applies the median TM-score gate of greater than 0.9 that the plan requires.

The module imports with only numpy and torch present. The structure metrics and
the heavy DPLM-2 tokenizer are imported lazily inside the functions that use them
so importing this file never pulls in the folding-model dependency stack; a
missing tokenizer or missing metrics module raises an actionable error pointing
at the download script and the pinned requirements files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow running this file directly (python evaluation/proteins/tokenizer_audit.py)
# as well as importing it as a package module; the absolute data and evaluation
# imports below require the repository root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.protein_structure_codec import (
    DEFAULT_STRUCT_CODEC,
    LFQBitCodec,
    build_full_roundtrip_fixture,
)

AUDIT_VERSION = "1.0"
DEFAULT_TM_GATE = 0.9
DEFAULT_LENGTH_BIN_EDGES: Tuple[int, ...] = (100, 200, 300, 400, 500)


# -----------------------------------------------------------------------------
# Input coercion helpers
# -----------------------------------------------------------------------------


def _extract_struct_record(
    item,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[str]]:
    """Return the structure ids, optional sequence ids, and source for one item.

    An item may be a raw array of structure ids, a mapping carrying at least a
    ``struct_index`` entry (and optionally ``seq_ids`` and ``source``), or an
    object exposing ``struct_index`` (and optionally ``seq_ids`` and ``source``)
    as attributes, such as a ``data.protein_multimodal.RowRecord``.
    """
    struct = None
    seq = None
    source = None
    if isinstance(item, dict):
        for key in ("struct_index", "struct_ids", "index"):
            if key in item:
                struct = item[key]
                break
        for key in ("seq_ids", "seq_index"):
            if key in item:
                seq = item[key]
                break
        source = item.get("source")
    elif hasattr(item, "struct_index"):
        struct = getattr(item, "struct_index")
        seq = getattr(item, "seq_ids", None)
        source = getattr(item, "source", None)
    else:
        struct = item
    if struct is None:
        raise ValueError("could not find structure ids in an audit input item")
    struct_arr = np.asarray(struct).reshape(-1).astype(np.int64)
    seq_arr = None if seq is None else np.asarray(seq).reshape(-1)
    source_str = None if source is None else str(source)
    return struct_arr, seq_arr, source_str


def _extract_coords_record(
    item,
) -> Tuple[np.ndarray, Optional[str], Optional[str]]:
    """Return the backbone coordinates, source, and stable id for one item.

    An item may be a raw coordinate array of shape ``[L, 4, 3]`` (or ``[L, 12]``),
    a mapping carrying a ``coords`` entry (and optionally ``source`` and a
    ``stable_id`` or ``id``), or an object exposing those as attributes.
    """
    coords = None
    source = None
    stable_id = None
    if isinstance(item, dict):
        for key in ("coords", "coord", "backbone"):
            if key in item:
                coords = item[key]
                break
        source = item.get("source")
        stable_id = item.get("stable_id", item.get("id"))
    elif hasattr(item, "coords"):
        coords = getattr(item, "coords")
        source = getattr(item, "source", None)
        stable_id = getattr(item, "stable_id", getattr(item, "id", None))
    else:
        coords = item
    if coords is None:
        raise ValueError(
            "could not find coordinates in a reconstruction input item"
        )
    return (
        _as_backbone_coords(coords),
        (None if source is None else str(source)),
        (None if stable_id is None else str(stable_id)),
    )


def _as_backbone_coords(coords) -> np.ndarray:
    """Coerce coordinates to a contiguous ``[L, 4, 3]`` float array (N, CA, C, O)."""
    arr = np.ascontiguousarray(np.asarray(coords, dtype=np.float64))
    if arr.ndim == 3 and arr.shape[1] >= 4 and arr.shape[2] == 3:
        return np.ascontiguousarray(arr[:, :4, :])
    if arr.ndim == 2 and arr.shape[1] == 12:
        return arr.reshape(-1, 4, 3)
    raise ValueError(
        f"coords must have shape [L, 4, 3] or [L, 12] for backbone atoms N, CA, C, O; "
        f"got shape {arr.shape}"
    )


# -----------------------------------------------------------------------------
# 1. Exact round-trip and length-consistency check
# -----------------------------------------------------------------------------


def exact_roundtrip_check(
    struct_index_arrays,
    *,
    seq_index_arrays: Optional[Sequence] = None,
    codec: LFQBitCodec = DEFAULT_STRUCT_CODEC,
) -> dict:
    """Prove structure ids round-trip through bits and match the sequence lengths.

    The check has two parts. The exhaustive part reuses
    ``data.protein_structure_codec.build_full_roundtrip_fixture`` to prove that
    every id in the whole codebook survives an id-to-bits-to-id round trip under
    ``codec``; this pins the bit order. The per-protein part expands each stored
    structure id array to bits and reads it back, confirming that the cached data
    is in range and lossless.

    ``struct_index_arrays`` is an iterable of items understood by
    :func:`_extract_struct_record`. When those items carry a sequence array, or
    when a parallel ``seq_index_arrays`` is supplied, the function asserts that
    each protein's sequence and structure residue counts agree. Any out-of-range
    id, any round-trip failure, or any residue-length mismatch raises a
    ``ValueError`` because each is a hard corruption that must block generator
    training. On success it returns a summary dictionary.
    """
    records = list(struct_index_arrays)
    seq_list = None if seq_index_arrays is None else list(seq_index_arrays)
    if seq_list is not None and len(seq_list) != len(records):
        raise ValueError(
            "seq_index_arrays length does not match struct_index_arrays length "
            f"({len(seq_list)} vs {len(records)})"
        )

    fixture = build_full_roundtrip_fixture(codec)
    if not fixture["exact_roundtrip"]:
        raise ValueError(
            "codec fails the exhaustive id-to-bits-to-id round trip; the LFQ bit "
            "order is corrupt and no structure data may be trusted"
        )

    num_proteins = 0
    num_residues = 0
    num_length_pairs = 0
    out_of_range: List[int] = []
    roundtrip_failures: List[int] = []
    length_mismatches: List[dict] = []

    for i, item in enumerate(records):
        struct, seq_from_item, _ = _extract_struct_record(item)
        num_proteins += 1
        length = int(struct.shape[0])
        num_residues += length

        if struct.size and (
            int(struct.min()) < 0 or int(struct.max()) >= codec.codebook_size
        ):
            out_of_range.append(i)
        else:
            bits = codec.index_to_bits_np(struct)
            recovered = codec.bits_to_index_np(bits)
            if not np.array_equal(struct, recovered):
                roundtrip_failures.append(i)

        seq = seq_from_item
        if seq is None and seq_list is not None:
            seq = np.asarray(seq_list[i]).reshape(-1)
        if seq is not None:
            num_length_pairs += 1
            seq_len = int(np.asarray(seq).reshape(-1).shape[0])
            if seq_len != length:
                length_mismatches.append(
                    {"index": i, "seq_len": seq_len, "struct_len": length}
                )

    if out_of_range:
        raise ValueError(
            f"structure ids out of range [0, {codec.codebook_size - 1}] in "
            f"{len(out_of_range)} protein(s); first offending index {out_of_range[0]}"
        )
    if roundtrip_failures:
        raise ValueError(
            f"id-to-bits-to-id round trip failed for {len(roundtrip_failures)} "
            f"protein(s); first offending index {roundtrip_failures[0]}"
        )
    if length_mismatches:
        first = length_mismatches[0]
        raise ValueError(
            "sequence and structure residue counts disagree for "
            f"{len(length_mismatches)} protein(s); first at index {first['index']} "
            f"has seq_len {first['seq_len']} and struct_len {first['struct_len']}"
        )

    return {
        "exact_roundtrip": True,
        "num_proteins": num_proteins,
        "num_residues": num_residues,
        "convention_hash": codec.convention_hash(),
        "codebook": {
            "codebook_size": int(fixture["codebook_size"]),
            "exact_roundtrip": bool(fixture["exact_roundtrip"]),
            "bits_hash": fixture["bits_hash"],
            "convention_hash": fixture["convention_hash"],
        },
        "length_check": {
            "num_pairs_checked": num_length_pairs,
            "num_mismatches": 0,
        },
    }


# -----------------------------------------------------------------------------
# 2. Bit-balance statistics
# -----------------------------------------------------------------------------


def _bernoulli_entropy_bits(prob: np.ndarray) -> np.ndarray:
    """Return the per-element Bernoulli entropy in bits for probabilities ``prob``."""
    p = np.clip(np.asarray(prob, dtype=np.float64), 0.0, 1.0)
    out = np.zeros_like(p)
    mask = (p > 0.0) & (p < 1.0)
    pm = p[mask]
    out[mask] = -(pm * np.log2(pm) + (1.0 - pm) * np.log2(1.0 - pm))
    return out


def _pairwise_bit_correlation(bits: np.ndarray) -> np.ndarray:
    """Return the Pearson correlation matrix of bit columns, constant bits as zero.

    ``bits`` is a ``[N, D]`` array. Columns with zero variance (a bit that is
    always 0 or always 1) have an undefined correlation with the other bits;
    those entries are reported as zero and the diagonal is set to one.
    """
    n = bits.shape[0]
    if n == 0:
        d = bits.shape[1]
        return np.eye(d, dtype=np.float64)
    centered = bits - bits.mean(axis=0, keepdims=True)
    std = bits.std(axis=0)
    covariance = (centered.T @ centered) / float(n)
    denom = np.outer(std, std)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0.0, covariance / denom, 0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def bit_balance_stats(
    struct_index_arrays,
    *,
    codec: LFQBitCodec = DEFAULT_STRUCT_CODEC,
    top_k_tokens: int = 10,
) -> dict:
    """Report the balance and diversity of the structure code, in pure numpy.

    The returned dictionary contains, over all residues pooled across proteins,
    the per-bit marginal frequency of a set bit, the per-bit Bernoulli entropy in
    bits, and the pairwise Pearson correlation matrix of the bits. It also reports
    token usage over the codebook: the number of distinct tokens observed, the
    fraction of the codebook covered, the empirical token entropy in bits, and the
    codebook perplexity (two raised to that entropy). Finally it summarizes
    per-protein token diversity, the ratio of distinct tokens to residues within
    each protein. This function has no external dependency beyond numpy.
    """
    records = list(struct_index_arrays)
    num_dims = int(codec.num_dims)
    codebook_size = int(codec.codebook_size)

    per_protein_diversity: List[float] = []
    parts: List[np.ndarray] = []
    for item in records:
        struct, _, _ = _extract_struct_record(item)
        if struct.size == 0:
            continue
        unique_tokens = int(np.unique(struct).size)
        per_protein_diversity.append(unique_tokens / float(struct.size))
        parts.append(struct)

    all_ids = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    num_residues = int(all_ids.size)

    if num_residues == 0:
        return {
            "num_residues": 0,
            "num_proteins": len(records),
            "num_dims": num_dims,
            "per_bit_frequency": [0.0] * num_dims,
            "per_bit_entropy_bits": [0.0] * num_dims,
            "mean_bit_entropy_bits": 0.0,
            "pairwise_bit_correlation": np.eye(num_dims).tolist(),
            "max_abs_offdiagonal_correlation": 0.0,
            "token_usage": {
                "unique_tokens": 0,
                "codebook_size": codebook_size,
                "coverage": 0.0,
                "entropy_bits": 0.0,
                "max_entropy_bits": float(np.log2(codebook_size)),
                "perplexity": 1.0,
                "most_frequent": [],
            },
            "per_protein_diversity": _stat_block([]),
        }

    bits = codec.index_to_bits_np(all_ids).astype(np.float64)
    marginal = bits.mean(axis=0)
    per_bit_entropy = _bernoulli_entropy_bits(marginal)
    correlation = _pairwise_bit_correlation(bits)
    offdiag = correlation[~np.eye(num_dims, dtype=bool)]
    max_abs_offdiag = float(np.max(np.abs(offdiag))) if offdiag.size else 0.0

    counts = np.bincount(all_ids, minlength=codebook_size).astype(np.int64)
    probs = counts / float(counts.sum())
    nonzero = probs[probs > 0.0]
    token_entropy = float(-(nonzero * np.log2(nonzero)).sum())
    unique_tokens = int((counts > 0).sum())
    coverage = unique_tokens / float(codebook_size)
    order = np.argsort(counts)[::-1]
    most_frequent = [
        [int(token), int(counts[token])]
        for token in order[: max(0, int(top_k_tokens))]
        if counts[token] > 0
    ]

    return {
        "num_residues": num_residues,
        "num_proteins": len(records),
        "num_dims": num_dims,
        "per_bit_frequency": marginal.tolist(),
        "per_bit_entropy_bits": per_bit_entropy.tolist(),
        "mean_bit_entropy_bits": float(per_bit_entropy.mean()),
        "pairwise_bit_correlation": correlation.tolist(),
        "max_abs_offdiagonal_correlation": max_abs_offdiag,
        "token_usage": {
            "unique_tokens": unique_tokens,
            "codebook_size": codebook_size,
            "coverage": coverage,
            "entropy_bits": token_entropy,
            "max_entropy_bits": float(np.log2(codebook_size)),
            "perplexity": float(2.0**token_entropy),
            "most_frequent": most_frequent,
        },
        "per_protein_diversity": _stat_block(per_protein_diversity),
    }


# -----------------------------------------------------------------------------
# 3. Reconstruction report
# -----------------------------------------------------------------------------


def _stat_block(values) -> dict:
    """Return count, median, mean, min, max, and std for a sequence of values."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "median": float("nan"),
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def _length_bin_label(length: int, edges: Sequence[int]) -> str:
    """Return a human-readable length-bin label for the given residue count."""
    low = 1
    for edge in edges:
        if length <= edge:
            return f"{low}-{int(edge)}"
        low = int(edge) + 1
    return f"{low}+"


def _grouped_metric_stats(
    per_protein: Sequence[dict], key_fn: Callable[[dict], str]
) -> dict:
    """Group per-protein records by ``key_fn`` and summarize each metric per group."""
    groups: Dict[str, List[dict]] = {}
    for record in per_protein:
        groups.setdefault(key_fn(record), []).append(record)
    summary: Dict[str, dict] = {}
    for key in sorted(groups):
        items = groups[key]
        summary[key] = {
            "count": len(items),
            "tm_score": _stat_block(x["tm_score"] for x in items),
            "rmsd": _stat_block(x["rmsd"] for x in items),
            "lddt": _stat_block(x["lddt"] for x in items),
        }
    return summary


def _resolve_tokenizer(tokenizer, device: str = "cpu"):
    """Return an object with ``encode`` and ``decode``, loading lazily if needed.

    ``tokenizer`` may already be a loaded runner exposing callable ``encode`` and
    ``decode`` methods, in which case it is returned unchanged, or a path or
    Hugging Face repo id for the frozen DPLM-2 structure tokenizer checkpoint, in
    which case ``evaluation.proteins.dplm_struct_tokenizer.load_struct_tokenizer``
    is imported lazily and used to load it. A missing tokenizer or a missing DPLM
    inference environment raises an actionable ``RuntimeError``.
    """
    if tokenizer is None:
        raise RuntimeError(
            "reconstruction_report requires a loaded structure tokenizer or a "
            "checkpoint path. Download the frozen DPLM-2 tokenizer with\n"
            "  python scripts/proteins/setup/download_dplm_tokenizer.py\n"
            "and install the DPLM inference environment listed in "
            "the dplm-inference extra (uv sync --extra dplm-inference) (see scripts/proteins/setup/setup_evaluation.sh)."
        )
    if callable(getattr(tokenizer, "encode", None)) and callable(
        getattr(tokenizer, "decode", None)
    ):
        return tokenizer
    if isinstance(tokenizer, (str, Path)):
        try:
            from evaluation.proteins.dplm_struct_tokenizer import (
                load_struct_tokenizer,
            )
        except Exception as exc:  # noqa: BLE001 - surface any import failure uniformly
            raise RuntimeError(
                "Could not import the DPLM structure tokenizer loader. Install the "
                "DPLM inference environment from the dplm-inference extra (uv sync --extra dplm-inference) "
                "(see scripts/proteins/setup/setup_evaluation.sh)."
            ) from exc
        return load_struct_tokenizer(tokenizer, device=device)
    raise TypeError(
        "tokenizer must be a loaded encode/decode object or a checkpoint path, "
        f"got {type(tokenizer).__name__}"
    )


def reconstruction_report(
    coords_list,
    tokenizer,
    *,
    tm_gate: float = DEFAULT_TM_GATE,
    device: str = "cpu",
    length_bin_edges: Sequence[int] = DEFAULT_LENGTH_BIN_EDGES,
) -> dict:
    """Encode and decode backbones through the tokenizer and score reconstruction.

    Each entry in ``coords_list`` is understood by :func:`_extract_coords_record`
    and supplies a backbone of shape ``[L, 4, 3]`` (atoms N, CA, C, O) plus an
    optional source label. Every backbone is encoded to structure ids and decoded
    back to coordinates through ``tokenizer``, and the reconstruction is scored
    with the alpha-carbon RMSD, the TM-score, and the lDDT from
    ``evaluation.proteins.structure_metrics``. Results are broken down by length
    bin and by source. The plan gate is applied to the median TM-score, which must
    exceed ``tm_gate`` (0.9 by default) for the tokenizer to be usable as a target.

    The structure metrics and the DPLM tokenizer are imported lazily so this
    function is the only place that touches heavy or external dependencies.
    """
    try:
        from evaluation.proteins.structure_metrics import lddt, rmsd, tm_score
    except Exception as exc:  # noqa: BLE001 - surface any import failure uniformly
        raise RuntimeError(
            "reconstruction_report needs evaluation.proteins.structure_metrics, "
            "which imports with numpy only. Install the protein evaluation "
            "requirements from the protein-eval extra (uv sync --extra protein-eval) "
            "(see scripts/proteins/setup/setup_evaluation.sh)."
        ) from exc

    runner = _resolve_tokenizer(tokenizer, device=device)

    per_protein: List[dict] = []
    for i, item in enumerate(coords_list):
        coords, source, stable_id = _extract_coords_record(item)
        length = int(coords.shape[0])
        index = np.asarray(runner.encode(coords)).reshape(-1)
        recon = _as_backbone_coords(runner.decode(index))
        ca_true = coords[:, 1, :]
        ca_recon = recon[:, 1, :]
        per_protein.append(
            {
                "index": i,
                "stable_id": stable_id if stable_id is not None else str(i),
                "source": source if source is not None else "unknown",
                "length": length,
                "tm_score": float(tm_score(ca_recon, ca_true)),
                "rmsd": float(rmsd(ca_recon, ca_true, superpose=True)),
                "lddt": float(lddt(recon, coords)),
            }
        )

    if not per_protein:
        raise ValueError(
            "reconstruction_report requires at least one structure"
        )

    tm_values = np.asarray(
        [p["tm_score"] for p in per_protein], dtype=np.float64
    )
    median_tm = float(np.median(tm_values))

    return {
        "num_structures": len(per_protein),
        "tm_gate": float(tm_gate),
        "median_tm_score": median_tm,
        "passed_tm_gate": bool(median_tm > tm_gate),
        "overall": {
            "tm_score": _stat_block(tm_values),
            "rmsd": _stat_block(p["rmsd"] for p in per_protein),
            "lddt": _stat_block(p["lddt"] for p in per_protein),
        },
        "by_length": _grouped_metric_stats(
            per_protein,
            lambda r: _length_bin_label(r["length"], length_bin_edges),
        ),
        "by_source": _grouped_metric_stats(per_protein, lambda r: r["source"]),
        "per_protein": per_protein,
    }


# -----------------------------------------------------------------------------
# Command-line entry point
# -----------------------------------------------------------------------------


def _git_commit() -> Optional[str]:
    """Return the current git commit hash, or None when git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best effort
        return None


def _json_default(value):
    """Coerce numpy scalars and arrays so the audit JSON is always serializable."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def _load_shard_struct_records(
    shard_dir: Path, split: Optional[str], limit: int
) -> List[dict]:
    """Read paired shards and return per-protein structure and sequence arrays.

    Each returned record carries ``struct_index``, ``seq_ids``, ``source``, and
    ``stable_id`` for one chain, read straight from the ragged ``.npz`` shards and
    JSON metadata sidecars produced by ``scripts/proteins/setup/prepare_dplm_paired.py``.
    """
    shard_dir = Path(shard_dir)
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Build paired shards with "
            "scripts/proteins/setup/prepare_dplm_paired.py or point --shard-dir at the cache."
        )
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    records: List[dict] = []
    for shard_name in manifest.get("shards", []):
        npz = np.load(shard_dir / f"{shard_name}.npz")
        offsets = np.asarray(npz["offsets"], dtype=np.int64)
        struct_index = np.asarray(npz["struct_index"])
        seq_ids = np.asarray(npz["seq_ids"])
        meta_path = shard_dir / f"{shard_name}.meta.json"
        rows_meta = None
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as handle:
                rows_meta = json.load(handle).get("rows")
        for row_idx in range(int(offsets.shape[0]) - 1):
            meta = (
                rows_meta[row_idx]
                if (rows_meta and row_idx < len(rows_meta))
                else {}
            )
            if split is not None and meta.get("split", "train") != split:
                continue
            start, end = int(offsets[row_idx]), int(offsets[row_idx + 1])
            records.append(
                {
                    "struct_index": struct_index[start:end],
                    "seq_ids": seq_ids[start:end],
                    "source": meta.get("source", "unknown"),
                    "stable_id": meta.get(
                        "stable_id", f"{shard_name}:{row_idx}"
                    ),
                }
            )
            if limit and len(records) >= limit:
                return records
    return records


def _load_coords_records(coords_dir: Path, limit: int) -> List[dict]:
    """Read raw backbone-coordinate ``.npz`` files into reconstruction records.

    Two on-disk layouts are supported: a single-chain file with a ``coords``
    array of shape ``[L, 4, 3]`` (or ``[L, 12]``), and a ragged multi-chain file
    that additionally carries an ``offsets`` array. The source label defaults to
    the first path component beneath ``coords_dir`` or the directory name.
    """
    coords_dir = Path(coords_dir)
    files = sorted(coords_dir.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No .npz coordinate files found under {coords_dir}. Each file must "
            "contain a 'coords' array of shape [L, 4, 3] (backbone N, CA, C, O), "
            "optionally with 'offsets' for a ragged multi-chain shard."
        )
    records: List[dict] = []
    for path in files:
        relative = path.relative_to(coords_dir)
        default_source = (
            relative.parts[0] if len(relative.parts) > 1 else coords_dir.name
        )
        with np.load(path, allow_pickle=True) as loaded:
            keys = list(loaded.files)
            if "coords" not in keys:
                raise ValueError(f"{path}: missing required 'coords' array")
            coords_raw = np.asarray(loaded["coords"])
            source = (
                str(loaded["source"]) if "source" in keys else default_source
            )
            offsets = (
                np.asarray(loaded["offsets"], dtype=np.int64)
                if "offsets" in keys
                else None
            )
        if offsets is not None:
            coords_flat = _as_backbone_coords(coords_raw)
            for row_idx in range(int(offsets.shape[0]) - 1):
                start, end = int(offsets[row_idx]), int(offsets[row_idx + 1])
                records.append(
                    {
                        "coords": coords_flat[start:end],
                        "source": source,
                        "stable_id": f"{path.stem}:{row_idx}",
                    }
                )
                if limit and len(records) >= limit:
                    return records
        else:
            records.append(
                {
                    "coords": _as_backbone_coords(coords_raw),
                    "source": source,
                    "stable_id": path.stem,
                }
            )
            if limit and len(records) >= limit:
                return records
    return records


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the tokenizer audit command-line entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen structure-tokenizer audit before any generator "
            "training and write the results to an audit JSON. The exact round-trip "
            "and bit-balance checks run on cached structure ids from a paired "
            "shard directory; the reconstruction check needs raw coordinates and "
            "the frozen DPLM-2 tokenizer."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="Paired shard directory with a manifest.json for the round-trip and "
        "bit-balance checks.",
    )
    parser.add_argument(
        "--coords-dir",
        type=Path,
        default=None,
        help="Directory of raw backbone-coordinate .npz files for the "
        "reconstruction check.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Local checkpoint or Hugging Face repo id for the frozen DPLM-2 "
        "structure tokenizer, required only for the reconstruction check.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device used by the tokenizer during reconstruction.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Restrict the shard rows to this split (for example train or valid).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap the number of proteins used for the id-level checks (0 disables).",
    )
    parser.add_argument(
        "--recon-limit",
        type=int,
        default=0,
        help="Cap the number of structures used for the reconstruction check "
        "(0 disables).",
    )
    parser.add_argument(
        "--tm-gate",
        type=float,
        default=DEFAULT_TM_GATE,
        help="Median TM-score the reconstruction must exceed to pass the gate.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path of the audit JSON to write.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the audit from the command line and write the audit JSON.

    Returns 0 when every requested gate passes and 1 when any gate fails so the
    audit can block a training pipeline. At least one of ``--shard-dir`` or
    ``--coords-dir`` must be supplied.
    """
    args = build_parser().parse_args(argv)
    if args.shard_dir is None and args.coords_dir is None:
        raise SystemExit("provide at least one of --shard-dir or --coords-dir")

    audit: Dict[str, object] = {
        "audit_version": AUDIT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "codec": DEFAULT_STRUCT_CODEC.spec(),
        "convention_hash": DEFAULT_STRUCT_CODEC.convention_hash(),
        "inputs": {
            "shard_dir": str(args.shard_dir) if args.shard_dir else None,
            "coords_dir": str(args.coords_dir) if args.coords_dir else None,
            "tokenizer_path": str(args.tokenizer_path)
            if args.tokenizer_path
            else None,
            "split": args.split,
            "limit": args.limit,
            "recon_limit": args.recon_limit,
        },
        "exact_roundtrip": None,
        "bit_balance": None,
        "reconstruction": None,
    }

    roundtrip_ok: Optional[bool] = None
    reconstruction_ok: Optional[bool] = None

    if args.shard_dir is not None:
        struct_records = _load_shard_struct_records(
            args.shard_dir, args.split, args.limit
        )
        try:
            audit["exact_roundtrip"] = exact_roundtrip_check(struct_records)
            roundtrip_ok = True
        except ValueError as exc:
            audit["exact_roundtrip"] = {
                "exact_roundtrip": False,
                "error": str(exc),
            }
            roundtrip_ok = False
        audit["bit_balance"] = bit_balance_stats(struct_records)

    if args.coords_dir is not None:
        coords_records = _load_coords_records(
            args.coords_dir, args.recon_limit
        )
        reconstruction = reconstruction_report(
            coords_records,
            args.tokenizer_path,
            tm_gate=args.tm_gate,
            device=args.device,
        )
        audit["reconstruction"] = reconstruction
        reconstruction_ok = bool(reconstruction["passed_tm_gate"])

    all_passed = all(
        gate is not False for gate in (roundtrip_ok, reconstruction_ok)
    )
    audit["gates"] = {
        "exact_roundtrip_ok": roundtrip_ok,
        "reconstruction_tm_gate_ok": reconstruction_ok,
        "all_passed": bool(all_passed),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, default=_json_default)

    print(f"tokenizer audit written to {args.out}", flush=True)
    print(
        "gates: "
        f"exact_roundtrip_ok={roundtrip_ok} "
        f"reconstruction_tm_gate_ok={reconstruction_ok} "
        f"all_passed={all_passed}",
        flush=True,
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
