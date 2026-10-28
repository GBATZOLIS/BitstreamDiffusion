#!/usr/bin/env python3
"""Prepare the frozen UniRef50 sequence replay subset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.protein_multimodal import (  # noqa: E402
    CANONICAL_AA,
    RowRecord,
    SequenceBitCodec,
    write_manifest,
    write_shard,
)
from data.protein_structure_codec import DEFAULT_STRUCT_CODEC  # noqa: E402

AA_TO_ID: dict[str, int] = {aa: i for i, aa in enumerate(CANONICAL_AA)}

DEFAULT_SOURCE_ROOT = "datasets/uniref50_evodiff_2020"
DEFAULT_OUT_DIR = "datasets/uniref50_seqonly_replay"
DEFAULT_MAX_ROWS = 400_000
DEFAULT_MIN_LEN = 40
DEFAULT_MAX_LEN = 512
DEFAULT_SHARD_SIZE = 8192
NONCANONICAL_ROW_DROP_FRACTION = 0.10
SOURCE_LABEL = "uniref50"
SPLIT_LABEL = "train"
LOG_EVERY = 50_000
LOG_PREFIX = "[prepare_uniref50_replay]"


def _log(message: str) -> None:
    """Print a progress line with a stable prefix and no decoration."""
    print(f"{LOG_PREFIX} {message}", flush=True)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Write JSON to path via a temporary file and an atomic replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def map_sequence_to_ids(aa_seq: str) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Map an amino-acid string to canonical-20 ids, a mask, and a flagged
    count.
    """
    length = len(aa_seq)
    seq_ids = np.zeros(length, dtype=np.uint8)
    seq_mask = np.zeros(length, dtype=bool)
    n_noncanonical = 0
    for i, residue in enumerate(aa_seq):
        token = AA_TO_ID.get(residue)
        if token is None:
            n_noncanonical += 1
            continue
        seq_ids[i] = token
        seq_mask[i] = True
    return seq_ids, seq_mask, n_noncanonical


def _resolve_data_dir(root: Path) -> Path:
    """
    Return the directory that holds consensus.fasta and the offset
    metadata.
    """
    manifest_path = root / "frozen_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Prepare the frozen EvoDiff UniRef50"
            f" dataset with "
            f"scripts/proteins/setup/prepare_evodiff_uniref50.py "
            f"--download."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_dir = root / manifest["dataset"].get("data_dir", "uniref50")
    if not (data_dir / "consensus.fasta").exists():
        raise FileNotFoundError(
            f"Missing consensus.fasta under {data_dir}; the UniRef50 "
            f"corpus is not fully prepared."
        )
    return data_dir


def iter_train_sequences(
    data_dir: Path,
    min_len: int,
    max_len: int,
    max_rows: int,
    *,
    target_residues: int = 0,
    seed: int = 0,
    crop: str = "random",
) -> tuple[list[RowRecord], dict[str, int]]:
    """Read kept sequence-only rows from the EvoDiff train split."""
    meta = np.load(data_dir / "lengths_and_offsets.npz", allow_pickle=False)
    offsets = np.asarray(meta["seq_offsets"], dtype=np.int64)
    ells = np.asarray(meta["ells"], dtype=np.int64)
    with (data_dir / "splits.json").open("r", encoding="utf-8") as handle:
        train_indices = np.asarray(json.load(handle)["train"], dtype=np.int64)

    # THE fix for defect 1. Without this the corpus is length-sorted.
    rng = np.random.default_rng(int(seed))
    train_indices = rng.permutation(train_indices)

    counts = {
        "input_rows_read": 0,
        "kept_rows": 0,
        "dropped_short": 0,
        "dropped_noncanonical": 0,
        "truncated_rows": 0,
        "residues_seq_flagged": 0,
        "residues_kept": 0,
    }
    rows: list[RowRecord] = []
    fasta = (data_dir / "consensus.fasta").open("rb")
    try:
        for source_idx in train_indices:
            if max_rows and len(rows) >= max_rows:
                break
            if target_residues and counts["residues_kept"] >= target_residues:
                break
            counts["input_rows_read"] += 1
            full_len = int(ells[source_idx])
            if full_len < min_len:
                counts["dropped_short"] += 1
                continue
            fasta.seek(int(offsets[source_idx]))
            raw = fasta.readline().rstrip(b"\r\n")
            if len(raw) != full_len:
                # Skip a corrupt/mismatched row rather than abort the
                # whole build.
                continue
            if full_len > max_len:
                # Defect 2. A random crop keeps the length distribution
                # at the cap but samples the whole protein rather than
                # always its N-terminus.
                if crop == "random":
                    start = int(rng.integers(0, full_len - max_len + 1))
                else:
                    start = 0
                raw = raw[start : start + max_len]
                counts["truncated_rows"] += 1
            aa_seq = raw.decode("ascii", errors="replace")
            length = len(aa_seq)
            seq_ids, seq_mask, n_noncanonical = map_sequence_to_ids(aa_seq)
            if n_noncanonical > NONCANONICAL_ROW_DROP_FRACTION * length:
                counts["dropped_noncanonical"] += 1
                continue
            counts["residues_seq_flagged"] += n_noncanonical
            counts["kept_rows"] += 1
            counts["residues_kept"] += length
            rows.append(
                RowRecord(
                    stable_id=f"uniref50_{int(source_idx)}",
                    seq_ids=seq_ids,
                    struct_index=np.zeros(length, dtype=np.uint16),
                    seq_mask=seq_mask,
                    struct_mask=np.zeros(length, dtype=bool),
                    source=SOURCE_LABEL,
                    cluster_id=f"uniref50_{int(source_idx)}",
                    split=SPLIT_LABEL,
                    accession=f"uniref50_{int(source_idx)}",
                )
            )
            if counts["input_rows_read"] % LOG_EVERY == 0:
                _log(
                    f"read {counts['input_rows_read']} rows, kept "
                    f"{counts['kept_rows']}"
                )
    finally:
        fasta.close()
    return rows, counts


def _length_histogram(
    rows: list[RowRecord], max_len: int
) -> dict[str, object]:
    """Summarise the achieved row lengths so a length bias is visible."""
    lengths = np.asarray([len(r.seq_ids) for r in rows], dtype=np.int64)
    if lengths.size == 0:
        return {}
    edges = [0, 64, 128, 192, 256, 320, 384, 448, max_len + 1]
    counts, _ = np.histogram(lengths, bins=edges)
    return {
        "mean": round(float(lengths.mean()), 2),
        "median": int(np.median(lengths)),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "fraction_at_max_len": round(float((lengths >= max_len).mean()), 4),
        "bin_edges": edges,
        "bin_counts": [int(c) for c in counts],
    }


def _build_params(
    args: argparse.Namespace, seq_codec: SequenceBitCodec
) -> dict[str, object]:
    """Return the parameter dict that keys idempotency for this build."""
    return {
        "source_root": str(args.source_root),
        "max_rows": int(args.max_rows),
        "target_residues": int(getattr(args, "target_residues", 0) or 0),
        "seed": int(getattr(args, "seed", 0) or 0),
        "crop": str(getattr(args, "crop", "random")),
        "min_len": int(args.min_len),
        "max_len": int(args.max_len),
        "shard_size": int(args.shard_size),
        "seq_codec_hash": seq_codec.hash(),
        "struct_convention_hash": DEFAULT_STRUCT_CODEC.convention_hash(),
        "noncanonical_row_drop_fraction": NONCANONICAL_ROW_DROP_FRACTION,
    }


def _already_built(out_dir: Path, params: dict[str, object]) -> bool:
    """Return True when a prior identical build is fully present on disk."""
    state_path = out_dir / "build_state.json"
    manifest_path = out_dir / "manifest.json"
    if not state_path.exists() or not manifest_path.exists():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if state.get("params") != params:
        return False
    for shard_name in manifest.get("shards", []):
        if not (out_dir / f"{shard_name}.npz").exists():
            return False
        if not (out_dir / f"{shard_name}.meta.json").exists():
            return False
    return True


def build(args: argparse.Namespace) -> dict[str, object]:
    """
    Run the full build and return the manifest extra payload that was
    written.
    """
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_codec = SequenceBitCodec()
    params = _build_params(args, seq_codec)

    if not args.force and _already_built(out_dir, params):
        _log(
            f"already built at {out_dir} with matching parameters; nothing"
            f" to do"
        )
        return {"skipped": True}

    data_dir = _resolve_data_dir(Path(args.source_root))
    _log(f"reading EvoDiff train sequences from {data_dir}")
    rows, counts = iter_train_sequences(
        data_dir,
        int(args.min_len),
        int(args.max_len),
        int(args.max_rows),
        target_residues=int(getattr(args, "target_residues", 0) or 0),
        seed=int(getattr(args, "seed", 0) or 0),
        crop=str(getattr(args, "crop", "random")),
    )
    if not rows:
        raise RuntimeError("no rows kept; check the source corpus and filters")

    shard_size = int(args.shard_size)
    shard_names: list[str] = []
    for shard_index, start in enumerate(range(0, len(rows), shard_size)):
        name = f"shard_{shard_index:05d}"
        write_shard(
            out_dir,
            name,
            rows[start : start + shard_size],
            tokenizer_revision="sequence_only_replay",
        )
        shard_names.append(name)
        _log(f"wrote {name} ({len(rows[start : start + shard_size])} rows)")

    extra = {
        "corpus": "uniref50_sequence_only_replay",
        "source_root": str(args.source_root),
        ("source_dataset"): (
            "EvoDiff March-2020 UniRef50 (frozen local snapshot)"
        ),
        "split_read": "train",
        "structure_tokens_origin": "none_sequence_only",
        "residue_policy": (
            f"canonical-20 ids; non-canonical residues masked, rows "
            f"dropped when non-canonical fraction exceeds "
            f"{NONCANONICAL_ROW_DROP_FRACTION}"
        ),
        "min_len": int(args.min_len),
        "max_len": int(args.max_len),
        "shard_size": shard_size,
        "max_rows": int(args.max_rows),
        "target_residues": int(getattr(args, "target_residues", 0) or 0),
        "seed": int(getattr(args, "seed", 0) or 0),
        "crop": str(getattr(args, "crop", "random")),
        "post_filter_counts": counts,
        "counts_per_split": {SPLIT_LABEL: counts["kept_rows"]},
        "counts_per_source": {SOURCE_LABEL: counts["kept_rows"]},
        # The achieved length distribution is recorded so the ordering
        # defect this build fixes stays visible if it ever comes back.
        # The defective corpus had 99.3% of rows sitting exactly at
        # max_len and a mean length of ~510 against a source mean of
        # 283.
        "length_histogram": _length_histogram(rows, int(args.max_len)),
    }
    write_manifest(out_dir, shard_names, seq_codec=seq_codec, extra=extra)
    _atomic_write_json(
        out_dir / "build_state.json",
        {"params": params, "n_shards": len(shard_names), "counts": counts},
    )
    _log(
        f"done: {counts['kept_rows']} kept of {counts['input_rows_read']} "
        f"read into {len(shard_names)} shards at {out_dir}"
    )
    return extra


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments for the replay-corpus builder."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a sequence-only UniRef50 replay corpus in the paired-shard "
            "format read by ProteinMultimodalDataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--min-len", type=int, default=DEFAULT_MIN_LEN)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument(
        "--target-residues",
        type=int,
        default=0,
        help=(
            "Stop once this many residues have been kept. Residues, not rows, "
            "are the unit a token budget is expressed in. 0 means use "
            "--max-rows instead."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Seeds the permutation of the source index. The source index is "
            "ordered by DESCENDING LENGTH, so without the permutation the "
            "corpus is fragments of the longest proteins only."
        ),
    )
    parser.add_argument(
        "--crop",
        choices=("random", "head"),
        default="random",
        help=(
            "How to shorten a sequence longer than --max-len. random samples "
            "the whole protein across the corpus; head keeps only the "
            "N-terminus, which is the old behaviour."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when an identical build is already present.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the build."""
    build(parse_args(argv))


if __name__ == "__main__":
    main()
