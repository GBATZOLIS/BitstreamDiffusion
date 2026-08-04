#!/usr/bin/env python3
"""Build a sequence-only UniRef50 replay corpus in the paired-shard format.

The multimodal training curriculum (plan section 6.6 step 3) interleaves roughly
25 percent sequence-only UniRef50 replay by token budget so the warm-started
sequence knowledge is not forgotten while the model learns structure. The
production ``Trainer`` mixes replay by pulling whole batches from a second
``ProteinMultimodalDataset`` whose rows carry a sequence but no structure
(``struct_mask`` all False). This script builds that corpus from the frozen local
EvoDiff March-2020 UniRef50 dataset so the config's ``sequence_replay_root`` points
at a real, loadable paired-shard directory.

Each kept row stores canonical-20 sequence ids with their per-residue mask, a
placeholder structure index of zeros with ``struct_mask`` all False (so the
collator marks structure ABSENT and the structure loss contributes nothing on a
replay step), source label ``uniref50``, and split ``train``. The manifest pins
the same sequence and structure codec convention hashes as the M0 paired cache so
``ProteinMultimodalDataset`` accepts it without a convention mismatch.

Residue policy mirrors ``prepare_dplm_paired.py``: canonical residues
(``ACDEFGHIKLMNPQRSTVWY``) get their id with the sequence mask set; every other
symbol is stored as id 0 with the mask cleared; a whole row is dropped only when
its non-canonical fraction exceeds ``NONCANONICAL_ROW_DROP_FRACTION``. Sequences
longer than ``--max-len`` are leading-cropped, matching the M0 build. Only the
frozen EvoDiff ``train`` split is read, so the replay pool never contains the
EvoDiff ``valid``/``rtest`` sequences.

Idempotency: the build records its parameters in ``build_state.json`` and skips
the whole build when the manifest, that state file, and every shard are already
present with the same parameters and shard count.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

AA_TO_ID: Dict[str, int] = {aa: i for i, aa in enumerate(CANONICAL_AA)}

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


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    """Write JSON to ``path`` via a temporary file and an atomic replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def map_sequence_to_ids(aa_seq: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Map an amino-acid string to canonical-20 ids, a mask, and a flagged count."""
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
    """Return the directory that holds consensus.fasta and the offset metadata."""
    manifest_path = root / "frozen_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Prepare the frozen EvoDiff UniRef50 dataset "
            "with scripts/proteins/setup/prepare_evodiff_uniref50.py --download."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_dir = root / manifest["dataset"].get("data_dir", "uniref50")
    if not (data_dir / "consensus.fasta").exists():
        raise FileNotFoundError(
            f"Missing consensus.fasta under {data_dir}; the UniRef50 corpus is "
            "not fully prepared."
        )
    return data_dir


def iter_train_sequences(
    data_dir: Path, min_len: int, max_len: int, max_rows: int
) -> Tuple[List[RowRecord], Dict[str, int]]:
    """Read up to ``max_rows`` kept sequence-only rows from the EvoDiff train split."""
    meta = np.load(data_dir / "lengths_and_offsets.npz", allow_pickle=False)
    offsets = np.asarray(meta["seq_offsets"], dtype=np.int64)
    ells = np.asarray(meta["ells"], dtype=np.int64)
    with (data_dir / "splits.json").open("r", encoding="utf-8") as handle:
        train_indices = np.asarray(json.load(handle)["train"], dtype=np.int64)

    counts = {
        "input_rows_read": 0,
        "kept_rows": 0,
        "dropped_short": 0,
        "dropped_noncanonical": 0,
        "truncated_rows": 0,
        "residues_seq_flagged": 0,
    }
    rows: List[RowRecord] = []
    fasta = (data_dir / "consensus.fasta").open("rb")
    try:
        for source_idx in train_indices:
            if len(rows) >= max_rows:
                break
            counts["input_rows_read"] += 1
            full_len = int(ells[source_idx])
            if full_len < min_len:
                counts["dropped_short"] += 1
                continue
            fasta.seek(int(offsets[source_idx]))
            raw = fasta.readline().rstrip(b"\r\n")
            if len(raw) != full_len:
                # Skip a corrupt/mismatched row rather than abort the whole build.
                continue
            if full_len > max_len:
                raw = raw[:max_len]
                counts["truncated_rows"] += 1
            aa_seq = raw.decode("ascii", errors="replace")
            length = len(aa_seq)
            seq_ids, seq_mask, n_noncanonical = map_sequence_to_ids(aa_seq)
            if n_noncanonical > NONCANONICAL_ROW_DROP_FRACTION * length:
                counts["dropped_noncanonical"] += 1
                continue
            counts["residues_seq_flagged"] += n_noncanonical
            counts["kept_rows"] += 1
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
                    f"read {counts['input_rows_read']} rows, kept {counts['kept_rows']}"
                )
    finally:
        fasta.close()
    return rows, counts


def _build_params(args: argparse.Namespace, seq_codec: SequenceBitCodec) -> Dict[str, object]:
    """Return the parameter dict that keys idempotency for this build."""
    return {
        "source_root": str(args.source_root),
        "max_rows": int(args.max_rows),
        "min_len": int(args.min_len),
        "max_len": int(args.max_len),
        "shard_size": int(args.shard_size),
        "seq_codec_hash": seq_codec.hash(),
        "struct_convention_hash": DEFAULT_STRUCT_CODEC.convention_hash(),
        "noncanonical_row_drop_fraction": NONCANONICAL_ROW_DROP_FRACTION,
    }


def _already_built(out_dir: Path, params: Dict[str, object]) -> bool:
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


def build(args: argparse.Namespace) -> Dict[str, object]:
    """Run the full build and return the manifest extra payload that was written."""
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_codec = SequenceBitCodec()
    params = _build_params(args, seq_codec)

    if not args.force and _already_built(out_dir, params):
        _log(f"already built at {out_dir} with matching parameters; nothing to do")
        return {"skipped": True}

    data_dir = _resolve_data_dir(Path(args.source_root))
    _log(f"reading EvoDiff train sequences from {data_dir}")
    rows, counts = iter_train_sequences(
        data_dir, int(args.min_len), int(args.max_len), int(args.max_rows)
    )
    if not rows:
        raise RuntimeError("no rows kept; check the source corpus and filters")

    shard_size = int(args.shard_size)
    shard_names: List[str] = []
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
        "source_dataset": "EvoDiff March-2020 UniRef50 (frozen local snapshot)",
        "split_read": "train",
        "structure_tokens_origin": "none_sequence_only",
        "residue_policy": (
            "canonical-20 ids; non-canonical residues masked, rows dropped when "
            f"non-canonical fraction exceeds {NONCANONICAL_ROW_DROP_FRACTION}"
        ),
        "min_len": int(args.min_len),
        "max_len": int(args.max_len),
        "shard_size": shard_size,
        "max_rows": int(args.max_rows),
        "post_filter_counts": counts,
        "counts_per_split": {SPLIT_LABEL: counts["kept_rows"]},
        "counts_per_source": {SOURCE_LABEL: counts["kept_rows"]},
    }
    write_manifest(out_dir, shard_names, seq_codec=seq_codec, extra=extra)
    _atomic_write_json(
        out_dir / "build_state.json",
        {"params": params, "n_shards": len(shard_names), "counts": counts},
    )
    _log(
        f"done: {counts['kept_rows']} kept of {counts['input_rows_read']} read "
        f"into {len(shard_names)} shards at {out_dir}"
    )
    return extra


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
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
        "--force",
        action="store_true",
        help="Rebuild even when an identical build is already present.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments and run the build."""
    build(parse_args(argv))


if __name__ == "__main__":
    main()
