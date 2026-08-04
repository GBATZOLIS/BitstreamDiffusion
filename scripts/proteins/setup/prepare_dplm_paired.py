#!/usr/bin/env python3
"""Build the M0 paired sequence+structure shards from airkingbd/pdb_swissprot.

This is the tier B / M0 data-preparation step of plan section 7.3. It reads the
released DPLM-2 paired corpus ``airkingbd/pdb_swissprot`` at a pinned revision
(220,482 rows across its ``train`` and ``valid`` configs), maps each amino-acid
chain to canonical-20 ids, parses the pre-computed DPLM structure-token string to
uint16 LFQ ids, builds the per-residue sequence and structure masks, assigns a
rebuilt contamination-aware split, and writes immutable shards in the format read
by ``data.protein_multimodal.ProteinMultimodalDataset``.

The heavy Hugging Face ``datasets`` dependency is imported lazily inside the
iteration function, so this module imports and runs ``--help`` with only numpy
and torch present. The DPLM structure tokenizer itself is never loaded here: the
structure tokens ship inside the dataset, so this step only parses them.

Documented residue policy. Each residue is flagged, not silently coerced. A
canonical residue (one of ``ACDEFGHIKLMNPQRSTVWY``) receives its id with the
sequence mask set true. Any non-canonical residue (X, B, Z, U, O, gaps, lower
case, or any other symbol) is stored as id 0 with the sequence mask cleared, so
the model treats that position as having no valid amino-acid identity. A whole
row is dropped only when the fraction of non-canonical residues exceeds
``NONCANONICAL_ROW_DROP_FRACTION``; such rows are counted in the manifest. A
structure token outside ``[0, 8191]`` is stored as 0 with the structure mask
cleared. Rows whose structure-token count disagrees with the sequence length, or
whose sequence is empty, are dropped and counted.

Documented split policy. The released train/valid configs are not inherited as a
train/validation partition. Instead the split is rebuilt so it is deterministic
and contamination aware. Rows from a recent held-out benchmark source (CAMEO or
CASP style labels, the temporal-holdout dimension of the policy) are assigned to
``test``. Every other row is assigned by a stable hash of its sequence-cluster id
(falling back to its accession when no cluster is recorded), which keeps every
member of a cluster in the same split and prevents near-duplicate leakage between
train and validation. The release carries no per-entry deposition date, so the
temporal dimension is realised through the recognised benchmark sources rather
than a finer per-row cutoff; this is recorded in the manifest.

Idempotency. Shards are written in a stable stream order and each is skipped when
an identical file with a matching content hash already exists, so a re-run only
rewrites shards whose contents actually changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

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
from data.protein_structure_codec import (  # noqa: E402
    DEFAULT_STRUCT_CODEC,
    DPLM_GIT_REPO,
    STRUCT_CODEBOOK_SIZE,
)

# -----------------------------------------------------------------------------
# Pinned source identity and policy constants
# -----------------------------------------------------------------------------

HF_DATASET_REPO = "airkingbd/pdb_swissprot"
# Pinned revision (main branch of the released dataset, dated 2025-05-09).
DEFAULT_HF_REVISION = "eafc13c5b50f51543dae8d4d2895332d4f8e5261"
DATASET_SOURCE_URL = f"https://huggingface.co/datasets/{HF_DATASET_REPO}"
# The pinned revision exposes exactly these two configs, each with a single
# "train" split. The valid config holds the CAMEO 2022 benchmark targets.
DATASET_CONFIG_SPLITS: Tuple[Tuple[str, str], ...] = (
    ("train", "train"),
    ("valid", "train"),
)
# The dataset card declares no machine-readable license; this records the known
# provenance of its two components for downstream attribution.
DATASET_LICENSE_NOTE = (
    "AlphaFold DB (afdb_swissprot) entries under CC-BY-4.0; RCSB PDB and CAMEO "
    "targets in the public domain; see the source dataset card for details."
)

AA_TO_ID: Dict[str, int] = {aa: i for i, aa in enumerate(CANONICAL_AA)}

# Drop a row only when it is mostly non-canonical; otherwise mask per residue.
NONCANONICAL_ROW_DROP_FRACTION = 0.10
# Deterministic hash split of the clustered training pool into train and val.
DEFAULT_VAL_FRACTION = 0.02
SPLIT_HASH_BUCKETS = 1_000_000
# Source labels routed to the temporal test holdout (recent benchmark targets).
TEMPORAL_HOLDOUT_SOURCE_PREFIXES = ("cameo", "casp")
SPLIT_POLICY_VERSION = "cluster_hash_v1"

DEFAULT_OUT_DIR = "datasets/dplm_paired_m0"
DEFAULT_MAX_LEN = 512
DEFAULT_SHARD_SIZE = 8192
LOG_EVERY = 20_000

LOG_PREFIX = "[prepare_dplm_paired]"


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _log(message: str) -> None:
    """Print a progress line with a stable prefix and no decoration."""
    print(f"{LOG_PREFIX} {message}", flush=True)


def _to_float(value: object) -> float:
    """Return ``value`` as a float, or NaN when it is missing or unparseable."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    """Write JSON to ``path`` via a temporary file and an atomic replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _require_datasets():
    """Import and return the Hugging Face ``datasets`` module, or explain how to install it."""
    try:
        import datasets  # noqa: WPS433 - lazy, heavy optional dependency
    except (
        Exception
    ) as exc:  # pragma: no cover - exercised only without the dep
        raise RuntimeError(
            "The Hugging Face 'datasets' library is required to build the paired "
            "shards but is not importable. Install it into the training venv with\n"
            "  pip install 'datasets>=2.19' 'huggingface_hub>=0.23'\n"
            "or provision the DPLM inference environment via "
            "scripts/proteins/setup/setup_evaluation.sh dplm "
            "(see the dplm-inference extra (uv sync --extra dplm-inference))."
        ) from exc
    return datasets


# -----------------------------------------------------------------------------
# Per-row conversion
# -----------------------------------------------------------------------------


def map_sequence_to_ids(aa_seq: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Map an amino-acid string to canonical-20 ids, a mask, and a flagged count.

    Returns ``(seq_ids, seq_mask, n_noncanonical)`` where ``seq_ids`` is uint8 in
    ``[0, 19]`` (non-canonical positions stored as 0), ``seq_mask`` is true only
    at canonical residues, and ``n_noncanonical`` counts the masked positions.
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


def parse_struct_tokens(struct_seq: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Parse a comma-separated DPLM structure-token string to uint16 LFQ ids.

    Returns ``(struct_index, struct_mask, n_out_of_range)``. Tokens outside the
    codebook range ``[0, STRUCT_CODEBOOK_SIZE)`` are stored as 0 with their mask
    cleared and counted. Raises ``ValueError`` when a field is not an integer.
    """
    fields = [
        tok for tok in struct_seq.replace(" ", ",").split(",") if tok != ""
    ]
    length = len(fields)
    struct_index = np.zeros(length, dtype=np.uint16)
    struct_mask = np.zeros(length, dtype=bool)
    n_out_of_range = 0
    for i, field in enumerate(fields):
        value = int(field)
        if 0 <= value < STRUCT_CODEBOOK_SIZE:
            struct_index[i] = value
            struct_mask[i] = True
        else:
            n_out_of_range += 1
    return struct_index, struct_mask, n_out_of_range


def is_temporal_holdout_source(source: str) -> bool:
    """Return whether a source label marks a recent benchmark temporal holdout."""
    lowered = (source or "").strip().lower()
    return lowered.startswith(TEMPORAL_HOLDOUT_SOURCE_PREFIXES)


def assign_split(
    accession: str,
    cluster: str,
    source: str,
    *,
    val_fraction: float,
    salt: str,
) -> str:
    """Assign a contamination-aware split for one row.

    Recent benchmark sources are routed to ``test`` (the temporal holdout). Every
    other row is placed by a stable hash of its cluster id, falling back to its
    accession, so a whole sequence cluster always lands in the same split.
    """
    if is_temporal_holdout_source(source):
        return "test"
    key = (cluster or "").strip() or (accession or "").strip()
    digest = hashlib.sha256(f"{salt}|{key}".encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % SPLIT_HASH_BUCKETS
    fraction = bucket / float(SPLIT_HASH_BUCKETS)
    return "val" if fraction < val_fraction else "train"


def _parse_pdb_chain(source: str, accession: str) -> str:
    """Return a chain identifier for PDB rows shaped ``<id>_<chain>``, else empty."""
    if source == "pdb" and "_" in accession:
        return accession.split("_", 1)[1]
    return ""


def build_row_record(
    row: Dict[str, object],
    *,
    max_len: int,
    release: str,
    val_fraction: float,
    salt: str,
) -> Tuple[Optional[RowRecord], str, Dict[str, int]]:
    """Convert one dataset row to a ``RowRecord`` or a drop reason.

    Returns ``(record, reason, stats)``. When ``record`` is None the ``reason``
    is one of the documented drop labels; otherwise ``reason`` is ``"kept"``. The
    ``stats`` dict reports flagged residues and whether the row was truncated.
    """
    stats = {
        "residues_seq_flagged": 0,
        "residues_struct_flagged": 0,
        "truncated": 0,
    }
    aa_seq = row.get("aa_seq") or ""
    struct_seq = row.get("struct_seq") or ""
    if not aa_seq or not struct_seq:
        return None, "empty", stats

    try:
        struct_index, struct_mask, _ = parse_struct_tokens(struct_seq)
    except ValueError:
        return None, "unparseable_struct", stats

    if struct_index.shape[0] != len(aa_seq):
        return None, "length_mismatch", stats

    length_full = len(aa_seq)
    length = min(length_full, int(max_len))
    if length < 1:
        return None, "empty", stats
    if length_full > length:
        stats["truncated"] = 1

    seq_ids, seq_mask, _ = map_sequence_to_ids(aa_seq[:length])
    struct_index = struct_index[:length].copy()
    struct_mask = struct_mask[:length].copy()

    n_noncanonical = int((~seq_mask).sum())
    if n_noncanonical > NONCANONICAL_ROW_DROP_FRACTION * length:
        return None, "noncanonical", stats
    stats["residues_seq_flagged"] = n_noncanonical
    stats["residues_struct_flagged"] = int((~struct_mask).sum())

    source = str(row.get("split") or "unknown")
    accession = str(row.get("pdb_name") or "")
    cluster = str(row.get("cluster") or "")
    split = assign_split(
        accession, cluster, source, val_fraction=val_fraction, salt=salt
    )

    record = RowRecord(
        stable_id=accession or f"row_{abs(hash(struct_seq)) & 0xFFFFFFFF:08x}",
        seq_ids=seq_ids,
        struct_index=struct_index,
        seq_mask=seq_mask,
        struct_mask=struct_mask,
        source=source,
        cluster_id=cluster,
        split=split,
        accession=accession,
        chain=_parse_pdb_chain(source, accession),
        release=release,
        plddt=_to_float(row.get("avg_plddt")),
        resolution=_to_float(row.get("resolution")),
    )
    return record, "kept", stats


# -----------------------------------------------------------------------------
# Lazy dataset iteration
# -----------------------------------------------------------------------------


def iter_dataset_rows(
    revision: str, cache_dir: Path, limit: Optional[int]
) -> Iterator[Dict[str, object]]:
    """Yield raw rows from every dataset config in a stable order, lazily.

    Uses ``datasets`` streaming so the full parquet corpus is not materialised on
    disk. Iteration stops once ``limit`` input rows have been yielded when set.
    """
    datasets = _require_datasets()
    yielded = 0
    for config_name, split_name in DATASET_CONFIG_SPLITS:
        stream = datasets.load_dataset(
            HF_DATASET_REPO,
            name=config_name,
            split=split_name,
            revision=revision,
            streaming=True,
            cache_dir=str(cache_dir),
        )
        for row in stream:
            yield row
            yielded += 1
            if limit is not None and yielded >= limit:
                return


# -----------------------------------------------------------------------------
# Shard content hashing and idempotency ledger
# -----------------------------------------------------------------------------


def shard_content_hash(rows: List[RowRecord], context: str) -> str:
    """Return a stable content hash over a shard's rows and the build context."""
    digest = hashlib.sha256()
    digest.update(context.encode("utf-8"))
    for record in rows:
        digest.update(b"\x00")
        digest.update(record.stable_id.encode("utf-8"))
        digest.update(record.split.encode("utf-8"))
        digest.update(record.source.encode("utf-8"))
        digest.update(record.cluster_id.encode("utf-8"))
        digest.update(np.asarray(record.seq_ids, dtype=np.uint8).tobytes())
        digest.update(
            np.asarray(record.struct_index, dtype=np.uint16).tobytes()
        )
        digest.update(np.asarray(record.seq_mask, dtype=bool).tobytes())
        digest.update(np.asarray(record.struct_mask, dtype=bool).tobytes())
    return digest.hexdigest()


def _load_ledger(path: Path) -> Dict[str, Dict[str, object]]:
    """Load the idempotency ledger mapping shard name to its recorded hash."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


# -----------------------------------------------------------------------------
# Build driver
# -----------------------------------------------------------------------------


def build(args: argparse.Namespace) -> Dict[str, object]:
    """Run the full build and return the manifest payload that was written."""
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Keep the Hugging Face cache inside the output directory so builds are
    # self-contained and do not depend on a writable global cache.
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))

    revision = str(args.hf_revision)
    max_len = int(args.max_len)
    shard_size = int(args.shard_size)
    val_fraction = float(DEFAULT_VAL_FRACTION)
    salt = f"{HF_DATASET_REPO}@{revision}"
    release = f"{HF_DATASET_REPO}@{revision[:12]}"
    tokenizer_revision = f"dplm2_struct_tokens:{release}"
    seq_codec = SequenceBitCodec()
    context = json.dumps(
        {
            "revision": revision,
            "max_len": max_len,
            "shard_size": shard_size,
            "val_fraction": val_fraction,
            "split_policy": SPLIT_POLICY_VERSION,
            "seq_codec_hash": seq_codec.hash(),
            "struct_convention_hash": DEFAULT_STRUCT_CODEC.convention_hash(),
        },
        sort_keys=True,
    )

    ledger_path = out_dir / "build_state.json"
    ledger = _load_ledger(ledger_path)

    counts = {
        "input_rows_read": 0,
        "kept_rows": 0,
        "dropped_empty": 0,
        "dropped_length_mismatch": 0,
        "dropped_unparseable_struct": 0,
        "dropped_noncanonical": 0,
        "truncated_rows": 0,
        "residues_seq_flagged": 0,
        "residues_struct_flagged": 0,
    }
    per_split: Dict[str, int] = {}
    per_source: Dict[str, int] = {}
    drop_reason_to_count = {
        "empty": "dropped_empty",
        "length_mismatch": "dropped_length_mismatch",
        "unparseable_struct": "dropped_unparseable_struct",
        "noncanonical": "dropped_noncanonical",
    }

    shard_names: List[str] = []
    buffer: List[RowRecord] = []
    shard_index = 0

    def flush() -> None:
        nonlocal shard_index
        if not buffer:
            return
        name = f"shard_{shard_index:05d}"
        content_hash = shard_content_hash(buffer, context)
        npz_path = out_dir / f"{name}.npz"
        meta_path = out_dir / f"{name}.meta.json"
        recorded = ledger.get(name)
        already = (
            npz_path.exists()
            and meta_path.exists()
            and isinstance(recorded, dict)
            and recorded.get("content_hash") == content_hash
            and recorded.get("n_rows") == len(buffer)
        )
        if already:
            _log(f"skip {name} ({len(buffer)} rows, hash matches)")
        else:
            write_shard(
                out_dir, name, buffer, tokenizer_revision=tokenizer_revision
            )
            ledger[name] = {
                "content_hash": content_hash,
                "n_rows": len(buffer),
            }
            _atomic_write_json(ledger_path, ledger)
            _log(f"wrote {name} ({len(buffer)} rows)")
        shard_names.append(name)
        shard_index += 1
        buffer.clear()

    for row in iter_dataset_rows(revision, cache_dir, args.limit):
        counts["input_rows_read"] += 1
        record, reason, stats = build_row_record(
            row,
            max_len=max_len,
            release=release,
            val_fraction=val_fraction,
            salt=salt,
        )
        if record is None:
            counts[drop_reason_to_count[reason]] += 1
        else:
            counts["kept_rows"] += 1
            counts["truncated_rows"] += stats["truncated"]
            counts["residues_seq_flagged"] += stats["residues_seq_flagged"]
            counts["residues_struct_flagged"] += stats[
                "residues_struct_flagged"
            ]
            per_split[record.split] = per_split.get(record.split, 0) + 1
            per_source[record.source] = per_source.get(record.source, 0) + 1
            buffer.append(record)
            if len(buffer) >= shard_size:
                flush()
        if counts["input_rows_read"] % LOG_EVERY == 0:
            _log(
                f"read {counts['input_rows_read']} rows, kept {counts['kept_rows']}"
            )
    flush()

    split_manifest = {
        "policy_version": SPLIT_POLICY_VERSION,
        "description": (
            "Rebuilt contamination-aware split. Recent benchmark sources "
            f"(prefixes {list(TEMPORAL_HOLDOUT_SOURCE_PREFIXES)}) are the temporal "
            "test holdout; all other rows are assigned by a stable hash of their "
            "sequence-cluster id (falling back to accession) so whole clusters "
            "stay together. The released train/valid configs are not inherited."
        ),
        "temporal_note": (
            "The release carries no per-entry deposition date, so the temporal "
            "dimension is realised through recognised benchmark sources rather "
            "than a per-row date cutoff."
        ),
        "val_fraction": val_fraction,
        "hash_salt": salt,
        "hash_buckets": SPLIT_HASH_BUCKETS,
        "temporal_holdout_prefixes": list(TEMPORAL_HOLDOUT_SOURCE_PREFIXES),
        "counts_per_split": per_split,
    }
    _atomic_write_json(out_dir / "split_manifest.json", split_manifest)

    extra = {
        "source_dataset": HF_DATASET_REPO,
        "source_url": DATASET_SOURCE_URL,
        "hf_revision": revision,
        "license": DATASET_LICENSE_NOTE,
        "dplm_repo": DPLM_GIT_REPO,
        "tokenizer_revision": tokenizer_revision,
        "structure_tokens_origin": "inherited_from_dataset",
        "max_len": max_len,
        "shard_size": shard_size,
        "limit": args.limit,
        "noncanonical_row_drop_fraction": NONCANONICAL_ROW_DROP_FRACTION,
        "residue_policy": (
            "canonical-20 ids; non-canonical residues masked, rows dropped when "
            f"non-canonical fraction exceeds {NONCANONICAL_ROW_DROP_FRACTION}"
        ),
        "post_filter_counts": counts,
        "counts_per_split": per_split,
        "counts_per_source": per_source,
        "split_policy": split_manifest,
    }
    write_manifest(out_dir, shard_names, seq_codec=seq_codec, extra=extra)

    _log(
        f"done: {counts['kept_rows']} kept of {counts['input_rows_read']} read "
        f"into {len(shard_names)} shards at {out_dir}"
    )
    _log(f"per-split counts: {per_split}")
    _log(f"per-source counts: {per_source}")
    return extra


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command-line arguments for the paired-shard builder."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the M0 paired sequence+structure shards from the pinned "
            "airkingbd/pdb_swissprot dataset into the ProteinMultimodalDataset "
            "shard format."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hf-revision",
        default=DEFAULT_HF_REVISION,
        help="Pinned Hugging Face dataset revision (commit sha) to read.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory to write shards, manifests, and the build ledger into.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help="Maximum stored residues per row; longer chains are leading-cropped.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Number of kept rows per shard.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on input rows read, for quick smoke builds.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments and run the build."""
    args = parse_args(argv)
    build(args)


if __name__ == "__main__":
    main()
    # The datasets streaming reader leaves pyarrow worker threads that race the
    # interpreter finalizer and print a spurious PyGILState_Release message after
    # every output has already been written. Flush and hard-exit to keep the CLI
    # output clean; all shards and manifests are on disk by this point.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
