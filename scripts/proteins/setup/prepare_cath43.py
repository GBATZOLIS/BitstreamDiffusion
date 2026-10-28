#!/usr/bin/env python
"""Download CATH validation and test backbones and retain resolved segments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

BASE_URL = (
    "https://dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206"
)
CHAIN_SET = "chain_set.jsonl"
SPLITS = "splits.json"

# Backbone atom order the evaluation stack expects.
ATOM_ORDER = ("N", "CA", "C", "O")

# Must match evaluation.evaluation_drivers.proteins.metrics.CANONICAL_AA,
# because that is
# the alphabet inverse_folding.py maps seq_ids back through.
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: i for i, aa in enumerate(CANONICAL_AA)}


def download(url: str, dest: Path, *, force: bool = False) -> Path:
    """Fetch a URL to dest, skipping the download when the file is complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        try:
            with urllib.request.urlopen(url) as r:  # noqa: S310
                remote = int(r.headers.get("Content-Length", 0))
            if remote and dest.stat().st_size == remote:
                print(f"  have {dest.name} ({dest.stat().st_size:,} bytes)")
                return dest
        except OSError:
            if dest.stat().st_size > 0:
                print(f"  have {dest.name} (size not verified, offline)")
                return dest
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:  # noqa: S310
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done / total
                print(
                    (
                        f"\r    {done / 1000000.0:.0f}/"
                        f"{total / 1000000.0:.0f} MB ({pct:.0f}%)"
                    ),
                    end="",
                    flush=True,
                )
    print()
    tmp.replace(dest)
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_record(record: dict):
    """Return (name, sequence, coords[L,4,3] float32) or None when unusable."""
    name = record.get("name") or record.get("chain_name") or ""
    seq = record.get("seq") or record.get("sequence") or ""
    coords = record.get("coords")
    if not seq or not isinstance(coords, dict):
        return None

    try:
        tracks = [np.asarray(coords[a], dtype=np.float64) for a in ATOM_ORDER]
    except KeyError:
        return None

    length = len(seq)
    if any(t.shape != (length, 3) for t in tracks):
        return None

    arr = np.stack(tracks, axis=1).astype(np.float32)  # [L, 4, 3]
    return name, seq, arr


def longest_resolved_run(arr: np.ndarray) -> tuple[int, int]:
    """Return [start, end) of the longest run of fully-resolved residues."""
    good = np.isfinite(arr).all(axis=(1, 2))  # [L]
    best_start = best_end = 0
    run_start = None
    for i, ok in enumerate(good):
        if ok and run_start is None:
            run_start = i
        elif not ok and run_start is not None:
            if i - run_start > best_end - best_start:
                best_start, best_end = run_start, i
            run_start = None
    if run_start is not None and len(good) - run_start > best_end - best_start:
        best_start, best_end = run_start, len(good)
    return best_start, best_end


def build_split(
    records: dict[str, tuple[str, np.ndarray]],
    names: list[str],
    out_path: Path,
    *,
    min_len: int,
    max_len: int,
    gap_policy: str = "longest-run",
    min_retained_fraction: float = 0.5,
) -> dict:
    """Pack one split into a single npz and return its statistics."""
    kept_names: list[str] = []
    seq_id_blocks: list[np.ndarray] = []
    coord_blocks: list[np.ndarray] = []
    offsets = [0]

    stats = {
        "requested": len(names),
        "missing": 0,
        "nonfinite": 0,
        "too_short": 0,
        "too_long": 0,
        "noncanonical": 0,
        "too_fragmented": 0,
        "kept": 0,
    }
    retained_fractions: list[float] = []

    for name in names:
        entry = records.get(name)
        if entry is None:
            stats["missing"] += 1
            continue
        seq, arr = entry
        full_length = arr.shape[0]

        if not np.isfinite(arr).all():
            if gap_policy == "strict":
                # Never imputed: an invented atom would change both the
                # tokenized structure and the RMSD it is scored against,
                # with no record that it was invented.
                stats["nonfinite"] += 1
                continue
            start, end = longest_resolved_run(arr)
            if end - start < min_len:
                stats["nonfinite"] += 1
                continue
            arr = arr[start:end]
            seq = seq[start:end]

        retained = arr.shape[0] / max(1, full_length)
        if retained < min_retained_fraction:
            # A chain reduced to a small fraction of itself is arguably
            # a different protein, and ProteinMPNN reports on full
            # chains with in-place masking. Dropping these keeps the
            # comparison honest.
            stats["too_fragmented"] += 1
            continue
        retained_fractions.append(retained)

        if arr.shape[0] < min_len:
            stats["too_short"] += 1
            continue
        if arr.shape[0] > max_len:
            stats["too_long"] += 1
            continue
        # inverse_folding.py maps seq_ids through the 20-letter
        # canonical alphabet and raises on anything outside it, so a
        # chain carrying X, U, B or Z cannot be scored and is dropped
        # here instead.
        if any(c not in AA_TO_ID for c in seq):
            stats["noncanonical"] += 1
            continue

        kept_names.append(name)
        seq_id_blocks.append(
            np.asarray([AA_TO_ID[c] for c in seq], dtype=np.int8)
        )
        coord_blocks.append(arr)
        offsets.append(offsets[-1] + arr.shape[0])
        stats["kept"] += 1

    if not coord_blocks:
        raise SystemExit(f"no usable chains for {out_path.name}")

    coords = np.concatenate(coord_blocks, axis=0)
    seq_ids = np.concatenate(seq_id_blocks, axis=0)
    lengths = np.diff(np.asarray(offsets, dtype=np.int64))

    # seq_ids is a FLAT per-residue array, not one string per chain,
    # because load_backbone_records slices only a fixed set of
    # per-residue keys (seq_ids, aatype, struct_index, ...) by the
    # offsets and passes every other key through whole. A per-chain
    # array of strings would reach every record unsliced and fail the
    # length check.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        coords=coords,
        offsets=np.asarray(offsets, dtype=np.int64),
        seq_ids=seq_ids,
        lengths=lengths,
    )
    # Chain names live beside the npz so the array file stays free of
    # object arrays, and so a result table can name the chain it scored.
    with out_path.with_suffix(".names.json").open("w") as f:
        json.dump(kept_names, f)
    stats["residues"] = int(coords.shape[0])
    stats["median_length"] = int(np.median(lengths))
    stats["file_bytes"] = out_path.stat().st_size
    stats["gap_policy"] = gap_policy
    if retained_fractions:
        rf = np.asarray(retained_fractions)
        # How much of each original CATH chain survived. Report it: our
        # recovery is measured on these segments, while a protocol that
        # masks gaps in place measures on the whole chain.
        stats["retained_fraction_mean"] = round(float(rf.mean()), 4)
        stats["retained_fraction_median"] = round(float(np.median(rf)), 4)
        stats["retained_fraction_min"] = round(float(rf.min()), 4)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/cath43")
    ap.add_argument("--cache", default="datasets/_downloads/cath43")
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="Chains longer than this are dropped. Keep in step "
        "with cfg.data.max_len so evaluation matches training.",
    )
    ap.add_argument(
        "--include-train",
        action="store_true",
        help="Also pack the 16,699 train chains. Needed by "
        "build_decontaminated_splits.py; not needed to "
        "evaluate.",
    )
    ap.add_argument(
        "--gap-policy",
        choices=("longest-run", "strict"),
        default="longest-run",
        help="How to handle residues with missing density. "
        "longest-run keeps the longest fully-resolved "
        "segment of each chain (1,880/1,882 test chains, "
        "median 94%% retained). strict drops any chain "
        "with a gap (keeps only 26%%).",
    )
    ap.add_argument(
        "--min-retained-fraction",
        type=float,
        default=0.5,
        help="Drop a chain whose longest resolved run is below "
        "this fraction of its full length. ProteinMPNN "
        "scores full chains with in-place masking, so a "
        "heavily fragmented chain is not comparable.",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache)
    out = Path(args.out)

    print("CATH 4.3 topology split")
    print("=" * 60)
    chain_path = download(
        f"{BASE_URL}/{CHAIN_SET}", cache / CHAIN_SET, force=args.force
    )
    splits_path = download(
        f"{BASE_URL}/{SPLITS}", cache / SPLITS, force=args.force
    )

    with splits_path.open() as f:
        splits = json.load(f)
    print(
        "  splits: "
        + ", ".join(f"{k}={len(v)}" for k, v in sorted(splits.items()))
    )

    wanted = set()
    split_names = ["test", "validation"]
    if args.include_train:
        split_names.append("train")
    for s in split_names:
        wanted.update(splits.get(s, []))

    print(f"  parsing {CHAIN_SET} for {len(wanted):,} chains ...")
    records: dict[str, tuple[str, np.ndarray]] = {}
    bad = 0
    with chain_path.open() as f:
        for i, line in enumerate(f):
            if i % 2000 == 0:
                print(
                    f"\r    line {i:,}  kept {len(records):,}",
                    end="",
                    flush=True,
                )
            line = line.strip()
            if not line:
                continue
            try:
                # json accepts the NaN literals this file uses.
                record = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            parsed = parse_record(record)
            if parsed is None:
                bad += 1
                continue
            name, seq, arr = parsed
            if name in wanted:
                records[name] = (seq, arr)
    print(
        f"\r    parsed; {len(records):,} of {len(wanted):,} wanted chains "
        f"found, {bad:,} unparsable lines"
    )

    manifest = {
        "source": BASE_URL,
        "chain_set_sha256": sha256(chain_path),
        "splits_sha256": sha256(splits_path),
        ("split_kind"): (
            "CATH 4.3 topology split (ProteinMPNN / ESM-IF protocol)"
        ),
        "atom_order": list(ATOM_ORDER),
        "min_len": args.min_len,
        "max_len": args.max_len,
        "gap_policy": args.gap_policy,
        "min_retained_fraction": args.min_retained_fraction,
        "dropped_policy": (
            "residues with missing density are NaN in the source; the "
            "longest fully-resolved run of each chain is kept, never "
            "imputed and never spliced across a gap"
        ),
        "splits": {},
    }

    for split in split_names:
        names = splits.get(split, [])
        print(f"\n  packing {split} ({len(names):,} chains)")
        split_path = out / f"{split}.npz"
        stats = build_split(
            records,
            names,
            split_path,
            min_len=args.min_len,
            max_len=args.max_len,
            gap_policy=args.gap_policy,
            min_retained_fraction=args.min_retained_fraction,
        )
        stats["sha256"] = sha256(split_path)
        manifest["splits"][split] = stats
        print(
            f"    kept {stats['kept']:,} / {stats['requested']:,}  "
            f"residues {stats['residues']:,}  median length "
            f"{stats['median_length']}"
        )
        print(
            f"    dropped: missing {stats['missing']}, unusable-gaps "
            f"{stats['nonfinite']}, too fragmented "
            f"{stats['too_fragmented']}, too short {stats['too_short']}, "
            f"too long {stats['too_long']}"
        )

    out.mkdir(parents=True, exist_ok=True)
    with (out / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nwrote {out}/manifest.json")
    print(
        "The evaluation driver encodes these backbones with the DPLM "
        "tokenizer."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
