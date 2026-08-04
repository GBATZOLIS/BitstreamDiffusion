"""Build a tiny, deterministic synthetic M0 paired fixture.

This is NOT real DPLM data. It writes a handful of paired sequence+structure
rows in the exact immutable shard format ``data.protein_multimodal`` expects
(``write_shard`` + ``write_manifest``) so the full multimodal training path can
be exercised end to end on CPU without downloading the 220k-row corpus or the
frozen tokenizer. The real M0 cache is built by
``scripts/proteins/setup/prepare_dplm_paired.py`` and has the identical schema,
so anything that runs against this fixture runs against real M0 unchanged.

The rows are drawn from a fixed RNG, span two sources and two length buckets so
the source/length-balanced sampler has real buckets to draw, and are split into
train/val. Sequence and structure are fully paired (``seq_mask`` and
``struct_mask`` all True). Structure ids are stored ``uint16`` in the released
DPLM MSB-first convention pinned by ``DEFAULT_STRUCT_CODEC``; the manifest pins
that convention hash so the dataset refuses to load under a different bit order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.protein_multimodal import RowRecord, write_manifest, write_shard
from data.protein_structure_codec import (
    DEFAULT_STRUCT_CODEC,
    STRUCT_CODEBOOK_SIZE,
)

SEQ_VOCAB_SIZE = 20  # canonical amino acids


def build_fixture(
    out_dir: str | Path,
    *,
    lengths: Sequence[int] = (40, 48),
    sources: Sequence[str] = ("pdb", "afdb_swissprot"),
    rows_per_bucket_train: int = 8,
    rows_per_bucket_val: int = 2,
    seed: int = 0,
    shard_name: str = "m0_fixture",
) -> Dict[str, object]:
    """Write a deterministic paired fixture; return the manifest dict."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)

    rows: List[RowRecord] = []
    counter = 0
    for split, n_rows in (
        ("train", rows_per_bucket_train),
        ("val", rows_per_bucket_val),
    ):
        for src in sources:
            for length in lengths:
                for _ in range(n_rows):
                    seq_ids = rng.integers(
                        0, SEQ_VOCAB_SIZE, size=length, dtype=np.uint8
                    )
                    struct_index = rng.integers(
                        0, STRUCT_CODEBOOK_SIZE, size=length, dtype=np.uint16
                    )
                    rows.append(
                        RowRecord(
                            stable_id=f"{split}_{src}_{length}_{counter:05d}",
                            seq_ids=seq_ids,
                            struct_index=struct_index,
                            seq_mask=np.ones(length, dtype=bool),
                            struct_mask=np.ones(length, dtype=bool),
                            source=src,
                            cluster_id=f"cl_{counter % 7}",
                            split=split,
                            accession=f"ACC{counter:05d}",
                            chain="A",
                            release="fixture-2020-01-01",
                            plddt=90.0,
                            resolution=2.0,
                        )
                    )
                    counter += 1

    write_shard(
        out_dir,
        shard_name,
        rows,
        tokenizer_revision="synthetic-fixture-v1",
    )
    write_manifest(
        out_dir,
        [shard_name],
        codec=DEFAULT_STRUCT_CODEC,
        extra={
            "fixture": True,
            "note": "synthetic tiny M0 fixture; not real DPLM data",
            "n_rows": len(rows),
        },
    )
    manifest = {"shards": [shard_name], "n_rows": len(rows)}
    print(
        f"[build_m0_fixture] wrote {len(rows)} rows to {out_dir} "
        f"(convention_hash={DEFAULT_STRUCT_CODEC.convention_hash()})"
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="datasets/dplm_paired_m0_fixture")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build_fixture(args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
