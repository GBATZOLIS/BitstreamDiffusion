#!/usr/bin/env python3
"""Extract offset-free BiCodec speaker token IDs from a packed cache."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def arguments():
    cascade_root = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[3]
    source_dir = repo_root / "datasets/textaudio"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=source_dir / "cache_textaudio_train.uint32")
    parser.add_argument("--source-meta", type=Path,
                        default=source_dir / "cache_textaudio_train.meta.json")
    parser.add_argument("--output", type=Path,
                        default=cascade_root / "data/speaker_tokens_train.uint16")
    parser.add_argument("--chunk-rows", type=int, default=65536)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    if args.chunk_rows < 1:
        raise ValueError("--chunk-rows must be positive")
    with args.source_meta.open() as handle:
        meta = json.load(handle)

    dtype = np.dtype(meta["dtype"])
    rows = int(meta["n_sequences"])
    width = int(meta["seq_len_tokens"])
    start = int(meta["text_seq_len"])
    length = int(meta["speaker_seq_len"])
    offset = int(meta["speaker_offset"])
    vocab = int(meta["speaker_vocab"])
    expected = rows * width * dtype.itemsize
    if args.source.stat().st_size != expected:
        raise ValueError(
            f"Source size mismatch: expected {expected:,}, "
            f"found {args.source.stat().st_size:,} bytes"
        )
    if vocab > np.iinfo(np.uint16).max + 1:
        raise ValueError("Speaker vocabulary does not fit in uint16")

    output_meta = args.output.with_suffix(".meta.json")
    if not args.overwrite and (args.output.exists() or output_meta.exists()):
        raise FileExistsError("Output exists; pass --overwrite to replace it")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary_meta = output_meta.with_name(output_meta.name + ".tmp")
    for path in (temporary, temporary_meta):
        if path.exists():
            path.unlink()

    source = np.memmap(args.source, dtype=dtype, mode="r", shape=(rows, width))
    output = np.memmap(
        temporary, dtype=np.uint16, mode="w+", shape=(rows, length)
    )
    observed_min, observed_max = vocab, -1
    try:
        for first in range(0, rows, args.chunk_rows):
            last = min(first + args.chunk_rows, rows)
            encoded = np.asarray(source[first:last, start:start + length])
            low, high = int(encoded.min()), int(encoded.max())
            if low < offset or high >= offset + vocab:
                raise ValueError(
                    f"Rows [{first}, {last}) contain encoded values "
                    f"[{low}, {high}], expected [{offset}, {offset + vocab})"
                )
            ids = encoded - offset
            output[first:last] = ids.astype(np.uint16, copy=False)
            observed_min = min(observed_min, int(ids.min()))
            observed_max = max(observed_max, int(ids.max()))
            print(f"Extracted {last:,}/{rows:,}", flush=True)
        output.flush()
    except BaseException:
        del output
        if temporary.exists():
            temporary.unlink()
        raise
    del output
    del source

    result_meta = {
        "cache_format": "speaker_token_ids",
        "dtype": "uint16",
        "n_sequences": rows,
        "seq_len_tokens": length,
        "speaker_vocab": vocab,
        "speaker_offset_removed": offset,
        "observed_token_min": observed_min,
        "observed_token_max": observed_max,
        "source_cache": str(args.source.resolve()),
        "source_meta": str(args.source_meta.resolve()),
        "source_columns": [start, start + length],
    }
    with temporary_meta.open("w") as handle:
        json.dump(result_meta, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, args.output)
    os.replace(temporary_meta, output_meta)
    print(f"Saved raw token IDs: {args.output}")
    print(f"Shape={rows:,}x{length}, dtype=uint16, range=[0,{vocab - 1}]")


if __name__ == "__main__":
    main()
