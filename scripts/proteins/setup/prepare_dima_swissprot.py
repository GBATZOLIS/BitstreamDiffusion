#!/usr/bin/env python3
"""Freeze the exact Swiss-Prot snapshot/protocol used by the released DiMA code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from huggingface_hub import snapshot_download

from datasets import load_dataset

HF_REPO_ID = "bayes-group-diffusion/swissprot"
HF_REVISION = "bf4b2f131664fe87ef5e0fc9e53f01f0d030dcab"
DIMA_REPO = "https://github.com/MeshchaninovViacheslav/DiMA"
DIMA_COMMIT = "18f4a2e67988efe1cc5593fcaec1ece8f09fdac0"
MAX_SEQUENCE_LEN = 254
CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def write_indexed_sequences(
    sequences: Iterable[str], output_dir: Path, split: str
) -> Tuple[Dict, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_path = output_dir / f"{split}.sequences.bin"
    offsets_path = output_dir / f"{split}.offsets.npy"
    lengths_path = output_dir / f"{split}.lengths.npy"

    tmp_binary = binary_path.with_suffix(binary_path.suffix + ".tmp")
    offsets = [0]
    lengths = []
    raw_lengths = []
    noncanonical_sequences = 0
    empty_sequences = 0

    with tmp_binary.open("wb") as handle:
        for sequence in sequences:
            if not isinstance(sequence, str):
                raise TypeError(
                    f"Expected sequence string, got {type(sequence).__name__}"
                )
            try:
                raw = sequence.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ValueError("Swiss-Prot sequence is not ASCII") from exc
            raw_lengths.append(len(raw))
            effective = raw[:MAX_SEQUENCE_LEN]
            if not effective:
                empty_sequences += 1
            if any(chr(code) not in CANONICAL_AA for code in effective):
                noncanonical_sequences += 1
            handle.write(effective)
            offsets.append(offsets[-1] + len(effective))
            lengths.append(len(effective))
    os.replace(tmp_binary, binary_path)

    offsets_array = np.asarray(offsets, dtype=np.uint64)
    lengths_array = np.asarray(lengths, dtype=np.uint16)
    raw_lengths_array = np.asarray(raw_lengths, dtype=np.int64)
    np.save(offsets_path, offsets_array, allow_pickle=False)
    np.save(lengths_path, lengths_array, allow_pickle=False)

    stats = {
        "num_sequences": int(lengths_array.size),
        "raw_length_min": int(raw_lengths_array.min()),
        "raw_length_max": int(raw_lengths_array.max()),
        "raw_length_mean": float(raw_lengths_array.mean()),
        "effective_length_min": int(lengths_array.min()),
        "effective_length_max": int(lengths_array.max()),
        "effective_length_mean": float(lengths_array.mean()),
        "num_truncated_to_254": int(
            np.sum(raw_lengths_array > MAX_SEQUENCE_LEN)
        ),
        "num_empty": int(empty_sequences),
        "num_noncanonical_effective": int(noncanonical_sequences),
    }
    return stats, raw_lengths_array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/swissprot_dima_bf4b2f13"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    snapshot_dir = output_dir / "upstream_snapshot"
    protocol_dir = output_dir / "protocol"
    manifest_path = output_dir / "frozen_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists() and not args.force:
        print(f"Frozen snapshot already exists: {manifest_path}")
        return

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=HF_REVISION,
        local_dir=snapshot_dir,
        allow_patterns=["README.md", ".gitattributes", "data/*.parquet"],
    )

    parquet_files = {
        "train": snapshot_dir / "data" / "train-00000-of-00001.parquet",
        "test": snapshot_dir / "data" / "test-00000-of-00001.parquet",
    }
    for path in parquet_files.values():
        if not path.exists():
            raise FileNotFoundError(path)

    dataset = load_dataset(
        "parquet",
        data_files={split: str(path) for split, path in parquet_files.items()},
        cache_dir=str(output_dir / ".cache"),
    )
    split_stats = {}
    raw_train_lengths = None
    for split in ("train", "test"):
        if "sequence" not in dataset[split].column_names:
            raise ValueError(f"Missing sequence column in {split}")
        stats, raw_lengths = write_indexed_sequences(
            dataset[split]["sequence"], protocol_dir, split
        )
        split_stats[split] = stats
        if split == "train":
            raw_train_lengths = raw_lengths

    assert raw_train_lengths is not None
    # Exact DiMA behavior: prepare_length_distribution counts raw lengths;
    # LengthSampler then slices [0:255] and renormalizes instead of assigning
    # overlength training examples to the 254 bin.
    raw_counts = np.bincount(raw_train_lengths)
    generation_counts = np.zeros(MAX_SEQUENCE_LEN + 1, dtype=np.int64)
    copied = min(generation_counts.size, raw_counts.size)
    generation_counts[:copied] = raw_counts[:copied]
    generation_probs = generation_counts.astype(np.float64)
    generation_probs /= generation_probs.sum()
    np.save(
        protocol_dir / "train_raw_length_counts_through_254.npy",
        generation_counts,
    )
    np.save(
        protocol_dir / "generation_length_probabilities.npy", generation_probs
    )

    files = {}
    for path in sorted(output_dir.rglob("*")):
        if (
            path.is_file()
            and path != manifest_path
            and ".cache" not in path.parts
        ):
            files[str(path.relative_to(output_dir))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream": {
            "huggingface_repo": HF_REPO_ID,
            "huggingface_revision": HF_REVISION,
            "dima_repo": DIMA_REPO,
            "dima_commit": DIMA_COMMIT,
        },
        "protocol": {
            "max_sequence_len": MAX_SEQUENCE_LEN,
            "training_rows": "all upstream rows, in upstream order",
            "training_transform": "ASCII sequence[:254] (matches tokenizer truncation)",
            "padding_loss": "excluded",
            "generation_length_distribution": (
                "raw train length histogram restricted to lengths <=254, then renormalized"
            ),
            "split_policy": "use upstream train and test splits unchanged",
        },
        "splits": split_stats,
        "files": files,
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest["splits"], indent=2, sort_keys=True))
    print(f"Frozen manifest: {manifest_path}")


if __name__ == "__main__":
    main()
