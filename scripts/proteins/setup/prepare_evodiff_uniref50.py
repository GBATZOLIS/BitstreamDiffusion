#!/usr/bin/env python3
"""Download, verify, extract, and freeze the EvoDiff UniRef50 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from data.uniref50 import (
    DPLM_COMMIT,
    EVODIFF_COMMIT,
    EVODIFF_UNIREF50_ALPHABET,
    EVODIFF_UNIREF50_ARCHIVE_MD5,
    EVODIFF_UNIREF50_URL,
)

REQUIRED_FILES = ("consensus.fasta", "lengths_and_offsets.npz", "splits.json")


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download_resumable(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:  # noqa: S310 - pinned HTTPS URL
        append = offset > 0 and response.status == 206
        mode = "ab" if append else "wb"
        if offset and not append:
            offset = 0
        total = response.headers.get("Content-Length")
        total_bytes = offset + int(total) if total is not None else None
        received = offset
        with partial.open(mode) as output:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                received += len(chunk)
                if total_bytes:
                    print(
                        f"\r[uniref50] {received / 1e9:.2f}/"
                        f"{total_bytes / 1e9:.2f} GB",
                        end="",
                        flush=True,
                    )
    print()
    os.replace(partial, destination)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if (
                destination_resolved not in target.parents
                and target != destination_resolved
            ):
                raise ValueError(f"Unsafe path in archive: {member.name}")
        bundle.extractall(destination)  # noqa: S202 - paths validated above


def find_data_dir(root: Path) -> Path:
    candidates = []
    for metadata in root.rglob("lengths_and_offsets.npz"):
        parent = metadata.parent
        if all((parent / name).exists() for name in REQUIRED_FILES):
            candidates.append(parent)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one extracted UniRef50 directory under {root}, "
            f"found {candidates}"
        )
    return candidates[0]


def write_manifest(root: Path, archive: Path, data_dir: Path) -> Path:
    with (data_dir / "splits.json").open("r", encoding="utf-8") as handle:
        splits = json.load(handle)
    required_splits = {"train", "valid", "test", "rtest"}
    missing = required_splits.difference(splits)
    if missing:
        raise ValueError(f"EvoDiff splits.json is missing {sorted(missing)}")

    metadata = np.load(
        data_dir / "lengths_and_offsets.npz", allow_pickle=False
    )
    n_sequences = int(len(metadata["ells"]))
    if len(metadata["seq_offsets"]) != n_sequences:
        # Upstream offsets identify the beginning of each line, not boundaries.
        raise ValueError("lengths_and_offsets.npz has inconsistent arrays")

    relative_data_dir = os.path.relpath(data_dir, root)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": "March 2020 UniRef50 with released EvoDiff splits",
            "zenodo_record": "6564798",
            "doi": "10.5281/zenodo.6564798",
            "url": EVODIFF_UNIREF50_URL,
            "archive": archive.name,
            "archive_bytes": archive.stat().st_size,
            "archive_md5": digest(archive, "md5"),
        },
        "upstream": {
            "evodiff_repo": "https://github.com/microsoft/evodiff",
            "evodiff_commit": EVODIFF_COMMIT,
            "dplm_repo": "https://github.com/bytedance/dplm",
            "dplm_commit": DPLM_COMMIT,
        },
        "dataset": {
            "data_dir": relative_data_dir,
            "num_sequences": n_sequences,
            "alphabet": EVODIFF_UNIREF50_ALPHABET,
            "split_map": {"train": "train", "val": "valid", "test": "rtest"},
            "upstream_split_counts": {
                name: len(splits[name]) for name in sorted(required_splits)
            },
            "crop_policy": {
                "train": "uniform random contiguous crop when sequence > max_len",
                "val": "deterministic leading crop when sequence > max_len",
                "test": "deterministic leading crop when sequence > max_len",
            },
        },
        "metadata_files": {
            name: {
                "bytes": (data_dir / name).stat().st_size,
                "sha256": digest(data_dir / name),
            }
            for name in ("lengths_and_offsets.npz", "splits.json")
        },
    }
    if payload["source"]["archive_md5"] != EVODIFF_UNIREF50_ARCHIVE_MD5:
        raise ValueError(
            f"Archive MD5 mismatch: got {payload['source']['archive_md5']}, "
            f"expected {EVODIFF_UNIREF50_ARCHIVE_MD5}"
        )
    manifest = root / "frozen_manifest.json"
    temporary = manifest.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("datasets/uniref50_evodiff_2020")
    )
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    archive = args.archive or root / "uniref50.tar.gz"
    if not archive.exists():
        if not args.download:
            raise FileNotFoundError(
                f"Missing {archive}. Re-run with --download or pass --archive PATH."
            )
        download_resumable(EVODIFF_UNIREF50_URL, archive)

    archive_md5 = digest(archive, "md5")
    if archive_md5 != EVODIFF_UNIREF50_ARCHIVE_MD5:
        raise ValueError(
            f"Archive MD5 mismatch: got {archive_md5}, "
            f"expected {EVODIFF_UNIREF50_ARCHIVE_MD5}"
        )
    print(f"[uniref50] archive verified: {archive_md5}")

    try:
        data_dir = find_data_dir(root)
    except RuntimeError:
        if args.verify_only:
            raise
        safe_extract(archive, root)
        data_dir = find_data_dir(root)
    manifest = write_manifest(root, archive, data_dir)
    print(f"[uniref50] data: {data_dir}")
    print(f"[uniref50] frozen manifest: {manifest}")


if __name__ == "__main__":
    main()
