#!/usr/bin/env python3
"""Download the pinned DPLM structure tokenizer config."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path

TOKENIZER_REVISION = "dd9b86fff06889d74ad00d45377a58a17df2d2c2"
CONFIG_URL = (
    f"https://huggingface.co/airkingbd/struct_tokenizer/resolve/"
    f"{TOKENIZER_REVISION}/config.yaml"
)
CONFIG_SHA256 = (
    "48e98663e61d034c966a52071bdc8cd5e34ac876c66bd691129ce8d5c9aa4ca3"
)
ROOT = Path(__file__).resolve().parents[2]


def download(url, destination, expected, algorithm="sha256"):
    """Download a file and verify its hash before replacing the destination."""
    destination = Path(destination)

    def digest(path):
        value = hashlib.new(algorithm)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()

    if destination.is_file():
        if digest(destination) == expected:
            print(f"Verified {destination}")
            return destination
        raise ValueError(f"Existing file has a different hash: {destination}")
    if not url.startswith("https://"):
        raise ValueError("Downloads require an HTTPS URL")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as source:
        with partial.open("wb") as output:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                output.write(block)
    if digest(partial) != expected:
        raise ValueError(f"Download hash mismatch: {partial}")
    os.replace(partial, destination)
    print(f"Downloaded {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "datasets/dplm_struct_tokenizer/config.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(f"{CONFIG_URL} -> {args.out}")
        return
    download(CONFIG_URL, args.out, CONFIG_SHA256)


if __name__ == "__main__":
    main()
