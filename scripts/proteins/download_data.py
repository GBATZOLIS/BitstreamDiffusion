#!/usr/bin/env python3
"""Download and prepare the protein training and evaluation data."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.proteins.download_config import download

METADATA_URL = (
    "https://zenodo.org/api/records/15424801/files/"
    "dplm2_metadata.tar.gz/content"
)
METADATA_MD5 = "7e1f1a584d6fe730e84ad374ed747f08"


def commands(component, python=sys.executable):
    """Return the preparation commands for one data component."""
    prefix = [python, "-m", "scripts.proteins.setup."]
    tasks = {
        "paired": [
            ("prepare_dplm_paired", ["--out-dir", "datasets/dplm_paired_m0"]),
        ],
        "replay": [
            ("prepare_evodiff_uniref50", ["--download"]),
            (
                "prepare_uniref50_replay",
                [
                    "--out-dir",
                    "datasets/uniref50_replay_v2",
                    "--max-rows",
                    "400000",
                    "--seed",
                    "0",
                    "--crop",
                    "random",
                    "--min-len",
                    "40",
                    "--max-len",
                    "512",
                ],
            ),
        ],
        "cath": [("prepare_cath43", [])],
        "cameo": [("prepare_cameo", [])],
    }
    return [
        prefix[:2] + [prefix[2] + name] + args
        for name, args in tasks[component]
    ]


def prepare_cameo_source():
    """Extract the verified DPLM metadata used by the CAMEO benchmark."""
    archive = download(
        METADATA_URL,
        ROOT / "datasets/_downloads/dplm2_metadata.tar.gz",
        METADATA_MD5,
        "md5",
    )
    target = ROOT / "datasets/dplm2_metadata"
    marker = target / ".complete"
    if marker.is_file() and marker.read_text().strip() == METADATA_MD5:
        return
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(target, filter="data")
    marker.write_text(METADATA_MD5 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "component",
        choices=["train", "eval", "all", "paired", "replay", "cath", "cameo"],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    groups = {
        "train": ["paired", "replay"],
        "eval": ["cath", "cameo"],
        "all": ["paired", "replay", "cath", "cameo"],
    }
    for component in groups.get(args.component, [args.component]):
        if component == "cameo":
            if args.dry_run:
                print(f"Download and verify {METADATA_URL}")
            else:
                prepare_cameo_source()
        for command in commands(component):
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
