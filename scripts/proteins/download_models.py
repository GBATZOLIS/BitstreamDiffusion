#!/usr/bin/env python3
"""Download the pinned code and weights used to evaluate proteins."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.proteins.download_config import (
    CONFIG_SHA256,
    CONFIG_URL,
    TOKENIZER_REVISION,
    download,
)

SOURCES = (
    ("dplm", "bytedance/dplm", "8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d"),
    (
        "openfold",
        "aqlaboratory/openfold",
        "103d0370ad9ce07579c20fa9c889a632f9b16618",
    ),
    (
        "ProteinMPNN",
        "dauparas/ProteinMPNN",
        "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57",
    ),
)
ESMFOLD_REVISION = "75a3841ee059df2bf4d56688166c8fb459ddd97a"


def source_revision(destination):
    """Read the revision of a downloaded source tree or a git checkout."""
    marker = destination / ".source.json"
    if marker.is_file():
        return json.loads(marker.read_text())["revision"]
    git = destination / ".git"
    head = git / "HEAD"
    if head.is_file():
        value = head.read_text().strip()
        if not value.startswith("ref:"):
            return value
        ref = value.split(":", 1)[1].strip()
        if (git / ref).is_file():
            return (git / ref).read_text().strip()
        packed = git / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if not line.startswith(("#", "^")) and line.endswith(
                    " " + ref
                ):
                    return line.split()[0]
    return None


def download_source(name, repository, revision):
    """Install one source archive without changing an existing checkout."""
    destination = ROOT / "external" / name
    if destination.exists():
        if source_revision(destination) != revision:
            raise ValueError(
                f"Existing source has a different revision: {destination}"
            )
        print(f"Verified {name} at {revision}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/{repository}/tar.gz/{revision}"
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        scratch = Path(scratch)
        archive = scratch / "source.tar.gz"
        with urllib.request.urlopen(url, timeout=120) as source:
            with archive.open("wb") as output:
                shutil.copyfileobj(source, output)
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(scratch / "source", filter="data")
        contents = list((scratch / "source").iterdir())
        if len(contents) != 1 or not contents[0].is_dir():
            raise ValueError(f"Unexpected archive layout: {repository}")
        (contents[0] / ".source.json").write_text(
            json.dumps({"repository": repository, "revision": revision}) + "\n"
        )
        contents[0].rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sources-only", action="store_true")
    args = parser.parse_args()
    for name, repository, revision in SOURCES:
        if args.dry_run:
            print(f"{repository}@{revision} -> external/{name}")
        else:
            download_source(name, repository, revision)
    if args.sources_only:
        return
    if args.dry_run:
        print(
            f"airkingbd/struct_tokenizer@{TOKENIZER_REVISION} (weights and"
            f" config)"
        )
        print(f"facebook/esmfold_v1@{ESMFOLD_REVISION} -> hf_cache")
        return
    download(
        CONFIG_URL,
        ROOT / "datasets/dplm_struct_tokenizer/config.yaml",
        CONFIG_SHA256,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.proteins.setup.download_dplm_tokenizer",
            "--revision",
            TOKENIZER_REVISION,
        ],
        cwd=ROOT,
        check=True,
    )
    os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
    from huggingface_hub import snapshot_download

    snapshot_download(
        "facebook/esmfold_v1",
        revision=ESMFOLD_REVISION,
        cache_dir=ROOT / "hf_cache/hub",
        allow_patterns=[
            "config.json",
            "vocab.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "pytorch_model.bin",
        ],
        max_workers=1,
    )


if __name__ == "__main__":
    main()
