#!/usr/bin/env python3
"""Upload CoBit text or protein model files to Hugging Face.

Use --model protein with --checkpoint for the protein model. The upload
includes the paper config and the checkpoint's frozen entropy tables.
Downloaded datasets, logs, and evaluation outputs are not collected.

Use --dry-run to check the file list and hashes without uploading.
Authenticate with HF_TOKEN or cached Hugging Face credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTROPY_TABLES = (
    "entropy_pdf.pt",
    "entropy_cdf.pt",
    "entropy_sigmas.pt",
    "entropy_edges.pt",
)


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def build_manifest(release_dir: Path) -> list[tuple[Path, str]]:
    """List the files for the original text model release."""
    release_dir = _resolve(release_dir)
    assets = REPO_ROOT / "assets/entropy_tables"
    tokenizer = REPO_ROOT / "datasets/openwebtext_gpt2_trainm100k"
    manifest = [
        (
            release_dir / "cobit_s_lm1b_step001000000_ema.pt",
            "checkpoints/cobit_s_lm1b_1M_ema.pt",
        ),
        (
            release_dir / "cobit_s_owt_step000750000_ema.pt",
            "checkpoints/cobit_s_owt_750k_ema.pt",
        ),
        (
            release_dir / "cobit_m_owt_step000750000_ema.pt",
            "checkpoints/cobit_m_owt_750k_ema.pt",
        ),
    ]
    for suffix in (".json", ".meta.json"):
        name = "tokenizer_gpt2id_bpe16_65536_base1024" + suffix
        manifest.append((tokenizer / name, f"tokenizer/{name}"))
    for dataset in ("lm1b", "owt", "owt_medium"):
        for name in ENTROPY_TABLES:
            manifest.append(
                (
                    assets / dataset / name,
                    f"entropy_tables/{dataset}/{name}",
                )
            )
    return manifest


def build_protein_manifest(
    checkpoint: Path,
    entropy_dir: Path | None = None,
    config: Path = Path("configs/proteins/proteins_400m.py"),
) -> list[tuple[Path, str]]:
    """List only the protein checkpoint, config, and entropy files."""
    checkpoint = _resolve(checkpoint)
    entropy_dir = (
        _resolve(entropy_dir)
        if entropy_dir is not None
        else checkpoint.parent.parent
    )
    return [
        (checkpoint, "checkpoints/best.pt"),
        (_resolve(config), "configs/proteins/proteins_400m.py"),
        (entropy_dir / "entropy_profile.json", "entropy_profile.json"),
        *((entropy_dir / name, name) for name in ENTROPY_TABLES),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protein_manifest(manifest: list[tuple[Path, str]]) -> None:
    """Check that the entropy files belong to the selected checkpoint."""
    files = {destination: source for source, destination in manifest}
    profile = json.loads(
        files["entropy_profile.json"].read_text(encoding="utf-8")
    )
    try:
        checkpoint_hash = profile["checkpoint"]["sha256"]
        table_hashes = profile["results"]["entropy_tables"]
    except (KeyError, TypeError) as exc:
        raise ValueError("The entropy profile is incomplete.") from exc
    if not isinstance(table_hashes, dict) or set(table_hashes) != set(
        ENTROPY_TABLES
    ):
        raise ValueError("The entropy profile must list all four tables.")
    if _sha256(files["checkpoints/best.pt"]) != checkpoint_hash:
        raise ValueError(
            "The entropy profile belongs to another checkpoint. "
            "Run the entropy task for this checkpoint first."
        )
    for name in ENTROPY_TABLES:
        if _sha256(files[name]) != table_hashes[name]:
            raise ValueError(f"Entropy table changed: {name}")


def upload_manifest(
    manifest: list[tuple[Path, str]],
    repo_id: str,
    token: str | None,
    private: bool,
) -> None:
    """Upload the listed files together in one Hub commit."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token or None)
    operations = [
        CommitOperationAdd(path_in_repo=destination, path_or_fileobj=source)
        for source, destination in manifest
    ]
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message="Upload CoBit model files",
        num_threads=1,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--model", choices=("text", "protein"), default="text")
    parser.add_argument(
        "--release-dir",
        default=Path("release"),
        type=Path,
        help="Directory holding the text release checkpoints.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Protein checkpoint to upload."
    )
    parser.add_argument(
        "--entropy-dir",
        type=Path,
        help="Entropy files directory; defaults to the checkpoint run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Protein config; defaults to configs/proteins/proteins_400m.py.",
    )
    parser.add_argument(
        "--token", default=os.environ.get("HF_TOKEN"), help="HF write token."
    )
    parser.add_argument(
        "--private", action="store_true", help="Create a private repository."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Check files without uploading."
    )
    args = parser.parse_args(argv)

    if args.model == "protein":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for --model protein.")
        manifest = build_protein_manifest(
            args.checkpoint,
            args.entropy_dir,
            args.config or Path("configs/proteins/proteins_400m.py"),
        )
    else:
        if any((args.checkpoint, args.entropy_dir, args.config)):
            parser.error(
                "--checkpoint, --entropy-dir, and --config require "
                "--model protein."
            )
        manifest = build_manifest(args.release_dir)

    missing = []
    print("Upload manifest:")
    for source, destination in manifest:
        if source.is_file():
            size = f"{source.stat().st_size / 1e9:.2f} GB"
        else:
            size = "MISSING"
            missing.append(source)
        print(f"  {destination} <- {source} ({size})")
    if missing:
        print(f"{len(missing)} required file(s) missing.", file=sys.stderr)
        if args.model == "protein":
            print(
                "Run scripts/proteins/evaluate.py entropy for this "
                "checkpoint first.",
                file=sys.stderr,
            )
        elif args.dry_run:
            print("Dry run: nothing uploaded.")
            return 0
        return 1
    if args.model == "protein":
        try:
            validate_protein_manifest(manifest)
        except (OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if args.dry_run:
        print("Dry run: all files checked; nothing uploaded.")
        return 0

    upload_manifest(manifest, args.repo_id, args.token, args.private)
    print(f"Done: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
