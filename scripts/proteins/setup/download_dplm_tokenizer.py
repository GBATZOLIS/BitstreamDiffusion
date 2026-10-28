#!/usr/bin/env python3
"""Download and verify the pinned DPLM structure tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.protein_structure_codec import (  # noqa: E402
    DEFAULT_STRUCT_CODEC,
    DPLM_STRUCT_TOKENIZER_HF_REPO,
    TokenizerProvenance,
)

DEFAULT_OUT_DIR = "datasets/dplm_struct_tokenizer"
PROVENANCE_FILENAME = "provenance.json"
SCHEMA_VERSION = 1

# Directory components written by the Hugging Face client or version
# control that must never be treated as tokenizer payload or entered
# into the provenance file.
_IGNORED_DIR_PARTS = frozenset({".cache", ".huggingface", ".git", ".locks"})


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_payload_file(rel_path: Path) -> bool:
    """
    Return True when a repo-relative path is tokenizer payload worth
    hashing.
    """
    if rel_path.name == PROVENANCE_FILENAME:
        return False
    if rel_path.suffix == ".tmp":
        return False
    return not any(part in _IGNORED_DIR_PARTS for part in rel_path.parts)


def compute_file_hashes(out_dir: Path) -> dict[str, str]:
    """
    Return a mapping of payload file paths (posix, relative) to SHA-256
    hashes.
    """
    hashes: dict[str, str] = {}
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(out_dir)
        if not is_payload_file(rel):
            continue
        hashes[rel.as_posix()] = sha256_file(path)
    return hashes


def load_provenance(provenance_path: Path) -> dict[str, object] | None:
    """
    Read and parse the provenance manifest, or None when it is absent or
    unreadable.
    """
    if not provenance_path.exists():
        return None
    try:
        return json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def verify_complete(
    out_dir: Path, provenance_path: Path, requested_revision: str | None
) -> tuple[bool, str]:
    """Decide whether the local download matches its recorded provenance."""
    record = load_provenance(provenance_path)
    if record is None:
        return False, "no readable provenance manifest found"

    file_hashes = record.get("file_hashes") or {}
    if not isinstance(file_hashes, dict) or not file_hashes:
        return False, "provenance manifest records no files"

    if requested_revision is not None:
        recorded_commit = record.get("hf_revision")
        download_meta = record.get("download") or {}
        recorded_request = (
            download_meta.get("requested_revision")
            if isinstance(download_meta, dict)
            else None
        )
        if requested_revision not in (recorded_commit, recorded_request):
            return (
                False,
                (
                    f"requested revision {requested_revision!r} does not "
                    f"match recorded revision {recorded_commit!r}"
                ),
            )

    for rel, expected in sorted(file_hashes.items()):
        path = out_dir / rel
        if not path.is_file():
            return False, f"recorded file is missing: {rel}"
        if sha256_file(path) != expected:
            return False, f"recorded file is corrupt (hash mismatch): {rel}"

    return (
        True,
        f"{len(file_hashes)} recorded files present with matching hashes",
    )


def _import_huggingface_hub():
    """
    Import huggingface_hub lazily, raising an actionable error when
    missing.
    """
    try:
        import huggingface_hub  # noqa: WPS433
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only without the dep
        raise RuntimeError(
            "huggingface_hub is required to download the tokenizer. "
            "Run ./setup.sh train to install it."
        ) from exc
    return huggingface_hub


def _resolve_commit(
    hf_module, repo_id: str, revision: str | None
) -> str | None:
    """
    Return the resolved commit SHA for a repo revision, or None if it
    cannot be read.
    """
    try:
        info = hf_module.HfApi().repo_info(repo_id=repo_id, revision=revision)
    except Exception:  # pragma: no cover - network/offline dependent
        return None
    return getattr(info, "sha", None)


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """
    Write JSON to a temporary sibling and atomically replace the target
    file.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def build_provenance_payload(
    repo_id: str,
    resolved_commit: str | None,
    requested_revision: str | None,
    file_hashes: dict[str, str],
) -> dict[str, object]:
    """
    Assemble the provenance dict from TokenizerProvenance plus download
    metadata.
    """
    provenance = TokenizerProvenance(
        hf_repo=repo_id,
        hf_revision=resolved_commit or requested_revision,
        file_hashes=file_hashes,
    )
    payload = provenance.to_dict()
    payload["codec_convention_hash"] = DEFAULT_STRUCT_CODEC.convention_hash()
    payload["download"] = {
        "schema_version": SCHEMA_VERSION,
        "downloader": "scripts/proteins/setup/download_dplm_tokenizer.py",
        "requested_revision": requested_revision,
        "resolved_commit": resolved_commit,
        "num_files": len(file_hashes),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    return payload


def download_tokenizer(
    repo_id: str,
    out_dir: Path,
    revision: str | None,
    force: bool,
) -> dict[str, object]:
    """Fetch the tokenizer snapshot and write a fresh provenance manifest."""
    provenance_path = out_dir / PROVENANCE_FILENAME
    had_prior_record = provenance_path.exists()
    force_download = bool(force or had_prior_record)

    hf = _import_huggingface_hub()
    out_dir.mkdir(parents=True, exist_ok=True)
    hf.snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(out_dir),
        max_workers=1,
        force_download=force_download,
    )

    file_hashes = compute_file_hashes(out_dir)
    if not file_hashes:
        raise RuntimeError(
            f"No files were downloaded into {out_dir} for repo {repo_id!r}"
            f"; the snapshot appears empty."
        )

    resolved_commit = _resolve_commit(hf, repo_id, revision)
    payload = build_provenance_payload(
        repo_id, resolved_commit, revision, file_hashes
    )
    atomic_write_json(provenance_path, payload)

    complete, reason = verify_complete(out_dir, provenance_path, revision)
    if not complete:
        raise RuntimeError(
            f"Download verification failed after fetching into {out_dir}: "
            f"{reason}"
        )
    return payload


def parse_args(argv: list | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the tokenizer downloader."""
    parser = argparse.ArgumentParser(
        description=(
            f"Idempotently download the released DPLM-2 structure "
            f"tokenizer ({DPLM_STRUCT_TOKENIZER_HF_REPO}) and record its "
            f"provenance."
        )
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=(
            f"Local directory for the tokenizer weights (default: "
            f"{DEFAULT_OUT_DIR})."
        ),
    )
    parser.add_argument(
        "--revision",
        default="dd9b86fff06889d74ad00d45377a58a17df2d2c2",
        help="Hugging Face revision (branch, tag, or commit SHA) to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-download even when the recorded hashes already match on disk."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list | None = None) -> int:
    """Run the idempotent download and print a short status summary."""
    args = parse_args(argv)
    repo_id = DPLM_STRUCT_TOKENIZER_HF_REPO
    out_dir = Path(args.out_dir).expanduser().resolve()
    provenance_path = out_dir / PROVENANCE_FILENAME

    if not args.force:
        complete, reason = verify_complete(
            out_dir, provenance_path, args.revision
        )
        if complete:
            print(
                f"DPLM structure tokenizer already complete at {out_dir} ("
                f"{reason}); nothing to download."
            )
            return 0
        print(f"Fetching DPLM structure tokenizer into {out_dir} ({reason}).")
    else:
        print(
            f"Forcing re-download of DPLM structure tokenizer into {out_dir}."
        )

    payload = download_tokenizer(repo_id, out_dir, args.revision, args.force)
    download_meta = payload["download"]
    revision = (
        download_meta["resolved_commit"] or download_meta["requested_revision"]
    )
    print(
        f"Downloaded {download_meta['num_files']} files from {repo_id} "
        f"(revision {revision})."
    )
    print(f"Wrote provenance manifest: {provenance_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
