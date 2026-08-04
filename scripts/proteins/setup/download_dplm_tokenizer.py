#!/usr/bin/env python3
"""Idempotent downloader for the released DPLM-2 structure tokenizer.

This script fetches the frozen DPLM-2 LFQ structure tokenizer from the Hugging
Face repository ``airkingbd/struct_tokenizer`` into a local directory and records
a ``provenance.json`` manifest next to the weights. The manifest pins the source
repository, the resolved HF revision, the per-file SHA-256 hashes, and the bit
convention hash of the local structure codec so a later run can decide whether
the download is already complete.

The download is idempotent. Before fetching anything the script verifies the
recorded hashes against the files on disk. If every recorded file is present and
its hash matches, and the requested revision agrees with the recorded one, the
script does no network work and exits. A missing file, a mismatched hash, or a
different requested revision is treated as a partial or corrupt download and the
snapshot is re-fetched.

Only numpy and torch are needed to import this module and to print ``--help``.
The Hugging Face client is imported lazily inside the function that downloads,
so a missing ``huggingface_hub`` raises a clear, actionable error only when an
actual download is attempted rather than at import time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

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

# Directory components written by the Hugging Face client or version control that
# must never be treated as tokenizer payload or entered into the provenance file.
_IGNORED_DIR_PARTS = frozenset({".cache", ".huggingface", ".git", ".locks"})


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_payload_file(rel_path: Path) -> bool:
    """Return True when a repo-relative path is tokenizer payload worth hashing."""
    if rel_path.name == PROVENANCE_FILENAME:
        return False
    if rel_path.suffix == ".tmp":
        return False
    return not any(part in _IGNORED_DIR_PARTS for part in rel_path.parts)


def compute_file_hashes(out_dir: Path) -> Dict[str, str]:
    """Return a mapping of payload file paths (posix, relative) to SHA-256 hashes."""
    hashes: Dict[str, str] = {}
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(out_dir)
        if not is_payload_file(rel):
            continue
        hashes[rel.as_posix()] = sha256_file(path)
    return hashes


def load_provenance(provenance_path: Path) -> Optional[Dict[str, object]]:
    """Read and parse the provenance manifest, or None when it is absent or unreadable."""
    if not provenance_path.exists():
        return None
    try:
        return json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def verify_complete(
    out_dir: Path, provenance_path: Path, requested_revision: Optional[str]
) -> Tuple[bool, str]:
    """Decide whether the local download matches its recorded provenance.

    Returns a ``(complete, reason)`` pair. The download is complete only when a
    readable provenance manifest exists, its requested revision agrees with the
    one asked for now, and every recorded file is present with a matching hash.
    """
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
                f"requested revision {requested_revision!r} does not match recorded "
                f"revision {recorded_commit!r}",
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
    """Import ``huggingface_hub`` lazily, raising an actionable error when missing."""
    try:
        import huggingface_hub  # noqa: WPS433
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only without the dep
        raise RuntimeError(
            "huggingface_hub is required to download the DPLM structure tokenizer but "
            "it is not installed. Install the DPLM inference dependencies with\n"
            "  scripts/proteins/setup/setup_evaluation.sh dplm\n"
            "or install the package directly with\n"
            "  pip install -r the dplm-inference extra (uv sync --extra dplm-inference)\n"
            "which pins huggingface_hub."
        ) from exc
    return huggingface_hub


def _resolve_commit(
    hf_module, repo_id: str, revision: Optional[str]
) -> Optional[str]:
    """Return the resolved commit SHA for a repo revision, or None if it cannot be read."""
    try:
        info = hf_module.HfApi().repo_info(repo_id=repo_id, revision=revision)
    except Exception:  # pragma: no cover - network/offline dependent
        return None
    return getattr(info, "sha", None)


def atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    """Write JSON to a temporary sibling and atomically replace the target file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def build_provenance_payload(
    repo_id: str,
    resolved_commit: Optional[str],
    requested_revision: Optional[str],
    file_hashes: Dict[str, str],
) -> Dict[str, object]:
    """Assemble the provenance dict from ``TokenizerProvenance`` plus download metadata."""
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
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return payload


def download_tokenizer(
    repo_id: str,
    out_dir: Path,
    revision: Optional[str],
    force: bool,
) -> Dict[str, object]:
    """Fetch the tokenizer snapshot and write a fresh provenance manifest.

    A full ``force_download`` is used when the caller asked for it or when a prior
    provenance manifest exists but no longer matches the files on disk, so that
    partial or corrupt payloads are actually re-fetched rather than trusted from
    the local etag cache.
    """
    provenance_path = out_dir / PROVENANCE_FILENAME
    had_prior_record = provenance_path.exists()
    force_download = bool(force or had_prior_record)

    hf = _import_huggingface_hub()
    out_dir.mkdir(parents=True, exist_ok=True)
    hf.snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(out_dir),
        force_download=force_download,
    )

    file_hashes = compute_file_hashes(out_dir)
    if not file_hashes:
        raise RuntimeError(
            f"No files were downloaded into {out_dir} for repo {repo_id!r}; "
            "the snapshot appears empty."
        )

    resolved_commit = _resolve_commit(hf, repo_id, revision)
    payload = build_provenance_payload(
        repo_id, resolved_commit, revision, file_hashes
    )
    atomic_write_json(provenance_path, payload)

    complete, reason = verify_complete(out_dir, provenance_path, revision)
    if not complete:
        raise RuntimeError(
            f"Download verification failed after fetching into {out_dir}: {reason}"
        )
    return payload


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments for the tokenizer downloader."""
    parser = argparse.ArgumentParser(
        description=(
            "Idempotently download the released DPLM-2 structure tokenizer "
            f"({DPLM_STRUCT_TOKENIZER_HF_REPO}) and record its provenance."
        )
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Local directory for the tokenizer weights (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Hugging Face revision (branch, tag, or commit SHA) to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when the recorded hashes already match on disk.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
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
                f"DPLM structure tokenizer already complete at {out_dir} ({reason}); "
                "nothing to download."
            )
            return 0
        print(f"Fetching DPLM structure tokenizer into {out_dir} ({reason}).")
    else:
        print(
            f"Forcing re-download of DPLM structure tokenizer into {out_dir}."
        )

    payload = download_tokenizer(repo_id, out_dir, args.revision, args.force)
    download_meta = payload["download"]
    print(
        f"Downloaded {download_meta['num_files']} files from {repo_id} "
        f"(revision {download_meta['resolved_commit'] or download_meta['requested_revision']})."
    )
    print(f"Wrote provenance manifest: {provenance_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
