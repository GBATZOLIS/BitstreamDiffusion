#!/usr/bin/env python3
"""Prepare the optional M2 ESMAtlas high-confidence representative structures.

This stages the raw predicted-coordinate shards for the ESMAtlas high-quality,
foldseek-clustered representative set (plan section 7.4, tier M2). It is a gated
scale and source-diversity ablation, not part of the main paper corpus: nothing is
downloaded unless the caller passes --enable, so setup.sh never pulls terabytes by
default.

The script mirrors the other preparation scripts under scripts/proteins: it works
from a pinned shard index (the frozen accession list), downloads each shard
resumably, records provenance and per-shard checksums in a frozen manifest, is
idempotent across re-runs, and prints a disk estimate before doing any work. These
structures arrive as raw coordinates only; they must later be passed through the
frozen DPLM structure tokenizer by scripts/proteins/setup/tokenize_structures.py to
produce the uint16 token cache used for training.

Helix-bias caveat: ESMAtlas coordinates are model predictions, and training on
predicted structures at this scale has been associated with an alpha-helix
secondary-structure bias. Treat this corpus as an ablation and audit
secondary-structure composition and topology diversity before promoting it. See
HELIX_BIAS_CAVEAT below.

The module imports with only numpy and torch present. Downloads use the standard
library. The optional backbone-extraction and .zst decompression paths import their
heavy dependencies lazily and raise a clear, actionable error when a tool is
missing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

# Pinned provenance for the ESMAtlas high-confidence representative set. The exact
# bulk-download layout and shard list are documented in the ESM atlas README; the
# frozen shard index (accession list) pins the precise files this script fetches.
ESM_REPO = "https://github.com/facebookresearch/esm"
ESM_COMMIT = "2b369911bb5b4b0dda914521b9475cad1656b2ac"
ESMATLAS_README = (
    "https://github.com/facebookresearch/esm/blob/main/scripts/atlas/README.md"
)
ESMATLAS_HOST = "https://esmatlas.com"
ESMATLAS_RELEASE = "v2023_02"
ESMATLAS_SUBSET = "highquality_clust30"
ESMATLAS_LICENSE = "CC-BY-4.0"
ESMATLAS_LICENSE_URL = "https://esmatlas.com/about#license"

# High-quality definition and clustering used upstream, recorded for reproducibility.
CLUSTER_SEQUENCE_IDENTITY = 0.30
MIN_MEAN_PLDDT = 0.70
MIN_PTM = 0.70

# Scale target for the clustered representative set (Yeti-style), used for the disk
# estimate when precise shard sizes are not declared in the index.
ROW_ESTIMATE = 2_085_441
AVG_RESIDUES_PER_CHAIN = 200
BACKBONE_ATOMS = 4  # N, CA, C, O
BYTES_PER_FLOAT16 = 2
BYTES_PER_UINT16 = 2
EST_BYTES_PER_SHARD_ARCHIVE = 512 * 1024 * 1024

BACKBONE_ATOM_ORDER = ("N", "CA", "C", "O")

HELIX_BIAS_CAVEAT = (
    "ESMAtlas structures are predicted, not experimental. Prior work reported an "
    "alpha-helix secondary-structure bias associated with training on predicted "
    "structures at this scale. This corpus is a gated scale ablation only: include "
    "it only if a source-balanced probe shows it improves topology diversity and "
    "co-generation on validation without worsening the helix bias. Audit "
    "secondary-structure fractions before and after adding it, and keep experimental "
    "PDB and curated Swiss-Prot oversampled relative to their raw counts."
)


def sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Dict) -> None:
    """Write JSON to a temporary file and atomically replace the destination."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def format_bytes(num_bytes: Optional[float]) -> str:
    """Format a byte count using binary units, or return 'unknown' for None."""
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024.0 or unit == "PiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def _require(module_name: str, install_hint: str):
    """Import a heavy dependency lazily and raise an actionable error if missing."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"'{module_name}' is required for this step but is not installed. "
            f"{install_hint}"
        ) from exc


def load_shard_index(path: Path) -> List[Dict]:
    """Load the frozen shard index that pins the representative files to download.

    Accepts JSON (a list of records) or a tab-separated file with a
    'filename<TAB>url' layout and optional 'sha256' and 'bytes' columns. Each record
    must provide a filename and a URL; expected checksums and sizes are optional but
    recommended so that downloads can be verified rather than merely completed.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, list):
            raise ValueError(f"Shard index {path} must contain a JSON list")
        records = [dict(item) for item in raw]
    else:
        records = []
        header: Optional[List[str]] = None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                fields = line.split("\t")
                if header is None and fields[0].strip().lower() == "filename":
                    header = [name.strip().lower() for name in fields]
                    continue
                columns = header or ["filename", "url", "sha256", "bytes"]
                record: Dict[str, object] = {}
                for name, value in zip(columns, fields):
                    value = value.strip()
                    if value:
                        record[name] = value
                records.append(record)
    cleaned: List[Dict] = []
    for record in records:
        filename = record.get("filename")
        url = record.get("url")
        if not filename or not url:
            raise ValueError(
                f"Shard index {path} has a record without a filename and url: {record}"
            )
        entry: Dict[str, object] = {"filename": str(filename), "url": str(url)}
        if record.get("sha256"):
            entry["sha256"] = str(record["sha256"]).lower()
        if record.get("bytes"):
            entry["bytes"] = int(record["bytes"])
        cleaned.append(entry)
    if not cleaned:
        raise ValueError(f"Shard index {path} is empty")
    return cleaned


def disk_estimate(records: Optional[List[Dict]]) -> Dict:
    """Estimate download and cache footprint for the representative set.

    Uses declared shard sizes from the index when present and extrapolates for shards
    without a size. The raw coordinate cache and the eventual uint16 token cache are
    estimated from the row-count target and an average chain length.
    """
    chains = ROW_ESTIMATE
    if records:
        num_shards = len(records)
        sized = [int(r["bytes"]) for r in records if r.get("bytes")]
        declared = sum(sized)
        if sized:
            average = declared / len(sized)
            archive_bytes = declared + average * (num_shards - len(sized))
        else:
            archive_bytes = float(EST_BYTES_PER_SHARD_ARCHIVE) * num_shards
    else:
        num_shards = None
        archive_bytes = None
    backbone_bytes = (
        chains
        * AVG_RESIDUES_PER_CHAIN
        * BACKBONE_ATOMS
        * 3
        * BYTES_PER_FLOAT16
    )
    token_bytes = chains * AVG_RESIDUES_PER_CHAIN * BYTES_PER_UINT16
    return {
        "estimated_chains": chains,
        "num_shards": num_shards,
        "download_archive_bytes": None
        if archive_bytes is None
        else int(archive_bytes),
        "backbone_coord_cache_bytes": int(backbone_bytes),
        "struct_token_cache_bytes": int(token_bytes),
        "avg_residues_per_chain": AVG_RESIDUES_PER_CHAIN,
    }


def print_banner(
    estimate: Dict, index_path: Path, records: Optional[List[Dict]]
) -> None:
    """Print provenance, the helix-bias caveat, and the disk estimate."""
    print(
        "[esmatlas] ESMAtlas high-confidence representatives (M2, gated scale tier)"
    )
    print(
        f"[esmatlas] source: {ESMATLAS_HOST} release {ESMATLAS_RELEASE} subset {ESMATLAS_SUBSET}"
    )
    print(
        f"[esmatlas] high-quality filter: mean pLDDT > {MIN_MEAN_PLDDT}, pTM > {MIN_PTM}, "
        f"clustered at <= {int(CLUSTER_SEQUENCE_IDENTITY * 100)}% sequence identity"
    )
    print(f"[esmatlas] license: {ESMATLAS_LICENSE} ({ESMATLAS_LICENSE_URL})")
    print(f"[esmatlas] access README: {ESMATLAS_README}")
    print(
        f"[esmatlas] shard index: {index_path} "
        + ("(found)" if records else "(not found)")
    )
    print(f"[esmatlas] caveat: {HELIX_BIAS_CAVEAT}")
    print("[esmatlas] disk estimate:")
    print(f"[esmatlas]   estimated chains: {estimate['estimated_chains']:,}")
    if estimate["num_shards"] is not None:
        print(f"[esmatlas]   shards to download: {estimate['num_shards']:,}")
    print(
        f"[esmatlas]   download archives: {format_bytes(estimate['download_archive_bytes'])}"
    )
    print(
        f"[esmatlas]   raw backbone coordinate cache (float16): "
        f"{format_bytes(estimate['backbone_coord_cache_bytes'])}"
    )
    print(
        f"[esmatlas]   uint16 structure-token cache (produced later): "
        f"{format_bytes(estimate['struct_token_cache_bytes'])}"
    )
    print(
        "[esmatlas]   keep full coordinate shards only for evaluation and audit "
        "subsets, not the whole corpus."
    )


def download_resumable(url: str, destination: Path, label: str) -> None:
    """Download a URL to a path, resuming from a partial file when the server allows.

    Writes to a .partial sibling and atomically renames on completion so that an
    interrupted run leaves a resumable partial rather than a truncated final file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:  # noqa: S310 - pinned HTTPS host
        append = offset > 0 and getattr(response, "status", 200) == 206
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
                        f"\r[esmatlas] {label}: {received / 1e9:.2f}/"
                        f"{total_bytes / 1e9:.2f} GB",
                        end="",
                        flush=True,
                    )
    print()
    os.replace(partial, destination)


def verify_shard(path: Path, record: Dict) -> Dict:
    """Check an existing shard against declared size and checksum where available.

    Returns a dict with keys ok, bytes, and sha256. A shard with no declared checksum
    is treated as present when it exists and is non-empty; its computed digest is still
    recorded for provenance.
    """
    if not path.exists():
        return {"ok": False, "bytes": 0, "sha256": None}
    size = path.stat().st_size
    if size == 0:
        return {"ok": False, "bytes": 0, "sha256": None}
    expected_bytes = record.get("bytes")
    if expected_bytes is not None and size != int(expected_bytes):
        return {"ok": False, "bytes": size, "sha256": None}
    digest = sha256(path)
    expected_sha = record.get("sha256")
    if expected_sha is not None and digest != str(expected_sha).lower():
        return {"ok": False, "bytes": size, "sha256": digest}
    return {"ok": True, "bytes": size, "sha256": digest}


def prepare_shards(
    records: List[Dict],
    coords_dir: Path,
    *,
    verify_only: bool,
    force: bool,
    max_shards: Optional[int],
) -> List[Dict]:
    """Download and verify each representative shard idempotently.

    Existing shards that match the declared size and checksum are skipped. In
    verify-only mode nothing is downloaded and any missing or corrupt shard raises.
    """
    coords_dir.mkdir(parents=True, exist_ok=True)
    selected = records if max_shards is None else records[:max_shards]
    results: List[Dict] = []
    missing: List[str] = []
    for index, record in enumerate(selected):
        filename = str(record["filename"])
        destination = coords_dir / filename
        status = verify_shard(destination, record)
        label = f"shard {index + 1}/{len(selected)} {filename}"
        if status["ok"] and not force:
            print(f"[esmatlas] {label}: present, skipping")
        elif verify_only:
            missing.append(filename)
            print(f"[esmatlas] {label}: MISSING or CORRUPT")
            continue
        else:
            download_resumable(str(record["url"]), destination, label)
            status = verify_shard(destination, record)
            if not status["ok"]:
                raise ValueError(
                    f"Shard {filename} failed verification after download; "
                    f"expected sha256={record.get('sha256')} bytes={record.get('bytes')}"
                )
        entry = {
            "filename": filename,
            "url": str(record["url"]),
            "bytes": status["bytes"],
            "sha256": status["sha256"],
        }
        results.append(entry)
    if verify_only and missing:
        raise RuntimeError(
            f"{len(missing)} shard(s) missing or corrupt: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )
    return results


def _open_structure_text(path: Path) -> str:
    """Read a structure file as text, decompressing .zst with a lazy dependency."""
    if path.suffix.lower() == ".zst":
        zstandard = _require(
            "zstandard",
            "Install 'zstandard' (pip install zstandard) to read compressed "
            "ESMAtlas .zst structure shards.",
        )
        decompressor = zstandard.ZstdDecompressor()
        with path.open("rb") as handle:
            data = decompressor.stream_reader(handle).read()
        return data.decode("utf-8")
    return path.read_text(encoding="utf-8")


def structure_file_to_backbone(path: Path) -> Optional[np.ndarray]:
    """Parse one structure file into an [L, 4, 3] backbone array using biotite.

    Returns coordinates in N, CA, C, O order per residue for residues with a complete
    backbone, or None if the file contains no usable amino-acid residues. Biotite is
    imported lazily so the module stays importable with numpy and torch only.
    """
    biotite_io = _require(
        "biotite.structure.io.pdbx",
        "Install biotite via the DPLM inference stack "
        "(the dplm-inference extra (uv sync --extra dplm-inference)), for example with "
        "scripts/proteins/setup/setup_evaluation.sh dplm.",
    )
    biotite_pdb = _require(
        "biotite.structure.io.pdb",
        "Install biotite via the DPLM inference stack "
        "(the dplm-inference extra (uv sync --extra dplm-inference)), for example with "
        "scripts/proteins/setup/setup_evaluation.sh dplm.",
    )
    biotite_struct = _require(
        "biotite.structure",
        "Install biotite via the DPLM inference stack "
        "(the dplm-inference extra (uv sync --extra dplm-inference)), for example with "
        "scripts/proteins/setup/setup_evaluation.sh dplm.",
    )
    import io

    text = _open_structure_text(path)
    suffix = path.suffix.lower()
    stem_suffix = (
        Path(path.stem).suffix.lower() if suffix == ".zst" else suffix
    )
    if stem_suffix in (".cif", ".mmcif", ".bcif"):
        cif_file = biotite_io.CIFFile.read(io.StringIO(text))
        atoms = biotite_io.get_structure(cif_file, model=1)
    else:
        pdb_file = biotite_pdb.PDBFile.read(io.StringIO(text))
        atoms = pdb_file.get_structure(model=1)

    atoms = atoms[biotite_struct.filter_amino_acids(atoms)]
    if atoms.array_length() == 0:
        return None

    residues: List[np.ndarray] = []
    starts = biotite_struct.get_residue_starts(atoms, add_exclusive_stop=True)
    for begin, end in zip(starts[:-1], starts[1:]):
        names = atoms.atom_name[begin:end]
        coords = atoms.coord[begin:end]
        backbone = np.full((BACKBONE_ATOMS, 3), np.nan, dtype=np.float32)
        complete = True
        for slot, atom_name in enumerate(BACKBONE_ATOM_ORDER):
            hits = np.where(names == atom_name)[0]
            if hits.size == 0:
                complete = False
                break
            backbone[slot] = coords[hits[0]]
        if complete:
            residues.append(backbone)
    if not residues:
        return None
    return np.stack(residues, axis=0)


def extract_backbone_shards(
    coords_dir: Path, output_dir: Path, shard_size: int
) -> Dict:
    """Convert downloaded structure files into sharded float16 backbone arrays.

    Walks the downloaded coordinate directory for individual .pdb/.cif structure
    files, extracts N, CA, C, O coordinates per residue, and writes them as object
    arrays in fixed-size shards alongside an index. This is a convenience for building
    the raw coordinate cache; the token cache is produced separately by
    tokenize_structures.py.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = ("*.pdb", "*.cif", "*.mmcif", "*.pdb.zst", "*.cif.zst")
    files = sorted(
        {path for pattern in patterns for path in coords_dir.rglob(pattern)}
    )
    index: List[Dict] = []
    shard: List[np.ndarray] = []
    shard_ids: List[str] = []
    shard_number = 0
    parsed = 0
    skipped = 0

    def flush() -> None:
        nonlocal shard_number, shard, shard_ids
        if not shard:
            return
        shard_path = output_dir / f"backbone_{shard_number:05d}.npy"
        array = np.empty(len(shard), dtype=object)
        for position, coords in enumerate(shard):
            array[position] = coords.astype(np.float16)
        np.save(shard_path, array, allow_pickle=True)
        for position, stable_id in enumerate(shard_ids):
            index.append(
                {
                    "stable_id": stable_id,
                    "shard": shard_path.name,
                    "row": position,
                }
            )
        shard_number += 1
        shard = []
        shard_ids = []

    for path in files:
        coords = structure_file_to_backbone(path)
        if coords is None:
            skipped += 1
            continue
        shard.append(coords)
        shard_ids.append(path.name)
        parsed += 1
        if len(shard) >= shard_size:
            flush()
    flush()

    atomic_write_json(output_dir / "backbone_index.json", {"rows": index})
    return {
        "structure_files": len(files),
        "chains_parsed": parsed,
        "chains_skipped": skipped,
        "shards_written": shard_number,
        "output_dir": str(output_dir),
    }


def write_manifest(
    output_dir: Path,
    manifest_path: Path,
    index_path: Path,
    shard_records: List[Dict],
    estimate: Dict,
    extraction: Optional[Dict],
) -> None:
    """Write the frozen manifest recording provenance, checksums, and estimates."""
    total_bytes = sum(int(entry["bytes"]) for entry in shard_records)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tier": "M2",
        "tier_status": "gated scale ablation; not part of the main paper corpus",
        "artifact": "raw predicted backbone coordinates",
        "source": {
            "name": "ESMAtlas high-confidence foldseek-clustered representatives",
            "host": ESMATLAS_HOST,
            "release": ESMATLAS_RELEASE,
            "subset": ESMATLAS_SUBSET,
            "license": ESMATLAS_LICENSE,
            "license_url": ESMATLAS_LICENSE_URL,
            "access_readme": ESMATLAS_README,
        },
        "upstream": {
            "esm_repo": ESM_REPO,
            "esm_commit": ESM_COMMIT,
        },
        "filters": {
            "min_mean_plddt": MIN_MEAN_PLDDT,
            "min_ptm": MIN_PTM,
            "cluster_sequence_identity": CLUSTER_SEQUENCE_IDENTITY,
        },
        "shard_index": {
            "path": str(index_path),
            "sha256": sha256(index_path) if index_path.exists() else None,
            "num_shards_downloaded": len(shard_records),
        },
        "shards": shard_records,
        "shards_total_bytes": total_bytes,
        "disk_estimate": estimate,
        "helix_bias_caveat": HELIX_BIAS_CAVEAT,
        "tokenization": {
            "note": (
                "Raw coordinates only. Pass every chain through the frozen DPLM "
                "structure tokenizer via scripts/proteins/setup/tokenize_structures.py to "
                "build the uint16 token cache used for training."
            ),
            "struct_tokenizer_repo": "airkingbd/struct_tokenizer",
        },
        "backbone_extraction": extraction,
    }
    atomic_write_json(manifest_path, manifest)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ESMAtlas preparation script."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            f"datasets/esmatlas_{ESMATLAS_SUBSET}_{ESMATLAS_RELEASE}"
        ),
        help="Destination directory for shards, index, and the frozen manifest.",
    )
    parser.add_argument(
        "--shard-index",
        type=Path,
        default=None,
        help=(
            "Frozen shard index (JSON list or TSV) pinning the representative files "
            "to download. Defaults to <output-dir>/shard_index.json."
        ),
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help=(
            "Required gate. Without it the script only prints provenance, the "
            "helix-bias caveat, and the disk estimate, then exits without downloading."
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing shards against the index without downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-verify and re-download shards even if they already match.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Process only the first N shards from the index (for staged fetches).",
    )
    parser.add_argument(
        "--extract-backbone",
        action="store_true",
        help=(
            "After download, parse structure files into sharded float16 backbone "
            "arrays. Requires biotite (lazy import)."
        ),
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=8192,
        help="Number of chains per extracted backbone shard.",
    )
    return parser


def main() -> None:
    """Entry point: gate, estimate, then optionally download and verify shards."""
    parser = build_parser()
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    coords_dir = output_dir / "coords"
    manifest_path = output_dir / "frozen_manifest.json"
    index_path = (
        args.shard_index or (output_dir / "shard_index.json")
    ).resolve()

    records: Optional[List[Dict]] = None
    if index_path.exists():
        records = load_shard_index(index_path)

    estimate = disk_estimate(records)
    print_banner(estimate, index_path, records)

    if not args.enable and not args.verify_only:
        print(
            "[esmatlas] gated tier: pass --enable to download the raw coordinate "
            "shards. The default run does nothing."
        )
        return

    if records is None:
        raise RuntimeError(
            f"Shard index not found at {index_path}. Obtain the ESMAtlas "
            f"high-quality clustered representative index per {ESMATLAS_README}, or "
            f"pass --shard-index PATH. The index is a JSON list or TSV of "
            f"'filename<TAB>url' with optional sha256 and bytes columns."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_records = prepare_shards(
        records,
        coords_dir,
        verify_only=args.verify_only,
        force=args.force,
        max_shards=args.max_shards,
    )

    extraction: Optional[Dict] = None
    if args.extract_backbone and not args.verify_only:
        extraction = extract_backbone_shards(
            coords_dir, output_dir / "backbone", args.shard_size
        )
        print(
            f"[esmatlas] extracted backbone for {extraction['chains_parsed']:,} chains "
            f"into {extraction['shards_written']:,} shard(s)"
        )

    write_manifest(
        output_dir,
        manifest_path,
        index_path,
        shard_records,
        estimate,
        extraction,
    )
    print(f"[esmatlas] verified {len(shard_records):,} shard(s)")
    print(f"[esmatlas] frozen manifest: {manifest_path}")


if __name__ == "__main__":
    main()
