#!/usr/bin/env python3
"""Prepare the plan 7.4 M1 AFDB-representative corpus as raw backbone coordinates.

This script downloads AlphaFold Database (AFDB) predicted structures for a list
of Foldseek cluster representatives and extracts their backbone coordinates
(atoms N, CA, C, O) into a raw, untokenized coordinate corpus for later offline
tokenization. The intended corpus is roughly 1.2 to 1.3 million confident
predicted structures. Because a full pass transfers on the order of a terabyte
of structure files over the network, the download is gated behind an explicit
``--enable`` flag so that the ordinary setup path never pulls it by default.

The script is deliberately narrow. It downloads structures, filters them by
confidence, extracts backbone coordinates, and records accession lists,
per-structure checksums, and the upstream license. It does not run the DPLM
structure tokenizer; converting these coordinates into structure token ids is
the job of ``tokenize_structures.py``.

Graceful degradation: importing this module and running ``--help`` require only
numpy (and a Python standard library). The structure parser depends on biotite
or biopython, both imported lazily inside the parsing function and guarded with
an actionable error. Network access uses the standard library ``urllib``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Frozen source, license, and format constants.
# -----------------------------------------------------------------------------

AFDB_MODEL_VERSION = 4
AFDB_FILE_URL_TEMPLATE = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F{fragment}-model_v{version}.cif"
AFDB_DATABASE_URL = "https://alphafold.ebi.ac.uk/"

# AlphaFold DB structures and derived data are released under CC-BY-4.0.
AFDB_LICENSE_NAME = "CC-BY-4.0"
AFDB_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
AFDB_ATTRIBUTION = (
    "Jumper et al., Highly accurate protein structure prediction with AlphaFold, "
    "Nature 596 (2021); Varadi et al., AlphaFold Protein Structure Database, "
    "Nucleic Acids Research 50 (2022)."
)
AFDB_CLUSTER_ATTRIBUTION = (
    "Barrio-Hernandez et al., Clustering-predicted structures at the scale of the "
    "known protein universe, Nature 622 (2023)."
)

USER_AGENT = "BitstreamDiffusion-AFDB-prep/1.0"

# Backbone atoms and their slot order in the [L, 4, 3] coordinate tensor. This
# matches the DPLM-2 structure tokenizer atom convention "N, CA, C, O".
BACKBONE_SLOTS: Dict[str, int] = {"N": 0, "CA": 1, "C": 2, "O": 3}

DEFAULT_MIN_PLDDT = 70.0
DEFAULT_OUT_DIR = Path("datasets/afdb_representatives")
# Planned corpus size, used only for the pre-download disk estimate when no
# cluster list or cap is supplied.
DEFAULT_PLANNED_STRUCTURES = 1_250_000

# Rough per-structure sizes used only for the disk estimate printed before any
# large work begins. Actual sizes vary with protein length.
ESTIMATED_RESIDUES = 350
ESTIMATED_CIF_BYTES = 1_000_000  # a typical single-chain AlphaFold CIF file
COORD_BYTES_PER_RESIDUE = 4 * 3 * 4  # [4 atoms, 3 coords] float32

# Accepted header names for the representative-accession column of a cluster list.
ACCESSION_COLUMN_NAMES = {
    "repid",
    "rep_id",
    "rep",
    "representative",
    "representativeid",
    "accession",
    "acc",
    "id",
    "member",
    "cluster_rep",
}

MANIFEST_NAME = "afdb_representatives_manifest.json"
CHECKSUMS_NAME = "checksums.jsonl"
ACCESSIONS_NAME = "accessions.txt"
LICENSE_NAME = "LICENSE_AFDB.txt"


# -----------------------------------------------------------------------------
# Small helpers.
# -----------------------------------------------------------------------------


def format_bytes(num: float) -> str:
    """Return a human-readable size string for a byte count."""
    value = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024.0 or unit == "PiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PiB"


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of a byte string."""
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to a path atomically by writing to a temporary file first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Dict) -> None:
    """Serialize a JSON payload to a path atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def normalize_accession(token: str) -> str:
    """Extract a bare UniProt accession from a cluster-list token.

    Cluster representatives may be recorded either as bare accessions such as
    ``P12345`` or as AlphaFold model identifiers such as
    ``AF-P12345-F1-model_v4``. Both map to the accession ``P12345``.
    """
    token = token.strip()
    match = re.match(r"^AF-([A-Za-z0-9]+)-F\d+", token)
    if match:
        return match.group(1)
    return token


def bucket_for(accession: str) -> str:
    """Return a two-character shard directory name for an accession.

    A stable hash bucket keeps any single directory from holding the whole
    corpus while remaining deterministic across runs.
    """
    return hashlib.sha1(accession.encode("utf-8")).hexdigest()[:2]


def coord_path(out_dir: Path, accession: str) -> Path:
    """Return the coordinate file path for an accession."""
    return out_dir / "coords" / bucket_for(accession) / f"AF-{accession}.npz"


def raw_path(out_dir: Path, accession: str) -> Path:
    """Return the retained raw CIF path for an accession."""
    filename = f"AF-{accession}-F1-model_v{AFDB_MODEL_VERSION}.cif"
    return out_dir / "raw" / bucket_for(accession) / filename


# -----------------------------------------------------------------------------
# Cluster list parsing.
# -----------------------------------------------------------------------------


def _sniff_delimiter(line: str) -> Optional[str]:
    """Guess the field delimiter of a cluster-list line."""
    if "\t" in line:
        return "\t"
    if "," in line:
        return ","
    return None


def read_cluster_list(path: Path) -> Iterator[Tuple[str, Optional[float]]]:
    """Yield ``(accession, avg_plddt)`` pairs from a cluster-list file.

    The file may be a plain list with one representative accession per line, or a
    delimited table. When a header row is present its column names are used to
    locate the accession column and an average-pLDDT column (any column whose
    name contains ``plddt``); the average pLDDT is returned so callers can filter
    without downloading. When no such column exists the pLDDT is ``None`` and the
    caller must rely on values parsed from the structures themselves. Blank lines
    and lines beginning with ``#`` are ignored.
    """
    with path.open("r", encoding="utf-8") as handle:
        header_consumed = False
        acc_col = 0
        plddt_col: Optional[int] = None
        delimiter: Optional[str] = None
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if not header_consumed:
                header_consumed = True
                delimiter = _sniff_delimiter(line)
                fields = line.split(delimiter) if delimiter else line.split()
                lowered = [field.strip().lower() for field in fields]
                is_header = any(
                    name in ACCESSION_COLUMN_NAMES or "plddt" in name
                    for name in lowered
                )
                if is_header:
                    for index, name in enumerate(lowered):
                        if name in ACCESSION_COLUMN_NAMES:
                            acc_col = index
                            break
                    for index, name in enumerate(lowered):
                        if "plddt" in name:
                            plddt_col = index
                            break
                    continue
            fields = line.split(delimiter) if delimiter else line.split()
            if acc_col >= len(fields):
                continue
            accession = normalize_accession(fields[acc_col])
            if not accession:
                continue
            plddt: Optional[float] = None
            if plddt_col is not None and plddt_col < len(fields):
                try:
                    plddt = float(fields[plddt_col])
                except ValueError:
                    plddt = None
            yield accession, plddt


def count_line_estimate(path: Path) -> int:
    """Return a fast, header-agnostic count of non-empty data lines."""
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip() and not raw_line.lstrip().startswith("#"):
                count += 1
    return max(count - 1, 0) if count else 0


# -----------------------------------------------------------------------------
# Structure parsing (lazy heavy dependency).
# -----------------------------------------------------------------------------


def _parse_with_biotite(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Parse backbone coordinates and pLDDT using biotite, or return ``None``.

    Returns ``None`` when biotite is not importable so the caller can try a
    fallback parser before raising.
    """
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdbx as pdbx
    except (
        Exception
    ):  # pragma: no cover - exercised only when biotite is absent
        return None

    cif = pdbx.CIFFile.read(str(path))
    atoms = pdbx.get_structure(cif, model=1, extra_fields=["b_factor"])
    atoms = atoms[struc.filter_amino_acids(atoms)]
    if atoms.array_length() == 0:
        raise ValueError("no amino-acid atoms in structure")
    first_chain = np.unique(atoms.chain_id)[0]
    atoms = atoms[atoms.chain_id == first_chain]

    _, inverse = np.unique(atoms.res_id, return_inverse=True)
    num_res = int(inverse.max()) + 1
    coords = np.full((num_res, 4, 3), np.nan, dtype=np.float32)
    plddt = np.full((num_res,), np.nan, dtype=np.float32)
    names = atoms.atom_name
    for name, slot in BACKBONE_SLOTS.items():
        mask = names == name
        if np.any(mask):
            coords[inverse[mask], slot] = atoms.coord[mask]
    ca_mask = names == "CA"
    if np.any(ca_mask):
        plddt[inverse[ca_mask]] = atoms.b_factor[ca_mask].astype(np.float32)
    return coords, plddt


def _parse_with_biopython(
    path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Parse backbone coordinates and pLDDT using biopython, or return ``None``."""
    try:
        from Bio.PDB import MMCIFParser
        from Bio.PDB.Polypeptide import is_aa
    except (
        Exception
    ):  # pragma: no cover - exercised only when biopython is absent
        return None

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("afdb", str(path))
    model = next(structure.get_models())
    coord_rows: List[np.ndarray] = []
    plddt_rows: List[float] = []
    for chain in model:
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            try:
                atoms = [residue[name].coord for name in ("N", "CA", "C", "O")]
            except KeyError:
                continue
            coord_rows.append(np.asarray(atoms, dtype=np.float32))
            plddt_rows.append(float(residue["CA"].bfactor))
        break  # AlphaFold predictions are single-chain; use the first chain only.
    if not coord_rows:
        return (
            np.empty((0, 4, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return (
        np.stack(coord_rows, axis=0).astype(np.float32),
        np.asarray(plddt_rows, dtype=np.float32),
    )


def parse_backbone_coords(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return complete backbone coordinates ``[L, 4, 3]`` and pLDDT ``[L]``.

    Residues missing any of the four backbone atoms are dropped so the result is
    directly consumable by the structure tokenizer. The pLDDT values are read
    from the CA B-factor column that AlphaFold uses to store per-residue
    confidence. Parsing requires biotite or biopython; when neither is installed
    a clear, actionable error is raised.
    """
    result = _parse_with_biotite(path)
    if result is None:
        result = _parse_with_biopython(path)
    if result is None:
        raise RuntimeError(
            "Parsing AlphaFold CIF files requires biotite or biopython, and "
            "neither is importable. Install the DPLM inference stack with\n"
            "  scripts/proteins/setup/setup_evaluation.sh dplm\n"
            "or add the dependencies from the dplm-inference extra (uv sync --extra dplm-inference) "
            "(biotite) or the protein-eval extra (uv sync --extra protein-eval) (biopython)."
        )
    coords, plddt = result
    if coords.shape[0] == 0:
        return coords, plddt
    complete = ~np.isnan(coords).any(axis=(1, 2))
    return coords[complete].astype(np.float32), plddt[complete].astype(
        np.float32
    )


# -----------------------------------------------------------------------------
# Download.
# -----------------------------------------------------------------------------


def download_structure_bytes(
    accession: str, *, retries: int = 4, timeout: float = 60.0
) -> Optional[bytes]:
    """Download an AlphaFold CIF file for an accession.

    Returns the file contents, or ``None`` when the entry is absent from AFDB
    (HTTP 404). Transient network errors are retried with a bounded backoff
    before a final failure is raised.
    """
    url = AFDB_FILE_URL_TEMPLATE.format(
        accession=accession, fragment=1, version=AFDB_MODEL_VERSION
    )
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - pinned HTTPS host
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last_error = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(min(2.0**attempt, 30.0))
    raise RuntimeError(f"failed to download {url}: {last_error}")


# -----------------------------------------------------------------------------
# Coordinate storage.
# -----------------------------------------------------------------------------


def save_coords(
    path: Path, coords: np.ndarray, plddt: np.ndarray
) -> Tuple[int, str]:
    """Write coordinates and pLDDT to a compressed npz atomically.

    Returns the byte size and SHA-256 digest of the written file so the caller
    can record it for idempotent size and hash checks.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez(
        buffer,
        coords=coords.astype(np.float32),
        plddt=plddt.astype(np.float32),
    )
    data = buffer.getvalue()
    atomic_write_bytes(path, data)
    return len(data), sha256_bytes(data)


def load_processed_accessions(checksums_path: Path) -> set:
    """Return the set of accessions already recorded in the checksums log."""
    processed: set = set()
    if not checksums_path.exists():
        return processed
    with checksums_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a truncated final line from an interrupted run
            accession = record.get("accession")
            if accession:
                processed.add(accession)
    return processed


# -----------------------------------------------------------------------------
# Disk estimate and gating message.
# -----------------------------------------------------------------------------


def estimate_disk(num_structures: int, keep_cif: bool) -> Tuple[int, int]:
    """Return estimated network transfer bytes and stored coordinate bytes."""
    transfer = num_structures * ESTIMATED_CIF_BYTES
    coord_bytes = num_structures * (
        ESTIMATED_RESIDUES * COORD_BYTES_PER_RESIDUE
        + ESTIMATED_RESIDUES * 4
        + 256
    )
    stored = coord_bytes + (transfer if keep_cif else 0)
    return transfer, stored


def resolve_planned_count(args: argparse.Namespace) -> int:
    """Return an approximate number of structures for the disk estimate."""
    if args.cluster_list is not None and Path(args.cluster_list).exists():
        candidate = count_line_estimate(Path(args.cluster_list))
        if args.max_structures is not None:
            return min(candidate, args.max_structures)
        return candidate or DEFAULT_PLANNED_STRUCTURES
    if args.max_structures is not None:
        return args.max_structures
    return DEFAULT_PLANNED_STRUCTURES


def print_plan_message(planned: int, args: argparse.Namespace) -> None:
    """Emit a clear description of the corpus and its disk cost before work."""
    transfer, stored = estimate_disk(planned, keep_cif=args.keep_cif)
    print(
        "[afdb] Plan 7.4 M1 AFDB-representative corpus (raw backbone coordinates)."
    )
    print(f"[afdb] Output directory: {Path(args.out_dir).resolve()}")
    print(f"[afdb] Minimum mean pLDDT filter: {args.min_plddt}")
    print(f"[afdb] Planned structures (approximate): {planned:,}")
    print(
        f"[afdb] Estimated network transfer (AlphaFold CIF files): {format_bytes(transfer)}"
    )
    print(f"[afdb] Estimated stored size on disk: {format_bytes(stored)}")
    if args.keep_cif:
        print(
            "[afdb] Raw CIF files will be retained (--keep-cif); this is the large term above."
        )
    else:
        print(
            "[afdb] Raw CIF files are parsed then discarded; pass --keep-cif to retain them."
        )
    print(
        "[afdb] This step does NOT tokenize. Run tokenize_structures.py afterwards."
    )
    print(
        f"[afdb] License of downloaded data: {AFDB_LICENSE_NAME} ({AFDB_LICENSE_URL})."
    )


# -----------------------------------------------------------------------------
# License artifact and manifest.
# -----------------------------------------------------------------------------


def write_license(out_dir: Path) -> Path:
    """Write the AFDB license and attribution notice to the output directory."""
    path = out_dir / LICENSE_NAME
    text = (
        "AlphaFold Protein Structure Database representative corpus\n"
        "\n"
        f"License: {AFDB_LICENSE_NAME}\n"
        f"License URL: {AFDB_LICENSE_URL}\n"
        f"Source database: {AFDB_DATABASE_URL}\n"
        "\n"
        "Attribution:\n"
        f"  {AFDB_ATTRIBUTION}\n"
        f"  {AFDB_CLUSTER_ATTRIBUTION}\n"
        "\n"
        "These files contain backbone coordinates extracted from AlphaFold DB\n"
        "predicted structures. Predicted confidence (pLDDT) is stored per\n"
        "residue. Retain this notice when redistributing the corpus.\n"
    )
    atomic_write_bytes(path, text.encode("utf-8"))
    return path


def build_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    counters: Dict[str, int],
    checksums_path: Path,
    accessions_path: Path,
) -> Dict:
    """Assemble the frozen manifest describing the prepared corpus."""
    artifacts: Dict[str, Dict[str, object]] = {}
    for path in (checksums_path, accessions_path, out_dir / LICENSE_NAME):
        if path.exists():
            artifacts[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_item": "7.4 M1 AFDB-representative corpus (raw coordinates)",
        "tokenized": False,
        "next_step": "scripts/proteins/setup/tokenize_structures.py",
        "source": {
            "database": "AlphaFold Protein Structure Database",
            "database_url": AFDB_DATABASE_URL,
            "file_url_template": AFDB_FILE_URL_TEMPLATE,
            "model_version": AFDB_MODEL_VERSION,
            "cluster_list": str(Path(args.cluster_list).resolve())
            if args.cluster_list is not None
            else None,
            "cluster_attribution": AFDB_CLUSTER_ATTRIBUTION,
        },
        "license": {
            "name": AFDB_LICENSE_NAME,
            "url": AFDB_LICENSE_URL,
            "attribution": AFDB_ATTRIBUTION,
        },
        "parameters": {
            "min_plddt": float(args.min_plddt),
            "max_structures": args.max_structures,
            "keep_cif": bool(args.keep_cif),
        },
        "coordinate_format": {
            "layout": "[L, 4, 3] float32",
            "atom_order": list(BACKBONE_SLOTS.keys()),
            "plddt": "[L] float32, per-residue CA B-factor confidence",
            "storage": "one compressed npz per structure under coords/<bucket>/",
        },
        "counts": dict(counters),
        "artifacts": artifacts,
    }


# -----------------------------------------------------------------------------
# Main preparation loop.
# -----------------------------------------------------------------------------


def prepare(args: argparse.Namespace) -> Dict[str, int]:
    """Download, filter, and store the AFDB-representative coordinate corpus.

    The loop is idempotent: structures whose coordinate file already exists or
    that are already recorded in the checksums log are skipped and counted toward
    the requested cap, so an interrupted run resumes without redoing work.
    """
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checksums_path = out_dir / CHECKSUMS_NAME
    accessions_path = out_dir / ACCESSIONS_NAME
    manifest_path = out_dir / MANIFEST_NAME
    write_license(out_dir)

    processed = load_processed_accessions(checksums_path)
    counters = {
        "candidates_seen": 0,
        "kept_total": 0,
        "newly_saved": 0,
        "skipped_existing": 0,
        "filtered_plddt_pre": 0,
        "filtered_plddt_post": 0,
        "missing_from_afdb": 0,
        "parse_failed": 0,
        "empty_structures": 0,
    }
    max_structures = args.max_structures

    checksums_handle = checksums_path.open("a", encoding="utf-8")
    accessions_handle = accessions_path.open("w", encoding="utf-8")
    try:
        for accession, avg_plddt in read_cluster_list(Path(args.cluster_list)):
            if (
                max_structures is not None
                and counters["kept_total"] >= max_structures
            ):
                break
            counters["candidates_seen"] += 1

            target = coord_path(out_dir, accession)
            if accession in processed or target.exists():
                counters["kept_total"] += 1
                counters["skipped_existing"] += 1
                accessions_handle.write(accession + "\n")
                processed.add(accession)
                continue

            if avg_plddt is not None and avg_plddt < args.min_plddt:
                counters["filtered_plddt_pre"] += 1
                continue

            data = download_structure_bytes(accession)
            if data is None:
                counters["missing_from_afdb"] += 1
                continue
            raw_sha = sha256_bytes(data)
            raw_bytes = len(data)

            if args.keep_cif:
                cif_path = raw_path(out_dir, accession)
                if not (
                    cif_path.exists() and cif_path.stat().st_size == raw_bytes
                ):
                    atomic_write_bytes(cif_path, data)
                parse_source = cif_path
                cleanup = None
            else:
                handle = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".cif", dir=str(out_dir)
                )
                handle.write(data)
                handle.close()
                parse_source = Path(handle.name)
                cleanup = parse_source

            try:
                coords, plddt = parse_backbone_coords(parse_source)
            except RuntimeError:
                raise  # missing parser dependency; surface immediately.
            except Exception:  # noqa: BLE001 - one malformed file must not stop the run
                counters["parse_failed"] += 1
                continue
            finally:
                if cleanup is not None:
                    cleanup.unlink(missing_ok=True)

            if coords.shape[0] == 0:
                counters["empty_structures"] += 1
                continue
            mean_plddt = float(np.nanmean(plddt)) if plddt.size else 0.0
            if mean_plddt < args.min_plddt:
                counters["filtered_plddt_post"] += 1
                continue

            coord_bytes, coord_sha = save_coords(target, coords, plddt)
            record = {
                "accession": accession,
                "num_residues": int(coords.shape[0]),
                "mean_plddt": round(mean_plddt, 3),
                "raw_bytes": raw_bytes,
                "raw_sha256": raw_sha,
                "coord_path": str(target.relative_to(out_dir)),
                "coord_bytes": coord_bytes,
                "coord_sha256": coord_sha,
            }
            checksums_handle.write(json.dumps(record, sort_keys=True) + "\n")
            checksums_handle.flush()
            accessions_handle.write(accession + "\n")
            processed.add(accession)
            counters["kept_total"] += 1
            counters["newly_saved"] += 1

            if counters["candidates_seen"] % 1000 == 0:
                print(
                    f"[afdb] seen={counters['candidates_seen']:,} "
                    f"kept={counters['kept_total']:,} "
                    f"saved={counters['newly_saved']:,} "
                    f"missing={counters['missing_from_afdb']:,}",
                    flush=True,
                )
    finally:
        checksums_handle.close()
        accessions_handle.close()

    manifest = build_manifest(
        out_dir, args, counters, checksums_path, accessions_path
    )
    atomic_write_json(manifest_path, manifest)
    print(
        f"[afdb] kept {counters['kept_total']:,} structures "
        f"({counters['newly_saved']:,} newly saved this run)."
    )
    print(f"[afdb] manifest: {manifest_path}")
    print(f"[afdb] checksums: {checksums_path}")
    print(f"[afdb] accessions: {accessions_path}")
    return counters


# -----------------------------------------------------------------------------
# Command-line interface.
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the corpus preparation script."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory that receives coordinates, checksums, license, and manifest.",
    )
    parser.add_argument(
        "--cluster-list",
        type=Path,
        default=None,
        help="File listing AFDB cluster-representative accessions (required with --enable).",
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="Optional cap on the number of kept structures.",
    )
    parser.add_argument(
        "--min-plddt",
        type=float,
        default=DEFAULT_MIN_PLDDT,
        help="Minimum mean per-residue pLDDT for a structure to be kept.",
    )
    parser.add_argument(
        "--keep-cif",
        action="store_true",
        help="Retain the raw downloaded CIF files instead of discarding them after parsing.",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Required gate. Without it the script only prints the plan and disk estimate.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, show the plan and estimate, and prepare the corpus when enabled."""
    args = build_parser().parse_args(argv)
    planned = resolve_planned_count(args)
    print_plan_message(planned, args)

    if not args.enable:
        print(
            "[afdb] --enable was not passed. This corpus is gated so that setup "
            "does not download terabytes by default. Re-run with --enable to proceed."
        )
        return 0

    if args.cluster_list is None:
        print(
            "[afdb] --cluster-list is required together with --enable. Provide a file "
            "of AFDB cluster-representative accessions.",
            file=sys.stderr,
        )
        return 2
    if not Path(args.cluster_list).exists():
        print(
            f"[afdb] cluster list not found: {args.cluster_list}",
            file=sys.stderr,
        )
        return 2

    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
