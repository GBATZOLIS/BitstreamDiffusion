#!/usr/bin/env python3
"""Prepare CAMEO 2022 targets from the verified DPLM metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
OPENFOLD_AA = "ARNDCQEGHILKMFPSTWYV"
BACKBONE37 = (0, 1, 2, 4)  # N, CA, C, O in OpenFold atom37.
ATOM_ORDER = ("N", "CA", "C", "O")

AA3_TO_1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


class RestrictedUnpickler(pickle.Unpickler):
    """Load only the basic NumPy containers present in official data files."""

    _ALLOWED = {
        ("collections", "defaultdict"): defaultdict,
        ("builtins", "list"): list,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        (
            "numpy.core.multiarray",
            "_reconstruct",
        ): np.core.multiarray._reconstruct,
        (
            "numpy._core.multiarray",
            "_reconstruct",
        ): np.core.multiarray._reconstruct,
        ("numpy.core.numeric", "_frombuffer"): np.core.numeric._frombuffer,
        ("numpy._core.numeric", "_frombuffer"): np.core.numeric._frombuffer,
    }

    def find_class(self, module: str, name: str):
        allowed = self._ALLOWED.get((module, name))
        if allowed is None:
            raise pickle.UnpicklingError(
                f"refusing pickle global {module}.{name}; input is not a "
                f"supported benchmark container"
            )
        return allowed


def restricted_load(path: Path):
    with path.open("rb") as handle:
        return RestrictedUnpickler(handle).load()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def parse_fasta(
    path: Path, *, comma_tokens: bool = False
) -> dict[str, object]:
    records: dict[str, object] = {}
    name = None
    chunks: list[str] = []

    def flush() -> None:
        nonlocal chunks
        if name is None:
            return
        text = "".join(chunks)
        if comma_tokens:
            records[name] = np.asarray(
                [int(value) for value in text.split(",") if value.strip()],
                dtype=np.uint16,
            )
        else:
            records[name] = text.strip().upper()
        chunks = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            flush()
            name = line[1:].split()[0]
            if name in records:
                raise ValueError(f"duplicate FASTA id {name!r} in {path}")
        else:
            chunks.append(line.strip())
    flush()
    return records


def select_fasta_residues(
    data: Mapping[str, object], sequence: str, target_id: str
) -> np.ndarray:
    """
    Map an official DPLM FASTA record back onto its source coordinate
    row.
    """
    aatype = np.asarray(data["aatype"], dtype=np.int64).reshape(-1)
    decoded = "".join(
        OPENFOLD_AA[int(value)] if 0 <= int(value) < 20 else "X"
        for value in aatype
    )
    modeled_idx = np.asarray(
        data.get("modeled_idx", np.arange(aatype.shape[0])), dtype=np.int64
    ).reshape(-1)
    if (
        modeled_idx.shape[0] == len(sequence)
        and "".join(decoded[int(index)] for index in modeled_idx) == sequence
    ):
        return modeled_idx

    # PDB_date FASTAs can retain an X residue that modeled_idx excludes.
    # In that case the FASTA is an exact contiguous chain in the full
    # atom37 record.  Resolve it explicitly and use modeled_idx overlap
    # to disambiguate repeated/symmetry-related chains.
    chain_index = np.asarray(
        data.get("chain_index", np.zeros(aatype.shape[0], dtype=np.int64))
    ).reshape(-1)
    candidates = []
    start = decoded.find(sequence)
    while start >= 0:
        stop = start + len(sequence)
        if stop <= chain_index.shape[0] and np.all(
            chain_index[start:stop] == chain_index[start]
        ):
            selection = np.arange(start, stop, dtype=np.int64)
            overlap = int(np.isin(selection, modeled_idx).sum())
            candidates.append((overlap, start, selection))
        start = decoded.find(sequence, start + 1)
    if not candidates:
        raise ValueError(
            f"{target_id}: official FASTA cannot be aligned to modeled_idx"
            f" or a contiguous source chain"
        )
    candidates.sort(key=lambda row: (row[0], -row[1]), reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise ValueError(
            f"{target_id}: official FASTA has an ambiguous "
            f"coordinate-chain mapping"
        )
    return candidates[0][2]


def prepare_dplm_structure_split(
    raw_root: Path,
    output_root: Path,
    *,
    raw_name: str,
    output_name: str,
    source_label: str,
    expected_count: int | None,
) -> dict:
    split_root = raw_root / raw_name
    fasta_path = split_root / "aatype.fasta"
    struct_path = split_root / "struct.fasta"
    pkl_dir = split_root / "preprocessed"
    for path in (fasta_path, struct_path, pkl_dir):
        if not path.exists():
            raise FileNotFoundError(
                f"missing extracted DPLM metadata input: {path}"
            )

    sequences = parse_fasta(fasta_path)
    tokens = parse_fasta(struct_path, comma_tokens=True)
    # DPLM stores CAMEO pickles directly in preprocessed and shards the
    # larger PDB-date split into two-character subdirectories.
    available_files = sorted(pkl_dir.rglob("*.pkl"))
    available_ids = [path.stem for path in available_files]
    if len(set(available_ids)) != len(available_ids):
        raise ValueError(f"{raw_name}: duplicate pickle ids across shards")
    if set(sequences) != set(tokens):
        raise ValueError(
            f"{raw_name}: sequence/token FASTA ids disagree: sequence="
            f"{len(sequences)} tokens={len(tokens)}"
        )
    file_by_id = dict(zip(available_ids, available_files, strict=True))
    missing_pickles = sorted(set(sequences).difference(file_by_id))
    if missing_pickles:
        raise ValueError(
            f"{raw_name}: {len(missing_pickles)} FASTA targets have no "
            f"coordinate pickle; first={missing_pickles[:5]}"
        )
    # PDB_date ships a larger sharded coordinate pool while its two
    # FASTAs define the official 449-target benchmark.  Only those FASTA
    # IDs belong in the evaluation manifest; the remaining pickles are
    # source metadata.
    source_ids = sorted(sequences)
    if expected_count is not None and len(source_ids) != int(expected_count):
        raise ValueError(
            f"{raw_name}: expected {expected_count} FASTA targets, found "
            f"{len(source_ids)}"
        )
    excluded_noncanonical = [
        target_id
        for target_id in source_ids
        if any(
            residue not in CANONICAL_AA
            for residue in str(sequences[target_id])
        )
    ]
    excluded_set = set(excluded_noncanonical)
    ids = [
        target_id for target_id in source_ids if target_id not in excluded_set
    ]
    files = [file_by_id[target_id] for target_id in ids]

    destination = output_root / output_name
    coords_dir = destination / "coords"
    coords_dir.mkdir(parents=True, exist_ok=True)
    targets = []
    total_residues = 0
    total_native = 0
    for path in files:
        target_id = path.stem
        data = restricted_load(path)
        sequence = str(sequences[target_id])
        struct_index = np.asarray(tokens[target_id], dtype=np.uint16).reshape(
            -1
        )
        selected_idx = select_fasta_residues(data, sequence, target_id)
        atom_positions = np.asarray(data["atom_positions"], dtype=np.float32)[
            selected_idx
        ]
        atom_mask = np.asarray(data["atom_mask"], dtype=bool)[selected_idx]
        coords = np.ascontiguousarray(atom_positions[:, BACKBONE37, :])
        native_mask = (
            np.asarray(data["bb_mask"], dtype=bool).reshape(-1)[selected_idx]
            & atom_mask[:, BACKBONE37].all(axis=1)
            & np.isfinite(coords).all(axis=(1, 2))
        )
        length = len(sequence)
        if coords.shape != (length, 4, 3) or struct_index.shape[0] != length:
            raise ValueError(
                f"{target_id}: sequence={length}, coords={coords.shape}, "
                f"tokens={struct_index.shape}"
            )
        aatype = np.asarray(data["aatype"], dtype=np.int64).reshape(-1)[
            selected_idx
        ]
        decoded = "".join(
            OPENFOLD_AA[int(value)] if 0 <= int(value) < 20 else "X"
            for value in aatype
        )
        if decoded != sequence:
            raise ValueError(
                f"{target_id}: aatype does not match official FASTA"
            )
        if not bool(native_mask.any()):
            raise ValueError(
                f"{target_id}: no resolved native backbone residues"
            )

        coordinate_file = coords_dir / f"{target_id}.npz"
        np.savez_compressed(
            coordinate_file,
            coords=coords,
            native_mask=native_mask,
            struct_index=struct_index,
            source=np.asarray(source_label),
        )
        targets.append(
            {
                "target_id": target_id,
                "sequence": sequence,
                "coords_path": str(Path("coords") / coordinate_file.name),
                "native_mask_path": str(Path("coords") / coordinate_file.name),
                "source": source_label,
                "length": length,
                "num_native_residues": int(native_mask.sum()),
                "native_coverage": float(native_mask.mean()),
            }
        )
        total_residues += length
        total_native += int(native_mask.sum())

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "benchmark": source_label,
        "source": {
            "record": "DPLM-2 Dataset, Zenodo 15424801",
            "doi": "10.5281/zenodo.15424801",
            "raw_directory": str(split_root),
            "aatype_fasta_sha256": sha256(fasta_path),
            "struct_fasta_sha256": sha256(struct_path),
        },
        "coordinate_convention": "OpenFold atom37 slots 0,1,2,4 -> N,CA,C,O",
        "native_mask_policy": (
            "Map each official FASTA exactly to modeled_idx, or to its "
            "exact contiguous source chain when the FASTA retains an X "
            "excluded by modeled_idx; within that selection score "
            "residues with bb_mask plus complete finite N/CA/C/O."
        ),
        "eligibility_policy": (
            "Evaluate only targets whose conditioning sequence uses the "
            "model's canonical 20-amino-acid vocabulary; never "
            "substitute an unknown residue."
        ),
        "num_source_targets": len(source_ids),
        "num_excluded_noncanonical": len(excluded_noncanonical),
        "excluded_noncanonical_target_ids": excluded_noncanonical,
        "num_targets": len(targets),
        "num_sequence_residues": total_residues,
        "num_native_residues": total_native,
        "targets": targets,
    }
    atomic_json(destination / "targets.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the CAMEO 2022 benchmark."
    )
    parser.add_argument(
        "--dplm-root", type=Path, default=Path("datasets/dplm2_metadata")
    )
    parser.add_argument(
        "--out-root", type=Path, default=Path("datasets/benchmarks")
    )
    args = parser.parse_args()
    result = prepare_dplm_structure_split(
        args.dplm_root,
        args.out_root,
        raw_name="cameo2022",
        output_name="cameo2022_dplm21",
        source_label="CAMEO 2022 (DPLM-2.1 split)",
        expected_count=183,
    )
    print(f"Prepared {result['num_targets']} CAMEO targets.")


if __name__ == "__main__":
    main()
