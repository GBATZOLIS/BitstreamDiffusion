"""Write backbone structures to PDB files."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"

AA1_TO_3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}

# Backbone atom order used throughout the protein evaluation stack.
ATOM_ORDER = ("N", "CA", "C", "O")
ATOM_ELEMENT = {"N": "N", "CA": "C", "C": "C", "O": "O"}


def _format_atom_name(name: str) -> str:
    """Return the 4-character PDB atom-name field."""
    if len(name) >= 4:
        return name[:4]
    return f" {name:<3s}"


def write_backbone_pdb(
    path,
    coords: np.ndarray,
    sequence: str,
    *,
    chain_id: str = "A",
    start_resseq: int = 1,
    bfactors: Sequence[float] | None = None,
    remarks: Iterable[str] = (),
) -> Path:
    """Write one chain of backbone coordinates as a PDB file."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[1:] != (4, 3):
        raise ValueError(
            f"coords must have shape [L, 4, 3] in order {ATOM_ORDER}, got "
            f"{coords.shape}"
        )
    if len(sequence) != coords.shape[0]:
        raise ValueError(
            f"sequence length {len(sequence)} does not match "
            f"{coords.shape[0]} residues"
        )
    if bfactors is not None and len(bfactors) != coords.shape[0]:
        raise ValueError("bfactors must have one value per residue")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"REMARK   1 {r}" for r in remarks]
    serial = 1
    for i, aa in enumerate(sequence):
        resname = AA1_TO_3.get(aa.upper())
        if resname is None:
            raise ValueError(
                f"residue {i} is {aa!r}, which is not one of the canonical"
                f" 20 ({CANONICAL_AA}). A PDB file cannot represent it as "
                f"a standard residue, and writing ALA instead would "
                f"silently change the protein."
            )
        resseq = start_resseq + i
        bfac = 0.0 if bfactors is None else float(bfactors[i])
        for j, atom in enumerate(ATOM_ORDER):
            x, y, z = coords[i, j]
            if not np.isfinite([x, y, z]).all():
                continue  # missing density: omit the atom, never invent one
            lines.append(
                f"ATOM  {serial:>5d} {_format_atom_name(atom)} "
                f"{resname:>3s} {chain_id}{resseq:>4d}    {x:>8.3f}"
                f"{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{bfac:>6.2f}          "
                f"{ATOM_ELEMENT[atom]:>2s}"
            )
            serial += 1
    lines.append(
        f"TER   {serial:>5d}      {AA1_TO_3[sequence[-1].upper()]:>3s} "
        f"{chain_id}{start_resseq + len(sequence) - 1:>4d}"
    )
    lines.append("END")

    path.write_text("\n".join(lines) + "\n")
    return path


def read_backbone_pdb(path):
    """Read back one chain as (sequence, coords[L,4,3])."""
    aa3_to_1 = {v: k for k, v in AA1_TO_3.items()}
    residues: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    names: dict[tuple[str, int], str] = {}
    order: list[tuple[str, int]] = []

    for line in Path(path).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        if atom not in ATOM_ORDER:
            continue
        key = (line[21], int(line[22:26]))
        if key not in residues:
            residues[key] = {}
            names[key] = line[17:20].strip()
            order.append(key)
        residues[key][atom] = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        )

    sequence = "".join(aa3_to_1.get(names[k], "X") for k in order)
    coords = np.full((len(order), 4, 3), np.nan)
    for i, k in enumerate(order):
        for j, atom in enumerate(ATOM_ORDER):
            if atom in residues[k]:
                coords[i, j] = residues[k][atom]
    return sequence, coords


def write_backbone_set(
    out_dir,
    coords_list: Sequence[np.ndarray],
    sequences: Sequence[str],
    names: Sequence[str],
    **kwargs,
) -> list[Path]:
    """Write many chains as individual PDB files, one per name."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for coords, seq, name in zip(coords_list, sequences, names, strict=True):
        safe = str(name).replace("/", "_").replace(" ", "_")
        written.append(
            write_backbone_pdb(out_dir / f"{safe}.pdb", coords, seq, **kwargs)
        )
    return written
