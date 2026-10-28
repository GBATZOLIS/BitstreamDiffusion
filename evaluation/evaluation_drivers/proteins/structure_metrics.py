"""Structure-evaluation primitives for generated protein backbones."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# Index of each backbone atom in the [L, 4, 3] layout.
_ATOM_N = 0
_ATOM_CA = 1
_ATOM_C = 2
_ATOM_O = 3

# Ideal bond lengths in Angstroms for a standard protein backbone.
_TARGET_N_CA = 1.458
_TARGET_CA_C = 1.525
_TARGET_C_N = 1.329
_TARGET_N_CA_C_ANGLE = 111.2
_TARGET_CA_C_N_ANGLE = 116.2
_TARGET_C_N_CA_ANGLE = 121.7

# Geometry thresholds in Angstroms.
_CLASH_DISTANCE = 2.0
_CHAIN_BREAK_DISTANCE = 4.5


def _as_ca(coords) -> np.ndarray:
    """
    Return a contiguous [L, 3] CA array from CA-only or full-backbone
    input.
    """
    array = np.ascontiguousarray(np.asarray(coords, dtype=np.float64))
    if array.ndim == 3 and array.shape[1] >= 2 and array.shape[2] == 3:
        return np.ascontiguousarray(array[:, _ATOM_CA, :])
    if array.ndim == 2 and array.shape[1] == 3:
        return array
    raise ValueError(
        f"Expected CA coordinates of shape [L, 3] or backbone coordinates "
        f"of shape [L, 4, 3], received array of shape {array.shape}."
    )


def _check_pair(a: np.ndarray, b: np.ndarray) -> None:
    """Validate that two point clouds have matching [L, 3] shapes."""
    if a.shape != b.shape:
        raise ValueError(
            f"Coordinate arrays must have equal shape, received {a.shape} "
            f"and {b.shape}."
        )
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(
            f"Expected point clouds of shape [L, 3], received {a.shape}."
        )
    if a.shape[0] == 0:
        raise ValueError("Coordinate arrays must contain at least one point.")


def _superpose(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Kabsch-superpose mobile onto reference and return the rotated mobile
    points.
    """
    mobile_centroid = mobile.mean(axis=0)
    reference_centroid = reference.mean(axis=0)
    centered_mobile = mobile - mobile_centroid
    centered_reference = reference - reference_centroid
    covariance = centered_mobile.T @ centered_reference
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, sign])
    rotation = vt.T @ correction @ u.T
    return centered_mobile @ rotation.T + reference_centroid


def rmsd(coords_a, coords_b, superpose: bool = True) -> float:
    """
    Root-mean-square deviation between two matched CA arrays of shape
    [L, 3].
    """
    a = np.ascontiguousarray(np.asarray(coords_a, dtype=np.float64))
    b = np.ascontiguousarray(np.asarray(coords_b, dtype=np.float64))
    _check_pair(a, b)
    if superpose:
        a = _superpose(a, b)
    difference = a - b
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))


def _tm_d0(length: int) -> float:
    """
    Return the standard TM-score length-normalization distance d0 in
    Angstroms.
    """
    d0 = 1.24 * np.cbrt(float(length) - 15.0) - 1.8
    return float(max(d0, 0.5))


def _tm_score_fallback(a: np.ndarray, b: np.ndarray) -> float:
    """
    Kabsch-superposed TM-score approximation normalized by the length of
    b.
    """
    _check_pair(a, b)
    aligned = _superpose(a, b)
    distances = np.linalg.norm(aligned - b, axis=1)
    d0 = _tm_d0(b.shape[0])
    return float(np.mean(1.0 / (1.0 + (distances / d0) ** 2)))


def tm_score(ca_a, ca_b) -> float:
    """TM-score between two CA arrays, normalized by the length of ca_b."""
    a = np.ascontiguousarray(np.asarray(ca_a, dtype=np.float64))
    b = np.ascontiguousarray(np.asarray(ca_b, dtype=np.float64))
    if a.ndim != 2 or a.shape[1] != 3 or b.ndim != 2 or b.shape[1] != 3:
        raise ValueError("tm_score expects CA arrays of shape [L, 3].")
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise ValueError("tm_score requires non-empty CA arrays.")
    try:
        from tmtools import tm_align
    except ImportError:
        return _tm_score_fallback(a, b)
    result = tm_align(a, b, "A" * a.shape[0], "A" * b.shape[0])
    value = getattr(result, "tm_norm_chain2", None)
    if value is None:
        value = getattr(result, "tm_norm_chan2", None)
    if value is None:
        raise RuntimeError(
            "tmtools returned a result without a chain-2 normalized "
            "TM-score; the installed tmtools version is not compatible. "
            "Reinstall tmtools>=0.2 from the eval extra (uv sync --extra"
            " eval)."
        )
    return float(value)


def lddt(
    coords_pred,
    coords_true,
    cutoff: float = 15.0,
    thresholds: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
) -> float:
    """Local Distance Difference Test over CA distances, pure numpy."""
    pred = _as_ca(coords_pred)
    true = _as_ca(coords_true)
    _check_pair(pred, true)
    length = true.shape[0]
    if length < 2:
        return 1.0
    dist_pred = np.linalg.norm(pred[:, None, :] - pred[None, :, :], axis=-1)
    dist_true = np.linalg.norm(true[:, None, :] - true[None, :, :], axis=-1)
    included = (dist_true < cutoff) & ~np.eye(length, dtype=bool)
    if not included.any():
        return 0.0
    delta = np.abs(dist_pred - dist_true)
    preserved = np.zeros_like(delta)
    for threshold in thresholds:
        preserved += (delta < threshold).astype(np.float64)
    preserved /= float(len(thresholds))
    return float(preserved[included].mean())


def backbone_geometry(coords) -> dict:
    """
    Report backbone bond-length, clash, chain-break, and chirality
    diagnostics.
    """
    array = np.ascontiguousarray(np.asarray(coords, dtype=np.float64))
    if array.ndim != 3 or array.shape[1] < 4 or array.shape[2] != 3:
        raise ValueError(
            f"backbone_geometry expects coordinates of shape [L, 4, 3] "
            f"with atom order N, CA, C, O, received array of shape "
            f"{array.shape}."
        )
    length = array.shape[0]
    nitrogen = array[:, _ATOM_N, :]
    alpha = array[:, _ATOM_CA, :]
    carbon = array[:, _ATOM_C, :]

    n_ca = np.linalg.norm(alpha - nitrogen, axis=1)
    ca_c = np.linalg.norm(carbon - alpha, axis=1)
    c_n = (
        np.linalg.norm(nitrogen[1:] - carbon[:-1], axis=1)
        if length > 1
        else np.zeros(0)
    )

    def _angles(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        if not len(a):
            return np.zeros(0, dtype=np.float64)
        left = a - b
        right = c - b
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(
            right, axis=1
        )
        cosine = np.sum(left * right, axis=1) / np.maximum(denominator, 1e-12)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    n_ca_c_angle = _angles(nitrogen, alpha, carbon)
    ca_c_n_angle = (
        _angles(alpha[:-1], carbon[:-1], nitrogen[1:])
        if length > 1
        else np.zeros(0)
    )
    c_n_ca_angle = (
        _angles(carbon[:-1], nitrogen[1:], alpha[1:])
        if length > 1
        else np.zeros(0)
    )

    def _dihedrals(
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
    ) -> np.ndarray:
        if not len(p0):
            return np.zeros(0, dtype=np.float64)
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1_unit = b1 / np.maximum(
            np.linalg.norm(b1, axis=1, keepdims=True), 1e-12
        )
        v = b0 - np.sum(b0 * b1_unit, axis=1, keepdims=True) * b1_unit
        w = b2 - np.sum(b2 * b1_unit, axis=1, keepdims=True) * b1_unit
        x = np.sum(v * w, axis=1)
        y = np.sum(np.cross(b1_unit, v) * w, axis=1)
        return np.degrees(np.arctan2(y, x))

    omega = (
        _dihedrals(alpha[:-1], carbon[:-1], nitrogen[1:], alpha[1:])
        if length > 1
        else np.zeros(0)
    )
    omega_planarity = 180.0 - np.abs(omega)

    def _summary(values: np.ndarray, target: float) -> dict:
        if values.size == 0:
            return {
                "mean": float("nan"),
                "target": target,
                "deviation": float("nan"),
            }
        return {
            "mean": float(values.mean()),
            "target": target,
            "deviation": float(np.mean(np.abs(values - target))),
        }

    omega_summary = {
        "mean_degrees": float(np.mean(omega)) if omega.size else float("nan"),
        "mean_planarity_deviation_degrees": (
            float(np.mean(np.abs(omega_planarity)))
            if omega.size
            else float("nan")
        ),
        "trans_fraction_30deg": (
            float(np.mean(np.abs(omega_planarity) <= 30.0))
            if omega.size
            else float("nan")
        ),
        "cis_like_count": (
            int(np.sum(np.abs(omega) < 30.0)) if omega.size else 0
        ),
    }

    clash_count = 0
    chain_break_count = 0
    if length > 1:
        ca_distances = np.linalg.norm(
            alpha[:, None, :] - alpha[None, :, :], axis=-1
        )
        separation = np.abs(
            np.arange(length)[:, None] - np.arange(length)[None, :]
        )
        clash_mask = (
            (ca_distances < _CLASH_DISTANCE)
            & (separation > 1)
            & np.triu(np.ones((length, length), dtype=bool), k=1)
        )
        clash_count = int(clash_mask.sum())
        consecutive = np.linalg.norm(alpha[1:] - alpha[:-1], axis=1)
        chain_break_count = int((consecutive > _CHAIN_BREAK_DISTANCE).sum())

    chirality_sign = 0.0
    chirality_consistency = float("nan")
    chirality_violations = 0
    if length > 1:
        n_vectors = nitrogen[:-1] - alpha[:-1]
        c_vectors = carbon[:-1] - alpha[:-1]
        step_vectors = alpha[1:] - alpha[:-1]
        signed_volume = np.sum(
            np.cross(n_vectors, c_vectors) * step_vectors, axis=1
        )
        signs = np.sign(signed_volume)
        nonzero = signs[signs != 0]
        if nonzero.size:
            positive = int((nonzero > 0).sum())
            negative = int((nonzero < 0).sum())
            majority = 1.0 if positive >= negative else -1.0
            chirality_sign = majority
            majority_count = max(positive, negative)
            chirality_consistency = majority_count / float(nonzero.size)
            chirality_violations = int(nonzero.size - majority_count)

    return {
        "length": int(length),
        "n_ca": _summary(n_ca, _TARGET_N_CA),
        "ca_c": _summary(ca_c, _TARGET_CA_C),
        "c_n": _summary(c_n, _TARGET_C_N),
        "bond_angles_degrees": {
            "n_ca_c": _summary(n_ca_c_angle, _TARGET_N_CA_C_ANGLE),
            "ca_c_n": _summary(ca_c_n_angle, _TARGET_CA_C_N_ANGLE),
            "c_n_ca": _summary(c_n_ca_angle, _TARGET_C_N_CA_ANGLE),
        },
        "omega": omega_summary,
        "clash_count": clash_count,
        "chain_break_count": chain_break_count,
        "chirality_sign": chirality_sign,
        "chirality_consistency": chirality_consistency,
        "chirality_violations": chirality_violations,
    }


def radius_of_gyration(ca) -> float:
    """Radius of gyration in Angstroms for a CA array of shape [L, 3]."""
    coords = _as_ca(ca)
    if coords.shape[0] == 0:
        raise ValueError(
            "radius_of_gyration requires at least one CA coordinate."
        )
    centered = coords - coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def _virtual_bond_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Return the angle at b, in degrees, spanned by the vectors b->a and
    b->c.
    """
    left = a - b
    right = c - b
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0.0:
        return float("nan")
    cosine = np.clip(np.dot(left, right) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _pseudo_dihedral(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> float:
    """
    Return the dihedral angle in degrees about the p1-p2 axis for four
    points.
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    norm_b1 = np.linalg.norm(b1)
    if norm_b1 == 0.0:
        return float("nan")
    b1 = b1 / norm_b1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def secondary_structure_labels(coords) -> list:
    """
    Approximate per-residue helix, sheet, and coil labels from CA
    geometry.
    """
    ca = _as_ca(coords)
    length = ca.shape[0]
    labels = ["coil"] * length
    for i in range(1, length - 2):
        tau = _virtual_bond_angle(ca[i - 1], ca[i], ca[i + 1])
        theta = _pseudo_dihedral(ca[i - 1], ca[i], ca[i + 1], ca[i + 2])
        if np.isnan(tau) or np.isnan(theta):
            continue
        if 70.0 <= tau <= 110.0 and 25.0 <= theta <= 95.0:
            labels[i] = "helix"
        elif 105.0 <= tau <= 150.0 and abs(theta) >= 120.0:
            labels[i] = "sheet"
    return labels


def secondary_structure_fractions(coords) -> dict:
    """Approximate helix, sheet, and coil fractions from CA geometry."""
    labels = secondary_structure_labels(coords)
    total = float(len(labels)) if labels else 1.0
    return {
        "helix": labels.count("helix") / total,
        "sheet": labels.count("sheet") / total,
        "coil": labels.count("coil") / total,
    }


_FOLDSEEK_HINT = (
    "Install Foldseek and place it on PATH. Releases are available at "
    "https://github.com/steineggerlab/foldseek."
)

_MMSEQS_HINT = (
    "Install MMseqs2 and place it on PATH. Releases are available at "
    "https://github.com/soedinglab/MMseqs2."
)


def _run_external_tool(
    executable: str, command: list, tool_name: str, install_hint: str
) -> None:
    """
    Run an external tool subprocess, raising an actionable error on
    failure.
    """
    location = shutil.which(executable)
    if location is None:
        raise RuntimeError(
            f"{tool_name} was not found on PATH. {install_hint}"
        )
    command = [location] + command[1:]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{tool_name} failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )


def _parse_cluster_tsv(tsv_path: Path) -> dict:
    """
    Parse a two-column representative/member cluster TSV into a summary
    dict.
    """
    if not tsv_path.exists():
        raise RuntimeError(
            f"Expected cluster output at {tsv_path} but the file was not "
            f"produced."
        )
    clusters: dict[str, list[str]] = {}
    with tsv_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            representative = parts[0]
            member = parts[1] if len(parts) > 1 else parts[0]
            clusters.setdefault(representative, []).append(member)
    sizes = sorted(
        (len(members) for members in clusters.values()), reverse=True
    )
    num_members = int(sum(sizes))
    singletons = int(sum(1 for size in sizes if size == 1))
    return {
        "num_members": num_members,
        "num_clusters": len(clusters),
        "cluster_sizes": sizes,
        "largest_cluster": sizes[0] if sizes else 0,
        "singleton_fraction": singletons / len(clusters) if clusters else 0.0,
        "clusters": clusters,
    }


def foldseek_cluster(pdb_dir, tmp_dir, min_tm: float = 0.5) -> dict:
    """
    Cluster PDB structures with foldseek easy-cluster and summarize the
    result.
    """
    pdb_path = Path(pdb_dir)
    tmp_path = Path(tmp_dir)
    if not pdb_path.is_dir():
        raise ValueError(f"pdb_dir {pdb_path} is not a directory.")
    tmp_path.mkdir(parents=True, exist_ok=True)
    result_prefix = tmp_path / "foldseek_cluster"
    work_dir = tmp_path / "foldseek_work"
    command = [
        "foldseek",
        "easy-cluster",
        str(pdb_path),
        str(result_prefix),
        str(work_dir),
        "--alignment-type",
        "1",
        "--tmscore-threshold",
        str(float(min_tm)),
    ]
    _run_external_tool(
        "foldseek",
        command,
        "foldseek",
        _FOLDSEEK_HINT,
    )
    return _parse_cluster_tsv(Path(str(result_prefix) + "_cluster.tsv"))


def mmseqs_cluster(fasta_path, tmp_dir, min_seq_id: float = 0.3) -> dict:
    """Cluster sequences with mmseqs easy-cluster and summarize the result."""
    fasta = Path(fasta_path)
    tmp_path = Path(tmp_dir)
    if not fasta.is_file():
        raise ValueError(f"fasta_path {fasta} is not a file.")
    tmp_path.mkdir(parents=True, exist_ok=True)
    result_prefix = tmp_path / "mmseqs_cluster"
    work_dir = tmp_path / "mmseqs_work"
    command = [
        "mmseqs",
        "easy-cluster",
        str(fasta),
        str(result_prefix),
        str(work_dir),
        "--min-seq-id",
        str(float(min_seq_id)),
        "-c",
        "0.8",
    ]
    _run_external_tool(
        "mmseqs",
        command,
        "mmseqs2",
        _MMSEQS_HINT,
    )
    return _parse_cluster_tsv(Path(str(result_prefix) + "_cluster.tsv"))


# Column names accepted by --format-output, taken from the installed
# binaries' own easy-search --help (foldseek 10-941cd33, mmseqs2
# 18-8cc5c). foldseek reports three differently normalized TM-scores:
# qtmscore over the query length, ttmscore over the target length, and
# alntmscore over the aligned region only.
_FOLDSEEK_SEARCH_COLUMNS = (
    "query",
    "target",
    "fident",
    "alnlen",
    "evalue",
    "bits",
    "qtmscore",
    "ttmscore",
    "alntmscore",
    "lddt",
    "rmsd",
)

_MMSEQS_SEARCH_COLUMNS = (
    "query",
    "target",
    "fident",
    "alnlen",
    "qcov",
    "tcov",
    "evalue",
    "bits",
)


def _parse_m8(
    path: Path, columns: Sequence[str], numeric: Sequence[str]
) -> list:
    """Parse a BLAST-tab (m8) hit table into a list of dicts."""
    if not path.exists():
        raise RuntimeError(
            f"Expected search output at {path} but the file was not produced."
        )
    hits = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < len(columns):
                continue
            row = dict(zip(columns, parts, strict=False))
            for key in numeric:
                try:
                    row[key] = float(row[key])
                except (TypeError, ValueError):
                    row[key] = float("nan")
            hits.append(row)
    return hits


def _best_hits(hits: Sequence, rank_by: str, exclude_self: bool) -> dict:
    """Reduce a hit table to the single best hit per query, keyed by query."""
    best: dict = {}
    for hit in hits:
        if exclude_self and hit["query"] == hit["target"]:
            continue
        score = hit.get(rank_by, float("nan"))
        if np.isnan(score):
            continue
        current = best.get(hit["query"])
        if current is None or score > current[rank_by]:
            best[hit["query"]] = hit
    return best


def _search_summary(
    hits: Sequence,
    rank_by: str,
    expected: Sequence[str] | None = None,
    exclude_self: bool = False,
) -> dict:
    """
    Summarize a search as one best hit per query plus aggregate
    statistics.
    """
    best = _best_hits(hits, rank_by, exclude_self)
    names = list(expected) if expected is not None else sorted(best)
    scores = [float(best[name][rank_by]) for name in names if name in best]
    without_hit = [name for name in names if name not in best]
    return {
        "rank_by": rank_by,
        "num_queries": len(names),
        "num_with_hit": len(scores),
        "num_without_hit": len(without_hit),
        "queries_without_hit": without_hit,
        "best_score_mean": float(np.mean(scores)) if scores else float("nan"),
        "best_score_median": (
            float(np.median(scores)) if scores else float("nan")
        ),
        "best_score_max": float(np.max(scores)) if scores else float("nan"),
        "best_score_min": float(np.min(scores)) if scores else float("nan"),
        "best_hits": best,
        "num_hits": len(hits),
    }


def _structure_names(pdb_dir: Path) -> list:
    """
    Return the stem of every structure file foldseek will read from a
    dir.
    """
    suffixes = {".pdb", ".cif", ".ent", ".mmcif", ".gz"}
    return sorted(
        path.stem
        for path in pdb_dir.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _fasta_names(fasta_path: Path) -> list:
    """Return the identifier of every record in a FASTA file."""
    names = []
    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                names.append(
                    line[1:].strip().split()[0] if line[1:].strip() else ""
                )
    return names


def foldseek_search(
    query_dir,
    target_dir,
    tmp_dir,
    min_tm: float = 0.0,
    max_seqs: int = 1000,
    exact_tmscore: bool = False,
) -> dict:
    """
    Search structures against a structure database with foldseek
    easy-search.
    """
    query_path = Path(query_dir)
    target_path = Path(target_dir)
    tmp_path = Path(tmp_dir)
    if not query_path.is_dir():
        raise ValueError(f"query_dir {query_path} is not a directory.")
    if not target_path.is_dir():
        raise ValueError(f"target_dir {target_path} is not a directory.")
    tmp_path.mkdir(parents=True, exist_ok=True)
    result_path = tmp_path / "foldseek_search.m8"
    work_dir = tmp_path / "foldseek_search_work"
    command = [
        "foldseek",
        "easy-search",
        str(query_path),
        str(target_path),
        str(result_path),
        str(work_dir),
        "--alignment-type",
        "1",
        "--tmscore-threshold",
        str(float(min_tm)),
        "--max-seqs",
        str(int(max_seqs)),
        "--format-output",
        ",".join(_FOLDSEEK_SEARCH_COLUMNS),
    ]
    if exact_tmscore:
        command += ["--exact-tmscore", "1"]
    _run_external_tool("foldseek", command, "foldseek", _FOLDSEEK_HINT)
    numeric = _FOLDSEEK_SEARCH_COLUMNS[2:]
    hits = _parse_m8(result_path, _FOLDSEEK_SEARCH_COLUMNS, numeric)
    # foldseek names queries per chain, not per file: a two-chain
    # 3ixt.pdb becomes the queries 3ixt_A and 3ixt_B. The per-query
    # bookkeeping is therefore chain-level and comes from the hit table,
    # and file coverage is reported separately by matching each chain
    # name back to its file stem.
    summary = _search_summary(hits, "qtmscore")
    stems = _structure_names(query_path)
    hit_queries = set(summary["best_hits"])
    summary["num_query_files"] = len(stems)
    summary["query_files_without_hit"] = [
        stem
        for stem in stems
        if not any(name.startswith(stem) for name in hit_queries)
    ]
    summary["tool"] = "foldseek easy-search"
    summary["result_path"] = str(result_path)
    return summary


def mmseqs_search(
    query_fasta,
    target_fasta,
    tmp_dir,
    min_seq_id: float = 0.0,
    evalue: float = 10.0,
    sensitivity: float = 7.5,
    max_seqs: int = 1000,
) -> dict:
    """Search sequences against a sequence database with mmseqs easy-search."""
    query_path = Path(query_fasta)
    target_path = Path(target_fasta)
    tmp_path = Path(tmp_dir)
    if not query_path.is_file():
        raise ValueError(f"query_fasta {query_path} is not a file.")
    if not target_path.is_file():
        raise ValueError(f"target_fasta {target_path} is not a file.")
    tmp_path.mkdir(parents=True, exist_ok=True)
    result_path = tmp_path / "mmseqs_search.m8"
    work_dir = tmp_path / "mmseqs_search_work"
    command = [
        "mmseqs",
        "easy-search",
        str(query_path),
        str(target_path),
        str(result_path),
        str(work_dir),
        "--min-seq-id",
        str(float(min_seq_id)),
        "-e",
        str(float(evalue)),
        "-s",
        str(float(sensitivity)),
        "--max-seqs",
        str(int(max_seqs)),
        "--format-output",
        ",".join(_MMSEQS_SEARCH_COLUMNS),
    ]
    _run_external_tool("mmseqs", command, "mmseqs2", _MMSEQS_HINT)
    numeric = _MMSEQS_SEARCH_COLUMNS[2:]
    hits = _parse_m8(result_path, _MMSEQS_SEARCH_COLUMNS, numeric)
    # Unlike foldseek, mmseqs keeps the FASTA identifier verbatim, so
    # the full query set is known here and a sequence with no hit can be
    # named.
    summary = _search_summary(hits, "fident", _fasta_names(query_path))
    summary["tool"] = "mmseqs easy-search"
    summary["result_path"] = str(result_path)
    return summary


def nearest_training_tm(query_ca, training_cas: Sequence) -> float:
    """
    Return the maximum TM-score of a query CA structure to a set of
    references.
    """
    references = list(training_cas)
    if not references:
        raise ValueError(
            "nearest_training_tm requires at least one reference structure."
        )
    return max(
        float(tm_score(query_ca, reference)) for reference in references
    )


def _self_check() -> None:
    """
    Build two random CA arrays and print rmsd, tm-score, and lddt to
    prove import.
    """
    rng = np.random.default_rng(0)
    length = 40
    reference = np.cumsum(rng.normal(scale=3.8, size=(length, 3)), axis=0)
    perturbed = reference + rng.normal(scale=0.5, size=(length, 3))
    print(f"rmsd={rmsd(perturbed, reference):.4f}")
    print(f"tm_score={tm_score(perturbed, reference):.4f}")
    print(f"lddt={lddt(perturbed, reference):.4f}")


if __name__ == "__main__":
    _self_check()
