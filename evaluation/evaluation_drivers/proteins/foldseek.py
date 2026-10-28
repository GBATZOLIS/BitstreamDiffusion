"""
Foldseek-backed PDB novelty and structural clustering for generated
proteins.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Iterable, MutableMapping, Sequence
from pathlib import Path


def validate_foldseek(foldseek_bin, database) -> tuple[Path, Path]:
    """Validate the executable and an extracted Foldseek database prefix."""
    binary = Path(foldseek_bin)
    database = Path(database)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(
            f"Foldseek executable is missing or not executable: {binary}"
        )
    dbtype = Path(f"{database}.dbtype")
    if not dbtype.is_file():
        raise FileNotFoundError(
            f"Foldseek database prefix {database} is incomplete; missing "
            f"{dbtype}"
        )
    return binary.resolve(), database.resolve()


def _sample_id(record: MutableMapping[str, object]) -> str:
    explicit = record.get("sample_id")
    if explicit:
        return str(explicit)
    return f"L{int(record['length'])}_{int(record['index']):03d}"


def _parse_hits(lines: Iterable[str]) -> dict[str, dict]:
    """Keep the maximum query-normalised TM hit for each query."""
    best: dict[str, dict] = {}
    for line in lines:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 6:
            continue
        query, target = Path(fields[0]).stem, fields[1]
        qtm, ttm, aln_tm, evalue = map(float, fields[2:6])
        candidate = {
            "target": target,
            "query_tm": qtm,
            "target_tm": ttm,
            "alignment_tm": aln_tm,
            "evalue": evalue,
        }
        if query not in best or qtm > float(best[query]["query_tm"]):
            best[query] = candidate
    return best


def _parse_clusters(lines: Iterable[str]) -> dict[str, str]:
    """Map every member id to its representative id from *_cluster.tsv."""
    clusters: dict[str, str] = {}
    for line in lines:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 2:
            continue
        representative, member = Path(fields[0]).stem, Path(fields[1]).stem
        clusters[member] = representative
    return clusters


def annotate_with_foldseek(
    records: Sequence[MutableMapping[str, object]],
    *,
    foldseek_bin,
    database,
    structures_dir,
    work_dir,
    novelty_tm_threshold: float = 0.5,
    cluster_tm_threshold: float = 0.5,
    threads: int = 16,
) -> dict:
    """
    Annotate records with nearest-PDB qTM and Foldseek cluster
    membership.
    """
    binary, database = validate_foldseek(foldseek_bin, database)
    structures_dir = Path(structures_dir).resolve()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    pdbs = sorted(structures_dir.glob("*.pdb"))
    if len(pdbs) != len(records):
        raise RuntimeError(
            f"Foldseek expected one PDB per record: {len(records)} records"
            f" but found {len(pdbs)} files in {structures_dir}"
        )

    with tempfile.TemporaryDirectory(prefix="foldseek-", dir=work_dir) as temp:
        temp_path = Path(temp)
        hits_path = temp_path / "pdb100_hits.tsv"
        subprocess.run(
            [
                str(binary),
                "easy-search",
                str(structures_dir),
                str(database),
                str(hits_path),
                str(temp_path / "search_tmp"),
                "--format-output",
                "query,target,qtmscore,ttmscore,alntmscore,evalue",
                "--max-seqs",
                "1000",
                "-s",
                "9.5",
                "--threads",
                str(int(threads)),
            ],
            check=True,
        )
        hit_text = hits_path.read_text(encoding="utf-8")
        hits = _parse_hits(hit_text.splitlines())

        cluster_prefix = temp_path / "generated"
        subprocess.run(
            [
                str(binary),
                "easy-cluster",
                str(structures_dir),
                str(cluster_prefix),
                str(temp_path / "cluster_tmp"),
                "--alignment-type",
                "1",
                "--cov-mode",
                "0",
                "--min-seq-id",
                "0",
                "--tmscore-threshold",
                str(float(cluster_tm_threshold)),
                "--threads",
                str(int(threads)),
            ],
            check=True,
        )
        cluster_path = Path(f"{cluster_prefix}_cluster.tsv")
        cluster_text = cluster_path.read_text(encoding="utf-8")
        clusters = _parse_clusters(cluster_text.splitlines())

    final_hits = work_dir / "pdb100_hits.tsv"
    final_clusters = work_dir / "generated_cluster.tsv"
    final_hits.write_text(hit_text, encoding="utf-8")
    final_clusters.write_text(cluster_text, encoding="utf-8")

    for record in records:
        sample_id = _sample_id(record)
        record["sample_id"] = sample_id
        hit = hits.get(sample_id)
        qtm = 0.0 if hit is None else float(hit["query_tm"])
        record["pdb_foldseek_query_tm"] = qtm
        record["pdb_nearest_id"] = None if hit is None else hit["target"]
        record["pdb_nearest_target_tm"] = (
            None if hit is None else float(hit["target_tm"])
        )
        record["pdb_nearest_alignment_tm"] = (
            None if hit is None else float(hit["alignment_tm"])
        )
        # Keep the generic field for the existing novelty
        # curves/headline, but make the method explicit in every report
        # and in the dedicated fields.
        record["structure_novelty_tm"] = qtm
        record["novel"] = qtm < float(novelty_tm_threshold)
        record["foldseek_cluster_id"] = clusters.get(sample_id, sample_id)
        record["designable_novel_diverse"] = (
            bool(record.get("designable"))
            and bool(record["novel"])
            and bool(record.get("diverse"))
        )

    cluster_sizes = {}
    for record in records:
        cluster_id = str(record["foldseek_cluster_id"])
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
    largest_cluster_size = max(cluster_sizes.values(), default=0)
    return {
        "method": "foldseek_easy_search_query_normalized_tm_and_easy_cluster",
        "database": str(database),
        "num_queries": len(records),
        "num_queries_with_hits": len(hits),
        "novelty_tm_threshold": float(novelty_tm_threshold),
        "cluster_tm_threshold": float(cluster_tm_threshold),
        "num_clusters": len(cluster_sizes),
        "cluster_fraction": (len(cluster_sizes) / len(records))
        if records
        else None,
        "largest_cluster_size": largest_cluster_size,
        "largest_cluster_fraction": (
            largest_cluster_size / len(records) if records else None
        ),
        "hits_tsv": str(final_hits),
        "clusters_tsv": str(final_clusters),
    }
