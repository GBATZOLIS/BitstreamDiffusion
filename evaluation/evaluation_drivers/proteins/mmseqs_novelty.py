"""MMseqs2-GPU sequence-novelty annotation for generated proteins."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import MutableSequence
from pathlib import Path


def annotate_with_mmseqs(
    records: MutableSequence[dict],
    *,
    mmseqs_bin: str | Path,
    reference_fasta: str | Path,
    work_dir: str | Path,
    threads: int = 16,
    gpu: bool = True,
    coverage: float = 0.8,
    novelty_identity: float = 0.3,
) -> dict:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    query = work / "generated.fasta"
    hits = work / "nearest_training_hits.tsv"
    tmp = work / "tmp"
    with query.open("w") as handle:
        for record in records:
            handle.write(f">{record['sample_id']}\n{record['sequence']}\n")
    command = [
        str(mmseqs_bin),
        "easy-search",
        str(query),
        str(reference_fasta),
        str(hits),
        str(tmp),
        "--max-seqs",
        "1",
        "-s",
        "7.5",
        "-c",
        str(coverage),
        "--cov-mode",
        "0",
        "--alignment-mode",
        "3",
        "--threads",
        str(int(threads)),
        "--format-output",
        "query,target,fident,alnlen,qcov,tcov,evalue",
    ]
    if gpu:
        command.extend(["--createdb-mode", "2", "--gpu", "1"])
    subprocess.run(command, check=True)

    nearest: dict[str, dict] = {}
    for line in hits.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) < 7:
            continue
        identity = float(fields[2])
        if identity > 1.0:
            identity /= 100.0
        row = {
            "target": fields[1],
            "identity": identity,
            "alignment_length": int(fields[3]),
            "query_coverage": float(fields[4]),
            "target_coverage": float(fields[5]),
            "evalue": float(fields[6]),
        }
        previous = nearest.get(fields[0])
        if previous is None or row["identity"] > previous["identity"]:
            nearest[fields[0]] = row

    values = []
    for record in records:
        hit = nearest.get(str(record["sample_id"]))
        record["mmseqs_nearest_training"] = hit
        identity = hit["identity"] if hit is not None else None
        record["mmseqs_training_identity"] = identity
        record["sequence_novel_mmseqs"] = (
            identity < novelty_identity if identity is not None else None
        )
        if identity is not None and math.isfinite(identity):
            values.append(identity)

    summary = {
        "tool": "MMseqs2-GPU" if gpu else "MMseqs2",
        "reference_fasta": str(reference_fasta),
        "coverage_threshold": coverage,
        "novelty_identity_threshold": novelty_identity,
        "num_queries": len(records),
        "num_with_hit": len(values),
        "mean_nearest_identity": sum(values) / len(values) if values else None,
        "fraction_novel": (
            sum(value < novelty_identity for value in values) / len(values)
            if values
            else None
        ),
        "command": command,
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
