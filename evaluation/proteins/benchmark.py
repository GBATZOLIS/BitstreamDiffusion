"""Frozen, model-agnostic EvoDiff UniRef50 generation benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .io import (
    EvoDiffReferenceStore,
    atomic_json,
    read_fasta,
    sha256,
    write_fasta,
)
from .metrics import (
    basic_metrics,
    cached_array,
    esm2_pppl_scores,
    esmfold_plddt_scores,
    frechet_distance,
    mmd_rbf,
    prot_t5_embeddings,
)
from .sampling import stratified_prefix_indices


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def _summaries_by_length(lengths: np.ndarray, values: np.ndarray) -> dict:
    return {
        str(length): float(values[lengths == length].mean())
        for length in sorted(set(lengths.tolist()))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Stable label used in tables"
    )
    parser.add_argument("--generated-fasta", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/uniref50_evodiff_2020"),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["basic", "esm2", "prot_t5", "esmfold"],
        default=["basic"],
    )
    parser.add_argument("--expected-samples", type=int, default=0)
    parser.add_argument("--esm2-samples", type=int, default=500)
    parser.add_argument("--esmfold-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    generated = read_fasta(args.generated_fasta)
    if args.expected_samples and len(generated) != args.expected_samples:
        raise ValueError(
            f"Expected {args.expected_samples} sequences, found {len(generated)}"
        )
    if not generated:
        raise ValueError("Generated FASTA is empty")
    lengths = np.asarray(
        [len(sequence) for sequence in generated], dtype=np.int64
    )
    store = EvoDiffReferenceStore(args.dataset_root)
    references = store.length_matched(lengths, split="rtest", seed=args.seed)

    out = args.out or args.generated_fasta.parent / "evodiff_benchmark.json"
    cache_dir = out.parent / "metric_cache"
    gen_hash = sha256(args.generated_fasta)
    reference_fasta = (
        cache_dir / f"rtest_matched_{gen_hash[:16]}_seed{args.seed}.fasta"
    )
    if not reference_fasta.exists():
        write_fasta(reference_fasta, references, "rtest")
    reference_hash = sha256(reference_fasta)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    results = basic_metrics(generated, references)
    by_length = {
        str(length): basic_metrics(
            [seq for seq in generated if len(seq) == length],
            [seq for seq in references if len(seq) == length],
        )
        for length in sorted(set(lengths.tolist()))
    }

    if "esm2" in args.metrics:
        count = min(args.esm2_samples, len(generated))
        selected_indices = stratified_prefix_indices(lengths, count)
        selected_generated = [generated[i] for i in selected_indices]
        selected_references = [references[i] for i in selected_indices]
        selected_lengths = lengths[selected_indices]
        gen_pppl = cached_array(
            cache_dir / f"{gen_hash}.n{count}.stratified.esm2_650m_pppl.npy",
            lambda: esm2_pppl_scores(selected_generated, device),
        )
        ref_pppl = cached_array(
            cache_dir
            / f"{reference_hash}.n{count}.stratified.esm2_650m_pppl.npy",
            lambda: esm2_pppl_scores(selected_references, device),
        )
        results["esm2_650m_pppl"] = float(gen_pppl.mean())
        results["esm2_650m_pppl_reference"] = float(ref_pppl.mean())
        results["esm2_650m_pppl_by_length"] = _summaries_by_length(
            selected_lengths, gen_pppl
        )

    if "prot_t5" in args.metrics:
        gen_embeddings = cached_array(
            cache_dir / f"{gen_hash}.prot_t5.npy",
            lambda: prot_t5_embeddings(generated, device),
        )
        ref_embeddings = cached_array(
            cache_dir / f"{reference_hash}.prot_t5.npy",
            lambda: prot_t5_embeddings(references, device),
        )
        results["prot_t5_frechet_distance"] = frechet_distance(
            gen_embeddings, ref_embeddings
        )
        results["prot_t5_mmd_rbf"] = mmd_rbf(gen_embeddings, ref_embeddings)
        results["prot_t5_by_length"] = {}
        for length in sorted(set(lengths.tolist())):
            mask = lengths == length
            results["prot_t5_by_length"][str(length)] = {
                "frechet_distance": frechet_distance(
                    gen_embeddings[mask], ref_embeddings[mask]
                ),
                "mmd_rbf": mmd_rbf(gen_embeddings[mask], ref_embeddings[mask]),
            }

    if "esmfold" in args.metrics:
        count = min(args.esmfold_samples, len(generated))
        selected_indices = stratified_prefix_indices(lengths, count)
        selected = [generated[i] for i in selected_indices]
        selected_lengths = lengths[selected_indices]
        plddt = cached_array(
            cache_dir / f"{gen_hash}.n{count}.stratified.esmfold_plddt.npy",
            lambda: esmfold_plddt_scores(selected, device),
        )
        results["esmfold_plddt"] = float(plddt.mean())
        results["esmfold_fraction_plddt_ge_70"] = float(np.mean(plddt >= 70.0))
        results["esmfold_plddt_by_length"] = _summaries_by_length(
            selected_lengths, plddt
        )

    manifest_path = args.dataset_root / "frozen_manifest.json"
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "EvoDiff UniRef50 March-2020; train/valid/rtest",
        "model": args.model,
        "generated_fasta": {
            "path": str(args.generated_fasta),
            "sha256": gen_hash,
        },
        "reference_fasta": {
            "path": str(reference_fasta),
            "sha256": reference_hash,
        },
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "metrics_requested": args.metrics,
        "sample_limits": {
            "all": len(generated),
            "esm2": min(args.esm2_samples, len(generated)),
            "esmfold": min(args.esmfold_samples, len(generated)),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
        },
        "results": results,
        "by_length": by_length,
    }
    atomic_json(out, payload)
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
