"""Prepare exact M0 training and generated-query FASTA files for MMseqs2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.protein_multimodal import CANONICAL_AA
from evaluation.proteins.io import atomic_json, read_fasta


SEEDS = (20260719, 20260720, 20260721, 20260722)


def _decode(ids: np.ndarray, mask: np.ndarray) -> str:
    return "".join(
        CANONICAL_AA[int(index)] if bool(valid) else "X"
        for index, valid in zip(ids, mask)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/dplm_paired_m0"))
    parser.add_argument("--eval-root", type=Path, default=Path("runs/proteins/m0_v2/protein_eval"))
    parser.add_argument(
        "--bundle-name-template",
        default="m0_v2_b_best_steps250_seed{seed}",
        help="Evaluation bundle directory template below --eval-root.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    training_path = args.out_dir / "training_sequences.fasta"
    training_count = 0
    with training_path.open("w", encoding="utf-8") as output:
        for shard_path in sorted(args.dataset.glob("shard_*.npz")):
            metadata_path = shard_path.with_suffix(".meta.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            archive = np.load(shard_path, allow_pickle=False)
            offsets = archive["offsets"]
            for row_index, row in enumerate(metadata["rows"]):
                if row.get("split") != "train":
                    continue
                start, end = int(offsets[row_index]), int(offsets[row_index + 1])
                sequence = _decode(
                    archive["seq_ids"][start:end], archive["seq_mask"][start:end]
                )
                output.write(f">train_{training_count} stable_id={row['stable_id']}\n{sequence}\n")
                training_count += 1

    query_path = args.out_dir / "generated_queries.fasta"
    queries = []
    with query_path.open("w", encoding="utf-8") as output:
        for seed in SEEDS:
            bundle = args.eval_root / args.bundle_name_template.format(seed=seed)
            marginal = read_fasta(bundle / "sequence_marginal.fasta")
            inverse = json.loads((bundle / "inverse_predictions.json").read_text())
            for index, sequence in enumerate(marginal):
                query_id = f"marginal_s{seed}_t{index:02d}"
                output.write(f">{query_id}\n{sequence}\n")
                queries.append({"query": query_id, "kind": "sequence_marginal", "seed": seed, "target": index, "length": len(sequence)})
            for record in inverse["records"]:
                samples = record["conditions"]["conditioned"]
                best = max(samples, key=lambda item: item["recovery"])
                index = int(record["target_position"])
                query_id = f"inverse_s{seed}_t{index:02d}"
                output.write(f">{query_id}\n{best['sequence']}\n")
                queries.append({"query": query_id, "kind": "inverse_conditioned_best_of_4", "seed": seed, "target": index, "length": len(best["sequence"]), "recovery": best["recovery"]})

    atomic_json(
        args.out_dir / "mmseqs_input_manifest.json",
        {
            "schema_version": 1,
            "training_sequences": training_count,
            "training_split": "metadata split == train",
            "training_fasta": str(training_path),
            "query_fasta": str(query_path),
            "queries": queries,
            "note": "Masked non-canonical training residues are represented as X.",
        },
    )
    print(f"Wrote {training_count} training sequences and {len(queries)} queries")


if __name__ == "__main__":
    main()
