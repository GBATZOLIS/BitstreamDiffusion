"""Export auditable per-residue predictions for the M0 inverse-folding test.

This focused evaluator replays the 3D->sequence part of
``evaluate_m0_dashboard.py`` with the same target selection and random seeds.
Unlike the dashboard evaluator, it preserves every sampled sequence and a
per-residue recovery mask so aggregate amino-acid recovery can be audited.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from data.protein_multimodal import CANONICAL_AA, ProteinMultimodalDataset
from evaluation.proteins.generate_multimodal import generate
from evaluation.proteins.io import atomic_json, load_binary_protein_model, write_fasta


def _sequence(ids: np.ndarray) -> str:
    return "".join(CANONICAL_AA[int(index)] for index in ids)


def _select(lengths: np.ndarray, count: int) -> np.ndarray:
    order = np.argsort(lengths, kind="stable")
    if count >= len(order):
        return order
    positions = np.linspace(0, len(order) - 1, count).round().astype(int)
    return order[positions]


def _match_mask(prediction: str, reference: str) -> list[int]:
    if len(prediction) != len(reference):
        raise ValueError("prediction and reference lengths differ")
    return [int(left == right) for left, right in zip(prediction, reference)]


def _normalized_profile(records: list[dict], bins: int = 20) -> list[dict]:
    totals = np.zeros(bins, dtype=np.int64)
    matches = np.zeros(bins, dtype=np.int64)
    for record in records:
        length = int(record["length"])
        for sample in record["conditions"]["conditioned"]:
            for position, matched in enumerate(sample["match_mask"]):
                index = min(bins - 1, int(position * bins / max(1, length)))
                totals[index] += 1
                matches[index] += int(matched)
    return [
        {
            "bin": index,
            "normalized_start": index / bins,
            "normalized_end": (index + 1) / bins,
            "positions": int(totals[index]),
            "recovery": float(matches[index] / totals[index]) if totals[index] else None,
        }
        for index in range(bins)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/proteins/m0_v3.py")
    parser.add_argument("--checkpoint", default="runs/proteins/m0_v2/checkpoints/best.pt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--targets", type=int, default=32)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = ProteinMultimodalDataset(
        "datasets/dplm_paired_m0", split="test", min_len=40, max_len=512
    )
    selected = _select(dataset.lengths, min(args.targets, len(dataset)))
    rows = [dataset[int(index)] for index in selected]
    model, cfg = load_binary_protein_model(
        args.config,
        args.checkpoint,
        args.device,
        num_steps=args.steps,
        context="M0 inverse prediction export",
    )
    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    fasta_sequences: list[str] = []

    for target_position, row in enumerate(rows):
        length = int(row["length"])
        reference = _sequence(np.asarray(row["seq_ids"]))
        structure = np.asarray(row["struct_index"], dtype=np.uint16)
        conditions = {
            "conditioned": structure,
            "shuffled": rng.permutation(structure),
            "unconditional": None,
        }
        record = {
            "target_position": target_position,
            "dataset_index": int(selected[target_position]),
            "length": length,
            "reference_sequence": reference,
            "conditions": {},
        }
        print(f"[{target_position + 1:02d}/{len(rows)}] L={length}", flush=True)
        for offset, (condition, observed_structure) in enumerate(
            conditions.items(), start=2
        ):
            torch.manual_seed(args.seed + target_position * 100 + offset)
            task = (
                "inverse_folding"
                if observed_structure is not None
                else "sequence_marginal"
            )
            observed = (
                {"struct_index": observed_structure}
                if observed_structure is not None
                else None
            )
            generated = generate(
                model,
                cfg,
                task,
                length,
                args.samples,
                args.device,
                observed=observed,
            )["seq_strings"]
            samples = []
            for sample_index, sequence in enumerate(generated):
                mask = _match_mask(sequence, reference)
                samples.append(
                    {
                        "sample_index": sample_index,
                        "sequence": sequence,
                        "recovery": float(np.mean(mask)),
                        "match_mask": mask,
                        "matched_positions": [
                            index for index, matched in enumerate(mask) if matched
                        ],
                    }
                )
                fasta_sequences.append(sequence)
            record["conditions"][condition] = samples
        records.append(record)

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "per-residue 3D-to-sequence audit on the dashboard target subset",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "steps": args.steps,
        "samples_per_target": args.samples,
        "dataset_indices": selected.tolist(),
        "records": records,
        "conditioned_normalized_position_profile": _normalized_profile(records),
        "limitations": [
            "Reference structures are frozen LFQ tokens from the CAMEO-labelled cache.",
            "This export measures residue recovery, not experimental fitness.",
        ],
    }
    atomic_json(args.out_dir / "inverse_predictions.json", payload)
    write_fasta(
        args.out_dir / "inverse_predictions.fasta", fasta_sequences, "m0_inverse"
    )
    print(f"Saved {len(fasta_sequences)} predictions to {args.out_dir}")


if __name__ == "__main__":
    main()
