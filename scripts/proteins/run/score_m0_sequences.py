"""Score M0 sequence samples with compact ESM-2 and design sanity metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from evaluation.proteins.io import atomic_json, read_fasta


HYDROPHOBIC = set("AVILMFWY")
POSITIVE = set("KR")
NEGATIVE = set("DE")


def _sanity(sequence: str) -> dict:
    counts = Counter(sequence); length = len(sequence); probabilities = np.asarray(list(counts.values()), dtype=float) / length
    longest = max((len(run) for aa in set(sequence) for run in sequence.split(aa) if False), default=0)
    # Explicit homopolymer scan avoids a regex dependency and handles every residue.
    current = best = 1
    for left, right in zip(sequence, sequence[1:]):
        current = current + 1 if left == right else 1; best = max(best, current)
    return {
        "entropy_bits": float(-np.sum(probabilities * np.log2(probabilities))),
        "hydrophobic_fraction": sum(aa in HYDROPHOBIC for aa in sequence) / length,
        "charge_proxy_per_residue": (sum(aa in POSITIVE for aa in sequence) - sum(aa in NEGATIVE for aa in sequence)) / length,
        "cysteine_fraction": sequence.count("C") / length,
        "longest_homopolymer": int(best if sequence else 0),
    }


@torch.no_grad()
def _pppl(sequences: list[str], device: torch.device, mask_batch: int) -> list[float]:
    import esm
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D(); model = model.to(device).eval(); converter = alphabet.get_batch_converter(); values = []
    for number, sequence in enumerate(sequences, start=1):
        _, _, tokens = converter([("protein", sequence)]); tokens = tokens.to(device); losses = []
        for start in range(0, len(sequence), mask_batch):
            positions = torch.arange(start + 1, min(len(sequence), start + mask_batch) + 1, device=device)
            masked = tokens.repeat(len(positions), 1); masked[torch.arange(len(positions), device=device), positions] = alphabet.mask_idx
            logits = model(masked, repr_layers=[], return_contacts=False)["logits"]
            target = tokens[0, positions]; row = torch.arange(len(positions), device=device)
            losses.append(-torch.log_softmax(logits[row, positions].float(), -1)[row, target])
        values.append(float(torch.cat(losses).mean().exp().cpu())); print(f"ESM2 [{number:02d}/{len(sequences)}] L={len(sequence)} pppl={values[-1]:.2f}", flush=True)
    return values


def _summary(values) -> dict:
    x = np.asarray(values, dtype=float); return {"mean": float(x.mean()), "median": float(np.median(x)), "std": float(x.std()), "min": float(x.min()), "max": float(x.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, default=Path("runs/proteins/m0_v1/protein_eval/dashboard/sequence_marginal.fasta"))
    parser.add_argument("--reference", type=Path, default=Path("runs/proteins/m0_v1/protein_eval/dashboard/test_references.fasta"))
    parser.add_argument("--out", type=Path, default=Path("runs/proteins/m0_v1/protein_eval/dashboard/sequence_judge_metrics.json"))
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--mask-batch", type=int, default=64)
    args = parser.parse_args(); generated = read_fasta(args.generated); references = read_fasta(args.reference); device = torch.device(args.device)
    gen_pppl = _pppl(generated, device, args.mask_batch); ref_pppl = _pppl(references, device, args.mask_batch)
    gen_sanity = [_sanity(sequence) for sequence in generated]; ref_sanity = [_sanity(sequence) for sequence in references]
    keys = gen_sanity[0].keys()
    payload = {
        "judge": "ESM-2 t6 8M masked pseudo-perplexity (compact exploratory judge)", "num_sequences": len(generated),
        "generated_pppl": _summary(gen_pppl), "reference_pppl": _summary(ref_pppl), "paired_pppl_ratio": float(np.mean(np.asarray(gen_pppl) / np.asarray(ref_pppl))),
        "generated_sanity": {key: _summary([row[key] for row in gen_sanity]) for key in keys},
        "reference_sanity": {key: _summary([row[key] for row in ref_sanity]) for key in keys},
        "per_sequence": {"generated_pppl": gen_pppl, "reference_pppl": ref_pppl},
        "limitations": ["8M-parameter ESM-2 is a compact screen, not the planned 650M judge", "pseudo-perplexity is not experimental fitness or designability"],
    }
    atomic_json(args.out, payload); print(json.dumps({key: payload[key] for key in ("generated_pppl", "reference_pppl", "paired_pppl_ratio")}, indent=2))


if __name__ == "__main__":
    main()
