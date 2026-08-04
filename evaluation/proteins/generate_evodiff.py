"""Generate the frozen length grid from official EvoDiff checkpoints."""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .io import atomic_json, sha256, write_fasta

MODEL_LOADERS = {
    "oa_dm_38m": "OA_DM_38M",
    "d3pm_uniform_38m": "D3PM_UNIFORM_38M",
    "d3pm_blosum_38m": "D3PM_BLOSUM_38M",
    "lr_ar_38m": "LR_AR_38M",
}


@torch.no_grad()
def _fixed_length_ar(
    model, tokenizer, length: int, count: int, device
) -> list[str]:
    sample = torch.full(
        (count, 1), tokenizer.start_id, dtype=torch.long, device=device
    )
    timestep = torch.zeros(count, dtype=torch.long, device=device)
    for _ in range(length):
        logits = model(sample, timestep)[:, -1, : len(tokenizer.all_aas)]
        next_token = torch.multinomial(torch.softmax(logits, -1), 1)
        sample = torch.cat((sample, next_token), dim=1)
    return [tokenizer.untokenize(row[1:]) for row in sample]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=sorted(MODEL_LOADERS), required=True
    )
    parser.add_argument(
        "--lengths", nargs="+", type=int, default=[100, 200, 300, 400, 500]
    )
    parser.add_argument("--samples-per-length", type=int, default=400)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evodiff-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.evodiff_root is not None:
        os.chdir(args.evodiff_root)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    import evodiff.pretrained as pretrained
    from evodiff.generate import generate_d3pm, generate_oaardm

    loader = getattr(pretrained, MODEL_LOADERS[args.model])
    is_d3pm = args.model.startswith("d3pm")
    checkpoint = loader(return_all=True) if is_d3pm else loader()
    if is_d3pm:
        model, _collater, tokenizer, _scheme, timesteps, q_bar, q = checkpoint
    else:
        model, _collater, tokenizer, _scheme = checkpoint
    device = torch.device(args.device)
    model = model.to(device).eval()

    sequences = []
    for length in args.lengths:
        remaining = args.samples_per_length
        while remaining:
            count = min(remaining, args.micro_batch_size)
            if args.model == "lr_ar_38m":
                batch = _fixed_length_ar(
                    model, tokenizer, length, count, device
                )
            elif is_d3pm:
                _tokens, batch = generate_d3pm(
                    model,
                    tokenizer,
                    q,
                    q_bar,
                    timesteps,
                    length,
                    batch_size=count,
                    device=device,
                )
            else:
                _tokens, batch = generate_oaardm(
                    model, tokenizer, length, batch_size=count, device=device
                )
            sequences.extend(batch)
            remaining -= count

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fasta = args.out_dir / "generated.fasta"
    write_fasta(fasta, sequences, args.model)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "source": "official microsoft/evodiff checkpoint loader",
        "generation": vars(args) | {"out_dir": str(args.out_dir)},
        "fasta": {"path": str(fasta), "sha256": sha256(fasta)},
    }
    atomic_json(args.out_dir / "manifest.json", manifest)
    print(f"Saved {len(sequences)} EvoDiff samples to {fasta}")


if __name__ == "__main__":
    main()
