"""Generate the frozen length grid from an official DPLM checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .dplm_loader import load_dplm_class
from .io import atomic_json, sha256, write_fasta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="airkingbd/dplm_150m")
    parser.add_argument("--dplm-root", type=Path, required=True)
    parser.add_argument(
        "--lengths", nargs="+", type=int, default=[100, 200, 300, 400, 500]
    )
    parser.add_argument("--samples-per-length", type=int, default=400)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--sampling-strategy", default="gumbel_argmax")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    DiffusionProteinLanguageModel = load_dplm_class(args.dplm_root)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    model = (
        DiffusionProteinLanguageModel.from_pretrained(args.model)
        .to(device)
        .eval()
    )
    tokenizer = model.tokenizer

    sequences = []
    for length in args.lengths:
        remaining = args.samples_per_length
        while remaining:
            count = min(remaining, args.micro_batch_size)
            masked = ["".join(["<mask>"] * length)] * count
            input_tokens = tokenizer.batch_encode_plus(
                masked,
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            )["input_ids"].to(device)
            partial_mask = input_tokens.ne(model.mask_id)
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ),
            ):
                output = model.generate(
                    input_tokens=input_tokens,
                    tokenizer=tokenizer,
                    max_iter=args.max_iter,
                    sampling_strategy=args.sampling_strategy,
                    partial_masks=partial_mask,
                )
            decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
            sequences.extend(
                "".join(sequence.split(" ")) for sequence in decoded
            )
            remaining -= count

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fasta = args.out_dir / "generated.fasta"
    write_fasta(fasta, sequences, "dplm")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "source": "official bytedance/dplm code and Hugging Face checkpoint",
        "generation": vars(args) | {"out_dir": str(args.out_dir)},
        "fasta": {"path": str(fasta), "sha256": sha256(fasta)},
    }
    atomic_json(args.out_dir / "manifest.json", manifest)
    print(f"Saved {len(sequences)} DPLM samples to {fasta}")


if __name__ == "__main__":
    main()
