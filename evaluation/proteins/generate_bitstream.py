"""Generate exact-length UniRef50 benchmark samples from a BitStream checkpoint."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from data.proteins import bitstreams_to_token_ids
from data.uniref50 import EVODIFF_UNIREF50_ALPHABET
from evaluation.utils import (
    load_checkpoint,
    load_config,
    sample_text_sequences_for_external,
    unwrap_all,
)
from models import create_model
from utils.ema import EMA

from .io import atomic_json, sha256, write_fasta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--lengths", nargs="+", type=int, default=[100, 200, 300, 400, 500]
    )
    parser.add_argument("--samples-per-length", type=int, default=400)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--terminal-sigma", type=float, default=0.08)
    parser.add_argument("--sampler", default="heun_karras")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if str(cfg.data.dataset) not in {
        "EvoDiffUniRef50",
        "evodiff_uniref50",
        "uniref50_evodiff",
    }:
        raise ValueError("Config is not the frozen EvoDiff UniRef50 protocol")
    if str(cfg.data.representation).lower() != "binary":
        raise ValueError("BitStream generation requires binary representation")
    cfg.train.generation.num_sampling_steps = args.num_steps
    cfg.train.generation.terminal_sigmas = [args.terminal_sigma]
    cfg.train.generation.entropic_blend_alpha = 0.0
    cfg.train.generation.entropy_ckpt_path = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg).to(device)
    ema = EMA(unwrap_all(model), decay=0.0)
    load_checkpoint(model, ema, args.checkpoint, device, apply_ema=True)
    model.eval()

    sequences = []
    for length in args.lengths:
        if length > int(cfg.data.max_len):
            raise ValueError(
                f"Requested length {length} exceeds model max_len"
            )
        remaining = args.samples_per_length
        local = copy.deepcopy(cfg)
        local.data.sequence_len_tokens = int(length)
        local.data.sequence_len = int(length) * 5
        while remaining:
            count = min(remaining, args.micro_batch_size)
            bits, _prefix, _condition = sample_text_sequences_for_external(
                cfg=local,
                model=model,
                device=device,
                num_samples=count,
                sampler_name=args.sampler,
                return_dict=False,
                decode_strategy="codebook_map",
                codebook_size=len(EVODIFF_UNIREF50_ALPHABET),
                bits_per_code=5,
                warmup=False,
                ddp=False,
            )
            token_ids = bitstreams_to_token_ids(bits.cpu(), 5)
            sequences.extend(
                "".join(EVODIFF_UNIREF50_ALPHABET[token] for token in row)
                for row in token_ids.tolist()
            )
            remaining -= count

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fasta = args.out_dir / "generated.fasta"
    write_fasta(fasta, sequences, "bitstream")
    dataset_manifest = Path(cfg.data.root) / "frozen_manifest.json"
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": args.config, "sha256": sha256(Path(args.config))},
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256(args.checkpoint),
        },
        "dataset_manifest": {
            "path": str(dataset_manifest),
            "sha256": sha256(dataset_manifest),
        },
        "generation": vars(args) | {"out_dir": str(args.out_dir)},
        "fasta": {"path": str(fasta), "sha256": sha256(fasta)},
    }
    atomic_json(args.out_dir / "manifest.json", payload)
    print(f"Saved {len(sequences)} BitStream samples to {fasta}")


if __name__ == "__main__":
    main()
