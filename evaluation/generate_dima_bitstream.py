"""Length-matched generation for the frozen DiMA Swiss-Prot protocol."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from data.proteins import DIMA_CANONICAL_AA, bitstreams_to_token_ids
from evaluation.utils import load_checkpoint, load_config, sample_text_sequences_for_external, unwrap_all
from models import create_model
from utils.ema import EMA


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--terminal-sigma", type=float, default=0.08)
    parser.add_argument("--sampler", default="heun_karras")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if str(cfg.data.representation).lower() != "binary":
        raise ValueError("This generator is for the BitStream representation")
    cfg.train.generation.num_sampling_steps = int(args.num_steps)
    cfg.train.generation.terminal_sigmas = [float(args.terminal_sigma)]
    cfg.train.generation.entropic_blend_alpha = 0.0
    cfg.train.generation.entropy_ckpt_path = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = create_model(cfg).to(device)
    ema = EMA(unwrap_all(model), decay=0.0)
    checkpoint = Path(args.checkpoint)
    load_checkpoint(model, ema, checkpoint, device, apply_ema=True)
    model.eval()

    probability_path = Path(cfg.evaluation.length_distribution)
    probabilities = np.load(probability_path, allow_pickle=False)
    probabilities = probabilities / probabilities.sum()
    rng = np.random.default_rng(args.seed)
    sampled_lengths = rng.choice(
        np.arange(len(probabilities)), size=args.num_samples, p=probabilities
    ).astype(np.int64)
    if sampled_lengths.min() < 128 or sampled_lengths.max() > 254:
        raise ValueError("Frozen DiMA length distribution produced an out-of-protocol length")

    sequences = [None] * int(args.num_samples)
    for length in sorted(np.unique(sampled_lengths).tolist()):
        positions = np.flatnonzero(sampled_lengths == length)
        local_cfg = copy.deepcopy(cfg)
        local_cfg.data.sequence_len_tokens = int(length)
        local_cfg.data.sequence_len = int(length) * 5
        for start in range(0, len(positions), int(args.micro_batch_size)):
            selected = positions[start : start + int(args.micro_batch_size)]
            bits, _prefix, _cond = sample_text_sequences_for_external(
                cfg=local_cfg,
                model=model,
                device=device,
                num_samples=len(selected),
                sampler_name=args.sampler,
                return_dict=False,
                decode_strategy="codebook_map",
                codebook_size=20,
                bits_per_code=5,
                warmup=False,
                ddp=False,
            )
            token_ids = bitstreams_to_token_ids(bits.cpu(), 5).tolist()
            for output_index, row in zip(selected.tolist(), token_ids):
                sequences[output_index] = "".join(DIMA_CANONICAL_AA[token] for token in row)

    if any(sequence is None for sequence in sequences):
        raise RuntimeError("Generation did not fill every sampled length")
    if any(len(sequence) != int(length) for sequence, length in zip(sequences, sampled_lengths)):
        raise RuntimeError("Generated sequence length mismatch")

    out_dir = args.out_dir or Path(cfg.evaluation.out_dir) / (
        f"frozen_seed{args.seed}_n{args.num_samples}_steps{args.num_steps}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / "generated.fasta"
    with fasta_path.open("w", encoding="utf-8") as handle:
        for index, sequence in enumerate(sequences):
            handle.write(f">gen_{index} length={len(sequence)}\n{sequence}\n")
    lengths_path = out_dir / "sampled_lengths.npy"
    np.save(lengths_path, sampled_lengths, allow_pickle=False)

    dataset_manifest = Path(cfg.data.root) / "frozen_manifest.json"
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": args.config, "sha256": sha256(Path(args.config))},
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "dataset_manifest": {
            "path": str(dataset_manifest),
            "sha256": sha256(dataset_manifest),
        },
        "generation": vars(args) | {"out_dir": str(out_dir)},
        "artifacts": {
            "fasta": {"path": str(fasta_path), "sha256": sha256(fasta_path)},
            "lengths": {"path": str(lengths_path), "sha256": sha256(lengths_path)},
        },
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(f"Saved {len(sequences)} exact-length samples to {fasta_path}")


if __name__ == "__main__":
    main()
