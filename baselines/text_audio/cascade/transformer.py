#!/usr/bin/env python3
"""Train a causal Transformer over 32-token BiCodec speaker-ID sequences."""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class SpeakerTokenDataset(Dataset):
    """Memory-mapped rows of raw (offset-free) speaker token IDs."""

    def __init__(self, cache, meta=None):
        self.path = Path(cache)
        self.meta_path = Path(meta) if meta else self.path.with_suffix(".meta.json")
        with self.meta_path.open() as handle:
            self.meta = json.load(handle)
        self.dtype = np.dtype(self.meta["dtype"])
        self.n_sequences = int(self.meta["n_sequences"])
        self.sequence_length = int(self.meta["seq_len_tokens"])
        self.vocab_size = int(self.meta["speaker_vocab"])
        expected = self.n_sequences * self.sequence_length * self.dtype.itemsize
        if self.path.stat().st_size != expected:
            raise ValueError(
                f"Cache size mismatch: expected {expected:,}, "
                f"found {self.path.stat().st_size:,} bytes"
            )
        self.tokens = None

    def _array(self):
        if self.tokens is None:
            self.tokens = np.memmap(
                self.path, dtype=self.dtype, mode="r",
                shape=(self.n_sequences, self.sequence_length)
            )
        return self.tokens

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, index):
        row = np.array(self._array()[index], dtype=np.int64, copy=True)
        if row.min() < 0 or row.max() >= self.vocab_size:
            raise ValueError(f"Out-of-range token in row {index}")
        return torch.from_numpy(row)


class SpeakerTokenTransformer(nn.Module):
    """Decoder-style causal Transformer for fixed-length speaker tokens."""

    def __init__(self, vocab_size=4096, sequence_length=32, d_model=128,
                 n_heads=4, n_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.bos_token_id = vocab_size
        self.token_embedding = nn.Embedding(vocab_size + 1, d_model)
        self.position_embedding = nn.Embedding(sequence_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            layer, n_layers, nn.LayerNorm(d_model)
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        length = input_ids.shape[1]
        if length > self.sequence_length:
            raise ValueError("Input exceeds configured sequence length")
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=input_ids.device),
            diagonal=1
        )
        hidden = self.transformer(hidden, mask=mask, is_causal=True)
        return self.lm_head(hidden)

    @torch.inference_mode()
    def sample(self, batch_size=1, temperature=1.0, top_k=None, device=None):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and not 1 <= top_k <= self.vocab_size:
            raise ValueError("top_k is outside the vocabulary")
        device = torch.device(device or next(self.parameters()).device)
        previous_mode = self.training
        self.eval()
        tokens = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        for _ in range(self.sequence_length):
            bos = torch.full(
                (batch_size, 1), self.bos_token_id,
                dtype=torch.long, device=device
            )
            logits = self(torch.cat((bos, tokens), dim=1))[:, -1] / temperature
            if top_k is not None:
                threshold = torch.topk(logits, top_k).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, -torch.inf)
            token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            tokens = torch.cat((tokens, token), dim=1)
        self.train(previous_mode)
        return tokens


def inputs_for(targets, bos_id):
    bos = torch.full(
        (targets.shape[0], 1), bos_id, dtype=targets.dtype, device=targets.device
    )
    return torch.cat((bos, targets[:, :-1]), dim=1)


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def arguments():
    cascade_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path,
                        default=cascade_root / "data/speaker_tokens_train.uint16")
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=cascade_root / "runs/speaker_transformer")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main():
    args = arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SpeakerTokenDataset(args.cache, args.meta)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0
    )
    model_config = {
        "vocab_size": dataset.vocab_size,
        "sequence_length": dataset.sequence_length,
        "d_model": args.dim,
        "n_heads": args.heads,
        "n_layers": args.layers,
        "dim_feedforward": args.mlp_dim,
        "dropout": args.dropout,
    }
    model = SpeakerTokenTransformer(**model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            1.0, float(step + 1) / max(1, args.warmup_steps)
        )
    )
    start_epoch, global_step = 0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    with (args.output_dir / "config.json").open("w") as handle:
        json.dump({
            "model": model_config,
            "training": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()},
            "n_parameters": parameter_count,
            "steps_per_epoch": len(loader),
        }, handle, indent=2)
        handle.write("\n")

    log_path = args.output_dir / "train.jsonl"
    print(f"{parameter_count:,} parameters; {len(dataset):,} sequences; "
          f"{len(loader):,} steps/epoch; device={device}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        weighted_loss, seen_tokens = 0.0, 0
        start_time = time.time()
        progress = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            total=len(loader),
            dynamic_ncols=True,
            unit="batch",
        )
        for batch_index, targets in enumerate(progress):
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda"
            ):
                logits = model(inputs_for(targets, model.bos_token_id))
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            weighted_loss += loss.item() * targets.numel()
            seen_tokens += targets.numel()

            if batch_index == 0 or (batch_index + 1) % 10 == 0:
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mean=f"{weighted_loss / seen_tokens:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    refresh=False,
                )

            if global_step == 1 or global_step % args.log_every == 0:
                record = {
                    "epoch": epoch, "batch": batch_index, "step": global_step,
                    "loss": loss.item(),
                    "perplexity": math.exp(min(loss.item(), 20)),
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_per_second": seen_tokens / (time.time() - start_time),
                }
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")

        record = {
            "epoch": epoch, "step": global_step,
            "mean_loss": weighted_loss / seen_tokens,
            "elapsed_seconds": time.time() - start_time,
            "sample_token_ids": model.sample(
                batch_size=4, top_k=100
            ).cpu().tolist(),
        }
        print(json.dumps(record), flush=True)
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_config": model_config,
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
        }
        atomic_save(checkpoint, args.output_dir / f"epoch_{epoch + 1}.pt")
        atomic_save(checkpoint, args.output_dir / "last.pt")


if __name__ == "__main__":
    main()
