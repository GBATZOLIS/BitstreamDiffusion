#!/usr/bin/env python
"""Produce a slim, EMA-baked release checkpoint from a full training checkpoint.

Training checkpoints written by ``trainers/trainer.py`` carry the live model
weights, the EMA shadow, the AdamW optimizer state, the LR scheduler, the grad
scaler and RNG state. For a 462M model that is ~7.4 GB, ~3.7 GB of which is the
optimizer moments — useless for inference.

Evaluation in this repo runs with ``apply_ema=True`` by default
(``evaluation/utils.py:load_checkpoint``): it loads ``ckpt["model"]`` and then
copies ``ckpt["ema"]["shadow"]`` over the trainable parameters. The paper /
Table-2 numbers were all produced this way.

This script *bakes* that exact operation into the weights: it copies the EMA
shadow over the matching ``ckpt["model"]`` entries (leaving non-parameter
buffers untouched) and drops everything else. The result is a single
``{"model": <ema-baked weights>}`` checkpoint (~1.85 GB) that, evaluated with
the default ``apply_ema=True`` (which becomes a no-op because there is no "ema"
key), is numerically identical to evaluating the original full checkpoint.

Usage:
    python scripts/owt/make_release_checkpoint.py \
        --in  runs/.../checkpoints/step=000750000.pt \
        --out release/cobit_m_owt_750k_ema.pt
"""
import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, type=Path, help="full training checkpoint")
    ap.add_argument("--out", dest="dst", required=True, type=Path, help="slim EMA-baked output")
    ap.add_argument("--no-bake", action="store_true",
                    help="keep separate model+ema keys (drop only optimizer) instead of baking")
    args = ap.parse_args()

    if not args.src.exists():
        raise FileNotFoundError(args.src)
    print(f"Loading {args.src} ...")
    ck = torch.load(args.src, map_location="cpu", weights_only=False)

    model = ck["model"]
    ema = ck.get("ema")
    if ema is None or "shadow" not in ema:
        raise ValueError("Checkpoint has no EMA shadow — refusing to make an EMA release checkpoint.")
    shadow = ema["shadow"]

    # Sanity: every shadow key must exist in the model state dict.
    missing = [k for k in shadow if k not in model]
    if missing:
        raise ValueError(f"{len(missing)} EMA shadow keys absent from model state_dict, e.g. {missing[:3]}")

    meta = {
        "global_step": ck.get("global_step"),
        "epoch": ck.get("epoch"),
        "ema_decay": ema.get("decay"),
        "source_checkpoint": str(args.src),
        "note": "Release checkpoint. EMA shadow baked into weights; evaluate with the default apply_ema=True.",
    }

    args.dst.parent.mkdir(parents=True, exist_ok=True)

    if args.no_bake:
        out = {"model": model, "ema": ema, **meta}
        print("Keeping separate model + ema keys (optimizer dropped).")
    else:
        baked = OrderedDict(model)
        n = 0
        for k, v in shadow.items():
            baked[k] = v.clone()
            n += 1
        out = {"model": baked, **meta}
        print(f"Baked {n} EMA params into the model weights ({len(model) - n} buffers untouched).")

    torch.save(out, args.dst)
    size_gb = args.dst.stat().st_size / 1e9
    print(f"✓ Wrote {args.dst}  ({size_gb:.2f} GB)  global_step={meta['global_step']} ema_decay={meta['ema_decay']}")


if __name__ == "__main__":
    main()
