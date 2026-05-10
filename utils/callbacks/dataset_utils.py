# callbacks/dataset_utils.py
from __future__ import annotations

import math
from typing import Any, Optional

import torch


def extract_dataset_attr(ds: Any, name: str):
    if hasattr(ds, name):
        return getattr(ds, name)
    if hasattr(ds, "dataset"):
        return extract_dataset_attr(ds.dataset, name)
    return None


def norm_ds(name: object) -> str:
    s = str(name or "").strip().lower()
    return s.replace("-", "").replace("_", "")


def is_text8(trainer) -> bool:
    return norm_ds(getattr(trainer.cfg.data, "dataset", "")) == "text8"


def is_wikitext(trainer) -> bool:
    ds = norm_ds(getattr(trainer.cfg.data, "dataset", ""))
    return ds in {"wikitext", "wikitext103"}


def is_text_dataset(trainer) -> bool:
    ds_name = norm_ds(getattr(trainer.cfg.data, "dataset", ""))
    if ds_name in {"text8", "wikitext", "wikitext103"}:
        return True
    ds = trainer.train_loader.dataset
    return bool(extract_dataset_attr(ds, "is_text_dataset"))


def seqs_to_images(seqs: torch.Tensor, trainer) -> torch.Tensor:
    ds = trainer.train_loader.dataset

    base_ds = ds
    while hasattr(base_ds, "dataset"):
        base_ds = base_ds.dataset

    reconstruct_fn = getattr(base_ds, "reconstruct_from_bits", None)
    if callable(reconstruct_fn):
        B = seqs.size(0)
        imgs = []
        for i in range(B):
            bits_i = seqs[i]
            if bits_i.is_floating_point():
                bits_i = (bits_i > 0.5).to(torch.float32)
            else:
                bits_i = (bits_i != 0).to(torch.float32)
            with torch.no_grad():
                img_chw = reconstruct_fn(bits_i)
            imgs.append(img_chw.unsqueeze(0))
        return torch.cat(imgs, dim=0)

    invperm = extract_dataset_attr(ds, "invperm")
    shape_hw = extract_dataset_attr(ds, "shape_hw")
    bits_per_pixel = extract_dataset_attr(ds, "bits_per_pixel")
    channels = extract_dataset_attr(ds, "channels") or 1

    if shape_hw is None:
        H = W = int(math.sqrt(seqs.shape[1]))
        inv = torch.arange(seqs.shape[1], device=seqs.device, dtype=torch.long)
        imgs = seqs[:, inv].view(-1, 1, H, W).to(torch.float32)
        return imgs

    H, W = shape_hw
    if invperm is None:
        inv = torch.arange(H * W, device=seqs.device, dtype=torch.long)
    else:
        inv = invperm.to(seqs.device).view(-1)

    B, S = seqs.shape

    if bits_per_pixel is None or int(bits_per_pixel) == 1:
        seqs_reordered = seqs[:, inv]
        imgs = seqs_reordered.view(B, 1, H, W)
        return imgs.to(torch.float32)

    bpp = int(bits_per_pixel)
    if bpp % channels != 0:
        raise ValueError(f"bits_per_pixel={bpp} is not divisible by channels={channels}.")
    bits_per_channel = bpp // channels

    if seqs.is_floating_point():
        bits = (seqs > 0.5).to(torch.uint8)
    else:
        bits = (seqs != 0).to(torch.uint8)

    bits = bits.view(B, H * W, bpp)
    bits = bits[:, inv, :]
    bits = bits.view(B, H * W, channels, bits_per_channel)

    vals = torch.zeros(B, H * W, channels, dtype=torch.uint8, device=seqs.device)
    for k in range(bits_per_channel):
        shift = bits_per_channel - 1 - k
        vals |= (bits[..., k] & 1) << shift

    vals = vals.view(B, H, W, channels).permute(0, 3, 1, 2)
    imgs = vals.to(torch.float32) / 255.0
    return imgs
