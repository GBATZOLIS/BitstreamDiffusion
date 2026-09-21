"""
data/mnist_sum_bits.py

Dataset for the MNIST-Sum image+text corpus (see docs/MNIST_SUM_EXPERIMENT.md).

Mirrors `data/multimodal_lfq_bits.py` in interface -- one flat {0,1} bit vector
per example -- but the image half is raw binarised pixels rather than LFQ codes,
so `reconstruct_batch_from_bits` needs no tokenizer and there is no
reconstruction floor.

Also exposes `labels(idx)` (true digits and true sum), which is what makes the
exact image->text scoring possible.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ml_collections import config_dict
from torch.utils.data import DataLoader, Dataset

from . import mnist_sum_codec as C


class MNISTSumBitsDataset(Dataset):
    def __init__(self, config: config_dict.ConfigDict, *, split: str):
        super().__init__()
        self.config = config
        self.split = "val" if split == "test" else split

        root = Path(str(config.data.precomputed_root))
        with open(root / "meta.json") as f:
            self.meta = json.load(f)

        self.bits_per_token = int(self.meta["bits_per_token"])
        self.caption_len_tokens = int(self.meta["caption_len_tokens"])
        self.num_image_tokens = int(self.meta["num_image_tokens"])
        self.num_positions = int(self.meta["num_positions"])
        self.sequence_len = self.num_positions * self.bits_per_token
        self.layout = C.build_layout(self.bits_per_token, self.caption_len_tokens)

        declared = getattr(config.data, "sequence_len", None)
        if declared is not None and int(declared) != self.sequence_len:
            raise ValueError(
                f"cfg.data.sequence_len={int(declared)} disagrees with dataset "
                f"sequence_len={self.sequence_len}")
        p_model = getattr(getattr(config, "model", object()), "patch_size", None)
        if p_model is not None and int(p_model) != self.bits_per_token:
            raise ValueError(
                f"cfg.model.patch_size={int(p_model)} disagrees with the corpus "
                f"bits_per_token={self.bits_per_token}. The trunk patches the flat "
                f"bitstream every patch_size bits, so these must match or positions "
                f"will straddle segment boundaries.")

        split_dir = root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"no split {self.split!r} under {root}")
        idx = json.load(open(split_dir / "shard_idx.json"))
        rec = idx["0"]
        self.num_images = int(rec["count"])
        self._mmap = np.memmap(split_dir / rec["file"], dtype=np.uint32, mode="r",
                               shape=(self.num_images, self.num_positions))

        lab = np.load(split_dir / "labels.npz")
        self._digits = lab["digits"]          # [N, 4] uint8
        self._sums = lab["sums"]              # [N] uint8

        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[MNISTSumBits/{self.split}] n={self.num_images:,} "
                  f"positions={self.num_positions} (text={self.layout.num_text_positions}"
                  f"+image={self.num_image_tokens}+markers=6) "
                  f"seq_len={self.sequence_len} P={self.bits_per_token} "
                  f"tuples={self.meta['splits'].get(self.split, {}).get('tuples', '?')}")

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, idx: int):
        codes = np.asarray(self._mmap[idx], dtype=np.int64)          # [N]
        shifts = np.arange(self.bits_per_token, dtype=np.int64)
        bits = ((codes[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
        return torch.from_numpy(bits.reshape(-1))                    # [N*P]

    # ---- ground truth, for exact scoring --------------------------------
    def labels(self, idx):
        i = np.asarray(idx)
        return (torch.from_numpy(self._digits[i].astype(np.int64)),
                torch.from_numpy(self._sums[i].astype(np.int64)))

    @property
    def modal_sum_baseline(self) -> float:
        return float(self.meta["splits"].get(self.split, {}).get("modal_sum_baseline", 0.0))

    # ---- decode ---------------------------------------------------------
    def reconstruct_batch_from_bits(self, bits: torch.Tensor):
        """
        [B, seq_len] (or [B, N, P]) bits -> {"image": [B,1,56,56] in [0,1],
                                             "quadrants": [B,4,28,28],
                                             "text": List[str],
                                             "text_codes": [B, Lt]}
        No tokenizer is involved in either direction.
        """
        if bits.dim() == 1:
            bits = bits.unsqueeze(0)
        B = bits.shape[0]
        bits_nb = bits.reshape(B, self.num_positions, self.bits_per_token)
        codes = C.bits_to_ids(bits_nb)
        parts = C.split_codes(self.layout, codes)

        quads = C.unpack_image_bits(self.layout, parts["image"])
        canvas = C.quadrants_to_canvas(quads)
        tcodes = C.unpack_text_codes(self.layout, parts["text_pos"])
        texts = [" ".join(C.codes_to_words(row.tolist())) for row in tcodes.cpu()]
        return {"image": canvas, "quadrants": quads,
                "text": texts, "text_codes": tcodes}

    def reconstruct_from_bits(self, bits: torch.Tensor):
        out = self.reconstruct_batch_from_bits(
            bits.unsqueeze(0) if bits.dim() == 1 else bits)
        return {"image": out["image"][0], "text": out["text"][0]}


def get_dataloaders(config, *, batch_size: Optional[int] = None, seed: int = 42):
    bs = batch_size or config.train.batch_size
    nw = int(getattr(config.data, "num_workers", 6))
    pin = bool(getattr(config.data, "pin_memory", True))

    train_ds = MNISTSumBitsDataset(config, split="train")
    val_ds = MNISTSumBitsDataset(config, split="val")

    def make(ds, shuffle, drop_last):
        kw = {}
        if nw > 0:
            kw["prefetch_factor"] = int(getattr(config.data, "prefetch_factor", 4))
            kw["persistent_workers"] = True
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
                          pin_memory=pin, drop_last=drop_last, **kw)

    return make(train_ds, True, True), make(val_ds, False, False), make(val_ds, False, False)
