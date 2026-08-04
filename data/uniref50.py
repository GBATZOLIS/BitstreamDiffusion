"""Frozen EvoDiff/DPLM UniRef50 protocol.

This module reads the March-2020 UniRef50 archive released with the
protein-sequence-models/EvoDiff work.  It intentionally keeps ``valid`` and
``rtest`` distinct: validation is used while training and ``rtest`` is only
used for final evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import BinaryIO, Optional

import numpy as np
import torch
from ml_collections import config_dict
from torch.utils.data import DataLoader, Dataset

from .uniref50_sampler import DistributedRandomLengthBatchSampler

from .proteins import (
    DistributedLengthBucketBatchSampler,
    _ddp_rank_world,
    build_token_to_bits_table,
)


EVODIFF_UNIREF50_ALPHABET = "ACDEFGHIKLMNPQRSTVWYBZXJOU"
EVODIFF_UNIREF50_ARCHIVE_MD5 = "e2278f1d93836e8309cf4192b8455828"
EVODIFF_UNIREF50_ZENODO_RECORD = "6564798"
EVODIFF_UNIREF50_URL = (
    "https://zenodo.org/records/6564798/files/uniref50.tar.gz?download=1"
)
EVODIFF_COMMIT = "33206e99446f799ec11cf6e57d66ffaec837be91"
DPLM_COMMIT = "8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d"

SPLIT_MAP = {"train": "train", "val": "valid", "test": "rtest"}


def _resolve_data_dir(root: Path, manifest: dict) -> Path:
    data_dir = Path(manifest["dataset"]["data_dir"])
    return data_dir if data_dir.is_absolute() else root / data_dir


class EvoDiffUniRef50Dataset(Dataset):
    """Padding-free UniRef50 rows using the exact released EvoDiff splits.

    Sequences longer than ``max_len`` use a random crop during training and a
    deterministic leading crop for validation/test.  This matches the upstream
    loader's crop size while making checkpoint selection and final evaluation
    repeatable.
    """

    is_text_dataset = False

    def __init__(self, config: config_dict.ConfigDict, *, split: str):
        super().__init__()
        if split not in SPLIT_MAP:
            raise ValueError(f"Unknown split {split!r}; expected train/val/test")

        self.requested_split = split
        self.split = SPLIT_MAP[split]
        self.root = Path(
            getattr(config.data, "root", "datasets/uniref50_evodiff_2020")
        )
        manifest_path = self.root / "frozen_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Prepare the frozen dataset with:\n"
                "  python scripts/proteins/setup/prepare_evodiff_uniref50.py --download"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        archive_md5 = self.manifest["source"]["archive_md5"]
        if archive_md5 != EVODIFF_UNIREF50_ARCHIVE_MD5:
            raise ValueError(
                f"Unexpected UniRef50 archive MD5 {archive_md5}; "
                f"expected {EVODIFF_UNIREF50_ARCHIVE_MD5}"
            )

        self.data_dir = _resolve_data_dir(self.root, self.manifest)
        metadata = np.load(
            self.data_dir / "lengths_and_offsets.npz", allow_pickle=False
        )
        all_offsets = metadata["seq_offsets"]
        all_lengths = metadata["ells"]
        with (self.data_dir / "splits.json").open("r", encoding="utf-8") as handle:
            split_indices = json.load(handle)[self.split]
        self.source_indices = np.asarray(split_indices, dtype=np.int64)

        self.max_len = int(getattr(config.data, "max_len", 1022))
        self.min_len = int(getattr(config.data, "min_len", 1))
        if self.max_len <= 0 or self.min_len <= 0 or self.min_len > self.max_len:
            raise ValueError("Require 0 < data.min_len <= data.max_len")

        source_lengths = np.asarray(all_lengths[self.source_indices], dtype=np.int64)
        keep = source_lengths >= self.min_len
        if not bool(np.all(keep)):
            self.source_indices = self.source_indices[keep]
            source_lengths = source_lengths[keep]

        limit_name = "limit_train" if split == "train" else "limit_eval"
        limit = int(getattr(config.data, limit_name, 0))
        if limit > 0:
            self.source_indices = self.source_indices[:limit]
            source_lengths = source_lengths[:limit]

        self.offsets = np.asarray(all_offsets, dtype=np.int64)
        self.source_lengths = source_lengths
        self.lengths = np.minimum(source_lengths, self.max_len).astype(np.int64)
        self.sequence_path = self.data_dir / "consensus.fasta"
        self._sequence_handle: Optional[BinaryIO] = None

        self.representation = str(
            getattr(config.data, "representation", "binary")
        ).lower()
        if self.representation not in {"binary", "tokens"}:
            raise ValueError("UniRef50 representation must be 'binary' or 'tokens'")
        self.alphabet = str(
            getattr(config.data, "alphabet", EVODIFF_UNIREF50_ALPHABET)
        )
        if self.alphabet != EVODIFF_UNIREF50_ALPHABET:
            raise ValueError(
                "The frozen EvoDiff protocol requires alphabet "
                f"{EVODIFF_UNIREF50_ALPHABET!r}"
            )
        self.vocab_size = len(self.alphabet)
        self.bits_per_token = int(getattr(config.data, "bits_per_token", 5))
        if self.bits_per_token != 5:
            raise ValueError("The 26-symbol EvoDiff alphabet requires 5 bits/token")
        self.byte_to_id = np.full(256, -1, dtype=np.int16)
        for token_id, residue in enumerate(self.alphabet):
            self.byte_to_id[ord(residue)] = token_id
        self.token_to_bits_table = build_token_to_bits_table(
            self.vocab_size, self.bits_per_token
        )

        print(
            f"[proteins:evodiff] requested={split} source={self.split} "
            f"n={len(self):,} length={int(self.lengths.min())}-"
            f"{int(self.lengths.max())} max_len={self.max_len}"
        )

    def __len__(self) -> int:
        return int(len(self.source_indices))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_sequence_handle"] = None
        return state

    def close(self) -> None:
        handle = getattr(self, "_sequence_handle", None)
        if handle is not None:
            handle.close()
            self._sequence_handle = None

    def __del__(self):
        self.close()

    def _handle(self) -> BinaryIO:
        if self._sequence_handle is None:
            self._sequence_handle = self.sequence_path.open("rb")
        return self._sequence_handle

    def sequence(self, idx: int) -> str:
        source_idx = int(self.source_indices[idx])
        handle = self._handle()
        handle.seek(int(self.offsets[source_idx]))
        raw = handle.readline().rstrip(b"\r\n")
        expected = int(self.source_lengths[idx])
        if len(raw) != expected:
            raise ValueError(
                f"Corrupt UniRef50 row {source_idx}: metadata length {expected}, "
                f"read {len(raw)} bytes"
            )
        if expected > self.max_len:
            if self.requested_split == "train":
                start = int(np.random.randint(0, expected - self.max_len + 1))
            else:
                start = 0
            raw = raw[start : start + self.max_len]
        return raw.decode("ascii")

    def __getitem__(self, idx: int) -> torch.Tensor:
        sequence = self.sequence(idx)
        residue_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        token_ids = self.byte_to_id[residue_bytes]
        if np.any(token_ids < 0):
            bad = sorted(set(chr(int(x)) for x in residue_bytes[token_ids < 0]))
            raise ValueError(
                f"UniRef50 row {int(self.source_indices[idx])} contains symbols "
                f"outside the frozen alphabet: {bad}"
            )
        ids = torch.from_numpy(np.asarray(token_ids, dtype=np.int64))
        if self.representation == "tokens":
            return ids
        return self.token_to_bits_table[ids].reshape(-1)


def get_evodiff_uniref50_loader(
    config: config_dict.ConfigDict,
    *,
    split: str,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    seed: int = 42,
) -> DataLoader:
    dataset = EvoDiffUniRef50Dataset(config, split=split)
    if shuffle is None:
        shuffle = split == "train"
    rank, world_size = _ddp_rank_world()
    local_batch_size = int(batch_size or config.train.batch_size)
    steps_per_epoch = int(getattr(config.train, "steps_per_epoch", 0))
    if split == "train" and steps_per_epoch > 0:
        sampler = DistributedRandomLengthBatchSampler(
            dataset.lengths,
            batch_size=local_batch_size,
            num_batches=steps_per_epoch,
            seed=int(seed),
            rank=rank,
            world_size=world_size,
        )
    else:
        sampler = DistributedLengthBucketBatchSampler(
            dataset.lengths,
            batch_size=local_batch_size,
            shuffle=bool(shuffle),
            seed=int(seed),
            rank=rank,
            world_size=world_size,
        )
    num_workers = int(getattr(config.data, "num_workers", 8))
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(getattr(config.data, "pin_memory", True)),
        persistent_workers=num_workers > 0,
        prefetch_factor=(
            int(getattr(config.data, "prefetch_factor", 4))
            if num_workers > 0
            else None
        ),
    )
