from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from ml_collections import config_dict
from torch.utils.data import DataLoader, Dataset

from utils.ecc_secded import ecc_chunk_len, ecc_encode_data_bits, ecc_from_cfg


# -----------------------------------------------------------------------------
# Gray helpers
# -----------------------------------------------------------------------------

def int_to_gray(n: int) -> int:
    return n ^ (n >> 1)


def _ceil_log2(x: int) -> int:
    return int(math.ceil(math.log2(max(2, int(x)))))


# -----------------------------------------------------------------------------
# Tokenizer + semantic map assets
# -----------------------------------------------------------------------------

def load_lm1b_tokenizer(root: Path, tokenizer_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError("Missing transformers. Install: pip install transformers") from e

    del root  # kept for signature symmetry / future-proofing
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    return tok


def load_lm1b_semantic_map(root: Path) -> tuple[Dict[int, int], Dict[int, int]]:
    map_path = root / "semantic_mapping_bert_base_uncased.json"
    if not map_path.exists():
        raise RuntimeError(
            f"Missing semantic map at {map_path}. "
            "Run scripts/prepare_lm1b_semantic_map.py after rebuilding the packed LM1B caches."
        )

    with open(map_path, "r", encoding="utf-8") as f:
        maps = json.load(f)

    old_to_new = {int(k): int(v) for k, v in maps["old_to_new"].items()}
    new_to_old = {int(k): int(v) for k, v in maps["new_to_old"].items()}
    return old_to_new, new_to_old


def load_lm1b_semantic_assets(root: Path, tokenizer_name: str):
    tok = load_lm1b_tokenizer(root, tokenizer_name)
    old_to_new, new_to_old = load_lm1b_semantic_map(root)
    return tok, old_to_new, new_to_old


# -----------------------------------------------------------------------------
# Cache helpers
# -----------------------------------------------------------------------------

def _load_memmap(cache_path: Path, meta_path: Path) -> tuple[np.memmap, dict]:
    if not cache_path.exists() or not meta_path.exists():
        raise RuntimeError(
            "Missing LM1B token cache files. Run scripts/build_lm1b_bert_caches.py first.\n"
            f"Expected: {cache_path} and {meta_path}"
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n = int(meta["n_sequences"])
    seq_len = int(meta["seq_len_tokens"])
    mm = np.memmap(cache_path, dtype=np.uint16, mode="r", shape=(n, seq_len))
    return mm, meta


# -----------------------------------------------------------------------------
# Token -> bits lookup cache
# -----------------------------------------------------------------------------
_TOKENBITS_CACHE: Dict[Tuple[str, str, int, int, int, int, int, int], torch.Tensor] = {}


def _get_token_to_bits_table(
    *,
    root: Path,
    vocab_size: int,
    data_bits_per_token: int,
    binarization: str,
    old_to_new: Optional[Dict[int, int]],
    ecc_enabled: bool,
    ecc_total_bits_per_token: int,
    ecc_include_overall_parity: bool,
) -> torch.Tensor:
    binarization = str(binarization).lower().strip()

    map_path = root / "semantic_mapping_bert_base_uncased.json"
    if binarization == "semantic":
        mtime = int(map_path.stat().st_mtime_ns) if map_path.exists() else 0
    else:
        mtime = 0

    key = (
        str(root.resolve()),
        str(binarization),
        int(vocab_size),
        int(data_bits_per_token),
        int(ecc_total_bits_per_token),
        int(1 if ecc_enabled else 0),
        int(1 if ecc_include_overall_parity else 0),
        mtime,
    )
    cached = _TOKENBITS_CACHE.get(key, None)
    if cached is not None:
        return cached

    m = int(data_bits_per_token)
    L = int(ecc_total_bits_per_token)

    data_table = torch.zeros((vocab_size, m), dtype=torch.long)

    for tid in range(vocab_size):
        if binarization == "semantic":
            if old_to_new is None:
                raise RuntimeError("binarization='semantic' requires a semantic map (old_to_new).")
            code = int_to_gray(int(old_to_new.get(tid, tid)))
        elif binarization == "raw_binary":
            code = int(tid)
        else:
            raise ValueError(
                f"Unknown LM1B binary binarization='{binarization}'. "
                "Supported: 'semantic', 'raw_binary'."
            )

        for k in range(m):
            data_table[tid, m - 1 - k] = (code >> k) & 1

    if not ecc_enabled:
        out_table = data_table
    else:
        ecc_cfg = ecc_from_cfg(
            type(
                "Cfg",
                (),
                {
                    "data": type(
                        "Data",
                        (),
                        {
                            "ecc": type(
                                "ECC",
                                (),
                                {
                                    "enabled": True,
                                    "data_bits": m,
                                    "parity_bits": None,
                                    "include_overall_parity": bool(ecc_include_overall_parity),
                                    "unk_token": "<unk>",
                                },
                            )()
                        },
                    )()
                },
            )()
        )
        out_table = ecc_encode_data_bits(data_table, ecc_cfg)
        if out_table.shape[-1] != L:
            raise RuntimeError(
                f"ECC table len mismatch: got {out_table.shape[-1]} expected {L}"
            )

    _TOKENBITS_CACHE[key] = out_table
    return out_table


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class LM1BDataset(Dataset):
    """
    LM1B packed-block benchmark dataset for:
      - discrete token runs
      - discrete bit runs
      - continuous bit runs

    Expected cache construction:
      - tokenizer: bert-base-uncased
      - add_special_tokens: False
      - raw examples concatenated into one token stream
      - optional separator token inserted between examples
      - stream chunked into fixed 128-token blocks
      - final incomplete tail dropped

    The dataset then returns either:
      - tokenizer ids
      - semantic-rank ids
      - semantic Gray bits
      - raw-binary bits
    """

    is_text_dataset = True

    def __init__(self, config: config_dict.ConfigDict, *, split: str):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.config = config
        self.split = split

        self.root = Path(getattr(config.data, "root", "datasets/lm1b"))
        self.tokenizer_name = str(getattr(config.data, "tokenizer_name", "bert-base-uncased"))

        self.repr = str(getattr(config.data, "representation", "tokens")).lower().strip()
        self.binarization = str(getattr(config.data, "binarization", "semantic")).lower().strip()
        self.token_space = str(getattr(config.data, "token_space", "semantic_rank")).lower().strip()

        if self.repr == "binary":
            if self.binarization not in {"semantic", "raw_binary"}:
                raise ValueError(
                    f"Unknown cfg.data.binarization={self.binarization!r} for LM1B binary mode. "
                    "Supported: 'semantic', 'raw_binary'."
                )
        elif self.repr == "tokens":
            if self.token_space not in {
                "semantic_rank",
                "semantic",
                "rank",
                "tokenizer_id",
                "raw",
                "tokenizer",
            }:
                raise ValueError(f"Unknown cfg.data.token_space={self.token_space}")
        else:
            raise ValueError(f"Unsupported LM1B representation={self.repr}")

        self.tokenizer = load_lm1b_tokenizer(self.root, self.tokenizer_name)
        self.old_to_new: Dict[int, int] = {}
        self.new_to_old: Dict[int, int] = {}

        need_semantic_map = (
            (self.repr == "binary" and self.binarization == "semantic")
            or (self.repr == "tokens" and self.token_space in {"semantic_rank", "semantic", "rank"})
        )
        if need_semantic_map:
            self.old_to_new, self.new_to_old = load_lm1b_semantic_map(self.root)

        self.vocab_size_base = int(self.tokenizer.vocab_size)

        if self.tokenizer.pad_token_id is None:
            raise RuntimeError(f"Tokenizer {self.tokenizer_name} does not expose a pad_token_id")

        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.unk_token_id = None if self.tokenizer.unk_token_id is None else int(self.tokenizer.unk_token_id)

        self.ecc = ecc_from_cfg(config)
        self.ecc_enabled = bool(self.ecc.enabled)

        data_bits_default = _ceil_log2(self.vocab_size_base)
        data_bits_cfg = getattr(config.data, "bits_per_token", None)
        self.data_bits_per_token = int(data_bits_cfg) if data_bits_cfg is not None else int(data_bits_default)

        min_bits_needed = _ceil_log2(self.vocab_size_base)
        if self.data_bits_per_token < min_bits_needed:
            raise ValueError(
                f"LM1B requires at least {min_bits_needed} data bits to encode vocab_size={self.vocab_size_base}, "
                f"but cfg.data.bits_per_token={self.data_bits_per_token}."
            )

        if self.ecc_enabled:
            if int(self.ecc.data_bits) != int(self.data_bits_per_token):
                raise ValueError(
                    f"ECC enabled but cfg.data.ecc.data_bits={int(self.ecc.data_bits)} does not match "
                    f"bits_per_token={int(self.data_bits_per_token)}."
                )
            self.bits_per_token = int(ecc_chunk_len(self.ecc))
        else:
            self.bits_per_token = int(self.data_bits_per_token)

        self.seq_len_tokens = int(getattr(config.data, "sequence_len_tokens", 128))
        self.seq_len_bits = self.seq_len_tokens * self.bits_per_token

        train_mm, train_meta = _load_memmap(
            self.root / "cache_train_tokens.uint16",
            self.root / "cache_train_tokens.meta.json",
        )
        test_mm, test_meta = _load_memmap(
            self.root / "cache_test_tokens.uint16",
            self.root / "cache_test_tokens.meta.json",
        )

        expected_cache_format = "packed_token_blocks"
        train_cache_format = str(train_meta.get("cache_format", "legacy_unknown"))
        test_cache_format = str(test_meta.get("cache_format", "legacy_unknown"))

        if train_cache_format != expected_cache_format or test_cache_format != expected_cache_format:
            raise RuntimeError(
                "LM1B cache files are not in the required packed-block format.\n"
                "Delete the old cache_*.uint16 / cache_*.meta.json files and rebuild them with:\n"
                "python -m scripts.build_lm1b_bert_caches --root datasets/lm1b "
                "--tokenizer_name bert-base-uncased --seq_len_tokens 128 --boundary_mode sep --force"
            )

        if int(train_meta["seq_len_tokens"]) != self.seq_len_tokens:
            raise RuntimeError(
                f"Train cache seq_len={train_meta['seq_len_tokens']} but config expects {self.seq_len_tokens}."
            )
        if int(test_meta["seq_len_tokens"]) != self.seq_len_tokens:
            raise RuntimeError(
                f"Test cache seq_len={test_meta['seq_len_tokens']} but config expects {self.seq_len_tokens}."
            )

        cache_tok_name_train = str(train_meta.get("tokenizer_name", self.tokenizer_name))
        cache_tok_name_test = str(test_meta.get("tokenizer_name", self.tokenizer_name))
        if cache_tok_name_train != self.tokenizer_name or cache_tok_name_test != self.tokenizer_name:
            raise RuntimeError(
                f"Cache tokenizer mismatch. "
                f"train={cache_tok_name_train!r}, test={cache_tok_name_test!r}, "
                f"config expects {self.tokenizer_name!r}."
            )

        self.cache_format = expected_cache_format
        self.cache_boundary_mode = train_meta.get("packing_boundary_mode", None)
        self.cache_boundary_token_id = train_meta.get("packing_boundary_token_id", None)

        self._train_mm = train_mm
        self._test_mm = test_mm

        frac_val = float(getattr(config.data, "val_fraction", 0.005))
        n_train_total = int(train_mm.shape[0])
        n_val = int(round(n_train_total * frac_val))
        n_val = max(1, n_val)
        n_train = n_train_total - n_val
        if n_train <= 0:
            raise RuntimeError("LM1B train split became empty after carving out validation rows.")

        if split == "train":
            self.mm = train_mm
            self.start = 0
            self.end = n_train
        elif split == "val":
            self.mm = train_mm
            self.start = n_train
            self.end = n_train_total
        else:
            self.mm = test_mm
            self.start = 0
            self.end = int(test_mm.shape[0])

        self.num_sequences = int(self.end - self.start)

        self.token_to_bits_table = None
        if self.repr == "binary":
            self.token_to_bits_table = _get_token_to_bits_table(
                root=self.root,
                vocab_size=self.vocab_size_base,
                data_bits_per_token=self.data_bits_per_token,
                binarization=self.binarization,
                old_to_new=self.old_to_new if self.binarization == "semantic" else None,
                ecc_enabled=self.ecc_enabled,
                ecc_total_bits_per_token=self.bits_per_token,
                ecc_include_overall_parity=bool(getattr(self.ecc, "include_overall_parity", True)),
            )

        self.token_to_rank_table = None
        if self.repr == "tokens" and self.token_space in {"semantic_rank", "semantic", "rank"}:
            rank_table = torch.empty(self.vocab_size_base, dtype=torch.long)
            for tid in range(self.vocab_size_base):
                rank_table[tid] = int(self.old_to_new.get(tid, tid))
            self.token_to_rank_table = rank_table

        print(
            f"[lm1b] split={split} cache_format={self.cache_format} "
            f"repr={self.repr} "
            f"{'binarization=' + self.binarization if self.repr == 'binary' else 'token_space=' + self.token_space} "
            f"vocab_base={self.vocab_size_base} seq_tokens={self.seq_len_tokens} "
            f"bits/token={self.bits_per_token} num_seq={self.num_sequences} "
            f"val_fraction={frac_val} boundary_mode={self.cache_boundary_mode}"
        )

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = np.array(self.mm[self.start + idx], dtype=np.int64, copy=True)
        toks_t = torch.from_numpy(row)

        if self.repr == "tokens":
            if self.token_space in {"semantic_rank", "semantic", "rank"}:
                return self.token_to_rank_table[toks_t].view(-1)
            return toks_t.view(-1)

        bits = self.token_to_bits_table[toks_t]
        return bits.view(-1)


# -----------------------------------------------------------------------------
# Dataloaders
# -----------------------------------------------------------------------------

def get_dataloaders(
    config: config_dict.ConfigDict,
    *,
    batch_size: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if str(config.data.dataset) != "LM1B":
        raise NotImplementedError("config.data.dataset must be 'LM1B' for data/lm1b.py")

    batch = int(batch_size or config.train.batch_size)

    train_ds = LM1BDataset(config, split="train")
    val_ds = LM1BDataset(config, split="val")
    test_ds = LM1BDataset(config, split="test")

    num_workers = int(getattr(config.data, "num_workers", 8))
    prefetch_factor = int(getattr(config.data, "prefetch_factor", 4))
    pin_memory = bool(getattr(config.data, "pin_memory", True))
    persistent_workers = num_workers > 0

    g = torch.Generator()
    g.manual_seed(int(seed))

    def _worker_init_fn(worker_id: int) -> None:
        base = int(seed) + int(worker_id)
        np.random.seed(base % (2**32 - 1))
        torch.manual_seed(base)

    def make_loader(ds: Dataset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            generator=g if shuffle else None,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )

    train_loader = make_loader(train_ds, shuffle=True, drop_last=True)
    val_loader = make_loader(val_ds, shuffle=False, drop_last=False)
    test_loader = make_loader(test_ds, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader