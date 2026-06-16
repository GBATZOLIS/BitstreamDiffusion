"""Sudoku task dataset for BitstreamDiffusion (S-FLM parity).

Sequence layout (identical to jdeschena/s-flm):

    [BOS] puzzle(89) [BOS] solution(89)            # 180 tokens
    prompt = [BOS] puzzle [BOS]                     # first 91 tokens
    solution occupies positions 91..179             # 89 tokens

Token vocabulary (size 12):
    0      empty cell (only in the puzzle/prompt)
    1..9   digits
    10     row separator (8 per grid)
    11     BOS

Each grid is 81 cells + 8 row separators = 89 tokens.

The model trains on a fixed-width binary bitstream: each of the 180 tokens
is encoded with bits_per_token = 4 bits (MSB-first), giving 720 bits. The
prompt region is conditioned on (kept clean) and excluded from the loss,
exactly matching S-FLM's attention_mask = [0]*91 + [1]*89.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from ml_collections import config_dict

from .sudoku_generator import generate_sudoku_dataset, DIFFICULTY_TO_CLUES
from .task_codec import token_ids_to_bits, token_mask_to_bit_mask
from .dist_build import rank0_first

# ---- Canonical S-FLM constants ----
VOCAB_SIZE = 12
BITS_PER_TOKEN = 4
EMPTY_TOKEN_ID = 0
ROW_SEPARATOR_ID = 10
BOS_TOKEN_ID = 11
GRID_TOKENS = 89          # 81 cells + 8 row separators
PROMPT_LEN_TOKENS = 91    # [BOS] + puzzle(89) + [BOS]
TOTAL_LEN_TOKENS = 180    # 91 prompt + 89 solution
TOTAL_LEN_BITS = TOTAL_LEN_TOKENS * BITS_PER_TOKEN  # 720


class SudokuTokenizer:
    """Minimal tokenizer struct consumed by sudoku_generator._tokenize_grids."""

    bos_token_id = BOS_TOKEN_ID
    row_separator_id = ROW_SEPARATOR_ID
    empty_token_id = EMPTY_TOKEN_ID
    prompt_len = PROMPT_LEN_TOKENS
    seq_len = GRID_TOKENS  # per-grid (solution) length used for the loss mask


def _cache_path(root: Path, difficulty: str, num_train: int, num_valid: int, seed: int) -> Path:
    name = f"sudoku_{difficulty}_train{num_train}_valid{num_valid}_seed{seed}.pt"
    return root / name


def build_or_load_sudoku(
    *,
    root: str,
    difficulty: str,
    num_train: int,
    num_valid: int,
    seed: int,
    num_workers: int = 1,
    overwrite: bool = False,
) -> dict:
    """Generate (with disk cache) the tokenized sudoku dataset."""
    if difficulty not in DIFFICULTY_TO_CLUES:
        raise ValueError(f"difficulty must be one of {list(DIFFICULTY_TO_CLUES)}, got {difficulty!r}")
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(root_p, difficulty, num_train, num_valid, seed)

    # DDP-safe: only rank 0 generates; other ranks wait then load.
    with rank0_first() as is_builder:
        if is_builder and (overwrite or not cache.exists()):
            data = generate_sudoku_dataset(
                num_train=num_train,
                num_valid=num_valid,
                difficulty=difficulty,
                seed=seed,
                tokenizer=SudokuTokenizer(),
                num_workers=num_workers,
            )
            out = {}
            for split in ("train", "validation"):
                out[split] = torch.tensor(data[split]["input_ids"], dtype=torch.uint8)
            tmp = cache.with_suffix(cache.suffix + ".tmp")
            torch.save(out, tmp)
            tmp.replace(cache)  # atomic rename

    return torch.load(cache)


class SudokuDataset(Dataset):
    is_text_dataset = True

    def __init__(self, config: config_dict.ConfigDict, *, split: str):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.config = config
        self.split = "validation" if split in {"val", "test"} else "train"

        d = config.data
        self.difficulty = str(getattr(d, "difficulty", "easy"))
        self.num_train = int(getattr(d, "num_train", 48000))
        self.num_valid = int(getattr(d, "num_valid", 2000))
        self.seed = int(getattr(d, "data_seed", 42))
        self.bits_per_token = int(getattr(d, "bits_per_token", BITS_PER_TOKEN))
        assert self.bits_per_token == BITS_PER_TOKEN, "Sudoku uses 4 bits/token"
        root = str(getattr(d, "root", "datasets/sudoku"))
        num_workers = int(getattr(d, "sudoku_num_workers", 1))

        cache = build_or_load_sudoku(
            root=root,
            difficulty=self.difficulty,
            num_train=self.num_train,
            num_valid=self.num_valid,
            seed=self.seed,
            num_workers=num_workers,
        )
        self.ids = cache[self.split].long()  # [N, 180]
        assert self.ids.shape[1] == TOTAL_LEN_TOKENS

        # Constant prompt mask (same for every example).
        prefix_tok = torch.arange(TOTAL_LEN_TOKENS) < PROMPT_LEN_TOKENS  # [180] bool
        self._prefix_bits = token_mask_to_bit_mask(prefix_tok, self.bits_per_token).bool()  # [720]

    def __len__(self) -> int:
        return self.ids.shape[0]

    def __getitem__(self, idx: int) -> dict:
        ids = self.ids[idx]  # [180] long
        bits = token_ids_to_bits(ids, self.bits_per_token)  # [720] long
        return {
            "x0": bits,                       # [720] long 0/1
            "prefix_mask": self._prefix_bits,  # [720] bool, True over prompt
            "input_ids": ids,                  # [180] long, for eval/debug
        }


def get_dataloaders(config, *, batch_size=None, seed: int = 42) -> Tuple[DataLoader, DataLoader, None]:
    train_ds = SudokuDataset(config, split="train")
    val_ds = SudokuDataset(config, split="val")
    bs = int(batch_size or config.train.batch_size)
    nw = int(getattr(config.data, "num_workers", 4))
    common = dict(
        num_workers=nw,
        pin_memory=bool(getattr(config.data, "pin_memory", True)),
        persistent_workers=nw > 0,
    )
    if nw > 0:
        common["prefetch_factor"] = int(getattr(config.data, "prefetch_factor", 2))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, None
