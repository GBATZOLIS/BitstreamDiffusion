"""TinyGSM training dataset + GSM8K test dataset (S-FLM parity).

Training (TinyGSM/TinyGSM, tokenizer HuggingFaceTB/SmolLM-135M):

    ids       = [BOS] + question + sep('\\n') + code + [EOS]      # padded to 512
    prompt_len = 1 + len(question) + len(sep)
    train_on_prompt = False, train_on_pad = True, filter_too_long = True

The model trains on a fixed-width binary bitstream: each of the 512 tokens is
encoded with bits_per_token = 16 bits (MSB-first) -> 8192 bits. The prompt
region (first prompt_len tokens) is conditioned on (kept clean) and excluded
from the loss; the loss therefore covers answer + EOS + padding, matching
S-FLM's train_on_pad=True. The prefix mask is per-example (variable length).

Evaluation (GSM8K test, datasets/gsm8k/gsm8k_test.json, 1319 examples):

    prompt = [BOS] + question + sep('\\n')
    sample the solution conditionally, decode the suffix, execute the code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from ml_collections import config_dict

from .task_codec import token_ids_to_bits, token_mask_to_bit_mask, bits_required
from .dist_build import rank0_first

DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM-135M"
SEP = "\n"


def get_tokenizer(name: str = DEFAULT_TOKENIZER):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        # S-FLM behaviour: add a dedicated PAD token if the tokenizer lacks one.
        tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok


def _special_ids(tok) -> Tuple[int, int, int]:
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    eos = tok.eos_token_id if tok.eos_token_id is not None else bos
    pad = tok.pad_token_id
    return int(bos), int(eos), int(pad)


# ---------------------------------------------------------------------------
# TinyGSM training cache
# ---------------------------------------------------------------------------

def build_or_load_tinygsm(
    *,
    root: str,
    tokenizer_name: str,
    block_size: int,
    val_ratio: float,
    val_seed: int,
    max_train_examples: Optional[int] = None,
    overwrite: bool = False,
):
    """Tokenize TinyGSM into a padded uint16 cache. Returns (split -> dict).

    Each split dict has:
        ids:        np.memmap/uint16 [N, block_size]
        prompt_len: np.ndarray int32 [N]
    """
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)
    safe = tokenizer_name.replace("/", "__")
    cap = "all" if max_train_examples is None else str(int(max_train_examples))
    tag = f"tinygsm_bs{block_size}_answeronly_trainonpad_filtered_{safe}_cap{cap}_vr{val_ratio}_vs{val_seed}"
    meta_path = root_p / f"{tag}.meta.json"

    def _load():
        meta = json.loads(meta_path.read_text())
        out = {}
        for split in ("train", "validation"):
            n = meta[split]["n"]
            ids = np.memmap(root_p / f"{tag}_{split}_ids.uint16", dtype=np.uint16, mode="r",
                            shape=(n, block_size))
            plen = np.fromfile(root_p / f"{tag}_{split}_plen.int32", dtype=np.int32)
            out[split] = {"ids": ids, "prompt_len": plen}
        return out

    # DDP-safe: only rank 0 tokenizes/writes the cache; others wait then load.
    with rank0_first() as is_builder:
        if is_builder and (overwrite or not meta_path.exists()):
            _build_tinygsm_cache(
                root_p=root_p, tag=tag, tokenizer_name=tokenizer_name,
                block_size=block_size, val_ratio=val_ratio, val_seed=val_seed,
                max_train_examples=max_train_examples,
            )
    return _load()


def _build_tinygsm_cache(*, root_p, tag, tokenizer_name, block_size, val_ratio,
                         val_seed, max_train_examples):
    from datasets import load_dataset

    tok = get_tokenizer(tokenizer_name)
    bos, eos, pad = _special_ids(tok)
    sep_ids = tok(SEP, add_special_tokens=False).input_ids

    BATCH = 50_000

    def _tok_rows(q_list, a_list):
        """Batch-encode (fast/rayon-parallel tokenizer) -> iterator of (q_ids, a_ids)."""
        q_enc = tok([q.strip() for q in q_list], add_special_tokens=False)["input_ids"]
        a_enc = tok([a.strip() for a in a_list], add_special_tokens=False)["input_ids"]
        return zip(q_enc, a_enc)

    def _emit(row_arr, plen_arr, w, q_ids, a_ids):
        """Assemble [BOS] q sep a [EOS] into preallocated buffers at index w.

        Buffers are pre-filled with `pad`, so we only write up to EOS (the rest
        stays padded). Returns the new write head (unchanged if filtered out).
        """
        total = 2 + len(q_ids) + len(sep_ids) + len(a_ids)  # +bos +eos
        if total > block_size:
            return w  # filter_too_long = True
        r = row_arr[w]
        p = 0
        r[p] = bos; p += 1
        r[p:p + len(q_ids)] = q_ids; p += len(q_ids)
        r[p:p + len(sep_ids)] = sep_ids; p += len(sep_ids)
        r[p:p + len(a_ids)] = a_ids; p += len(a_ids)
        r[p] = eos
        plen_arr[w] = 1 + len(q_ids) + len(sep_ids)
        return w + 1

    def _write(split_name, ids_arr, plen_arr, w):
        ids_arr[:w].tofile(root_p / f"{tag}_{split_name}_ids.uint16")
        plen_arr[:w].tofile(root_p / f"{tag}_{split_name}_plen.int32")
        return {"n": int(w)}  # _load() reads meta[split]["n"]

    meta = {}

    if max_train_examples is not None:
        # Smoke / capped path: stream a small subset (no full download).
        stream = load_dataset("TinyGSM/TinyGSM", split="train", streaming=True)
        rows = []
        for i, ex in enumerate(stream):
            if i >= int(max_train_examples):
                break
            rows.append((ex["question"], ex["code"]))
        n_val = max(1, int(len(rows) * val_ratio))
        parts = {"train": rows[:-n_val], "validation": rows[-n_val:]}
        for split_name, rws in parts.items():
            cap = len(rws)
            ids_arr = np.full((cap, block_size), pad, dtype=np.uint16)
            plen_arr = np.empty((cap,), dtype=np.int32)
            w = 0
            for i in range(0, cap, BATCH):
                chunk = rws[i:i + BATCH]
                for q_ids, a_ids in _tok_rows([c[0] for c in chunk], [c[1] for c in chunk]):
                    w = _emit(ids_arr, plen_arr, w, q_ids, a_ids)
            meta[split_name] = _write(split_name, ids_arr, plen_arr, w)
    else:
        # Full corpus: ONE sequential pass over the unshuffled arrow (sequential
        # I/O — avoids the random-access reads a shuffled train_test_split induces
        # on Lustre), routing each row into a seeded 1% val holdout. Tokenization
        # is batched (fast/rayon-parallel); preallocated uint16 buffers keep memory
        # bounded vs. building Python lists of lists.
        ds = load_dataset("TinyGSM/TinyGSM", split="train")
        N = len(ds)
        n_val = max(1, int(N * val_ratio))
        rng = np.random.default_rng(val_seed)
        is_val = np.zeros(N, dtype=bool)
        is_val[rng.permutation(N)[:n_val]] = True

        train_ids = np.full((N - n_val, block_size), pad, dtype=np.uint16)
        train_plen = np.empty((N - n_val,), dtype=np.int32)
        val_ids = np.full((n_val, block_size), pad, dtype=np.uint16)
        val_plen = np.empty((n_val,), dtype=np.int32)
        tw = vw = 0
        gi = 0
        for batch in ds.iter(batch_size=BATCH):
            for q_ids, a_ids in _tok_rows(batch["question"], batch["code"]):
                if is_val[gi]:
                    vw = _emit(val_ids, val_plen, vw, q_ids, a_ids)
                else:
                    tw = _emit(train_ids, train_plen, tw, q_ids, a_ids)
                gi += 1
        meta["train"] = _write("train", train_ids, train_plen, tw)
        meta["validation"] = _write("validation", val_ids, val_plen, vw)

    meta["tokenizer_len"] = int(len(tok))
    # Write meta last (atomically) — its presence signals a complete cache.
    meta_path = root_p / f"{tag}.meta.json"
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(meta_path)


class TinyGSMDataset(Dataset):
    is_text_dataset = True

    def __init__(self, config: config_dict.ConfigDict, *, split: str):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.split = "validation" if split in {"val", "test"} else "train"
        d = config.data
        self.block_size = int(getattr(d, "sequence_len_tokens", 512))
        self.bits_per_token = int(getattr(d, "bits_per_token", 16))
        self.tokenizer_name = str(getattr(d, "tokenizer_name", DEFAULT_TOKENIZER))
        root = str(getattr(d, "root", "datasets/tinygsm"))
        max_train = getattr(d, "max_train_examples", None)
        max_train = int(max_train) if max_train is not None else None

        cache = build_or_load_tinygsm(
            root=root,
            tokenizer_name=self.tokenizer_name,
            block_size=self.block_size,
            val_ratio=float(getattr(d, "val_ratio", 0.01)),
            val_seed=int(getattr(d, "val_seed", 42)),
            max_train_examples=max_train,
        )
        self.ids = cache[self.split]["ids"]
        self.prompt_len = cache[self.split]["prompt_len"]

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def __getitem__(self, idx: int) -> dict:
        ids = torch.from_numpy(np.asarray(self.ids[idx], dtype=np.int64))  # [block]
        plen = int(self.prompt_len[idx])
        bits = token_ids_to_bits(ids, self.bits_per_token)  # [block*bpt]
        prefix_tok = torch.arange(self.block_size) < plen
        prefix_bits = token_mask_to_bit_mask(prefix_tok, self.bits_per_token).bool()
        return {"x0": bits, "prefix_mask": prefix_bits, "input_ids": ids}


# ---------------------------------------------------------------------------
# GSM8K test set (evaluation only)
# ---------------------------------------------------------------------------

class GSM8KTestDataset(Dataset):
    """Conditioning prompts for GSM8K eval. x0 suffix is unused (masked)."""

    is_text_dataset = True

    def __init__(self, config: config_dict.ConfigDict):
        super().__init__()
        d = config.data
        self.block_size = int(getattr(d, "sequence_len_tokens", 512))
        self.bits_per_token = int(getattr(d, "bits_per_token", 16))
        self.tokenizer_name = str(getattr(d, "tokenizer_name", DEFAULT_TOKENIZER))
        path = str(getattr(d, "gsm8k_test_path", "datasets/gsm8k/gsm8k_test.json"))
        self.records = json.loads(Path(path).read_text())

        self.tok = get_tokenizer(self.tokenizer_name)
        self.bos, self.eos, self.pad = _special_ids(self.tok)
        self.sep_ids = self.tok(SEP, add_special_tokens=False).input_ids

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        q_ids = self.tok(rec["prompt"].strip(), add_special_tokens=False).input_ids
        prompt = [self.bos] + q_ids + self.sep_ids
        plen = len(prompt)
        if plen > self.block_size:
            prompt = prompt[: self.block_size]
            plen = self.block_size
        ids = prompt + [self.pad] * (self.block_size - plen)
        ids_t = torch.tensor(ids, dtype=torch.long)
        bits = token_ids_to_bits(ids_t, self.bits_per_token)
        prefix_tok = torch.arange(self.block_size) < plen
        prefix_bits = token_mask_to_bit_mask(prefix_tok, self.bits_per_token).bool()
        return {
            "x0": bits,
            "prefix_mask": prefix_bits,
            "prompt_len_tokens": plen,
            "idx": idx,
            "prompt": rec["prompt"],
            "response_ground_truth": rec["response_ground_truth"],
        }


def get_dataloaders(config, *, batch_size=None, seed: int = 42) -> Tuple[DataLoader, DataLoader, None]:
    train_ds = TinyGSMDataset(config, split="train")
    val_ds = TinyGSMDataset(config, split="val")
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
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=True, **common)
    return train_loader, val_loader, None
