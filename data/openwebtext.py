from __future__ import annotations

import itertools
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from filelock import FileLock
from ml_collections import config_dict
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from utils.ecc_secded import ecc_from_cfg, ecc_chunk_len, ecc_encode_data_bits


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def int_to_gray(n: int) -> int:
    return n ^ (n >> 1)


def _ceil_log2(x: int) -> int:
    return int(math.ceil(math.log2(max(2, int(x)))))


def _safe_name(s: str) -> str:
    return str(s).replace("/", "_").replace("-", "_").replace(":", "_")


def _ddp_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _ddp_rank_world() -> tuple[int, int]:
    if not _ddp_is_on():
        return 0, 1
    return int(dist.get_rank()), int(dist.get_world_size())


def _is_rank0() -> bool:
    rank, _ = _ddp_rank_world()
    return rank == 0


def _dist_barrier() -> None:
    if _ddp_is_on():
        dist.barrier()


# -----------------------------------------------------------------------------
# Tokenizer + semantic assets
# -----------------------------------------------------------------------------

def load_gpt2_tokenizer(tokenizer_name: str = "gpt2"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    eos_id = tok.eos_token_id
    if eos_id is None:
        raise RuntimeError(f"Tokenizer {tokenizer_name!r} does not expose eos_token_id")

    bos_id = tok.bos_token_id
    if bos_id is None:
        bos_id = eos_id

    return tok, int(bos_id), int(eos_id)


def load_openwebtext_semantic_map(
    root: Path,
    tokenizer_name: str,
) -> tuple[Dict[int, int], Dict[int, int]]:
    safe_tok = _safe_name(tokenizer_name)
    map_path = root / f"semantic_mapping_{safe_tok}.json"
    if not map_path.exists():
        raise RuntimeError(
            f"Missing semantic map at {map_path}. "
            f"Run scripts/prepare_openwebtext_semantic_map.py first."
        )

    with open(map_path, "r", encoding="utf-8") as f:
        maps = json.load(f)

    old_to_new = {int(k): int(v) for k, v in maps["old_to_new"].items()}
    new_to_old = {int(k): int(v) for k, v in maps["new_to_old"].items()}
    return old_to_new, new_to_old


def load_gpt2id_bpe16_tokenizer_and_meta(
    root: Path,
    tokenizer_path: Optional[str] = None,
    meta_path: Optional[str] = None,
):
    from tokenizers import Tokenizer

    if tokenizer_path is None:
        raise RuntimeError(
            "cfg.data.code_tokenizer_path must be set for sequence_codec='gpt2id_bpe16'"
        )

    tok_path = Path(tokenizer_path)
    if not tok_path.is_absolute():
        tok_path = root / tok_path

    if meta_path is None:
        meta_file = tok_path.with_suffix(".meta.json")
    else:
        meta_file = Path(meta_path)
        if not meta_file.is_absolute():
            meta_file = root / meta_file

    if not tok_path.exists():
        raise RuntimeError(f"Missing code tokenizer JSON: {tok_path}")
    if not meta_file.exists():
        raise RuntimeError(f"Missing code tokenizer meta: {meta_file}")

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    tok = Tokenizer.from_file(str(tok_path))
    return tok, meta


def gpt2_ids_to_pua_string(
    token_ids,
    *,
    base_char_offset: int,
) -> str:
    return "".join(chr(int(base_char_offset) + int(t)) for t in token_ids)


# -----------------------------------------------------------------------------
# Flat token cache from HF openwebtext
# -----------------------------------------------------------------------------

def _load_memmap(cache_path: Path, meta_path: Path) -> tuple[np.memmap, dict]:
    if not cache_path.exists() or not meta_path.exists():
        raise RuntimeError(f"Missing cache files: {cache_path} / {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n = int(meta["n_tokens"])
    mm = np.memmap(cache_path, dtype=np.uint16, mode="r", shape=(n,))
    return mm, meta


def build_or_load_openwebtext_flat_cache(
    *,
    root: Path,
    split_name: str,
    hf_split: str,
    tokenizer_name: str = "gpt2",
    insert_eos: bool = True,
    encode_batch_size: int = 1000,
    write_batch_size: int = 8192,
    num_proc: Optional[int] = None,
    overwrite: bool = False,
) -> tuple[np.memmap, dict]:
    """
    DDP-safe builder/loader for a flat uint16 OpenWebText token cache.
    """
    from datasets import load_dataset
    from tqdm.auto import tqdm

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    safe_tok = _safe_name(tokenizer_name)
    cache_path = root / f"cache_{split_name}_{safe_tok}_flat_eos{int(insert_eos)}.uint16"
    meta_path = root / f"cache_{split_name}_{safe_tok}_flat_eos{int(insert_eos)}.meta.json"
    tmp_path = root / f"cache_{split_name}_{safe_tok}_flat_eos{int(insert_eos)}.uint16.tmp"
    lock_path = root / f"cache_{split_name}_{safe_tok}_flat_eos{int(insert_eos)}.lock"

    if (not overwrite) and cache_path.exists() and meta_path.exists():
        print(f"[owt_cache] Reusing existing cache: {cache_path}")
        return _load_memmap(cache_path, meta_path)

    tok0, bos_id, eos_id = load_gpt2_tokenizer(tokenizer_name)
    vocab_size = int(tok0.vocab_size)

    if vocab_size > 65535:
        raise RuntimeError(f"Tokenizer vocab_size={vocab_size} too large for uint16 storage.")

    if num_proc is None:
        cpu_total = mp.cpu_count()
        fallback = cpu_total // 2 if cpu_total >= 8 else max(1, cpu_total - 1)
        num_proc = max(1, min(16, fallback))
    num_proc = max(1, int(num_proc))

    with FileLock(str(lock_path)):
        if (not overwrite) and cache_path.exists() and meta_path.exists():
            print(f"[owt_cache] Reusing existing cache after waiting: {cache_path}")
            return _load_memmap(cache_path, meta_path)

        if overwrite:
            if tmp_path.exists():
                print(f"[owt_cache] Removing stale temporary file: {tmp_path}")
                tmp_path.unlink(missing_ok=True)
            if cache_path.exists():
                print(f"[owt_cache] overwrite=True -> removing existing cache: {cache_path}")
                cache_path.unlink(missing_ok=True)
            if meta_path.exists():
                print(f"[owt_cache] overwrite=True -> removing existing meta: {meta_path}")
                meta_path.unlink(missing_ok=True)

        if tmp_path.exists():
            print(f"[owt_cache] Removing stale temporary file from interrupted run: {tmp_path}")
            tmp_path.unlink(missing_ok=True)

        print(
            f"[owt_cache] Loading HF dataset split={hf_split!r} into {root / 'hf_cache'} "
            f"(split_name={split_name}, tokenizer={tokenizer_name}, num_proc={num_proc})"
        )

        t0 = time.time()
        ds = load_dataset(
            "openwebtext",
            split=hf_split,
            cache_dir=str(root / "hf_cache"),
        )
        t_load = time.time() - t0

        try:
            n_examples = len(ds)
        except Exception:
            n_examples = None

        if n_examples is not None:
            print(f"[owt_cache] Loaded dataset with {n_examples:,} examples in {t_load:.1f}s")
        else:
            print(f"[owt_cache] Loaded dataset in {t_load:.1f}s")

        _TOK_CACHE = {"tok": None}

        def tokenize_batch(examples):
            tok = _TOK_CACHE["tok"]
            if tok is None:
                tok, _, _ = load_gpt2_tokenizer(tokenizer_name)
                _TOK_CACHE["tok"] = tok

            texts = [
                (t or "").replace("\r\n", "\n").replace("\r", "\n").strip()
                for t in examples["text"]
            ]

            enc = tok(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            out_ids = []
            for ids in enc["input_ids"]:
                if not ids:
                    out_ids.append([])
                else:
                    out_ids.append(ids + [eos_id] if insert_eos else ids)
            return {"input_ids": out_ids}

        print(
            f"[owt_cache] Tokenizing split={split_name} "
            f"(batch_size={encode_batch_size}, num_proc={num_proc})..."
        )
        t1 = time.time()

        tokenized_ds = ds.map(
            tokenize_batch,
            batched=True,
            batch_size=int(encode_batch_size),
            num_proc=num_proc,
            remove_columns=ds.column_names,
            desc=f"Tokenizing {split_name}",
            load_from_cache_file=True,
            keep_in_memory=False,
        )

        t_tok = time.time() - t1
        print(f"[owt_cache] Tokenization finished in {t_tok / 60.0:.2f} min")

        n_tokens = 0
        n_docs = 0

        print(
            f"[owt_cache] Flattening + writing uint16 stream to {cache_path.name} "
            f"(write_batch_size={write_batch_size})..."
        )
        t2 = time.time()

        total_rows = len(tokenized_ds) if hasattr(tokenized_ds, "__len__") else None

        try:
            with open(tmp_path, "wb") as fout:
                row_bar = tqdm(
                    total=total_rows,
                    desc=f"Writing {split_name}",
                    unit="docs",
                    dynamic_ncols=True,
                )

                for batch in tokenized_ds.iter(batch_size=int(write_batch_size)):
                    ids_batch = batch["input_ids"]

                    valid_docs = [ids for ids in ids_batch if ids]
                    if valid_docs:
                        flat_ids = list(itertools.chain.from_iterable(valid_docs))
                        arr = np.asarray(flat_ids, dtype=np.uint16)
                        fout.write(arr.tobytes())
                        n_tokens += int(arr.size)
                        n_docs += int(len(valid_docs))

                    row_bar.update(len(ids_batch))
                    row_bar.set_postfix(
                        docs=f"{n_docs:,}",
                        toks=f"{n_tokens:,}",
                    )

                row_bar.close()

            if n_tokens <= 0:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"No tokens written for split_name={split_name}, hf_split={hf_split}")

            os.replace(tmp_path, cache_path)

            t_write = time.time() - t2
            t_total = time.time() - t0

            meta = {
                "dataset_name": "openwebtext",
                "split_name": split_name,
                "hf_split": hf_split,
                "tokenizer_name": tokenizer_name,
                "vocab_size": vocab_size,
                "n_tokens": int(n_tokens),
                "n_docs": int(n_docs),
                "insert_eos": bool(insert_eos),
                "bos_token_id": int(bos_id),
                "eos_token_id": int(eos_id),
                "num_proc": int(num_proc),
                "encode_batch_size": int(encode_batch_size),
                "write_batch_size": int(write_batch_size),
                "build_seconds_total": float(t_total),
                "build_seconds_load_dataset": float(t_load),
                "build_seconds_tokenize": float(t_tok),
                "build_seconds_write": float(t_write),
            }

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            toks_per_sec = n_tokens / max(t_total, 1e-8)
            docs_per_sec = n_docs / max(t_total, 1e-8)
            print(
                f"[owt_cache] Done. docs={n_docs:,} toks={n_tokens:,} "
                f"time={t_total / 60.0:.2f} min "
                f"({docs_per_sec:.1f} docs/s, {toks_per_sec:.1f} toks/s)"
            )

        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        return _load_memmap(cache_path, meta_path)


# -----------------------------------------------------------------------------
# Fixed-length second-stage code cache for gpt2id_bpe16
# -----------------------------------------------------------------------------

def build_or_load_openwebtext_code_cache(
    *,
    root: Path,
    split_name: str,
    base_mm: np.memmap,
    base_meta: dict,
    tokenizer_name: str,
    code_tokenizer_path: str,
    code_tokenizer_meta_path: Optional[str],
    base_sequence_len_tokens: int,
    code_sequence_len_tokens: int,
    batch_size: int = 2048,
    overwrite: bool = False,
) -> tuple[np.memmap, dict]:
    from tqdm.auto import tqdm

    root = Path(root)
    safe_tok = _safe_name(tokenizer_name)

    code_tok, code_meta = load_gpt2id_bpe16_tokenizer_and_meta(
        root=root,
        tokenizer_path=code_tokenizer_path,
        meta_path=code_tokenizer_meta_path,
    )

    code_vocab_size = int(code_meta["actual_code_vocab_size"])
    if code_vocab_size > 65536:
        raise RuntimeError(f"Code vocab too large for uint16 cache: {code_vocab_size}")

    pad_id = int(code_meta["pad_id"])
    eoseq_id = int(code_meta["eoseq_id"])
    base_char_offset = int(code_meta["base_char_offset"])

    content_len = int(base_sequence_len_tokens) - 2
    if content_len <= 0:
        raise RuntimeError("base_sequence_len_tokens must be >= 3")

    n_base_tokens = int(base_mm.shape[0])
    n_seq = n_base_tokens // content_len
    if n_seq <= 0:
        raise RuntimeError("No sequences available for code-cache construction")

    cache_path = root / (
        f"cache_{split_name}_{safe_tok}_gpt2id_bpe16_"
        f"v{code_vocab_size}_base{base_sequence_len_tokens}_len{code_sequence_len_tokens}.uint16"
    )
    meta_path = cache_path.with_suffix(".meta.json")
    lock_path = root / (cache_path.name + ".lock")
    tmp_path = root / (cache_path.name + ".tmp")

    if (not overwrite) and cache_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        mm = np.memmap(
            cache_path,
            dtype=np.uint16,
            mode="r",
            shape=(int(meta["num_sequences"]), int(meta["code_sequence_len_tokens"])),
        )
        print(f"[owt_code_cache] Reusing existing cache: {cache_path}")
        return mm, meta

    with FileLock(str(lock_path)):
        if (not overwrite) and cache_path.exists() and meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            mm = np.memmap(
                cache_path,
                dtype=np.uint16,
                mode="r",
                shape=(int(meta["num_sequences"]), int(meta["code_sequence_len_tokens"])),
            )
            print(f"[owt_code_cache] Reusing existing cache after waiting: {cache_path}")
            return mm, meta

        if overwrite:
            cache_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            tmp_path.unlink(missing_ok=True)

        if tmp_path.exists():
            print(f"[owt_code_cache] Removing stale temporary file: {tmp_path}")
            tmp_path.unlink(missing_ok=True)

        bos_id = int(base_meta["bos_token_id"])
        eos_id = int(base_meta["eos_token_id"])
        bos_ch = chr(base_char_offset + bos_id)
        eos_ch = chr(base_char_offset + eos_id)

        print(
            f"[owt_code_cache] Building code cache for split={split_name} "
            f"num_sequences={n_seq:,}, batch_size={int(batch_size)} -> {cache_path.name}"
        )

        arr = np.memmap(
            tmp_path,
            dtype=np.uint16,
            mode="w+",
            shape=(n_seq, int(code_sequence_len_tokens)),
        )

        try:
            for start in tqdm(
                range(0, n_seq, int(batch_size)),
                desc=f"Encoding {split_name} code-cache",
                dynamic_ncols=True,
            ):
                end = min(start + int(batch_size), n_seq)

                batch_strings = []
                for i in range(start, end):
                    s = i * content_len
                    e = s + content_len
                    content = base_mm[s:e]
                    body = "".join(map(chr, content.astype(np.uint32, copy=False) + base_char_offset))
                    batch_strings.append(bos_ch + body + eos_ch)

                encs = code_tok.encode_batch(batch_strings)

                for j, enc in enumerate(encs):
                    ids = list(enc.ids)
                    ids.append(eoseq_id)

                    if len(ids) > int(code_sequence_len_tokens):
                        ids = ids[: int(code_sequence_len_tokens)]
                        ids[-1] = eoseq_id

                    if len(ids) < int(code_sequence_len_tokens):
                        ids.extend([pad_id] * (int(code_sequence_len_tokens) - len(ids)))

                    arr[start + j, :] = np.asarray(ids, dtype=np.uint16)

            arr.flush()
            del arr
            os.replace(tmp_path, cache_path)

        except Exception:
            try:
                del arr
            except Exception:
                pass
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        meta = {
            "format": "openwebtext_gpt2id_bpe16_cache",
            "split_name": split_name,
            "num_sequences": int(n_seq),
            "base_sequence_len_tokens": int(base_sequence_len_tokens),
            "code_sequence_len_tokens": int(code_sequence_len_tokens),
            "code_vocab_size": int(code_vocab_size),
            "pad_id": int(pad_id),
            "eoseq_id": int(eoseq_id),
            "source_flat_hf_split": base_meta.get("hf_split"),
            "source_flat_cache_tokens": int(base_mm.shape[0]),
            "code_tokenizer_path": str(code_tokenizer_path),
            "code_tokenizer_meta_path": str(code_tokenizer_meta_path) if code_tokenizer_meta_path is not None else None,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        mm = np.memmap(
            cache_path,
            dtype=np.uint16,
            mode="r",
            shape=(int(meta["num_sequences"]), int(meta["code_sequence_len_tokens"])),
        )
        print(f"[owt_code_cache] Finished building: {cache_path}")
        return mm, meta


# -----------------------------------------------------------------------------
# Cache prebuild coordinator
# -----------------------------------------------------------------------------

def ensure_openwebtext_caches_ready(
    config: config_dict.ConfigDict,
    *,
    split: str,
) -> None:
    """
    Ensure all caches required for this split exist.

    In DDP:
      - rank 0 builds if needed
      - all ranks barrier
      - nonzero ranks then proceed to load only

    Outside DDP:
      - build/load directly
    """
    assert split in {"train", "val", "test"}

    root = Path(getattr(config.data, "root", "datasets/openwebtext"))
    tokenizer_name = str(getattr(config.data, "tokenizer_name", "gpt2"))

    sequence_codec = str(getattr(config.data, "sequence_codec", "base")).lower().strip()

    train_split = str(getattr(config.data, "train_split", "train[:-100000]"))
    valid_split = str(getattr(config.data, "valid_split", "train[-100000:]"))
    insert_train_eos = bool(getattr(config.data, "insert_train_eos", True))
    insert_valid_eos = bool(getattr(config.data, "insert_valid_eos", True))
    encode_batch_size = int(getattr(config.data, "cache_encode_batch_size", 1000))
    write_batch_size = int(getattr(config.data, "cache_write_batch_size", 8192))
    cache_num_proc = getattr(config.data, "cache_num_proc", None)
    cache_overwrite = bool(getattr(config.data, "cache_overwrite", False))

    base_sequence_len_tokens = int(
        getattr(config.data, "base_sequence_len_tokens", getattr(config.data, "sequence_len_tokens", 1024))
    )
    code_sequence_len_tokens = int(getattr(config.data, "sequence_len_tokens", 1024))
    code_cache_batch_size = int(getattr(config.data, "code_cache_batch_size", 2048))
    code_cache_overwrite = bool(getattr(config.data, "code_cache_overwrite", False))

    if split == "train":
        split_name = "train"
        hf_split = train_split
        insert_eos = insert_train_eos
    else:
        split_name = "val"
        hf_split = valid_split
        insert_eos = insert_valid_eos

    def _build_all():
        base_mm, base_meta = build_or_load_openwebtext_flat_cache(
            root=root,
            split_name=split_name,
            hf_split=hf_split,
            tokenizer_name=tokenizer_name,
            insert_eos=insert_eos,
            encode_batch_size=encode_batch_size,
            write_batch_size=write_batch_size,
            num_proc=cache_num_proc,
            overwrite=cache_overwrite,
        )

        if sequence_codec == "gpt2id_bpe16":
            code_tokenizer_path = str(getattr(config.data, "code_tokenizer_path"))
            code_tokenizer_meta_path = getattr(config.data, "code_tokenizer_meta_path", None)

            build_or_load_openwebtext_code_cache(
                root=root,
                split_name=split_name,
                base_mm=base_mm,
                base_meta=base_meta,
                tokenizer_name=tokenizer_name,
                code_tokenizer_path=code_tokenizer_path,
                code_tokenizer_meta_path=code_tokenizer_meta_path,
                base_sequence_len_tokens=base_sequence_len_tokens,
                code_sequence_len_tokens=code_sequence_len_tokens,
                batch_size=code_cache_batch_size,
                overwrite=code_cache_overwrite,
            )

    if not _ddp_is_on():
        _build_all()
        return

    rank, world = _ddp_rank_world()
    if _is_rank0():
        print(f"[owt_cache][rank0/{world}] Ensuring caches for split={split} are ready...")
        _build_all()

    _dist_barrier()


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
    tokenizer_name: str,
) -> torch.Tensor:
    binarization = str(binarization).lower().strip()

    safe_tok = _safe_name(tokenizer_name)
    map_path = root / f"semantic_mapping_{safe_tok}.json"
    mtime = int(map_path.stat().st_mtime_ns) if (binarization == "semantic" and map_path.exists()) else 0

    key = (
        str(root.resolve()),
        str(binarization),
        int(vocab_size),
        int(data_bits_per_token),
        int(ecc_total_bits_per_token),
        int(1 if ecc_enabled else 0),
        int(1 if ecc_include_overall_parity else 0),
        int(mtime),
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
                raise RuntimeError("semantic binarization requested but semantic map is missing")
            code = int_to_gray(int(old_to_new.get(tid, tid)))
        elif binarization == "raw_binary":
            code = int(tid)
        else:
            raise ValueError(
                f"Unknown binarization={binarization!r}. Supported: 'raw_binary', 'semantic'."
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
                f"ECC output length mismatch: got {out_table.shape[-1]}, expected {L}"
            )

    _TOKENBITS_CACHE[key] = out_table
    return out_table


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class OpenWebTextDataset(Dataset):
    """
    Paper-faithful OpenWebText setup with optional second-stage fixed-length
    gpt2id_bpe16 codec.

    Base path:
      [BOS] + content_tokens + [EOS]  in GPT-2 space

    Optional codec path:
      fixed GPT-2 sequence -> gpt2id_bpe16 code ids -> fixed code tokens
    """

    is_text_dataset = True

    def __init__(self, config: config_dict.ConfigDict, *, split: str):
        super().__init__()
        assert split in {"train", "val", "test"}

        self.config = config
        self.split = split

        self.root = Path(getattr(config.data, "root", "datasets/openwebtext"))
        self.tokenizer_name = str(getattr(config.data, "tokenizer_name", "gpt2"))

        self.sequence_codec = str(getattr(config.data, "sequence_codec", "base")).lower().strip()
        if self.sequence_codec not in {"base", "gpt2id_bpe16"}:
            raise ValueError(f"Unsupported cfg.data.sequence_codec={self.sequence_codec!r}")

        self.repr = str(getattr(config.data, "representation", "binary")).lower().strip()
        self.binarization = str(getattr(config.data, "binarization", "raw_binary")).lower().strip()
        self.token_space = str(getattr(config.data, "token_space", "tokenizer_id")).lower().strip()

        if self.repr == "binary":
            if self.binarization not in {"raw_binary", "semantic"}:
                raise ValueError(
                    f"Unsupported cfg.data.binarization={self.binarization!r}; "
                    f"expected 'raw_binary' or 'semantic'."
                )
            if self.sequence_codec == "gpt2id_bpe16" and self.binarization != "raw_binary":
                raise ValueError(
                    "For sequence_codec='gpt2id_bpe16', use cfg.data.binarization='raw_binary'."
                )
        elif self.repr == "tokens":
            if self.token_space not in {
                "tokenizer_id",
                "tokenizer",
                "raw",
                "semantic_rank",
                "semantic",
                "rank",
            }:
                raise ValueError(f"Unsupported cfg.data.token_space={self.token_space!r}")
        else:
            raise ValueError(f"Unsupported representation={self.repr!r}")

        self.tokenizer, self.bos_token_id, self.eos_token_id = load_gpt2_tokenizer(self.tokenizer_name)
        self.vocab_size_base = int(self.tokenizer.vocab_size)

        self.old_to_new: Dict[int, int] = {}
        self.new_to_old: Dict[int, int] = {}
        need_semantic_map = (
            (self.repr == "binary" and self.binarization == "semantic")
            or (self.repr == "tokens" and self.token_space in {"semantic_rank", "semantic", "rank"})
        )
        if need_semantic_map:
            self.old_to_new, self.new_to_old = load_openwebtext_semantic_map(
                self.root, self.tokenizer_name
            )

        self.ecc = ecc_from_cfg(config)
        self.ecc_enabled = bool(self.ecc.enabled)

        data_bits_default = _ceil_log2(self.vocab_size_base)
        data_bits_cfg = getattr(config.data, "bits_per_token", None)
        self.data_bits_per_token = int(data_bits_cfg) if data_bits_cfg is not None else int(data_bits_default)

        self.base_seq_len_tokens = int(
            getattr(config.data, "base_sequence_len_tokens", getattr(config.data, "sequence_len_tokens", 1024))
        )
        self.seq_len_tokens = int(getattr(config.data, "sequence_len_tokens", 1024))

        min_bits_needed = 16 if self.sequence_codec == "gpt2id_bpe16" else _ceil_log2(self.vocab_size_base)
        if self.data_bits_per_token < min_bits_needed:
            raise ValueError(
                f"Need at least {min_bits_needed} bits/token, "
                f"but got cfg.data.bits_per_token={self.data_bits_per_token}."
            )

        if self.ecc_enabled:
            if int(self.ecc.data_bits) != int(self.data_bits_per_token):
                raise ValueError(
                    f"ECC enabled but cfg.data.ecc.data_bits={int(self.ecc.data_bits)} "
                    f"!= bits_per_token={int(self.data_bits_per_token)}"
                )
            self.bits_per_token = int(ecc_chunk_len(self.ecc))
        else:
            self.bits_per_token = int(self.data_bits_per_token)

        self.wrap = bool(getattr(config.data, "wrap", True))
        if not self.wrap:
            raise ValueError("OpenWebText requires cfg.data.wrap=True")

        if self.base_seq_len_tokens < 3:
            raise ValueError("base_sequence_len_tokens must be at least 3")

        self.base_content_len_tokens = self.base_seq_len_tokens - 2
        self.seq_len_bits = self.seq_len_tokens * self.bits_per_token

        train_split = str(getattr(config.data, "train_split", "train[:-100000]"))
        valid_split = str(getattr(config.data, "valid_split", "train[-100000:]"))
        insert_train_eos = bool(getattr(config.data, "insert_train_eos", True))
        insert_valid_eos = bool(getattr(config.data, "insert_valid_eos", True))
        encode_batch_size = int(getattr(config.data, "cache_encode_batch_size", 1000))
        write_batch_size = int(getattr(config.data, "cache_write_batch_size", 8192))
        cache_num_proc = getattr(config.data, "cache_num_proc", None)
        cache_overwrite = bool(getattr(config.data, "cache_overwrite", False))

        if split == "train":
            self.mm, self.meta = build_or_load_openwebtext_flat_cache(
                root=self.root,
                split_name="train",
                hf_split=train_split,
                tokenizer_name=self.tokenizer_name,
                insert_eos=insert_train_eos,
                encode_batch_size=encode_batch_size,
                write_batch_size=write_batch_size,
                num_proc=cache_num_proc,
                overwrite=cache_overwrite,
            )
            cache_split_name = "train"
        else:
            self.mm, self.meta = build_or_load_openwebtext_flat_cache(
                root=self.root,
                split_name="val",
                hf_split=valid_split,
                tokenizer_name=self.tokenizer_name,
                insert_eos=insert_valid_eos,
                encode_batch_size=encode_batch_size,
                write_batch_size=write_batch_size,
                num_proc=cache_num_proc,
                overwrite=cache_overwrite,
            )
            cache_split_name = "val"

        self.n_tokens = int(self.mm.shape[0])

        self.code_tokenizer = None
        self.code_meta = None
        self.code_mm = None
        self.code_cache_meta = None
        self.code_pad_id = None
        self.code_eoseq_id = None
        self.code_vocab_size = None

        if self.sequence_codec == "gpt2id_bpe16":
            code_tokenizer_path = str(getattr(config.data, "code_tokenizer_path"))
            code_tokenizer_meta_path = getattr(config.data, "code_tokenizer_meta_path", None)

            self.code_tokenizer, self.code_meta = load_gpt2id_bpe16_tokenizer_and_meta(
                root=self.root,
                tokenizer_path=code_tokenizer_path,
                meta_path=code_tokenizer_meta_path,
            )

            self.code_vocab_size = int(self.code_meta["actual_code_vocab_size"])
            self.code_pad_id = int(self.code_meta["pad_id"])
            self.code_eoseq_id = int(self.code_meta["eoseq_id"])

            self.code_mm, self.code_cache_meta = build_or_load_openwebtext_code_cache(
                root=self.root,
                split_name=cache_split_name,
                base_mm=self.mm,
                base_meta=self.meta,
                tokenizer_name=self.tokenizer_name,
                code_tokenizer_path=code_tokenizer_path,
                code_tokenizer_meta_path=code_tokenizer_meta_path,
                base_sequence_len_tokens=int(self.base_seq_len_tokens),
                code_sequence_len_tokens=int(self.seq_len_tokens),
                batch_size=int(getattr(config.data, "code_cache_batch_size", 2048)),
                overwrite=bool(getattr(config.data, "code_cache_overwrite", False)),
            )

            self.num_sequences = int(self.code_mm.shape[0])
            self.vocab_size_model = int(self.code_vocab_size)
        else:
            self.num_sequences = self.n_tokens // self.base_content_len_tokens
            self.vocab_size_model = int(self.vocab_size_base)

        if self.num_sequences <= 0:
            raise RuntimeError(
                f"Split {split!r} too small: n_tokens={self.n_tokens}, "
                f"base_content_len_tokens={self.base_content_len_tokens}"
            )

        self.token_to_bits_table = None
        if self.repr == "binary":
            self.token_to_bits_table = _get_token_to_bits_table(
                root=self.root,
                vocab_size=self.vocab_size_model,
                data_bits_per_token=self.data_bits_per_token,
                binarization=self.binarization,
                old_to_new=self.old_to_new if self.binarization == "semantic" else None,
                ecc_enabled=self.ecc_enabled,
                ecc_total_bits_per_token=self.bits_per_token,
                ecc_include_overall_parity=bool(getattr(self.ecc, "include_overall_parity", True)),
                tokenizer_name=self.tokenizer_name,
            )

        self.token_to_rank_table = None
        if self.repr == "tokens" and self.token_space in {"semantic_rank", "semantic", "rank"}:
            if self.sequence_codec != "base":
                raise ValueError(
                    "token_space semantic rank is only supported for sequence_codec='base'."
                )
            rank_table = torch.empty(self.vocab_size_base, dtype=torch.long)
            for tid in range(self.vocab_size_base):
                rank_table[tid] = int(self.old_to_new.get(tid, tid))
            self.token_to_rank_table = rank_table

        print(
            f"[owt] split={split} codec={self.sequence_codec} repr={self.repr} "
            f"{'binarization=' + self.binarization if self.repr == 'binary' else 'token_space=' + self.token_space} "
            f"tokenizer={self.tokenizer_name} vocab_base={self.vocab_size_base} vocab_model={self.vocab_size_model} "
            f"base_seq_tokens={self.base_seq_len_tokens} seq_tokens={self.seq_len_tokens} "
            f"bits/token={self.bits_per_token} seq_bits={self.seq_len_bits} "
            f"n_flat_tokens={self.n_tokens} num_seq={self.num_sequences} "
            f"hf_split={self.meta['hf_split']}"
        )

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.sequence_codec == "gpt2id_bpe16":
            toks_t = torch.from_numpy(np.array(self.code_mm[idx], dtype=np.uint16, copy=True)).long()

            if self.repr == "tokens":
                return toks_t.view(-1)

            bits = self.token_to_bits_table[toks_t]
            return bits.view(-1)

        s = idx * self.base_content_len_tokens
        e = s + self.base_content_len_tokens

        content = np.array(self.mm[s:e], dtype=np.uint16, copy=True)
        content_t = torch.from_numpy(content).long()

        toks_t = torch.empty(self.base_seq_len_tokens, dtype=torch.long)
        toks_t[0] = int(self.bos_token_id)
        toks_t[1:-1] = content_t
        toks_t[-1] = int(self.eos_token_id)

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
    dataset_name = str(config.data.dataset)
    if dataset_name not in {"OpenWebText", "openwebtext"}:
        raise NotImplementedError(
            "config.data.dataset must be 'OpenWebText' or 'openwebtext' for this file."
        )

    # Ensure required caches exist before each rank instantiates datasets.
    ensure_openwebtext_caches_ready(config, split="train")
    ensure_openwebtext_caches_ready(config, split="val")

    batch = int(batch_size or config.train.batch_size)

    train_ds = OpenWebTextDataset(config, split="train")
    val_ds = OpenWebTextDataset(config, split="val")
    test_ds = OpenWebTextDataset(config, split="test")

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

    rank, world_size = _ddp_rank_world()

    def make_loader(ds: Dataset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        sampler = None
        loader_shuffle = shuffle

        if _ddp_is_on():
            sampler = DistributedSampler(
                ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=int(seed),
            )
            loader_shuffle = False

        return DataLoader(
            ds,
            batch_size=batch,
            shuffle=loader_shuffle,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            generator=g if (shuffle and sampler is None) else None,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )

    train_loader = make_loader(train_ds, shuffle=True, drop_last=True)
    val_loader = make_loader(val_ds, shuffle=False, drop_last=False)
    test_loader = make_loader(test_ds, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader