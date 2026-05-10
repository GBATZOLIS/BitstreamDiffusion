from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import numpy as np
from tqdm import tqdm


def count_nonempty_lines(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def iter_nonempty_lines(path: Path) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def batched_texts(lines: Iterable[str], batch_size: int) -> Iterator[List[str]]:
    batch: List[str] = []
    for line in lines:
        batch.append(line)
        if len(batch) >= int(batch_size):
            yield batch
            batch = []
    if batch:
        yield batch


def get_boundary_token_id(tokenizer, boundary_mode: Optional[str]) -> Optional[int]:
    if boundary_mode is None:
        return None

    mode = str(boundary_mode).lower().strip()
    if mode in {"none", ""}:
        return None

    if mode == "sep":
        tid = getattr(tokenizer, "sep_token_id", None)
        if tid is None:
            raise RuntimeError(
                f"Requested boundary_mode='sep' but tokenizer {tokenizer.name_or_path} "
                "does not expose sep_token_id."
            )
        return int(tid)

    if mode == "eos":
        tid = getattr(tokenizer, "eos_token_id", None)
        if tid is None:
            raise RuntimeError(
                f"Requested boundary_mode='eos' but tokenizer {tokenizer.name_or_path} "
                "does not expose eos_token_id."
            )
        return int(tid)

    if mode == "cls":
        tid = getattr(tokenizer, "cls_token_id", None)
        if tid is None:
            raise RuntimeError(
                f"Requested boundary_mode='cls' but tokenizer {tokenizer.name_or_path} "
                "does not expose cls_token_id."
            )
        return int(tid)

    raise ValueError(
        f"Unknown boundary_mode={boundary_mode!r}. Supported: none, sep, eos, cls."
    )


def tokenize_batch(
    tokenizer,
    texts: List[str],
    *,
    add_special_tokens: bool,
) -> List[List[int]]:
    enc = tokenizer(
        texts,
        add_special_tokens=bool(add_special_tokens),
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = enc["input_ids"]
    if not isinstance(ids, list):
        raise RuntimeError("Tokenizer returned unexpected input_ids structure.")
    return ids


def count_total_stream_tokens(
    *,
    text_path: Path,
    tokenizer,
    seq_len: int,
    add_special_tokens: bool,
    batch_texts: int,
    boundary_token_id: Optional[int],
    keep_remainder: bool,
) -> dict:
    del seq_len  # kept for signature symmetry / future-proofing

    n_nonempty_lines = count_nonempty_lines(text_path)

    total_stream_tokens = 0
    n_nonempty_examples = 0

    for batch in tqdm(
        batched_texts(iter_nonempty_lines(text_path), batch_texts),
        total=math.ceil(n_nonempty_lines / batch_texts) if n_nonempty_lines > 0 else 0,
        desc=f"Counting tokens in {text_path.name}",
    ):
        batch_ids = tokenize_batch(
            tokenizer,
            batch,
            add_special_tokens=add_special_tokens,
        )
        for ids in batch_ids:
            if not ids:
                continue
            if n_nonempty_examples > 0 and boundary_token_id is not None:
                total_stream_tokens += 1
            total_stream_tokens += len(ids)
            n_nonempty_examples += 1

    if keep_remainder:
        n_sequences = int(math.ceil(total_stream_tokens / 128)) if total_stream_tokens > 0 else 0
    else:
        n_sequences = total_stream_tokens // 128

    return {
        "n_nonempty_lines": int(n_nonempty_lines),
        "n_nonempty_examples": int(n_nonempty_examples),
        "total_stream_tokens": int(total_stream_tokens),
        "n_sequences_if_seq_len_128": int(n_sequences),
    }


def build_packed_cache(
    *,
    text_path: Path,
    cache_path: Path,
    meta_path: Path,
    tokenizer_name: str,
    seq_len: int,
    add_special_tokens: bool,
    batch_texts: int,
    boundary_mode: Optional[str],
    keep_remainder: bool,
    force: bool,
) -> None:
    if cache_path.exists() and meta_path.exists() and (not force):
        print(f"[lm1b-cache] exists -> {cache_path} (skip; pass --force to rebuild)")
        return

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit("Missing transformers. Install: pip install transformers") from e

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    vocab_size = int(tok.vocab_size)
    if vocab_size > 65535:
        raise RuntimeError(f"vocab_size={vocab_size} too large for uint16 memmap")

    if tok.pad_token_id is None:
        raise RuntimeError(
            f"Tokenizer {tokenizer_name} does not expose a pad_token_id. "
            "A pad token is required for optional tail-padding metadata consistency."
        )

    boundary_token_id = get_boundary_token_id(tok, boundary_mode)

    stats = count_total_stream_tokens(
        text_path=text_path,
        tokenizer=tok,
        seq_len=seq_len,
        add_special_tokens=add_special_tokens,
        batch_texts=batch_texts,
        boundary_token_id=boundary_token_id,
        keep_remainder=keep_remainder,
    )

    total_stream_tokens = int(stats["total_stream_tokens"])
    if keep_remainder:
        n_sequences = int(math.ceil(total_stream_tokens / seq_len)) if total_stream_tokens > 0 else 0
    else:
        n_sequences = total_stream_tokens // seq_len

    dropped_tail_tokens = 0
    padded_tail_tokens = 0
    if keep_remainder:
        padded_tail_tokens = (n_sequences * seq_len - total_stream_tokens) if n_sequences > 0 else 0
    else:
        dropped_tail_tokens = total_stream_tokens - n_sequences * seq_len

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(cache_path, dtype=np.uint16, mode="w+", shape=(n_sequences, seq_len))

    n_nonempty_lines = int(stats["n_nonempty_lines"])
    row = 0
    buffer: List[int] = []
    seen_any_example = False

    for batch in tqdm(
        batched_texts(iter_nonempty_lines(text_path), batch_texts),
        total=math.ceil(n_nonempty_lines / batch_texts) if n_nonempty_lines > 0 else 0,
        desc=f"Packing {text_path.name}",
    ):
        batch_ids = tokenize_batch(
            tok,
            batch,
            add_special_tokens=add_special_tokens,
        )

        for ids in batch_ids:
            if not ids:
                continue

            if seen_any_example and boundary_token_id is not None:
                buffer.append(int(boundary_token_id))

            buffer.extend(int(x) for x in ids)
            seen_any_example = True

            while len(buffer) >= seq_len:
                arr[row] = np.asarray(buffer[:seq_len], dtype=np.uint16)
                del buffer[:seq_len]
                row += 1

    if keep_remainder and len(buffer) > 0:
        padded = buffer + [int(tok.pad_token_id)] * (seq_len - len(buffer))
        arr[row] = np.asarray(padded, dtype=np.uint16)
        row += 1

    arr.flush()

    if row != n_sequences:
        raise RuntimeError(
            f"Packed cache row mismatch for {text_path.name}: wrote {row}, expected {n_sequences}."
        )

    meta = {
        "text_path": str(text_path),
        "cache_path": str(cache_path),
        "cache_format": "packed_token_blocks",
        "packing_strategy": "concatenate_examples_then_chunk_fixed_length",
        "seq_len_tokens": int(seq_len),
        "n_sequences": int(n_sequences),
        "tokenizer_name": tokenizer_name,
        "vocab_size": int(vocab_size),
        "add_special_tokens": bool(add_special_tokens),
        "packing_boundary_mode": None if boundary_mode is None else str(boundary_mode),
        "packing_boundary_token_id": None if boundary_token_id is None else int(boundary_token_id),
        "keep_remainder": bool(keep_remainder),
        "dropped_tail_tokens": int(dropped_tail_tokens),
        "padded_tail_tokens": int(padded_tail_tokens),
        "total_stream_tokens": int(total_stream_tokens),
        "n_nonempty_lines": int(stats["n_nonempty_lines"]),
        "n_nonempty_examples": int(stats["n_nonempty_examples"]),
        "pad_token_id": int(tok.pad_token_id),
        "unk_token_id": None if tok.unk_token_id is None else int(tok.unk_token_id),
        "cls_token_id": None if tok.cls_token_id is None else int(tok.cls_token_id),
        "sep_token_id": None if tok.sep_token_id is None else int(tok.sep_token_id),
        "mask_token_id": None if tok.mask_token_id is None else int(tok.mask_token_id),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[lm1b-cache] wrote packed cache: {n_sequences:,} x {seq_len} -> {cache_path}")
    print(
        f"[lm1b-cache] total_stream_tokens={total_stream_tokens:,} "
        f"dropped_tail_tokens={dropped_tail_tokens:,} padded_tail_tokens={padded_tail_tokens:,}"
    )
    print(f"[lm1b-cache] meta -> {meta_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="datasets/lm1b")
    ap.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    ap.add_argument("--seq_len_tokens", type=int, default=128)
    ap.add_argument("--batch_texts", type=int, default=2048)

    # For the benchmark we want raw benchmark tokenizer ids packed into fixed blocks.
    ap.add_argument("--add_special_tokens", action="store_true")

    # Recommended for LM1B packed benchmark: insert [SEP] between raw examples.
    ap.add_argument(
        "--boundary_mode",
        type=str,
        default="sep",
        choices=["none", "sep", "eos", "cls"],
        help="Boundary token inserted between consecutive raw examples before chunking.",
    )

    # Benchmark default: drop final incomplete tail rather than pad it.
    ap.add_argument(
        "--keep_remainder",
        action="store_true",
        help="If set, keep the last incomplete block and right-pad it with pad_token_id. "
             "For the benchmark this should usually stay OFF.",
    )

    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    train_txt = root / "train.txt"
    test_txt = root / "test.txt"

    if not train_txt.exists() or not test_txt.exists():
        raise SystemExit(
            "Missing train.txt / test.txt. Run scripts/download_lm1b.py first.\n"
            f"Expected: {train_txt} and {test_txt}"
        )

    boundary_mode = None if str(args.boundary_mode).lower() == "none" else str(args.boundary_mode)

    build_packed_cache(
        text_path=train_txt,
        cache_path=root / "cache_train_tokens.uint16",
        meta_path=root / "cache_train_tokens.meta.json",
        tokenizer_name=args.tokenizer_name,
        seq_len=int(args.seq_len_tokens),
        add_special_tokens=bool(args.add_special_tokens),
        batch_texts=int(args.batch_texts),
        boundary_mode=boundary_mode,
        keep_remainder=bool(args.keep_remainder),
        force=bool(args.force),
    )
    build_packed_cache(
        text_path=test_txt,
        cache_path=root / "cache_test_tokens.uint16",
        meta_path=root / "cache_test_tokens.meta.json",
        tokenizer_name=args.tokenizer_name,
        seq_len=int(args.seq_len_tokens),
        add_special_tokens=bool(args.add_special_tokens),
        batch_texts=int(args.batch_texts),
        boundary_mode=boundary_mode,
        keep_remainder=bool(args.keep_remainder),
        force=bool(args.force),
    )


if __name__ == "__main__":
    main()