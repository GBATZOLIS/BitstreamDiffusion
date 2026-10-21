#scripts/owt/train_owt_gpt2id_bpe16.py
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from tqdm import tqdm


def _safe_name(s: str) -> str:
    return str(s).replace("/", "_").replace("-", "_").replace(":", "_")


def _load_flat_memmap(cache_path: Path, meta_path: Path) -> tuple[np.memmap, dict]:
    if not cache_path.exists() or not meta_path.exists():
        raise RuntimeError(f"Missing cache files: {cache_path} / {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    n = int(meta["n_tokens"])
    mm = np.memmap(cache_path, dtype=np.uint16, mode="r", shape=(n,))
    return mm, meta


def _load_gpt2_assets(tokenizer_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    eos_id = tok.eos_token_id
    if eos_id is None:
        raise RuntimeError(f"Tokenizer {tokenizer_name!r} has no eos_token_id")
    bos_id = tok.bos_token_id
    if bos_id is None:
        bos_id = eos_id

    return tok, int(bos_id), int(eos_id)


def gpt2_ids_to_pua_string(
    token_ids: Iterable[int],
    *,
    base_char_offset: int,
) -> str:
    return "".join(chr(base_char_offset + int(t)) for t in token_ids)


def pua_string_to_gpt2_ids(
    s: str,
    *,
    base_char_offset: int,
    vocab_size_gpt2: int,
) -> List[int]:
    out: List[int] = []
    for ch in s:
        tid = ord(ch) - int(base_char_offset)
        if 0 <= tid < int(vocab_size_gpt2):
            out.append(int(tid))
        else:
            raise RuntimeError(
                f"Decoded character outside GPT-2 id range: ord={ord(ch)} "
                f"offset={base_char_offset} -> tid={tid}"
            )
    return out


def iter_gpt2_sequences_from_flat_cache(
    mm: np.memmap,
    *,
    base_seq_len_tokens: int,
    bos_id: int,
    eos_id: int,
):
    """
    Reconstruct the exact final GPT-2 sequences used by the OWT dataset:
        [BOS] + content_tokens + [EOS]
    where content_tokens are consecutive chunks from the flat cache.
    """
    if int(base_seq_len_tokens) < 3:
        raise ValueError("base_seq_len_tokens must be >= 3")

    content_len = int(base_seq_len_tokens) - 2
    n_tokens = int(mm.shape[0])
    n_seq = n_tokens // content_len

    for i in range(n_seq):
        s = i * content_len
        e = s + content_len
        content = np.array(mm[s:e], dtype=np.uint16, copy=False)
        seq = np.empty((base_seq_len_tokens,), dtype=np.uint32)
        seq[0] = int(bos_id)
        seq[1:-1] = content.astype(np.uint32, copy=False)
        seq[-1] = int(eos_id)
        yield seq


def iter_gpt2_pua_lines_from_flat_cache(
    mm: np.memmap,
    *,
    base_seq_len_tokens: int,
    bos_id: int,
    eos_id: int,
    base_char_offset: int,
):
    """
    Yield one PUA string per final GPT-2 sequence:
        [BOS] + content_tokens + [EOS]

    This avoids constructing an intermediate integer sequence array and avoids
    seq.tolist(), which significantly reduces Python overhead in Step 1.
    """
    if int(base_seq_len_tokens) < 3:
        raise ValueError("base_seq_len_tokens must be >= 3")

    content_len = int(base_seq_len_tokens) - 2
    n_tokens = int(mm.shape[0])
    n_seq = n_tokens // content_len

    bos_ch = chr(int(base_char_offset) + int(bos_id))
    eos_ch = chr(int(base_char_offset) + int(eos_id))
    offset = int(base_char_offset)

    for i in range(n_seq):
        s = i * content_len
        e = s + content_len
        content = mm[s:e]  # memmap view; no copy

        # Convert the content tokens directly to their PUA characters.
        # astype(copy=False) is cheap here and avoids accidental uint16 overflow
        # in the offset addition on some platforms.
        body = "".join(map(chr, content.astype(np.uint32, copy=False) + offset))
        yield bos_ch + body + eos_ch


def train_tokenizer_from_sequences(
    train_strings_path: Path,
    *,
    vocab_size: int,
    min_frequency: int,
    base_char_offset: int,
    vocab_size_gpt2: int,
    out_tokenizer_path: Path,
) -> Tuple[int, List[str]]:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer

    out_tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    special_tokens = ["[PAD]", "[UNK]", "[EOSEQ]"]

    # Force every GPT-2 base symbol into the tokenizer alphabet.
    initial_alphabet = [chr(base_char_offset + i) for i in range(int(vocab_size_gpt2))]

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    trainer = BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        show_progress=True,
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet,
    )

    tokenizer.train([str(train_strings_path)], trainer)
    tokenizer.save(str(out_tokenizer_path))

    vocab = tokenizer.get_vocab()
    actual_vocab_size = int(len(vocab))

    return actual_vocab_size, special_tokens


def encode_length_stats(
    tokenizer_json_path: Path,
    strings_path: Path,
    *,
    limit_lines: int | None = None,
    batch_size: int = 2048,
) -> dict:
    """
    Compute encoded-length stats using batched tokenization to reduce Python
    overhead relative to line-by-line tok.encode(...).
    """
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_json_path))
    lengths: List[int] = []

    def _flush(batch: List[str]) -> None:
        if not batch:
            return
        encs = tok.encode_batch(batch)
        lengths.extend(len(enc.ids) for enc in encs)

    batch: List[str] = []
    seen = 0

    with open(strings_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Encoding length scan", unit="lines"):
            if limit_lines is not None and seen >= int(limit_lines):
                break
            batch.append(line.rstrip("\n"))
            seen += 1
            if len(batch) >= int(batch_size):
                _flush(batch)
                batch = []

    _flush(batch)

    if not lengths:
        raise RuntimeError("No sequences found while computing encoding length stats")

    arr = np.asarray(lengths, dtype=np.int64)
    return {
        "num_sequences_scanned": int(arr.size),
        "encoded_len_min": int(arr.min()),
        "encoded_len_mean": float(arr.mean()),
        "encoded_len_max": int(arr.max()),
        "encoded_len_p50": int(np.percentile(arr, 50)),
        "encoded_len_p90": int(np.percentile(arr, 90)),
        "encoded_len_p95": int(np.percentile(arr, 95)),
        "encoded_len_p99": int(np.percentile(arr, 99)),
        "encoded_len_p999": int(np.percentile(arr, 99.9)),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, default="datasets/openwebtext_gpt2_trainm100k")
    ap.add_argument("--tokenizer_name", type=str, default="gpt2")

    # This is the *base* GPT-2 sequence length whose distribution we preserve.
    ap.add_argument("--base_seq_len_tokens", type=int, default=1024)

    # Target second-stage code vocabulary.
    ap.add_argument("--code_vocab_size", type=int, default=65536)
    ap.add_argument("--min_frequency", type=int, default=2)

    # Unicode Private Use Area offset.
    ap.add_argument("--base_char_offset", type=lambda x: int(x, 0), default=0xF0000)

    # Which flat cache to use as training source.
    ap.add_argument("--split_name", type=str, default="train", choices=["train", "val"])
    ap.add_argument("--insert_eos", type=int, default=1)

    # Optional limits for fast experiments.
    ap.add_argument("--limit_sequences", type=int, default=None)
    ap.add_argument("--length_stats_limit", type=int, default=None)
    ap.add_argument("--length_stats_batch_size", type=int, default=2048)

    ap.add_argument(
        "--out_tokenizer",
        type=str,
        default=None,
        help="If omitted, stored under root as tokenizer_gpt2id_bpe16_<vocab>.json",
    )
    ap.add_argument(
        "--out_meta",
        type=str,
        default=None,
        help="If omitted, stored next to the tokenizer JSON",
    )
    ap.add_argument(
        "--keep_train_strings",
        type=int,
        default=0,
        help="If 1, keep the intermediate .train_strings.txt file. Default: delete it.",
    )

    args = ap.parse_args()

    root = Path(args.root)
    safe_tok = _safe_name(args.tokenizer_name)

    cache_path = root / f"cache_{args.split_name}_{safe_tok}_flat_eos{int(args.insert_eos)}.uint16"
    meta_path = root / f"cache_{args.split_name}_{safe_tok}_flat_eos{int(args.insert_eos)}.meta.json"

    mm, flat_meta = _load_flat_memmap(cache_path, meta_path)
    gpt2_tok, bos_id, eos_id = _load_gpt2_assets(args.tokenizer_name)
    vocab_size_gpt2 = int(gpt2_tok.vocab_size)

    if vocab_size_gpt2 + int(args.base_char_offset) > 0x10FFFF:
        raise RuntimeError(
            f"base_char_offset={hex(args.base_char_offset)} too large for vocab_size_gpt2={vocab_size_gpt2}"
        )

    if args.out_tokenizer is None:
        out_tokenizer_path = (
            root / f"tokenizer_gpt2id_bpe16_{int(args.code_vocab_size)}_base{int(args.base_seq_len_tokens)}.json"
        )
    else:
        out_tokenizer_path = Path(args.out_tokenizer)

    if args.out_meta is None:
        out_meta_path = out_tokenizer_path.with_suffix(".meta.json")
    else:
        out_meta_path = Path(args.out_meta)

    out_tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: write training strings, one line per fixed GPT-2 sequence
    # ------------------------------------------------------------------
    train_strings_path = out_tokenizer_path.with_suffix(".train_strings.txt")

    n_written = 0
    print(f"\n[gpt2id_bpe16] Starting Step 1: Writing sequences to {train_strings_path}")

    content_len = int(args.base_seq_len_tokens) - 2
    if content_len <= 0:
        raise RuntimeError("base_seq_len_tokens must be >= 3")

    total_sequences = int(mm.shape[0]) // content_len
    if args.limit_sequences is not None:
        total_sequences = min(total_sequences, int(args.limit_sequences))

    with open(train_strings_path, "w", encoding="utf-8", buffering=16 * 1024 * 1024) as fout:
        it = iter_gpt2_pua_lines_from_flat_cache(
            mm,
            base_seq_len_tokens=int(args.base_seq_len_tokens),
            bos_id=int(bos_id),
            eos_id=int(eos_id),
            base_char_offset=int(args.base_char_offset),
        )

        for i, line in enumerate(
            tqdm(
                it,
                total=total_sequences,
                desc="Writing Sequences",
                unit="seq",
            )
        ):
            if args.limit_sequences is not None and i >= int(args.limit_sequences):
                break
            fout.write(line)
            fout.write("\n")
            n_written += 1

    if n_written <= 0:
        raise RuntimeError("No GPT-2 sequences were written for tokenizer training")

    print(f"\n[gpt2id_bpe16] wrote {n_written:,} training sequences to {train_strings_path}")

    # ------------------------------------------------------------------
    # Step 2: train second-stage tokenizer
    # ------------------------------------------------------------------
    print("\n[gpt2id_bpe16] Starting Step 2: Training BPE tokenizer...")
    actual_vocab_size, special_tokens = train_tokenizer_from_sequences(
        train_strings_path,
        vocab_size=int(args.code_vocab_size),
        min_frequency=int(args.min_frequency),
        base_char_offset=int(args.base_char_offset),
        vocab_size_gpt2=int(vocab_size_gpt2),
        out_tokenizer_path=out_tokenizer_path,
    )

    print(
        f"\n[gpt2id_bpe16] trained tokenizer saved to {out_tokenizer_path} "
        f"(actual_vocab_size={actual_vocab_size})"
    )

    # ------------------------------------------------------------------
    # Step 3: load tokenizer to resolve special ids and scan encoding lengths
    # ------------------------------------------------------------------
    print("\n[gpt2id_bpe16] Starting Step 3: Scanning encoding lengths...")
    from tokenizers import Tokenizer

    code_tok = Tokenizer.from_file(str(out_tokenizer_path))
    vocab = code_tok.get_vocab()

    pad_id = vocab.get("[PAD]", None)
    unk_id = vocab.get("[UNK]", None)
    eoseq_id = vocab.get("[EOSEQ]", None)

    if pad_id is None or unk_id is None or eoseq_id is None:
        raise RuntimeError(
            f"Missing special ids in trained tokenizer. "
            f"pad_id={pad_id}, unk_id={unk_id}, eoseq_id={eoseq_id}"
        )

    length_stats = encode_length_stats(
        out_tokenizer_path,
        train_strings_path,
        limit_lines=args.length_stats_limit,
        batch_size=int(args.length_stats_batch_size),
    )

    # Recommended fixed model length in code-token space:
    # we need one extra slot for explicit EOSEQ that we append after encoding.
    recommended_code_seq_len_tokens = int(length_stats["encoded_len_max"] + 1)

    meta = {
        "format": "gpt2id_bpe16",
        "root": str(root),
        "source_flat_cache_path": str(cache_path),
        "source_flat_meta_path": str(meta_path),
        "source_flat_hf_split": flat_meta.get("hf_split"),
        "tokenizer_name_gpt2": args.tokenizer_name,
        "vocab_size_gpt2": int(vocab_size_gpt2),
        "bos_token_id_gpt2": int(bos_id),
        "eos_token_id_gpt2": int(eos_id),
        "base_seq_len_tokens": int(args.base_seq_len_tokens),
        "content_len_tokens": int(args.base_seq_len_tokens - 2),
        "base_char_offset": int(args.base_char_offset),
        "target_code_vocab_size": int(args.code_vocab_size),
        "actual_code_vocab_size": int(actual_vocab_size),
        "special_tokens": special_tokens,
        "pad_id": int(pad_id),
        "unk_id": int(unk_id),
        "eoseq_id": int(eoseq_id),
        "train_strings_path": str(train_strings_path) if int(args.keep_train_strings) else None,
        "num_training_sequences": int(n_written),
        "encoding_length_stats": length_stats,
        "recommended_code_seq_len_tokens": int(recommended_code_seq_len_tokens),
        "bits_per_code_token": int(math.ceil(math.log2(max(2, int(actual_vocab_size))))),
    }

    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[gpt2id_bpe16] meta saved to {out_meta_path}")
    print(
        f"[gpt2id_bpe16] recommended_code_seq_len_tokens={recommended_code_seq_len_tokens} "
        f"(this already includes +1 for explicit EOSEQ)"
    )

    if not int(args.keep_train_strings):
        try:
            train_strings_path.unlink()
            print(f"[gpt2id_bpe16] deleted intermediate file {train_strings_path}")
        except FileNotFoundError:
            pass

    print()


if __name__ == "__main__":
    main()