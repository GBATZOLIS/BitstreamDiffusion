from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

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


def _load_bpe_model_from_tokenizer_json(tokenizer_json_path: Path) -> dict:
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    model = obj.get("model", {})
    if model.get("type") != "BPE":
        raise RuntimeError(
            f"Expected BPE tokenizer JSON at {tokenizer_json_path}, "
            f"got model type {model.get('type')!r}."
        )

    return model


def _load_bpe_merges_from_tokenizer_json(tokenizer_json_path: Path) -> list[str]:
    model = _load_bpe_model_from_tokenizer_json(tokenizer_json_path)
    return list(model.get("merges", []))


def _load_bpe_vocab_from_tokenizer_json(tokenizer_json_path: Path) -> dict:
    model = _load_bpe_model_from_tokenizer_json(tokenizer_json_path)
    return dict(model.get("vocab", {}))


def _assert_warm_start_prefix(
    warm_start_tokenizer_json: Path,
    new_tokenizer_json: Path,
) -> None:
    old_merges = _load_bpe_merges_from_tokenizer_json(warm_start_tokenizer_json)
    new_merges = _load_bpe_merges_from_tokenizer_json(new_tokenizer_json)

    if len(new_merges) < len(old_merges):
        raise RuntimeError(
            "Warm-start prefix check failed: new tokenizer has fewer merges than old tokenizer. "
            f"old={len(old_merges):,}, new={len(new_merges):,}"
        )

    for i, old_merge in enumerate(old_merges):
        if new_merges[i] != old_merge:
            raise RuntimeError(
                "Warm-start prefix check failed: the old merges are not a prefix "
                "of the new tokenizer merges. This means the tokenizer was not "
                "continued inductively.\n"
                f"First mismatch at merge index {i:,}:\n"
                f"  old: {old_merges[i]!r}\n"
                f"  new: {new_merges[i]!r}"
            )

    old_vocab = _load_bpe_vocab_from_tokenizer_json(warm_start_tokenizer_json)
    new_vocab = _load_bpe_vocab_from_tokenizer_json(new_tokenizer_json)

    missing = [tok for tok in old_vocab.keys() if tok not in new_vocab]
    if missing:
        raise RuntimeError(
            "Warm-start vocab check failed: some old vocab tokens are missing from "
            "the new tokenizer.\n"
            f"Example missing token: {missing[0]!r}"
        )

    print(
        f"[large_vocab_sweep] Warm-start prefix check passed: "
        f"{len(old_merges):,} old merges are a prefix of "
        f"{len(new_merges):,} new merges.",
        flush=True,
    )


def iter_gpt2_pua_lines_from_flat_cache(
    mm: np.memmap,
    *,
    base_seq_len_tokens: int,
    bos_id: int,
    eos_id: int,
    base_char_offset: int,
    limit_sequences: int | None = None,
):
    """
    Yield one Unicode Private Use Area string per final GPT-2 sequence:

        [BOS] + content_tokens + [EOS]

    The resulting string has length base_seq_len_tokens.
    Each character represents exactly one GPT-2 token ID.
    """
    if int(base_seq_len_tokens) < 3:
        raise ValueError("base_seq_len_tokens must be >= 3")

    content_len = int(base_seq_len_tokens) - 2
    n_tokens = int(mm.shape[0])
    n_seq = n_tokens // content_len

    if limit_sequences is not None:
        n_seq = min(n_seq, int(limit_sequences))

    bos_ch = chr(int(base_char_offset) + int(bos_id))
    eos_ch = chr(int(base_char_offset) + int(eos_id))
    offset = int(base_char_offset)

    for i in range(n_seq):
        s = i * content_len
        e = s + content_len
        content = mm[s:e]

        body = "".join(map(chr, content.astype(np.uint32, copy=False) + offset))
        yield bos_ch + body + eos_ch


def write_pua_sequences_once(
    mm: np.memmap,
    out_path: Path,
    *,
    base_seq_len_tokens: int,
    bos_id: int,
    eos_id: int,
    base_char_offset: int,
    limit_sequences: int | None,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    content_len = int(base_seq_len_tokens) - 2
    total_sequences = int(mm.shape[0]) // content_len
    if limit_sequences is not None:
        total_sequences = min(total_sequences, int(limit_sequences))

    n_written = 0

    with open(out_path, "w", encoding="utf-8", buffering=16 * 1024 * 1024) as fout:
        it = iter_gpt2_pua_lines_from_flat_cache(
            mm,
            base_seq_len_tokens=base_seq_len_tokens,
            bos_id=bos_id,
            eos_id=eos_id,
            base_char_offset=base_char_offset,
            limit_sequences=limit_sequences,
        )

        for line in tqdm(
            it,
            total=total_sequences,
            desc="Writing GPT2-ID PUA sequences",
            unit="seq",
        ):
            fout.write(line)
            fout.write("\n")
            n_written += 1

    if n_written <= 0:
        raise RuntimeError("No PUA sequences were written.")

    return n_written


def train_bpe_tokenizer(
    train_strings_path: Path,
    *,
    vocab_size: int,
    min_frequency: int,
    base_char_offset: int,
    vocab_size_gpt2: int,
    out_tokenizer_path: Path,
    warm_start_tokenizer_json: Path | None = None,
    require_warm_start_prefix: bool = True,
) -> tuple[int, list[str]]:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer

    out_tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    special_tokens = ["[PAD]", "[UNK]", "[EOSEQ]"]

    initial_alphabet = [
        chr(int(base_char_offset) + i)
        for i in range(int(vocab_size_gpt2))
    ]

    if warm_start_tokenizer_json is not None:
        warm_start_tokenizer_json = Path(warm_start_tokenizer_json)

        if not warm_start_tokenizer_json.exists():
            raise RuntimeError(f"Warm-start tokenizer does not exist: {warm_start_tokenizer_json}")

        old_vocab = _load_bpe_vocab_from_tokenizer_json(warm_start_tokenizer_json)
        old_merges = _load_bpe_merges_from_tokenizer_json(warm_start_tokenizer_json)

        print(
            f"[large_vocab_sweep] Warm-starting BPE from existing tokenizer:\n"
            f"  {warm_start_tokenizer_json}\n"
            f"[large_vocab_sweep] Warm-start tokenizer stats:\n"
            f"  old vocab size: {len(old_vocab):,}\n"
            f"  old merges:     {len(old_merges):,}\n"
            f"  target vocab:   {int(vocab_size):,}",
            flush=True,
        )

        if int(vocab_size) <= len(old_vocab):
            raise RuntimeError(
                f"Target vocab_size={int(vocab_size):,} must be larger than "
                f"warm-start vocab size={len(old_vocab):,}."
            )

        tokenizer = Tokenizer.from_file(str(warm_start_tokenizer_json))

    else:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    trainer = BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        show_progress=True,
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet,
    )

    print(
        f"[large_vocab_sweep] Calling tokenizer.train(...) with target vocab_size={int(vocab_size):,}",
        flush=True,
    )

    tokenizer.train([str(train_strings_path)], trainer)
    tokenizer.save(str(out_tokenizer_path))

    actual_vocab_size = int(len(tokenizer.get_vocab()))

    if warm_start_tokenizer_json is not None and require_warm_start_prefix:
        _assert_warm_start_prefix(
            Path(warm_start_tokenizer_json),
            Path(out_tokenizer_path),
        )

    return actual_vocab_size, special_tokens


def encode_length_stats(
    tokenizer_json_path: Path,
    strings_path: Path,
    *,
    limit_lines: int | None,
    batch_size: int,
) -> dict:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_json_path))

    lengths: list[int] = []
    seen = 0
    batch: list[str] = []

    def flush(batch_: list[str]) -> None:
        if not batch_:
            return
        encs = tok.encode_batch(batch_)
        lengths.extend(len(enc.ids) for enc in encs)

    with open(strings_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Scanning lengths: {tokenizer_json_path.name}", unit="seq"):
            if limit_lines is not None and seen >= int(limit_lines):
                break

            batch.append(line.rstrip("\n"))
            seen += 1

            if len(batch) >= int(batch_size):
                flush(batch)
                batch = []

    flush(batch)

    if not lengths:
        raise RuntimeError("No sequences found while computing length stats.")

    arr = np.asarray(lengths, dtype=np.int64)

    stats = {
        "num_sequences_scanned": int(arr.size),
        "encoded_len_min": int(arr.min()),
        "encoded_len_max": int(arr.max()),
        "encoded_len_mean": float(arr.mean()),
        "encoded_len_std": float(arr.std(ddof=0)),
        "encoded_len_p50": float(np.percentile(arr, 50)),
        "encoded_len_p90": float(np.percentile(arr, 90)),
        "encoded_len_p95": float(np.percentile(arr, 95)),
        "encoded_len_p99": float(np.percentile(arr, 99)),
        "encoded_len_p999": float(np.percentile(arr, 99.9)),
    }

    stats["encoded_len_with_eoseq_min"] = int(stats["encoded_len_min"] + 1)
    stats["encoded_len_with_eoseq_max"] = int(stats["encoded_len_max"] + 1)
    stats["encoded_len_with_eoseq_mean"] = float(stats["encoded_len_mean"] + 1.0)
    stats["encoded_len_with_eoseq_p90"] = float(stats["encoded_len_p90"] + 1.0)
    stats["encoded_len_with_eoseq_p95"] = float(stats["encoded_len_p95"] + 1.0)
    stats["encoded_len_with_eoseq_p99"] = float(stats["encoded_len_p99"] + 1.0)
    stats["encoded_len_with_eoseq_p999"] = float(stats["encoded_len_p999"] + 1.0)

    return stats


def optionally_write_encoded_memmap(
    tokenizer_json_path: Path,
    strings_path: Path,
    out_memmap_path: Path,
    out_lengths_path: Path,
    *,
    eoseq_id: int,
    pad_id: int,
    fixed_len: int,
    limit_lines: int | None,
    batch_size: int,
) -> int:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_json_path))

    n = 0
    with open(strings_path, "r", encoding="utf-8") as f:
        for _ in f:
            if limit_lines is not None and n >= int(limit_lines):
                break
            n += 1

    if n <= 0:
        raise RuntimeError("No sequences available for memmap writing.")

    out_memmap_path.parent.mkdir(parents=True, exist_ok=True)
    out_lengths_path.parent.mkdir(parents=True, exist_ok=True)

    encoded_mm = np.memmap(
        out_memmap_path,
        dtype=np.uint32,
        mode="w+",
        shape=(n, int(fixed_len)),
    )
    encoded_mm[:] = int(pad_id)

    lengths = np.memmap(
        out_lengths_path,
        dtype=np.uint32,
        mode="w+",
        shape=(n,),
    )

    row = 0
    batch: list[str] = []

    def flush(batch_: list[str]) -> None:
        nonlocal row

        if not batch_:
            return

        encs = tok.encode_batch(batch_)

        for enc in encs:
            ids = list(map(int, enc.ids))
            ids.append(int(eoseq_id))

            if len(ids) > fixed_len:
                raise RuntimeError(
                    f"Encoded sequence length {len(ids)} exceeds fixed_len={fixed_len}."
                )

            encoded_mm[row, : len(ids)] = np.asarray(ids, dtype=np.uint32)
            lengths[row] = int(len(ids))
            row += 1

    with open(strings_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Writing encoded memmap: {tokenizer_json_path.name}", unit="seq"):
            if limit_lines is not None and row + len(batch) >= int(limit_lines):
                break

            batch.append(line.rstrip("\n"))

            if len(batch) >= int(batch_size):
                flush(batch)
                batch = []

    flush(batch)

    encoded_mm.flush()
    lengths.flush()

    return int(row)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        type=str,
        default="datasets/openwebtext_gpt2_trainm100k",
        help="Root containing the existing GPT-2 flat cache.",
    )
    ap.add_argument("--tokenizer_name", type=str, default="gpt2")

    ap.add_argument(
        "--base_seq_len_tokens",
        type=int,
        default=1024,
        help="Original GPT-2 sequence length: [BOS] + content + [EOS].",
    )

    ap.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[17, 18, 19, 20],
        help="Second-stage vocabulary bit sizes to train.",
    )

    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--base_char_offset", type=lambda x: int(x, 0), default=0xF0000)

    ap.add_argument("--split_name", type=str, default="train", choices=["train", "val"])
    ap.add_argument("--insert_eos", type=int, default=1)

    ap.add_argument(
        "--out_dir",
        type=str,
        default="runs/LargeVocabularies",
    )

    ap.add_argument(
        "--train_limit_sequences",
        type=int,
        default=None,
        help="Optional cap on number of 1024-token GPT-2 sequences used for BPE training.",
    )

    ap.add_argument(
        "--stats_limit_sequences",
        type=int,
        default=None,
        help="Optional cap on number of sequences used for length statistics.",
    )

    ap.add_argument("--batch_size", type=int, default=2048)

    ap.add_argument(
        "--target_max_len",
        type=int,
        default=512,
        help="Target maximum encoded length, including explicit EOSEQ.",
    )

    ap.add_argument(
        "--save_encoded_memmaps",
        type=int,
        default=0,
        help="If 1, save padded uint32 encoded memmaps for each codec.",
    )

    ap.add_argument(
        "--keep_pua_strings",
        type=int,
        default=1,
        help="If 1, keep the shared PUA strings file.",
    )

    ap.add_argument(
        "--pua_strings_path",
        type=str,
        default=None,
        help="Optional existing PUA strings file to reuse.",
    )

    ap.add_argument(
        "--warm_start_tokenizer_json",
        type=str,
        default=None,
        help="Optional existing BPE tokenizer JSON to warm-start from.",
    )

    ap.add_argument(
        "--require_warm_start_prefix",
        type=int,
        default=1,
        help="If 1, verify that the warm-start merges are a prefix of the new merges.",
    )

    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    tok_dir = out_dir / "tokenizers"
    stats_dir = out_dir / "stats"
    encoded_dir = out_dir / "encoded_memmaps"

    out_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    safe_tok = _safe_name(args.tokenizer_name)

    cache_path = root / f"cache_{args.split_name}_{safe_tok}_flat_eos{int(args.insert_eos)}.uint16"
    meta_path = root / f"cache_{args.split_name}_{safe_tok}_flat_eos{int(args.insert_eos)}.meta.json"

    print(f"\n[large_vocab_sweep] Loading flat cache:", flush=True)
    print(f"  cache: {cache_path}", flush=True)
    print(f"  meta:  {meta_path}", flush=True)

    mm, flat_meta = _load_flat_memmap(cache_path, meta_path)

    gpt2_tok, bos_id, eos_id = _load_gpt2_assets(args.tokenizer_name)
    vocab_size_gpt2 = int(gpt2_tok.vocab_size)

    if vocab_size_gpt2 + int(args.base_char_offset) > 0x10FFFF:
        raise RuntimeError(
            f"base_char_offset={hex(args.base_char_offset)} too large for "
            f"vocab_size_gpt2={vocab_size_gpt2}"
        )

    content_len = int(args.base_seq_len_tokens) - 2
    if content_len <= 0:
        raise RuntimeError("base_seq_len_tokens must be >= 3")

    total_available_sequences = int(mm.shape[0]) // content_len
    total_written_sequences = total_available_sequences
    if args.train_limit_sequences is not None:
        total_written_sequences = min(total_written_sequences, int(args.train_limit_sequences))

    print(f"\n[large_vocab_sweep] GPT-2 vocab size: {vocab_size_gpt2}", flush=True)
    print(f"[large_vocab_sweep] BOS id: {bos_id}", flush=True)
    print(f"[large_vocab_sweep] EOS id: {eos_id}", flush=True)
    print(f"[large_vocab_sweep] Base GPT-2 seq len: {args.base_seq_len_tokens}", flush=True)
    print(f"[large_vocab_sweep] Content len: {content_len}", flush=True)
    print(f"[large_vocab_sweep] Available sequences: {total_available_sequences:,}", flush=True)
    print(f"[large_vocab_sweep] Sequences used for training/eval strings: {total_written_sequences:,}", flush=True)

    # ---------------------------------------------------------------------
    # Step 1: write or reuse shared PUA strings
    # ---------------------------------------------------------------------
    if args.pua_strings_path is not None:
        pua_strings_path = Path(args.pua_strings_path)

        if not pua_strings_path.exists():
            raise RuntimeError(f"Requested PUA strings file does not exist: {pua_strings_path}")

        print(f"\n[large_vocab_sweep] Using user-provided PUA strings:", flush=True)
        print(f"  {pua_strings_path}", flush=True)

        n_written = total_written_sequences

    else:
        pua_strings_path = out_dir / (
            f"owt_{safe_tok}_gpt2id_pua_base{int(args.base_seq_len_tokens)}"
            f"_n{total_written_sequences}.txt"
        )

        if pua_strings_path.exists():
            print(f"\n[large_vocab_sweep] Reusing existing PUA strings:", flush=True)
            print(f"  {pua_strings_path}", flush=True)
            n_written = total_written_sequences
        else:
            print(f"\n[large_vocab_sweep] Writing shared PUA strings:", flush=True)
            print(f"  {pua_strings_path}", flush=True)

            n_written = write_pua_sequences_once(
                mm,
                pua_strings_path,
                base_seq_len_tokens=int(args.base_seq_len_tokens),
                bos_id=int(bos_id),
                eos_id=int(eos_id),
                base_char_offset=int(args.base_char_offset),
                limit_sequences=args.train_limit_sequences,
            )

    # ---------------------------------------------------------------------
    # Step 2: train codecs and scan lengths
    # ---------------------------------------------------------------------
    summary_rows: list[dict] = []

    for bits in args.bits:
        bits = int(bits)
        vocab_size = int(2 ** bits)

        print("\n" + "=" * 80, flush=True)
        print(f"[large_vocab_sweep] Training {bits}-bit codec: vocab_size={vocab_size:,}", flush=True)
        print("=" * 80, flush=True)

        tokenizer_path = tok_dir / (
            f"tokenizer_gpt2id_bpe{bits}_{vocab_size}_base{int(args.base_seq_len_tokens)}.json"
        )
        meta_out_path = tok_dir / (
            f"tokenizer_gpt2id_bpe{bits}_{vocab_size}_base{int(args.base_seq_len_tokens)}.meta.json"
        )
        stats_out_path = stats_dir / (
            f"length_stats_gpt2id_bpe{bits}_{vocab_size}_base{int(args.base_seq_len_tokens)}.json"
        )

        if tokenizer_path.exists():
            print(f"[large_vocab_sweep] Reusing existing tokenizer: {tokenizer_path}", flush=True)

            from tokenizers import Tokenizer

            tmp_tok = Tokenizer.from_file(str(tokenizer_path))
            actual_vocab_size = int(len(tmp_tok.get_vocab()))
            special_tokens = ["[PAD]", "[UNK]", "[EOSEQ]"]

        else:
            actual_vocab_size, special_tokens = train_bpe_tokenizer(
                pua_strings_path,
                vocab_size=vocab_size,
                min_frequency=int(args.min_frequency),
                base_char_offset=int(args.base_char_offset),
                vocab_size_gpt2=int(vocab_size_gpt2),
                out_tokenizer_path=tokenizer_path,
                warm_start_tokenizer_json=(
                    Path(args.warm_start_tokenizer_json)
                    if args.warm_start_tokenizer_json is not None
                    else None
                ),
                require_warm_start_prefix=bool(int(args.require_warm_start_prefix)),
            )

        print(
            f"[large_vocab_sweep] Tokenizer ready: {tokenizer_path} "
            f"(actual_vocab_size={actual_vocab_size:,})",
            flush=True,
        )

        from tokenizers import Tokenizer

        code_tok = Tokenizer.from_file(str(tokenizer_path))
        vocab = code_tok.get_vocab()

        pad_id = vocab.get("[PAD]")
        unk_id = vocab.get("[UNK]")
        eoseq_id = vocab.get("[EOSEQ]")

        if pad_id is None or unk_id is None or eoseq_id is None:
            raise RuntimeError(
                f"Missing special ids for {tokenizer_path}. "
                f"pad_id={pad_id}, unk_id={unk_id}, eoseq_id={eoseq_id}"
            )

        print(f"[large_vocab_sweep] Scanning encoded lengths for {bits}-bit codec...", flush=True)

        length_stats = encode_length_stats(
            tokenizer_path,
            pua_strings_path,
            limit_lines=args.stats_limit_sequences,
            batch_size=int(args.batch_size),
        )

        max_with_eoseq = int(length_stats["encoded_len_with_eoseq_max"])
        mean_with_eoseq = float(length_stats["encoded_len_with_eoseq_mean"])

        passes_target = bool(max_with_eoseq <= int(args.target_max_len))
        recommended_code_seq_len_tokens = max_with_eoseq

        meta = {
            "format": f"gpt2id_bpe{bits}",
            "root": str(root),
            "out_dir": str(out_dir),
            "source_flat_cache_path": str(cache_path),
            "source_flat_meta_path": str(meta_path),
            "source_flat_hf_split": flat_meta.get("hf_split"),
            "tokenizer_name_gpt2": args.tokenizer_name,
            "vocab_size_gpt2": int(vocab_size_gpt2),
            "bos_token_id_gpt2": int(bos_id),
            "eos_token_id_gpt2": int(eos_id),
            "base_seq_len_tokens": int(args.base_seq_len_tokens),
            "content_len_tokens": int(content_len),
            "base_char_offset": int(args.base_char_offset),
            "target_bits_per_code_token": int(bits),
            "target_code_vocab_size": int(vocab_size),
            "actual_code_vocab_size": int(actual_vocab_size),
            "actual_bits_per_code_token": int(math.ceil(math.log2(max(2, actual_vocab_size)))),
            "warm_start_tokenizer_json": (
                str(args.warm_start_tokenizer_json)
                if args.warm_start_tokenizer_json is not None
                else None
            ),
            "require_warm_start_prefix": bool(int(args.require_warm_start_prefix)),
            "special_tokens": special_tokens,
            "pad_id": int(pad_id),
            "unk_id": int(unk_id),
            "eoseq_id": int(eoseq_id),
            "pua_strings_path": str(pua_strings_path),
            "num_training_sequences": int(n_written),
            "stats_limit_sequences": args.stats_limit_sequences,
            "encoding_length_stats": length_stats,
            "recommended_code_seq_len_tokens": int(recommended_code_seq_len_tokens),
            "target_max_len_including_eoseq": int(args.target_max_len),
            "passes_target_max_len": passes_target,
        }

        with open(meta_out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        with open(stats_out_path, "w", encoding="utf-8") as f:
            json.dump(length_stats, f, indent=2)

        print(f"[large_vocab_sweep] Meta saved:  {meta_out_path}", flush=True)
        print(f"[large_vocab_sweep] Stats saved: {stats_out_path}", flush=True)

        encoded_memmap_path = None
        encoded_lengths_path = None

        if int(args.save_encoded_memmaps):
            fixed_len = int(recommended_code_seq_len_tokens)

            encoded_memmap_path = encoded_dir / (
                f"encoded_gpt2id_bpe{bits}_{vocab_size}_base{int(args.base_seq_len_tokens)}"
                f"_fixed{fixed_len}.uint32"
            )
            encoded_lengths_path = encoded_dir / (
                f"encoded_gpt2id_bpe{bits}_{vocab_size}_base{int(args.base_seq_len_tokens)}"
                f"_lengths.uint32"
            )

            n_encoded = optionally_write_encoded_memmap(
                tokenizer_path,
                pua_strings_path,
                encoded_memmap_path,
                encoded_lengths_path,
                eoseq_id=int(eoseq_id),
                pad_id=int(pad_id),
                fixed_len=fixed_len,
                limit_lines=args.stats_limit_sequences,
                batch_size=int(args.batch_size),
            )

            print(f"[large_vocab_sweep] Encoded memmap saved: {encoded_memmap_path}", flush=True)
            print(f"[large_vocab_sweep] Encoded lengths saved: {encoded_lengths_path}", flush=True)
            print(f"[large_vocab_sweep] Encoded sequences: {n_encoded:,}", flush=True)

        row = {
            "bits": int(bits),
            "target_vocab_size": int(vocab_size),
            "actual_vocab_size": int(actual_vocab_size),
            "num_sequences_scanned": int(length_stats["num_sequences_scanned"]),
            "encoded_len_min": int(length_stats["encoded_len_min"]),
            "encoded_len_mean": float(length_stats["encoded_len_mean"]),
            "encoded_len_std": float(length_stats["encoded_len_std"]),
            "encoded_len_max": int(length_stats["encoded_len_max"]),
            "encoded_len_p50": float(length_stats["encoded_len_p50"]),
            "encoded_len_p90": float(length_stats["encoded_len_p90"]),
            "encoded_len_p95": float(length_stats["encoded_len_p95"]),
            "encoded_len_p99": float(length_stats["encoded_len_p99"]),
            "encoded_len_p999": float(length_stats["encoded_len_p999"]),
            "encoded_len_with_eoseq_mean": mean_with_eoseq,
            "encoded_len_with_eoseq_max": max_with_eoseq,
            "encoded_len_with_eoseq_p90": float(length_stats["encoded_len_with_eoseq_p90"]),
            "encoded_len_with_eoseq_p95": float(length_stats["encoded_len_with_eoseq_p95"]),
            "encoded_len_with_eoseq_p99": float(length_stats["encoded_len_with_eoseq_p99"]),
            "encoded_len_with_eoseq_p999": float(length_stats["encoded_len_with_eoseq_p999"]),
            "compression_ratio_mean": float(args.base_seq_len_tokens) / float(length_stats["encoded_len_mean"]),
            "compression_ratio_max_safe": float(args.base_seq_len_tokens) / float(length_stats["encoded_len_max"]),
            "attention_reduction_mean_approx": (
                float(args.base_seq_len_tokens) / float(length_stats["encoded_len_mean"])
            ) ** 2,
            "recommended_code_seq_len_tokens": int(recommended_code_seq_len_tokens),
            "target_max_len_including_eoseq": int(args.target_max_len),
            "passes_target_max_len": int(passes_target),
            "tokenizer_path": str(tokenizer_path),
            "meta_path": str(meta_out_path),
            "stats_path": str(stats_out_path),
            "encoded_memmap_path": str(encoded_memmap_path) if encoded_memmap_path is not None else "",
            "encoded_lengths_path": str(encoded_lengths_path) if encoded_lengths_path is not None else "",
        }

        summary_rows.append(row)

        print("\n[large_vocab_sweep] Result:", flush=True)
        print(f"  bits:                         {bits}", flush=True)
        print(f"  actual vocab size:            {actual_vocab_size:,}", flush=True)
        print(f"  mean encoded len:             {length_stats['encoded_len_mean']:.2f}", flush=True)
        print(f"  std encoded len:              {length_stats['encoded_len_std']:.2f}", flush=True)
        print(f"  p90 encoded len:              {length_stats['encoded_len_p90']:.2f}", flush=True)
        print(f"  p95 encoded len:              {length_stats['encoded_len_p95']:.2f}", flush=True)
        print(f"  p99 encoded len:              {length_stats['encoded_len_p99']:.2f}", flush=True)
        print(f"  min encoded len:              {length_stats['encoded_len_min']}", flush=True)
        print(f"  max encoded len:              {length_stats['encoded_len_max']}", flush=True)
        print(f"  max with explicit EOSEQ:      {max_with_eoseq}", flush=True)
        print(f"  target max length:            {args.target_max_len}", flush=True)
        print(f"  passes target:                {passes_target}", flush=True)

    # ---------------------------------------------------------------------
    # Step 3: save combined summary
    # ---------------------------------------------------------------------
    summary_json_path = out_dir / "large_vocab_sweep_summary.json"
    summary_csv_path = out_dir / "large_vocab_sweep_summary.csv"

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print("\n" + "=" * 80, flush=True)
    print("[large_vocab_sweep] Finished sweep.", flush=True)
    print(f"[large_vocab_sweep] Summary JSON: {summary_json_path}", flush=True)
    print(f"[large_vocab_sweep] Summary CSV:  {summary_csv_path}", flush=True)
    print("=" * 80, flush=True)

    print("\nSummary:", flush=True)
    for row in summary_rows:
        print(
            f"  {row['bits']:>2}-bit | "
            f"vocab={row['actual_vocab_size']:,} | "
            f"mean={row['encoded_len_mean']:.2f} | "
            f"std={row['encoded_len_std']:.2f} | "
            f"p95+EOSEQ={row['encoded_len_with_eoseq_p95']:.0f} | "
            f"p99+EOSEQ={row['encoded_len_with_eoseq_p99']:.0f} | "
            f"max+EOSEQ={row['encoded_len_with_eoseq_max']} | "
            f"passes<{args.target_max_len + 1}: {bool(row['passes_target_max_len'])}",
            flush=True,
        )

    if not int(args.keep_pua_strings):
        try:
            pua_strings_path.unlink()
            print(f"\n[large_vocab_sweep] Deleted shared PUA strings: {pua_strings_path}", flush=True)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()