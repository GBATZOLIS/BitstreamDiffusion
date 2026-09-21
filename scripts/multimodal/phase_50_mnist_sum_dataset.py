"""
scripts/multimodal/phase_50_mnist_sum_dataset.py

Build the MNIST-Sum image+text corpus. See docs/MNIST_SUM_EXPERIMENT.md.

An image is four binarised MNIST digits in a 2x2 grid (56x56, pristine 28x28
digits, no resizing). The text is the raster-order equation and its sum:

    image: [7][6]     text: "seven + six + three + two = eighteen"
           [3][2]

Every direction has a unique correct answer, so image->text, text->image and
joint co-generation are all scored exactly rather than by proxy.

Sampling is NATURAL (uniform digits) by default. `--balance_sums` exists but is
off deliberately: rejection-sampling for a uniform sum distorts the digit
marginals badly (sum=0 forces 0,0,0,0), which corrupts the perception task. The
"always answer the modal sum" shortcut it would guard against is already handled
by scoring per-digit perception separately -- that cannot be gamed by guessing a
sum -- and the builder records the baseline in meta.json regardless.

Usage:
  python scripts/multimodal/phase_50_mnist_sum_dataset.py \
      --out datasets/mnist_sum_p28 --patch_size 28 \
      --n_train 1000000 --n_val 10000 --holdout_combos 0.1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from data import mnist_sum_codec as C   # noqa: E402


def load_mnist(raw_dir: Path, split: str):
    """Read raw IDX directly -- no torchvision, no network (compute nodes are offline)."""
    img_f = "train-images-idx3-ubyte" if split == "train" else "t10k-images-idx3-ubyte"
    lbl_f = "train-labels-idx1-ubyte" if split == "train" else "t10k-labels-idx1-ubyte"
    with open(raw_dir / img_f, "rb") as f:
        assert int.from_bytes(f.read(4), "big") == 2051
        n = int.from_bytes(f.read(4), "big")
        h = int.from_bytes(f.read(4), "big")
        w = int.from_bytes(f.read(4), "big")
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, h, w).copy()
    with open(raw_dir / lbl_f, "rb") as f:
        assert int.from_bytes(f.read(4), "big") == 2049
        int.from_bytes(f.read(4), "big")
        lbls = np.frombuffer(f.read(), dtype=np.uint8).copy()
    return imgs, lbls


def sample_tuples(rng, n, *, balance, allowed):
    if not balance:
        out = rng.integers(0, 10, size=(n, C.N_DIGITS), dtype=np.int64)
        if allowed is None:
            return out
        bad = np.array([tuple(r) not in allowed for r in out])
        while bad.any():
            idx = np.flatnonzero(bad)
            out[idx] = rng.integers(0, 10, size=(idx.size, C.N_DIGITS), dtype=np.int64)
            bad = np.array([tuple(r) not in allowed for r in out])
        return out
    out = np.empty((n, C.N_DIGITS), dtype=np.int64)
    i = 0
    while i < n:
        s = int(rng.integers(0, C.MAX_SUM + 1))
        for _ in range(400):
            t = rng.integers(0, 10, size=C.N_DIGITS)
            if int(t.sum()) != s:
                continue
            if allowed is not None and tuple(t) not in allowed:
                continue
            out[i] = t; i += 1
            break
        else:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mnist_raw",
                    default="/rds/project/rds-LlrDsbHU5UM/gb511/datasets/MNISTDataset/raw")
    ap.add_argument("--patch_size", type=int, default=28,
                    help="P: bits per trunk position (must divide 784 and be a multiple of 7)")
    ap.add_argument("--caption_len", type=int, default=12, help="text TOKENS")
    ap.add_argument("--n_train", type=int, default=1_000_000)
    ap.add_argument("--n_val", type=int, default=10_000)
    ap.add_argument("--threshold", type=int, default=127)
    ap.add_argument("--balance_sums", action="store_true")
    ap.add_argument("--holdout_combos", type=float, default=0.1,
                    help="fraction of the 10^4 digit tuples reserved for a held-out val split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=50_000)
    args = ap.parse_args()

    layout = C.build_layout(args.patch_size, args.caption_len)
    out = Path(args.out)
    rng = np.random.default_rng(args.seed)
    print(f"[mnist-sum] P={args.patch_size} Lt={args.caption_len} tokens "
          f"({C.tokens_per_text_position(args.patch_size)}/position) -> "
          f"text_pos={layout.num_text_positions} image_pos={layout.num_image_tokens} "
          f"total={layout.num_positions} bits={layout.total_bits}")

    train_allowed = val_allowed = None
    if args.holdout_combos > 0:
        all_t = [(a, b, c, d) for a in range(10) for b in range(10)
                 for c in range(10) for d in range(10)]
        rng.shuffle(all_t)
        k = int(len(all_t) * args.holdout_combos)
        val_allowed, train_allowed = set(all_t[:k]), set(all_t[k:])
        print(f"[mnist-sum] tuple holdout: {len(train_allowed)} train / "
              f"{len(val_allowed)} held-out")

    # Three splits: train, val (in-distribution), val_holdout (unseen tuples).
    splits = [("train", args.n_train, train_allowed, "train"),
              ("val", args.n_val, train_allowed, "test")]
    if val_allowed:
        splits.append(("val_holdout", args.n_val, val_allowed, "test"))

    meta_splits = {}
    for split, n_ex, allowed, mnist_split in splits:
        (out / split).mkdir(parents=True, exist_ok=True)
        imgs, lbls = load_mnist(Path(args.mnist_raw), mnist_split)
        pool = {d: np.flatnonzero(lbls == d) for d in range(10)}

        tuples = sample_tuples(rng, n_ex, balance=args.balance_sums, allowed=allowed)
        sums = tuples.sum(axis=1)

        shard = out / split / "shard_00000.uint32.mmap"
        mm = np.memmap(shard, dtype=np.uint32, mode="w+",
                       shape=(n_ex, layout.num_positions))
        eq_f = open(out / split / "equations.txt", "w")
        for s0 in range(0, n_ex, args.chunk):
            s1 = min(s0 + args.chunk, n_ex)
            buf = np.zeros((s1 - s0, layout.num_positions), dtype=np.int64)
            for i in range(s0, s1):
                quad = np.zeros((C.N_DIGITS, C.DIGIT_H, C.DIGIT_W), dtype=np.uint8)
                for q, d in enumerate(tuples[i]):
                    p = pool[int(d)]
                    quad[q] = (imgs[p[rng.integers(0, p.size)]] > args.threshold)
                codes = C.equation_to_codes(tuples[i])
                if len(codes) > args.caption_len:
                    raise SystemExit(
                        f"ERROR: equation needs {len(codes)} tokens > caption_len="
                        f"{args.caption_len}; the answer would be truncated.")
                buf[i - s0] = C.assemble_code_row(layout, quad, codes)
                eq_f.write(" ".join(C.equation_words(tuples[i])) + "\n")
            mm[s0:s1] = buf.astype(np.uint32)
            if s0 % (args.chunk * 10) == 0:
                print(f"    {split}: {s1:,}/{n_ex:,}", flush=True)
        mm.flush(); del mm
        eq_f.close()

        json.dump({"0": {"shard_idx": 0, "file": shard.name, "start": 0,
                         "count": int(n_ex), "status": "done"}},
                  open(out / split / "shard_idx.json", "w"), indent=1)
        np.savez_compressed(out / split / "labels.npz",
                            digits=tuples.astype(np.uint8), sums=sums.astype(np.uint8))

        cnt = Counter(int(s) for s in sums)
        modal, modal_n = cnt.most_common(1)[0]
        meta_splits[split] = {
            "count": int(n_ex),
            "modal_sum": int(modal),
            "modal_sum_baseline": round(modal_n / n_ex, 5),
            "tuples": "held_out" if split == "val_holdout" else "in_distribution",
        }
        print(f"[mnist-sum/{split}] n={n_ex:,} | always-'{C.num2words(modal)}' "
              f"baseline = {modal_n / n_ex:.3%}")

    meta = {
        "dataset": "MNISTSumBits",
        "task": "four binarised MNIST digits in a 2x2 grid; text is the raster-order "
                "addition (TL + TR + BL + BR) and its sum, in words",
        "image_tokenizer": None,
        "image_tokenizer_note": "none -- binary pixels are carried as raw bits, "
                                f"{args.patch_size} pixels per position. No reconstruction "
                                "floor: generated and real images are exactly comparable.",
        "text_tokenizer": "mnist_sum_codec/7bit",
        "text_bits": C.TEXT_BITS,
        "text_vocab_used": C.N_TEXT_CODES,
        "tokens_per_text_position": C.tokens_per_text_position(args.patch_size),
        "bits_per_token": args.patch_size,
        "caption_len_tokens": args.caption_len,
        "num_text_positions": layout.num_text_positions,
        "num_image_tokens": layout.num_image_tokens,
        "num_positions": layout.num_positions,
        "sequence_len_bits": layout.total_bits,
        "canvas": [2 * C.DIGIT_H, 2 * C.DIGIT_W],
        "digit_shape": [C.DIGIT_H, C.DIGIT_W],
        "digit_order": list(C.DIGIT_SLOT_NAMES),
        "flatten_order": "digit_major_row_major",
        "bit_layout": "raw_binary_pixels_lsb_first",
        "binarize_threshold": args.threshold,
        "balance_sums": bool(args.balance_sums),
        "holdout_combos": args.holdout_combos,
        "seed": args.seed,
        "splits": meta_splits,
        "marker_codes": C.marker_codes(args.patch_size),
    }
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print(f"[mnist-sum] wrote {out}/meta.json")


if __name__ == "__main__":
    main()
