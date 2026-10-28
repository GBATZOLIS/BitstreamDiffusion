from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Optional

from tqdm import tqdm


# -----------------------------------------------------------------------------
# Detokenizer used by SEDD / MDLM-style LM1B preprocessing
# Source convention: Lou et al. preprocessing family
# -----------------------------------------------------------------------------
import re


def lm1b_detokenizer(x: str) -> str:
    x = (x or "").strip()
    if not x:
        return ""

    x = x.replace("http : / / ", "http://")
    x = x.replace("https : / / ", "https://")
    x = re.sub(r" \'(\w+)", r"'\1", x)
    x = re.sub(r" (\w+) \. ", r" \1. ", x)
    x = re.sub(r" (\w+) \.$", r" \1.", x)
    x = x.replace(" ? ", "? ")
    x = re.sub(r" \?$", "?", x)
    x = x.replace(" ! ", "! ")
    x = re.sub(r" \!$", "!", x)
    x = x.replace(" , ", ", ")
    x = x.replace(" : ", ": ")
    x = x.replace(" ; ", "; ")
    x = x.replace(" / ", "/")
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r"\' ([^\']+) \'", r"'\1'", x)
    x = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", x)
    x = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", x)
    x = x.replace("$ ", "$")
    x = x.replace("£ ", "£")
    return x.strip()


# -----------------------------------------------------------------------------
# Dataset backends
# -----------------------------------------------------------------------------

def _iter_tfds_split(split: str, *, data_dir: Optional[str]) -> Iterator[str]:
    try:
        import tensorflow_datasets as tfds
    except ImportError as e:
        raise ImportError(
            "Missing tensorflow-datasets. Install with: pip install tensorflow-datasets"
        ) from e

    # Avoid TensorFlow reserving GPU memory on multi-GPU training machines.
    try:
        import tensorflow as tf

        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
    except Exception:
        tf = None  # noqa: F841

    ds = tfds.load("lm1b", split=split, data_dir=data_dir, shuffle_files=False)
    for ex in tfds.as_numpy(ds):
        txt = ex.get("text", b"")
        if isinstance(txt, bytes):
            txt = txt.decode("utf-8", errors="ignore")
        if txt:
            yield txt



def _iter_hf_split(split: str, *, cache_dir: Optional[str]) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Missing datasets. Install with: pip install datasets") from e

    # The public HF mirror has changed names over time. Try a small list.
    candidates = [
        ("lm1b", None),
        ("billion-word-benchmark/lm1b", None),
        ("jdeschena/lm1b", None),
        ("dvruette/lm1b", None),
    ]

    last_err: Optional[Exception] = None
    for repo_id, subset in candidates:
        try:
            ds = load_dataset(repo_id, subset, split=split, cache_dir=cache_dir)
            for ex in ds:
                txt = ex.get("text", "")
                if txt:
                    yield txt
            return
        except Exception as e:  # pragma: no cover - backend fallback path
            last_err = e
            continue

    raise RuntimeError(
        "Could not load LM1B from Hugging Face. Last error:\n"
        f"{type(last_err).__name__}: {last_err}"
    )



def iter_lm1b_split(
    split: str,
    *,
    backend: str,
    tfds_data_dir: Optional[str],
    hf_cache_dir: Optional[str],
) -> Iterator[str]:
    backend = backend.lower().strip()

    if backend in {"tfds", "tensorflow_datasets"}:
        yield from _iter_tfds_split(split, data_dir=tfds_data_dir)
        return

    if backend in {"hf", "huggingface"}:
        yield from _iter_hf_split(split, cache_dir=hf_cache_dir)
        return

    if backend == "auto":
        try:
            yield from _iter_tfds_split(split, data_dir=tfds_data_dir)
            return
        except Exception:
            yield from _iter_hf_split(split, cache_dir=hf_cache_dir)
            return

    raise ValueError(f"Unknown backend={backend!r}")


# -----------------------------------------------------------------------------
# Materialization
# -----------------------------------------------------------------------------

def write_text_file(
    texts: Iterable[str],
    out_path: Path,
    *,
    max_examples: Optional[int],
    progress_desc: str,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for txt in tqdm(texts, desc=progress_desc):
            txt = lm1b_detokenizer(txt)
            if not txt:
                continue
            fout.write(txt + "\n")
            n += 1
            if max_examples is not None and n >= int(max_examples):
                break
    return n


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", type=str, default="auto", choices=["auto", "tfds", "hf"])
    ap.add_argument("--out_dir", type=str, default="datasets/lm1b")
    ap.add_argument("--train_out", type=str, default=None)
    ap.add_argument("--test_out", type=str, default=None)
    ap.add_argument("--tfds_data_dir", type=str, default=None)
    ap.add_argument("--hf_cache_dir", type=str, default=None)
    ap.add_argument("--max_train_examples", type=int, default=None)
    ap.add_argument("--max_test_examples", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out = Path(args.train_out) if args.train_out else (out_dir / "train.txt")
    test_out = Path(args.test_out) if args.test_out else (out_dir / "test.txt")
    meta_out = out_dir / "materialization_meta.json"

    if (train_out.exists() or test_out.exists()) and not args.force:
        raise SystemExit(
            "Refusing to overwrite existing files. Pass --force to regenerate.\n"
            f"Existing: {train_out if train_out.exists() else ''} {test_out if test_out.exists() else ''}"
        )

    print(f"[lm1b] backend={args.backend}")
    print(f"[lm1b] train_out={train_out}")
    print(f"[lm1b] test_out={test_out}")

    n_train = write_text_file(
        iter_lm1b_split(
            "train",
            backend=args.backend,
            tfds_data_dir=args.tfds_data_dir,
            hf_cache_dir=args.hf_cache_dir,
        ),
        train_out,
        max_examples=args.max_train_examples,
        progress_desc="LM1B train -> text",
    )

    # TFDS uses a standard test split for LM1B.
    n_test = write_text_file(
        iter_lm1b_split(
            "test",
            backend=args.backend,
            tfds_data_dir=args.tfds_data_dir,
            hf_cache_dir=args.hf_cache_dir,
        ),
        test_out,
        max_examples=args.max_test_examples,
        progress_desc="LM1B test -> text",
    )

    meta = {
        "backend": args.backend,
        "train_out": str(train_out),
        "test_out": str(test_out),
        "n_train_examples": int(n_train),
        "n_test_examples": int(n_test),
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[lm1b] wrote {n_train:,} train examples -> {train_out}")
    print(f"[lm1b] wrote {n_test:,} test examples  -> {test_out}")
    print(f"[lm1b] meta -> {meta_out}")


if __name__ == "__main__":
    main()