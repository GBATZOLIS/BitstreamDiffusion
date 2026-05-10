from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
from tqdm import tqdm


def _safe_name(s: str) -> str:
    return str(s).replace("/", "_").replace("-", "_").replace(":", "_")


# -----------------------------------------------------------------------------
# Optional safety clean
# -----------------------------------------------------------------------------

def clean_cache_if_exists(dataset_dir: Path) -> None:
    patterns = ["*.pt", "cached_*.pt"]
    found = False
    for pattern in patterns:
        for f in dataset_dir.glob(pattern):
            print(f"🧹 Safety Clean: Deleting old cache file {f.name}...")
            try:
                os.remove(f)
                found = True
            except OSError as e:
                print(f"   (warn) Could not delete {f}: {e}")
    if found:
        print("   (This helps ensure the new map stays in sync with the dataset.)")


# -----------------------------------------------------------------------------
# Iterator over HF openwebtext examples
# -----------------------------------------------------------------------------

class OpenWebTextTokenIterator:
    """
    Streams HF openwebtext examples and yields token-id sequences as list[str]
    for Word2Vec training.
    """

    def __init__(
        self,
        *,
        split: str,
        tokenizer_name: str,
        cache_dir: Path,
        sample_chars: int,
        max_examples: Optional[int],
        batch_texts: int,
        chunk_tokens: int,
        min_chunk_len: int,
    ):
        self.split = str(split)
        self.tokenizer_name = str(tokenizer_name)
        self.cache_dir = Path(cache_dir)
        self.sample_chars = int(sample_chars)
        self.max_examples = None if max_examples is None else int(max_examples)
        self.batch_texts = int(batch_texts)
        self.chunk_tokens = int(chunk_tokens)
        self.min_chunk_len = int(min_chunk_len)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)

        from datasets import load_dataset
        self.ds = load_dataset(
            "openwebtext",
            split=self.split,
            cache_dir=str(self.cache_dir),
            trust_remote_code=True,
        )

    def __iter__(self) -> Iterator[List[str]]:
        total_chars = 0
        seen_examples = 0
        batch: List[str] = []

        for ex in self.ds:
            if self.max_examples is not None and seen_examples >= self.max_examples:
                break

            txt = (ex.get("text") or "")
            txt = txt.replace("\r\n", "\n").replace("\r", "\n").strip()
            if not txt:
                continue

            batch.append(txt)
            total_chars += len(txt)
            seen_examples += 1

            if len(batch) >= self.batch_texts:
                yield from self._flush_batch(batch)
                batch = []

            if total_chars >= self.sample_chars:
                break

        if batch:
            yield from self._flush_batch(batch)

        print(
            f"[semantic_owt] Iterator done: read ~{total_chars} chars over {seen_examples} examples "
            f"from HF split {self.split}."
        )

    def _flush_batch(self, batch_texts: List[str]) -> Iterator[List[str]]:
        enc = self.tokenizer(
            batch_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        K = self.chunk_tokens

        for ids in enc["input_ids"]:
            if not ids:
                continue
            for i in range(0, len(ids), K):
                chunk = ids[i:i + K]
                if len(chunk) >= self.min_chunk_len:
                    yield [str(t) for t in chunk]


# -----------------------------------------------------------------------------
# Orderings
# -----------------------------------------------------------------------------

def pca_1d_order(vectors: np.ndarray) -> List[int]:
    X = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = Vt[0].astype(np.float32)
    scores = (X @ pc1).astype(np.float32)
    order = np.argsort(scores)
    return [int(i) for i in order.tolist()]


def greedy_nn_order(vectors: np.ndarray, start: int = 0) -> List[int]:
    V = vectors.shape[0]
    visited = np.zeros(V, dtype=bool)
    path: List[int] = []

    curr = int(start) % V
    visited[curr] = True
    path.append(curr)

    use_matrix = False
    sim_matrix = None
    try:
        sim_matrix = vectors @ vectors.T
        np.fill_diagonal(sim_matrix, -9999.0)
        use_matrix = True
        print("[semantic_owt] Using full similarity matrix.")
    except MemoryError:
        sim_matrix = None
        use_matrix = False
        print("[semantic_owt] Full similarity matrix too large; using on-the-fly dot products.")

    for _ in tqdm(range(V - 1), desc="Sorting (greedy NN)"):
        if use_matrix:
            sims = sim_matrix[curr]
            masked = sims.copy()
            masked[visited] = -9999.0
            nxt = int(np.argmax(masked))
        else:
            sims = vectors @ vectors[curr]
            masked = sims.copy()
            masked[visited] = -9999.0
            nxt = int(np.argmax(masked))

        visited[nxt] = True
        path.append(nxt)
        curr = nxt

    return path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, default="datasets/openwebtext")
    ap.add_argument("--tokenizer_name", type=str, default="gpt2")
    ap.add_argument("--split", type=str, default="train[:-100000]")
    ap.add_argument("--out", type=str, default=None)

    ap.add_argument("--method", type=str, default="greedy", choices=["greedy", "pca"])

    ap.add_argument("--sample_chars", type=int, default=80_000_000)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--batch_texts", type=int, default=256)
    ap.add_argument("--chunk_tokens", type=int, default=128)
    ap.add_argument("--min_chunk_len", type=int, default=8)

    ap.add_argument("--vec_dim", type=int, default=64)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--start_id", type=int, default=0)

    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dataset_dir", type=str, default="datasets/openwebtext")
    ap.add_argument("--no_safety_clean", action="store_true")

    args = ap.parse_args()

    try:
        from gensim.models import Word2Vec
    except ImportError as e:
        raise SystemExit("Missing `gensim`. Install: pip install gensim") from e

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    safe_tok = _safe_name(args.tokenizer_name)
    out_path = Path(args.out) if args.out is not None else (root / f"semantic_mapping_{safe_tok}.json")

    if out_path.exists() and not args.force:
        print(f"[semantic_owt] ✅ Map found at {out_path}")
        print("[semantic_owt] Skipping generation. Use --force to overwrite.")
        return

    dataset_dir = Path(args.dataset_dir)
    if not args.no_safety_clean and dataset_dir.exists():
        clean_cache_if_exists(dataset_dir)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    vocab_size = int(tokenizer.vocab_size)
    print(f"[semantic_owt] Loaded tokenizer {args.tokenizer_name}. vocab_size={vocab_size}")

    sentences = OpenWebTextTokenIterator(
        split=args.split,
        tokenizer_name=args.tokenizer_name,
        cache_dir=root / "hf_cache",
        sample_chars=int(args.sample_chars),
        max_examples=args.max_examples,
        batch_texts=int(args.batch_texts),
        chunk_tokens=int(args.chunk_tokens),
        min_chunk_len=int(args.min_chunk_len),
    )

    print(
        f"[semantic_owt] Training Word2Vec:\n"
        f"  tokenizer={args.tokenizer_name}, split={args.split}\n"
        f"  vec_dim={args.vec_dim}, window={args.window}, epochs={args.epochs}, workers={args.workers}\n"
        f"  sample_chars={args.sample_chars}"
    )

    model = Word2Vec(
        sentences=sentences,
        vector_size=int(args.vec_dim),
        window=int(args.window),
        min_count=1,
        sg=1,
        workers=int(args.workers),
        epochs=int(args.epochs),
        seed=int(args.seed),
        compute_loss=False,
    )

    rng = np.random.default_rng(int(args.seed))
    vectors = rng.standard_normal((vocab_size, int(args.vec_dim))).astype(np.float32)

    found = 0
    for tid in range(vocab_size):
        key = str(tid)
        if key in model.wv:
            vectors[tid] = model.wv[key]
            found += 1

    print(f"[semantic_owt] Vectors ready. Found {found}/{vocab_size} token vectors.")

    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

    if str(args.method).lower() == "greedy":
        print("[semantic_owt] Running GREEDY nearest-neighbor ordering...")
        path = greedy_nn_order(vectors, start=int(args.start_id))
    else:
        print("[semantic_owt] Running PCA-1D ordering...")
        path = pca_1d_order(vectors)

    old_to_new = {int(old_id): int(new_idx) for new_idx, old_id in enumerate(path)}
    new_to_old = {int(new_idx): int(old_id) for new_idx, old_id in enumerate(path)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"old_to_new": old_to_new, "new_to_old": new_to_old}, f)

    print(f"[semantic_owt] ✅ Saved semantic map ({args.method}) to: {out_path}")
    print("[semantic_owt] Gray coding is applied later in the dataset.")


if __name__ == "__main__":
    main()