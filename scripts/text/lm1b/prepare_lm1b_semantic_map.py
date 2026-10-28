from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np
from tqdm import tqdm


class TokenRowIterator:
    """
    Streams token-cache rows as lists[str] for Word2Vec.

    We intentionally remove PAD and other reserved ids before feeding Word2Vec,
    so the learned semantic geometry is not dominated by padding / control tokens.
    """

    def __init__(
        self,
        *,
        cache_path: Path,
        meta_path: Path,
        ignore_ids: Sequence[int],
        max_rows: Optional[int] = None,
    ):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.n_rows = int(meta["n_sequences"])
        self.seq_len = int(meta["seq_len_tokens"])
        self.arr = np.memmap(cache_path, dtype=np.uint16, mode="r", shape=(self.n_rows, self.seq_len))
        self.ignore_ids = set(int(x) for x in ignore_ids)
        self.max_rows = None if max_rows is None else int(max_rows)

    def __iter__(self) -> Iterator[List[str]]:
        n = self.n_rows if self.max_rows is None else min(self.n_rows, self.max_rows)
        for i in range(n):
            row = self.arr[i]
            toks = [str(int(t)) for t in row if int(t) not in self.ignore_ids]
            if toks:
                yield toks


def pca_1d_order(vectors: np.ndarray, ids: List[int]) -> List[int]:
    X = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = Vt[0].astype(np.float32)
    scores = (X @ pc1).astype(np.float32)
    order = np.argsort(scores)
    return [int(ids[i]) for i in order.tolist()]


def _token_str(tokenizer, tid: int) -> str:
    try:
        tok = tokenizer.convert_ids_to_tokens(int(tid))
        return tok if tok is not None else str(int(tid))
    except Exception:
        return str(int(tid))


def _mean_adjacent_cosine(vectors: np.ndarray, ordered_ids: Sequence[int]) -> float:
    if len(ordered_ids) < 2:
        return float("nan")
    idx = np.asarray(list(ordered_ids), dtype=np.int64)
    a = vectors[idx[:-1]]
    b = vectors[idx[1:]]
    sims = (a * b).sum(axis=1)
    return float(np.mean(sims))


def _mean_adjacent_cosine_positions(vectors: np.ndarray, ordered_pos: Sequence[int]) -> float:
    if len(ordered_pos) < 2:
        return float("nan")
    idx = np.asarray(list(ordered_pos), dtype=np.int64)
    a = vectors[idx[:-1]]
    b = vectors[idx[1:]]
    sims = (a * b).sum(axis=1)
    return float(np.mean(sims))


def _random_order_baseline(
    *,
    vectors: np.ndarray,
    ordered_ids: Sequence[int],
    trials: int,
    seed: int,
) -> tuple[float, float]:
    if len(ordered_ids) < 2:
        return float("nan"), float("nan")

    idx = np.asarray(list(ordered_ids), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    vals = []

    for _ in range(max(1, int(trials))):
        perm = rng.permutation(idx)
        vals.append(_mean_adjacent_cosine(vectors, perm.tolist()))

    vals = np.asarray(vals, dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=0))


def _safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return float("nan")
    return float(a / b)


def _find_token_id_for_probe(tokenizer, word: str) -> Optional[int]:
    """
    Prefer exact vocab match. Fall back to tokenizing the word and using the
    single resulting token if possible.
    """
    vocab = tokenizer.get_vocab()
    if word in vocab:
        return int(vocab[word])

    toks = tokenizer.tokenize(word)
    if len(toks) == 1:
        ids = tokenizer.convert_tokens_to_ids(toks)
        if isinstance(ids, list):
            if len(ids) == 1 and ids[0] is not None:
                return int(ids[0])
        elif ids is not None:
            return int(ids)

    return None


def _format_rank_window(tokenizer, ordered_ids: Sequence[int], lo: int, hi: int, center_rank: int) -> str:
    parts = []
    for r in range(lo, hi):
        tid = int(ordered_ids[r])
        tok = _token_str(tokenizer, tid)
        marker = "*" if r == center_rank else " "
        parts.append(f"{marker}{r}:{tid}:{tok}")
    return " | ".join(parts)


def _print_probe_neighborhoods(
    *,
    tokenizer,
    ordered_ids: Sequence[int],
    probe_words: Sequence[str],
    half_window: int,
) -> None:
    if not probe_words:
        return

    rank_of = {int(tid): int(r) for r, tid in enumerate(ordered_ids)}

    print("\n[semantic-lm1b] Probe-centered semantic neighborhoods:")
    for word in probe_words:
        tid = _find_token_id_for_probe(tokenizer, word)
        if tid is None:
            print(f"  probe='{word}' -> not found as a single vocabulary token")
            continue

        if tid not in rank_of:
            print(f"  probe='{word}' token_id={tid} -> not present in semantic order")
            continue

        r = rank_of[tid]
        lo = max(0, r - half_window)
        hi = min(len(ordered_ids), r + half_window + 1)

        print(f"  probe='{word}' token_id={tid} semantic_rank={r}")
        print("    " + _format_rank_window(tokenizer, ordered_ids, lo, hi, r))
        print()


def _choose_start_positions(
    *,
    vocab_size: int,
    num_starts: int,
    start_pos: int,
    strategy: str,
    seed: int,
) -> List[int]:
    if vocab_size <= 0:
        return []

    num_starts = max(1, min(int(num_starts), int(vocab_size)))
    start_pos = int(start_pos) % int(vocab_size)
    strategy = str(strategy).lower()

    if num_starts == 1:
        return [start_pos]

    starts: List[int] = []

    if strategy == "even":
        stride = float(vocab_size) / float(num_starts)
        starts = [int(round(start_pos + k * stride)) % vocab_size for k in range(num_starts)]
    elif strategy == "random":
        rng = np.random.default_rng(int(seed))
        starts = [start_pos]
        remaining = np.arange(vocab_size, dtype=np.int64)
        remaining = remaining[remaining != start_pos]
        if len(remaining) > 0 and num_starts > 1:
            extra = rng.choice(remaining, size=num_starts - 1, replace=False)
            starts.extend(int(x) for x in extra.tolist())
    else:
        raise ValueError(f"Unknown start strategy '{strategy}'. Expected one of: even, random")

    deduped: List[int] = []
    seen = set()
    for s in starts:
        s = int(s) % vocab_size
        if s not in seen:
            seen.add(s)
            deduped.append(s)

    if len(deduped) < num_starts:
        for s in range(vocab_size):
            if s not in seen:
                seen.add(s)
                deduped.append(s)
                if len(deduped) >= num_starts:
                    break

    return deduped[:num_starts]


def _build_similarity_matrix(vectors: np.ndarray) -> Optional[np.ndarray]:
    sim_matrix = None
    try:
        sim_matrix = (vectors @ vectors.T).astype(np.float32, copy=False)
        np.fill_diagonal(sim_matrix, -9999.0)
        print("[semantic-lm1b] Using full similarity matrix.")
        return sim_matrix
    except MemoryError:
        print("[semantic-lm1b] Full similarity matrix too large; using on-the-fly dot products.")
        return None


def _pair_sims(
    *,
    vectors: np.ndarray,
    sim_matrix: Optional[np.ndarray],
    left: int,
    right: np.ndarray,
) -> np.ndarray:
    if sim_matrix is not None:
        return sim_matrix[left, right]
    return (vectors[right] @ vectors[left]).astype(np.float32, copy=False)


def _adjacent_sims(
    *,
    vectors: np.ndarray,
    sim_matrix: Optional[np.ndarray],
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    if sim_matrix is not None:
        return sim_matrix[left, right]
    return np.einsum("ij,ij->i", vectors[left], vectors[right]).astype(np.float32, copy=False)


def two_opt_refine_positions(
    *,
    vectors: np.ndarray,
    path_pos: Sequence[int],
    sim_matrix: Optional[np.ndarray],
    max_window: int,
    max_passes: int,
    min_delta: float = 1e-8,
) -> tuple[List[int], dict]:
    """
    Lightweight local 2-opt refinement for the path objective

        W(path) = sum_k S(path[k], path[k+1])

    Because cosine similarity is symmetric, reversing the middle segment preserves
    all internal edge contributions, so a 2-opt reversal changes only the two
    boundary edges. Therefore any accepted move with delta > 0 strictly improves W.

    We restrict candidates to j <= i + max_window for practicality.
    """
    P = np.asarray(list(path_pos), dtype=np.int64).copy()
    N = int(P.shape[0])

    if N < 4 or int(max_window) < 2 or int(max_passes) < 1:
        return P.tolist(), {
            "passes_run": 0,
            "total_swaps": 0,
            "total_gain": 0.0,
        }

    max_window = int(max_window)
    max_passes = int(max_passes)
    min_delta = float(min_delta)

    total_swaps = 0
    total_gain = 0.0
    passes_run = 0

    print(
        "[semantic-lm1b] Running lightweight 2-opt refinement:\n"
        f"  max_window={max_window}\n"
        f"  max_passes={max_passes}\n"
        f"  min_delta={min_delta}"
    )

    for pass_idx in range(max_passes):
        improved = False
        pass_swaps = 0
        pass_gain = 0.0

        i = 0
        while i < N - 3:
            a = int(P[i])
            b = int(P[i + 1])

            if sim_matrix is not None:
                old_ab = float(sim_matrix[a, b])
            else:
                old_ab = float(np.dot(vectors[a], vectors[b]))

            j_lo = i + 2
            j_hi = min(N - 2, i + max_window)

            if j_lo > j_hi:
                i += 1
                continue

            js = np.arange(j_lo, j_hi + 1, dtype=np.int64)
            c = P[js]
            d = P[js + 1]

            ac = _pair_sims(vectors=vectors, sim_matrix=sim_matrix, left=a, right=c)
            bd = _pair_sims(vectors=vectors, sim_matrix=sim_matrix, left=b, right=d)
            cd = _adjacent_sims(vectors=vectors, sim_matrix=sim_matrix, left=c, right=d)

            delta = ac + bd - old_ab - cd

            best_rel = int(np.argmax(delta))
            best_delta = float(delta[best_rel])

            if best_delta > min_delta:
                j = int(js[best_rel])

                # Reverse the middle segment [i+1, ..., j]
                P[i + 1 : j + 1] = P[i + 1 : j + 1][::-1]

                improved = True
                pass_swaps += 1
                total_swaps += 1
                pass_gain += best_delta
                total_gain += best_delta

                # Step back slightly because nearby edges changed
                i = max(0, i - 1)
            else:
                i += 1

        passes_run += 1
        print(
            f"[semantic-lm1b] 2-opt pass {pass_idx + 1}/{max_passes}: "
            f"swaps={pass_swaps}, gain={pass_gain:.6f}"
        )

        if not improved:
            print("[semantic-lm1b] 2-opt reached local optimum within the chosen window.")
            break

    return P.tolist(), {
        "passes_run": int(passes_run),
        "total_swaps": int(total_swaps),
        "total_gain": float(total_gain),
    }


def _greedy_nn_order_positions(
    *,
    vectors: np.ndarray,
    start_pos: int = 0,
    sim_matrix: Optional[np.ndarray] = None,
    show_progress: bool = True,
    progress_desc: str = "Ordering (greedy NN)",
) -> List[int]:
    V = int(vectors.shape[0])
    if V == 0:
        return []

    visited = np.zeros(V, dtype=bool)
    path_pos: List[int] = []

    curr = int(start_pos) % V
    visited[curr] = True
    path_pos.append(curr)

    iterator = range(V - 1)
    if show_progress:
        iterator = tqdm(iterator, desc=progress_desc)

    for _ in iterator:
        if sim_matrix is not None:
            sims = sim_matrix[curr]
        else:
            sims = vectors @ vectors[curr]

        masked = sims.copy()
        masked[visited] = -9999.0
        nxt = int(np.argmax(masked))

        visited[nxt] = True
        path_pos.append(nxt)
        curr = nxt

    return path_pos


def greedy_nn_order(vectors: np.ndarray, ids: List[int], start_pos: int = 0) -> List[int]:
    """
    Same greedy NN traversal style as the original script, kept as a thin wrapper
    for backward compatibility with the old single-start behavior.
    """
    sim_matrix = _build_similarity_matrix(vectors)
    path_pos = _greedy_nn_order_positions(
        vectors=vectors,
        start_pos=start_pos,
        sim_matrix=sim_matrix,
        show_progress=True,
        progress_desc="Ordering (greedy NN)",
    )
    return [int(ids[p]) for p in path_pos]


def greedy_nn_multi_start_order(
    *,
    vectors: np.ndarray,
    ids: List[int],
    tokenizer,
    num_starts: int,
    start_pos: int,
    start_strategy: str,
    seed: int,
    sim_matrix: Optional[np.ndarray] = None,
) -> tuple[List[int], dict]:
    V = int(vectors.shape[0])
    if V == 0:
        return [], {
            "best_score": float("nan"),
            "best_start_pos": -1,
            "best_start_token_id": -1,
            "best_path_pos": [],
            "per_start": [],
        }

    start_positions = _choose_start_positions(
        vocab_size=V,
        num_starts=num_starts,
        start_pos=start_pos,
        strategy=start_strategy,
        seed=seed,
    )

    if len(start_positions) == 1:
        print(
            "[semantic-lm1b] Running GREEDY nearest-neighbor ordering "
            f"from a single start: pos={start_positions[0]}"
        )
    else:
        print(
            "[semantic-lm1b] Running multi-start GREEDY nearest-neighbor ordering:\n"
            f"  num_starts={len(start_positions)}, strategy={start_strategy}, seed={seed}"
        )

    best_score = -float("inf")
    best_start = -1
    best_path_pos: List[int] = []
    per_start = []

    for idx, sp in enumerate(start_positions, start=1):
        start_tid = int(ids[sp])
        start_tok = _token_str(tokenizer, start_tid)

        desc = f"Ordering (greedy NN start {idx}/{len(start_positions)})"
        path_pos = _greedy_nn_order_positions(
            vectors=vectors,
            start_pos=sp,
            sim_matrix=sim_matrix,
            show_progress=(len(start_positions) == 1),
            progress_desc=desc,
        )
        score = _mean_adjacent_cosine_positions(vectors, path_pos)

        rec = {
            "start_index": int(idx - 1),
            "start_pos": int(sp),
            "start_token_id": int(start_tid),
            "score": float(score),
        }
        per_start.append(rec)

        print(
            f"[semantic-lm1b] start {idx:>2}/{len(start_positions)}: "
            f"pos={sp:>5} token_id={start_tid:>5} token={start_tok:<20} score={score:.6f}"
        )

        if score > best_score:
            best_score = float(score)
            best_start = int(sp)
            best_path_pos = path_pos

    best_start_tid = int(ids[best_start])
    print(
        "[semantic-lm1b] Winning greedy start:\n"
        f"  pos={best_start}\n"
        f"  token_id={best_start_tid}\n"
        f"  token={_token_str(tokenizer, best_start_tid)}\n"
        f"  score={best_score:.6f}"
    )

    best_ids = [int(ids[p]) for p in best_path_pos]
    meta = {
        "best_score": float(best_score),
        "best_start_pos": int(best_start),
        "best_start_token_id": int(best_start_tid),
        "best_path_pos": [int(p) for p in best_path_pos],
        "per_start": per_start,
    }
    return best_ids, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="datasets/lm1b")
    ap.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    ap.add_argument("--cache", type=str, default=None)
    ap.add_argument("--cache_meta", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)

    ap.add_argument("--method", type=str, default="greedy", choices=["greedy", "pca"])
    ap.add_argument("--vec_dim", type=int, default=64)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--start_pos", type=int, default=0)
    ap.add_argument("--num_starts", type=int, default=1)
    ap.add_argument("--start_strategy", type=str, default="even", choices=["even", "random"])
    
    ap.add_argument("--two_opt_window", type=int, default=0)
    ap.add_argument("--two_opt_passes", type=int, default=0)
    ap.add_argument("--two_opt_min_delta", type=float, default=1e-8)
    
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--random_baseline_trials", type=int, default=16)

    ap.add_argument(
        "--probe_words",
        type=str,
        nargs="*",
        default=[
            "attackers",
            "evacuated",
            "driver",
            "car",
            "television",
            "german",
            "music",
            "science",
            "church",
            "king",
        ],
    )
    ap.add_argument("--probe_half_window", type=int, default=6)

    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit("Missing transformers. Install: pip install transformers") from e

    try:
        from gensim.models import Word2Vec
    except ImportError as e:
        raise SystemExit("Missing gensim. Install: pip install gensim") from e

    root = Path(args.root)
    cache_path = Path(args.cache) if args.cache else (root / "cache_train_tokens.uint16")
    cache_meta = Path(args.cache_meta) if args.cache_meta else (root / "cache_train_tokens.meta.json")
    out_path = Path(args.out) if args.out else (root / "semantic_mapping_bert_base_uncased.json")

    if out_path.exists() and not args.force:
        print(f"[semantic-lm1b] exists -> {out_path} (skip; pass --force to overwrite)")
        return

    if not cache_path.exists() or not cache_meta.exists():
        raise SystemExit(
            "Missing token cache. Run scripts/build_lm1b_bert_caches.py first.\n"
            f"Expected: {cache_path} and {cache_meta}"
        )

    tok = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    vocab_size = int(tok.vocab_size)

    reserved = []
    for tid in [tok.pad_token_id, tok.unk_token_id, tok.cls_token_id, tok.sep_token_id, tok.mask_token_id]:
        if tid is None:
            continue
        tid = int(tid)
        if 0 <= tid < vocab_size and tid not in reserved:
            reserved.append(tid)

    print(f"[semantic-lm1b] reserved token ids kept fixed at front: {reserved}")
    if reserved:
        print(
            "[semantic-lm1b] reserved tokens: "
            + ", ".join(f"{tid}:{_token_str(tok, tid)}" for tid in reserved)
        )

    sentences = TokenRowIterator(
        cache_path=cache_path,
        meta_path=cache_meta,
        ignore_ids=reserved,
        max_rows=args.max_rows,
    )

    print(
        "[semantic-lm1b] Training Word2Vec (skip-gram) on token rows:\n"
        f"  vec_dim={args.vec_dim}, window={args.window}, epochs={args.epochs}, workers={args.workers}, seed={args.seed}\n"
        f"  cache={cache_path}\n"
        f"  max_rows={args.max_rows}"
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

    print(f"[semantic-lm1b] Vectors ready. Found {found}/{vocab_size} token vectors in Word2Vec model.")

    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

    free_ids = [tid for tid in range(vocab_size) if tid not in reserved]
    free_vecs = vectors[np.asarray(free_ids, dtype=np.int64)]

    need_sim_matrix = (args.method == "greedy") or (
        int(args.two_opt_passes) > 0 and int(args.two_opt_window) > 0
    )
    sim_matrix = _build_similarity_matrix(free_vecs) if need_sim_matrix else None

    search_meta = None
    if args.method == "greedy":
        ordered_free, search_meta = greedy_nn_multi_start_order(
            vectors=free_vecs,
            ids=free_ids,
            tokenizer=tok,
            num_starts=int(args.num_starts),
            start_pos=int(args.start_pos),
            start_strategy=str(args.start_strategy),
            seed=int(args.seed),
            sim_matrix=sim_matrix,
        )
    else:
        print("[semantic-lm1b] Running PCA-1D ordering...")
        ordered_free = pca_1d_order(free_vecs, free_ids)

    if int(args.two_opt_passes) > 0 and int(args.two_opt_window) > 0:
        if search_meta is not None and "best_path_pos" in search_meta:
            ordered_free_pos = [int(x) for x in search_meta["best_path_pos"]]
        else:
            id_to_pos = {int(tid): pos for pos, tid in enumerate(free_ids)}
            ordered_free_pos = [int(id_to_pos[int(tid)]) for tid in ordered_free]

        refined_pos, refine_meta = two_opt_refine_positions(
            vectors=free_vecs,
            path_pos=ordered_free_pos,
            sim_matrix=sim_matrix,
            max_window=int(args.two_opt_window),
            max_passes=int(args.two_opt_passes),
            min_delta=float(args.two_opt_min_delta),
        )

        ordered_free = [int(free_ids[p]) for p in refined_pos]

        print(
            "[semantic-lm1b] 2-opt summary:\n"
            f"  passes_run={refine_meta['passes_run']}\n"
            f"  total_swaps={refine_meta['total_swaps']}\n"
            f"  total_gain={refine_meta['total_gain']:.6f}"
        )

    final_order = list(reserved) + ordered_free
    if len(final_order) != vocab_size:
        raise RuntimeError("Final semantic order does not cover the full vocabulary")

    old_to_new = {int(old_id): int(new_idx) for new_idx, old_id in enumerate(final_order)}
    new_to_old = {int(new_idx): int(old_id) for new_idx, old_id in enumerate(final_order)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tokenizer_name": args.tokenizer_name,
                "reserved_token_ids": reserved,
                "old_to_new": old_to_new,
                "new_to_old": new_to_old,
            },
            f,
            indent=2,
        )

    print(f"[semantic-lm1b] wrote semantic map -> {out_path}")

    if search_meta is not None:
        print(
            "[semantic-lm1b] Greedy search summary:\n"
            f"  best_start_pos={search_meta['best_start_pos']}\n"
            f"  best_start_token_id={search_meta['best_start_token_id']}\n"
            f"  best_score={search_meta['best_score']:.6f}"
        )

    semantic_adj = _mean_adjacent_cosine(vectors, ordered_free)
    raw_id_adj = _mean_adjacent_cosine(vectors, free_ids)
    rand_mean, rand_std = _random_order_baseline(
        vectors=vectors,
        ordered_ids=ordered_free,
        trials=int(args.random_baseline_trials),
        seed=int(args.seed),
    )

    print("\n[semantic-lm1b] Adjacency cosine report (non-reserved vocabulary only)")
    print(f"  semantic-order mean adjacent cosine : {semantic_adj:.6f}")
    print(f"  raw token-id order cosine           : {raw_id_adj:.6f}")
    print(f"  random-order cosine                 : {rand_mean:.6f} ± {rand_std:.6f}")
    print(f"  gain vs random (abs)                : {semantic_adj - rand_mean:.6f}")
    print(f"  gain vs raw-id (abs)                : {semantic_adj - raw_id_adj:.6f}")
    print(f"  ratio vs random                     : {_safe_ratio(semantic_adj, rand_mean):.4f}x")
    print(f"  ratio vs raw-id                     : {_safe_ratio(semantic_adj, raw_id_adj):.4f}x")

    _print_probe_neighborhoods(
        tokenizer=tok,
        ordered_ids=ordered_free,
        probe_words=args.probe_words,
        half_window=int(args.probe_half_window),
    )


if __name__ == "__main__":
    main()