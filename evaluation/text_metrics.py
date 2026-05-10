from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

import torch

_WORD_RE = re.compile(r"\S+")


def _words(s: str) -> List[str]:
    return _WORD_RE.findall((s or "").lower())


def rep_n(texts: List[str], n: int) -> float:
    total = 0
    repeated = 0
    for txt in texts:
        toks = _words(txt)
        if len(toks) < n:
            continue
        grams = [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]
        counts = Counter(grams)
        total += len(grams)
        repeated += sum(c - 1 for c in counts.values() if c > 1)
    return 100.0 * repeated / max(total, 1)


def distinct_n(texts: List[str], n: int) -> float:
    total = 0
    uniq = set()
    for txt in texts:
        toks = _words(txt)
        if len(toks) < n:
            continue
        grams = [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        uniq.update(grams)
    return 100.0 * len(uniq) / max(total, 1)


def diversity_234(texts: List[str]) -> float:
    r2 = rep_n(texts, 2) / 100.0
    r3 = rep_n(texts, 3) / 100.0
    r4 = rep_n(texts, 4) / 100.0
    return (1.0 - r2) * (1.0 - r3) * (1.0 - r4)


def unigram_entropy(text: str) -> float:
    toks = _words(text)
    T = len(toks)
    if T == 0:
        return 0.0
    counts = Counter(toks)
    H = 0.0
    for c in counts.values():
        p = c / T
        H -= p * math.log(p)  # nats
    return H


def avg_unigram_entropy(texts: List[str]) -> float:
    if not texts:
        return float("nan")
    return sum(unigram_entropy(t) for t in texts) / len(texts)


@torch.no_grad()
def token_unigram_entropy_from_token_ids(tokens_1d: torch.Tensor) -> float:
    """
    tokens_1d: [T] integer token ids
    returns unigram entropy in nats
    """
    tokens_1d = tokens_1d.view(-1).to(torch.long)
    if tokens_1d.numel() == 0:
        return 0.0
    _, counts = torch.unique(tokens_1d, return_counts=True, sorted=False)
    p = counts.float() / counts.sum()
    return float(torch.special.entr(p).sum().item())


@torch.no_grad()
def avg_token_unigram_entropy_from_token_ids(samples_2d: torch.Tensor) -> float:
    """
    samples_2d: [N, T] integer token ids
    returns mean per-sample unigram entropy in nats
    """
    samples_2d = samples_2d.to(torch.long)
    if samples_2d.numel() == 0:
        return float("nan")
    vals = [token_unigram_entropy_from_token_ids(row) for row in samples_2d]
    return float(sum(vals) / max(len(vals), 1))