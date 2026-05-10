#evaluation/mauve.py
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


def _require_mauve():
    try:
        import mauve  # type: ignore
        return mauve
    except Exception as e:
        raise ImportError(
            "MAUVE not found. Install it via: pip install mauve-text\n"
            "Then re-run."
        ) from e


@dataclass
class MauveConfig:
    featurize_model_name: str = "gpt2"
    max_text_length: int = 256
    num_buckets: int = 25
    seed: int = 0
    verbose: bool = False
    device_id: Optional[int] = None
    batch_size: int = 128


def compute_mauve(
    *,
    p_text: List[str],
    q_text: List[str],
    cfg: Optional[MauveConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = MauveConfig()

    mauve = _require_mauve()

    print(
        f"[Mauve Internal] Starting compute_mauve on {len(p_text)} samples "
        f"with batch_size={cfg.batch_size}, model={cfg.featurize_model_name}, "
        f"max_len={cfg.max_text_length}, device_id={cfg.device_id}",
        file=sys.stderr,
    )
    print(
        f"[Mauve Internal] OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')} "
        f"TOKENIZERS_PARALLELISM={os.environ.get('TOKENIZERS_PARALLELISM')}",
        file=sys.stderr,
    )

    t0 = time.time()

    out = mauve.compute_mauve(
        p_text=p_text,
        q_text=q_text,
        device_id=cfg.device_id,
        featurize_model_name=cfg.featurize_model_name,
        max_text_length=int(cfg.max_text_length),
        num_buckets=int(cfg.num_buckets),
        batch_size=int(cfg.batch_size),
        verbose=bool(cfg.verbose),
    )

    dt = time.time() - t0
    print(f"[Mauve Internal] compute_mauve finished in {dt:.2f}s", file=sys.stderr)

    score = float(getattr(out, "mauve", np.nan))
    res: Dict[str, Any] = {"mauve": score}

    extra_keys = (
        "frontier_integral",
        "p_entropy",
        "q_entropy",
        "kl_pq",
        "kl_qp",
        "mauve_star",
        "frontier_integral_star",
    )
    for k in extra_keys:
        if hasattr(out, k):
            try:
                val = getattr(out, k)
                if isinstance(val, (int, float, np.number)):
                    res[k] = float(val)
            except Exception:
                pass

    if hasattr(out, "divergence_curve"):
        res["divergence_curve"] = getattr(out, "divergence_curve")

    return res


class MauveEvaluator:
    def __init__(self, *, cfg: Optional[MauveConfig] = None):
        self.cfg = cfg or MauveConfig()

    def score(
        self,
        *,
        p_text: List[str],
        q_text: List[str],
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        if batch_size is not None:
            cfg = MauveConfig(
                featurize_model_name=cfg.featurize_model_name,
                max_text_length=cfg.max_text_length,
                num_buckets=cfg.num_buckets,
                seed=cfg.seed,
                verbose=cfg.verbose,
                device_id=cfg.device_id,
                batch_size=int(batch_size),
            )
        return compute_mauve(p_text=p_text, q_text=q_text, cfg=cfg)