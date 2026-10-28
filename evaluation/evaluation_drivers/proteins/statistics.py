"""Small deterministic uncertainty helpers for protein evaluation reports."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260810


def power_of_two_prefix_budgets(num_candidates: int) -> list[int]:
    """Return nested reporting budgets such as [1, 2, 4, 8]."""
    num_candidates = int(num_candidates)
    if num_candidates < 1:
        return []
    budgets = []
    budget = 1
    while budget < num_candidates:
        budgets.append(budget)
        budget *= 2
    budgets.append(num_candidates)
    return budgets


def finite_values(values: Iterable[object]) -> np.ndarray:
    """Return the finite numeric members of *values* as a float64 vector."""
    result = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return np.asarray(result, dtype=np.float64)


def bootstrap_mean_ci(
    values: Iterable[object],
    *,
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict | None:
    """Percentile-bootstrap confidence interval for a target-level mean."""
    array = finite_values(values)
    if array.size == 0:
        return None
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    if int(n_resamples) < 1:
        raise ValueError("n_resamples must be positive")
    if array.size == 1:
        low = high = float(array[0])
    else:
        rng = np.random.default_rng(int(seed))
        means = np.empty(int(n_resamples), dtype=np.float64)
        batch_size = max(1, min(512, int(n_resamples)))
        for start in range(0, int(n_resamples), batch_size):
            stop = min(start + batch_size, int(n_resamples))
            draw = rng.integers(
                0, array.size, size=(stop - start, array.size), endpoint=False
            )
            means[start:stop] = array[draw].mean(axis=1)
        alpha = (1.0 - float(confidence)) / 2.0
        low, high = np.quantile(means, [alpha, 1.0 - alpha]).tolist()
    return {
        "low": float(low),
        "high": float(high),
        "confidence": float(confidence),
        "method": "target_level_percentile_bootstrap_mean",
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> dict | None:
    """Wilson 95% score interval for a binomial success proportion."""
    successes = int(successes)
    total = int(total)
    if total == 0:
        return None
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total")
    p = successes / total
    z2 = float(z) ** 2
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    half = (
        float(z)
        * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
        / denominator
    )
    return {
        "low": float(max(0.0, center - half)),
        "high": float(min(1.0, center + half)),
        "confidence": 0.95,
        "method": "wilson_score",
        "successes": successes,
        "total": total,
    }


def fraction_with_wilson(
    flags: Sequence[object],
) -> tuple[float | None, dict | None]:
    """Return a defined-flag fraction and its Wilson interval."""
    defined = [bool(flag) for flag in flags if flag is not None]
    if not defined:
        return None, None
    successes = sum(defined)
    return successes / len(defined), wilson_interval(successes, len(defined))
