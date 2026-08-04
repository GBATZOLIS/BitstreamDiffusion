from __future__ import annotations

import numpy as np


def stratified_prefix_indices(lengths: np.ndarray, limit: int) -> np.ndarray:
    """Select a deterministic, approximately even prefix from each length."""
    lengths = np.asarray(lengths)
    limit = min(int(limit), len(lengths))
    values = sorted(set(lengths.tolist()))
    base, remainder = divmod(limit, len(values))
    selected = []
    for position, value in enumerate(values):
        quota = base + int(position < remainder)
        selected.extend(np.flatnonzero(lengths == value)[:quota].tolist())
    if len(selected) < limit:
        used = set(selected)
        selected.extend(i for i in range(len(lengths)) if i not in used)
    return np.asarray(selected[:limit], dtype=np.int64)
