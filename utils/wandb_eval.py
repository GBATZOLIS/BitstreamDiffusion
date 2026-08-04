"""Attach evaluation metrics to the training run's W&B run.

Evaluation runs as a separate process (evaluation/run_eval.py), after training,
so its metrics would otherwise land in a different place from the training
curves. Training writes a ``wandb_run.json`` into the run directory recording the
W&B run identity; this helper reads it and resumes that exact run so eval metrics
(logged under an ``eval/`` prefix and mirrored into the run summary) appear on the
same W&B run as the loss curves.

Scalar metrics become both a logged point and a summary value; non-scalar entries
(dicts, lists, NaNs) are skipped for the scalar log but kept in the summary as-is.
It is a no-op (returns False) when W&B is unavailable, no ``wandb_run.json`` is
present, or logging is disabled, so callers can invoke it unconditionally.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional

try:
    import wandb

    _WANDB = True
except ImportError:  # pragma: no cover - optional dependency
    wandb = None
    _WANDB = False


def _is_scalar(value: object) -> bool:
    """True for a finite int/float (bool excluded) that W&B can chart."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def read_run_identity(run_dir: Path) -> Optional[dict]:
    """Return the recorded W&B run identity for a run directory, or None."""
    path = Path(run_dir) / "wandb_run.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and data.get("id") else None


def log_eval_metrics(
    run_dir: str,
    metrics: Mapping[str, object],
    *,
    prefix: str = "eval",
    tag: Optional[str] = None,
) -> bool:
    """Resume the training run's W&B run and log ``metrics`` onto it.

    ``run_dir`` is the training run directory (holds ``wandb_run.json``).
    ``prefix`` namespaces the keys (default ``eval``); ``tag`` optionally adds a
    second level (e.g. the eval split or task) so ``eval/<tag>/<metric>``. Returns
    True when metrics were logged, False on any no-op condition.
    """
    if not _WANDB or wandb is None:
        return False
    identity = read_run_identity(Path(run_dir))
    if identity is None:
        return False

    parts = [prefix] + ([tag] if tag else [])
    key_prefix = "/".join(parts)
    scalar_payload = {}
    summary_payload = {}
    for key, value in metrics.items():
        full = f"{key_prefix}/{key}"
        summary_payload[full] = value
        if _is_scalar(value):
            scalar_payload[full] = float(value)

    run = wandb.init(
        id=identity["id"],
        entity=identity.get("entity"),
        project=identity.get("project"),
        resume="allow",
        job_type="eval",
        reinit=True,
    )
    if scalar_payload:
        wandb.log(scalar_payload)
    for key, value in summary_payload.items():
        try:
            run.summary[key] = value
        except (TypeError, ValueError):
            run.summary[key] = str(value)
    wandb.finish()
    return True
