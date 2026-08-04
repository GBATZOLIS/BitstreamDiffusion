"""Shared Weights & Biases defaults for every protein experiment.

Call ``enable_wandb(cfg, group=...)`` from a protein config so the run logs to
W&B by default with the full TensorBoard stream mirrored in (sync_tensorboard),
and writes a ``wandb_run.json`` that later eval can attach to. Every value falls
back to an environment variable so a run can be redirected or silenced without
editing configs:

  COBIT_WANDB=0        -> disable W&B for this run (default on)
  WANDB_MODE=offline   -> log locally, `wandb sync` later (default online)
  WANDB_MODE=disabled  -> hard off (honored by the trainer and by wandb itself)
  WANDB_ENTITY=<name>  -> override the entity (default "trentini")
  WANDB_PROJECT=<name> -> override the project (default "cobit-proteins")

The trainer reads the same environment variables at init, so the two stay in
agreement regardless of which side sets them.
"""

from __future__ import annotations

import os

DEFAULT_ENTITY = "trentini"
DEFAULT_PROJECT = "cobit-proteins"


def _wandb_enabled_default() -> bool:
    """W&B is on unless COBIT_WANDB is a falsey value or WANDB_MODE is disabled."""
    if os.environ.get("WANDB_MODE", "").lower() in ("disabled", "dryrun"):
        return False
    return os.environ.get("COBIT_WANDB", "1").lower() not in ("0", "false", "no", "off")


def enable_wandb(cfg, *, group: str):
    """Set the W&B logging fields on ``cfg`` from environment-backed defaults."""
    cfg.logging.use_wandb = _wandb_enabled_default()
    cfg.logging.mode = os.environ.get("WANDB_MODE", "online")
    cfg.logging.entity = os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY)
    cfg.logging.project = os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT)
    cfg.logging.group = group
    # Mirror the full TensorBoard stream (scalars, figures, images, histograms,
    # text) into W&B so every metric appears in the dashboard automatically.
    cfg.logging.sync_tensorboard = True
    return cfg
