#!/usr/bin/env python3
"""Backfill existing TensorBoard runs into Weights & Biases.

Runs trained before W&B logging was enabled only have local TensorBoard event
files under ``runs/<experiment>/training_logs`` (or ``training_logs_ar``). This
script replays each run's scalar history into a W&B run so the whole history is
visible in one place, without retraining.

It is idempotent: each imported run uses a stable W&B id derived from its run
directory, so re-running updates the same W&B run instead of creating duplicates.
Scalars are replayed against their original global step. Media (figures/images/
histograms) are not backfilled - only scalar curves - because that is the part
that matters for comparing runs; new runs capture media live via sync_tensorboard.

Usage:
  python scripts/proteins/setup/import_tb_runs_to_wandb.py                 # all runs/proteins/*
  python scripts/proteins/setup/import_tb_runs_to_wandb.py --runs-root runs/proteins --entity trentini --project cobit-proteins
  python scripts/proteins/setup/import_tb_runs_to_wandb.py --only swissprot_char5 evodiff_uniref50_bitstream_seed0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_RUNS_ROOT = "runs/proteins"
DEFAULT_ENTITY = "trentini"
DEFAULT_PROJECT = "cobit-proteins"
TB_SUBDIRS = ("training_logs", "training_logs_ar")


def _stable_id(key: str) -> str:
    """A deterministic 8-byte hex W&B run id so re-imports update in place."""
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def find_tb_dirs(runs_root: Path, only: Optional[List[str]]) -> List[Path]:
    """Return every run directory that holds a TensorBoard event file."""
    found: List[Path] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        if only and run_dir.name not in only:
            continue
        for sub in TB_SUBDIRS:
            tb = run_dir / sub
            if tb.is_dir() and any(tb.glob("events.out.tfevents.*")):
                found.append(tb)
    return found


def _load_run_config(run_dir: Path) -> Dict[str, object]:
    """Read the run's saved config.json if present, for W&B run config."""
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_scalars(tb_dir: Path) -> Dict[str, List]:
    """Return {tag: [(step, value), ...]} for every scalar tag in a TB dir."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    ea = EventAccumulator(
        str(tb_dir), size_guidance={"scalars": 0}
    )  # 0 = load all
    ea.Reload()
    out: Dict[str, List] = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = [(int(s.step), float(s.value)) for s in ea.Scalars(tag)]
    return out


def import_one(
    tb_dir: Path,
    *,
    entity: str,
    project: str,
    mode: str,
) -> Optional[str]:
    """Replay one TB dir into a W&B run; return the run URL (or None if empty)."""
    import wandb

    run_dir = tb_dir.parent
    scalars = read_scalars(tb_dir)
    if not scalars:
        return None

    experiment = run_dir.name
    suffix = "" if tb_dir.name == "training_logs" else f"-{tb_dir.name}"
    run_name = f"{experiment}{suffix}"
    run_id = _stable_id(str(tb_dir.resolve()))
    saved_cfg = _load_run_config(run_dir)
    group = None
    if isinstance(saved_cfg.get("logging"), dict):
        group = saved_cfg["logging"].get("group")

    run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="allow",
        name=run_name,
        group=group,
        job_type="tb-import",
        tags=["imported", "tensorboard-backfill"],
        config=saved_cfg or None,
        mode=mode,
        reinit=True,
    )
    # Replay in global-step order so W&B's step stays monotonic across all tags.
    per_step: Dict[int, Dict[str, float]] = {}
    for tag, series in scalars.items():
        for step, value in series:
            per_step.setdefault(step, {})[tag] = value
    for step in sorted(per_step):
        wandb.log(per_step[step], step=step)
    url = run.get_url()
    n = sum(len(v) for v in scalars.values())
    print(f"  imported {run_name}: {len(scalars)} tags, {n} points -> {url}")
    wandb.finish()
    return url


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the TensorBoard backfill importer."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--entity", default=DEFAULT_ENTITY)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--mode", default="online", choices=["online", "offline"])
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Only import these run directory names (default: all).",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Import every matching TensorBoard run directory into W&B."""
    args = parse_args(argv)
    runs_root = Path(args.runs_root)
    if not runs_root.is_dir():
        raise SystemExit(f"runs root not found: {runs_root}")
    tb_dirs = find_tb_dirs(runs_root, args.only)
    if not tb_dirs:
        print(f"No TensorBoard runs found under {runs_root}.")
        return 0
    print(f"Importing {len(tb_dirs)} TensorBoard run(s) into {args.entity}/{args.project}:")
    for tb_dir in tb_dirs:
        try:
            import_one(
                tb_dir, entity=args.entity, project=args.project, mode=args.mode
            )
        except Exception as exc:  # pragma: no cover - per-run robustness
            print(f"  FAILED {tb_dir}: {exc!r}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
