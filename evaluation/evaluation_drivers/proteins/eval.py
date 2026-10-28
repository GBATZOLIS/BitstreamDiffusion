"""Unified-evaluator adapter for the protein benchmark protocols."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_OFFLINE_SCRIPT = _ROOT / "scripts" / "proteins" / "evaluate.py"
_DEFAULT_TASKS = ("entropy", "inverse", "forward", "cogeneration")


def evaluate_proteins(args, cfg) -> None:
    """Run selected protein protocols through ``evaluation.run_eval``."""
    tasks = tuple(getattr(args, "protein_tasks", None) or _DEFAULT_TASKS)
    checkpoint = getattr(args, "protein_checkpoint", None)
    out_dir = getattr(args, "protein_out_dir", None)
    device = getattr(args, "protein_device", None) or str(cfg.device)

    for task in tasks:
        command = [
            sys.executable,
            str(_OFFLINE_SCRIPT),
            str(task),
            "--config",
            str(args.config),
            "--device",
            device,
        ]
        if checkpoint is not None:
            command.extend(["--checkpoint", str(checkpoint)])
        if out_dir is not None and task != "entropy":
            command.extend(["--out", str(Path(out_dir) / f"{task}.json")])
        if bool(getattr(args, "protein_resume", False)):
            command.append("--resume")
        subprocess.run(command, cwd=_ROOT, check=True)
