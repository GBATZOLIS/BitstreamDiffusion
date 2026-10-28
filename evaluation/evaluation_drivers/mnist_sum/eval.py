"""Unified-evaluator adapter for the paper-grade MNIST-sum protocol."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[3]
_OFFLINE_SCRIPT = _ROOT / "scripts" / "mnist_sum" / "eval_mnist_sum_offline.py"


def evaluate_mnist_sum(args, cfg) -> None:
    """Run the existing offline protocol through ``evaluation.run_eval``."""
    checkpoint = getattr(args, "mnist_sum_checkpoint", None)
    if checkpoint is None:
        checkpoint = getattr(cfg.evaluation, "checkpoint_path", None)
    if not checkpoint:
        raise ValueError(
            "MNIST-sum evaluation needs --mnist_sum_checkpoint or "
            "cfg.evaluation.checkpoint_path"
        )

    out = getattr(args, "mnist_sum_out", None)
    if out is None:
        out = getattr(cfg.evaluation, "out_dir", "results/mnist_sum")

    command = [
        sys.executable,
        str(_OFFLINE_SCRIPT),
        "--config",
        str(args.config),
        "--ckpt",
        str(checkpoint),
        "--out",
        str(out),
    ]
    scalar_options = {
        "mnist_sum_n": "--n",
        "mnist_sum_n_joint": "--n_joint",
        "mnist_sum_micro_batch_size": "--micro",
        "mnist_sum_classifier_path": "--clf_path",
    }
    for attribute, option in scalar_options.items():
        value = getattr(args, attribute, None)
        if value is not None:
            command.extend([option, str(value)])

    skip = getattr(args, "mnist_sum_skip", None)
    if skip:
        command.extend(["--skip", *map(str, skip)])
    if bool(getattr(args, "no_ema", False)):
        command.append("--no-ema")
    if bool(getattr(args, "mnist_sum_no_compile", False)):
        command.append("--no-compile")
    if bool(getattr(args, "mnist_sum_fp32", False)):
        command.append("--fp32")
    if bool(getattr(args, "mnist_sum_allow_karras", False)):
        command.append("--allow-karras")

    subprocess.run(command, cwd=_ROOT, check=True)
