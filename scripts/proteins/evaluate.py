#!/usr/bin/env python3
"""Run protein evaluation with the paper config."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_config(path):
    spec = importlib.util.spec_from_file_location("protein_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def command_for(
    task, cfg, config, checkpoint, out, device, python=sys.executable
):
    """Build one evaluation command without running it."""
    ev = cfg.evaluation
    common = [
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--device",
        device,
        "--seed",
        str(ev.seed),
        "--out",
        str(out),
    ]
    if task == "entropy":
        profile = ev.entropy
        module = "entropy_profile"
        extra = [
            "--validation-data",
            ev.validation_data,
            "--entropy-out-dir",
            str(checkpoint.parent.parent),
            "--struct-tokenizer",
            ev.struct_tokenizer,
            "--num-validation-backbones",
            str(profile.num_backbones),
            "--samples-per-unit",
            str(profile.samples_per_unit),
            "--num-bins",
            str(profile.num_bins),
            "--subset-seed",
            str(profile.seed),
        ]
    elif task == "inverse":
        module = "inverse_folding"
        extra = [
            "--data",
            ev.inverse_data,
            "--struct-tokenizer",
            ev.struct_tokenizer,
            "--methods",
            "model",
            "--num-samples",
            str(ev.num_samples),
            "--sc-designs",
            "1",
        ]
    elif task == "forward":
        module = "folding"
        extra = [
            "--targets",
            ev.forward_targets,
            "--tokenizer-path",
            ev.struct_tokenizer,
            "--num-samples",
            str(ev.num_samples),
            "--skip-controls",
            "--skip-ceiling",
        ]
    elif task == "cogeneration":
        module = "cogeneration"
        extra = [
            "--struct-tokenizer-path",
            ev.struct_tokenizer,
            "--lengths",
            *map(str, ev.length_grid),
            "--samples-per-length",
            str(ev.samples_per_length),
            "--n-design-seqs",
            "1",
            "--sc-temperature",
            "0.1",
        ]
    else:
        raise ValueError(f"Unknown evaluation task: {task}")
    module_name = "evaluation.evaluation_drivers.proteins." + module
    return [python, "-m", module_name] + common + extra


def environment():
    """Set local model paths for evaluation."""
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(ROOT / "hf_cache"))
    env.setdefault("HF_HUB_CACHE", str(ROOT / "hf_cache/hub"))
    env.setdefault("PROTEINMPNN_DIR", str(ROOT / "external/ProteinMPNN"))
    env.setdefault(
        "PROTEINMPNN_WEIGHTS",
        str(ROOT / "external/ProteinMPNN/vanilla_model_weights/v_48_020.pt"),
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT / "evaluation/evaluation_drivers/proteins/compat"),
            str(ROOT / "external/openfold"),
            str(ROOT),
        ]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(key, "1")
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task", choices=["entropy", "inverse", "forward", "cogeneration"]
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/proteins/proteins_400m.py",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    checkpoint = (
        args.checkpoint or ROOT / cfg.evaluation.checkpoint_path
    ).resolve()
    profile = checkpoint.parent.parent / "entropy_profile.json"
    out = args.out or (
        profile
        if args.task == "entropy"
        else ROOT / cfg.evaluation.out_dir / (args.task + ".json")
    )
    command = command_for(
        args.task, cfg, args.config, checkpoint, out, args.device
    )
    if args.resume:
        command.append("--resume")
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    if args.task != "entropy":
        if not profile.is_file():
            parser.error("Run the entropy task for this checkpoint first.")
        from evaluation.evaluation_drivers.proteins.io import sha256

        report = json.loads(profile.read_text())
        if report["checkpoint"]["sha256"] != sha256(checkpoint):
            parser.error("The entropy profile belongs to another checkpoint.")
        for name, digest in report["results"]["entropy_tables"].items():
            if sha256(profile.parent / name) != digest:
                parser.error(f"Entropy table changed: {name}")
    subprocess.run(command, cwd=ROOT, env=environment(), check=True)


if __name__ == "__main__":
    main()
