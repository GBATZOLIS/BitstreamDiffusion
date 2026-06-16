"""Pre-build a task dataset cache in a single process (before torchrun).

Ensures the Sudoku / TinyGSM cache exists before DDP training starts, so no
rank generates data under the process group. Usage:

    python scripts/tasks/prebuild_task.py --config configs/tasks/sudoku_bits.py
"""
import argparse
import importlib.util
import os
import sys

# Ensure repo root is importable when run as a script from scripts/tasks/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("cfg", args.config)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.get_config()

    name = str(cfg.data.dataset).lower()
    if name in {"sudoku"}:
        from data.sudoku import SudokuDataset as DS
    elif name in {"tinygsm", "gsm8k"}:
        from data.tinygsm import TinyGSMDataset as DS
    else:
        raise SystemExit(f"prebuild not supported for dataset={name!r}")

    print(f"[prebuild] building train split for {name} ...", flush=True)
    tr = DS(cfg, split="train")
    print(f"[prebuild] train n={len(tr)}", flush=True)
    print(f"[prebuild] building val split ...", flush=True)
    va = DS(cfg, split="val")
    print(f"[prebuild] val n={len(va)} — cache ready", flush=True)


if __name__ == "__main__":
    main()
