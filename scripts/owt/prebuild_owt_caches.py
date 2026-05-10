#scripts/owt/prebuild_owt_caches.py
from __future__ import annotations

import argparse
from pathlib import Path

from ml_collections import ConfigDict

# Adjust this import path if your dataset file is named differently
from data.openwebtext import ensure_openwebtext_caches_ready

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Python config module path, e.g. configs/owt/rate_debug.py")
    args = parser.parse_args()

    # Load the config dynamically
    cfg_globals = {}
    with open(args.config, "r", encoding="utf-8") as f:
        code = compile(f.read(), args.config, "exec")
        exec(code, cfg_globals)

    if "get_config" not in cfg_globals:
        raise RuntimeError(f"{args.config} does not define get_config()")

    cfg = cfg_globals["get_config"]()

    print(f"\n[prebuild] Starting offline cache generation using config: {args.config}")
    
    print("\n[prebuild] Ensuring TRAIN caches...")
    ensure_openwebtext_caches_ready(cfg, split="train")

    print("\n[prebuild] Ensuring VAL caches...")
    ensure_openwebtext_caches_ready(cfg, split="val")

    print("\n[prebuild] All caches successfully built! You are ready to train.")

if __name__ == "__main__":
    main()