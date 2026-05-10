# autoregression/utils.py
from __future__ import annotations

import os
import json
import random
import importlib.util
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist


def load_config(path: str):
    spec = importlib.util.spec_from_file_location("config", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.get_config()


def cfg_to_dict(cfg) -> Dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if isinstance(cfg, dict):
        return {k: cfg_to_dict(v) for k, v in cfg.items()}
    out: Dict[str, Any] = {}
    for k, v in cfg.__dict__.items():
        if k.startswith("_"):
            continue
        out[k] = cfg_to_dict(v) if hasattr(v, "__dict__") else v
    return out


def maybe_init_distributed(cfg):
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    inferred = env_world > 1

    if not hasattr(cfg, "system"):
        from ml_collections import config_dict
        cfg.system = config_dict.ConfigDict()

    if getattr(cfg.system, "distributed", False) or inferred:
        cfg.system.distributed = True
        cfg.system.world_size = env_world
        cfg.system.global_rank = int(os.environ.get("RANK", "0"))
        cfg.system.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
    else:
        cfg.system.distributed = False
        cfg.system.world_size = 1
        cfg.system.global_rank = 0
        cfg.system.local_rank = 0

    return cfg


def destroy_distributed_if_needed(cfg):
    if bool(getattr(cfg.system, "distributed", False)) and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def unwrap_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    clean = {}
    for k, v in state_dict.items():
        k = k.replace("_orig_mod.", "").replace("module.", "")
        clean[k] = v
    return clean


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def human_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)
