# train.py
import argparse
import importlib.util
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from ml_collections import config_dict
from trainers import Trainer
import datetime

def _is_distributed():
    """Check if we are running under torchrun."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ

def _setup_ddp():
    """Initialize DDP process group if distributed."""
    if _is_distributed():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # Bind this rank to its GPU and pass device_id so NCCL knows the
        # rank->GPU mapping (avoids the "devices currently unknown" barrier
        # warning / potential hang on some clusters; see docs NCCL tip).
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(minutes=20),
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        # Only print setup info on master or for debugging
        if rank == 0:
            print(f"🚀 DDP Initialized: Global Rank {rank}, World Size {world_size}")
        return rank, local_rank, world_size
    return 0, 0, 1

def _cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def _update_cfg_from_dict(cfg, d, skip_sections=("logging", "system", "evaluation")):
    """Recursively update config, ignoring specific sections."""
    for k, v in d.items():
        if k in skip_sections:
            continue
        if isinstance(v, dict):
            if not hasattr(cfg, k) or getattr(cfg, k) is None:
                setattr(cfg, k, config_dict.ConfigDict())
            _update_cfg_from_dict(getattr(cfg, k), v, skip_sections=skip_sections)
        else:
            setattr(cfg, k, v)


def main():
    # 1. Setup Distributed Environment
    rank, local_rank, world_size = _setup_ddp()

    parser = argparse.ArgumentParser(description="Train a diffusion model.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.")
    args = parser.parse_args()

    # 2. Load Python Config
    spec = importlib.util.spec_from_file_location("config", args.config)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = cfg_module.get_config()

    # 3. Inject DDP info into config for Trainer
    # (We attach this to cfg so we don't need to change Trainer __init__ signature)
    cfg.system = config_dict.ConfigDict()
    cfg.system.distributed = (world_size > 1)
    cfg.system.global_rank = rank
    cfg.system.local_rank = local_rank
    cfg.system.world_size = world_size

    # 4. Resume Logic (Only Rank 0 prints, but all must know to resume)
    # Note: We rely on the Trainer to handle the actual loading logic safely.
    run_dir = Path("runs") / cfg.experiment
    saved_cfg_path = run_dir / "config.json"
    ckpt_dir = run_dir / "checkpoints"
    last_ckpt = ckpt_dir / "last.pt"

    if last_ckpt.exists() and saved_cfg_path.exists():
        with open(saved_cfg_path, "r") as f:
            saved_cfg_dict = json.load(f)
        _update_cfg_from_dict(cfg, saved_cfg_dict)
        if rank == 0:
            print(f"✓ Resuming experiment '{cfg.experiment}' using saved config.")
    elif rank == 0:
        print("🏁 No previous run with saved config found – using Python config.")

    # Store original config path
    cfg._config_path = str(Path(args.config).resolve())

    # 5. Start Training
    try:
        trainer = Trainer(cfg)
        trainer.train()
    finally:
        _cleanup_ddp()

if __name__ == "__main__":
    main()