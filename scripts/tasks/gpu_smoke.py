"""CSD3 A100 GPU smoke test for CoBit TinyGSM.

Run under torchrun (nproc=2). Validates the env-specific risks before
committing the 4-GPU training chain:
  * CUDA available, device count, compute capability, bf16
  * the EXACT train.py NCCL path: init_process_group(nccl) with NO device_id,
    then dist.barrier() (the documented Isambard-vs-CSD3 hang). Short timeout
    so a hang fails fast instead of blocking 20 min.
  * create_model(cfg) builds (param count ~133.8M) and moves to GPU
  * bf16 scaled_dot_product_attention
  * torch.compile of a small module (inductor works on these nodes)
"""
import os
import datetime
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def load_cfg():
    spec = importlib.util.spec_from_file_location("cfg", "configs/tasks/tinygsm_bits.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def main():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    dev = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[cuda] torch {torch.__version__} avail={torch.cuda.is_available()} "
              f"ndev={torch.cuda.device_count()} name={torch.cuda.get_device_name(local_rank)} "
              f"cap={torch.cuda.get_device_capability(local_rank)} "
              f"bf16={torch.cuda.is_bf16_supported()}", flush=True)

    # --- DDP exactly as train.py does it (NO device_id), short timeout --------
    if world > 1:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=2))
        torch.cuda.set_device(local_rank)
        if rank == 0:
            print("[dist] init_process_group OK; calling barrier (this is the risky step)...", flush=True)
        dist.barrier()
        if rank == 0:
            print("[dist] barrier OK -> no device_id patch needed", flush=True)
        t = torch.ones(4, device=dev) * (rank + 1)
        dist.all_reduce(t)
        if rank == 0:
            print(f"[dist] all_reduce sum (expect {sum(range(1, world+1))}) = {t[0].item()}", flush=True)
    else:
        torch.cuda.set_device(local_rank)

    # --- bf16 SDPA ------------------------------------------------------------
    q = torch.randn(2, 12, 512, 64, device=dev, dtype=torch.bfloat16)
    y = F.scaled_dot_product_attention(q, q, q)
    if rank == 0:
        print(f"[sdpa] bf16 ok out={tuple(y.shape)}", flush=True)

    # --- build the real model -------------------------------------------------
    cfg = load_cfg()
    from models import create_model
    model = create_model(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"[model] create_model OK params={n_params/1e6:.1f}M", flush=True)

    # --- torch.compile a small module ----------------------------------------
    m = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256)).to(dev)
    mc = torch.compile(m)
    x = torch.randn(8, 256, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = mc(x)
        loss = out.float().pow(2).mean()
    loss.backward()
    if rank == 0:
        print(f"[compile] torch.compile fwd+bwd ok loss={loss.item():.4f}", flush=True)
        print("[SMOKE] ALL CHECKS PASSED", flush=True)

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
