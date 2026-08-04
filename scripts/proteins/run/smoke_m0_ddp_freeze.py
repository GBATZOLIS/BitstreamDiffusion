"""DDP smoke for the warm-start curriculum: trunk freeze/unfreeze under DDP.

The plain M0 DDP smoke uses freeze_trunk_steps=0, so it never exercises the
combination that is fragile under DistributedDataParallel: freezing the trunk
(requires_grad=False) AFTER the DDP wrap makes those params stop producing
gradients mid-run, which errors with the default reducer. This driver builds a
matching 5-bit sequence source checkpoint, warm-starts an 18-bit model from it,
freezes the trunk for a few steps, and trains PAST the unfreeze boundary under
DDP to prove the reducer handles it (find_unused_parameters is enabled when a
freeze window is configured).

    torchrun --nproc_per_node=4 scripts/proteins/run/smoke_m0_ddp_freeze.py
"""

from __future__ import annotations

import datetime
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from ml_collections import config_dict

from scripts.proteins.run.smoke_m0_trainer import _tiny_cfg
from scripts.proteins.setup.build_m0_fixture import build_fixture


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size > 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if ddp:
        dist.init_process_group(
            backend="nccl" if device == "cuda" else "gloo",
            timeout=datetime.timedelta(minutes=10),
        )
        if device == "cuda":
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
    is_master = rank == 0

    exp = "proteins/_smoke_m0_ddp_freeze"
    run_dir = Path("runs") / exp
    fixture_dir = Path("datasets/_m0_ddp_freeze_fx")
    src_ckpt = run_dir / "seq_src.pt"

    if is_master:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        build_fixture(fixture_dir, seed=2)
        # Build a matching 5-bit sequence source checkpoint (same trunk dims).
        from models.sdt import SequenceVDTContinuousModel

        base = _tiny_cfg(str(fixture_dir), exp, device="cpu", total_steps=1)
        src = config_dict.ConfigDict(base.to_dict())
        src.model.multimodal = False
        src.model.patch_size = 5
        src.data.representation = "binary"
        src.data.vocab_size = 2
        m5 = SequenceVDTContinuousModel(src)
        sd = m5.state_dict()
        torch.save({"model": sd, "ema": {"decay": 0.9, "shadow": sd}}, src_ckpt)
    if ddp:
        dist.barrier()

    cfg = _tiny_cfg(str(fixture_dir), exp, device=device, total_steps=12)
    cfg.system = config_dict.ConfigDict()
    cfg.system.distributed = ddp
    cfg.system.global_rank = rank
    cfg.system.local_rank = local_rank
    cfg.system.world_size = world_size
    cfg.train.steps_per_epoch = 12
    cfg.model.warm_start_seq_checkpoint = str(src_ckpt)
    cfg.model.warm_start_freeze_trunk_steps = 4  # freeze under DDP, then unfreeze
    cfg.model.warm_start_trunk_lr_mult = 0.1

    from trainers import Trainer

    t = Trainer(cfg)
    assert t._trunk_frozen, "trunk should start frozen"
    t.train()  # must cross the unfreeze boundary without a DDP reducer error

    ok = (not t._trunk_frozen) and t.global_step >= 12
    comps = t._last_mm_components or {}
    finite = comps and all(
        torch.isfinite(torch.tensor(float(v))) for v in comps.values()
    )
    if is_master:
        print(
            f"[ddp-freeze] world_size={world_size} unfroze={not t._trunk_frozen} "
            f"global_step={t.global_step} finite_loss={bool(finite)} comps={comps}"
        )
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(fixture_dir, ignore_errors=True)
        print(
            "\n✅ DDP warm-start freeze/unfreeze smoke PASSED"
            if (ok and finite)
            else "\n❌ FAILED"
        )
    if ddp:
        dist.destroy_process_group()
    return 0 if (ok and finite) else 1


if __name__ == "__main__":
    raise SystemExit(main())
