from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import math

# ────────────────────────────── LR schedulers ─────────────────────────────
class WarmUpCosineDecayScheduler:
    """Linear warm-up → cosine decay to `min_lr`. Call `.step()` each batch."""

    def __init__(self, optimizer: optim.Optimizer, *, warmup_steps: int, total_steps: int,
                 base_lr: float, min_lr: float = 1e-6, global_step: int = 0):
        self.opt = optimizer
        self.warm = warmup_steps
        self.tot = total_steps
        self.base = base_lr
        self.min = min_lr
        self.step_idx = global_step

    def step(self):
        self.step_idx += 1
        if self.step_idx < self.warm:
            lr = self.base * self.step_idx / max(1, self.warm)
        else:
            prog = (self.step_idx - self.warm) / max(1, self.tot - self.warm)
            prog = min(max(prog, 0.0), 1.0)
            lr = self.min + (self.base - self.min) * 0.5 * (1 + math.cos(math.pi * prog))
        for g in self.opt.param_groups:
            g["lr"] = lr

    # state helpers for checkpointing -------------------------------------
    def state_dict(self):
        return {"step": self.step_idx}

    def load_state_dict(self, d):
        self.step_idx = d.get("step", 0)


class WarmUpScheduler:
    """Linear warm-up then constant LR (backwards-compatible)."""

    def __init__(self, optimizer: optim.Optimizer, *, warmup_steps: int, base_lr: float, global_step: int = 0):
        self.opt = optimizer
        self.warm = warmup_steps
        self.base = base_lr
        self.step_idx = global_step

    def step(self):
        self.step_idx += 1
        lr = self.base if self.step_idx >= self.warm else self.base * self.step_idx / max(1, self.warm)
        for g in self.opt.param_groups:
            g["lr"] = lr

    def state_dict(self):
        return {"step": self.step_idx}

    def load_state_dict(self, d):
        self.step_idx = d.get("step", 0)


# ───────────────────────── optimiser factory ──────────────────────────────

def get_optimizer_and_scheduler(model: nn.Module, cfg, global_step: int = 0) -> Tuple[optim.Optimizer, object]:
    """Return `(optimizer, scheduler)` based on `cfg.optim` fields."""
    o_cfg = cfg.optim
    lr = o_cfg.get("lr", 1e-4)
    wd = o_cfg.get("weight_decay", 1e-2)

    opt_name = o_cfg.get("optimizer", "adamw").lower()

    if opt_name == "adamw":
        # Defaults match previous hardcoded values: betas=(0.9, 0.95), eps=1e-8
        beta1 = o_cfg.get("beta1", 0.9)
        beta2 = o_cfg.get("beta2", 0.95)
        eps = o_cfg.get("eps", 1e-8)
        opt = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=wd,
        )
    elif opt_name == "adam":
        # Defaults match previous hardcoded values: betas=(0.9, 0.99), eps=1e-8
        beta1 = o_cfg.get("beta1", 0.9)
        beta2 = o_cfg.get("beta2", 0.99)
        eps = o_cfg.get("eps", 1e-8)
        opt = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=wd,
        )
    else:
        raise ValueError(f"Optimizer '{opt_name}' not supported.")

    sched_type = o_cfg.get("scheduler", "cosine_decay").lower()
    if sched_type == "cosine_decay":
        total_steps = o_cfg.get("total_steps", 500_000)
        warmup = o_cfg.get("warmup", 1000)
        sched = WarmUpCosineDecayScheduler(
            opt,
            warmup_steps=warmup,
            total_steps=total_steps,
            base_lr=lr,
            global_step=global_step,
        )
    elif sched_type in {"constant", "warmup_only", "none"}:
        warmup = o_cfg.get("warmup", 1000)
        sched = WarmUpScheduler(
            opt,
            warmup_steps=warmup,
            base_lr=lr,
            global_step=global_step,
        )
    else:
        raise ValueError(f"Scheduler '{sched_type}' not supported.")

    return opt, sched
