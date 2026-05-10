# callbacks/sigma_grad_norm.py
from __future__ import annotations

import contextlib
import math
from typing import Any

import torch
from torch.cuda.amp import autocast

from .base import Callback, maybe_disable_dynamo


@maybe_disable_dynamo
def run_sigma_grad_norm(trainer: Any, epoch: int, batch_size: int):
    dev = torch.device(trainer.cfg.device)
    trainer.ema.apply(trainer.model)
    trainer.model.eval()

    try:
        raw = next(iter(trainer.val_loader))
        x0 = (raw[0] if isinstance(raw, (list, tuple)) else raw)[:128].to(dev).float()
        x0 = x0.view(x0.size(0), -1)

        s_min = max(trainer.cfg.diffusion.continuous.sigma_min, 0.01)
        s_max = trainer.cfg.diffusion.continuous.sigma_max
        sigmas = torch.logspace(math.log10(s_max), math.log10(s_min), steps=8, device=dev)

        use_amp = bool(getattr(trainer.cfg.train, "use_fp16", False))
        amp_dtype = getattr(trainer, "amp_dtype", torch.float16)

        model = trainer.model
        is_ddp = getattr(trainer, "is_distributed", False) and hasattr(model, "no_sync")

        grad_dict = {}

        for σ in sigmas:
            ctx = model.no_sync() if is_ddp else contextlib.nullcontext()
            with ctx:
                model.zero_grad(set_to_none=True)

                B = min(batch_size, x0.size(0))
                x0b = x0[:B]
                xt = x0b + σ * torch.randn_like(x0b)
                σb = σ.expand(B)

                with autocast(enabled=use_amp, dtype=amp_dtype):
                    logits = model(xt, σb)
                    loss = trainer.loss_fn(logits, x0b, σb, trainer.cfg)

                loss.backward()

            total = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total += p.grad.detach().pow(2).sum()
            grad = total.sqrt().item()
            key = f"{σ.item():.2g}"
            grad_dict[key] = grad

            trainer.writer.add_scalar(f"grad_diag/norm_per_sigma/{key}", grad, epoch)

        trainer._log_wandb({f"grad_diag/norm_per_sigma/{k}": v for k, v in grad_dict.items()})
    finally:
        trainer.ema.restore(trainer.model)


class SigmaGradNormCallback(Callback):
    def __init__(self, every_k_epochs: int = 20, batch_size: int = 16):
        self.every_k = every_k_epochs
        self.B = batch_size

    def on_epoch_end(self, trainer: Any, epoch: int):
        if (epoch + 1) % self.every_k:
            return
        print(f"\n— SigmaGradNormCallback (Epoch {epoch+1})")
        run_sigma_grad_norm(trainer, epoch, self.B)
