# callbacks/sigma_data.py
from __future__ import annotations

from typing import Any

import torch

from .base import Callback


class SigmaDataEstimator(Callback):
    def __init__(self, num_batches: int = 10):
        self.num_batches = num_batches
        self.done = False

    @torch.no_grad()
    def on_train_begin(self, trainer: Any):
        if self.done:
            return
        print(f"— SigmaDataEstimator: analysing {self.num_batches} batches …")
        n_pix, mean, m2 = 0, 0.0, 0.0

        for k, batch in enumerate(trainer.train_loader):
            if k >= self.num_batches:
                break
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(trainer.device).float()
            if x.max() > 1:
                x = x / x.max()

            flat = x.view(-1)
            n_new = flat.numel()
            n_tot = n_pix + n_new
            delta = flat.mean() - mean
            mean += delta * n_new / max(n_tot, 1)
            m2 += flat.var(unbiased=False) * n_new + (delta**2) * n_pix * n_new / max(n_tot, 1)
            n_pix = n_tot

        var = m2 / max(n_pix, 1)
        sigma = float(var.sqrt().item())
        trainer.cfg.diffusion.continuous.sigma_data = sigma
        trainer.writer.add_scalar("sigma_data/estimate", sigma, 0)
        trainer._log_wandb({"sigma_data/estimate": sigma})
        print(f"✓ σ_data estimated: {sigma:.4f}")
        self.done = True
