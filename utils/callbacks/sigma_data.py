# callbacks/sigma_data.py
from __future__ import annotations

import json
from pathlib import Path
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
            if isinstance(batch, dict):
                _xb = batch["x0"]
            elif isinstance(batch, (list, tuple)):
                _xb = batch[0]
            else:
                _xb = batch
            x = _xb.to(trainer.device).float()
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
        # Persist so EVAL uses the trained value (configs hardcode 0.5, which does
        # not match). evaluation/tasks/_task_common.resolve_sigma_data reads this.
        try:
            run_dir = getattr(trainer, "run_dir", None)
            if run_dir is not None:
                p = Path(run_dir) / "sigma_data.json"
                p.write_text(json.dumps({"sigma_data": sigma}))
                print(f"✓ σ_data persisted -> {p}")
        except Exception as e:  # never let persistence break training
            print(f"[SigmaDataEstimator] WARN: could not persist sigma_data ({e})")
        print(f"✓ σ_data estimated: {sigma:.4f}")
        self.done = True
