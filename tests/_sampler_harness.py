"""Shared CPU test harness for the continuous samplers.

Builds a tiny *deterministic* stand-in denoiser and a real (but CPU-forced,
compile/flash-attn disabled) Sudoku config so sampler control flow — prompt
clamping, self-conditioning carry, the PF-ODE/EM integrators, and the FKC
particle loop — can be exercised without a checkpoint or GPU.

The stand-in is NOT trained; it only has to be a smooth, deterministic function
of (x_t, sigma, x0_hat) so that trajectories are reproducible under a fixed
seed and genuinely sigma/state dependent (needed by the FKC ancestry and
Gaussian-moment gates). Correctness gates that assert *equality between two
samplers* hold for any deterministic denoiser.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from evaluation.tasks._task_common import load_config, configure_stochastic


class TinyBinaryDenoiser(nn.Module):
    """Deterministic binary-mode denoiser stand-in returning a raw logit [B, S].

    The public postprocessing (matched-filter residual) is applied downstream by
    `_model_logits_continuous`, so this returns only the learned component.
    """

    def __init__(self, seq_len: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.register_buffer("w", torch.randn(seq_len, generator=g) * 0.7)
        self.register_buffer("b", torch.randn(seq_len, generator=g) * 0.1)
        # A real parameter so device inference / .to() behave like a normal module.
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_t, sigma, x0_hat=None):
        if isinstance(sigma, torch.Tensor):
            s = sigma.reshape(-1)[0].to(x_t.dtype)
        else:
            s = torch.tensor(float(sigma), dtype=x_t.dtype, device=x_t.device)
        target = torch.sigmoid(self.w).unsqueeze(0)                 # [1, S] in (0,1)
        # Pull x toward a fixed per-bit target, damped at high sigma; deterministic.
        drive = (target - (x_t - 0.5)) / (1.0 + s)
        sc = 0.0 if x0_hat is None else (x0_hat - 0.5) * 0.1
        return (drive + self.b.unsqueeze(0) + sc) * self.gain


def make_cpu_cfg(*, self_condition: bool = True, num_steps: int = 12):
    """Real Sudoku config, forced onto CPU with compile/flash-attn off."""
    cfg = load_config("configs/tasks/sudoku_bits.py")
    cfg.device = "cpu"
    cfg.model.use_flash_attn = False
    cfg.model.self_condition = bool(self_condition)
    cfg.train.use_compile = False
    cfg.evaluation.num_sampling_steps = int(num_steps)
    configure_stochastic(cfg, mode="deterministic", gamma=0.0, num_steps=int(num_steps))
    return cfg


def make_conditioning(B: int, S: int, n_prompt: int, *, seed: int = 0):
    """Return (prefix_full [B,S] in {0,1}, prefix_mask [B,S] bool, True=prompt)."""
    g = torch.Generator().manual_seed(int(seed))
    prefix_full = (torch.rand(B, S, generator=g) > 0.5).float()
    prefix_mask = torch.zeros(B, S, dtype=torch.bool)
    prefix_mask[:, :n_prompt] = True
    return prefix_full, prefix_mask
