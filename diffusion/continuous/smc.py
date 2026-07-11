"""Sequential Monte Carlo utilities for the Feynman-Kac corrector sampler.

These back `FeynmanKacEulerMaruyamaSampler`, which realises the tempered target
pi_beta(y | c) ~ p_theta(y | c)^beta by running K weighted particles per prompt
and resampling toward the FKC potential. The layout convention throughout is

    per-particle tensors : [B, K, ...]   (B = prompt groups, K = particles)
    log-weights          : [B, K] float64

Resampling is *independent within each prompt group* (along the K axis) -- a
group is one conditioning prompt, and particles must never migrate across
prompts. Weights and ESS are computed in float64 because the FKC log-weight
increment 1/2 beta(beta-1) Delta(sigma^2) ||s||^2_free is extensive in the number
of free bits (hundreds to thousands) and overflows/loses precision in float32.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class SMCConfig:
    """Static configuration for one FKC particle run."""
    num_particles: int = 8
    resampling_method: str = "systematic"          # only 'systematic' implemented
    resampling_policy: str = "ess"                 # 'ess' | 'every_step_active' | 'never'
    ess_threshold_fraction: float = 0.5            # resample when ESS < frac * K
    final_resample: bool = True
    weight_dtype: torch.dtype = torch.float64


@dataclass
class FKCOutput:
    """Result of a FeynmanKacEulerMaruyamaSampler.sample_particles run.

    All per-particle tensors are [B, K, S]. `bits`/`probs`/`x` are the final
    target population (after the mandatory final resample when enabled);
    `pre_resample_*` expose the proposal population *before* that final resample
    so proposal coverage can be distinguished from post-concentration.
    """
    bits: torch.Tensor                 # [B, K, S] long 0/1
    probs: torch.Tensor                # [B, K, S] float
    x: torch.Tensor                    # [B, K, S] final continuous state
    pre_resample_bits: torch.Tensor    # [B, K, S] long 0/1 (before final resample)
    log_weights_final: torch.Tensor    # [B, K] float64 (before final resample)
    ancestors: torch.Tensor            # [B, K] int64 (origin particle per slot)
    diagnostics: "SMCDiagnostics"


@dataclass
class SMCDiagnostics:
    """Per-step SMC telemetry (each list is appended once per active step)."""
    sigmas: list = field(default_factory=list)
    ess: list = field(default_factory=list)                    # [B] per step
    log_weight_mean: list = field(default_factory=list)
    log_weight_std: list = field(default_factory=list)
    max_weight: list = field(default_factory=list)             # max normalized weight
    unique_ancestors: list = field(default_factory=list)       # [B] per resample
    resampled: list = field(default_factory=list)              # bool per step
    potential_per_free_bit: list = field(default_factory=list)

    def as_summary(self) -> dict:
        def _stack(xs):
            return torch.stack(xs) if xs else torch.empty(0)
        ess = _stack(self.ess)
        return {
            "num_steps_logged": len(self.sigmas),
            "num_resample_events": int(sum(1 for r in self.resampled if bool(r))),
            "min_ess": (float(ess.min()) if ess.numel() else None),
            "median_ess": (float(ess.median()) if ess.numel() else None),
            "final_unique_ancestors": (
                [int(v) for v in self.unique_ancestors[-1].tolist()]
                if self.unique_ancestors else None
            ),
        }


def effective_sample_size(log_weights: torch.Tensor) -> torch.Tensor:
    """ESS = (sum_k w_k)^2 / sum_k w_k^2, per group. log_weights [B, K] -> [B].

    Computed stably via logsumexp; result lies in [1, K].
    """
    lw = log_weights.to(torch.float64)
    lse1 = torch.logsumexp(lw, dim=1)
    lse2 = torch.logsumexp(2.0 * lw, dim=1)
    return torch.exp(2.0 * lse1 - lse2)


def normalized_weights(log_weights: torch.Tensor) -> torch.Tensor:
    """Softmax of log-weights along the particle axis. [B, K] -> [B, K] float64."""
    return torch.softmax(log_weights.to(torch.float64), dim=1)


def systematic_resample_indices(
    log_weights: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
    u0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Systematic (low-variance) resampling indices, independent per group.

    Draws one uniform u0 ~ U[0, 1/K) per group and reads the K equally spaced
    positions u0 + k/K against the weight CDF. Returns ancestor indices [B, K]
    (int64). Pass `u0` [B, 1] to make the draw deterministic (Gate 7).
    """
    w = normalized_weights(log_weights)                 # [B, K] float64
    B, K = w.shape
    device = w.device
    cdf = torch.cumsum(w, dim=1)
    cdf[:, -1] = 1.0                                    # guard tiny fp drift
    if u0 is None:
        u0 = torch.rand(B, 1, generator=generator, device=device, dtype=torch.float64) / K
    else:
        u0 = u0.to(device=device, dtype=torch.float64).reshape(B, 1)
    positions = u0 + torch.arange(K, device=device, dtype=torch.float64).unsqueeze(0) / K
    idx = torch.searchsorted(cdf.contiguous(), positions.contiguous())
    return idx.clamp_(0, K - 1).to(torch.int64)


def gather_particles(t, indices: torch.Tensor):
    """Gather along the particle axis (dim=1) using ancestor `indices` [B, K].

    Handles None (returns None) and tuples/lists (gathers each element) so the
    same call resamples x, the self-conditioning tensor, the score, and any
    future CFG self-conditioning tuple with identical indices.
    """
    if t is None:
        return None
    if isinstance(t, (tuple, list)):
        return type(t)(gather_particles(e, indices) for e in t)
    # t: [B, K, ...]; expand indices to match trailing dims.
    idx = indices
    while idx.dim() < t.dim():
        idx = idx.unsqueeze(-1)
    idx = idx.expand(indices.shape[0], indices.shape[1], *t.shape[2:])
    return torch.gather(t, 1, idx)


def unique_ancestor_count(indices: torch.Tensor) -> torch.Tensor:
    """Number of distinct ancestors selected per group. indices [B, K] -> [B]."""
    B, K = indices.shape
    counts = torch.zeros(B, dtype=torch.int64, device=indices.device)
    for b in range(B):
        counts[b] = int(torch.unique(indices[b]).numel())
    return counts
