# diffusion/discrete/samplers.py
from __future__ import annotations

from typing import Optional, List, Iterable

import torch
from tqdm import tqdm

__all__ = ["TweedieTauLeapingSampler", "EulerRateSampler"]


def _maybe_push_traj(
    buf: List[torch.Tensor], xt: torch.Tensor, keep: bool, *, to_cpu: bool
) -> None:
    if not keep:
        return
    dev = "cpu" if to_cpu else xt.device
    # int32 is safe for 32k vocab + possible mask token
    buf.append(xt.detach().to(dev).to(torch.int32).clone())


def _iter_steps(n: int, *, use_tqdm: bool, desc: str) -> Iterable[int]:
    it = range(n)
    return tqdm(it, desc=desc) if use_tqdm else it


def _clamp_prefix(
    xt: torch.Tensor,
    prefix: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
) -> None:
    """
    In-place clamp: xt[mask] = prefix[mask]
    Expects:
      xt: [B,S] long
      prefix: [B,S] long
      mask: [B,S] bool
    """
    if prefix is None or mask is None:
        return

    if prefix.device != xt.device:
        prefix = prefix.to(xt.device)
    if mask.device != xt.device:
        mask = mask.to(xt.device)

    prefix = prefix.to(torch.long)
    mask = mask.to(torch.bool)

    if prefix.shape != xt.shape:
        raise ValueError(f"prefix shape mismatch: {prefix.shape} vs xt {xt.shape}")
    if mask.shape != xt.shape:
        raise ValueError(f"mask shape mismatch: {mask.shape} vs xt {xt.shape}")

    if mask.any():
        xt[mask] = prefix[mask]

class EulerRateSampler:
    """
    Euler predictor in rate space (SEDD-style rate Euler).
    """

    def __init__(
        self,
        *,
        model,
        process,
        device,
        vocab_size: int,
        is_absorb: bool,
        mask_id: Optional[int],
        seq_len: int,
        num_steps: int,
        t_eps: float,
        log_score_clip: float = 30.0,
        prob_floor: float = 1e-12,
        disallow_mask_in_final: bool = True,
        save_trajectories: bool = False,
        trajectory_to_cpu: bool = True,
        progress_bar: bool = False,
    ):
        self.model = model
        self.process = process
        self.device = torch.device(device)

        self.vocab_size = int(vocab_size)
        self.is_absorb = bool(is_absorb)
        self.mask_id = int(mask_id) if mask_id is not None else None

        self.seq_len = int(seq_len)
        self.num_steps = int(num_steps)
        self.t_eps = float(t_eps)

        self.log_score_clip = float(log_score_clip)
        self.prob_floor = float(prob_floor)
        self.disallow_mask_in_final = bool(disallow_mask_in_final)

        self.keep_trajectory = bool(save_trajectories)
        self.trajectory_to_cpu = bool(trajectory_to_cpu)
        self.progress_bar = bool(progress_bar)

        self.trajectory: Optional[torch.Tensor] = None

        if self.seq_len <= 0:
            raise ValueError(f"EulerRateSampler: seq_len must be > 0, got {self.seq_len}")
        if self.num_steps <= 0:
            raise ValueError(f"EulerRateSampler: num_steps must be > 0, got {self.num_steps}")
        if not (0.0 < self.t_eps < 1.0):
            raise ValueError(f"EulerRateSampler: t_eps must be in (0,1), got {self.t_eps}")
        if self.is_absorb and self.mask_id is None:
            raise ValueError("EulerRateSampler: is_absorb=True but mask_id is None.")

    @torch.no_grad()
    def sample(
        self,
        *,
        num_samples: int,
        conditioning_prefix: Optional[torch.Tensor] = None,  # [B,S] long
        cond_mask: Optional[torch.Tensor] = None,            # [B,S] bool
    ) -> torch.Tensor:
        B = int(num_samples)
        S = self.seq_len
        V = self.vocab_size

        if B == 0:
            return torch.empty((0, S), device=self.device, dtype=torch.long)

        # Defensive device / dtype normalization for conditioning tensors
        if conditioning_prefix is not None:
            conditioning_prefix = conditioning_prefix.to(device=self.device, dtype=torch.long)
            if conditioning_prefix.shape != (B, S):
                raise ValueError(
                    f"conditioning_prefix shape mismatch: got {conditioning_prefix.shape}, expected {(B, S)}"
                )

        if cond_mask is not None:
            cond_mask = cond_mask.to(device=self.device, dtype=torch.bool)
            if cond_mask.shape != (B, S):
                raise ValueError(
                    f"cond_mask shape mismatch: got {cond_mask.shape}, expected {(B, S)}"
                )

        # Init
        if self.is_absorb:
            xt = torch.full((B, S), int(self.mask_id), device=self.device, dtype=torch.long)
        else:
            xt = torch.randint(0, V, (B, S), device=self.device, dtype=torch.long)

        _clamp_prefix(xt, conditioning_prefix, cond_mask)

        steps = self.num_steps
        t_eps = self.t_eps
        ts = torch.linspace(1.0, t_eps, steps + 1, device=self.device)
        dt = (1.0 - t_eps) / steps

        traj: List[torch.Tensor] = []
        _maybe_push_traj(traj, xt, self.keep_trajectory, to_cpu=self.trajectory_to_cpu)

        clip = float(self.log_score_clip)

        for i in _iter_steps(steps, use_tqdm=self.progress_bar, desc="Euler Sampler"):
            t = ts[i].expand(B)  # (B,)
            sigma_total, dsigma_dt = self.process.noise_total_and_rate(t)

            log_scores = self.model(xt, sigma_total)

            if not torch.isfinite(log_scores).all():
                raise RuntimeError(
                    f"[Euler sampler] non-finite model logits at step={i}: "
                    f"nan={torch.isnan(log_scores).any().item()} "
                    f"inf={torch.isinf(log_scores).any().item()} "
                    f"sigma_min={sigma_total.min().item():.6g} "
                    f"sigma_max={sigma_total.max().item():.6g}"
                )

            log_scores = torch.nan_to_num(
                log_scores,
                nan=0.0,
                posinf=clip,
                neginf=-clip,
            ).clamp(-clip, clip)

            scores = log_scores.float().exp()  # (B,S,V)

            scale = (dt * dsigma_dt).view(-1, 1, 1).to(dtype=scores.dtype)
            rev_rate = scale * self.process.graph.reverse_rate(xt, scores)

            rev_rate = torch.nan_to_num(rev_rate, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)

            # Make off-diagonals nonnegative, rebuild diagonal as negative row sum
            rev_rate = rev_rate.clone()
            rev_rate.scatter_(-1, xt.unsqueeze(-1), 0.0)
            rev_rate.clamp_min_(0.0)
            rev_rate.scatter_(-1, xt.unsqueeze(-1), -rev_rate.sum(dim=-1, keepdim=True))

            if not torch.isfinite(rev_rate).all():
                raise RuntimeError(
                    f"[Euler sampler] non-finite reverse rates at step={i}: "
                    f"nan={torch.isnan(rev_rate).any().item()} "
                    f"inf={torch.isinf(rev_rate).any().item()}"
                )

            xt = self.process.graph.sample_rate(xt, rev_rate)

            _clamp_prefix(xt, conditioning_prefix, cond_mask)
            _maybe_push_traj(traj, xt, self.keep_trajectory, to_cpu=self.trajectory_to_cpu)

        # Final denoise at t=t_eps
        t_last = ts[-1].expand(B)
        sigma_last, _ = self.process.noise_total_and_rate(t_last)

        log_scores = self.model(xt, sigma_last)

        if not torch.isfinite(log_scores).all():
            raise RuntimeError(
                "[Euler sampler final denoise] non-finite model logits: "
                f"nan={torch.isnan(log_scores).any().item()} "
                f"inf={torch.isinf(log_scores).any().item()}"
            )

        log_scores = torch.nan_to_num(
            log_scores,
            nan=0.0,
            posinf=clip,
            neginf=-clip,
        ).clamp(-clip, clip)

        scores = log_scores.float().exp()

        stag = self.process.staggered_score(scores, sigma_last)
        rowT = self.process.transp_transition(xt, sigma_last)

        probs = (stag * rowT).to(torch.float32)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs.clamp_min_(0.0)

        if self.is_absorb and self.disallow_mask_in_final and self.mask_id is not None:
            if 0 <= self.mask_id < probs.size(-1):
                probs[..., int(self.mask_id)] = 0.0

        row_sums = probs.sum(-1, keepdim=True)
        bad_rows = row_sums <= self.prob_floor

        if bad_rows.any():
            fallback = torch.ones_like(probs)
            if self.is_absorb and self.disallow_mask_in_final and self.mask_id is not None:
                if 0 <= self.mask_id < fallback.size(-1):
                    fallback[..., int(self.mask_id)] = 0.0
            fallback = fallback / fallback.sum(-1, keepdim=True).clamp_min(self.prob_floor)
            probs = torch.where(bad_rows, fallback, probs)
            row_sums = probs.sum(-1, keepdim=True)

        probs = probs / row_sums.clamp_min(self.prob_floor)

        if not torch.isfinite(probs).all():
            raise RuntimeError(
                "[Euler sampler final denoise] non-finite probs: "
                f"nan={torch.isnan(probs).any().item()} "
                f"inf={torch.isinf(probs).any().item()}"
            )

        xt = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, S)

        _clamp_prefix(xt, conditioning_prefix, cond_mask)
        _maybe_push_traj(traj, xt, self.keep_trajectory, to_cpu=self.trajectory_to_cpu)

        self.trajectory = torch.stack(traj, dim=0) if self.keep_trajectory else None
        return xt


class TweedieTauLeapingSampler:
    """
    Analytic Tweedie τ-leaping sampler (SEDD-style).
    """

    def __init__(
        self,
        *,
        model,
        process,
        device,
        vocab_size: int,
        is_absorb: bool,
        mask_id: Optional[int],
        seq_len: int,
        num_steps: int,
        t_eps: float,
        denoise: bool = True,
        log_score_clip: float = 30.0,
        prob_floor: float = 1e-12,
        disallow_mask_in_final: bool = True,
        save_trajectories: bool = False,
        trajectory_to_cpu: bool = True,
        progress_bar: bool = True,
    ):
        self.model = model
        self.process = process
        self.device = torch.device(device)

        self.vocab_size = int(vocab_size)
        self.is_absorb = bool(is_absorb)
        self.mask_id = int(mask_id) if mask_id is not None else None

        self.seq_len = int(seq_len)
        self.num_steps = int(num_steps)
        self.t_eps = float(t_eps)

        self.denoise = bool(denoise)
        self.log_score_clip = float(log_score_clip)
        self.prob_floor = float(prob_floor)
        self.disallow_mask_in_final = bool(disallow_mask_in_final)

        self.keep_trajectory = bool(save_trajectories)
        self.trajectory_to_cpu = bool(trajectory_to_cpu)
        self.progress_bar = bool(progress_bar)

        self.trajectory: Optional[torch.Tensor] = None

        if self.seq_len <= 0:
            raise ValueError(f"TweedieTauLeapingSampler: seq_len must be > 0, got {self.seq_len}")
        if self.num_steps <= 0:
            raise ValueError(f"TweedieTauLeapingSampler: num_steps must be > 0, got {self.num_steps}")
        if not (0.0 < self.t_eps < 1.0):
            raise ValueError(f"TweedieTauLeapingSampler: t_eps must be in (0,1), got {self.t_eps}")
        if self.is_absorb and self.mask_id is None:
            raise ValueError("TweedieTauLeapingSampler: is_absorb=True but mask_id is None.")

    @torch.no_grad()
    def sample(
        self,
        *,
        num_samples: int,
        conditioning_prefix: Optional[torch.Tensor] = None,  # [B,S] long
        cond_mask: Optional[torch.Tensor] = None,            # [B,S] bool
    ) -> torch.Tensor:
        B = int(num_samples)
        S = self.seq_len
        V = self.vocab_size

        if B == 0:
            return torch.empty((0, S), device=self.device, dtype=torch.long)

        # Defensive device / dtype normalization for conditioning tensors
        if conditioning_prefix is not None:
            conditioning_prefix = conditioning_prefix.to(device=self.device, dtype=torch.long)
            if conditioning_prefix.shape != (B, S):
                raise ValueError(
                    f"conditioning_prefix shape mismatch: got {conditioning_prefix.shape}, expected {(B, S)}"
                )

        if cond_mask is not None:
            cond_mask = cond_mask.to(device=self.device, dtype=torch.bool)
            if cond_mask.shape != (B, S):
                raise ValueError(
                    f"cond_mask shape mismatch: got {cond_mask.shape}, expected {(B, S)}"
                )

        # Init
        if self.is_absorb:
            xt = torch.full((B, S), int(self.mask_id), device=self.device, dtype=torch.long)
        else:
            xt = torch.randint(0, V, (B, S), device=self.device, dtype=torch.long)

        _clamp_prefix(xt, conditioning_prefix, cond_mask)

        steps = self.num_steps
        t_eps = self.t_eps
        ts = torch.linspace(1.0, t_eps, steps + 1, device=self.device)

        traj: List[torch.Tensor] = []
        _maybe_push_traj(traj, xt, self.keep_trajectory, to_cpu=self.trajectory_to_cpu)

        clip = float(self.log_score_clip)

        for i in _iter_steps(steps, use_tqdm=self.progress_bar, desc="Tweedie Sampler"):
            t_now, t_next = ts[i], ts[i + 1]
            sigma_now, _ = self.process.noise_total_and_rate(t_now.expand(B))
            sigma_next, _ = self.process.noise_total_and_rate(t_next.expand(B))
            d_sigma = sigma_now - sigma_next  # (B,)

            log_scores = self.model(xt, sigma_now)

            if not torch.isfinite(log_scores).all():
                raise RuntimeError(
                    f"[Tweedie sampler] non-finite model logits at step={i}: "
                    f"nan={torch.isnan(log_scores).any().item()} "
                    f"inf={torch.isinf(log_scores).any().item()} "
                    f"sigma_now_min={sigma_now.min().item():.6g} "
                    f"sigma_now_max={sigma_now.max().item():.6g}"
                )

            log_scores = torch.nan_to_num(
                log_scores,
                nan=0.0,
                posinf=clip,
                neginf=-clip,
            ).clamp(-clip, clip)

            scores = log_scores.float().exp()  # float32 for safety

            stag = self.process.staggered_score(scores, d_sigma)
            rowT = self.process.transp_transition(xt, d_sigma)

            probs = (stag * rowT).to(torch.float32)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = probs.clamp_min_(0.0)

            row_sums = probs.sum(-1, keepdim=True)
            bad_rows = row_sums <= self.prob_floor

            if bad_rows.any():
                fallback = torch.ones_like(probs)

                # If final mask is undesirable in absorbing mode, avoid it in fallback too
                if self.is_absorb and self.mask_id is not None:
                    if 0 <= self.mask_id < fallback.size(-1):
                        fallback[..., int(self.mask_id)] = 0.0

                fallback = fallback / fallback.sum(-1, keepdim=True).clamp_min(self.prob_floor)
                probs = torch.where(bad_rows, fallback, probs)
                row_sums = probs.sum(-1, keepdim=True)

            probs = probs / row_sums.clamp_min(self.prob_floor)

            if not torch.isfinite(probs).all():
                raise RuntimeError(
                    f"[Tweedie sampler] non-finite probs at step={i}: "
                    f"nan={torch.isnan(probs).any().item()} "
                    f"inf={torch.isinf(probs).any().item()}"
                )

            xt = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, S)

            _clamp_prefix(xt, conditioning_prefix, cond_mask)
            _maybe_push_traj(traj, xt, self.keep_trajectory, to_cpu=self.trajectory_to_cpu)

        if self.denoise:
            sigma_eps, _ = self.process.noise_total_and_rate(ts[-1].expand(B))
            log_scores = self.model(xt, sigma_eps)

            if not torch.isfinite(log_scores).all():
                raise RuntimeError(
                    "[Tweedie sampler final denoise] non-finite model logits: "
                    f"nan={torch.isnan(log_scores).any().item()} "
                    f"inf={torch.isinf(log_scores).any().item()}"
                )

            log_scores = torch.nan_to_num(
                log_scores,
                nan=0.0,
                posinf=clip,
                neginf=-clip,
            ).clamp(-clip, clip)

            scores = log_scores.float().exp()

            stag = self.process.staggered_score(scores, sigma_eps)
            rowT = self.process.transp_transition(xt, sigma_eps)

            probs = (stag * rowT).to(torch.float32)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = probs.clamp_min_(0.0)

            if self.is_absorb and self.disallow_mask_in_final and self.mask_id is not None:
                if 0 <= self.mask_id < probs.size(-1):
                    probs[..., int(self.mask_id)] = 0.0

            row_sums = probs.sum(-1, keepdim=True)
            bad_rows = row_sums <= self.prob_floor

            if bad_rows.any():
                fallback = torch.ones_like(probs)

                if self.is_absorb and self.disallow_mask_in_final and self.mask_id is not None:
                    if 0 <= self.mask_id < fallback.size(-1):
                        fallback[..., int(self.mask_id)] = 0.0

                fallback = fallback / fallback.sum(-1, keepdim=True).clamp_min(self.prob_floor)
                probs = torch.where(bad_rows, fallback, probs)
                row_sums = probs.sum(-1, keepdim=True)

            probs = probs / row_sums.clamp_min(self.prob_floor)

            if not torch.isfinite(probs).all():
                raise RuntimeError(
                    "[Tweedie sampler final denoise] non-finite probs: "
                    f"nan={torch.isnan(probs).any().item()} "
                    f"inf={torch.isinf(probs).any().item()}"
                )

            xt = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, S)

            _clamp_prefix(xt, conditioning_prefix, cond_mask)
            _maybe_push_traj(traj, xt, self.keep_trajectory, to_cpu=self.trajectory_to_cpu)

        self.trajectory = torch.stack(traj, dim=0) if self.keep_trajectory else None
        return xt
