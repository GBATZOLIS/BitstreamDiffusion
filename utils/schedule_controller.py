#utils/schedule_controller.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import torch

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None


# -------------------------------------------------------------------------
# Gabriel regularized entropy gate:
#   g(σ) = σ^n / (σ^n + c^n) = x/(1+x) where x=(σ/c)^n
# -------------------------------------------------------------------------
def _generalized_regularizer(sigmas: torch.Tensor, c: float, n: float) -> torch.Tensor:
    c = float(max(c, 1e-12))
    n = float(n)
    x = (sigmas.clamp_min(1e-12) / c).pow(n)
    return x / (1.0 + x)


class EntropyScheduleController:
    """
    Entropy-rate based sigma scheduling for continuous diffusion.

    DESIGN GOALS (your requirements):
      - FIFO buffer lives in Trainer (ring buffer on CPU).
      - Sample log-uniform sigma until:
            (i) buffer has enough samples and
            (ii) warmup is done
      - During transition, mix base and entropy with gamma ramp.
      - After warmup + transition -> fully entropic (gamma=1).

    ALIGNMENT WITH GABRIEL:
      - Regularized entropy mode:
            q(σ) ∝ g(σ; c,n) * r(σ)^power
        where:
            g(σ) = σ^n / (σ^n + c^n)
            r(σ) is your entropy-rate proxy per sigma-bin
            power = 1.0   (rate) or 0.5 (sqrt-rate)

    DDP SAFETY:
      - Rank0 computes schedule.
      - Other ranks receive schedule via broadcast.
      - A broadcasted "updated_flag" prevents deadlocks.
    """

    def __init__(self, trainer, proc):
        self.trainer = trainer
        self.proc = proc

        # Sanity: trainer must have FIFO state
        assert hasattr(trainer, "_entropy_sig_buf"), "Trainer must allocate _entropy_sig_buf in __init__"
        assert hasattr(trainer, "_entropy_metric_buf"), "Trainer must allocate _entropy_metric_buf in __init__"
        assert hasattr(trainer, "_entropy_buf_ptr"), "Trainer must define _entropy_buf_ptr in __init__"
        assert hasattr(trainer, "_entropy_buf_len"), "Trainer must define _entropy_buf_len in __init__"

        # Schedule tensors (live on GPU device)
        self.trainer._entropy_ready = bool(getattr(self.trainer, "_entropy_ready", False))
        self.trainer._entropy_pdf = getattr(self.trainer, "_entropy_pdf", None)
        self.trainer._entropy_cdf = getattr(self.trainer, "_entropy_cdf", None)
        self.trainer._entropy_sigmas = getattr(self.trainer, "_entropy_sigmas", None)  # midpoints
        self.trainer._entropy_edges = getattr(self.trainer, "_entropy_edges", None)   # edges

        # Diagnostics
        self.trainer._entropy_ln_mu = getattr(self.trainer, "_entropy_ln_mu", None)
        self.trainer._entropy_ln_std = getattr(self.trainer, "_entropy_ln_std", None)

    # ──────────────────────────────────────────────────────────────────────
    # Disk IO (optional but useful)
    # ──────────────────────────────────────────────────────────────────────
    def entropy_paths(self) -> Tuple[Path, Path, Path, Path]:
        run_dir = Path(self.trainer.run_dir)
        return (
            run_dir / "entropy_pdf.pt",
            run_dir / "entropy_cdf.pt",
            run_dir / "entropy_sigmas.pt",
            run_dir / "entropy_edges.pt",
        )

    def save_entropy_tables(
        self,
        pdf: torch.Tensor,
        cdf: torch.Tensor,
        sigmas: torch.Tensor,
        edges: torch.Tensor,
    ) -> None:
        if not bool(getattr(self.trainer, "is_master", False)):
            return
        pdf_p, cdf_p, sig_p, edg_p = self.entropy_paths()
        torch.save(pdf.detach().cpu(), pdf_p)
        torch.save(cdf.detach().cpu(), cdf_p)
        torch.save(sigmas.detach().cpu(), sig_p)
        torch.save(edges.detach().cpu(), edg_p)

    def load_entropy_tables_if_any(self) -> None:
        pdf_p, cdf_p, sig_p, edg_p = self.entropy_paths()

        if pdf_p.exists() and cdf_p.exists() and sig_p.exists():
            dev = self.trainer.device
            self.trainer._entropy_pdf = torch.load(pdf_p, map_location=dev)
            self.trainer._entropy_cdf = torch.load(cdf_p, map_location=dev)
            self.trainer._entropy_sigmas = torch.load(sig_p, map_location=dev)
            self.trainer._entropy_edges = torch.load(edg_p, map_location=dev) if edg_p.exists() else None
            self.trainer._entropy_ready = True
            self.fit_lognormal_to_entropy_profile()

            if bool(getattr(self.trainer, "is_master", False)):
                K = int(self.trainer._entropy_sigmas.numel())
                print(f"✓ Entropy schedule restored from disk ({K} bins).")

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────
    def _ddp_is_on(self) -> bool:
        return (
            dist is not None
            and dist.is_available()
            and dist.is_initialized()
            and int(getattr(self.trainer, "world_size", 1)) > 1
        )

    def _enough_fifo_samples(self) -> bool:
        """
        Require enough sigma/metric pairs to make a stable histogram.
        Default: at least 10 samples per bin.
        """
        buf_len = int(getattr(self.trainer, "_entropy_buf_len", 0))
        num_bins = int(getattr(self.trainer, "entropy_num_bins", 256))
        min_per_bin = int(getattr(self.trainer.cfg.train, "entropy_min_per_bin", 10))
        return buf_len >= (num_bins * min_per_bin)

    def _get_edges_on_device(self) -> torch.Tensor:
        """
        Prefer stored edges, otherwise reconstruct edges from midpoints.
        """
        dev = self.trainer.device
        edges = getattr(self.trainer, "_entropy_edges", None)

        if isinstance(edges, torch.Tensor) and edges.numel() >= 2:
            return edges.to(dev).float().clamp(min=1e-12)

        # reconstruct from midpoints if needed (backward compatibility)
        mids = getattr(self.trainer, "_entropy_sigmas", None)
        if not isinstance(mids, torch.Tensor) or mids.numel() < 1:
            return torch.tensor(
                [float(self.proc.sigma_min), float(self.proc.sigma_max)],
                device=dev,
                dtype=torch.float32,
            )

        mids = mids.to(dev).float().clamp(min=1e-12)
        logm = mids.log()
        K = int(logm.numel())

        if K == 1:
            edges_log = torch.tensor(
                [math.log(float(self.proc.sigma_min)), math.log(float(self.proc.sigma_max))],
                device=dev,
                dtype=torch.float32,
            )
            return edges_log.exp()

        internal = 0.5 * (logm[:-1] + logm[1:])  # [K-1]
        edges_log = torch.empty((K + 1,), device=dev, dtype=logm.dtype)
        edges_log[0] = math.log(float(self.proc.sigma_min))
        edges_log[-1] = math.log(float(self.proc.sigma_max))
        edges_log[1:-1] = internal
        edges = edges_log.exp().clamp(min=1e-12)

        self.trainer._entropy_edges = edges
        return edges

    # ──────────────────────────────────────────────────────────────────────
    # Gamma schedule (warmup -> transition -> full entropic)
    # ──────────────────────────────────────────────────────────────────────
    def entropy_gamma(self) -> float:
        """
        gamma(step) = 0 for step<warmup
                      linear 0->gamma_max over transition
                      gamma_max afterwards

        If gamma_max=1.0 => becomes fully entropic after warmup+transition.
        """

        if bool(getattr(self.trainer, "entropy_offline_enabled", False)):
            return 1.0 if bool(getattr(self.trainer, "_entropy_ready", False)) else 0.0

        if not bool(getattr(self.trainer, "entropy_use_for_sampling", False)):
            return 0.0

        step = int(getattr(self.trainer, "global_step", 0))
        warm = int(getattr(self.trainer, "entropy_warmup_steps", 0))
        trans = int(getattr(self.trainer, "entropy_transition_steps", 1))
        gamma_max = float(getattr(self.trainer, "entropy_gamma_max", 0.0))

        if step < warm:
            return 0.0

        frac = min(float(step - warm) / max(1, trans), 1.0)
        gamma = gamma_max * frac
        return max(0.0, min(gamma, 1.0))

    # ──────────────────────────────────────────────────────────────────────
    # Sampling from schedule
    # ──────────────────────────────────────────────────────────────────────
    def sample_entropy_sigma(self, bsz: int) -> torch.Tensor:
        """
        Sample σ from learned entropy schedule:
          1) pick a bin via CDF
          2) sample log-uniform inside [edge_i, edge_{i+1}]
        """
        if not bool(getattr(self.trainer, "_entropy_ready", False)):
            # if schedule not ready, fallback
            return self.proc.sample_sigma(bsz, strategy="log-uniform").to(self.trainer.device)

        dev = self.trainer.device

        cdf = self.trainer._entropy_cdf.to(dev).float().clamp(min=0.0)
        cdf[-1] = 1.0  # numerical safety

        u = torch.rand(bsz, device=dev)
        idx = torch.searchsorted(cdf, u, right=False)
        idx = torch.clamp(idx, 0, cdf.numel() - 1)

        edges = self._get_edges_on_device()  # [K+1]
        lo = edges[idx].clamp(min=float(self.proc.sigma_min))
        hi = edges[idx + 1].clamp(max=float(self.proc.sigma_max))

        # sample log-uniform within the bin
        u2 = torch.rand(bsz, device=dev)
        log_sigma = lo.log() + u2 * (hi.log() - lo.log())
        sigma = log_sigma.exp()

        return sigma.clamp(min=float(self.proc.sigma_min), max=float(self.proc.sigma_max))

    def draw_sigma(self, bsz: int) -> torch.Tensor:
        """
        Training-time σ sampling.

        Behavior:
          - Base strategy from cfg.train.sigma_sampling_strategy ("log-uniform"/"log-normal").
          - If entropy_use_for_sampling:
              * before warmup: base only
              * warmup->transition: mixture base/entropy (gamma ramps)
              * after: fully entropic if gamma_max=1.0
          - Entropy can only be used when _entropy_ready=True.
        """
        if self.trainer.cfg.framework != "continuous_score":
            raise RuntimeError("draw_sigma only valid for continuous_score.")

        if self.trainer.is_master and self.trainer.global_step < 5:
            print("draw_sigma source:", "offline-entropic" if self.trainer.entropy_offline_enabled and self.trainer._entropy_ready else "base")

        strat = str(getattr(self.trainer.cfg.train, "sigma_sampling_strategy", "log-uniform")).lower()

        # Base sampling
        if strat == "log-normal":
            p_mean = getattr(self.trainer.cfg.diffusion.continuous, "p_mean", -1.2)
            p_std = getattr(self.trainer.cfg.diffusion.continuous, "p_std", 1.2)
            sigma_base = self.proc.sample_sigma(bsz, strategy="log-normal", p_mean=p_mean, p_std=p_std)
        else:
            # default: log-uniform
            sigma_base = self.proc.sample_sigma(bsz, strategy="log-uniform")

        sigma_base = sigma_base.to(self.trainer.device)

        # OFFLINE MODE: if a fixed profile exists, use it immediately (no warmup/transition)
        if bool(getattr(self.trainer, "entropy_offline_enabled", False)):
            if bool(getattr(self.trainer, "_entropy_ready", False)):
                return self.sample_entropy_sigma(bsz).to(self.trainer.device)
            return sigma_base

        # Entropy scheduling off -> base only
        if not bool(getattr(self.trainer, "entropy_use_for_sampling", False)):
            return sigma_base

        # Not ready -> base only
        if not bool(getattr(self.trainer, "_entropy_ready", False)):
            return sigma_base

        gamma = self.entropy_gamma()
        if gamma <= 0.0:
            return sigma_base

        # mixture: choose entropy with prob gamma
        u = torch.rand(bsz, device=self.trainer.device)
        use_ent = u < gamma
        n_ent = int(use_ent.sum().item())
        if n_ent == 0:
            return sigma_base

        sigma_mix = sigma_base.clone()
        sigma_mix[use_ent] = self.sample_entropy_sigma(n_ent).to(self.trainer.device)
        return sigma_mix

    # ──────────────────────────────────────────────────────────────────────
    # FIFO buffer updates (delegated to Trainer)
    # ──────────────────────────────────────────────────────────────────────
    def update_entropy_buffer(self, sigma: torch.Tensor, entropy_metric: torch.Tensor) -> None:
        """
        Call this from training step *after* you compute the entropy metric.
        """
        if not bool(getattr(self.trainer, "entropy_compute", False)):
            return
        self.trainer.entropy_fifo_push(sigma, entropy_metric)

    # ──────────────────────────────────────────────────────────────────────
    # Recompute schedule (rank0 computes, broadcast to all)
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def recompute_entropy_from_buffer(self) -> None:
        """
        Build entropy schedule from FIFO buffer in RegularizedEntropyMode:

            rate_samples = metric / σ^2
            mean_rate(bin) = mean(rate_samples in bin)

            q(bin) ∝ g(σ_mid; c,n) * mean_rate(bin)^power

        Robustness goals:
        - FIFO buffers are on CPU (Trainer.entropy_fifo_push enforces this), but we still
            defensively move to CPU here.
        - DDP-safe: rank0 decides whether to update and broadcasts a flag; if updating,
            rank0 broadcasts the schedule tensors to all ranks.
        - Config plumbing: reads bins/power/target from cfg first, then falls back to trainer attrs.
        - Numerically safe: clamps, handles empty/degenerate histograms, enforces proper CDF.
        """
        if not bool(getattr(self.trainer, "entropy_compute", False)):
            return

        is_ddp = self._ddp_is_on()
        is_master = bool(getattr(self.trainer, "is_master", False))
        dev = self.trainer.device

        # ----------------------------
        # Decide if we should update (DDP-safe)
        # ----------------------------
        if is_ddp:
            flag = torch.zeros((), device=dev, dtype=torch.int32)
            if is_master:
                do_update = self._enough_fifo_samples()
                flag.fill_(1 if do_update else 0)
            dist.broadcast(flag, src=0)
            if int(flag.item()) == 0:
                return
        else:
            if not self._enough_fifo_samples():
                return

        # ----------------------------
        # Resolve num_bins + power robustly
        # ----------------------------
        cfg_train = getattr(self.trainer.cfg, "train", None)

        def _cfg_get(name: str, default):
            return getattr(cfg_train, name, default) if cfg_train is not None else default

        num_bins = int(
            _cfg_get("entropy_num_bins", getattr(self.trainer, "entropy_num_bins", 256))
        )
        num_bins = max(2, num_bins)  # need at least 2 bins

        # Determine power from config if possible.
        # Priority:
        #   1) cfg.train.entropy_rate_power (explicit float)
        #   2) cfg.train.entropy_target in {"rate","sqrt"}  -> {1.0, 0.5}
        #   3) trainer.entropy_rate_power
        power = _cfg_get("entropy_rate_power", None)
        if power is None:
            target = str(_cfg_get("entropy_target", "")).lower().strip()
            if target in ("sqrt", "sqrt-rate", "sqrt_rate", "sqrt_rate_samples"):
                power = 0.5
            elif target in ("rate", "rate-only", "rate_only"):
                power = 1.0
            else:
                power = float(getattr(self.trainer, "entropy_rate_power", 1.0))
        power = float(power)

        # Regularizer params from cfg (stable defaults)
        c = float(_cfg_get("entropy_regularizer_c", 0.1))
        n = float(_cfg_get("entropy_regularizer_n", 3.0))

        # ----------------------------
        # Non-master ranks: just receive schedule from rank0
        # ----------------------------
        if is_ddp and (not is_master):
            pdf_dev = torch.empty((num_bins,), device=dev, dtype=torch.float32)
            cdf_dev = torch.empty((num_bins,), device=dev, dtype=torch.float32)
            mid_dev = torch.empty((num_bins,), device=dev, dtype=torch.float32)
            edges_dev = torch.empty((num_bins + 1,), device=dev, dtype=torch.float32)

            dist.broadcast(pdf_dev, src=0)
            dist.broadcast(cdf_dev, src=0)
            dist.broadcast(mid_dev, src=0)
            dist.broadcast(edges_dev, src=0)

            # Minimal sanity (avoid NaNs silently propagating)
            if not torch.isfinite(pdf_dev).all() or not torch.isfinite(cdf_dev).all():
                # Fallback to uniform to avoid crashing training
                pdf_dev = torch.full((num_bins,), 1.0 / float(num_bins), device=dev, dtype=torch.float32)
                cdf_dev = torch.cumsum(pdf_dev, dim=0)
                cdf_dev[-1] = 1.0

            self.trainer._entropy_pdf = pdf_dev
            self.trainer._entropy_cdf = cdf_dev
            self.trainer._entropy_sigmas = mid_dev
            self.trainer._entropy_edges = edges_dev
            self.trainer._entropy_ready = True

            self.fit_lognormal_to_entropy_profile()
            return

        # ----------------------------
        # Rank0 (master) computes schedule from FIFO ring buffer
        # ----------------------------
        # Buffer metadata
        cap = int(getattr(self.trainer, "_entropy_buf_cap", self.trainer._entropy_sig_buf.numel()))
        buf_len = int(getattr(self.trainer, "_entropy_buf_len", 0))
        cap = max(1, cap)
        buf_len = max(0, min(buf_len, cap))

        if buf_len <= 0:
            return  # should be impossible given the update gate, but safe

        # Extract filled portion (order irrelevant for histogram)
        if buf_len < cap:
            sig = self.trainer._entropy_sig_buf[:buf_len]
            metr = self.trainer._entropy_metric_buf[:buf_len]
        else:
            sig = self.trainer._entropy_sig_buf
            metr = self.trainer._entropy_metric_buf

        # Defensive: ensure CPU float32
        sig = sig.detach().to("cpu", dtype=torch.float32).flatten()
        metr = metr.detach().to("cpu", dtype=torch.float32).flatten()

        # Filter non-finite entries (robust to occasional NaNs/Infs)
        finite = torch.isfinite(sig) & torch.isfinite(metr)
        if finite.any():
            sig = sig[finite]
            metr = metr[finite]
        else:
            # degenerate: fallback uniform schedule
            pdf = torch.full((num_bins,), 1.0 / float(num_bins), dtype=torch.float32)
            cdf = torch.cumsum(pdf, dim=0)
            cdf[-1] = 1.0
            sigma_min = float(self.proc.sigma_min)
            sigma_max = float(self.proc.sigma_max)
            edges_log = torch.linspace(math.log(sigma_min), math.log(sigma_max), num_bins + 1, dtype=torch.float32)
            mids = (0.5 * (edges_log[:-1] + edges_log[1:])).exp().clamp(min=1e-12)
            edges_sigma = edges_log.exp().clamp(min=1e-12)
            pdf_dev, cdf_dev, mid_dev, edges_dev = pdf.to(dev), cdf.to(dev), mids.to(dev), edges_sigma.to(dev)
            if is_ddp:
                dist.broadcast(pdf_dev, src=0)
                dist.broadcast(cdf_dev, src=0)
                dist.broadcast(mid_dev, src=0)
                dist.broadcast(edges_dev, src=0)
            self.trainer._entropy_pdf = pdf_dev
            self.trainer._entropy_cdf = cdf_dev
            self.trainer._entropy_sigmas = mid_dev
            self.trainer._entropy_edges = edges_dev
            self.trainer._entropy_ready = True
            if is_master:
                self.save_entropy_tables(pdf_dev, cdf_dev, mid_dev, edges_dev)
            self.fit_lognormal_to_entropy_profile()
            return

        # Clamp sigma into supported range (also ensures positive)
        sigma_min = float(self.proc.sigma_min)
        sigma_max = float(self.proc.sigma_max)
        sigma_min = max(sigma_min, 1e-12)
        sigma_max = max(sigma_max, sigma_min * 1.0000001)  # avoid log_max==log_min
        sig = sig.clamp(min=sigma_min, max=sigma_max)

        # Build log-spaced edges on CPU
        log_min = math.log(sigma_min)
        log_max = math.log(sigma_max)
        edges_log = torch.linspace(log_min, log_max, num_bins + 1, dtype=torch.float32)

        # Bin index in [0, num_bins-1]
        log_sig = sig.log()
        bin_idx = torch.bucketize(log_sig, edges_log, right=False) - 1
        bin_idx.clamp_(0, num_bins - 1)

        # Rate proxy: metric / sigma^2
        rate_samples = metr / (sig * sig + 1e-12)
        rate_samples = torch.clamp(rate_samples, min=0.0)  # keep nonnegative

        # Aggregate per bin
        sum_rate = torch.zeros((num_bins,), dtype=torch.float32)
        count = torch.zeros((num_bins,), dtype=torch.float32)
        sum_rate.scatter_add_(0, bin_idx, rate_samples)
        count.scatter_add_(0, bin_idx, torch.ones_like(rate_samples, dtype=torch.float32))

        mean_rate = sum_rate / (count + 1e-8)
        mean_rate = torch.clamp(mean_rate, min=0.0)

        # Midpoints & edges (sigma-domain)
        mid_log = 0.5 * (edges_log[:-1] + edges_log[1:])
        mids = mid_log.exp().clamp(min=1e-12)
        edges_sigma = edges_log.exp().clamp(min=1e-12)

        # Regularizer g(sigma_mid)
        reg = _generalized_regularizer(mids, c=c, n=n).to(torch.float32)

        # Rate exponent
        rate_term = mean_rate.pow(power) if power != 0.0 else torch.ones_like(mean_rate)

        unnormalized = reg * rate_term

        # Handle degeneracies / NaNs / all-zero
        good = torch.isfinite(unnormalized).all() and float(unnormalized.sum().item()) > 0.0
        if not good:
            pdf = torch.full((num_bins,), 1.0 / float(num_bins), dtype=torch.float32)
        else:
            pdf = unnormalized / unnormalized.sum()

        # CDF must be monotone and end at 1
        cdf = torch.cumsum(pdf, dim=0)
        # enforce monotonicity (rarely needed, but safe)
        cdf = torch.maximum(cdf, torch.cummax(cdf, dim=0)[0])
        cdf[-1] = 1.0

        # Move to device (GPU) for sampling
        pdf_dev = pdf.to(dev, dtype=torch.float32)
        cdf_dev = cdf.to(dev, dtype=torch.float32)
        mid_dev = mids.to(dev, dtype=torch.float32)
        edges_dev = edges_sigma.to(dev, dtype=torch.float32)

        # Broadcast schedule to all ranks
        if is_ddp:
            dist.broadcast(pdf_dev, src=0)
            dist.broadcast(cdf_dev, src=0)
            dist.broadcast(mid_dev, src=0)
            dist.broadcast(edges_dev, src=0)

        # Commit schedule
        self.trainer._entropy_pdf = pdf_dev
        self.trainer._entropy_cdf = cdf_dev
        self.trainer._entropy_sigmas = mid_dev
        self.trainer._entropy_edges = edges_dev
        self.trainer._entropy_ready = True

        # Save + diagnostics
        if is_master:
            self.save_entropy_tables(pdf_dev, cdf_dev, mid_dev, edges_dev)

        self.fit_lognormal_to_entropy_profile()

        if is_master:
            print(
                f"✓ Entropy schedule updated: FIFO={buf_len}/{cap}, bins={num_bins}, "
                f"mode=regularized, c={c}, n={n}, power={power}"
            )


    # ──────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────
    def fit_lognormal_to_entropy_profile(self) -> None:
        """
        Diagnostic-only: fit LogNormal(mu, std_log) to discrete PDF over midpoints.
        """
        if not bool(getattr(self.trainer, "_entropy_ready", False)):
            self.trainer._entropy_ln_mu = None
            self.trainer._entropy_ln_std = None
            return

        pdf = self.trainer._entropy_pdf.detach().float().cpu()
        mids = self.trainer._entropy_sigmas.detach().float().cpu()

        pdf = pdf.clamp(min=0.0)
        tot = float(pdf.sum())
        if not math.isfinite(tot) or tot <= 0.0:
            self.trainer._entropy_ln_mu = None
            self.trainer._entropy_ln_std = None
            return

        pdf = pdf / tot
        log_m = mids.clamp_min(1e-12).log()

        mu = (pdf * log_m).sum()
        var = (pdf * (log_m - mu).pow(2)).sum()
        std = torch.sqrt(var.clamp_min(1e-12))

        self.trainer._entropy_ln_mu = float(mu.item())
        self.trainer._entropy_ln_std = float(std.item())

        if bool(getattr(self.trainer, "is_master", False)):
            print(f"✓ (diag) fitted LogNormal: μ={self.trainer._entropy_ln_mu:.3f}, σlog={self.trainer._entropy_ln_std:.3f}")
