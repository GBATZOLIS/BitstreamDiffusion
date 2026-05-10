# callbacks/offline_entropy_profile.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None

from .base import Callback


class OfflineEntropyProfileCallback(Callback):
    """
    Offline entropy-rate profile estimation at the start of training, using a
    *separate* probe model loaded from a checkpoint.

    GOAL:
      Build a FIXED sigma sampling distribution that matches the *online*
      EntropyScheduleController.recompute_entropy_from_buffer() construction:

        rate_samples = metric / sigma^2
        mean_rate(bin) = mean(rate_samples in bin)

        q(bin) ∝ g(σ_mid; c,n) * mean_rate(bin)^power

      where g is Gabriel's regularizer:
        g(σ) = σ^n / (σ^n + c^n)

      and power is derived from cfg.train.entropy_target:
        "rate" -> 1.0
        "sqrt" -> 0.5
        or a numeric string / float.

    CONDITIONAL SUPPORT (matches Trainer._step_continuous entropy metric logic):
      - If cfg.cond.enabled=True and cL>0:
          * if noise_prefix=False: keep prefix clean and noise suffix only
            and compute metric on suffix only.
          * No CFG dropout in offline profiling (prefix always present).

    SELF-CONDITIONING SUPPORT (matches Trainer._step_continuous):
      - if cfg.model.self_condition=True:
          compute x0_hat via an extra no-grad forward on a random subset
          sc_mask ~ Bernoulli(p_sc=cfg.train.self_condition_prob)
      - then do main forward using x0_hat
      - in conditional clean-prefix mode, overwrite x0_hat prefix with true prefix

    DDP:
      - run_on_all_ranks=True
      - work is sharded across ranks; then reduced via all_reduce.

    Outputs (stored exactly like online controller expects):
      trainer._entropy_pdf   [K]
      trainer._entropy_cdf   [K]
      trainer._entropy_sigmas (midpoints) [K]
      trainer._entropy_edges (edges)      [K+1]
      trainer._entropy_ready = True

      also saved to disk via trainer._save_entropy_tables(...)
      and diagnostic log-normal fit via trainer._fit_lognormal_to_entropy_profile()
    """

    run_on_all_ranks = True

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _cond_len_bits(cfg, seq_len_bits: int) -> int:
        cond_cfg = getattr(cfg, "cond", None)
        if cond_cfg is None or not bool(getattr(cond_cfg, "enabled", False)):
            return 0
        bits_per_char = int(getattr(getattr(cfg, "data", object()), "bits_per_char", 1))
        cond_len_chars = int(getattr(cond_cfg, "cond_len_chars", 0))
        cL = int(cond_len_chars * bits_per_char)
        return max(0, min(cL, seq_len_bits))

    @staticmethod
    def _clean_state_dict(sd: dict) -> dict:
        clean = {}
        for k, v in sd.items():
            k = k.replace("_orig_mod.", "")
            if k.startswith("module."):
                k = k[7:]
            clean[k] = v
        return clean

    @staticmethod
    def _generalized_regularizer(sigmas: torch.Tensor, c: float, n: float) -> torch.Tensor:
        # Same as EntropyScheduleController._generalized_regularizer
        c = float(max(c, 1e-12))
        n = float(n)
        x = (sigmas.clamp_min(1e-12) / c).pow(n)
        return x / (1.0 + x)

    @staticmethod
    def _get_entropy_power(trainer, off_cfg) -> float:
        """
        Priority:
          1) entropy_offline.power (explicit override)
          2) trainer.entropy_rate_power (parsed in Trainer.__init__)
          3) parse cfg.train.entropy_target
        """
        power_override = getattr(off_cfg, "power", None)
        if power_override is not None:
            return float(power_override)

        p = getattr(trainer, "entropy_rate_power", None)
        if p is not None:
            return float(p)

        target = getattr(trainer.cfg.train, "entropy_target", "rate")
        if isinstance(target, str):
            t = target.lower()
            if t == "rate":
                return 1.0
            if t in {"sqrt", "sqrt_rate", "sqrt-rate"}:
                return 0.5
            return float(t)
        return float(target)

    def _build_probe_model_from_checkpoint(self, trainer, ckpt_path: str):
        from models import create_model
        from utils.ema import EMA

        device = trainer.device
        ckpt = torch.load(ckpt_path, map_location="cpu")

        probe = create_model(trainer.cfg).to(device)
        probe.load_state_dict(self._clean_state_dict(ckpt["model"]), strict=False)
        probe.eval()

        off = getattr(trainer.cfg.train, "entropy_offline", None)
        use_ckpt_ema = True if off is None else bool(getattr(off, "checkpoint_use_ema", True))
        if trainer.is_master:
            print(f"— OfflineEntropyProfile: checkpoint_use_ema={use_ckpt_ema}")

        # Checkpoint EMA format: {"decay": float, "shadow": {...}} (your current format)
        if use_ckpt_ema and ("ema" in ckpt) and (ckpt["ema"] is not None):
            ema_probe = EMA(probe, decay=trainer.cfg.train.ema_decay)
            ema_probe.load_state_dict(ckpt["ema"])
            ema_probe.to(device)
            ema_probe.apply(probe)

        return probe

    def _model_forward(
        self,
        model,
        xt: torch.Tensor,
        sigma: torch.Tensor,
        x0_hat: Optional[torch.Tensor],
    ):
        """
        Backward compatible forward:
          - old: model(xt, sigma)
          - new: model(xt, sigma, x0_hat)
        """
        if x0_hat is None:
            return model(xt, sigma)
        try:
            return model(xt, sigma, x0_hat)
        except TypeError:
            return model(xt, sigma)

    def _log_entropy_profile_to_tb(self, trainer, pdf, cdf, sigmas_mid, edges_sigma):
        if not getattr(trainer, "is_master", False):
            return
        if getattr(trainer, "writer", None) is None:
            return

        import numpy as np
        import matplotlib.pyplot as plt

        pdf_cpu = pdf.detach().float().cpu()
        cdf_cpu = cdf.detach().float().cpu()
        mid_cpu = sigmas_mid.detach().float().cpu().clamp(min=1e-12)
        edges_cpu = edges_sigma.detach().float().cpu().clamp(min=1e-12)

        if pdf_cpu.numel() == 0 or mid_cpu.numel() == 0 or edges_cpu.numel() < 2:
            return

        log10_mid = mid_cpu.log10().numpy()

        n_samp = int(getattr(trainer.cfg.train.entropy_offline, "tb_num_samples", 50_000))
        n_samp = max(1_000, n_samp)

        # sample bins then sample log-uniform within bin
        idx = torch.multinomial(pdf_cpu, num_samples=n_samp, replacement=True)
        lo = edges_cpu[idx].clamp(min=1e-12)
        hi = edges_cpu[idx + 1].clamp(min=1e-12)

        u = torch.rand(n_samp)
        log_sigma = lo.log() + u * (hi.log() - lo.log())
        samp_sig = log_sigma.exp()
        samp_log10 = samp_sig.clamp(min=1e-12).log10().numpy()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
        ax1.plot(log10_mid, pdf_cpu.numpy(), marker="o", linewidth=1)
        ax1.set_ylabel("p(bin) (discrete)")
        ax1.set_title("Offline entropy σ profile (bins) + within-bin log-uniform sampling")

        ax2.plot(log10_mid, cdf_cpu.numpy(), marker=".", linewidth=1)
        ax2.set_ylabel("CDF")
        ax2.set_ylim(0.0, 1.0)

        bins = np.linspace(samp_log10.min(), samp_log10.max(), 80)
        ax3.hist(samp_log10, bins=bins, density=True, alpha=0.6, label="empirical σ samples")
        ax3.set_xlabel("log10 σ")
        ax3.set_ylabel("density")
        ax3.legend(loc="best")

        fig.tight_layout()
        step = int(getattr(trainer, "global_step", 0))
        trainer.writer.add_figure("entropy_offline/profile_pdf_cdf_hist", fig, global_step=step)
        plt.close(fig)

        trainer.writer.add_histogram("entropy_offline/sigma_samples", samp_sig, global_step=step)
        trainer.writer.add_scalar("entropy_offline/pdf_max", float(pdf_cpu.max().item()), step)
        trainer.writer.add_scalar("entropy_offline/pdf_min", float(pdf_cpu.min().item()), step)

    # ──────────────────────────────────────────────────────────────────────
    # Main
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def on_train_begin(self, trainer: Any):
        if getattr(trainer.cfg, "framework", "") != "continuous_score":
            return

        off = getattr(trainer.cfg.train, "entropy_offline", None)
        if off is None or not bool(getattr(off, "enabled", False)):
            return

        rng_state = trainer._rng_state() if hasattr(trainer, "_rng_state") else None
        model_for_profile = None

        try:
            overwrite = bool(getattr(off, "overwrite_existing", False))

            # --- DDP-safe: all ranks must make the SAME skip/run decision ---
            if (
                getattr(trainer, "is_distributed", False)
                and dist is not None
                and dist.is_available()
                and dist.is_initialized()
            ):
                # rank0 decides; broadcast to all ranks
                flag = torch.zeros((), device=trainer.device, dtype=torch.int32)
                if trainer.is_master:
                    already_ready = bool(getattr(trainer, "_entropy_ready", False))
                    do_profile = overwrite or (not already_ready)
                    flag.fill_(1 if do_profile else 0)
                dist.broadcast(flag, src=0)
                do_profile = bool(int(flag.item()))
            else:
                already_ready = bool(getattr(trainer, "_entropy_ready", False))
                do_profile = overwrite or (not already_ready)

            if not do_profile:
                if trainer.is_master:
                    print("✓ OfflineEntropyProfileCallback: entropy profile already ready; skipping.")
                return


            # which split to probe
            split = str(getattr(off, "data_split", "val")).lower()
            if split == "train":
                loader = trainer.train_loader
                if getattr(trainer, "is_distributed", False) and hasattr(loader.sampler, "set_epoch"):
                    loader.sampler.set_epoch(0)
            else:
                loader = trainer.val_loader

            # binning
            num_bins = int(getattr(off, "num_bins", 512))
            samples_per_bin = int(getattr(off, "samples_per_bin", 2048))
            within = str(getattr(off, "sigma_within_bin", "log-uniform")).lower()  # "log-uniform" | "midpoint"

            sigma_min = float(trainer.proc.sigma_min)
            sigma_max = float(trainer.proc.sigma_max)
            log_min = math.log(sigma_min)
            log_max = math.log(sigma_max)

            device = trainer.device

            # IMPORTANT: use the same binning style as online controller: equal-width in log-domain
            edges_log = torch.linspace(log_min, log_max, num_bins + 1, device=device, dtype=torch.float32)
            edges_sigma = edges_log.exp().clamp(min=1e-12)
            mid_log = 0.5 * (edges_log[:-1] + edges_log[1:])
            sigmas_mid = mid_log.exp().clamp(min=1e-12)  # [K]

            # checkpoint
            ckpt_path = getattr(off, "checkpoint_path", None)
            if ckpt_path is None or str(ckpt_path) == "":
                raise ValueError("cfg.train.entropy_offline.checkpoint_path must be set.")
            ckpt_path = str(ckpt_path)
            if not Path(ckpt_path).exists():
                raise FileNotFoundError(f"OfflineEntropyProfile checkpoint not found: {ckpt_path}")

            model_for_profile = self._build_probe_model_from_checkpoint(trainer, ckpt_path)
            if trainer.is_master:
                print(f"— OfflineEntropyProfile: using probe checkpoint for σ-profile: {ckpt_path}")

            # Work allocation: exact (bin,sample) pairs, shuffled deterministically, then strided by rank.
            N_total = num_bins * samples_per_bin
            bin_ids = torch.arange(num_bins, dtype=torch.int64).repeat_interleave(samples_per_bin)

            base_seed = int(getattr(trainer.cfg.train, "seed", 0))
            g = torch.Generator(device="cpu")
            g.manual_seed(base_seed + 12345)
            perm = torch.randperm(bin_ids.numel(), generator=g)
            bin_ids = bin_ids[perm]

            n_local = int(bin_ids.numel())
            if trainer.is_master:
                print(
                    f"— OfflineEntropyProfile: bins={num_bins}, samples/bin={samples_per_bin} "
                    f"(total={N_total}), world_size={trainer.world_size}, local≈{n_local}"
                )

            # Accumulators (rate = metric/sigma^2)
            sum_rate = torch.zeros(num_bins, device=device, dtype=torch.float32)
            count = torch.zeros(num_bins, device=device, dtype=torch.float32)

            use_amp = bool(getattr(trainer.cfg.train, "use_fp16", False))
            amp_dtype = getattr(trainer, "amp_dtype", torch.float16)

            # Self-conditioning knobs (match training)
            sc_enabled = bool(getattr(trainer.cfg.model, "self_condition", False))
            p_sc = float(getattr(trainer.cfg.train, "self_condition_prob", 0.5))

            it = iter(loader)
            processed = 0
            pbar = tqdm(
                total=n_local,
                desc="OfflineEntropyProfile",
                disable=not trainer.is_master,
                leave=False,
            )

            while processed < n_local:
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)

                x0 = batch[0] if isinstance(batch, (tuple, list)) else batch
                B_full = int(x0.size(0))
                B = min(B_full, n_local - processed)

                x0 = x0[:B].to(device, non_blocking=True).view(B, -1).float()  # [B,S]
                S = int(x0.size(1))
                bins = bin_ids[processed : processed + B].to(device=device, dtype=torch.int64)  # [B]

                # sigma within bin
                if within == "midpoint":
                    log_sigma = 0.5 * (edges_log[bins] + edges_log[bins + 1])
                else:
                    u = torch.rand(B, device=device)
                    log_sigma = edges_log[bins] + u * (edges_log[bins + 1] - edges_log[bins])

                sigma = log_sigma.exp().clamp(min=sigma_min, max=sigma_max)  # [B]

                # conditional setup (NO CFG DROPOUT in offline profiling)
                cL = self._cond_len_bits(trainer.cfg, S)
                cond_cfg = getattr(trainer.cfg, "cond", None)
                cond_enabled = (cond_cfg is not None) and bool(getattr(cond_cfg, "enabled", False)) and (cL > 0)

                noise_prefix = bool(getattr(cond_cfg, "noise_prefix", False)) if cond_enabled else True

                # build xt
                if cond_enabled and (not noise_prefix):
                    prefix = x0[:, :cL]
                    suffix = x0[:, cL:]
                    xt = x0.clone()
                    xt[:, :cL] = prefix
                    xt[:, cL:] = suffix + sigma.view(-1, 1) * torch.randn_like(suffix)
                else:
                    xt = x0 + sigma.view(-1, 1) * torch.randn_like(x0)

                # base x0_hat (match training)
                x0_hat = torch.zeros_like(x0)  # [B,S]
                if cond_enabled and (not noise_prefix):
                    x0_hat[:, :cL] = x0[:, :cL]  # true prefix always present

                # self-conditioning pass (match training; no-grad extra forward on subset)
                if sc_enabled and (p_sc > 0.0):
                    sc_mask = (torch.rand(B, device=device) < p_sc)
                    if sc_mask.any():
                        xt_sc = xt[sc_mask]
                        sigma_sc = sigma[sc_mask]
                        x0_hat_sc_in = torch.zeros_like(xt_sc)

                        with autocast(enabled=use_amp, dtype=amp_dtype):
                            logits_sc = self._model_forward(model_for_profile, xt_sc, sigma_sc, x0_hat_sc_in)

                        if logits_sc.dim() == 3 and logits_sc.size(-1) == 1:
                            logits_sc = logits_sc.squeeze(-1)  # [Bsc,S]

                        x0_hat_sc = torch.sigmoid(logits_sc.float()).to(dtype=x0_hat.dtype)
                        x0_hat[sc_mask] = x0_hat_sc

                        # keep conditional prefix consistent (like training)
                        if cond_enabled and (not noise_prefix):
                            x0_hat[:, :cL] = x0[:, :cL]

                # main forward + entropy metric (must match Trainer entropy_metric rule)
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    logits = self._model_forward(model_for_profile, xt, sigma, x0_hat)
                    if logits.dim() == 3 and logits.size(-1) == 1:
                        logits = logits.squeeze(-1)  # [B,S]

                    # entropy metric: suffix-only if clean-prefix conditioning; else full
                    if cond_enabled and (not noise_prefix):
                        logits_m = logits[:, cL:]
                        x0_m = x0[:, cL:]
                    else:
                        logits_m = logits
                        x0_m = x0

                    # Use the same entropy metric definition as training loss_fn provides
                    _, entropy_metric = trainer.loss_fn(
                        logits_m, x0_m, sigma, trainer.cfg, return_entropy_metric=True
                    )  # [B]

                # rate samples
                rate = entropy_metric / (sigma.pow(2) + 1e-12)  # [B]

                sum_rate.scatter_add_(0, bins, rate.to(torch.float32))
                count.scatter_add_(0, bins, torch.ones_like(rate, dtype=torch.float32))

                processed += B
                pbar.update(B)

            pbar.close()

            # sync across ranks
            if (
                getattr(trainer, "is_distributed", False)
                and dist is not None
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(sum_rate, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)

            mean_rate = (sum_rate / (count + 1e-8)).clamp_min(0.0)  # [K]

            # --- MATCH ONLINE CONTROLLER EXACTLY: regularized * rate^power ---
            c_reg = float(getattr(trainer.cfg.train, "entropy_regularizer_c", 0.1))
            n_reg = float(getattr(trainer.cfg.train, "entropy_regularizer_n", 3.0))
            reg = self._generalized_regularizer(sigmas_mid, c=c_reg, n=n_reg)

            power = self._get_entropy_power(trainer, off)
            unnormalized = reg * mean_rate.pow(power)

            if (unnormalized.sum() <= 0) or (not torch.isfinite(unnormalized).all()):
                pdf = torch.ones_like(unnormalized) / float(num_bins)
            else:
                pdf = unnormalized / unnormalized.sum()

            cdf = torch.cumsum(pdf, dim=0)
            cdf[-1] = 1.0

            # store (as controller expects)
            trainer._entropy_pdf = pdf
            trainer._entropy_cdf = cdf
            trainer._entropy_sigmas = sigmas_mid
            trainer._entropy_edges = edges_sigma
            trainer._entropy_ready = True

            if trainer.is_master:
                trainer._save_entropy_tables(pdf, cdf, sigmas_mid, edges_sigma)

            trainer._fit_lognormal_to_entropy_profile()
            self._log_entropy_profile_to_tb(trainer, pdf, cdf, sigmas_mid, edges_sigma)

            if trainer.is_master:
                mode = "conditional" if (getattr(trainer.cfg, "cond", None) is not None and bool(getattr(trainer.cfg.cond, "enabled", False))) else "unconditional"
                print(
                    f"✓ Offline entropy profile ready ({mode}). "
                    f"(bins={num_bins}, samples/bin={samples_per_bin}, within={within}, "
                    f"c={c_reg}, n={n_reg}, power={power}, self_cond={sc_enabled})"
                )

        finally:
            # cleanup probe model VRAM
            if model_for_profile is not None:
                del model_for_profile
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # restore RNG state so offline probing doesn't perturb training randomness
            if rng_state is not None and hasattr(trainer, "_set_rng_state"):
                trainer._set_rng_state(rng_state)
