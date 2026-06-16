from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast

from .base import Callback
from diffusion.continuous.logit_postprocess import _model_logits_continuous


class EntropySchedulePlotCallback(Callback):
    """
    Plots the current entropic σ profile and γ(global_step) every K epochs.

    Works for BOTH:
      - continuous binary runs
      - continuous one-hot token runs

    Adds a 3rd panel:
      - denoise error proxy vs σ
    """

    def __init__(
        self,
        every_k_epochs: int = 50,
        n_hist_samples: int = 20_000,
        denoise_curve_K: int = 64,
        denoise_curve_num_batches: int = 3,
        denoise_curve_batch_size: int = 64,
    ):
        super().__init__()
        self.every_k_epochs = every_k_epochs
        self.n_hist_samples = n_hist_samples
        self.denoise_curve_K = int(denoise_curve_K)
        self.denoise_curve_num_batches = int(denoise_curve_num_batches)
        self.denoise_curve_batch_size = int(denoise_curve_batch_size)

    @staticmethod
    def _maybe_load_entropy_tables(trainer) -> bool:
        if getattr(trainer, "_entropy_ready", False):
            return True

        off = getattr(trainer.cfg.train, "entropy_offline", None)
        offline_enabled = bool(getattr(off, "enabled", False)) if off is not None else False
        if not offline_enabled:
            return False

        entropy_run_dir = getattr(off, "entropy_run_dir", None)
        if entropy_run_dir is None:
            entropy_run_dir = getattr(trainer, "run_dir", None)
        if entropy_run_dir is None:
            return False

        entropy_run_dir = Path(entropy_run_dir).expanduser().resolve()
        pdf_p = entropy_run_dir / "entropy_pdf.pt"
        cdf_p = entropy_run_dir / "entropy_cdf.pt"
        sig_p = entropy_run_dir / "entropy_sigmas.pt"

        if not (pdf_p.exists() and cdf_p.exists() and sig_p.exists()):
            return False

        dev = trainer.device
        try:
            pdf = torch.load(pdf_p, map_location=dev).to(dev).float()
            cdf = torch.load(cdf_p, map_location=dev).to(dev).float()
            sig = torch.load(sig_p, map_location=dev).to(dev).float()
        except Exception as e:
            if getattr(trainer, "is_master", False):
                print(f"[WARN] Failed to load entropy tables from {entropy_run_dir}: {e}")
            return False

        if pdf.numel() == 0 or cdf.numel() == 0 or sig.numel() == 0:
            return False

        cdf[-1] = 1.0

        trainer._entropy_pdf = pdf
        trainer._entropy_cdf = cdf
        trainer._entropy_sigmas = sig
        trainer._entropy_ready = True

        if getattr(trainer, "is_master", False):
            print(f"✓ EntropySchedulePlotCallback loaded tables from {entropy_run_dir}")

        return True

    @staticmethod
    @torch.no_grad()
    def _estimate_denoise_curve(
        trainer,
        sigmas_full,
        *,
        K: int = 64,
        num_batches: int = 1,
        batch_size=None,
    ):
        device = trainer.device
        sigmas_full = sigmas_full.to(device).float()

        if sigmas_full.numel() < 2:
            return sigmas_full.detach().cpu(), torch.zeros_like(sigmas_full).detach().cpu()

        K = int(min(K, sigmas_full.numel()))
        idx = torch.linspace(0, sigmas_full.numel() - 1, K, device=device).long()
        sig_points = sigmas_full[idx]

        use_amp = bool(getattr(trainer.cfg.train, "use_fp16", False))
        amp_dtype = getattr(trainer, "amp_dtype", torch.float16)

        repr_mode = str(getattr(trainer.cfg.data, "representation", "binary")).lower()
        is_cont_tokens = (repr_mode == "tokens")
        self_condition = bool(getattr(trainer.cfg.model, "self_condition", False))

        curve_sum = torch.zeros(K, device=device, dtype=torch.float32)
        curve_count = 0

        it = iter(trainer.val_loader)
        for _ in range(int(num_batches)):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(trainer.val_loader)
                batch = next(it)

            if isinstance(batch, dict):
                x0 = batch["x0"]
            elif isinstance(batch, (tuple, list)):
                x0 = batch[0]
            else:
                x0 = batch
            x0 = x0.to(device, non_blocking=True)

            if batch_size is not None and x0.size(0) > batch_size:
                x0 = x0[:batch_size]

            B = x0.size(0)
            if B == 0:
                continue

            if is_cont_tokens:
                V = int(trainer.cfg.data.vocab_size)
                x0_target = x0.long().view(B, -1)  # [B,S]
                x0_clean = F.one_hot(x0_target, num_classes=V).to(
                    dtype=(amp_dtype if use_amp else torch.float32)
                )  # [B,S,V]
                loss_target = x0_target
            else:
                x0_clean = x0.view(B, -1).to(torch.float32)  # [B,S]
                loss_target = x0_clean

            for i in range(K):
                s = sig_points[i]
                sigma_rep = s.expand(B)  # [B]

                xt = x0_clean + s.to(dtype=x0_clean.dtype) * torch.randn_like(x0_clean)
                x0_hat = torch.zeros_like(xt) if self_condition else None

                with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                    logits = _model_logits_continuous(
                        trainer.model,
                        trainer.cfg,
                        xt,
                        sigma_rep,
                        x0_hat,
                    )

                    _, entropy_metric = trainer.loss_fn(
                        logits,
                        loss_target,
                        sigma_rep,
                        trainer.cfg,
                        return_entropy_metric=True,
                    )

                curve_sum[i] += entropy_metric.float().mean()

            curve_count += 1

        if curve_count > 0:
            curve_sum /= float(curve_count)

        return sig_points.detach().cpu(), curve_sum.detach().cpu()

    def on_epoch_end(self, trainer, epoch: int):
        if not self._maybe_load_entropy_tables(trainer):
            return

        if self.every_k_epochs <= 0:
            return
        if (epoch + 1) % self.every_k_epochs != 0:
            return

        pdf = trainer._entropy_pdf.detach().cpu()
        sigmas = trainer._entropy_sigmas.detach().cpu()

        gamma = float(trainer._entropy_gamma()) if hasattr(trainer, "_entropy_gamma") else 0.0
        if pdf.numel() == 0 or sigmas.numel() == 0:
            return

        import matplotlib.pyplot as plt
        import numpy as np

        # x-axis in log10(σ) for readability
        log10_sig = sigmas.clamp(min=1e-12).log10().numpy()
        y_pdf = pdf.numpy()

        fig, (ax1, ax3, ax2) = plt.subplots(3, 1, figsize=(7, 11), sharex=True)

        # ------------------------------------------------------------------
        # Panel 1: entropy PDF over log-spaced σ bins + Normal fit in log(σ)
        # ------------------------------------------------------------------
        ax1.plot(log10_sig, y_pdf, marker="o", linewidth=1, label="entropy PDF")

        mu = getattr(trainer, "_entropy_ln_mu", None)
        std = getattr(trainer, "_entropy_ln_std", None)
        if (
            mu is not None
            and std is not None
            and math.isfinite(float(mu))
            and math.isfinite(float(std))
            and float(std) > 0.0
        ):
            log_sig = sigmas.clamp(min=1e-12).log()  # natural log
            normal_log = torch.exp(-0.5 * ((log_sig - float(mu)) / float(std)) ** 2)
            normal_log = normal_log / (float(std) * math.sqrt(2.0 * math.pi))
            normal_log = (normal_log / normal_log.sum()).numpy()

            ax1.plot(log10_sig, normal_log, linewidth=2, label="Normal fit on log σ")

        ax1.set_ylabel("p(bin) (discrete)")
        ax1.set_title(f"Entropy profile at epoch {epoch+1}, γ={gamma:.3f}")
        ax1.legend(loc="best")

        ax1b = ax1.twinx()
        ax1b.axhline(gamma, linestyle="--", alpha=0.6)
        ax1b.set_ylabel("γ (mixture weight)")
        ax1b.set_ylim(0.0, max(1.0, gamma * 1.1))

        # ------------------------------------------------------------------
        # Panel 2: denoise curve vs σ
        # ------------------------------------------------------------------
        sig_pts_cpu, eps2_cpu = self._estimate_denoise_curve(
            trainer,
            trainer._entropy_sigmas,
            K=self.denoise_curve_K,
            num_batches=self.denoise_curve_num_batches,
            batch_size=self.denoise_curve_batch_size,
        )
        log10_sig_pts = sig_pts_cpu.clamp(min=1e-12).log10().numpy()

        ax3.plot(log10_sig_pts, eps2_cpu.numpy(), marker="o", linewidth=1)
        ax3.set_xlabel("log10 σ")
        ax3.set_ylabel("MSE / loss proxy (log scale)")
        ax3.set_yscale("log")
        ax3.grid(True, which="both", ls="-", alpha=0.2)

        # ------------------------------------------------------------------
        # Panel 3: histograms of base sampler and actual training sampler
        # ------------------------------------------------------------------
        n = int(self.n_hist_samples)
        with torch.no_grad():
            base_strategy = (
                getattr(trainer.cfg.train, "sigma_sampling_strategy", "log-uniform") or "log-uniform"
            ).lower()

            if base_strategy not in {"log-uniform", "log-normal"}:
                base_strategy = "log-uniform"
            if base_strategy == "entropy":
                base_strategy = "log-uniform"

            sig_base = trainer.proc.sample_sigma(n, strategy=base_strategy).cpu()
            sig_mix = trainer._draw_sigma(n).cpu() if hasattr(trainer, "_draw_sigma") else sig_base

        log_base = sig_base.clamp(min=1e-12).log10().numpy()
        log_mix = sig_mix.clamp(min=1e-12).log10().numpy()

        x_min = min(log_base.min(), log_mix.min())
        x_max = max(log_base.max(), log_mix.max())
        bins = np.linspace(x_min, x_max, 80)

        ax2.hist(log_base, bins=bins, density=True, alpha=0.5, label=f"base ({base_strategy})")
        ax2.hist(log_mix, bins=bins, density=True, alpha=0.5, label="train σ (draw_sigma)")
        ax2.set_ylabel("Empirical density (log10 σ)")
        ax2.legend(loc="best")

        # ------------------------------------------------------------------
        # Save / log
        # ------------------------------------------------------------------
        fig.tight_layout()

        trainer.writer.add_figure(
            "entropy/profile_gamma_and_denoise_curve",
            fig,
            global_step=trainer.global_step,
        )

        out_dir = Path(trainer.run_dir) / "entropy_plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"entropy_epoch{epoch+1:04d}.png", bbox_inches="tight")
        plt.close(fig)