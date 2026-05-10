from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

import torch

from models import create_model
from utils.ema import EMA
from data import get_loader
from evaluation.utils import load_config, unwrap_all, load_checkpoint
from diffusion.continuous.losses import binary_score_interpolation_loss

# [ECC IMPORTS] required by conditioning helpers
from utils.ecc_secded import ecc_from_cfg, ecc_chunk_len

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: DDP (multi-GPU, single node)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None


def _ddp_is_on() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return int(dist.get_rank()) if _ddp_is_on() else 0


def _world() -> int:
    return int(dist.get_world_size()) if _ddp_is_on() else 1


def _rank0() -> bool:
    return _rank() == 0


def _setup_ddp_if_needed(enable_ddp: bool) -> None:
    """
    Enable multi-GPU parallelization on a single node (torchrun).
    Does nothing in single GPU / single process mode.
    """
    if not enable_ddp:
        return

    if dist is None:
        raise RuntimeError("torch.distributed is not available but --ddp was requested.")

    if _ddp_is_on():
        return  # already initialized

    # torchrun sets these
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "--ddp requested but RANK/WORLD_SIZE not found. "
            "Launch with torchrun, e.g. `torchrun --standalone --nproc_per_node=2 -m evaluation.analyse_model ... --ddp`"
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()


def _cleanup_ddp() -> None:
    if _ddp_is_on():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()


def _tqdm_if_rank0(it, **kwargs):
    """
    Only rank0 shows a progress bar (avoids messy multi-line output).
    """
    return tqdm(it, **kwargs) if _rank0() else it


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ONSET_PERCENTILE = 0.0005
DEFAULT_SATURATION_PERCENTILE = 0.999


# -----------------------------------------------------------------------------
# Conditioning helpers (shared with training)  [USER-PROVIDED]
# -----------------------------------------------------------------------------
def _bits_per_unit(cfg) -> int:
    data = getattr(cfg, "data", object())

    ecc_cfg = getattr(data, "ecc", None)
    if ecc_cfg is not None and bool(getattr(ecc_cfg, "enabled", False)):
        ecc = ecc_from_cfg(cfg)
        return int(ecc_chunk_len(ecc))  # e.g. 21

    bpt = getattr(data, "bits_per_token", None)
    if bpt is not None:
        return int(bpt)
    return int(getattr(data, "bits_per_char", 1))


def _cond_len_bits_fixed(cfg, seq_len_bits: int) -> int:
    """
    Fixed prefix length in bits (backward compatible).
    Prefers cfg.cond.cond_len_tokens (semantic/BPE), else cfg.cond.cond_len_chars.
    """
    cond_cfg = getattr(cfg, "cond", None)
    if cond_cfg is None or not bool(getattr(cond_cfg, "enabled", False)):
        return 0

    bits_per = _bits_per_unit(cfg)

    n_units = getattr(cond_cfg, "cond_len_tokens", None)
    if n_units is None:
        n_units = int(getattr(cond_cfg, "cond_len_chars", 0))
    else:
        n_units = int(n_units)

    cL = int(n_units * bits_per)
    return max(0, min(int(cL), int(seq_len_bits)))


def _sample_cond_len_bits_per_example(cfg, B: int, seq_len_bits: int, device) -> torch.Tensor:
    """
    Returns cL_bits per example: [B] int64, in [0, seq_len_bits].
    If cfg.cond.sample_prompt_len=False (default), returns fixed length repeated.
    """
    cond_cfg = getattr(cfg, "cond", None)
    if cond_cfg is None or not bool(getattr(cond_cfg, "enabled", False)):
        return torch.zeros(B, device=device, dtype=torch.long)

    sample_len = bool(getattr(cond_cfg, "sample_prompt_len", False))
    if not sample_len:
        cL = _cond_len_bits_fixed(cfg, seq_len_bits)
        return torch.full((B,), int(cL), device=device, dtype=torch.long)

    bits_per = _bits_per_unit(cfg)

    # min/max in units (tokens or chars)
    mn = getattr(cond_cfg, "cond_len_tokens_min", None)
    mx = getattr(cond_cfg, "cond_len_tokens_max", None)
    if mn is None or mx is None:
        # fallback to legacy names
        mn = int(getattr(cond_cfg, "cond_len_chars_min", 0))
        mx = int(getattr(cond_cfg, "cond_len_chars_max", 0))
    else:
        mn = int(mn)
        mx = int(mx)

    mn = max(0, mn)
    mx = max(mn, mx)

    # uniform integer in [mn, mx]
    if mx == mn:
        units = torch.full((B,), mn, device=device, dtype=torch.long)
    else:
        units = torch.randint(low=mn, high=mx + 1, size=(B,), device=device, dtype=torch.long)

    cL_bits = units * int(bits_per)
    cL_bits = torch.clamp(cL_bits, min=0, max=int(seq_len_bits)).to(torch.long)
    return cL_bits


def _make_prefix_mask_from_lengths(cL_bits: torch.Tensor, S: int) -> torch.Tensor:
    """
    cL_bits: [B] long
    returns prefix_mask: [B,S] bool where True indicates "prefix / conditioned" positions.
    """
    B = int(cL_bits.numel())
    ar = torch.arange(S, device=cL_bits.device).view(1, S).expand(B, S)
    return ar < cL_bits.view(B, 1)


def _make_null_value(cfg, device, dtype) -> torch.Tensor:
    cond_cfg = getattr(cfg, "cond", None)
    strategy = str(getattr(cond_cfg, "null_strategy", "half")) if cond_cfg is not None else "half"
    if strategy == "half":
        return torch.tensor(0.5, device=device, dtype=dtype)
    if strategy == "zeros":
        return torch.tensor(0.0, device=device, dtype=dtype)
    if strategy == "random":
        # NOTE: random prefix is sampled per position later; here return sentinel
        return torch.tensor(float("nan"), device=device, dtype=dtype)
    raise ValueError(f"Unknown cfg.cond.null_strategy={strategy}")


def _make_null_prefix_full(x0_full: torch.Tensor, prefix_mask: torch.Tensor, cfg) -> torch.Tensor:
    """
    x0_full:    [B,S] float
    prefix_mask:[B,S] bool
    returns:    [B,S] float, with null value in prefix positions (else unchanged).
    """
    B, S = x0_full.shape
    cond_cfg = getattr(cfg, "cond", None)
    strategy = str(getattr(cond_cfg, "null_strategy", "half")) if cond_cfg is not None else "half"

    if strategy == "random":
        # Bernoulli(0.5) on prefix positions; elsewhere keep original (unused)
        rnd = torch.bernoulli(torch.full((B, S), 0.5, device=x0_full.device, dtype=x0_full.dtype))
        out = x0_full.clone()
        out[prefix_mask] = rnd[prefix_mask]
        return out

    null_val = _make_null_value(cfg, x0_full.device, x0_full.dtype)
    out = x0_full.clone()
    out[prefix_mask] = null_val
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Core utilities (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def _generalized_regularizer(sigmas: torch.Tensor, c: float, n: float) -> torch.Tensor:
    c = float(max(c, 1e-12))
    n = float(n)
    x = (sigmas.clamp_min(1e-12) / c).pow(n)
    return x / (1.0 + x)


def format_pct(p):
    val = p * 100
    if abs(val - round(val)) < 1e-9:
        return f"{val:.0f}%"
    return f"{val:.3f}".rstrip("0").rstrip(".") + "%"


def get_percentile(log_sigmas, cdf, percentile):
    try:
        f = interp1d(cdf, log_sigmas, kind="linear", bounds_error=False, fill_value="extrapolate")
        return float(f(percentile))
    except Exception:
        return float(np.interp(percentile, cdf, log_sigmas))


def find_rightmost_local_min(mse, sigmas):
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    mse_smooth = np.convolve(mse, kernel, mode="same")

    min_indices = argrelextrema(mse_smooth, np.less)[0]
    valid_indices = [idx for idx in min_indices if idx > 2 and idx < len(mse) - 3]

    if not valid_indices:
        best_idx = np.argmin(mse)
        if _rank0():
            print("[Min Detection] No local minima found. Using Global Minimum.")
        return sigmas[best_idx], mse[best_idx]

    best_idx = valid_indices[-1]
    return sigmas[best_idx], mse[best_idx]


def theoretical_ber(sigma):
    from scipy.special import erfc

    s = np.maximum(sigma, 1e-12)
    return 0.5 * erfc(0.5 / (s * np.sqrt(2)))


def generate_plot(
    sigmas,
    mse,
    ber,
    out_path,
    title_suffix,
    reg_c,
    reg_n,
    power,
    onset_pct,
    sat_pct,
    sigma_min_mark=None,
    user_sigma_min_mark=None, # <-- NEW ARGUMENT
):
    log_sigmas = np.log10(sigmas)

    # 1. Compute PDF/CDF for this specific range
    rate = mse / (sigmas**2)
    reg = _generalized_regularizer(torch.from_numpy(sigmas), c=reg_c, n=reg_n).numpy()
    unnormalized_pdf = reg * (rate**power)

    if np.sum(unnormalized_pdf) == 0:
        pdf = np.ones_like(unnormalized_pdf) / len(unnormalized_pdf)
    else:
        pdf = unnormalized_pdf / np.sum(unnormalized_pdf)

    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]

    # 2. Find Percentiles
    onset_pt = get_percentile(log_sigmas, cdf, onset_pct)
    sat_pt = get_percentile(log_sigmas, cdf, sat_pct)

    peak_idx = np.argmax(pdf)
    peak_pt = log_sigmas[peak_idx]

    onset_sigma = 10**onset_pt
    sat_sigma = 10**sat_pt

    theory_ber_curve = theoretical_ber(sigmas)

    # --- PLOTTING ---
    plt.close("all")
    plt.figure(figsize=(10, 14), dpi=150)
    grid_kwargs = dict(which="major", linestyle="-", alpha=0.3)
    minor_grid_kwargs = dict(which="minor", linestyle=":", alpha=0.1)

    # Plot 1: PDF
    ax1 = plt.subplot(3, 1, 1)
    plt.grid(True, **grid_kwargs)
    plt.grid(True, **minor_grid_kwargs)
    color_pdf = "#2ca02c"

    plt.plot(log_sigmas, pdf, label="Implied Entropy PDF", color=color_pdf, lw=2)
    plt.axvline(x=onset_pt, color=color_pdf, linestyle=":", linewidth=1.5, alpha=0.6)
    plt.scatter(onset_pt, 0, color=color_pdf, s=40, marker="^", zorder=10, clip_on=False)
    plt.axvline(x=sat_pt, color=color_pdf, linestyle="--", linewidth=1.5, alpha=0.6)
    plt.scatter(sat_pt, 0, color=color_pdf, s=40, marker="v", zorder=10, clip_on=False)

    plt.ylabel("Probability Density")
    plt.title(f"Entropic Profile {title_suffix}\n(Onset {format_pct(onset_pct)}, Sat {format_pct(sat_pct)})")
    ax1.legend(loc="upper right")

    # Plot 2: MSE
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    plt.grid(True, **grid_kwargs)
    plt.grid(True, **minor_grid_kwargs)
    color_mse = "#1f77b4"

    plt.plot(log_sigmas, mse, label="MSE (Prob Space)", color=color_mse, lw=2)
    ax2.set_yscale("log")

    if sigma_min_mark is not None:
        min_log = math.log10(sigma_min_mark)
        plt.axvline(x=min_log, color="red", linestyle="-", linewidth=1.5, label="Adaptive Min")
        
    # <-- NEW PLOTTING LOGIC FOR USER SIGMA MIN
    if user_sigma_min_mark is not None:
        user_log = math.log10(user_sigma_min_mark)
        plt.axvline(x=user_log, color="purple", linestyle="-.", linewidth=1.5, label=f"User Min ($\sigma$={user_sigma_min_mark})")

    plt.axvline(x=onset_pt, color=color_pdf, linestyle=":", alpha=0.4)
    plt.axvline(x=sat_pt, color=color_pdf, linestyle="--", alpha=0.4)

    plt.ylabel("MSE (Log Scale)")
    plt.title("Denoising Error (MSE)")
    ax2.legend(loc="upper left")

    # Plot 3: BER
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    plt.grid(True, **grid_kwargs)
    plt.grid(True, **minor_grid_kwargs)

    floor = 1e-7
    ber_safe = np.maximum(ber, floor)
    theory_safe = np.maximum(theory_ber_curve, floor)

    color_model = "#d62728"
    color_theory = "black"

    plt.plot(log_sigmas, ber_safe, label="Empirical BER (NN Model)", color=color_model, lw=2)
    plt.plot(
        log_sigmas,
        theory_safe,
        label="Theoretical BER (Threshold)",
        color=color_theory,
        linestyle="--",
        lw=2,
        alpha=0.7,
    )

    plt.axvline(x=onset_pt, color=color_pdf, linestyle=":", alpha=0.4)
    plt.axvline(x=sat_pt, color=color_pdf, linestyle="--", alpha=0.4)

    if sigma_min_mark is not None:
        plt.axvline(x=math.log10(sigma_min_mark), color="red", linestyle="-", linewidth=1.5, alpha=0.5)
        
    # <-- NEW PLOTTING LOGIC FOR USER SIGMA MIN
    if user_sigma_min_mark is not None:
        plt.axvline(x=math.log10(user_sigma_min_mark), color="purple", linestyle="-.", linewidth=1.5, alpha=0.5)

    ax3.set_yscale("log")
    plt.ylim(floor, 1.1)
    plt.xlabel(r"Log$_{10}$ Sigma ($\sigma$)")
    plt.ylabel("Bit Error Rate (Log Scale)")
    plt.title("Error Rate: Contextual vs Non-Contextual")
    ax3.legend(loc="upper left")

    # Stats Box
    stats_text = (
        f"Peak (Mode): {peak_pt:.3f} (σ={10**peak_pt:.2e})\n"
        f"Onset: {onset_pt:.3f} (σ={onset_sigma:.2e})\n"
        f"Sat:   {sat_pt:.3f} (σ={sat_sigma:.2e})"
    )
    ax1.text(
        0.02,
        0.95,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

    plt.tight_layout()

    if out_path.exists():
        try:
            os.remove(out_path)
        except OSError:
            pass
    plt.savefig(out_path)
    if _rank0():
        print(f"✅ Plot saved to {out_path}")
    plt.close("all")

    return onset_sigma, sat_sigma


# ─────────────────────────────────────────────────────────────────────────────
# Training-aligned construction for (xt, x0_hat, mask)
# ─────────────────────────────────────────────────────────────────────────────
def _build_training_aligned_inputs_for_sigmas(
    cfg,
    model,
    x0: torch.Tensor,                 # [B,S] float
    chunk_sigmas: torch.Tensor,        # [k]  (device)
    use_fp16: bool,
    amp_dtype: torch.dtype,
):
    """
    Replicates training distribution for conditional setup, but for a sigma grid.

    Returns:
      xt     : [B*k, S] float
      x0_rep : [B*k, S] float
      sig_rep: [B*k]    float
      x0_hat : [B*k, S] float
      loss_mask: Optional[[B*k, S] float]  (suffix-only mask, or None)
      prefix_mask_rep: Optional[[B*k, S] bool] (for BER masking; None if not used)
    """
    device = x0.device
    B, S = x0.shape
    k_eff = int(chunk_sigmas.numel())

    # ------------------------------------------------------------
    # Conditional setup (match training)
    # ------------------------------------------------------------
    cond_cfg = getattr(cfg, "cond", None)
    cond_enabled_cfg = (cond_cfg is not None) and bool(getattr(cond_cfg, "enabled", False))

    if not cond_enabled_cfg:
        cond_enabled = False
        prefix_mask = None
        cL_bits = torch.zeros(B, device=device, dtype=torch.long)
    else:
        cL_bits = _sample_cond_len_bits_per_example(cfg, B, S, device=device)  # [B]
        cond_enabled = bool((cL_bits.max().item() if B > 0 else 0) > 0)
        prefix_mask = _make_prefix_mask_from_lengths(cL_bits, S) if cond_enabled else None  # [B,S] bool

    noise_prefix = True
    suffix_only_loss = False
    p_uncond = 0.0

    if cond_enabled:
        noise_prefix = bool(getattr(cond_cfg, "noise_prefix", False))
        suffix_only_loss = bool(getattr(cond_cfg, "loss_on_suffix_only", True))
        p_uncond = float(getattr(cond_cfg, "p_uncond", 0.0))
        p_uncond = max(0.0, min(1.0, p_uncond))

    # ------------------------------------------------------------
    # Expand data across sigmas (same example, multiple sigmas)
    # ------------------------------------------------------------
    x0_rep = x0.repeat_interleave(k_eff, dim=0)          # [B*k, S]
    sig_rep = chunk_sigmas.repeat(B)                    # [B*k]

    # ------------------------------------------------------------
    # Construct xt exactly like training
    #   - If cond_enabled and noise_prefix=False: clean prefix + noise suffix
    #   - else: noise entire sequence
    # ------------------------------------------------------------
    if cond_enabled and (not noise_prefix):
        assert prefix_mask is not None
        drop_mask = (torch.rand(B, device=device) < p_uncond)  # [B] bool

        prefix_used_full = x0.clone()  # [B,S]
        null_full = _make_null_prefix_full(x0, prefix_mask, cfg)  # [B,S]
        if drop_mask.any():
            dm = drop_mask.view(B, 1).expand(B, S)
            replace = dm & prefix_mask
            prefix_used_full[replace] = null_full[replace]

        # Replicate prefix structures across sigmas (same per-example prompt/drop)
        prefix_mask_rep = prefix_mask.repeat_interleave(k_eff, dim=0)           # [B*k, S] bool
        prefix_used_rep = prefix_used_full.repeat_interleave(k_eff, dim=0)      # [B*k, S] float

        # Noise full then overwrite prefix (equivalent to "noise only suffix")
        eps = torch.randn_like(x0_rep)
        xt = x0_rep + sig_rep.view(-1, 1) * eps
        xt[prefix_mask_rep] = prefix_used_rep[prefix_mask_rep]

        # Loss mask: suffix-only positions
        loss_mask = None
        if suffix_only_loss:
            loss_mask = (~prefix_mask_rep).to(dtype=torch.float32)  # [B*k, S]

    else:
        # unconditional or "noise_prefix=True" mode (old behavior): noise entire sequence
        eps = torch.randn_like(x0_rep)
        xt = x0_rep + sig_rep.view(-1, 1) * eps
        prefix_mask_rep = None
        prefix_used_rep = None
        loss_mask = None

    # ------------------------------------------------------------
    # Self-conditioning (match training; per-example coinflip)
    # In analysis, each (example, sigma) pair is a distinct sample,
    # so we sample sc_mask over the replicated batch.
    # ------------------------------------------------------------
    sc_enabled = bool(getattr(cfg.model, "self_condition", False))
    p_sc = float(getattr(cfg.train, "self_condition_prob", 0.5))
    x0_hat = torch.zeros_like(xt)  # [B*k, S]

    if sc_enabled and p_sc > 0.0:
        sc_mask = (torch.rand(xt.size(0), device=device) < p_sc)  # [B*k] bool
        if sc_mask.any():
            xt_sc = xt[sc_mask]
            sigma_sc = sig_rep[sc_mask]
            x0_hat_sc = torch.zeros_like(xt_sc)

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_fp16, dtype=amp_dtype):
                    # Training uses model(xt, sigma, x0_hat)
                    try:
                        logits_sc = model(xt_sc, sigma_sc, x0_hat_sc)
                    except TypeError:
                        # Backward compat: some models accept only (xt, sigma)
                        logits_sc = model(xt_sc, sigma_sc)

                if logits_sc.dim() == 3 and logits_sc.size(-1) == 1:
                    logits_sc = logits_sc.squeeze(-1)  # [Bsc,S]

                x0_hat_all = torch.sigmoid(logits_sc.float()).to(dtype=xt.dtype)  # [Bsc,S]

            x0_hat[sc_mask] = x0_hat_all

    # IMPORTANT: in clean-prefix conditional mode, clamp SC prefix like training
    if cond_enabled and (not noise_prefix):
        assert prefix_mask_rep is not None and prefix_used_rep is not None
        x0_hat[prefix_mask_rep] = prefix_used_rep[prefix_mask_rep]

    return xt, x0_rep, sig_rep, x0_hat, loss_mask, prefix_mask_rep


def analyze_sigma_profile(
    cfg,
    model,
    loader,
    device,
    sigma_min=1e-4,
    sigma_max=200.0,
    steps=200,
    max_batches=None,
    out_dir=None,
    reg_c=0.1,
    reg_n=3.0,
    power=1.0,
    onset_pct=DEFAULT_ONSET_PERCENTILE,
    sat_pct=DEFAULT_SATURATION_PERCENTILE,
    max_effective_bs=512,
    denoising_sigma_min=None, # <-- NEW ARGUMENT
):
    if _rank0():
        print(f"Running Sigma Analysis: [{sigma_min:.1e}, {sigma_max:.1e}] over {steps} points.")

    # Match training AMP knobs (cfg.train.*)
    use_fp16 = bool(getattr(cfg.train, "use_fp16", False))
    amp_dtype_str = str(getattr(cfg.train, "amp_dtype", "float16")).lower()
    amp_dtype = torch.float16
    if amp_dtype_str == "bf16" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    if _rank0():
        print(f"AMP Enabled: {use_fp16} ({amp_dtype})")

    log_min = math.log10(sigma_min)
    log_max = math.log10(sigma_max)
    log_sigmas = torch.linspace(log_min, log_max, steps, device=device)
    sigmas = 10**log_sigmas  # [steps]

    # -------------------------------------------------------------------------
    # DDP SPEEDUP STRATEGY (NO SEMANTIC CHANGE):
    #   - Each rank gets a disjoint slice of sigma indices.
    #   - Each rank iterates the SAME dataset batches, but only evaluates its sigma slice.
    #   - We all_gather the per-rank accumulators and reconstruct full-length arrays.
    #   - Plotting is rank0 only.
    # -------------------------------------------------------------------------
    world = _world()
    rank = _rank()
    idx_start = (steps * rank) // world
    idx_end = (steps * (rank + 1)) // world
    local_steps = max(0, idx_end - idx_start)
    if local_steps <= 0:
        raise RuntimeError(f"Rank {rank} got empty sigma slice: [{idx_start}, {idx_end}) with steps={steps}")

    local_sigmas = sigmas[idx_start:idx_end]  # [local_steps]

    mse_accum_local = torch.zeros(local_steps, device=device, dtype=torch.float32)
    ber_accum_local = torch.zeros(local_steps, device=device, dtype=torch.float32)
    count_accum_local = 0

    model.eval()
    total_batches = len(loader) if hasattr(loader, "__len__") else None

    iterable = _tqdm_if_rank0(loader, desc="Analyzing Dataset", total=total_batches)
    for i, batch in enumerate(iterable):
        if max_batches is not None and i >= max_batches:
            break

        if isinstance(batch, (tuple, list)):
            batch = batch[0]

        x0_full = batch.to(device)
        if x0_full.dim() > 2:
            x0_full = x0_full.view(x0_full.size(0), -1)
        x0_full = x0_full.float()  # [B,S]

        full_batch_size = x0_full.size(0)
        B_chunk = min(full_batch_size, max_effective_bs)

        for start_idx in range(0, full_batch_size, B_chunk):
            end_idx = min(start_idx + B_chunk, full_batch_size)
            x0 = x0_full[start_idx:end_idx]  # [B_sub,S]
            B_sub = x0.size(0)
            S_len = x0.size(1)

            # Keep your original effective batch scheduling
            sigma_chunk_size = max(1, max_effective_bs // B_sub)

            # Iterate over LOCAL sigma indices only
            for s_local in range(0, local_steps, sigma_chunk_size):
                s_end = min(s_local + sigma_chunk_size, local_steps)
                chunk_sigmas = local_sigmas[s_local:s_end]  # [k_eff]
                k_eff = int(chunk_sigmas.numel())

                # ---- Build (xt, x0_hat, loss_mask) to match training distribution ----
                xt, x0_rep, sig_rep, x0_hat, loss_mask, prefix_mask_rep = _build_training_aligned_inputs_for_sigmas(
                    cfg=cfg,
                    model=model,
                    x0=x0,
                    chunk_sigmas=chunk_sigmas,
                    use_fp16=use_fp16,
                    amp_dtype=amp_dtype,
                )

                # ---- Forward (match training signature) ----
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=use_fp16, dtype=amp_dtype):
                        try:
                            logits = model(xt, sig_rep, x0_hat)  # training signature
                        except TypeError:
                            logits = model(xt, sig_rep)          # backward compat

                        if logits.dim() == 3 and logits.size(-1) == 1:
                            logits = logits.squeeze(-1)  # [B*k,S]

                        # ---- Entropy metric / MSE aligned with training loss mask ----
                        _, entropy_metric = binary_score_interpolation_loss(
                            logits,
                            x0_rep,
                            sig_rep,
                            cfg,
                            return_entropy_metric=True,
                            mask=loss_mask,  # suffix-only if training used it
                        )  # entropy_metric: [B*k]

                        # ---- BER aligned with the same positions as training supervision ----
                        preds = (logits > 0).to(dtype=x0_rep.dtype)  # [B*k,S]
                        errors = (preds != x0_rep).to(dtype=torch.float32)  # [B*k,S]

                        errors_reshaped = errors.view(B_sub, k_eff, S_len)  # [B,k,S]

                        if loss_mask is not None:
                            # loss_mask is [B*k, S] float; reshape and apply
                            m = loss_mask.view(B_sub, k_eff, S_len)  # [B,k,S]
                            num = (errors_reshaped * m).sum(dim=2)  # [B,k]
                            den = m.sum(dim=2).clamp_min(1.0)       # [B,k]
                            batch_ber = num / den                  # [B,k]
                        else:
                            batch_ber = errors_reshaped.mean(dim=2)  # [B,k]

                ent_reshaped = entropy_metric.view(B_sub, k_eff)  # [B,k]

                # Accumulate sums over examples for each sigma slot
                mse_accum_local[s_local:s_end] += ent_reshaped.sum(dim=0).float()
                ber_accum_local[s_local:s_end] += batch_ber.sum(dim=0).float()

            count_accum_local += B_sub

    # -------------------------------------------------------------------------
    # DDP aggregation
    # -------------------------------------------------------------------------
    if _ddp_is_on():
        # counts should match across ranks because everyone sees same data
        cnt = torch.tensor([int(count_accum_local)], device=device, dtype=torch.long)
        cnt_all = [torch.zeros_like(cnt) for _ in range(world)]
        dist.all_gather(cnt_all, cnt)

        if _rank0():
            counts = [int(t.item()) for t in cnt_all]
            if len(set(counts)) != 1:
                raise RuntimeError(f"DDP count mismatch across ranks: {counts}")
            count_accum = counts[0]
        else:
            count_accum = int(cnt_all[0].item())

        # gather variable-length slices via all_gather_object
        mse_cpu = mse_accum_local.detach().cpu()
        ber_cpu = ber_accum_local.detach().cpu()
        gathered = [None for _ in range(world)]
        dist.all_gather_object(gathered, (idx_start, idx_end, mse_cpu, ber_cpu))

        if _rank0():
            mse_full = np.zeros((steps,), dtype=np.float64)
            ber_full = np.zeros((steps,), dtype=np.float64)
            for (s0, s1, mse_part, ber_part) in gathered:
                mse_full[s0:s1] = np.asarray(mse_part, dtype=np.float64)
                ber_full[s0:s1] = np.asarray(ber_part, dtype=np.float64)
        else:
            mse_full = None
            ber_full = None

        # broadcast reconstructed arrays from rank0 so return is consistent everywhere
        if not _rank0():
            mse_full = np.empty((steps,), dtype=np.float64)
            ber_full = np.empty((steps,), dtype=np.float64)

        mse_t = torch.from_numpy(mse_full).to(device=device, dtype=torch.float64)
        ber_t = torch.from_numpy(ber_full).to(device=device, dtype=torch.float64)
        dist.broadcast(mse_t, src=0)
        dist.broadcast(ber_t, src=0)

        mse_full = mse_t.cpu().numpy()
        ber_full = ber_t.cpu().numpy()

    else:
        count_accum = count_accum_local
        mse_full = np.zeros((steps,), dtype=np.float64)
        ber_full = np.zeros((steps,), dtype=np.float64)
        mse_full[idx_start:idx_end] = mse_accum_local.detach().cpu().numpy()
        ber_full[idx_start:idx_end] = ber_accum_local.detach().cpu().numpy()

    # Final averages (mean over examples, per sigma)
    avg_mse_full = (mse_full / float(count_accum)).astype(np.float64)
    avg_ber_full = (ber_full / float(count_accum)).astype(np.float64)
    sigmas_full = sigmas.detach().cpu().numpy()

    # ─────────────────────────────────────────────────────────────────────────
    # PLOTTING + ADAPTIVE LOGIC (rank0 only for plots/prints)
    # ─────────────────────────────────────────────────────────────────────────
    if out_dir and _rank0():
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_data_path = out_dir / "analysis_raw_metrics.npz"
        np.savez(
            raw_data_path,
            sigmas=sigmas_full,
            mse=avg_mse_full,
            ber=avg_ber_full,
            count=count_accum
        )
        print(f"\n[Export] Saved raw metrics to: {raw_data_path}")
        
        print("\n[Plotting] Generating Full Range Plot...")
        generate_plot(
            sigmas_full,
            avg_mse_full,
            avg_ber_full,
            out_path=out_dir / "entropy_analysis_full_range.png",
            title_suffix="(Full Range)",
            reg_c=reg_c,
            reg_n=reg_n,
            power=power,
            onset_pct=onset_pct,
            sat_pct=sat_pct,
            sigma_min_mark=None,
            user_sigma_min_mark=denoising_sigma_min, # <-- PASSING IT IN
        )

    sigma_min_new, min_mse_val = find_rightmost_local_min(avg_mse_full, sigmas_full)
    if _rank0():
        print(f"\n[Adaptive] Detected Rightmost Minimum: sigma={sigma_min_new:.6f} (MSE={min_mse_val:.6f})")

    valid_mask = sigmas_full >= sigma_min_new
    sigmas_trunc = sigmas_full[valid_mask]
    mse_trunc = avg_mse_full[valid_mask]
    ber_trunc = avg_ber_full[valid_mask]

    if _rank0():
        print(f"[Adaptive] Truncated range: [{sigmas_trunc[0]:.6f}, {sigmas_trunc[-1]:.6f}]")

    onset_sigma = 0.0
    sat_sigma = 0.0

    if out_dir and _rank0():
        print("[Plotting] Generating Truncated Range Plot...")
        onset_sigma, sat_sigma = generate_plot(
            sigmas_trunc,
            mse_trunc,
            ber_trunc,
            out_path=out_dir / "entropy_analysis_adaptive.png",
            title_suffix="(Adaptive Truncation)",
            reg_c=reg_c,
            reg_n=reg_n,
            power=power,
            onset_pct=onset_pct,
            sat_pct=sat_pct,
            sigma_min_mark=sigma_min_new,
            user_sigma_min_mark=denoising_sigma_min, # <-- PASSING IT IN HERE TOO
        )

    # Broadcast key scalars to all ranks (so return dict matches everywhere)
    if _ddp_is_on():
        scalars = torch.tensor([sigma_min_new, onset_sigma, sat_sigma], device=device, dtype=torch.float64)
        dist.broadcast(scalars, src=0)
        sigma_min_new, onset_sigma, sat_sigma = [float(x) for x in scalars.tolist()]

    return {"adaptive_min_sigma": sigma_min_new, "onset_sigma": onset_sigma, "sat_sigma": sat_sigma}


def main():
    parser = argparse.ArgumentParser(description="Analyze Sigma Profile for Continuous Diffusion (training-aligned)")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path override")

    parser.add_argument("--max_effective_bs", type=int, default=512, help="Max batch size passed to the model per forward pass")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit number of batches")

    parser.add_argument("--sigma_min", type=float, default=2e-4)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--steps", type=int, default=200)

    parser.add_argument("--reg_c", type=float, default=0.1)
    parser.add_argument("--reg_n", type=float, default=3.0)
    parser.add_argument("--power", type=float, default=1.0, help="Entropy Power")

    parser.add_argument("--onset_pct", type=float, default=DEFAULT_ONSET_PERCENTILE)
    parser.add_argument("--sat_pct", type=float, default=DEFAULT_SATURATION_PERCENTILE)
    
    # <-- NEW ARGUMENT EXPOSED TO USER
    parser.add_argument(
        "--denoising_sigma_min", 
        type=float, 
        default=None, 
        help="Optional user-specified sigma_min used in reverse denoising to plot as a vertical line."
    )

    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Enable multi-GPU parallelism on a single node (launch with torchrun). "
        "Work is split across sigma grid; plots are produced on rank0 only.",
    )

    args = parser.parse_args()

    _setup_ddp_if_needed(args.ddp)

    cfg = load_config(args.config)

    # Respect LOCAL_RANK when ddp is enabled; otherwise fall back to cfg.device
    if args.ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(cfg.device)

    ckpt_path = args.ckpt or getattr(cfg.evaluation, "checkpoint_path", None)
    if not ckpt_path:
        ckpt_path = f"runs/{cfg.experiment}/checkpoints/last.pt"

    if _rank0():
        print(f"Loading model from {ckpt_path}...")

    model = create_model(cfg).to(device)
    ema = EMA(unwrap_all(model), decay=0.0)
    load_checkpoint(model, ema, Path(ckpt_path), device, apply_ema=True)
    model.eval()

    # Keep loader behavior unchanged (DDP speedup is sigma-splitting, not data-sharding)
    split = "val" if hasattr(cfg.data, "val_fraction") else "test"
    loader = get_loader(cfg, split=split)

    out_dir = Path(f"runs/{cfg.experiment}/analysis")

    results = analyze_sigma_profile(
        cfg,
        model,
        loader,
        device,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        steps=args.steps,
        max_batches=args.max_batches,
        out_dir=out_dir,
        reg_c=args.reg_c,
        reg_n=args.reg_n,
        power=args.power,
        onset_pct=args.onset_pct,
        sat_pct=args.sat_pct,
        max_effective_bs=args.max_effective_bs,
        denoising_sigma_min=args.denoising_sigma_min, # <-- PASSING IT DOWN
    )

    if _rank0():
        print("-" * 60)
        print("SUGGESTED PARAMETERS")
        print("-" * 60)
        print(f"Adaptive Sigma Min (MSE Floor): {results['adaptive_min_sigma']:.6f}")
        print(f"Recalculated Sigma Min (Onset): {results['onset_sigma']:.6f}")
        print(f"Recalculated Sigma Max (Sat):   {results['sat_sigma']:.6f}")
        print("-" * 60)

    _cleanup_ddp()


if __name__ == "__main__":
    main()