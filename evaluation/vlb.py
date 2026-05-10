from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None

from diffusion.continuous.logit_postprocess import _model_logits_continuous


# -----------------------------------------------------------------------------
# Helpers for truncated log-normal importance sampling over y = log(sigma)
# -----------------------------------------------------------------------------
def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _trunc_normal_logpdf(y: torch.Tensor, *, mu: float, std: float, a: float, b: float) -> torch.Tensor:
    """
    log q(y) for y ~ Normal(mu,std^2) truncated to [a,b], where y = log(sigma).
    """
    mu_t = torch.tensor(mu, device=y.device, dtype=y.dtype)
    std_t = torch.tensor(std, device=y.device, dtype=y.dtype)

    z = (y - mu_t) / std_t
    log_phi = -0.5 * z * z - math.log(math.sqrt(2.0 * math.pi))
    log_pdf = log_phi - torch.log(std_t)

    a_t = torch.tensor(a, device=y.device, dtype=y.dtype)
    b_t = torch.tensor(b, device=y.device, dtype=y.dtype)
    za = (a_t - mu_t) / std_t
    zb = (b_t - mu_t) / std_t

    Z = (_normal_cdf(zb) - _normal_cdf(za)).clamp_min(1e-12)
    return log_pdf - torch.log(Z)


def _ddp_is_on() -> bool:
    return (dist is not None) and dist.is_available() and dist.is_initialized()


def _is_master() -> bool:
    if not _ddp_is_on():
        return True
    try:
        return dist.get_rank() == 0
    except Exception:
        return True


# -----------------------------------------------------------------------------
# Representation helpers
# -----------------------------------------------------------------------------
def _repr_mode(cfg) -> str:
    return str(getattr(getattr(cfg, "data", object()), "representation", "binary")).lower().strip()


def _is_token_mode(cfg) -> bool:
    return _repr_mode(cfg) == "tokens"


def _is_binary_mode(cfg) -> bool:
    return _repr_mode(cfg) == "binary"


def _token_vocab_size(cfg) -> int:
    v = int(getattr(getattr(cfg, "data", object()), "vocab_size", 0))
    if v <= 0:
        raise ValueError("Token-mode VLB requires cfg.data.vocab_size > 0")
    return v


def _cond_len_positions(cfg, S: int) -> int:
    """
    Return conditional prefix length in the native sequence dimension of the current
    representation:
      - binary mode: number of bit positions
      - token mode: number of token positions
    """
    cond_cfg = getattr(cfg, "cond", None)
    if cond_cfg is None or not bool(getattr(cond_cfg, "enabled", False)):
        return 0

    if _is_token_mode(cfg):
        cond_len_tokens = getattr(cond_cfg, "cond_len_tokens", None)
        if cond_len_tokens is not None:
            return max(0, min(int(S), int(cond_len_tokens)))

        cond_len_chars = getattr(cond_cfg, "cond_len_chars", None)
        if cond_len_chars is not None:
            return max(0, min(int(S), int(cond_len_chars)))

        return 0

    bits_per_token = getattr(getattr(cfg, "data", object()), "bits_per_token", None)
    if bits_per_token is not None:
        cond_len_tokens = getattr(cond_cfg, "cond_len_tokens", None)
        if cond_len_tokens is not None:
            cL = int(cond_len_tokens) * int(bits_per_token)
            return max(0, min(int(S), int(cL)))

    bpc = int(getattr(getattr(cfg, "data", object()), "bits_per_char", 0) or 0)
    clen_chars = int(getattr(cond_cfg, "cond_len_chars", 0) or 0)
    cL = int(clen_chars * bpc) if (clen_chars > 0 and bpc > 0) else 0
    return max(0, min(int(S), int(cL)))


def _make_null_prefix(prefix: torch.Tensor, cfg) -> torch.Tensor:
    cond_cfg = getattr(cfg, "cond", None)
    strategy = str(getattr(cond_cfg, "null_strategy", "half")) if cond_cfg is not None else "half"
    strategy = strategy.lower().strip()

    if strategy == "half":
        return torch.full_like(prefix, 0.5)
    if strategy == "zeros":
        return torch.zeros_like(prefix)
    if strategy == "random":
        return torch.bernoulli(torch.full_like(prefix, 0.5))
    if strategy == "data_center":
        data_center = float(getattr(getattr(cfg.diffusion, "continuous", object()), "data_center", 0.5))
        return torch.full_like(prefix, data_center)

    raise ValueError(f"Unknown cfg.cond.null_strategy={strategy}")


@dataclass
class VLBResult:
    vlb_bpd: float
    recon_bpd: float
    diff_bpd: float
    prior_bpd: float
    sigma_min_eval: float
    sigma_max_eval: float
    K: int
    sigma_sampling: str
    num_examples: int
    S_dim: int
    mode: str


def _qstats(name: str, x: torch.Tensor) -> str:
    x = x.detach().float().flatten()
    qs = torch.quantile(x, torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0], device=x.device))
    vals = [float(v.item()) for v in qs]
    return (
        f"{name} q=[min {vals[0]:.4g}, p1 {vals[1]:.4g}, p5 {vals[2]:.4g}, "
        f"p50 {vals[3]:.4g}, p95 {vals[4]:.4g}, p99 {vals[5]:.4g}, max {vals[6]:.4g}]"
    )


def _bin_means(
    sig: torch.Tensor,
    loss_sum: torch.Tensor,
    ratio: torch.Tensor,
    *,
    smin: float,
    smax: float,
    nbins: int,
    label: str,
) -> str:
    nb = max(1, int(nbins))
    edges = torch.logspace(math.log10(smin), math.log10(smax), steps=nb + 1, device=sig.device)
    out = []
    for i in range(nb):
        lo, hi = edges[i], edges[i + 1]
        m = (sig >= lo) & (sig < hi) if i < nb - 1 else (sig >= lo) & (sig <= hi)
        if m.any():
            out.append(
                f"  bin[{i}] sigma∈[{float(lo):.3g},{float(hi):.3g}]: "
                f"{label}={float(loss_sum[m].mean().item()):.3g}  {label}/s^2={float(ratio[m].mean().item()):.3g}"
            )
        else:
            out.append(f"  bin[{i}] sigma∈[{float(lo):.3g},{float(hi):.3g}]: (no samples)")
    return "\n".join(out)


# -----------------------------------------------------------------------------
# FAST DEBUG HELPERS
# -----------------------------------------------------------------------------
def _dbg_bits(
    name: str,
    x: torch.Tensor,
    *,
    bi: int,
    tol: float = 1e-3,
    max_show: int = 24,
    show_rows: int = 2,
) -> None:
    xt = x.detach()
    B = int(xt.shape[0]) if xt.ndim >= 1 else 1
    xf = xt.float().view(B, -1) if xt.ndim > 1 else xt.float().view(B, 1)

    mn = float(xf.min().item())
    mx = float(xf.max().item())

    frac_lt0 = float((xf < -tol).float().mean().item())
    frac_gt1 = float((xf > 1.0 + tol).float().mean().item())
    dist_to_bit = torch.minimum(xf.abs(), (xf - 1.0).abs())
    frac_not_bit = float((dist_to_bit > tol).float().mean().item())

    sample = xf[: min(B, 2), : min(xf.shape[1], 4096)].reshape(-1).cpu()
    uniq = torch.unique(sample)
    uniq_show = uniq[: min(uniq.numel(), 20)].tolist()

    print(
        f"[VLB-DBG] bi={bi} {name}: shape={tuple(xt.shape)} dtype={xt.dtype} device={xt.device} "
        f"min={mn:.6g} max={mx:.6g} frac<0={frac_lt0:.3g} frac>1={frac_gt1:.3g} frac_not_bit={frac_not_bit:.3g} "
        f"uniq(sample)={uniq_show}"
    )

    xcpu = xf[: min(B, show_rows), : max_show].cpu()
    for i in range(xcpu.shape[0]):
        print(f"[VLB-DBG] bi={bi} {name}[{i}][:{max_show}] = {xcpu[i].tolist()}")


def _dbg_ptr(name: str, x: torch.Tensor, *, bi: int) -> None:
    try:
        storage_ptr = x.untyped_storage().data_ptr()
    except Exception:
        storage_ptr = -1
    print(
        f"[VLB-DBG-PTR] bi={bi} {name}: data_ptr={x.data_ptr()} storage_ptr={storage_ptr} "
        f"stride={tuple(x.stride())} contiguous={x.is_contiguous()}"
    )


# -----------------------------------------------------------------------------
# Representation-aware losses for VLB estimator
# -----------------------------------------------------------------------------
def _recon_term_binary(logits_eval: torch.Tensor, target_eval: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits_eval.float(),
        target_eval.float(),
        reduction="none",
    ).sum(dim=1)


def _recon_term_tokens(logits_eval: torch.Tensor, target_ids_eval: torch.Tensor) -> torch.Tensor:
    B, S, V = logits_eval.shape
    ce = F.cross_entropy(
        logits_eval.float().reshape(B * S, V),
        target_ids_eval.long().reshape(B * S),
        reduction="none",
    ).view(B, S)
    return ce.sum(dim=1)


def _diffusion_loss_sum_binary(probs_eval: torch.Tensor, target_eval: torch.Tensor) -> torch.Tensor:
    return ((probs_eval - target_eval[:, None, :]) ** 2).sum(dim=2)


def _diffusion_loss_sum_tokens(probs_eval: torch.Tensor, target_ids_eval: torch.Tensor) -> torch.Tensor:
    """
    Compute, for each [B,K], the sum over sequence positions of

        ||p - e_c||^2 = sum_v p_v^2 - 2 p_c + 1

    without materializing one-hot targets.

    Inputs:
      probs_eval:      [B, K, S, V]
      target_ids_eval: [B, S]

    Returns:
      loss_sum:        [B, K]
    """
    if probs_eval.dim() != 4:
        raise ValueError(f"Expected probs_eval [B,K,S,V], got {tuple(probs_eval.shape)}")
    if target_ids_eval.dim() != 2:
        raise ValueError(f"Expected target_ids_eval [B,S], got {tuple(target_ids_eval.shape)}")

    B, K, S, V = probs_eval.shape
    if target_ids_eval.shape != (B, S):
        raise ValueError(
            f"target_ids_eval shape {tuple(target_ids_eval.shape)} incompatible with probs_eval {tuple(probs_eval.shape)}"
        )

    sumsq = (probs_eval * probs_eval).sum(dim=-1)
    gather_idx = target_ids_eval.long().unsqueeze(1).unsqueeze(-1).expand(B, K, S, 1)
    p_c = probs_eval.gather(dim=-1, index=gather_idx).squeeze(-1)

    sq_err = sumsq - 2.0 * p_c + 1.0
    return sq_err.sum(dim=2)


@torch.no_grad()
def compute_vlb_over_loader(
    *,
    model: torch.nn.Module,
    cfg,
    data_loader,
    device: torch.device,
    sigma_min_eval: Optional[float] = None,
    sigma_max_eval: Optional[float] = None,
    sigma_sampling: str = "log-uniform",
    num_mc_samples_per_batch: int = 1,
    include_prior: bool = False,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    max_batches: Optional[int] = None,
    progress: bool = True,
    allow_conditional_clean_prefix: bool = True,
    force_unconditional_path: bool = False,
    debug_integrand: bool = False,
    debug_first_n_batches: int = 1,
    debug_num_sigma_bins: int = 6,
    debug_compare_null_prefix: bool = True,
    debug_compare_noise_prefix: bool = True,
    null_prefix_value: float = 0.0,
    null_prefix_mode: str = "constant",
) -> VLBResult:
    del null_prefix_value, null_prefix_mode  # kept for API compatibility

    if data_loader is None:
        raise ValueError("data_loader is None")

    c_cfg = cfg.diffusion.continuous
    sigma_min_cfg = float(c_cfg.sigma_min)
    sigma_max_cfg = float(c_cfg.sigma_max)

    if sigma_min_eval is None:
        sigma_min_eval = float(
            getattr(getattr(cfg.train, "vlb", object()), "sigma_min_eval", None) or sigma_min_cfg
        )
    if sigma_max_eval is None:
        sigma_max_eval = float(
            getattr(getattr(cfg.train, "vlb", object()), "sigma_max_eval", None) or sigma_max_cfg
        )

    sigma_min_eval = max(float(sigma_min_eval), 1e-12)
    sigma_max_eval = max(float(sigma_max_eval), sigma_min_eval)

    a = math.log(sigma_min_eval)
    b = math.log(sigma_max_eval)
    ln_range = b - a

    mu = float(getattr(c_cfg, "p_mean", -1.2))
    std = float(getattr(c_cfg, "p_std", 1.2))

    K = max(1, int(num_mc_samples_per_batch))
    sigma_sampling = str(sigma_sampling).lower().strip()

    sum_prior = torch.zeros((), device=device, dtype=torch.float64)
    sum_recon = torch.zeros((), device=device, dtype=torch.float64)
    sum_diff = torch.zeros((), device=device, dtype=torch.float64)
    sum_count = torch.zeros((), device=device, dtype=torch.float64)

    S_dim: Optional[int] = None
    mode_str = "FORCE-UNCOND" if force_unconditional_path else "COND/UNCOND-AUTO"

    token_mode = _is_token_mode(cfg)
    binary_mode = _is_binary_mode(cfg)
    if not (token_mode or binary_mode):
        raise ValueError(f"Unsupported representation for VLB: {_repr_mode(cfg)!r}")

    # Key patch: state tensors use AMP dtype when enabled.
    state_dtype = amp_dtype if use_amp else torch.float32

    V = _token_vocab_size(cfg) if token_mode else None

    model.eval()

    iterator = data_loader
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="VLB", leave=True)
        except Exception:
            pass

    saw_clean_prefix_mode = False

    for bi, batch in enumerate(iterator):
        if max_batches is not None and bi >= int(max_batches):
            break

        x0 = batch[0] if isinstance(batch, (tuple, list)) else batch
        x0 = x0.to(device, non_blocking=True)

        B = int(x0.size(0))
        if B <= 0:
            continue

        if token_mode:
            x0_ids = x0.long().view(B, -1).contiguous()  # [B,S]
            x0_dense = F.one_hot(x0_ids, num_classes=V).to(dtype=state_dtype)  # [B,S,V]
            S_full = int(x0_ids.size(1))
            x0_debug = x0_dense
        else:
            x0_ids = None
            x0_dense = x0.view(B, -1).to(dtype=torch.float32).contiguous()  # [B,S]
            S_full = int(x0_dense.size(1))
            x0_debug = x0_dense

        if debug_integrand and bi < int(debug_first_n_batches):
            _dbg_bits("x0_dense", x0_debug, bi=bi)
            _dbg_ptr("x0_dense", x0_debug, bi=bi)

        if S_dim is None:
            S_dim = S_full

        if force_unconditional_path:
            cL = 0
            clean_prefix_mode = False
            prefix = None
            x0_eval_dense = x0_dense
            x0_eval_ids = x0_ids
        else:
            cL = _cond_len_positions(cfg, S_full)
            cond_cfg = getattr(cfg, "cond", None)
            cond_enabled = (cond_cfg is not None) and bool(getattr(cond_cfg, "enabled", False)) and (cL > 0)
            noise_prefix = bool(getattr(cond_cfg, "noise_prefix", False)) if cond_enabled else True

            clean_prefix_mode = bool(allow_conditional_clean_prefix) and cond_enabled and (not noise_prefix)

            if clean_prefix_mode:
                saw_clean_prefix_mode = True
                prefix = x0_dense[:, :cL].contiguous()
                x0_eval_dense = x0_dense[:, cL:]
                x0_eval_ids = x0_ids[:, cL:] if token_mode else None
            else:
                prefix = None
                x0_eval_dense = x0_dense
                x0_eval_ids = x0_ids

        S_eval = int(x0_eval_dense.size(1))

        if debug_integrand and bi < int(debug_first_n_batches):
            print(
                f"[VLB-DBG] bi={bi} mode={mode_str} repr={_repr_mode(cfg)} "
                f"clean_prefix_mode={clean_prefix_mode} cL={cL} S_full={S_full} S_eval={S_eval}"
            )

        # -----------------------------
        # Prior term (optional)
        # -----------------------------
        if include_prior:
            prior_per_ex = 0.5 * (x0_eval_dense.float() ** 2).reshape(B, -1).sum(dim=1) / (sigma_max_eval**2)
        else:
            prior_per_ex = torch.zeros(B, device=device, dtype=torch.float32)

        # -----------------------------
        # Recon term at sigma_min_eval
        # -----------------------------
        sigma0_vec = torch.full((B,), sigma_min_eval, device=device, dtype=torch.float32)

        if clean_prefix_mode:
            x0_target_dense = x0_dense[:, cL:].contiguous()
            x0_target_ids = x0_ids[:, cL:].contiguous() if token_mode else None

            z0_suffix = (
                x0_target_dense + sigma0_vec.view(-1, 1, 1) * torch.randn_like(x0_target_dense)
                if token_mode
                else x0_target_dense + sigma0_vec.view(-1, 1) * torch.randn_like(x0_target_dense)
            )

            z0_in = x0_dense.clone()
            z0_in[:, cL:] = z0_suffix

            x0_hat0 = torch.zeros_like(z0_in)
            x0_hat0[:, :cL] = prefix
        else:
            x0_target_dense = x0_eval_dense.contiguous().clone()
            x0_target_ids = x0_eval_ids.contiguous().clone() if token_mode else None

            z0_in = (
                x0_target_dense + sigma0_vec.view(-1, 1, 1) * torch.randn_like(x0_target_dense)
                if token_mode
                else x0_target_dense + sigma0_vec.view(-1, 1) * torch.randn_like(x0_target_dense)
            )
            x0_hat0 = torch.zeros_like(z0_in)

        with autocast(enabled=use_amp, dtype=amp_dtype):
            logits0 = _model_logits_continuous(model, cfg, z0_in, sigma0_vec, x0_hat0)
            if binary_mode and logits0.dim() == 3 and logits0.size(-1) == 1:
                logits0 = logits0.squeeze(-1)

        logits0_eval = logits0[:, cL:] if clean_prefix_mode else logits0

        if token_mode:
            recon_per_ex = _recon_term_tokens(logits0_eval, x0_target_ids)
        else:
            recon_per_ex = _recon_term_binary(logits0_eval, x0_target_dense)

        # -----------------------------
        # Diffusion integral MC estimate over log-sigma
        # -----------------------------
        if ln_range <= 0:
            diff_per_ex = torch.zeros(B, device=device, dtype=torch.float32)
            sigma = torch.full((B, K), sigma_min_eval, device=device, dtype=torch.float32)
            inv_qy = torch.full((B, K), ln_range, device=device, dtype=torch.float32)
            loss_sum = torch.zeros((B, K), device=device, dtype=torch.float32)
            ratio = torch.zeros((B, K), device=device, dtype=torch.float32)
        else:
            if sigma_sampling in {"loguniform", "log-uniform", "uniform"}:
                y = torch.rand(B, K, device=device, dtype=torch.float32) * ln_range + a
                inv_qy = torch.full((B, K), ln_range, device=device, dtype=torch.float32)
            elif sigma_sampling in {"lognormal", "log-normal", "edm"}:
                y = torch.empty(B, K, device=device, dtype=torch.float32)
                mask = torch.ones(B, K, device=device, dtype=torch.bool)
                max_rounds = 128
                rounds = 0
                while mask.any() and rounds < max_rounds:
                    nmask = int(mask.sum().item())
                    y_prop = mu + std * torch.randn(nmask, device=device, dtype=torch.float32)
                    ok = (y_prop >= a) & (y_prop <= b)
                    idx = mask.nonzero(as_tuple=False)
                    if ok.any():
                        y[idx[ok, 0], idx[ok, 1]] = y_prop[ok]
                        mask[idx[ok, 0], idx[ok, 1]] = False
                    rounds += 1
                if mask.any():
                    idx = mask.nonzero(as_tuple=False)
                    y[idx[:, 0], idx[:, 1]] = torch.empty(
                        idx.size(0), device=device, dtype=torch.float32
                    ).uniform_(a, b)
                    mask[idx[:, 0], idx[:, 1]] = False

                log_qy = _trunc_normal_logpdf(y, mu=mu, std=std, a=a, b=b)
                inv_qy = torch.exp(-log_qy).clamp_max(1e12)
            else:
                raise ValueError(f"Unknown sigma_sampling='{sigma_sampling}'")

            sigma = torch.exp(y)

            if clean_prefix_mode:
                x0_for_loss_dense = x0_dense[:, cL:].contiguous()
                x0_for_loss_ids = x0_ids[:, cL:].contiguous() if token_mode else None

                if token_mode:
                    noise = torch.randn(B, K, S_eval, V, device=device, dtype=x0_for_loss_dense.dtype)
                    zt_suffix = x0_for_loss_dense[:, None, :, :] + sigma[:, :, None, None] * noise
                    zt = x0_dense[:, None, :, :].expand(B, K, S_full, V).contiguous()
                    zt[:, :, cL:, :] = zt_suffix
                else:
                    noise = torch.randn(B, K, S_eval, device=device, dtype=x0_for_loss_dense.dtype)
                    zt_suffix = x0_for_loss_dense[:, None, :] + sigma[:, :, None] * noise
                    zt = x0_dense[:, None, :].expand(B, K, S_full).contiguous()
                    zt[:, :, cL:] = zt_suffix
            else:
                x0_for_loss_dense = x0_eval_dense.contiguous().clone()
                x0_for_loss_ids = x0_eval_ids.contiguous().clone() if token_mode else None

                if token_mode:
                    noise = torch.randn(B, K, S_eval, V, device=device, dtype=x0_for_loss_dense.dtype)
                    zt = x0_for_loss_dense[:, None, :, :] + sigma[:, :, None, None] * noise
                else:
                    noise = torch.randn(B, K, S_eval, device=device, dtype=x0_for_loss_dense.dtype)
                    zt = x0_for_loss_dense[:, None, :] + sigma[:, :, None] * noise

            if token_mode:
                zt_flat = zt.reshape(B * K, S_full if clean_prefix_mode else S_eval, V)
            else:
                zt_flat = zt.reshape(B * K, S_full if clean_prefix_mode else S_eval)

            sigma_flat = sigma.reshape(B * K)

            x0_hat_flat = torch.zeros_like(zt_flat)
            if clean_prefix_mode:
                if token_mode:
                    prefix_rep = (
                        prefix[:, None, :, :]
                        .expand(B, K, cL, V)
                        .reshape(B * K, cL, V)
                        .contiguous()
                    )
                    x0_hat_flat[:, :cL, :] = prefix_rep
                else:
                    prefix_rep = (
                        prefix[:, None, :]
                        .expand(B, K, cL)
                        .reshape(B * K, cL)
                        .contiguous()
                    )
                    x0_hat_flat[:, :cL] = prefix_rep

            with autocast(enabled=use_amp, dtype=amp_dtype):
                logits = _model_logits_continuous(model, cfg, zt_flat, sigma_flat, x0_hat_flat)
                if binary_mode and logits.dim() == 3 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)

            if token_mode:
                probs = torch.softmax(logits.float(), dim=-1).view(B, K, -1, V)
                probs_eval = probs[:, :, cL:, :] if clean_prefix_mode else probs
                loss_sum = _diffusion_loss_sum_tokens(probs_eval, x0_for_loss_ids)
            else:
                probs = torch.sigmoid(logits.float()).view(B, K, -1)
                probs_eval = probs[:, :, cL:] if clean_prefix_mode else probs
                loss_sum = _diffusion_loss_sum_binary(probs_eval, x0_for_loss_dense)

            ratio = loss_sum / (sigma**2)
            weighted = ratio * inv_qy
            diff_per_ex = weighted.mean(dim=1)

        if debug_integrand and bi < int(debug_first_n_batches):
            header = (
                f"[VLB-DBG] mode={'FORCE-UNCOND' if force_unconditional_path else ('COND-CLEAN' if saw_clean_prefix_mode else 'UNCOND-AUTO')}"
                f" bi={bi} B={B} K={K} cL={cL} S_full={S_full} S_eval={S_eval} "
                f"sigma_min_eval={sigma_min_eval:g} sigma_max_eval={sigma_max_eval:g} "
                f"dist={sigma_sampling} repr={_repr_mode(cfg)}"
            )
            print(header)
            print(" " + _qstats("sigma", sigma))
            if ln_range > 0:
                print(" " + _qstats("loss_sum", loss_sum))
                print(" " + _qstats("loss_sum/sigma^2", ratio))
                print(" " + _qstats("inv_qy", inv_qy))
                print(
                    _bin_means(
                        sigma,
                        loss_sum,
                        ratio,
                        smin=sigma_min_eval,
                        smax=sigma_max_eval,
                        nbins=debug_num_sigma_bins,
                        label="loss_sum",
                    )
                )

        sum_prior += prior_per_ex.double().sum()
        sum_recon += recon_per_ex.double().sum()
        sum_diff += diff_per_ex.double().sum()
        sum_count += torch.tensor(float(B), device=device, dtype=torch.float64)

    if _ddp_is_on():
        dist.all_reduce(sum_prior, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_recon, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_diff, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_count, op=dist.ReduceOp.SUM)

    n = int(sum_count.item())
    if n <= 0 or S_dim is None:
        raise RuntimeError("No examples processed for VLB.")

    prior_mean = (sum_prior / sum_count).float()
    recon_mean = (sum_recon / sum_count).float()
    diff_mean = (sum_diff / sum_count).float()

    S_eval_final = int(S_dim)
    if (not force_unconditional_path) and saw_clean_prefix_mode and allow_conditional_clean_prefix:
        cL_full = _cond_len_positions(cfg, int(S_dim))
        S_eval_final = max(1, int(S_dim - cL_full))

    denom = float(math.log(2.0) * max(1, int(S_eval_final)))
    vlb_bpd = (prior_mean + recon_mean + diff_mean) / denom

    final_mode = (
        "FORCE-UNCOND"
        if force_unconditional_path
        else ("COND-CLEAN" if saw_clean_prefix_mode else "UNCOND-AUTO")
    )

    return VLBResult(
        vlb_bpd=float(vlb_bpd.item()),
        recon_bpd=float((recon_mean / denom).item()),
        diff_bpd=float((diff_mean / denom).item()),
        prior_bpd=float((prior_mean / denom).item()),
        sigma_min_eval=float(sigma_min_eval),
        sigma_max_eval=float(sigma_max_eval),
        K=int(K),
        sigma_sampling=str(sigma_sampling),
        num_examples=int(n),
        S_dim=int(S_eval_final),
        mode=str(final_mode),
    )