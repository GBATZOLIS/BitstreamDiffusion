#diffusion/continuous/samplers.py
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from tqdm import tqdm

from utils.ecc_secded import ecc_from_cfg, ecc_chunk_len
from diffusion.continuous.logit_postprocess import _model_logits_continuous


def _normalize_sc_refresh_mode(mode: Optional[str]) -> str:
    mode = "refined" if mode is None else str(mode).lower()
    aliases = {
        "refined": "refined",
        "refresh": "refined",
        "full": "refined",
        "carry": "carry",
        "unrefined": "carry",
        "no_refresh": "carry",
        "no-refine": "carry",
    }
    if mode not in aliases:
        raise ValueError(f"Unknown sc_refresh_mode='{mode}'")
    return aliases[mode]


def _infer_model_device(model, cfg_device: str) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        pass
    try:
        return next(model.buffers()).device
    except StopIteration:
        pass
    return torch.device(cfg_device)


def logits_to_x0_hat(
    logits: torch.Tensor,
    dtype: torch.dtype,
    *,
    is_cont_tokens: bool = False,
) -> torch.Tensor:
    """
    Convert canonical logits -> probability-like x0_hat.

    Expects:
      - binary mode: logits [B,S]
      - token mode:  logits [B,S,V]
    """
    if is_cont_tokens:
        if logits.dim() != 3:
            raise ValueError(
                f"Continuous token mode expects logits [B,S,V], got {tuple(logits.shape)}"
            )
        return torch.softmax(logits.float(), dim=-1).to(dtype=dtype)

    if logits.dim() != 2:
        raise ValueError(
            f"Continuous binary mode expects logits [B,S], got {tuple(logits.shape)}"
        )
    return torch.sigmoid(logits.float()).to(dtype=dtype)


def get_score_from_logits(logits, x_t, sigma):
    """
    Binary helper kept for backward compatibility.

    Expects canonical binary logits:
      logits: [B,S]
      x_t:    [B,S]

    Interprets:
      D(x, σ) = sigmoid(logits)
      score   = (D(x, σ) - x) / σ²
    """
    if logits.dim() != 2:
        raise ValueError(f"Expected binary logits [B,S], got {tuple(logits.shape)}")
    if x_t.dim() != 2:
        raise ValueError(f"Expected binary state x_t [B,S], got {tuple(x_t.shape)}")

    if isinstance(sigma, float):
        sigma = torch.tensor(sigma, device=x_t.device, dtype=x_t.dtype)
    elif isinstance(sigma, torch.Tensor) and sigma.device != x_t.device:
        sigma = sigma.to(device=x_t.device)

    if sigma.dim() == 0:
        sigma = sigma.expand(x_t.size(0))

    sigma2 = (sigma**2).view(-1, 1).to(torch.float32)
    probs = torch.sigmoid(logits.to(torch.float32))
    return (probs - x_t.to(torch.float32)) / sigma2


# -----------------------------------------------------------------------------
# Helpers for ATI (Asymmetric Time Intervals)
# -----------------------------------------------------------------------------

def _resolve_ati_eta(cfg, ati_eta: Optional[float]) -> float:
    if ati_eta is not None:
        return max(0.0, float(ati_eta))
    ev = getattr(cfg, "evaluation", None)
    if ev is not None:
        ati_cfg = getattr(ev, "ati", None)
        if ati_cfg is not None:
            if not bool(getattr(ati_cfg, "enabled", True)):
                return 0.0
            return max(0.0, float(getattr(ati_cfg, "eta", 0.0)))
        legacy = getattr(ev, "ati_eta", None)
        if legacy is not None:
            return max(0.0, float(legacy))
    return 0.0


def _ati_shift_sigma_label(
    sigma_state: torch.Tensor,
    sigma_noisier: Optional[torch.Tensor],
    eta: float,
) -> torch.Tensor:
    """
    Local ATI label shift in log-sigma space.
    """
    if eta <= 0.0 or sigma_noisier is None:
        return sigma_state

    s = sigma_state.to(torch.float32)
    n = sigma_noisier.to(device=s.device, dtype=torch.float32)

    if torch.all(n <= s):
        return sigma_state

    log_s = torch.log(s.clamp_min(1e-20))
    log_n = torch.log(n.clamp_min(1e-20))
    sigma_eval = torch.exp(log_s + float(eta) * (log_n - log_s))
    sigma_eval = torch.maximum(sigma_eval, s)
    sigma_eval = torch.minimum(sigma_eval, n)
    return sigma_eval.to(dtype=sigma_state.dtype)


# -----------------------------------------------------------------------------
# Helpers for stochastic EDM-style churn
# -----------------------------------------------------------------------------

@dataclass
class _StochasticSamplerCfg:
    enabled: bool
    s_churn: float = 0.0
    s_noise: float = 1.0
    s_tmin: float = 0.0
    s_tmax: float = float("inf")
    window_mode: str = "deterministic"


def _clamp_prob01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _resolve_sigma_bounds(
    cfg,
    *,
    sigma_min_override: Optional[float],
    sigma_max_override: Optional[float],
) -> tuple[float, float]:
    sigma_min = (
        float(sigma_min_override)
        if sigma_min_override is not None
        else float(cfg.diffusion.continuous.sigma_min)
    )
    sigma_max = (
        float(sigma_max_override)
        if sigma_max_override is not None
        else float(cfg.diffusion.continuous.sigma_max)
    )
    if sigma_max < sigma_min:
        sigma_max, sigma_min = sigma_min, sigma_max
    return sigma_min, sigma_max


def _compute_edm_gamma(
    sigma_cur: torch.Tensor | float,
    *,
    num_intervals: int,
    s_churn: float,
    s_tmin: float,
    s_tmax: float,
) -> float:
    """
    EDM-style gamma:
      gamma_i = min(S_churn / N, sqrt(2)-1) if sigma_i in [S_tmin, S_tmax]
                0 otherwise
    """
    s = float(sigma_cur.item()) if isinstance(sigma_cur, torch.Tensor) else float(sigma_cur)
    if s < float(s_tmin) or s > float(s_tmax):
        return 0.0
    if num_intervals <= 0:
        return 0.0
    return max(
        0.0,
        min(float(s_churn) / float(num_intervals), math.sqrt(2.0) - 1.0),
    )


def _apply_edm_churn(
    x: torch.Tensor,
    sigma_cur: torch.Tensor,
    *,
    gamma: float,
    s_noise: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    EDM Algorithm 2:
      sigma_hat = sigma_cur * (1 + gamma)
      x_hat = x + sqrt(sigma_hat^2 - sigma_cur^2) * eps,
      eps ~ N(0, S_noise^2 I)
    """
    if gamma <= 0.0:
        return x, sigma_cur

    sigma_hat = sigma_cur * (1.0 + float(gamma))
    sigma_delta = (sigma_hat.square() - sigma_cur.square()).clamp_min(0.0).sqrt()

    eps = torch.randn_like(x) * float(s_noise)
    x_hat = x + sigma_delta.to(device=x.device, dtype=x.dtype) * eps
    return x_hat, sigma_hat


# -----------------------------------------------------------------------------
# Helpers for conditional prompting + CFG
# -----------------------------------------------------------------------------

def _bits_per_unit(cfg) -> int:
    ecc = ecc_from_cfg(cfg)
    if ecc.enabled:
        return int(ecc_chunk_len(ecc))

    data = getattr(cfg, "data", object())
    bpt = getattr(data, "bits_per_token", None)
    if bpt is not None:
        return int(bpt)
    return int(getattr(data, "bits_per_char", 1))


def _get_cond_len_bits(cfg, seq_len: int, cond_len_bits_override: Optional[int] = None) -> int:
    """
    Backward-compatible prompt length in current model-space positions.

    For binary runs:
      returns bit positions.

    For token runs:
      returns token positions, because seq_len is already token length.
    """
    if cond_len_bits_override is not None:
        return max(0, min(int(cond_len_bits_override), int(seq_len)))

    cond_cfg = getattr(cfg, "cond", None)
    if cond_cfg is None or not bool(getattr(cond_cfg, "enabled", False)):
        return 0

    n_units = getattr(cond_cfg, "cond_len_tokens", None)
    if n_units is None:
        n_units = int(getattr(cond_cfg, "cond_len_chars", 0))
    else:
        n_units = int(n_units)

    repr_mode = str(getattr(getattr(cfg, "data", object()), "representation", "binary")).lower()
    if repr_mode == "tokens":
        cL = int(n_units)
    else:
        bits_per = _bits_per_unit(cfg)
        cL = int(n_units * bits_per)

    return max(0, min(cL, int(seq_len)))


def _make_null_value(
    cfg,
    device,
    dtype,
    *,
    is_cont_tokens: bool = False,
    vocab_size: int | None = None,
) -> torch.Tensor:
    cond_cfg = getattr(cfg, "cond", None)
    strategy = str(getattr(cond_cfg, "null_strategy", "half")) if cond_cfg is not None else "half"

    if is_cont_tokens:
        if strategy in {"half", "data_center"}:
            if vocab_size is None:
                raise ValueError("vocab_size required for token null value")
            dc = float(getattr(cfg.diffusion.continuous, "data_center", 1.0 / vocab_size))
            return torch.tensor(dc, device=device, dtype=dtype)
        if strategy == "zeros":
            return torch.tensor(0.0, device=device, dtype=dtype)
        if strategy == "random":
            return torch.tensor(float("nan"), device=device, dtype=dtype)
        raise ValueError(f"Unknown cfg.cond.null_strategy={strategy}")

    if strategy == "half":
        return torch.tensor(0.5, device=device, dtype=dtype)
    if strategy == "data_center":
        return torch.tensor(
            float(getattr(cfg.diffusion.continuous, "data_center", 0.5)),
            device=device,
            dtype=dtype,
        )
    if strategy == "zeros":
        return torch.tensor(0.0, device=device, dtype=dtype)
    if strategy == "random":
        return torch.tensor(float("nan"), device=device, dtype=dtype)
    raise ValueError(f"Unknown cfg.cond.null_strategy={strategy}")


def _make_null_full(
    prefix_full: torch.Tensor,
    prefix_mask: torch.Tensor,
    cfg,
    *,
    is_cont_tokens: bool = False,
    vocab_size: int = 2,
) -> torch.Tensor:
    """
    Build the unconditional/dropped-prompt prefix_full for CFG.

    Binary:
      prefix_full [B,S], prefix_mask [B,S]

    Token:
      prefix_full [B,S,V], prefix_mask [B,S]
    """
    out = prefix_full.clone()
    if not prefix_mask.any():
        return out

    cond_cfg = getattr(cfg, "cond", None)
    strategy = str(getattr(cond_cfg, "null_strategy", "half")) if cond_cfg is not None else "half"

    if is_cont_tokens:
        pm = prefix_mask.unsqueeze(-1).expand_as(prefix_full)

        if strategy == "random":
            rnd = torch.full_like(prefix_full, 1.0 / float(vocab_size))
            out[pm] = rnd[pm]
            return out

        null_val = _make_null_value(
            cfg,
            prefix_full.device,
            prefix_full.dtype,
            is_cont_tokens=True,
            vocab_size=vocab_size,
        )
        out[pm] = null_val
        return out

    if strategy == "random":
        rnd = torch.bernoulli(
            torch.full(
                prefix_full.shape,
                0.5,
                device=prefix_full.device,
                dtype=prefix_full.dtype,
            )
        )
        out[prefix_mask] = rnd[prefix_mask]
        return out

    null_val = _make_null_value(
        cfg,
        prefix_full.device,
        prefix_full.dtype,
        is_cont_tokens=False,
    )
    out[prefix_mask] = null_val
    return out


def _expand_prefix_to_batch(prefix: torch.Tensor, B: int, device, dtype) -> torch.Tensor:
    prefix = prefix.to(device=device, dtype=dtype)
    if prefix.dim() == 1:
        prefix = prefix.unsqueeze(0).expand(B, -1).contiguous()
    elif prefix.dim() == 2:
        if prefix.size(0) != B:
            raise ValueError(f"conditioning_prefix batch mismatch: got {prefix.size(0)} vs B={B}")
    else:
        raise ValueError("conditioning_prefix must have shape [cL] or [B,cL]")
    return prefix


def _clamp_prefix_(x: torch.Tensor, prefix: torch.Tensor, cL: int) -> None:
    if cL > 0:
        x[:, :cL] = prefix


def _clamp_mask_(x: torch.Tensor, full: torch.Tensor, mask: torch.Tensor) -> None:
    """
    In-place clamp:
      binary: x/full [B,S], mask [B,S]
      token:  x/full [B,S,V], mask [B,S]
    """
    if mask is None or (not bool(mask.any().item())):
        return

    if x.dim() == 3 and mask.dim() == 2:
        mask = mask.unsqueeze(-1).expand_as(x)

    x[mask] = full[mask]


def _zero_mask_(d: torch.Tensor, mask: torch.Tensor) -> None:
    if mask is None or (not bool(mask.any().item())):
        return

    if d.dim() == 3 and mask.dim() == 2:
        mask = mask.unsqueeze(-1).expand_as(d)

    d[mask] = 0.0


def _score_from_probs(
    probs: torch.Tensor,
    x_t: torch.Tensor,
    sigma: torch.Tensor,
    *,
    is_cont_tokens: bool = False,
) -> torch.Tensor:
    """
    probs:
      - binary: [B,S]
      - token:  [B,S,V]
    """
    if isinstance(sigma, float):
        sigma = torch.tensor(sigma, device=x_t.device, dtype=x_t.dtype)
    if isinstance(sigma, torch.Tensor) and sigma.device != x_t.device:
        sigma = sigma.to(device=x_t.device)
    if sigma.dim() == 0:
        sigma = sigma.expand(x_t.size(0))

    if is_cont_tokens:
        sigma2 = (sigma**2).view(-1, 1, 1).to(torch.float32)
    else:
        sigma2 = (sigma**2).view(-1, 1).to(torch.float32)

    probs_f = probs.to(torch.float32)
    x_f = x_t.to(torch.float32)
    return (probs_f - x_f) / sigma2


def _build_mask_conditioning(
    *,
    cfg,
    B: int,
    S: int,
    device: torch.device,
    conditioning_prefix_full: Optional[torch.Tensor],
    cond_prefix_mask: Optional[torch.Tensor],
    conditioning_prefix: Optional[torch.Tensor],
    cond_len_bits: Optional[int],
    is_cont_tokens: bool = False,
    vocab_size: int = 2,
) -> Tuple[bool, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Unifies legacy and new conditioning APIs.

    Returns:
      cond_enabled: bool
      prefix_full:
        - binary: [B,S]
        - token:  [B,S,V]
      prefix_mask: [B,S] bool
      null_full: same shape as prefix_full
    """

    def _to_onehot_prefix(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 3:
            if t.size(0) != B or t.size(1) != S or t.size(2) != vocab_size:
                raise ValueError(
                    f"conditioning_prefix_full must be [B,S,V]; got {tuple(t.shape)} vs {(B, S, vocab_size)}"
                )
            return t.to(device=device, dtype=torch.float32)

        if t.dim() == 1:
            if t.numel() != S:
                raise ValueError(f"conditioning_prefix_full has length {t.numel()} but expected S={S}")
            t = t.view(1, S).expand(B, S)

        elif t.dim() == 2:
            if t.size(0) != B or t.size(1) != S:
                raise ValueError(f"conditioning_prefix_full must be [B,S]; got {tuple(t.shape)}")
        else:
            raise ValueError("conditioning_prefix_full must have shape [S], [B,S], or [B,S,V]")

        return torch.nn.functional.one_hot(
            t.to(device=device, dtype=torch.long),
            num_classes=vocab_size,
        ).float()

    # ------------------------------------------------------------------
    # New API: full prefix + mask
    # ------------------------------------------------------------------
    if conditioning_prefix_full is not None or cond_prefix_mask is not None:
        if conditioning_prefix_full is None or cond_prefix_mask is None:
            raise ValueError(
                "Must provide BOTH conditioning_prefix_full and cond_prefix_mask (or neither)."
            )

        if is_cont_tokens:
            pf = _to_onehot_prefix(conditioning_prefix_full)
        else:
            pf = conditioning_prefix_full.to(device=device, dtype=torch.float32)
            if pf.dim() == 1:
                if pf.numel() != S:
                    raise ValueError(
                        f"conditioning_prefix_full has length {pf.numel()} but expected S={S}"
                    )
                pf = pf.view(1, S).expand(B, S).contiguous()
            elif pf.dim() == 2:
                if pf.size(0) != B or pf.size(1) != S:
                    raise ValueError(
                        f"conditioning_prefix_full must be [B,S]; got {tuple(pf.shape)}"
                    )
            else:
                raise ValueError(
                    "conditioning_prefix_full must have shape [S] or [B,S] in binary mode"
                )

        pm = cond_prefix_mask.to(device=device)
        if pm.dtype != torch.bool:
            pm = pm.to(torch.bool)
        if pm.dim() == 1:
            if pm.numel() != S:
                raise ValueError(f"cond_prefix_mask has length {pm.numel()} but expected S={S}")
            pm = pm.view(1, S).expand(B, S).contiguous()
        elif pm.dim() == 2:
            if pm.size(0) != B or pm.size(1) != S:
                raise ValueError(f"cond_prefix_mask must be [B,S]; got {tuple(pm.shape)}")
        else:
            raise ValueError("cond_prefix_mask must have shape [S] or [B,S]")

        cond_enabled = bool(pm.any().item())
        if not cond_enabled:
            return False, None, None, None

        null_full = _make_null_full(
            pf,
            pm,
            cfg,
            is_cont_tokens=is_cont_tokens,
            vocab_size=vocab_size,
        )
        return True, pf, pm, null_full

    # ------------------------------------------------------------------
    # Legacy API: fixed prefix length
    # ------------------------------------------------------------------
    cL = _get_cond_len_bits(cfg, S, cond_len_bits_override=cond_len_bits)
    cond_enabled = (conditioning_prefix is not None) and (cL > 0)
    if not cond_enabled:
        return False, None, None, None

    if is_cont_tokens:
        cp = conditioning_prefix.to(device=device)
        if cp.dim() == 1:
            cp = cp.view(1, cL).expand(B, cL)
        elif cp.dim() == 2:
            if cp.size(0) != B or cp.size(1) != cL:
                raise ValueError(
                    f"conditioning_prefix must be [B,cL] or [cL], got {tuple(cp.shape)}"
                )
        else:
            raise ValueError(
                "conditioning_prefix must have shape [cL] or [B,cL] in token mode"
            )

        cp_oh = torch.nn.functional.one_hot(cp.long(), num_classes=vocab_size).float()
        prefix_full = torch.full((B, S, vocab_size), 0.0, device=device, dtype=torch.float32)
        prefix_mask = torch.zeros((B, S), device=device, dtype=torch.bool)
        prefix_full[:, :cL, :] = cp_oh
        prefix_mask[:, :cL] = True
    else:
        cond_prefix = _expand_prefix_to_batch(conditioning_prefix, B, device, torch.float32)
        if cond_prefix.size(1) != cL:
            raise ValueError(
                f"conditioning_prefix has {cond_prefix.size(1)} bits but expected cL={cL}"
            )

        prefix_full = torch.zeros((B, S), device=device, dtype=torch.float32)
        prefix_mask = torch.zeros((B, S), device=device, dtype=torch.bool)
        prefix_full[:, :cL] = cond_prefix
        prefix_mask[:, :cL] = True

    null_full = _make_null_full(
        prefix_full,
        prefix_mask,
        cfg,
        is_cont_tokens=is_cont_tokens,
        vocab_size=vocab_size,
    )
    return True, prefix_full, prefix_mask, null_full

# -----------------------------------------------------------------------------
# Shared sigma-schedule provider
# -----------------------------------------------------------------------------


class SigmaSchedule:
    """
    Provides sampling sigma schedules (karras/entropic) and entropy table IO.
    """

    def __init__(self, process, cfg, device: torch.device):
        self.process = process
        self.cfg = cfg
        self.device = device

    @staticmethod
    def _entropy_run_dir_from_ckpt(ckpt_path: Path) -> Path:
        return ckpt_path.parent.parent

    def _default_entropy_run_dir(self) -> Path:
        ckpt_path = Path(self.cfg.evaluation.checkpoint_path).expanduser().resolve()
        return self._entropy_run_dir_from_ckpt(ckpt_path)

    def _load_entropy_tables(
        self,
        *,
        entropy_run_dir: Optional[Path] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if entropy_run_dir is None:
            entropy_run_dir = self._default_entropy_run_dir()
        else:
            if isinstance(entropy_run_dir, str):
                entropy_run_dir = Path(entropy_run_dir)
            else:
                try:
                    entropy_run_dir = Path(entropy_run_dir)
                except TypeError:
                    pass

        pdf_p = entropy_run_dir / "entropy_pdf.pt"
        cdf_p = entropy_run_dir / "entropy_cdf.pt"
        sig_p = entropy_run_dir / "entropy_sigmas.pt"

        if not (pdf_p.exists() and cdf_p.exists() and sig_p.exists()):
            return None, None, None

        pdf = torch.load(pdf_p, map_location=self.device, weights_only=True).to(self.device)
        cdf = torch.load(cdf_p, map_location=self.device, weights_only=True).to(self.device)
        sigs = torch.load(sig_p, map_location=self.device, weights_only=True).to(self.device)
        return pdf, cdf, sigs

    def entropy_quantile(
        self,
        q: float,
        *,
        entropy_run_dir: Optional[Path] = None,
    ) -> Optional[float]:
        """
        Return sigma at quantile q of the saved entropy CDF.
        Assumes the saved sigma table is ascending in sigma.
        """
        _, cdf, sigmas_base = self._load_entropy_tables(entropy_run_dir=entropy_run_dir)
        if cdf is None or sigmas_base is None:
            return None

        cdf = cdf.to(self.device).float().clone()
        sig = sigmas_base.to(self.device).float()

        if cdf.numel() == 0 or sig.numel() == 0:
            return None

        cdf[-1] = 1.0
        q = _clamp_prob01(q)
        q_t = torch.tensor(q, device=self.device, dtype=torch.float32)

        idx = torch.searchsorted(cdf, q_t, right=False)
        idx = idx.clamp(min=0, max=cdf.numel() - 1)

        i = int(idx.item())
        if i == 0:
            return float(sig[0].item())

        cdf_lo = cdf[i - 1]
        cdf_hi = cdf[i]
        sig_lo = sig[i - 1]
        sig_hi = sig[i]
        denom = (cdf_hi - cdf_lo).clamp_min(1e-20)
        w = (q_t - cdf_lo) / denom
        sig_q = sig_lo + w * (sig_hi - sig_lo)
        return float(sig_q.item())

    def resolve_stochastic_cfg(
        self,
        *,
        entropy_run_dir: Optional[Path] = None,
        sigma_min_override: Optional[float] = None,
        sigma_max_override: Optional[float] = None,
    ) -> _StochasticSamplerCfg:
        """
        Resolve evaluation-time stochastic sampler config.

        window_mode:
          - deterministic/off/none
          - full
          - fixed
          - entropy_cdf
        """
        ev = getattr(self.cfg, "evaluation", None)
        st = getattr(ev, "stochastic", None)

        if st is None or not bool(getattr(st, "enabled", False)):
            return _StochasticSamplerCfg(enabled=False)

        sigma_min, sigma_max = _resolve_sigma_bounds(
            self.cfg,
            sigma_min_override=sigma_min_override,
            sigma_max_override=sigma_max_override,
        )

        window_mode = str(getattr(st, "window_mode", "entropy_cdf")).lower().strip()
        fallback = str(getattr(st, "entropy_fallback", "deterministic")).lower().strip()

        s_churn = max(0.0, float(getattr(st, "s_churn", 0.0)))
        s_noise = max(0.0, float(getattr(st, "s_noise", 1.0)))

        def _finalize(lo: float, hi: float) -> _StochasticSamplerCfg:
            lo = max(sigma_min, min(float(lo), sigma_max))
            hi = max(sigma_min, min(float(hi), sigma_max))
            if hi < lo:
                lo, hi = hi, lo
            return _StochasticSamplerCfg(
                enabled=(s_churn > 0.0 and hi >= lo),
                s_churn=s_churn,
                s_noise=s_noise,
                s_tmin=lo,
                s_tmax=hi,
                window_mode=window_mode,
            )

        if window_mode in {"deterministic", "off", "none"}:
            return _StochasticSamplerCfg(enabled=False)

        if window_mode == "full":
            return _finalize(sigma_min, sigma_max)

        if window_mode == "fixed":
            lo = sigma_min if getattr(st, "s_tmin", None) is None else float(st.s_tmin)
            hi = sigma_max if getattr(st, "s_tmax", None) is None else float(st.s_tmax)
            return _finalize(lo, hi)

        if window_mode == "entropy_cdf":
            q_lo = _clamp_prob01(getattr(st, "entropy_quantile_lo", 0.10))
            q_hi = _clamp_prob01(getattr(st, "entropy_quantile_hi", 0.90))
            if q_hi < q_lo:
                q_lo, q_hi = q_hi, q_lo

            lo = self.entropy_quantile(q_lo, entropy_run_dir=entropy_run_dir)
            hi = self.entropy_quantile(q_hi, entropy_run_dir=entropy_run_dir)

            if lo is not None and hi is not None:
                return _finalize(lo, hi)

            if fallback == "full":
                return _finalize(sigma_min, sigma_max)

            if fallback == "fixed":
                lo = sigma_min if getattr(st, "s_tmin", None) is None else float(st.s_tmin)
                hi = sigma_max if getattr(st, "s_tmax", None) is None else float(st.s_tmax)
                return _finalize(lo, hi)

            return _StochasticSamplerCfg(enabled=False)

        raise ValueError(f"Unknown evaluation.stochastic.window_mode='{window_mode}'")

    def _karras_schedule(self, N: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
        rho = float(getattr(self.cfg.diffusion.continuous, "rho", 7.0))
        if N < 2:
            return torch.tensor([sigma_max], device=self.device, dtype=torch.float32)
        t = torch.linspace(0.0, 1.0, N, device=self.device, dtype=torch.float32)
        inv_rho = 1.0 / rho
        smax = float(sigma_max) ** inv_rho
        smin = float(sigma_min) ** inv_rho
        sigmas = (smax + t * (smin - smax)) ** rho
        sigmas[0] = float(sigma_max)
        sigmas[-1] = float(sigma_min)
        return sigmas

    @staticmethod
    def _interp1d_monotone(x: torch.Tensor, y: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
        x0 = x[0]
        x1 = x[-1]
        xq_clamped = xq.clamp(min=x0, max=x1)
        idx = torch.searchsorted(x, xq_clamped, right=False)
        idx = idx.clamp(min=1, max=x.numel() - 1)
        x_lo = x[idx - 1]
        x_hi = x[idx]
        y_lo = y[idx - 1]
        y_hi = y[idx]
        denom = (x_hi - x_lo).clamp_min(1e-20)
        w = (xq_clamped - x_lo) / denom
        return y_lo + w * (y_hi - y_lo)

    def _inverse_cdf_sample_truncated(
        self,
        sigmas_base: torch.Tensor,
        cdf: torch.Tensor,
        *,
        N: int,
        sigma_min: float,
        sigma_max: float,
    ) -> torch.Tensor:
        sig = sigmas_base.to(self.device).float()
        F = cdf.to(self.device).float()

        if sig.numel() < 2:
            out = torch.full((N,), float(sigma_max), device=self.device, dtype=torch.float32)
            out[-1] = float(sigma_min)
            return out

        table_min = float(sig[0].item())
        table_max = float(sig[-1].item())
        sigma_min_eff = float(max(sigma_min, table_min))
        sigma_max_eff = float(min(sigma_max, table_max))

        if sigma_min_eff > sigma_max_eff:
            val = float(max(min(sigma_max, table_max), table_min))
            out = torch.full((N,), val, device=self.device, dtype=torch.float32)
            out[-1] = val
            return out

        s_min_t = torch.tensor(sigma_min_eff, device=self.device, dtype=torch.float32)
        s_max_t = torch.tensor(sigma_max_eff, device=self.device, dtype=torch.float32)
        F_min = self._interp1d_monotone(sig, F, s_min_t)
        F_max = self._interp1d_monotone(sig, F, s_max_t)

        if float((F_max - F_min).abs().item()) < 1e-12:
            lo = torch.log(torch.tensor(sigma_min_eff, device=self.device))
            hi = torch.log(torch.tensor(sigma_max_eff, device=self.device))
            s_fwd = torch.exp(torch.linspace(lo, hi, N, device=self.device, dtype=torch.float32))
            s = torch.flip(s_fwd, dims=[0])
            s[0] = sigma_max_eff
            s[-1] = sigma_min_eff
            return s

        u = torch.linspace(0.0, 1.0, N, device=self.device, dtype=torch.float32)
        u = F_min + u * (F_max - F_min)
        u = u.clamp(min=F[0].item(), max=1.0 - 1e-7)

        idx = torch.searchsorted(F, u, right=False).clamp(min=1, max=F.numel() - 1)
        F_lo = F[idx - 1]
        F_hi = F[idx]
        s_lo = sig[idx - 1]
        s_hi = sig[idx]
        denom = (F_hi - F_lo).clamp_min(1e-20)
        w = (u - F_lo) / denom
        sig_forward = s_lo + w * (s_hi - s_lo)
        sig_forward[0] = sigma_min_eff
        sig_forward[-1] = sigma_max_eff
        sigmas = torch.flip(sig_forward, dims=[0])
        sigmas[0] = sigma_max_eff
        sigmas[-1] = sigma_min_eff
        return sigmas

    def prepare(
        self,
        *,
        schedule: Optional[str] = None,
        num_steps: Optional[int] = None,
        entropic_blend_alpha: Optional[float] = None,
        entropy_run_dir: Optional[Path] = None,
        sigma_min_override: Optional[float] = None,
        sigma_max_override: Optional[float] = None,
    ) -> torch.Tensor:
        schedule_name = schedule if schedule is not None else getattr(
            self.cfg.evaluation, "schedule", "karras"
        )
        schedule_name = str(schedule_name).lower()
        N = int(
            num_steps
            if num_steps is not None
            else getattr(self.cfg.evaluation, "num_sampling_steps", 400)
        )

        sigma_max = (
            float(sigma_max_override)
            if sigma_max_override is not None
            else float(self.cfg.diffusion.continuous.sigma_max)
        )
        sigma_min = (
            float(sigma_min_override)
            if sigma_min_override is not None
            else float(self.cfg.diffusion.continuous.sigma_min)
        )

        if sigma_max < sigma_min:
            sigma_max, sigma_min = sigma_min, sigma_max

        if schedule_name == "karras":
            return self._karras_schedule(N, sigma_min=sigma_min, sigma_max=sigma_max)

        if schedule_name == "entropic":
            _, cdf, sigmas_base = self._load_entropy_tables(entropy_run_dir=entropy_run_dir)
            if cdf is None or sigmas_base is None:
                resolved_dir = entropy_run_dir if entropy_run_dir is not None else self._default_entropy_run_dir()
                raise FileNotFoundError(
                    "Entropic schedule was requested but the entropy tables "
                    "(entropy_pdf.pt, entropy_cdf.pt, entropy_sigmas.pt) are missing.\n"
                    f"Looked in: {resolved_dir}\n"
                    "The released LM1B and OWT eval configs point at "
                    "assets/entropy_tables/<dataset>/, which ships with the repo. "
                    "If you moved or deleted those files, see the README section "
                    "'Entropic schedule artefacts' for download/restore instructions."
                )

            cdf = cdf.to(self.device).clone().float()
            cdf[-1] = 1.0
            sigmas_base = sigmas_base.to(self.device).float()

            sigmas = self._inverse_cdf_sample_truncated(
                sigmas_base,
                cdf,
                N=N,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )

            blend = float(
                entropic_blend_alpha
                if entropic_blend_alpha is not None
                else getattr(self.cfg.evaluation, "entropic_blend_alpha", 0.0)
            )
            if blend > 0:
                karras = self._karras_schedule(
                    N,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                ).to(dtype=sigmas.dtype)
                sigmas = (1.0 - blend) * sigmas + blend * karras
            return sigmas

        return self._karras_schedule(N, sigma_min=sigma_min, sigma_max=sigma_max)


# -----------------------------------------------------------------------------
# Heun sampler
# -----------------------------------------------------------------------------

class HeunSampler:
    """
    Second-order ODE solver with:
      - centering fix
      - prompt conditioning
      - CFG via fused 2B batching
      - configurable self-conditioning refresh mode
      - ATI support
      - EDM stochastic churn support
      - continuous binary and continuous one-hot token support
    """

    def __init__(self, model, forward_process, cfg):
        self.model = model
        self.process = forward_process
        self.cfg = cfg
        self.device = _infer_model_device(model, cfg.device)
        self.sigmas = SigmaSchedule(self.process, self.cfg, self.device)
        self.sc_enabled = bool(getattr(self.cfg.model, "self_condition", False))
        self.data_center = float(getattr(self.cfg.diffusion.continuous, "data_center", 0.5))

        self.repr_mode = str(getattr(cfg.data, "representation", "binary")).lower()
        self.is_cont_tokens = (self.repr_mode == "tokens")
        self.vocab_size = int(getattr(cfg.data, "vocab_size", 1)) if self.is_cont_tokens else 1

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        seq_len: int,
        *,
        conditioning_prefix_full: Optional[torch.Tensor] = None,
        cond_prefix_mask: Optional[torch.Tensor] = None,
        conditioning_prefix: Optional[torch.Tensor] = None,
        cond_len_bits: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        schedule: Optional[str] = None,
        num_steps: Optional[int] = None,
        entropic_blend_alpha: Optional[float] = None,
        entropy_run_dir: Optional[Path] = None,
        sigma_min_override: Optional[float] = None,
        sigma_max_override: Optional[float] = None,
        sc_refresh_mode: str = "refined",
        ati_eta: Optional[float] = None,
        return_probs: bool = False,
        progress: bool = True,
    ):
        sc_refresh_mode = _normalize_sc_refresh_mode(sc_refresh_mode)
        ati_eta = _resolve_ati_eta(self.cfg, ati_eta)

        B = int(num_samples)
        S = int(seq_len)

        sigmas = self.sigmas.prepare(
            schedule=schedule,
            num_steps=num_steps,
            entropic_blend_alpha=entropic_blend_alpha,
            entropy_run_dir=entropy_run_dir,
            sigma_min_override=sigma_min_override,
            sigma_max_override=sigma_max_override,
        )
        stoch_cfg = self.sigmas.resolve_stochastic_cfg(
            entropy_run_dir=entropy_run_dir,
            sigma_min_override=sigma_min_override,
            sigma_max_override=sigma_max_override,
        )
        sigma0 = sigmas[0]

        cond_enabled, prefix_full, prefix_mask, null_full = _build_mask_conditioning(
            cfg=self.cfg,
            B=B,
            S=S,
            device=self.device,
            conditioning_prefix_full=conditioning_prefix_full,
            cond_prefix_mask=cond_prefix_mask,
            conditioning_prefix=conditioning_prefix,
            cond_len_bits=cond_len_bits,
            is_cont_tokens=self.is_cont_tokens,
            vocab_size=self.vocab_size,
        )

        use_cfg = False
        if guidance_scale is None:
            guidance_scale = float(
                getattr(getattr(self.cfg, "evaluation", object()), "guidance_scale", 0.0)
            )
        guidance_scale = float(guidance_scale)
        use_cfg = bool(cond_enabled and (guidance_scale > 0.0))

        if self.is_cont_tokens:
            x = torch.randn(B, S, self.vocab_size, device=self.device, dtype=torch.float32) * sigma0
        else:
            x = torch.randn(B, S, device=self.device, dtype=torch.float32) * sigma0
        x = x + self.data_center

        if cond_enabled:
            _clamp_mask_(x, prefix_full, prefix_mask)

        if self.sc_enabled:
            if use_cfg:
                x0_hat_c = torch.zeros_like(x)
                x0_hat_u = torch.zeros_like(x)
                _clamp_mask_(x0_hat_c, prefix_full, prefix_mask)
                _clamp_mask_(x0_hat_u, null_full, prefix_mask)
            else:
                x0_hat = torch.zeros_like(x)
                if cond_enabled:
                    _clamp_mask_(x0_hat, prefix_full, prefix_mask)
        else:
            x0_hat_c = x0_hat_u = None
            x0_hat = None

        indices = range(len(sigmas) - 1)
        if progress:
            indices = tqdm(indices, desc="Heun Sampler", leave=False)

        for i in indices:
            sigma_cur, sigma_next = sigmas[i], sigmas[i + 1]
            sigma_prev = sigmas[i - 1] if i > 0 else None

            if cond_enabled:
                _clamp_mask_(x, prefix_full, prefix_mask)

            # ------------------------------------------------------------
            # Optional EDM-style stochastic churn:
            # move from (x, sigma_cur) to (x_state, sigma_state=sigma_hat),
            # then evaluate the denoiser at that perturbed state.
            # ------------------------------------------------------------
            gamma_i = _compute_edm_gamma(
                sigma_cur,
                num_intervals=max(1, len(sigmas) - 1),
                s_churn=stoch_cfg.s_churn,
                s_tmin=stoch_cfg.s_tmin,
                s_tmax=stoch_cfg.s_tmax,
            ) if stoch_cfg.enabled else 0.0

            x_state, sigma_state = _apply_edm_churn(
                x,
                sigma_cur,
                gamma=gamma_i,
                s_noise=stoch_cfg.s_noise,
            )

            if cond_enabled:
                _clamp_mask_(x_state, prefix_full, prefix_mask)

            sigma_eval_cur = _ati_shift_sigma_label(sigma_state, sigma_prev, ati_eta)
            sigma_eval_next = _ati_shift_sigma_label(sigma_next, sigma_state, ati_eta)
            h = sigma_next - sigma_state

            # ------------------------------------------------------------
            # 1) Evaluate at (x_state, sigma_state)
            # ------------------------------------------------------------
            if use_cfg:
                x_cat = torch.cat([x_state, x_state], dim=0)
                sig_cat = sigma_eval_cur.expand(2 * B)

                _clamp_mask_(x_cat[:B], prefix_full, prefix_mask)
                _clamp_mask_(x_cat[B:], null_full, prefix_mask)

                if self.sc_enabled:
                    cond_cat = torch.cat([x0_hat_c, x0_hat_u], dim=0)
                    _clamp_mask_(cond_cat[:B], prefix_full, prefix_mask)
                    _clamp_mask_(cond_cat[B:], null_full, prefix_mask)
                else:
                    cond_cat = torch.zeros_like(x_cat)

                logits_cat = _model_logits_continuous(self.model, self.cfg, x_cat, sig_cat, cond_cat)
                probs_c = logits_to_x0_hat(
                    logits_cat[:B],
                    dtype=x.dtype,
                    is_cont_tokens=self.is_cont_tokens,
                )
                probs_u = logits_to_x0_hat(
                    logits_cat[B:],
                    dtype=x.dtype,
                    is_cont_tokens=self.is_cont_tokens,
                )

                _clamp_mask_(probs_c, prefix_full, prefix_mask)
                _clamp_mask_(probs_u, null_full, prefix_mask)

                probs_g = probs_u + guidance_scale * (probs_c - probs_u)
                _clamp_mask_(probs_g, prefix_full, prefix_mask)

                score_cur = _score_from_probs(
                    probs_g,
                    x_state,
                    sigma_state,
                    is_cont_tokens=self.is_cont_tokens,
                )
                d_cur = -sigma_state * score_cur
                _zero_mask_(d_cur, prefix_mask)

                if self.sc_enabled:
                    x0_hat_cur_c = probs_c
                    x0_hat_cur_u = probs_u

            else:
                sig_B = sigma_eval_cur.expand(B)
                cond_in = x0_hat if self.sc_enabled else torch.zeros_like(x_state)

                if cond_enabled:
                    _clamp_mask_(x_state, prefix_full, prefix_mask)
                    if self.sc_enabled:
                        _clamp_mask_(cond_in, prefix_full, prefix_mask)

                logits = _model_logits_continuous(self.model, self.cfg, x_state, sig_B, cond_in)
                probs = logits_to_x0_hat(
                    logits,
                    dtype=x.dtype,
                    is_cont_tokens=self.is_cont_tokens,
                )

                if cond_enabled:
                    _clamp_mask_(probs, prefix_full, prefix_mask)

                score_cur = _score_from_probs(
                    probs,
                    x_state,
                    sigma_state,
                    is_cont_tokens=self.is_cont_tokens,
                )
                d_cur = -sigma_state * score_cur
                _zero_mask_(d_cur, prefix_mask)

                if self.sc_enabled:
                    x0_hat_cur = probs

            x_pred = x_state + h * d_cur
            if cond_enabled:
                _clamp_mask_(x_pred, prefix_full, prefix_mask)

            # ------------------------------------------------------------
            # 2) Evaluate at (x_pred, sigma_next)
            # ------------------------------------------------------------
            if use_cfg:
                x_pred_cat = torch.cat([x_pred, x_pred], dim=0)
                sig_next_cat = sigma_eval_next.expand(2 * B)

                _clamp_mask_(x_pred_cat[:B], prefix_full, prefix_mask)
                _clamp_mask_(x_pred_cat[B:], null_full, prefix_mask)

                if self.sc_enabled:
                    cond2_cat = torch.cat([x0_hat_cur_c, x0_hat_cur_u], dim=0)
                    _clamp_mask_(cond2_cat[:B], prefix_full, prefix_mask)
                    _clamp_mask_(cond2_cat[B:], null_full, prefix_mask)
                else:
                    cond2_cat = torch.zeros_like(x_pred_cat)

                logits2_cat = _model_logits_continuous(
                    self.model,
                    self.cfg,
                    x_pred_cat,
                    sig_next_cat,
                    cond2_cat,
                )
                probs2_c = logits_to_x0_hat(
                    logits2_cat[:B],
                    dtype=x.dtype,
                    is_cont_tokens=self.is_cont_tokens,
                )
                probs2_u = logits_to_x0_hat(
                    logits2_cat[B:],
                    dtype=x.dtype,
                    is_cont_tokens=self.is_cont_tokens,
                )

                _clamp_mask_(probs2_c, prefix_full, prefix_mask)
                _clamp_mask_(probs2_u, null_full, prefix_mask)

                probs2_g = probs2_u + guidance_scale * (probs2_c - probs2_u)
                _clamp_mask_(probs2_g, prefix_full, prefix_mask)

                score_next = _score_from_probs(
                    probs2_g,
                    x_pred,
                    sigma_next,
                    is_cont_tokens=self.is_cont_tokens,
                )
                d_next = -sigma_next * score_next
                _zero_mask_(d_next, prefix_mask)

            else:
                sig_next_B = sigma_eval_next.expand(B)
                cond2 = x0_hat_cur if self.sc_enabled else torch.zeros_like(x_pred)

                if cond_enabled:
                    _clamp_mask_(x_pred, prefix_full, prefix_mask)
                    if self.sc_enabled:
                        _clamp_mask_(cond2, prefix_full, prefix_mask)

                logits2 = _model_logits_continuous(self.model, self.cfg, x_pred, sig_next_B, cond2)
                probs2 = logits_to_x0_hat(
                    logits2,
                    dtype=x.dtype,
                    is_cont_tokens=self.is_cont_tokens,
                )

                if cond_enabled:
                    _clamp_mask_(probs2, prefix_full, prefix_mask)

                score_next = _score_from_probs(
                    probs2,
                    x_pred,
                    sigma_next,
                    is_cont_tokens=self.is_cont_tokens,
                )
                d_next = -sigma_next * score_next
                _zero_mask_(d_next, prefix_mask)

            x = x_state + 0.5 * h * (d_cur + d_next)
            if cond_enabled:
                _clamp_mask_(x, prefix_full, prefix_mask)

            # ------------------------------------------------------------
            # 3) SC refresh
            # ------------------------------------------------------------
            if self.sc_enabled:
                if use_cfg:
                    if sc_refresh_mode == "refined":
                        x_ref_cat = torch.cat([x, x], dim=0)
                        sig_ref_cat = sigma_eval_next.expand(2 * B)

                        _clamp_mask_(x_ref_cat[:B], prefix_full, prefix_mask)
                        _clamp_mask_(x_ref_cat[B:], null_full, prefix_mask)

                        cond_ref_cat = torch.cat([x0_hat_cur_c, x0_hat_cur_u], dim=0)
                        _clamp_mask_(cond_ref_cat[:B], prefix_full, prefix_mask)
                        _clamp_mask_(cond_ref_cat[B:], null_full, prefix_mask)

                        logits_ref_cat = _model_logits_continuous(
                            self.model,
                            self.cfg,
                            x_ref_cat,
                            sig_ref_cat,
                            cond_ref_cat,
                        )
                        x0_hat_c = logits_to_x0_hat(
                            logits_ref_cat[:B],
                            dtype=x.dtype,
                            is_cont_tokens=self.is_cont_tokens,
                        )
                        x0_hat_u = logits_to_x0_hat(
                            logits_ref_cat[B:],
                            dtype=x.dtype,
                            is_cont_tokens=self.is_cont_tokens,
                        )
                        _clamp_mask_(x0_hat_c, prefix_full, prefix_mask)
                        _clamp_mask_(x0_hat_u, null_full, prefix_mask)
                    else:
                        x0_hat_c = x0_hat_cur_c
                        x0_hat_u = x0_hat_cur_u
                else:
                    if sc_refresh_mode == "refined":
                        sig_ref_B = sigma_eval_next.expand(B)
                        logits_ref = _model_logits_continuous(
                            self.model,
                            self.cfg,
                            x,
                            sig_ref_B,
                            x0_hat_cur,
                        )
                        x0_hat = logits_to_x0_hat(
                            logits_ref,
                            dtype=x.dtype,
                            is_cont_tokens=self.is_cont_tokens,
                        )
                        if cond_enabled:
                            _clamp_mask_(x0_hat, prefix_full, prefix_mask)
                    else:
                        x0_hat = x0_hat_cur

        # ------------------------------------------------------------
        # Final denoised probabilities
        # ------------------------------------------------------------
        # Keep the existing public return_probs contract unchanged:
        #   - binary: returns (x, probs [B,S])
        #   - tokens: returns (x, probs [B,S,V])
        if return_probs:
            sigma_final = _ati_shift_sigma_label(
                sigmas[-1],
                sigmas[-2] if len(sigmas) > 1 else None,
                ati_eta,
            )

            if use_cfg:
                x_cat = torch.cat([x, x], dim=0)
                sig_cat = sigma_final.expand(2 * B)

                _clamp_mask_(x_cat[:B], prefix_full, prefix_mask)
                _clamp_mask_(x_cat[B:], null_full, prefix_mask)

                if self.sc_enabled:
                    cond_cat = torch.cat([x0_hat_c, x0_hat_u], dim=0)
                else:
                    cond_cat = torch.zeros_like(x_cat)

                _clamp_mask_(cond_cat[:B], prefix_full, prefix_mask)
                _clamp_mask_(cond_cat[B:], null_full, prefix_mask)

                logits_cat = _model_logits_continuous(
                    self.model,
                    self.cfg,
                    x_cat,
                    sig_cat,
                    cond_cat,
                )

                probs_c = logits_to_x0_hat(
                    logits_cat[:B],
                    dtype=x.dtype,
                    is_cont_tokens=self.is_cont_tokens,
                )
                probs_u = logits_to_x0_hat(
                    logits_cat[B:],
                    dtype=x.dtype,
                    is_cont_tokens=self.is_cont_tokens,
                )

                _clamp_mask_(probs_c, prefix_full, prefix_mask)
                _clamp_mask_(probs_u, null_full, prefix_mask)

                probs_g = probs_u + guidance_scale * (probs_c - probs_u)
                _clamp_mask_(probs_g, prefix_full, prefix_mask)

                return x, probs_g

            sig_B = sigma_final.expand(B)
            cond_in = x0_hat if self.sc_enabled else torch.zeros_like(x)

            logits = _model_logits_continuous(
                self.model,
                self.cfg,
                x,
                sig_B,
                cond_in,
            )

            probs = logits_to_x0_hat(
                logits,
                dtype=x.dtype,
                is_cont_tokens=self.is_cont_tokens,
            )

            if cond_enabled:
                _clamp_mask_(probs, prefix_full, prefix_mask)

            return x, probs

        # ------------------------------------------------------------
        # Token-only generation fix:
        # decode tokens from the final denoised categorical distribution,
        # not from the noisy continuous state x.
        # ------------------------------------------------------------
        if self.is_cont_tokens:
            sigma_final = _ati_shift_sigma_label(
                sigmas[-1],
                sigmas[-2] if len(sigmas) > 1 else None,
                ati_eta,
            )

            if use_cfg:
                x_cat = torch.cat([x, x], dim=0)
                sig_cat = sigma_final.expand(2 * B)

                _clamp_mask_(x_cat[:B], prefix_full, prefix_mask)
                _clamp_mask_(x_cat[B:], null_full, prefix_mask)

                if self.sc_enabled:
                    cond_cat = torch.cat([x0_hat_c, x0_hat_u], dim=0)
                else:
                    cond_cat = torch.zeros_like(x_cat)

                _clamp_mask_(cond_cat[:B], prefix_full, prefix_mask)
                _clamp_mask_(cond_cat[B:], null_full, prefix_mask)

                logits_cat = _model_logits_continuous(
                    self.model,
                    self.cfg,
                    x_cat,
                    sig_cat,
                    cond_cat,
                )

                probs_c = logits_to_x0_hat(
                    logits_cat[:B],
                    dtype=x.dtype,
                    is_cont_tokens=True,
                )
                probs_u = logits_to_x0_hat(
                    logits_cat[B:],
                    dtype=x.dtype,
                    is_cont_tokens=True,
                )

                _clamp_mask_(probs_c, prefix_full, prefix_mask)
                _clamp_mask_(probs_u, null_full, prefix_mask)

                probs_out = probs_u + guidance_scale * (probs_c - probs_u)
                _clamp_mask_(probs_out, prefix_full, prefix_mask)

            else:
                sig_B = sigma_final.expand(B)
                cond_in = x0_hat if self.sc_enabled else torch.zeros_like(x)

                if cond_enabled and self.sc_enabled:
                    _clamp_mask_(cond_in, prefix_full, prefix_mask)

                logits = _model_logits_continuous(
                    self.model,
                    self.cfg,
                    x,
                    sig_B,
                    cond_in,
                )

                probs_out = logits_to_x0_hat(
                    logits,
                    dtype=x.dtype,
                    is_cont_tokens=True,
                )

                if cond_enabled:
                    _clamp_mask_(probs_out, prefix_full, prefix_mask)

            if probs_out.dim() != 3:
                raise RuntimeError(
                    f"Expected final continuous-token probabilities [B,S,V], "
                    f"got {tuple(probs_out.shape)}"
                )

            return probs_out.argmax(dim=-1)

        # Binary branch unchanged: return the final continuous state.
        return x


# -----------------------------------------------------------------------------
# DDIM sampler
# -----------------------------------------------------------------------------

class DDIMSampler:
    """
    First-order ODE sampler (Euler/DDIM-style) with:
      - centering fix
      - prompt conditioning
      - CFG via fused 2B batching
      - configurable self-conditioning refresh mode
      - ATI support
      - EDM stochastic churn support
      - continuous binary and continuous one-hot token support
    """

    def _integrate_step(
        self,
        x_state: torch.Tensor,
        h: torch.Tensor,
        d_cur: torch.Tensor,
        *,
        sigma_cur: torch.Tensor,
        sigma_next: torch.Tensor,
        prefix_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """First-order probability-flow (DDIM/Euler) update: x_{i+1} = x_i + h·d_i.

        Factored out so reverse-SDE subclasses (EulerMaruyamaSampler) can inject
        a LambdaProfile-gated Langevin term WITHOUT touching the shared
        denoise/SC/CFG machinery in `sample`. Base implementation is
        behavior-preserving (guarded by the DDIM regression / Gate 0).
        """
        return x_state + h * d_cur

    def __init__(self, model, forward_process, cfg):
        self.model = model
        self.process = forward_process
        self.cfg = cfg
        self.device = _infer_model_device(model, cfg.device)
        self.sigmas = SigmaSchedule(self.process, self.cfg, self.device)
        self.sc_enabled = bool(getattr(self.cfg.model, "self_condition", False))

        data_center = 0.5
        try:
            data_center = float(
                getattr(getattr(self.cfg.diffusion, "continuous", object()), "data_center", 0.5)
            )
        except Exception:
            data_center = 0.5
        self.data_center = float(data_center)

        self.repr_mode = str(getattr(cfg.data, "representation", "binary")).lower()
        self.is_cont_tokens = (self.repr_mode == "tokens")
        self.vocab_size = int(getattr(cfg.data, "vocab_size", 1)) if self.is_cont_tokens else 1

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        seq_len: int,
        *,
        conditioning_prefix_full: Optional[torch.Tensor] = None,
        cond_prefix_mask: Optional[torch.Tensor] = None,
        conditioning_prefix: Optional[torch.Tensor] = None,
        cond_len_bits: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        schedule: Optional[str] = None,
        num_steps: Optional[int] = None,
        entropic_blend_alpha: Optional[float] = None,
        entropy_run_dir: Optional[Path] = None,
        sigma_min_override: Optional[float] = None,
        sigma_max_override: Optional[float] = None,
        sc_refresh_mode: str = "refined",
        ati_eta: Optional[float] = None,
        return_probs: bool = False,
        progress: bool = True,
    ):
        sc_refresh_mode = _normalize_sc_refresh_mode(sc_refresh_mode)
        ati_eta = _resolve_ati_eta(self.cfg, ati_eta)

        B = int(num_samples)
        S = int(seq_len)

        sigmas = self.sigmas.prepare(
            schedule=schedule,
            num_steps=num_steps,
            entropic_blend_alpha=entropic_blend_alpha,
            entropy_run_dir=entropy_run_dir,
            sigma_min_override=sigma_min_override,
            sigma_max_override=sigma_max_override,
        )
        stoch_cfg = self.sigmas.resolve_stochastic_cfg(
            entropy_run_dir=entropy_run_dir,
            sigma_min_override=sigma_min_override,
            sigma_max_override=sigma_max_override,
        )
        sigma0 = sigmas[0]

        cond_enabled, prefix_full, prefix_mask, null_full = _build_mask_conditioning(
            cfg=self.cfg,
            B=B,
            S=S,
            device=self.device,
            conditioning_prefix_full=conditioning_prefix_full,
            cond_prefix_mask=cond_prefix_mask,
            conditioning_prefix=conditioning_prefix,
            cond_len_bits=cond_len_bits,
            is_cont_tokens=self.is_cont_tokens,
            vocab_size=self.vocab_size,
        )

        if guidance_scale is None:
            guidance_scale = float(
                getattr(getattr(self.cfg, "evaluation", object()), "guidance_scale", 0.0)
            )
        guidance_scale = float(guidance_scale)
        use_cfg = bool(cond_enabled and (guidance_scale > 0.0))

        if self.is_cont_tokens:
            x = torch.randn(B, S, self.vocab_size, device=self.device, dtype=torch.float32) * sigma0
        else:
            x = torch.randn(B, S, device=self.device, dtype=torch.float32) * sigma0
        x = x + self.data_center

        if cond_enabled:
            _clamp_mask_(x, prefix_full, prefix_mask)

        if self.sc_enabled:
            if use_cfg:
                x0_hat_c = torch.zeros_like(x)
                x0_hat_u = torch.zeros_like(x)
                _clamp_mask_(x0_hat_c, prefix_full, prefix_mask)
                _clamp_mask_(x0_hat_u, null_full, prefix_mask)
            else:
                x0_hat = torch.zeros_like(x)
                if cond_enabled:
                    _clamp_mask_(x0_hat, prefix_full, prefix_mask)
        else:
            x0_hat_c = x0_hat_u = None
            x0_hat = None

        indices = range(len(sigmas) - 1)
        if progress:
            indices = tqdm(indices, desc="DDIM Sampler", leave=False)

        for i in indices:
            sigma_cur, sigma_next = sigmas[i], sigmas[i + 1]
            sigma_prev = sigmas[i - 1] if i > 0 else None

            if cond_enabled:
                _clamp_mask_(x, prefix_full, prefix_mask)

            # ------------------------------------------------------------
            # Optional EDM-style stochastic churn before the denoiser call.
            # ------------------------------------------------------------
            gamma_i = _compute_edm_gamma(
                sigma_cur,
                num_intervals=max(1, len(sigmas) - 1),
                s_churn=stoch_cfg.s_churn,
                s_tmin=stoch_cfg.s_tmin,
                s_tmax=stoch_cfg.s_tmax,
            ) if stoch_cfg.enabled else 0.0

            x_state, sigma_state = _apply_edm_churn(
                x,
                sigma_cur,
                gamma=gamma_i,
                s_noise=stoch_cfg.s_noise,
            )

            if cond_enabled:
                _clamp_mask_(x_state, prefix_full, prefix_mask)

            sigma_eval_cur = _ati_shift_sigma_label(sigma_state, sigma_prev, ati_eta)
            sigma_eval_next = _ati_shift_sigma_label(sigma_next, sigma_state, ati_eta)
            h = sigma_next - sigma_state

            # ------------------------------------------------------------
            # Evaluate at (x_state, sigma_state)
            # ------------------------------------------------------------
            if use_cfg:
                x_cat = torch.cat([x_state, x_state], dim=0)
                sig_cat = sigma_eval_cur.expand(2 * B)

                _clamp_mask_(x_cat[:B], prefix_full, prefix_mask)
                _clamp_mask_(x_cat[B:], null_full, prefix_mask)

                if self.sc_enabled:
                    cond_cat = torch.cat([x0_hat_c, x0_hat_u], dim=0)
                    _clamp_mask_(cond_cat[:B], prefix_full, prefix_mask)
                    _clamp_mask_(cond_cat[B:], null_full, prefix_mask)
                else:
                    cond_cat = torch.zeros_like(x_cat)

                logits_cat = _model_logits_continuous(self.model, self.cfg, x_cat, sig_cat, cond_cat)
                probs_c = logits_to_x0_hat(
                    logits_cat[:B],
                    dtype=x.dtype,
                    is_cont_tokens=self.is_cont_tokens,
                )
                probs_u = logits_to_x0_hat(
                    logits_cat[B:],
                    dtype=x.dtype,
                    is_cont_tokens=self.is_cont_tokens,
                )

                x0_hat_cur_c = probs_c
                x0_hat_cur_u = probs_u

                _clamp_mask_(x0_hat_cur_c, prefix_full, prefix_mask)
                _clamp_mask_(x0_hat_cur_u, null_full, prefix_mask)
                _clamp_mask_(probs_c, prefix_full, prefix_mask)
                _clamp_mask_(probs_u, null_full, prefix_mask)

                probs_g = probs_u + guidance_scale * (probs_c - probs_u)
                _clamp_mask_(probs_g, prefix_full, prefix_mask)

                score_cur = _score_from_probs(
                    probs_g,
                    x_state,
                    sigma_state,
                    is_cont_tokens=self.is_cont_tokens,
                )
                d_cur = -sigma_state * score_cur
                _zero_mask_(d_cur, prefix_mask)

                x = self._integrate_step(
                    x_state, h, d_cur,
                    sigma_cur=sigma_state, sigma_next=sigma_next, prefix_mask=prefix_mask,
                )
                if cond_enabled:
                    _clamp_mask_(x, prefix_full, prefix_mask)

                if self.sc_enabled:
                    if sc_refresh_mode == "refined":
                        x_ref_cat = torch.cat([x, x], dim=0)
                        sig_ref_cat = sigma_eval_next.expand(2 * B)

                        _clamp_mask_(x_ref_cat[:B], prefix_full, prefix_mask)
                        _clamp_mask_(x_ref_cat[B:], null_full, prefix_mask)

                        cond_ref_cat = torch.cat([x0_hat_cur_c, x0_hat_cur_u], dim=0)
                        _clamp_mask_(cond_ref_cat[:B], prefix_full, prefix_mask)
                        _clamp_mask_(cond_ref_cat[B:], null_full, prefix_mask)

                        logits_ref_cat = _model_logits_continuous(
                            self.model,
                            self.cfg,
                            x_ref_cat,
                            sig_ref_cat,
                            cond_ref_cat,
                        )
                        x0_hat_c = logits_to_x0_hat(
                            logits_ref_cat[:B],
                            dtype=x.dtype,
                            is_cont_tokens=self.is_cont_tokens,
                        )
                        x0_hat_u = logits_to_x0_hat(
                            logits_ref_cat[B:],
                            dtype=x.dtype,
                            is_cont_tokens=self.is_cont_tokens,
                        )
                        _clamp_mask_(x0_hat_c, prefix_full, prefix_mask)
                        _clamp_mask_(x0_hat_u, null_full, prefix_mask)
                    else:
                        x0_hat_c = x0_hat_cur_c
                        x0_hat_u = x0_hat_cur_u

            else:
                sig_B = sigma_eval_cur.expand(B)
                cond_in = x0_hat if self.sc_enabled else torch.zeros_like(x_state)

                if cond_enabled:
                    _clamp_mask_(x_state, prefix_full, prefix_mask)
                    if self.sc_enabled:
                        _clamp_mask_(cond_in, prefix_full, prefix_mask)

                logits = _model_logits_continuous(self.model, self.cfg, x_state, sig_B, cond_in)
                probs = logits_to_x0_hat(
                    logits,
                    dtype=x.dtype,
                    is_cont_tokens=self.is_cont_tokens,
                )

                x0_hat_cur = probs
                if cond_enabled:
                    _clamp_mask_(x0_hat_cur, prefix_full, prefix_mask)
                    _clamp_mask_(probs, prefix_full, prefix_mask)

                score_cur = _score_from_probs(
                    probs,
                    x_state,
                    sigma_state,
                    is_cont_tokens=self.is_cont_tokens,
                )
                d_cur = -sigma_state * score_cur
                _zero_mask_(d_cur, prefix_mask)

                x = self._integrate_step(
                    x_state, h, d_cur,
                    sigma_cur=sigma_state, sigma_next=sigma_next, prefix_mask=prefix_mask,
                )
                if cond_enabled:
                    _clamp_mask_(x, prefix_full, prefix_mask)

                if self.sc_enabled:
                    if sc_refresh_mode == "refined":
                        sig_next_B = sigma_eval_next.expand(B)
                        logits_ref = _model_logits_continuous(
                            self.model,
                            self.cfg,
                            x,
                            sig_next_B,
                            x0_hat_cur,
                        )
                        x0_hat = logits_to_x0_hat(
                            logits_ref,
                            dtype=x.dtype,
                            is_cont_tokens=self.is_cont_tokens,
                        )
                        if cond_enabled:
                            _clamp_mask_(x0_hat, prefix_full, prefix_mask)
                    else:
                        x0_hat = x0_hat_cur

        # ------------------------------------------------------------
        # Final denoised probabilities
        # ------------------------------------------------------------
        # Keep the existing public return_probs contract unchanged:
        #   - binary: returns (x, probs [B,S])
        #   - tokens: returns (x, probs [B,S,V])
        if return_probs:
            sigma_final = _ati_shift_sigma_label(
                sigmas[-1],
                sigmas[-2] if len(sigmas) > 1 else None,
                ati_eta,
            )

            if use_cfg:
                x_cat = torch.cat([x, x], dim=0)
                sig_cat = sigma_final.expand(2 * B)

                _clamp_mask_(x_cat[:B], prefix_full, prefix_mask)
                _clamp_mask_(x_cat[B:], null_full, prefix_mask)

                if self.sc_enabled:
                    cond_cat = torch.cat([x0_hat_c, x0_hat_u], dim=0)
                else:
                    cond_cat = torch.zeros_like(x_cat)

                _clamp_mask_(cond_cat[:B], prefix_full, prefix_mask)
                _clamp_mask_(cond_cat[B:], null_full, prefix_mask)

                logits_cat = _model_logits_continuous(
                    self.model,
                    self.cfg,
                    x_cat,
                    sig_cat,
                    cond_cat,
                )

                probs_c = logits_to_x0_hat(
                    logits_cat[:B],
                    dtype=x.dtype,
                    is_cont_tokens=self.is_cont_tokens,
                )
                probs_u = logits_to_x0_hat(
                    logits_cat[B:],
                    dtype=x.dtype,
                    is_cont_tokens=self.is_cont_tokens,
                )

                _clamp_mask_(probs_c, prefix_full, prefix_mask)
                _clamp_mask_(probs_u, null_full, prefix_mask)

                probs_g = probs_u + guidance_scale * (probs_c - probs_u)
                _clamp_mask_(probs_g, prefix_full, prefix_mask)

                return x, probs_g

            sig_B = sigma_final.expand(B)
            cond_in = x0_hat if self.sc_enabled else torch.zeros_like(x)

            logits = _model_logits_continuous(
                self.model,
                self.cfg,
                x,
                sig_B,
                cond_in,
            )

            probs = logits_to_x0_hat(
                logits,
                dtype=x.dtype,
                is_cont_tokens=self.is_cont_tokens,
            )

            if cond_enabled:
                _clamp_mask_(probs, prefix_full, prefix_mask)

            return x, probs

        # ------------------------------------------------------------
        # Token-only generation fix:
        # decode tokens from the final denoised categorical distribution,
        # not from the noisy continuous state x.
        # ------------------------------------------------------------
        if self.is_cont_tokens:
            sigma_final = _ati_shift_sigma_label(
                sigmas[-1],
                sigmas[-2] if len(sigmas) > 1 else None,
                ati_eta,
            )

            if use_cfg:
                x_cat = torch.cat([x, x], dim=0)
                sig_cat = sigma_final.expand(2 * B)

                _clamp_mask_(x_cat[:B], prefix_full, prefix_mask)
                _clamp_mask_(x_cat[B:], null_full, prefix_mask)

                if self.sc_enabled:
                    cond_cat = torch.cat([x0_hat_c, x0_hat_u], dim=0)
                else:
                    cond_cat = torch.zeros_like(x_cat)

                _clamp_mask_(cond_cat[:B], prefix_full, prefix_mask)
                _clamp_mask_(cond_cat[B:], null_full, prefix_mask)

                logits_cat = _model_logits_continuous(
                    self.model,
                    self.cfg,
                    x_cat,
                    sig_cat,
                    cond_cat,
                )

                probs_c = logits_to_x0_hat(
                    logits_cat[:B],
                    dtype=x.dtype,
                    is_cont_tokens=True,
                )
                probs_u = logits_to_x0_hat(
                    logits_cat[B:],
                    dtype=x.dtype,
                    is_cont_tokens=True,
                )

                _clamp_mask_(probs_c, prefix_full, prefix_mask)
                _clamp_mask_(probs_u, null_full, prefix_mask)

                probs_out = probs_u + guidance_scale * (probs_c - probs_u)
                _clamp_mask_(probs_out, prefix_full, prefix_mask)

            else:
                sig_B = sigma_final.expand(B)
                cond_in = x0_hat if self.sc_enabled else torch.zeros_like(x)

                if cond_enabled and self.sc_enabled:
                    _clamp_mask_(cond_in, prefix_full, prefix_mask)

                logits = _model_logits_continuous(
                    self.model,
                    self.cfg,
                    x,
                    sig_B,
                    cond_in,
                )

                probs_out = logits_to_x0_hat(
                    logits,
                    dtype=x.dtype,
                    is_cont_tokens=True,
                )

                if cond_enabled:
                    _clamp_mask_(probs_out, prefix_full, prefix_mask)

            if probs_out.dim() != 3:
                raise RuntimeError(
                    f"Expected final continuous-token probabilities [B,S,V], "
                    f"got {tuple(probs_out.shape)}"
                )

            return probs_out.argmax(dim=-1)

        # Binary branch unchanged: return the final continuous state.
        return x

class EulerMaruyamaSampler(DDIMSampler):
    """Explicit entropy-gated reverse-SDE sampler (Euler-Maruyama).

    Reverse SDE (see docs/EM_PC_SAMPLER_PLAN.md and the entropy-gated-SDE note):
        dx = (1 + lambda(sigma)) * sigma * s_theta dr + sqrt(2 lambda sigma) dW_r
    Discretized (codebase h/d convention, h = sigma_next - sigma_cur < 0,
    d = -sigma * score, Delta = sigma_cur - sigma_next > 0):
        x_det = x + h * (1 + lambda) * d
        x_new = x_det + sqrt(2 * lambda * sigma_cur * Delta) * z,  z ~ N(0, I)

    Reuses DDIMSampler's denoise / self-conditioning / CFG / prefix-clamp
    machinery verbatim and overrides ONLY the per-step integrator. Stochasticity
    is owned entirely by the LambdaProfile; EDM-style churn is refused. With
    lambda_zero == 0 the integrator returns x + h*d and makes NO randn call, so
    the sampler is bit-identical to deterministic DDIM (Gate 1).
    """

    def __init__(
        self,
        model,
        forward_process,
        cfg,
        *,
        lambda_profile_name: str = "entropy_rate",
        lambda_zero: float = 0.0,
        lambda_profile_normalize: str = "peak",
    ):
        super().__init__(model, forward_process, cfg)
        if float(lambda_zero) < 0.0:
            raise ValueError(f"lambda_zero must be >= 0, got {lambda_zero}")
        self.lambda_profile_name = str(lambda_profile_name)
        self.lambda_zero = float(lambda_zero)
        self.lambda_profile_normalize = str(lambda_profile_normalize)
        self._current_profile = None

    def _build_profile(self, entropy_run_dir):
        from diffusion.continuous.lambda_profiles import (
            FlatLambdaProfile,
            make_lambda_profile,
        )
        name = self.lambda_profile_name.lower().strip()
        if self.lambda_zero <= 0.0 or name in {"flat", "constant"}:
            return FlatLambdaProfile(lambda_zero=self.lambda_zero)
        if entropy_run_dir is None:
            entropy_run_dir = self.sigmas._default_entropy_run_dir()
        return make_lambda_profile(
            self.lambda_profile_name,
            lambda_zero=self.lambda_zero,
            entropy_run_dir=entropy_run_dir,
            device=self.device,
            normalize=self.lambda_profile_normalize,
        )

    def _integrate_step(
        self,
        x_state,
        h,
        d_cur,
        *,
        sigma_cur,
        sigma_next,
        prefix_mask,
    ):
        # Determinism gate: no profile eval, no randn -> identical to DDIM PF step.
        if self.lambda_zero == 0.0:
            return x_state + h * d_cur
        lam = self._current_profile.evaluate(sigma_cur, state=x_state)
        x_det = x_state + h * (1.0 + lam) * d_cur
        delta_i = (sigma_cur - sigma_next).clamp_min(0.0)
        z = torch.randn_like(x_state)
        if prefix_mask is not None:
            _zero_mask_(z, prefix_mask)
        sigma_noise = (2.0 * lam * sigma_cur * delta_i).clamp_min(0.0).sqrt()
        return x_det + sigma_noise * z

    @torch.no_grad()
    def sample(self, *args, entropy_run_dir=None, **kwargs):
        st = getattr(getattr(self.cfg, "evaluation", object()), "stochastic", None)
        if st is not None and bool(getattr(st, "enabled", False)) and float(getattr(st, "s_churn", 0.0)) > 0.0:
            raise RuntimeError(
                "EulerMaruyamaSampler does not support EDM-style churn. "
                "Set cfg.evaluation.stochastic.enabled=False / s_churn=0; "
                "stochasticity is controlled by lambda_zero / the LambdaProfile."
            )
        self._current_profile = self._build_profile(entropy_run_dir)
        try:
            return super().sample(*args, entropy_run_dir=entropy_run_dir, **kwargs)
        finally:
            self._current_profile = None


class PredictorCorrectorSampler(DDIMSampler):
    """PF-ODE predictor + LambdaProfile-gated Langevin corrector (entropy-gated SDE).

    Per step sigma_i -> sigma_{i+1} (h = sigma_next - sigma_cur < 0,
    Delta = sigma_cur - sigma_next > 0):

      Predictor (PF-ODE Euler at sigma_i, guided score with weight w):
        score_p = (D_pred - x)/sigma_i^2 ; x_tilde = x + h*(-sigma_i*score_p)
      Corrector (Langevin at sigma_{i+1}, gated by lambda):
        eta = lambda * sigma_{i+1} * Delta
        x_new = x_tilde + eta*score_c + sqrt(2*eta)*z

    Guidance modes (constructor `guidance_mode`):
      - 'predictor_only' (default, the entropy-gated-SDE-correct CFG): the
        PREDICTOR uses the guided score s_u + w(s_c - s_u); the CORRECTOR uses
        the plain CONDITIONAL score (w_corr = 1.0). This removes the (1+lambda)
        amplification of the guidance term that naive guide-everywhere CFG
        suffers under stochastic sampling, while keeping guidance on transport.
      - 'all': both predictor and corrector use the guided score (naive CFG).

    Asymmetry only engages when w > 1 (actual guidance); for w in {0,1} both
    calls coincide. Stochasticity is owned by the LambdaProfile; EDM churn is
    refused. lambda_zero=0 => corrector adds nothing (no randn) => deterministic
    PF predictor (NOT bit-identical to DDIM when self-conditioning is on, since
    PC carries the corrector's sigma_next estimate as SC).
    """

    def __init__(
        self,
        model,
        forward_process,
        cfg,
        *,
        lambda_profile_name: str = "entropy_rate",
        lambda_zero: float = 0.0,
        lambda_profile_normalize: str = "peak",
        corrector_step_rule: str = "em_match",
        guidance_mode: str = "predictor_only",
    ):
        super().__init__(model, forward_process, cfg)
        if float(lambda_zero) < 0.0:
            raise ValueError(f"lambda_zero must be >= 0, got {lambda_zero}")
        if corrector_step_rule not in ("em_match", "sigma_cur"):
            raise ValueError(f"corrector_step_rule must be em_match|sigma_cur, got {corrector_step_rule!r}")
        if guidance_mode not in ("predictor_only", "all"):
            raise ValueError(f"guidance_mode must be predictor_only|all, got {guidance_mode!r}")
        self.lambda_profile_name = str(lambda_profile_name)
        self.lambda_zero = float(lambda_zero)
        self.lambda_profile_normalize = str(lambda_profile_normalize)
        self.corrector_step_rule = str(corrector_step_rule)
        self.guidance_mode = str(guidance_mode)
        self._current_profile = None

    def _build_profile(self, entropy_run_dir):
        from diffusion.continuous.lambda_profiles import FlatLambdaProfile, make_lambda_profile
        name = self.lambda_profile_name.lower().strip()
        if self.lambda_zero <= 0.0 or name in {"flat", "constant"}:
            return FlatLambdaProfile(lambda_zero=self.lambda_zero)
        if entropy_run_dir is None:
            entropy_run_dir = self.sigmas._default_entropy_run_dir()
        return make_lambda_profile(
            self.lambda_profile_name, lambda_zero=self.lambda_zero,
            entropy_run_dir=entropy_run_dir, device=self.device,
            normalize=self.lambda_profile_normalize,
        )

    def _denoise_probs(self, x_state, sigma_eval, sc_cond, *, prefix_full, prefix_mask,
                       null_full, cond_enabled, guidance_scale, B):
        """Faithful copy of DDIMSampler's per-step denoise. Returns (probs_used, sc_carry).
        cfg path (cond & w>0): sc_cond is a (c,u) tuple, returns (probs_g, (probs_c,probs_u)).
        non-cfg path:          sc_cond is a single tensor,  returns (probs, probs).
        """
        use_cfg = bool(cond_enabled and (guidance_scale > 0.0))
        if use_cfg:
            x_cat = torch.cat([x_state, x_state], dim=0)
            sig_cat = sigma_eval.expand(2 * B)
            _clamp_mask_(x_cat[:B], prefix_full, prefix_mask)
            _clamp_mask_(x_cat[B:], null_full, prefix_mask)
            if self.sc_enabled:
                cond_cat = torch.cat([sc_cond[0], sc_cond[1]], dim=0)
                _clamp_mask_(cond_cat[:B], prefix_full, prefix_mask)
                _clamp_mask_(cond_cat[B:], null_full, prefix_mask)
            else:
                cond_cat = torch.zeros_like(x_cat)
            logits_cat = _model_logits_continuous(self.model, self.cfg, x_cat, sig_cat, cond_cat)
            probs_c = logits_to_x0_hat(logits_cat[:B], dtype=x_state.dtype, is_cont_tokens=self.is_cont_tokens)
            probs_u = logits_to_x0_hat(logits_cat[B:], dtype=x_state.dtype, is_cont_tokens=self.is_cont_tokens)
            _clamp_mask_(probs_c, prefix_full, prefix_mask)
            _clamp_mask_(probs_u, null_full, prefix_mask)
            probs_g = probs_u + guidance_scale * (probs_c - probs_u)
            _clamp_mask_(probs_g, prefix_full, prefix_mask)
            return probs_g, (probs_c, probs_u)
        else:
            sig_B = sigma_eval.expand(B)
            cond_in = sc_cond if self.sc_enabled else torch.zeros_like(x_state)
            if cond_enabled:
                _clamp_mask_(x_state, prefix_full, prefix_mask)
                if self.sc_enabled:
                    _clamp_mask_(cond_in, prefix_full, prefix_mask)
            logits = _model_logits_continuous(self.model, self.cfg, x_state, sig_B, cond_in)
            probs = logits_to_x0_hat(logits, dtype=x_state.dtype, is_cont_tokens=self.is_cont_tokens)
            if cond_enabled:
                _clamp_mask_(probs, prefix_full, prefix_mask)
            return probs, probs

    @torch.no_grad()
    def sample(self, num_samples, seq_len, *, conditioning_prefix_full=None,
               cond_prefix_mask=None, conditioning_prefix=None, cond_len_bits=None,
               guidance_scale=None, schedule=None, num_steps=None, entropic_blend_alpha=None,
               entropy_run_dir=None, sigma_min_override=None, sigma_max_override=None,
               sc_refresh_mode="refined", ati_eta=None, return_probs=False, progress=True):
        st = getattr(getattr(self.cfg, "evaluation", object()), "stochastic", None)
        if st is not None and bool(getattr(st, "enabled", False)) and float(getattr(st, "s_churn", 0.0)) > 0.0:
            raise RuntimeError("PredictorCorrectorSampler refuses EDM churn; set stochastic.enabled=False / s_churn=0. Use lambda_zero.")
        sc_refresh_mode = _normalize_sc_refresh_mode(sc_refresh_mode)
        ati_eta = _resolve_ati_eta(self.cfg, ati_eta)
        B, S = int(num_samples), int(seq_len)
        self._current_profile = self._build_profile(entropy_run_dir)
        try:
            sigmas = self.sigmas.prepare(schedule=schedule, num_steps=num_steps,
                entropic_blend_alpha=entropic_blend_alpha, entropy_run_dir=entropy_run_dir,
                sigma_min_override=sigma_min_override, sigma_max_override=sigma_max_override)
            sigma0 = sigmas[0]
            cond_enabled, prefix_full, prefix_mask, null_full = _build_mask_conditioning(
                cfg=self.cfg, B=B, S=S, device=self.device,
                conditioning_prefix_full=conditioning_prefix_full, cond_prefix_mask=cond_prefix_mask,
                conditioning_prefix=conditioning_prefix, cond_len_bits=cond_len_bits,
                is_cont_tokens=self.is_cont_tokens, vocab_size=self.vocab_size)
            if guidance_scale is None:
                guidance_scale = float(getattr(getattr(self.cfg, "evaluation", object()), "guidance_scale", 0.0))
            w = float(guidance_scale)
            w_pred = w
            # corrector uses conditional (w=1) under predictor_only when guidance is active (w>1)
            w_corr = 1.0 if (self.guidance_mode == "predictor_only" and w > 1.0) else w
            use_cfg = bool(cond_enabled and (w_pred > 0.0))

            if self.is_cont_tokens:
                x = torch.randn(B, S, self.vocab_size, device=self.device, dtype=torch.float32) * sigma0
            else:
                x = torch.randn(B, S, device=self.device, dtype=torch.float32) * sigma0
            x = x + self.data_center
            if cond_enabled:
                _clamp_mask_(x, prefix_full, prefix_mask)

            # init SC state matching the predictor's cfg mode
            if self.sc_enabled:
                if use_cfg:
                    sc_state = (torch.zeros_like(x), torch.zeros_like(x))
                    _clamp_mask_(sc_state[0], prefix_full, prefix_mask)
                    _clamp_mask_(sc_state[1], null_full, prefix_mask)
                else:
                    sc_state = torch.zeros_like(x)
                    if cond_enabled:
                        _clamp_mask_(sc_state, prefix_full, prefix_mask)
            else:
                sc_state = None

            indices = range(len(sigmas) - 1)
            if progress:
                indices = tqdm(indices, desc="PC Sampler", leave=False)

            for i in indices:
                sigma_cur, sigma_next = sigmas[i], sigmas[i + 1]
                sigma_prev = sigmas[i - 1] if i > 0 else None
                if cond_enabled:
                    _clamp_mask_(x, prefix_full, prefix_mask)
                sigma_eval_cur = _ati_shift_sigma_label(sigma_cur, sigma_prev, ati_eta)
                sigma_eval_next = _ati_shift_sigma_label(sigma_next, sigma_cur, ati_eta)
                h = sigma_next - sigma_cur
                delta_i = (sigma_cur - sigma_next).clamp_min(0.0)

                # ---- Predictor (guided) ----
                probs_p, sc_carry_p = self._denoise_probs(
                    x, sigma_eval_cur, sc_state, prefix_full=prefix_full, prefix_mask=prefix_mask,
                    null_full=null_full, cond_enabled=cond_enabled, guidance_scale=w_pred, B=B)
                score_p = _score_from_probs(probs_p, x, sigma_cur, is_cont_tokens=self.is_cont_tokens)
                d_pred = -sigma_cur * score_p
                _zero_mask_(d_pred, prefix_mask)
                x_tilde = x + h * d_pred
                if cond_enabled:
                    _clamp_mask_(x_tilde, prefix_full, prefix_mask)

                # ---- Corrector (conditional under predictor_only) ----
                probs_cc, sc_carry_c = self._denoise_probs(
                    x_tilde, sigma_eval_next, sc_carry_p, prefix_full=prefix_full, prefix_mask=prefix_mask,
                    null_full=null_full, cond_enabled=cond_enabled, guidance_scale=w_corr, B=B)

                if self.lambda_zero == 0.0:
                    x = x_tilde
                else:
                    score_c = _score_from_probs(probs_cc, x_tilde, sigma_next, is_cont_tokens=self.is_cont_tokens)
                    _zero_mask_(score_c, prefix_mask)
                    lam = self._current_profile.evaluate(sigma_next, state=x_tilde)
                    if self.corrector_step_rule == "em_match":
                        eta = (lam * sigma_next * delta_i).clamp_min(0.0)
                    else:
                        eta = (lam * sigma_cur * delta_i).clamp_min(0.0)
                    z = torch.randn_like(x_tilde)
                    _zero_mask_(z, prefix_mask)
                    x = x_tilde + eta * score_c + (2.0 * eta).clamp_min(0.0).sqrt() * z
                if cond_enabled:
                    _clamp_mask_(x, prefix_full, prefix_mask)

                # ---- SC carry across steps ----
                if self.sc_enabled:
                    if sc_refresh_mode == "refined":
                        _, sc_state = self._denoise_probs(
                            x, sigma_eval_next, sc_carry_c, prefix_full=prefix_full, prefix_mask=prefix_mask,
                            null_full=null_full, cond_enabled=cond_enabled, guidance_scale=w_corr, B=B)
                    else:
                        sc_state = sc_carry_c

            # ---- final denoised probs (binary return_probs contract) ----
            if return_probs:
                sigma_final = _ati_shift_sigma_label(sigmas[-1], sigmas[-2] if len(sigmas) > 1 else None, ati_eta)
                probs_final, _ = self._denoise_probs(
                    x, sigma_final, sc_state, prefix_full=prefix_full, prefix_mask=prefix_mask,
                    null_full=null_full, cond_enabled=cond_enabled, guidance_scale=w_pred, B=B)
                return x, probs_final
            return x
        finally:
            self._current_profile = None
