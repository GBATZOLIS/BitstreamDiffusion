#diffusion/continuous/logit_postprocess.py
from __future__ import annotations

from typing import Optional
import torch

def canonicalize_continuous_logits(
    logits: torch.Tensor,
    *,
    is_cont_tokens: bool,
) -> torch.Tensor:
    """
    Canonicalize continuous-model logits to the public pipeline shape.

    Public contract outside the model:
      - binary mode: [B,S]
      - token mode:  [B,S,V]

    The model may internally emit binary logits as [B,S,1]; this helper
    converts them to [B,S].
    """
    if is_cont_tokens:
        if logits.dim() != 3:
            raise ValueError(
                f"Continuous token mode expects logits [B,S,V], got {tuple(logits.shape)}"
            )
        return logits

    # binary mode
    if logits.dim() == 2:
        return logits

    if logits.dim() == 3 and logits.size(-1) == 1:
        return logits.squeeze(-1)

    raise ValueError(
        f"Continuous binary mode expects logits [B,S] or [B,S,1], got {tuple(logits.shape)}"
    )


def apply_continuous_logit_postprocessing(
    logits: torch.Tensor,
    sigma: torch.Tensor,
    *,
    mode: str = "none",
    x_t: Optional[torch.Tensor] = None,
    data_center: float = 0.5,
    matched_filter_center: float | None = None,
    matched_filter_scale: float = 1.0,
    matched_filter_clip: float | None = None,
    is_cont_tokens: bool = False,
) -> torch.Tensor:
    """
    Apply continuous-output postprocessing outside the compiled model.

    Public output contract:
      - binary mode: returns logits [B,S]
      - token mode:  returns logits [B,S,V]

    Supported modes:
      - none
      - inv_sigma2
      - matched_filter_residual
      - matched_filter_only

    Notes:
      * binary runs typically use:
            data_center = 0.5
            matched_filter_center = 0.5
      * continuous one-hot token runs often use:
            data_center = 1 / V         (for input centering)
            matched_filter_center = 0.5 or data_center, depending on design
    """
    mode = str(mode).lower()
    if mode == "matched_filter":
        mode = "matched_filter_residual"

    logits = canonicalize_continuous_logits(logits, is_cont_tokens=is_cont_tokens)

    if mode == "none":
        return logits

    if mode == "inv_sigma2":
        if logits.dim() == 2:
            sigma2 = (sigma.to(dtype=logits.dtype) ** 2).view(-1, 1)
        elif logits.dim() == 3:
            sigma2 = (sigma.to(dtype=logits.dtype) ** 2).view(-1, 1, 1)
        else:
            raise ValueError(f"Unexpected logits ndim={logits.dim()}, expected 2 or 3")

        sigma2 = sigma2.clamp_min(torch.finfo(logits.dtype).tiny)
        return logits / sigma2

    if mode in {"matched_filter_residual", "matched_filter_only"}:
        if x_t is None:
            raise ValueError(f"{mode} requires x_t")

        center = float(data_center if matched_filter_center is None else matched_filter_center)
        x_f32 = x_t.to(torch.float32)

        if is_cont_tokens:
            if x_f32.dim() != 3:
                raise ValueError(
                    f"Continuous token matched-filter expects x_t [B,S,V], got {tuple(x_t.shape)}"
                )
            sigma2 = sigma.to(torch.float32).square().view(-1, 1, 1).clamp_min(1e-12)
        else:
            if x_f32.dim() != 2:
                raise ValueError(
                    f"Continuous binary matched-filter expects x_t [B,S], got {tuple(x_t.shape)}"
                )
            sigma2 = sigma.to(torch.float32).square().view(-1, 1).clamp_min(1e-12)

        mf = matched_filter_scale * (x_f32 - center) / sigma2

        if matched_filter_clip is not None:
            c = float(matched_filter_clip)
            mf = mf.clamp(-c, c)

        if logits.shape != mf.shape:
            raise ValueError(
                f"Shape mismatch in apply_continuous_logit_postprocessing: "
                f"logits.shape={tuple(logits.shape)} vs mf.shape={tuple(mf.shape)}"
            )

        mf = mf.to(dtype=logits.dtype)

        if mode == "matched_filter_only":
            return mf
        return logits + mf

    raise ValueError(f"Unknown continuous_logit_scaling='{mode}'")

def _is_cont_tokens_cfg(cfg) -> bool:
    return str(getattr(cfg.data, "representation", "binary")).lower() == "tokens"


def _model_logits_continuous(
    model,
    cfg,
    x_t: torch.Tensor,
    sigma: torch.Tensor,
    x0_hat: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Shared cfg-aware helper for continuous models.

    Returns canonical public logits:
      - binary mode: [B,S]
      - token mode:  [B,S,V]
    """
    logits_raw = model(x_t, sigma, x0_hat)

    return apply_continuous_logit_postprocessing(
        logits_raw,
        sigma,
        mode=str(getattr(cfg.model, "continuous_logit_scaling", "none")),
        x_t=x_t,
        data_center=float(getattr(cfg.diffusion.continuous, "data_center", 0.5)),
        matched_filter_center=float(
            getattr(
                cfg.model,
                "matched_filter_center",
                getattr(cfg.diffusion.continuous, "data_center", 0.5),
            )
        ),
        matched_filter_scale=float(getattr(cfg.model, "matched_filter_scale", 1.0)),
        matched_filter_clip=getattr(cfg.model, "matched_filter_clip", None),
        is_cont_tokens=_is_cont_tokens_cfg(cfg),
    )
    