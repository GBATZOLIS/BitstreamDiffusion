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
    posterior_temp: float = 1.0,
    posterior_temp_target: str = "learned",
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

    Posterior temperature (CoBit's continuous analogue of MDLM/Duo low-T
    decoding). With T<1 the per-bit Bernoulli posterior sigmoid(logit/T) is
    sharpened toward 0/1 (the joint mode of the factorized code). Two targets:
      - "learned": sharpen ONLY the network's learned logit, leaving the
        analytic matched-filter data-consistency term at T=1:
            logit_out = logit_raw / T + mf
        This is the principled default (the MF term is the likelihood gradient,
        not a belief to sharpen).
      - "full": sharpen the whole postprocessed logit (logit_raw + mf) / T.
    For non-matched-filter modes the two targets coincide (logit / T).
    T == 1.0 reproduces the untempered output bit-for-bit.
    """
    mode = str(mode).lower()
    if mode == "matched_filter":
        mode = "matched_filter_residual"

    logits = canonicalize_continuous_logits(logits, is_cont_tokens=is_cont_tokens)

    T = float(posterior_temp)
    apply_temp = abs(T - 1.0) > 1e-8
    if apply_temp and T <= 0.0:
        raise ValueError(f"posterior_temp must be > 0, got {T}")
    target = str(posterior_temp_target).lower()

    if mode == "none":
        return logits / T if apply_temp else logits

    if mode == "inv_sigma2":
        if logits.dim() == 2:
            sigma2 = (sigma.to(dtype=logits.dtype) ** 2).view(-1, 1)
        elif logits.dim() == 3:
            sigma2 = (sigma.to(dtype=logits.dtype) ** 2).view(-1, 1, 1)
        else:
            raise ValueError(f"Unexpected logits ndim={logits.dim()}, expected 2 or 3")

        sigma2 = sigma2.clamp_min(torch.finfo(logits.dtype).tiny)
        out = logits / sigma2
        return out / T if apply_temp else out

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
            # No learned component to sharpen; "learned" target leaves mf intact.
            if apply_temp and target == "full":
                return mf / T
            return mf

        # matched_filter_residual
        if apply_temp:
            if target == "learned":
                return logits / T + mf
            if target == "full":
                return (logits + mf) / T
            raise ValueError(f"Unknown posterior_temp_target='{target}'")
        return logits + mf

    raise ValueError(f"Unknown continuous_logit_scaling='{mode}'")

def _is_cont_tokens_cfg(cfg) -> bool:
    return str(getattr(cfg.data, "representation", "binary")).lower() == "tokens"


def _matched_filter_binary(cfg, x_t: torch.Tensor, sigma: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Recompute the binary matched-filter residual mf = scale*(x_t-center)/sigma^2
    (clipped) as a separate tensor [B,S], matching
    apply_continuous_logit_postprocessing. Returns None when the model does not
    use a matched-filter postprocessing mode (then mf is implicitly 0 and the
    'learned'/'full' token targets coincide).
    """
    mode = str(getattr(cfg.model, "continuous_logit_scaling", "none")).lower()
    if mode == "matched_filter":
        mode = "matched_filter_residual"
    if mode not in {"matched_filter_residual", "matched_filter_only"}:
        return None
    center = float(
        getattr(cfg.model, "matched_filter_center",
                getattr(cfg.diffusion.continuous, "data_center", 0.5))
    )
    scale = float(getattr(cfg.model, "matched_filter_scale", 1.0))
    clip = getattr(cfg.model, "matched_filter_clip", None)
    sigma2 = sigma.to(torch.float32).square().view(-1, 1).clamp_min(1e-12)
    mf = scale * (x_t.to(torch.float32) - center) / sigma2
    if clip is not None:
        c = float(clip)
        mf = mf.clamp(-c, c)
    return mf


def _codeword_sharpen(
    raw_logits: torch.Tensor,
    mf: Optional[torch.Tensor],
    *,
    temp: float,
    target: str,
    codebook: torch.Tensor,
    topk: Optional[int] = None,
    chunk_rows: int = 4096,
) -> torch.Tensor:
    """
    Joint, valid-codeword temperature decode (the continuous analogue of MDLM/Duo
    categorical temperature). Per token position with per-bit logits ell:
        full:    L_T(v) = <C_v, ell_raw + mf> / T
        learned: L_T(v) = <C_v, ell_raw> / T + <C_v, mf>
        q_T = softmax over VALID tokens v of L_T(v)
        D_b = E_{v~q_T}[c_b(v)] = (q_T C)_b          (returned as the x0 bit estimate)
    Restricting the softmax to the V valid codewords makes invalid codes
    structurally unreachable, so as T->0 it goes to the joint MAP over valid
    tokens (true greedy) rather than the per-bit MAP. Processed in row-chunks to
    bound the [chunk, V] memory spike.

    raw_logits, mf: [B, S] with S = P * m (m = bits/token).
    codebook: [V, m] float in {0,1}.
    Returns D in (0,1), shape [B, S].
    """
    B, S = raw_logits.shape
    m = int(codebook.shape[1])
    if S % m != 0:
        raise ValueError(f"seq bits {S} not divisible by bits/token {m}")
    Ct = codebook.to(torch.float32).t().contiguous()          # [m, V]
    raw = raw_logits.to(torch.float32).reshape(B * S // m, m)  # [N, m]
    mf_r = mf.to(torch.float32).reshape(B * S // m, m) if mf is not None else None
    T = float(temp)
    full = (str(target).lower() == "full")
    N = raw.shape[0]
    out = torch.empty_like(raw)
    for i in range(0, N, chunk_rows):
        r = raw[i:i + chunk_rows]
        L = r @ Ct                                            # [n, V] = <C_v, ell_raw>
        if mf_r is not None:
            Lm = mf_r[i:i + chunk_rows] @ Ct                  # <C_v, mf>
            L = (L + Lm) / T if full else L / T + Lm
        else:
            L = L / T
        if topk is not None and topk < Ct.shape[1]:
            vals, idx = L.topk(int(topk), dim=-1)             # [n, k]
            q = torch.softmax(vals, dim=-1)
            out[i:i + chunk_rows] = (q.unsqueeze(-1) * codebook.to(torch.float32)[idx]).sum(-2)
        else:
            q = torch.softmax(L, dim=-1)                      # [n, V]
            out[i:i + chunk_rows] = q @ codebook.to(torch.float32)
    return out.reshape(B, S)


def _model_logits_continuous(
    model,
    cfg,
    x_t: torch.Tensor,
    sigma: torch.Tensor,
    x0_hat: Optional[torch.Tensor],
    *,
    posterior_temp: float = 1.0,
    posterior_temp_target: Optional[str] = None,
    pt_ctx: Optional[dict] = None,
) -> torch.Tensor:
    """
    Shared cfg-aware helper for continuous models.

    Returns canonical public logits:
      - binary mode: [B,S]
      - token mode:  [B,S,V]

    posterior_temp: per-call temperature applied to the bit posterior (see
    apply_continuous_logit_postprocessing). Defaults to 1.0 => no change.
    posterior_temp_target: "learned" (default) or "full"; falls back to
    cfg.model.posterior_temp_target then "learned" when None.
    pt_ctx: optional dict enabling joint valid-codeword (token-space) sharpening.
      When pt_ctx["space"] == "token" (binary mode only), the per-bit posterior is
      replaced by the temperature-sharpened expectation over valid codewords and
      returned as a logit logit(D) so the downstream sigmoid recovers D exactly
      (the sampler/score path is unchanged). Keys: space, codebook [V,m], topk.
    """
    target = (
        posterior_temp_target
        if posterior_temp_target is not None
        else str(getattr(cfg.model, "posterior_temp_target", "learned"))
    )

    if (
        pt_ctx is not None
        and str(pt_ctx.get("space", "bit")).lower() == "token"
        and not _is_cont_tokens_cfg(cfg)
    ):
        raw = canonicalize_continuous_logits(
            model(x_t, sigma, x0_hat), is_cont_tokens=False
        )
        mf = _matched_filter_binary(cfg, x_t, sigma)
        D = _codeword_sharpen(
            raw, mf,
            temp=float(posterior_temp),
            target=target,
            codebook=pt_ctx["codebook"],
            topk=pt_ctx.get("topk"),
        )
        D = D.clamp(1e-6, 1.0 - 1e-6)
        return torch.log(D / (1.0 - D)).to(dtype=raw.dtype)  # logit(D); sigmoid recovers D

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
        posterior_temp=float(posterior_temp),
        posterior_temp_target=target,
    )
    