#diffusion/continuous/losses.py
from __future__ import annotations

import inspect
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _HAS_TRITON = False


# -----------------------------------------------------------------------------
# Global runtime guards
# -----------------------------------------------------------------------------
_TOKEN_SM_TRITON_DISABLED_RUNTIME = False
_TOKEN_SM_TRITON_WARNED = False


# ============================================================================
# Common weighting
# ============================================================================

def _sigma_weight(cfg, sigma: torch.Tensor, ndim: int) -> torch.Tensor:
    sigma2 = (sigma.to(torch.float32) ** 2).view(-1, *([1] * (ndim - 1)))
    weighting = str(getattr(cfg.train, "loss_weighting", "edm")).lower()
    sigma_data_f32 = float(cfg.diffusion.continuous.sigma_data)

    if weighting in {"none", "unit", "1"}:
        return torch.ones_like(sigma2)

    if weighting in {"edm", "karras"}:
        return (sigma2 + sigma_data_f32**2) / (sigma2 * (sigma_data_f32**2))

    raise ValueError(f"Unknown cfg.train.loss_weighting='{weighting}'")


# ============================================================================
# Validation helpers
# ============================================================================

def _validate_token_sm_inputs_flat(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.dim() != 2:
        raise ValueError(f"Expected logits [N,V], got {tuple(logits.shape)}")
    if targets.dim() != 1:
        raise ValueError(f"Expected targets [N], got {tuple(targets.shape)}")
    if logits.size(0) != targets.size(0):
        raise ValueError(
            f"logits batch {logits.size(0)} incompatible with targets {targets.size(0)}"
        )
    if logits.size(1) <= 0:
        raise ValueError(f"Expected vocab size > 0, got {logits.size(1)}")

    if targets.numel() > 0:
        tmin = int(targets.min().item())
        tmax = int(targets.max().item())
        if tmin < 0 or tmax >= logits.size(1):
            raise ValueError(
                f"targets out of range for vocab size {logits.size(1)}: min={tmin}, max={tmax}"
            )


def _validate_token_sm_inputs_batched(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.dim() != 3:
        raise ValueError(f"Expected logits [B,S,V], got {tuple(logits.shape)}")
    if targets.dim() != 2:
        raise ValueError(f"Expected targets [B,S], got {tuple(targets.shape)}")

    b, s, v = logits.shape
    if targets.shape != (b, s):
        raise ValueError(
            f"targets shape {tuple(targets.shape)} incompatible with logits {tuple(logits.shape)}"
        )
    if v <= 0:
        raise ValueError(f"Expected vocab size > 0, got {v}")

    if targets.numel() > 0:
        tmin = int(targets.min().item())
        tmax = int(targets.max().item())
        if tmin < 0 or tmax >= v:
            raise ValueError(
                f"targets out of range for vocab size {v}: min={tmin}, max={tmax}"
            )


def _prepare_mask_3d(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        mask_ = mask.unsqueeze(-1)
    elif mask.dim() == 3:
        mask_ = mask
    else:
        raise ValueError(f"mask must have dim 2 or 3, got shape {tuple(mask.shape)}")

    return mask_.to(dtype=ref.dtype, device=ref.device)


# ============================================================================
# Exact PyTorch fallback for token score matching / exact metric
# ============================================================================

def _token_sm_squared_error_exact_chunked_flat(
    logits: torch.Tensor,   # [N, V]
    targets: torch.Tensor,  # [N]
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Exact squared error ||softmax(logits) - one_hot(target)||^2 without
    materializing the full probability tensor.
    """
    _validate_token_sm_inputs_flat(logits, targets)
    chunk_size = max(1, int(chunk_size))

    logits_f32 = logits.to(torch.float32)
    targets = targets.long()

    n, v = logits_f32.shape
    m = logits_f32.amax(dim=-1)  # [N]

    Z = torch.zeros((n,), device=logits_f32.device, dtype=logits_f32.dtype)
    Q = torch.zeros((n,), device=logits_f32.device, dtype=logits_f32.dtype)

    for start in range(0, v, chunk_size):
        end = min(start + chunk_size, v)
        chunk = logits_f32[:, start:end]      # [N, C]
        shifted = chunk - m[:, None]          # [N, C]
        e = torch.exp(shifted)                # [N, C]
        Z = Z + e.sum(dim=-1)                 # [N]
        Q = Q + (e * e).sum(dim=-1)           # [N]

    denom = Z.clamp_min(1e-30)
    l_c = logits_f32.gather(dim=-1, index=targets[:, None]).squeeze(-1)  # [N]
    p_c = torch.exp(l_c - m) / denom
    sum_sq_probs = Q / denom.square()

    # ||p - e_c||^2 = sum_i p_i^2 - 2 p_c + 1
    sq_err = sum_sq_probs - 2.0 * p_c + 1.0
    return sq_err


def _token_sm_squared_error_exact_chunked(
    logits: torch.Tensor,   # [B, S, V]
    targets: torch.Tensor,  # [B, S]
    chunk_size: int = 2048,
) -> torch.Tensor:
    _validate_token_sm_inputs_batched(logits, targets)

    b, s, v = logits.shape
    logits_2d = logits.reshape(b * s, v)
    targets_1d = targets.reshape(b * s)
    sq_err_flat = _token_sm_squared_error_exact_chunked_flat(
        logits_2d,
        targets_1d,
        chunk_size=chunk_size,
    )
    return sq_err_flat.reshape(b, s)


def _token_sm_squared_error_naive_flat(
    logits: torch.Tensor,   # [N, V]
    targets: torch.Tensor,  # [N]
) -> torch.Tensor:
    """
    Naive reference implementation that materializes softmax probabilities
    and one-hot targets. Useful for testing / benchmarking only.
    """
    _validate_token_sm_inputs_flat(logits, targets)

    logits_f32 = logits.to(torch.float32)
    probs = torch.softmax(logits_f32, dim=-1)
    one_hot = F.one_hot(targets.long(), num_classes=logits.size(1)).to(torch.float32)
    return ((probs - one_hot) ** 2).sum(dim=-1)


# ============================================================================
# Triton fused token score-matching kernels
# ============================================================================

def _triton_token_sm_available(x: torch.Tensor) -> bool:
    if not _HAS_TRITON:
        return False
    if not isinstance(x, torch.Tensor):
        return False
    if not x.is_cuda:
        return False
    if not torch.cuda.is_available():
        return False
    if x.dim() != 2:
        return False
    if x.numel() == 0:
        return False
    if x.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return False
    return True


def _triton_autotune_compat(configs, key):
    kwargs = {
        "configs": configs,
        "key": key,
    }
    try:
        sig = inspect.signature(triton.autotune)
        if "cache_results" in sig.parameters:
            kwargs["cache_results"] = True
    except Exception:
        pass
    return triton.autotune(**kwargs)


if _HAS_TRITON:
    _TOKEN_SM_AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=4),
    ]

    @_triton_autotune_compat(
        configs=_TOKEN_SM_AUTOTUNE_CONFIGS,
        key=["V"],
    )
    @triton.jit
    def _token_sm_fwd_kernel(
        logits_ptr,            # [N, V]
        targets_ptr,           # [N] int32
        loss_ptr,              # [N] fp32
        m_ptr,                 # [N] fp32
        invz_ptr,              # [N] fp32
        ssq_ptr,               # [N] fp32
        stride_logits_row,     # stride in elements
        V,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_ptr = logits_ptr + row * stride_logits_row
        target = tl.load(targets_ptr + row)

        offs = tl.arange(0, BLOCK_SIZE)

        m = -float("inf")
        for start in range(0, V, BLOCK_SIZE):
            cols = start + offs
            mask = cols < V
            x = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
            m = tl.maximum(m, tl.max(x, axis=0))

        z = 0.0
        q = 0.0
        for start in range(0, V, BLOCK_SIZE):
            cols = start + offs
            mask = cols < V
            x = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
            e = tl.exp(x - m)
            e = tl.where(mask, e, 0.0)
            z += tl.sum(e, axis=0)
            q += tl.sum(e * e, axis=0)

        invz = 1.0 / z
        ssq = q * invz * invz

        lt = tl.load(row_ptr + target).to(tl.float32)
        pc = tl.exp(lt - m) * invz

        loss = ssq - 2.0 * pc + 1.0

        tl.store(loss_ptr + row, loss)
        tl.store(m_ptr + row, m)
        tl.store(invz_ptr + row, invz)
        tl.store(ssq_ptr + row, ssq)

    @_triton_autotune_compat(
        configs=_TOKEN_SM_AUTOTUNE_CONFIGS,
        key=["V"],
    )
    @triton.jit
    def _token_sm_bwd_kernel(
        logits_ptr,            # [N, V]
        targets_ptr,           # [N] int32
        grad_out_ptr,          # [N] fp32
        grad_logits_ptr,       # [N, V], same dtype as logits tensor
        m_ptr,                 # [N] fp32
        invz_ptr,              # [N] fp32
        ssq_ptr,               # [N] fp32
        stride_logits_row,     # stride in elements
        stride_grad_row,       # stride in elements
        V,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_ptr = logits_ptr + row * stride_logits_row
        grad_row_ptr = grad_logits_ptr + row * stride_grad_row

        target = tl.load(targets_ptr + row)
        grad_out = tl.load(grad_out_ptr + row).to(tl.float32)
        m = tl.load(m_ptr + row)
        invz = tl.load(invz_ptr + row)
        ssq = tl.load(ssq_ptr + row)

        lt = tl.load(row_ptr + target).to(tl.float32)
        pc = tl.exp(lt - m) * invz

        offs = tl.arange(0, BLOCK_SIZE)

        for start in range(0, V, BLOCK_SIZE):
            cols = start + offs
            mask = cols < V

            x = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
            pk = tl.exp(x - m) * invz

            delta = (cols == target).to(tl.float32)

            # dL/dl_k = 2 p_k (p_k - sum_i p_i^2 + p_c) - 2 p_c * 1[k=c]
            grad = 2.0 * pk * (pk - ssq + pc) - 2.0 * pc * delta
            grad = grad * grad_out

            tl.store(grad_row_ptr + cols, grad, mask=mask)


class _FusedTokenSMLoss(torch.autograd.Function):
    """
    CUDA/Triton fast path only.
    Use _token_sm_loss_flat(...) below for safe dispatch.
    """

    @staticmethod
    def forward(ctx, logits_2d: torch.Tensor, targets_1d: torch.Tensor):
        _validate_token_sm_inputs_flat(logits_2d, targets_1d)

        if not _triton_token_sm_available(logits_2d):
            raise RuntimeError("_FusedTokenSMLoss called without Triton/CUDA availability")

        if not logits_2d.is_contiguous():
            logits_2d = logits_2d.contiguous()

        targets_i32 = targets_1d.to(torch.int32).contiguous()

        n_rows, V = logits_2d.shape
        loss = torch.empty((n_rows,), device=logits_2d.device, dtype=torch.float32)
        m = torch.empty_like(loss)
        invz = torch.empty_like(loss)
        ssq = torch.empty_like(loss)

        grid = (n_rows,)
        _token_sm_fwd_kernel[grid](
            logits_2d,
            targets_i32,
            loss,
            m,
            invz,
            ssq,
            logits_2d.stride(0),
            V,
        )

        ctx.save_for_backward(logits_2d, targets_i32, m, invz, ssq)
        ctx.V = V
        return loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        logits_2d, targets_i32, m, invz, ssq = ctx.saved_tensors

        grad_out = grad_out.contiguous().to(torch.float32)
        grad_logits = torch.empty_like(logits_2d)

        grid = (logits_2d.shape[0],)
        _token_sm_bwd_kernel[grid](
            logits_2d,
            targets_i32,
            grad_out,
            grad_logits,
            m,
            invz,
            ssq,
            logits_2d.stride(0),
            grad_logits.stride(0),
            ctx.V,
        )
        return grad_logits, None


def _token_sm_loss_flat(
    logits_2d: torch.Tensor,   # [N, V]
    targets_1d: torch.Tensor,  # [N]
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Safe dispatcher for differentiable token score-matching:
      - Triton fast path on CUDA
      - exact chunked PyTorch fallback elsewhere
      - auto-disables Triton path for this process if runtime launch fails
    """
    global _TOKEN_SM_TRITON_DISABLED_RUNTIME, _TOKEN_SM_TRITON_WARNED

    _validate_token_sm_inputs_flat(logits_2d, targets_1d)

    if (not _TOKEN_SM_TRITON_DISABLED_RUNTIME) and _triton_token_sm_available(logits_2d):
        try:
            return _FusedTokenSMLoss.apply(logits_2d, targets_1d)
        except Exception as e:
            _TOKEN_SM_TRITON_DISABLED_RUNTIME = True
            if not _TOKEN_SM_TRITON_WARNED:
                print(
                    "[token_sm] Triton fused path failed at runtime; "
                    f"disabling it for this process and falling back to exact chunked PyTorch. "
                    f"Error: {type(e).__name__}: {e}"
                )
                _TOKEN_SM_TRITON_WARNED = True

    return _token_sm_squared_error_exact_chunked_flat(
        logits_2d,
        targets_1d,
        chunk_size=chunk_size,
    )


@torch.no_grad()
def _token_sm_metric_flat(
    logits_2d: torch.Tensor,   # [N, V]
    targets_1d: torch.Tensor,  # [N]
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Exact no-grad metric path for ||softmax(logits)-one_hot(target)||^2.

    Uses the Triton forward kernel when available, otherwise falls back to
    exact chunked PyTorch. This is ideal for entropy/rate estimation when the
    optimization loss is CE.
    """
    global _TOKEN_SM_TRITON_DISABLED_RUNTIME, _TOKEN_SM_TRITON_WARNED

    _validate_token_sm_inputs_flat(logits_2d, targets_1d)

    if (not _TOKEN_SM_TRITON_DISABLED_RUNTIME) and _triton_token_sm_available(logits_2d):
        try:
            if not logits_2d.is_contiguous():
                logits_2d = logits_2d.contiguous()

            targets_i32 = targets_1d.to(torch.int32).contiguous()

            n_rows, V = logits_2d.shape
            loss = torch.empty((n_rows,), device=logits_2d.device, dtype=torch.float32)
            m = torch.empty_like(loss)
            invz = torch.empty_like(loss)
            ssq = torch.empty_like(loss)

            grid = (n_rows,)
            _token_sm_fwd_kernel[grid](
                logits_2d,
                targets_i32,
                loss,
                m,
                invz,
                ssq,
                logits_2d.stride(0),
                V,
            )
            return loss

        except Exception as e:
            _TOKEN_SM_TRITON_DISABLED_RUNTIME = True
            if not _TOKEN_SM_TRITON_WARNED:
                print(
                    "[token_sm_metric] Triton forward-only metric path failed at runtime; "
                    f"disabling it for this process and falling back to exact chunked PyTorch. "
                    f"Error: {type(e).__name__}: {e}"
                )
                _TOKEN_SM_TRITON_WARNED = True

    return _token_sm_squared_error_exact_chunked_flat(
        logits_2d,
        targets_1d,
        chunk_size=chunk_size,
    )


@torch.no_grad()
def maybe_warmup_token_sm_triton(
    vocab_size: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    n_rows: int = 4096,
) -> bool:
    """
    Optional one-time warmup/autotune helper.
    Call once per rank on the actual CUDA compute node before the hot loop.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return False
    if not _HAS_TRITON or not torch.cuda.is_available():
        return False
    if _TOKEN_SM_TRITON_DISABLED_RUNTIME:
        return False

    logits = torch.randn((n_rows, vocab_size), device=device, dtype=dtype)
    targets = torch.randint(0, vocab_size, (n_rows,), device=device, dtype=torch.int64)
    _ = _token_sm_metric_flat(logits, targets)
    torch.cuda.synchronize(device)
    return True


# ============================================================================
# Public losses
# ============================================================================

def binary_score_interpolation_loss(
    logits: torch.Tensor,
    x0: torch.Tensor,
    sigma: torch.Tensor,
    cfg,
    return_entropy_metric: bool = False,
    mask: Optional[torch.Tensor] = None,
):
    """
    Weighted denoising objective for binary score interpolation.
    """
    loss_type = str(getattr(cfg.train, "loss_type", "binary_ce")).lower()

    if logits.dim() == 2:
        logits = logits.unsqueeze(-1)
    elif logits.dim() == 3:
        if logits.size(-1) != 1:
            raise ValueError(f"Expected logits last dim = 1 if 3D, got {tuple(logits.shape)}")
    else:
        raise ValueError(f"Expected logits dim 2 or 3, got {tuple(logits.shape)}")

    b = logits.size(0)
    target = x0.view_as(logits).to(torch.float32)
    weight = _sigma_weight(cfg, sigma, ndim=3)

    logits_f32 = logits.to(torch.float32)
    probs = torch.sigmoid(logits_f32)

    if loss_type == "binary_sm":
        per_pos = F.mse_loss(probs, target, reduction="none")
    elif loss_type == "binary_ce":
        per_pos = F.binary_cross_entropy_with_logits(logits_f32, target, reduction="none")
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'")

    if mask is not None:
        mask_ = _prepare_mask_3d(mask, per_pos)
        num = (weight * per_pos * mask_).sum()
        den = mask_.sum().clamp_min(1.0)
        mean_loss = num / den
    else:
        mean_loss = (weight * per_pos).mean()

    if not return_entropy_metric:
        return mean_loss

    sq_err = (probs - target) ** 2
    if mask is not None:
        mask_ = _prepare_mask_3d(mask, sq_err)
        num = (sq_err * mask_).view(b, -1).sum(dim=1)
        den = mask_.view(b, -1).sum(dim=1).clamp_min(1.0)
        entropy_metric = num / den
    else:
        entropy_metric = sq_err.view(b, -1).mean(dim=1)

    return mean_loss, entropy_metric


def token_score_interpolation_loss(
    logits: torch.Tensor,
    x0: torch.Tensor,
    sigma: torch.Tensor,
    cfg,
    return_entropy_metric: bool = False,
    mask: Optional[torch.Tensor] = None,
):
    """
    Continuous one-hot token denoising loss.

    token_sm path:
      - Triton fused exact token score-matching on CUDA
      - exact chunked PyTorch fallback elsewhere

    token_ce path:
      - CE for optimization
      - exact ||p - e_c||^2 entropy proxy via no-grad Triton/chunked helper
    """
    _validate_token_sm_inputs_batched(logits, x0)

    b, s, v = logits.shape
    loss_type = str(getattr(cfg.train, "loss_type", "token_sm")).lower()
    weight = _sigma_weight(cfg, sigma, ndim=3)
    targets = x0.long()

    if loss_type in {"token_sm", "mse"}:
        chunk_size = max(1, int(getattr(cfg.train, "token_sm_chunk_size", 2048)))
        logits_2d = logits.contiguous().reshape(b * s, v)
        targets_1d = targets.contiguous().reshape(b * s)

        loss_flat = _token_sm_loss_flat(
            logits_2d,
            targets_1d,
            chunk_size=chunk_size,
        )  # [B*S]

        sq_err = loss_flat.reshape(b, s)   # [B,S]
        per_pos = sq_err.unsqueeze(-1)     # [B,S,1]

    elif loss_type in {"token_ce", "ce"}:
        chunk_size = max(1, int(getattr(cfg.train, "token_sm_chunk_size", 2048)))
        logits_2d = logits.contiguous().reshape(b * s, v)
        targets_1d = targets.contiguous().reshape(b * s)

        per_pos = F.cross_entropy(
            logits_2d.float(),
            targets_1d,
            reduction="none",
        ).reshape(b, s, 1)

        sq_err = None
        if return_entropy_metric:
            with torch.no_grad():
                sq_err = _token_sm_metric_flat(
                    logits_2d.detach(),
                    targets_1d,
                    chunk_size=chunk_size,
                ).reshape(b, s)

    else:
        raise ValueError(f"Unknown token continuous loss_type '{loss_type}'")

    if mask is not None:
        mask_ = _prepare_mask_3d(mask, per_pos)
        num = (weight * per_pos * mask_).sum()
        den = mask_.sum().clamp_min(1.0)
        mean_loss = num / den
    else:
        mean_loss = (weight * per_pos).mean()

    if not return_entropy_metric:
        return mean_loss

    if sq_err is None:
        entropy_metric = per_pos.squeeze(-1).mean(dim=1)
        return mean_loss, entropy_metric

    if mask is not None:
        mask2 = mask.squeeze(-1) if mask.dim() == 3 else mask
        mask2 = mask2.to(dtype=sq_err.dtype, device=sq_err.device)
        num = (sq_err * mask2).sum(dim=1)
        den = mask2.sum(dim=1).clamp_min(1.0)
        entropy_metric = num / den
    else:
        entropy_metric = sq_err.mean(dim=1)

    return mean_loss, entropy_metric