import math
import torch

__all__ = ["dwdse_loss"]

_LN2 = math.log(2.0)


def dwdse_loss(
    log_model_scores: torch.Tensor,  # (B,S,V)
    x0: torch.Tensor,                # (B,S)
    xt: torch.Tensor,                # (B,S)
    forward_process,
    t: torch.Tensor,                 # (B,)
    cfg,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    DWDSE loss.

    Notes:
      - `forward_process.score_entropy(...)` is expected to return a [B,S]
        tensor in float32.
      - All loss reductions are carried out in float32 for numerical stability.
      - If `mask` is provided, loss is averaged only over masked positions when
        cfg.train.loss_normalize_by_seq=True, or summed over masked positions otherwise.
    """
    B = x0.size(0)
    device = x0.device

    sigma, dsigma = forward_process.noise_total_and_rate(t.to(device))
    se = forward_process.score_entropy(log_model_scores, sigma, xt, x0)  # [B,S], float32

    dsigma_f32 = dsigma.view(B, 1).to(torch.float32)
    weighted = dsigma_f32 * se  # [B,S], float32

    normalize_by_seq = bool(getattr(cfg.train, "loss_normalize_by_seq", True))

    if mask is not None:
        m = mask.to(device=weighted.device, dtype=weighted.dtype)
        if normalize_by_seq:
            denom = m.sum(dim=-1).clamp_min(1.0)
            loss_per_ex_nats = (weighted * m).sum(dim=-1) / denom
        else:
            loss_per_ex_nats = (weighted * m).sum(dim=-1)
    else:
        if normalize_by_seq:
            loss_per_ex_nats = weighted.mean(dim=-1)
        else:
            loss_per_ex_nats = weighted.sum(dim=-1)

    loss_nats = loss_per_ex_nats.mean()

    units = str(getattr(cfg.train, "loss_units", "nats")).lower()
    if units in {"bits", "bpd", "bpb", "bits_per_dim", "bits_per_bit"}:
        return loss_nats / _LN2
    if units in {"nats", "nat"}:
        return loss_nats

    raise ValueError(f"Unknown cfg.train.loss_units={units!r} (use 'nats' or 'bpb'/'bpd').")