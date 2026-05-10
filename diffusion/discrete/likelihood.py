# diffusion/discrete/likelihood.py
import torch
from tqdm import tqdm

__all__ = ["bits_per_dim_dwdse"]


# ------------------------ Utilities: endpoint KL terms ------------------------

@torch.no_grad()
def _kl_forward_to_prior_absorb(sigma_T: torch.Tensor, V: int) -> torch.Tensor:
    """
    KL( p_T|0(·|x0) || p_base ) per POSITION for the absorbing kernel.

    Forward endpoint per position (given initial non-mask token):
        P(mask)   = 1 - stay
        P(x0)     = stay
        P(other)  = 0
    Base prior with full support (finite-time, avoids ∞ KL):
        P_base(mask)  = 1 - stay
        P_base(other) = stay / (V-1)

    Returns: (B,) per-example KL aggregated over S later by caller.
    """
    # sigma_T is (B,), stay = exp(-Sigma(1))
    stay = torch.exp(-sigma_T).clamp(0.0, 1.0)        # (B,)
    p_mask = 1.0 - stay                               # (B,)
    p_nonmask = (stay / max(V - 1, 1))                # (B,)

    # Forward endpoint puts mass only on {mask, original token}
    # KL per position:
    #   stay * log(stay / p_nonmask) + (1 - stay) * log((1 - stay) / p_mask)
    t1 = torch.where(stay > 0, stay * (stay / (p_nonmask + 1e-300)).log(), stay)
    t2 = torch.where(p_mask > 0, p_mask * (p_mask / (p_mask + 1e-300)).log(), p_mask)  # becomes 0
    # The second simplifies to 0, but we keep the pattern for clarity.
    # Explicit form:
    t2 = (1.0 - stay) * torch.where(
        p_mask > 0, ((1.0 - stay) / (p_mask + 1e-300)).log(), torch.zeros_like(p_mask)
    )
    kl = t1 + t2
    return kl  # (B,)


@torch.no_grad()
def _kl_forward_to_prior_uniform(sigma_T: torch.Tensor, V: int) -> torch.Tensor:
    """
    KL( p_T|0(·|x0) || Uniform(V) ) per POSITION for the uniform kernel.

    Forward endpoint per position:
        P(i->i)   = stay + (1 - stay)/V
        P(i->j≠i) = (1 - stay)/V
    Prior: uniform u = 1/V.

    Returns: scalar (B,), KL per position (independent of x0), to be multiplied by S by caller.
    """
    stay = torch.exp(-sigma_T).clamp(0.0, 1.0)     # (B,)
    u = 1.0 / float(V)
    p_same = stay + (1.0 - stay) * u               # (B,)
    p_other = (1.0 - stay) * u                     # (B,)

    # KL per position:
    #   p_same * log(p_same/u) + (V-1) * p_other * log(p_other/u)
    term_same = torch.where(p_same > 0, p_same * (p_same / u).log(), torch.zeros_like(p_same))
    term_other = (V - 1) * torch.where(
        p_other > 0, p_other * (p_other / u).log(), torch.zeros_like(p_other)
    )
    kl = term_same + term_other                    # (B,)
    return kl


@torch.no_grad()
def _endpoint_kl_per_example(forward_process, cfg, B: int, S: int, device) -> torch.Tensor:
    """
    Returns per-example KL(x_T|x0 || p_base) summed over positions: shape (B,).

    Uses the same finite-time priors you already use elsewhere:
      • Absorb: P_base(mask)=1-exp(-Σ(1)), P_base(other)=exp(-Σ(1))/(V-1)
      • Uniform: uniform over V
    """
    V = int(cfg.data.vocab_size)
    Sigma = lambda t: forward_process.noise_total_and_rate(t)[0]

    sigma_T = Sigma(torch.ones(B, device=device))  # (B,)
    if forward_process.is_absorb:
        kl_pos = _kl_forward_to_prior_absorb(sigma_T, V)  # (B,)
    else:
        kl_pos = _kl_forward_to_prior_uniform(sigma_T, V)  # (B,)

    # sum over S independent positions
    return S * kl_pos  # (B,)


# --------------------- Theorem 3.6: DWDSE likelihood bound -------------------

@torch.no_grad()
def bits_per_dim_dwdse(
    model,
    forward_process,          # DiscreteForwardProcess
    cfg,
    x0_tokens: torch.Tensor,  # (B,S) long
    time_samples: int = 64,   # Monte Carlo samples over t ~ U(ε, 1]
    clamp_absorb: float = 80.0,
    clamp_uniform: float = 8.0,
) -> float:
    """
    Implements Theorem 3.6 (SEDD) likelihood upper bound:

        -log p_θ(x0)  ≤  ∫ E_{x_t|x0} [ dΣ(t) * SE(log s_θ(·|x_t), Σ(t), x_t, x0) ] dt
                        + KL( p_{T|0}(·|x0) || p_base )

    We estimate the time integral by Monte Carlo over 'time_samples' draws of t ~ U(ε,1].
    The inner expectation is the same per-position score-entropy used in training (SEDD).

    Returns: bits/dim averaged over the batch.
    """
    assert int(cfg.data.vocab_size) >= 2
    if forward_process.is_absorb:
        assert 0 <= int(cfg.data.mask_token_id) < int(cfg.data.vocab_size)

    device = x0_tokens.device
    B, S = x0_tokens.shape

    # Monte Carlo over times
    # Use the schedule's epsilon if present; default 1e-3
    eps = float(getattr(getattr(cfg.diffusion, "discrete", object()), "eps", 1e-3))
    t_low = eps
    t_high = 1.0

    # Accumulate per-example DWDSE integrand averages
    accum = torch.zeros(B, device=device, dtype=torch.float64)

    for _ in range(time_samples):
        # t ~ Uniform(eps, 1]; vectorized over batch for variance reduction
        t = torch.empty(B, device=device).uniform_(t_low, t_high)  # (B,)
        sigma, dsigma = forward_process.noise_total_and_rate(t)    # (B,), (B,)

        # Sample x_t from the analytic forward kernel
        xt = forward_process.sample_xt(x0_tokens, t)               # (B,S) long

        # Model logits at (xt, Σ(t)); clamp for stability
        log_scores = model(xt, sigma)
        if forward_process.is_absorb:
            log_scores = log_scores.clamp(-clamp_absorb, clamp_absorb)
        else:
            log_scores = log_scores.clamp(-clamp_uniform, clamp_uniform)

        # Per-position score-entropy (same as in training)
        se = forward_process.score_entropy(log_scores, sigma, xt, x0_tokens)  # (B,S) dtype=model
        se = se.to(torch.float64)

        # Diffusion weighting dΣ(t) and sum over positions → per-example
        accum += (dsigma.view(B, 1).to(torch.float64) * se).sum(dim=-1)  # (B,)

    # Average over time samples to approximate the integral (Riemann MC estimator)
    dwdse = accum / max(1, time_samples)  # (B,)

    # Endpoint KL per example (sum over positions internally)
    kl_end = _endpoint_kl_per_example(forward_process, cfg, B, S, device).to(torch.float64)  # (B,)

    # Upper bound (nats), convert to bits/dim
    nll_upper = dwdse + kl_end  # (B,)
    bpd = (nll_upper / (S * torch.log(torch.tensor(2.0, device=device)))).mean().item()
    return bpd