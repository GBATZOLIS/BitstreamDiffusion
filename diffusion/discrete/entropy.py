# diffusion/discrete/entropy.py
import torch
from tqdm.auto import tqdm

__all__ = [
    "estimate_entropy_rate_uniform",
    "estimate_entropy_rate_absorb",
    "estimate_entropy_rate_batch",
    "entropy_rate_curve",
    "entropy_from_rate",
    "entropy_and_rate_curve",
    # NEW (optional-use): diagnostics helpers / wrappers
    "curl_diagnostics_uniform",
    "curl_diagnostics_absorb",
    "entropy_rate_curve_with_curl",
    "entropy_and_rate_curve_with_curl",
]

@torch.no_grad()
def _project_to_potentials(log_r: torch.Tensor, xt: torch.Tensor, V: int) -> torch.Tensor:
    """
    Project raw predicted log-ratios to the closest matrix of the form lambda_i - lambda_j
    (Frobenius least-squares) to remove per-column bias and enforce 3-cycle consistency.

    Args:
        log_r : (B,S,V) float64. Assumed diagonal j|j is already zeroed.
        xt    : (B,S)   long. Current token at each site: the column index j.
        V     : vocabulary size.

    Returns:
        log_r_proj : (B,S,V) float64, with (lambda_i - lambda_j) subtracted.
    """
    B, S, _ = log_r.shape
    device = log_r.device

    # --- column means \bar A_{·|j} over sites with xt=j ----------------------
    counts = torch.bincount(xt.reshape(-1), minlength=V).to(torch.float64)   # (V,)
    col_sums = torch.zeros(V, V, dtype=torch.float64, device=device)         # (i,j)
    for j in range(V):
        mask = (xt == j)
        if mask.any():
            col_sums[:, j] = log_r[mask].reshape(-1, V).sum(0)

    col_means = torch.where(
        counts[None, :] > 0,
        col_sums / torch.clamp_min(counts[None, :], 1.0),
        torch.zeros_like(col_sums),
    )

    # zero diagonal and (optionally) antisymmetrize to reduce variance
    eye = torch.eye(V, dtype=torch.bool, device=device)
    col_means[eye] = 0.0
    col_means = 0.5 * (col_means - col_means.T)

    # --- solve normal equations: lambda_k = (R_k - C_k)/(2V), gauge sum lambda=0
    R = col_means.sum(1)            # (V,) row-sum over j
    C = col_means.sum(0)            # (V,) col-sum over i
    lam = (R - C) / (2.0 * V)       # (V,)
    lam = lam - lam.mean()

    # --- subtract (lambda_i - lambda_j) at every site ------------------------
    lam_i = lam.view(1, 1, V)             # broadcast over (B,S)
    lam_j = lam[xt].unsqueeze(-1)         # (B,S,1)
    log_r_proj = log_r - (lam_i - lam_j)

    # keep diagonal j|j exactly zero (numerical hygiene)
    log_r_proj.scatter_(-1, xt.unsqueeze(-1), torch.zeros_like(log_r_proj[..., :1]))
    return log_r_proj


# ======================= NEW: diagnostics (vectorized, fast) ==================

@torch.no_grad()
def _batch_mean_matrix(log_r: torch.Tensor, xt: torch.Tensor, V: int) -> torch.Tensor:
    device = log_r.device
    xt_flat = xt.reshape(-1)                                # (B*S,)
    L_flat_T = log_r.reshape(-1, V).T.contiguous()          # (V, B*S)
    col_sums = torch.zeros(V, V, dtype=torch.float64, device=device)
    col_sums.index_add_(1, xt_flat, L_flat_T)
    counts = torch.bincount(xt_flat, minlength=V).to(torch.float64)
    A = torch.where(
        counts[None, :] > 0,
        col_sums / torch.clamp_min(counts[None, :], 1.0),
        torch.zeros_like(col_sums),
    )
    eye = torch.eye(V, dtype=torch.bool, device=device)
    A[eye] = 0.0
    return A

@torch.no_grad()
def _solve_lambda_from_A(A: torch.Tensor) -> torch.Tensor:
    V = A.shape[0]
    R = A.sum(1)
    C = A.sum(0)
    lam = (R - C) / (2.0 * V)
    return lam - lam.mean()

@torch.no_grad()
def _curl_distance(A: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Distance to gradient field: ||A - (lam_i - lam_j)||_F / V."""
    V = A.shape[0]
    gradA = lam.view(-1,1) - lam.view(1,-1)
    resid = A - gradA
    return torch.linalg.norm(resid, ord="fro") / V

@torch.no_grad()
def _sym_part_norm(A: torch.Tensor) -> torch.Tensor:
    """Symmetric-part (curl) magnitude: ||A + A^T||_F / V. Projection-invariant."""
    V = A.shape[0]
    return torch.linalg.norm(A + A.T, ord="fro") / V

@torch.no_grad()
def _anti_part_norm(A: torch.Tensor) -> torch.Tensor:
    """Antisymmetric-part magnitude: ||A - A^T||_F / V. Should drop after subtracting gradient."""
    V = A.shape[0]
    return torch.linalg.norm(A - A.T, ord="fro") / V

@torch.no_grad()
def _grad_component_norm(lam: torch.Tensor) -> torch.Tensor:
    """||lambda_i - lambda_j||_F / V for the best-fit gradient component."""
    V = lam.numel()
    gradA = lam.view(-1,1) - lam.view(1,-1)
    return torch.linalg.norm(gradA, ord="fro") / V

@torch.no_grad()
def _column_bias_std(A: torch.Tensor) -> torch.Tensor:
    """Destination-mean bias: std_j( mean_i A_{i|j} ). Zero after projection on uniform."""
    means = A.mean(dim=0)
    return means.std()

@torch.no_grad()
def curl_diagnostics_uniform(
    model,
    forward_process,
    x0: torch.Tensor,
    t: torch.Tensor,
    clamp: float = 8.0,
):
    """
    UNIFORM kernel diagnostics based on the batch-mean log-ratio matrix A:
      - sym_part_norm            : ||A + A^T||_F/V  (projection-invariant)
      - anti_part_norm_before    : ||A - A^T||_F/V  (before subtracting gradient)
      - anti_part_norm_after     : ...              (after residual; should drop)
      - grad_component_norm      : ||lambda_i - lambda_j||_F/V
      - curl_before/curl_after   : distance to gradient field (after == before)
      - colbias_std_before/after : destination-mean bias (after ≈ 0)
      - ev_explained_variance    : sitewise explained variance by projection
      - energy_before/energy_after: sitewise mean squared magnitude pre/post
    """
    device = x0.device
    V = forward_process.vocab_size
    sigma, _ = forward_process.noise_total_and_rate(t.to(device))
    xt = forward_process.sample_xt(x0, t)

    # raw predictions (float64), zero diag, clamp
    log_r = model(xt, sigma).to(torch.float64)
    if clamp is not None:
        log_r = log_r.clamp(-float(clamp), float(clamp))
    log_r.scatter_(-1, xt.unsqueeze(-1), torch.zeros_like(log_r[..., :1]))

    # --- batch-mean matrix A and gradient fit
    A_before = _batch_mean_matrix(log_r, xt, V)
    lam = _solve_lambda_from_A(A_before)

    grad_norm = _grad_component_norm(lam)
    curl_before = _curl_distance(A_before, lam)
    sym_part = _sym_part_norm(A_before)
    anti_before = _anti_part_norm(A_before)
    colbias_before = _column_bias_std(A_before)

    # residual on A (projection effect there)
    gradA = lam.view(-1,1) - lam.view(1,-1)
    A_resid = A_before - gradA
    curl_after = _curl_distance(A_resid, _solve_lambda_from_A(A_resid))  # ~== curl_before
    anti_after = _anti_part_norm(A_resid)
    colbias_after = _column_bias_std(A_resid)

    # --- NEW: sitewise EV / energies (effect on raw logits across all sites)
    grad_col = lam[xt].unsqueeze(-1)        # (B,S,1)
    grad_i   = lam.view(1,1,-1)             # (1,1,V)
    delta    = grad_i - grad_col            # (B,S,V)

    L_before = log_r
    L_after  = log_r - delta

    num = torch.sum(L_after**2) + 1e-12     # residual energy
    den = torch.sum(L_before**2) + 1e-12    # total energy
    ev_explained_variance = 1.0 - (num / den)

    energy_before = den / L_before.numel()
    energy_after  = num / L_after.numel()

    return {
        "sym_part_norm":           float(sym_part.item()),
        "anti_part_norm_before":   float(anti_before.item()),
        "anti_part_norm_after":    float(anti_after.item()),
        "grad_component_norm":     float(grad_norm.item()),
        "curl_before":             float(curl_before.item()),
        "curl_after":              float(curl_after.item()),
        "colbias_std_before":      float(colbias_before.item()),
        "colbias_std_after":       float(colbias_after.item()),
        # NEW:
        "ev_explained_variance":   float(ev_explained_variance.item()),
        "energy_before":           float(energy_before.item()),
        "energy_after":            float(energy_after.item()),
    }



@torch.no_grad()
def curl_diagnostics_absorb(model, forward_process, x0: torch.Tensor, t: torch.Tensor, clamp: float = 80.0):
    """
    MASK/star kernel: no projection needed; report the same interpretable pieces as 'before'.
    """
    device = x0.device
    V = forward_process.vocab_size
    sigma, _ = forward_process.noise_total_and_rate(t.to(device))
    xt = forward_process.sample_xt(x0, t)

    log_scores = model(xt, sigma).to(torch.float64).clamp(-float(clamp), float(clamp))
    log_scores.scatter_(-1, xt.unsqueeze(-1), torch.zeros_like(log_scores[..., :1]))

    A = _batch_mean_matrix(log_scores, xt, V)
    lam = _solve_lambda_from_A(A)

    return {
        "sym_part_norm":       float(_sym_part_norm(A).item()),
        "anti_part_norm":      float(_anti_part_norm(A).item()),
        "grad_component_norm": float(_grad_component_norm(lam).item()),
        "curl_before":         float(_curl_distance(A, lam).item()),
        "colbias_std":         float(_column_bias_std(A).item()),
    }

# ========================= (unchanged) estimators =============================

@torch.no_grad()
def estimate_entropy_rate_uniform(
    model,
    forward_process,
    x0,
    t,
    clamp: float = 8.0,          # <- small clamp for uniform kernel
    form: str = "current",       # "current" (=U2) or "column" (=U3); see Fix B below
    project: bool = True,        # <- turn off to compare raw vs projected
):
    """
    Uniform kernel ONLY. Returns dH/dt in nats.

    If form=="current": uses (U2)  dotH = (dsigma/V) E[(1-r)*(-log r)].
    If form=="column":  uses (U3)  dotH = (dsigma/V) E[sum_i -log r_{i|j}]  (Fix B).

    With project=True, raw logits are first projected to the closest
    difference-of-potentials gauge to remove per-column bias (Fix A).
    """
    B, S = x0.shape
    V = forward_process.vocab_size
    device = x0.device

    sigma, dsigma = forward_process.noise_total_and_rate(t.to(device))    # (B,), (B,)
    xt = forward_process.sample_xt(x0, t)                                  # (B,S)

    # raw model logits -> float64; tight clamp to avoid late-time blowups
    log_r = model(xt, sigma).to(torch.float64)
    if clamp is not None:
        log_r = log_r.clamp(-float(clamp), float(clamp))

    # enforce r_{j|j}=1, log r_{j|j}=0 exactly
    idx = xt.unsqueeze(-1)                                                 # (B,S,1)
    log_r.scatter_(-1, idx, torch.zeros_like(log_r[..., :1]))

    # ---- Fix A: projection to potentials (works for any V>=2)
    if project:
        log_r = _project_to_potentials(log_r, xt, V)

    if form == "column":
        # ---- Fix B (linear/one-sided form = U3 column form)
        inner_sum = (-log_r).sum(dim=-1)                                   # (B,S)
        scale = (dsigma.view(B, 1).to(torch.float64) / float(V))           # (B,1)
        dotH_per_ex = (scale * inner_sum).sum(dim=-1)                      # (B,)
        return dotH_per_ex.mean()

    # ---- default: current-weighted form (U2) with projection ---------------
    r = torch.exp(log_r)
    r.scatter_(-1, idx, torch.ones_like(r[..., :1]))
    inner = (1.0 - r) * (-log_r)                                           # (B,S,V)
    inner_sum = inner.sum(dim=-1)                                          # (B,S)
    scale = (dsigma.view(B, 1).to(torch.float64) / float(V))               # (B,1)
    dotH_per_ex = (scale * inner_sum).sum(dim=-1)                          # (B,)
    return dotH_per_ex.mean()


@torch.no_grad()
def estimate_entropy_rate_absorb(model, forward_process, x0, t, clamp=80.0):
    B, S = x0.shape
    m = forward_process.mask_id

    sigma, dsigma = forward_process.noise_total_and_rate(t.to(x0.device))  # (B,), (B,)
    xt = forward_process.sample_xt(x0, t)                                   # (B,S)

    log_scores = model(xt, sigma).clamp(-clamp, clamp)                      # (B,S,V)
    r = log_scores.exp()                                                    # (B,S,V)

    is_mask = (xt == m)                                                     # (B,S)
    not_mask = ~is_mask

    # These two pieces were previously added with a + sign; flip them so the
    # function returns dH/dt (which is ≤ 0 for absorbing diffusion)
    neg_log_r_mask = (-log_scores[..., m])                                  # (B,S)
    term_not_mask = -(dsigma.view(B, 1) / 2.0) * (neg_log_r_mask * not_mask).sum(dim=1)  # (B,)

    r_except_m = r[..., :m]
    log_r_except_m = log_scores[..., :m]
    term_mask_pos = (r_except_m * log_r_except_m).sum(-1)                   # (B,S)
    term_mask = -(dsigma.view(B, 1) / 2.0) * (term_mask_pos * is_mask).sum(dim=1)        # (B,)

    return (term_not_mask + term_mask).mean()


@torch.no_grad()
def estimate_entropy_rate_batch(
    model,
    forward_process,
    x0,
    t,
    clamp_uniform: float = 8.0,        # <- tight for uniform (Fix A default)
    clamp_absorb: float = 80.0,        # <- your previous default for masked
    uniform_form: str = "current",     # "current" (U2, with projection) or "column" (U3)
    project_uniform: bool = True,      # apply Fix A?
):
    if getattr(forward_process, "is_absorb", False):
        return estimate_entropy_rate_absorb(model, forward_process, x0, t, clamp=clamp_absorb)
    else:
        return estimate_entropy_rate_uniform(
            model, forward_process, x0, t,
            clamp=clamp_uniform, form=uniform_form, project=project_uniform
        )


@torch.no_grad()
def entropy_rate_curve(
    model,
    forward_process,
    val_loader,
    t_grid,
    device,
    clamp_uniform: float = 8.0,
    clamp_absorb: float = 80.0,
    uniform_form: str = "current",
    project_uniform: bool = True,
    show_progress: bool = True,
    units: str = "bits",
):
    import numpy as np

    times = t_grid.detach().cpu().numpy()
    dotH_list_nats = []

    outer_iter = tqdm(t_grid, total=len(t_grid), desc="Entropy-rate: times", position=0, leave=True) if show_progress else t_grid
    for tk in outer_iter:
        accum = 0.0
        count_batches = 0
        inner_iter = tqdm(val_loader, total=len(val_loader) if show_progress else None,
                          desc=f"t={float(tk):.4f}", position=1, leave=False) if show_progress else val_loader

        for batch in inner_iter:
            x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
            x0 = x0.to(device).view(x0.size(0), -1).long()
            t = torch.full((x0.size(0),), float(tk.item()), device=device)

            est_nats = estimate_entropy_rate_batch(
                model, forward_process, x0, t,
                clamp_uniform=clamp_uniform,
                clamp_absorb=clamp_absorb,
                uniform_form=uniform_form,
                project_uniform=project_uniform,
            )
            accum += float(est_nats.item())
            count_batches += 1

        dotH_list_nats.append(accum / max(1, count_batches))

    dotH = np.asarray(dotH_list_nats, dtype=float)
    if units.lower() == "bits":
        dotH = dotH / np.log(2.0)
    return times, dotH



# -------- integrate rate -> entropy --------------------------------

def entropy_from_rate(
    times,
    dotH,
    seq_len,
    vocab_size,
    is_absorb,
    known_H_end=None,
    sign: int = +1,
    units: str = "bits",
):
    """
    Integrate Ḣ(t) over t to recover H(t). 'dotH' must already be in the requested units.
    sign=+1 if your dotH is actually -dH/dt; sign=-1 if dotH = dH/dt.
    Returns H(t) in the same units as dotH.
    """
    import numpy as np

    # Endpoint in requested units
    if known_H_end is not None:
        H_end = float(known_H_end)
    else:
        if is_absorb:
            H_end = 0.0
        else:
            if units.lower() == "bits":
                H_end = float(seq_len) * float(np.log2(vocab_size))
            else:  # nats
                H_end = float(seq_len) * float(np.log(vocab_size))

    # Right-to-left trapezoidal integration
    H = np.empty_like(dotH)
    H[-1] = H_end
    for k in range(len(times) - 2, -1, -1):
        dt = times[k + 1] - times[k]
        area = 0.5 * (dotH[k + 1] + dotH[k]) * dt
        H[k] = H[k + 1] + sign * area
    return H


@torch.no_grad()
def entropy_and_rate_curve(
    model,
    forward_process,
    val_loader,
    t_grid,
    device,
    clamp_uniform: float = 8.0,
    clamp_absorb: float = 80.0,
    uniform_form: str = "current",   # "current" or "column"
    project_uniform: bool = True,
    show_progress: bool = True,
    known_H_end: float = None,
    sign: int = +1,
    units: str = "bits",
):
    times, dotH = entropy_rate_curve(
        model, forward_process, val_loader, t_grid, device,
        clamp_uniform=clamp_uniform,
        clamp_absorb=clamp_absorb,
        uniform_form=uniform_form,
        project_uniform=project_uniform,
        show_progress=show_progress,
        units=units,
    )
    is_absorb = getattr(forward_process, "is_absorb", False)
    S = getattr(forward_process.cfg.data, "sequence_len", None)
    V = getattr(forward_process, "vocab_size", None)

    H = entropy_from_rate(
        times, dotH, seq_len=S, vocab_size=V, is_absorb=is_absorb,
        known_H_end=known_H_end, sign=sign, units=units
    )
    return times, dotH, H


# ================= NEW: wrappers that add curl diagnostics (opt-in) ===========

@torch.no_grad()
def entropy_rate_curve_with_curl(
    model,
    forward_process,
    val_loader,
    t_grid,
    device,
    clamp_uniform: float = 8.0,
    clamp_absorb: float = 80.0,
    uniform_form: str = "current",
    project_uniform: bool = True,
    show_progress: bool = True,
    units: str = "bits",
):
    """
    Same as entropy_rate_curve but also returns a per-t list of dicts with curl diagnostics.
    Leaves the original entropy_rate_curve unmodified.
    """
    import numpy as np

    is_absorb = getattr(forward_process, "is_absorb", False)
    times = t_grid.detach().cpu().numpy()
    dotH_list_nats = []
    curl_diag_per_t = []

    outer_iter = tqdm(t_grid, total=len(t_grid), desc="Entropy-rate: times", position=0, leave=True) if show_progress else t_grid
    for tk in outer_iter:
        accum = 0.0
        count_batches = 0

        curl_accum = None

        inner_iter = tqdm(val_loader, total=len(val_loader) if show_progress else None,
                          desc=f"t={float(tk):.4f}", position=1, leave=False) if show_progress else val_loader

        for batch_idx, batch in enumerate(inner_iter):
            x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
            x0 = x0.to(device).view(x0.size(0), -1).long()
            t = torch.full((x0.size(0),), float(tk.item()), device=device)

            # estimator (unchanged)
            est_nats = estimate_entropy_rate_batch(
                model, forward_process, x0, t,
                clamp_uniform=clamp_uniform,
                clamp_absorb=clamp_absorb,
                uniform_form=uniform_form,
                project_uniform=project_uniform,
            )
            accum += float(est_nats.item())
            count_batches += 1

            # diagnostics (fast)
            if is_absorb:
                di = curl_diagnostics_absorb(model, forward_process, x0, t, clamp=clamp_absorb)
            else:
                di = curl_diagnostics_uniform(model, forward_process, x0, t, clamp=clamp_uniform)

            if curl_accum is None:
                curl_accum = {k: 0.0 for k in di.keys()}
            for k, v in di.items():
                curl_accum[k] += float(v)

        dotH_list_nats.append(accum / max(1, count_batches))
        if curl_accum is None:
            curl_diag_per_t.append({})
        else:
            avg_di = {k: (v / max(1, count_batches)) for k, v in curl_accum.items()}
            curl_diag_per_t.append(avg_di)

    dotH = np.asarray(dotH_list_nats, dtype=float)
    if units.lower() == "bits":
        dotH = dotH / np.log(2.0)
    return times, dotH, curl_diag_per_t


@torch.no_grad()
def entropy_and_rate_curve_with_curl(
    model,
    forward_process,
    val_loader,
    t_grid,
    device,
    clamp_uniform: float = 8.0,
    clamp_absorb: float = 80.0,
    uniform_form: str = "current",   # "current" or "column"
    project_uniform: bool = True,
    show_progress: bool = True,
    known_H_end: float = None,
    sign: int = +1,
    units: str = "bits",
):
    """
    Same as entropy_and_rate_curve, but returns curl diagnostics too.
    Leaves the original function unmodified.
    """
    times, dotH, curl_diag_per_t = entropy_rate_curve_with_curl(
        model, forward_process, val_loader, t_grid, device,
        clamp_uniform=clamp_uniform,
        clamp_absorb=clamp_absorb,
        uniform_form=uniform_form,
        project_uniform=project_uniform,
        show_progress=show_progress,
        units=units,
    )
    is_absorb = getattr(forward_process, "is_absorb", False)
    S = getattr(forward_process.cfg.data, "sequence_len", None)
    V = getattr(forward_process, "vocab_size", None)

    H = entropy_from_rate(
        times, dotH, seq_len=S, vocab_size=V, is_absorb=is_absorb,
        known_H_end=known_H_end, sign=sign, units=units
    )
    return times, dotH, H, curl_diag_per_t
