import torch
import torch.nn.functional as F

__all__ = ["DiscreteForwardProcess"]


_LOG_SCORE_CLAMP = 30.0
_F32_TINY = torch.finfo(torch.float32).tiny


def _stable_expm1_sigma_f32(sigma: torch.Tensor) -> torch.Tensor:
    """
    Compute exp(sigma) - 1 in float32 with a stable small-sigma branch.
    Returns strictly positive values clamped away from zero.
    """
    sigma_f32 = sigma.to(torch.float32)
    out = torch.where(
        sigma_f32 < 0.5,
        torch.expm1(sigma_f32),
        torch.exp(sigma_f32) - 1.0,
    )
    return out.clamp_min(_F32_TINY)


def _clamped_log_scores_f32(log_score: torch.Tensor) -> torch.Tensor:
    """
    Clamp model log-scores in float32 for stable downstream exp/log algebra.
    """
    return log_score.to(torch.float32).clamp(-_LOG_SCORE_CLAMP, _LOG_SCORE_CLAMP)


class _UniformGraph:
    """Fully-connected uniform graph. Analytic ops; no matrix_exp."""

    def __init__(self, dim: int):
        self.dim = dim
        self.absorb = False
        self.mask_id = None  # not used

    # --- RATE MATRIX API (needed by Euler) --------------------------------
    def rate(self, i: torch.Tensor) -> torch.Tensor:
        # i: (B,S) -> (B,S,V)
        V = self.dim
        edge = torch.ones(*i.shape, V, device=i.device, dtype=torch.float32) / V
        edge = edge.scatter(-1, i.unsqueeze(-1), -(V - 1) / V)
        return edge

    def transp_rate(self, i: torch.Tensor) -> torch.Tensor:
        # symmetric for uniform graph
        return self.rate(i)

    def reverse_rate(self, i: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """
        Q^T(x) ⊙ score, with diagonal set to negative row-sum so rows sum to 0.
        i: (B,S), score: (B,S,V) in prob-space (exp(log-score)).
        """
        rr = self.transp_rate(i) * score  # (B,S,V)
        rr.scatter_(-1, i.unsqueeze(-1), 0.0)
        rr.scatter_(-1, i.unsqueeze(-1), -rr.sum(dim=-1, keepdim=True))
        return rr

    def sample_rate(self, i: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        """
        Sample from one_hot(i) + rate (not a prob simplex; requires nonnegativity).
        """
        V = self.dim
        w = F.one_hot(i, num_classes=V).to(rate.dtype) + rate
        w = w.clamp_min(0)
        B, S, Vp = w.shape
        idx = torch.multinomial(w.view(-1, Vp), 1).view(B, S)
        return idx

    # ---- forward transition row for ΔΣ (row-stochastic) ------------------
    # Consistent with sample_transition: with prob move = 1 - e^{-ΔΣ}, redraw token
    # uniformly in {0..V-1}
    # → P(i->i) = e^{-ΔΣ} + (1 - e^{-ΔΣ})/V,  P(i->j≠i) = (1 - e^{-ΔΣ})/V
    def transition_row(self, i: torch.Tensor, sigma_delta: torch.Tensor) -> torch.Tensor:
        # i: (B,S), sigma_delta: (B,)
        B, S = i.shape
        V = self.dim

        ds = sigma_delta.view(B, 1, 1).to(device=i.device, dtype=torch.float32)
        stay_cont = torch.exp(-ds)
        move = 1.0 - stay_cont

        p_change = move / V
        p_stay = stay_cont + p_change

        row = p_change.expand(B, S, V).clone()
        row.scatter_(2, i.unsqueeze(-1), p_stay.expand(B, S, 1))
        return row

    # transpose transition equals transition for uniform graph
    def transp_transition(self, i: torch.Tensor, sigma_delta: torch.Tensor) -> torch.Tensor:
        return self.transition_row(i, sigma_delta)

    # x_t ~ p(. | x0, Σ)
    def sample_transition(self, x0: torch.Tensor, sigma_total: torch.Tensor) -> torch.Tensor:
        move = (1.0 - torch.exp(-sigma_total)).view(-1, 1)
        rand = torch.rand_like(x0, dtype=torch.float32)
        new = torch.randint(0, self.dim, x0.shape, device=x0.device)
        return torch.where(rand < move, new, x0)

    # e^{-ΔΣ Q} s   (Tweedie “staggered score”)
    def staggered_score(self, score: torch.Tensor, sigma_delta: torch.Tensor) -> torch.Tensor:
        # score: (B,S,V)
        B, S, V = score.shape
        ds = sigma_delta.view(B, 1, 1).to(score.dtype)
        epow = torch.exp(-ds)
        mean = score.mean(dim=-1, keepdim=True)
        return ((epow - 1.0) / (V * epow)) * mean + score / epow

    def score_entropy(self, log_score, sigma_total, x, x0):
        """
        Uniform-graph score entropy.

        Returns float32 tensor of shape [B, S].
        All numerically sensitive algebra is done in float32.
        """
        B, S, V = log_score.shape
        V_f32 = float(V)

        # Stable exp(sigma) - 1 and associated ratio
        esigm1_f32 = _stable_expm1_sigma_f32(sigma_total).view(B, 1)  # [B,1]
        ratio_f32 = esigm1_f32 / (esigm1_f32 + V_f32)                 # [B,1]
        ratio_safe = ratio_f32.clamp_min(_F32_TINY)

        # Single clamped fp32 view of logits
        ls32 = _clamped_log_scores_f32(log_score)                     # [B,S,V]

        # Negative term in log-space
        ls_mean = ls32.mean(dim=-1)                                  # [B,S]
        ls_x = torch.gather(ls32, -1, x.unsqueeze(-1)).squeeze(-1)   # [B,S]
        base_neg = ls_mean - (ls_x / V_f32)                          # [B,S]

        same = (x == x0)
        ls_x0 = torch.gather(ls32, -1, x0.unsqueeze(-1)).squeeze(-1) # [B,S]

        neg_term = torch.where(
            same,
            ratio_f32 * base_neg,
            (ls_x0 / esigm1_f32) + base_neg,
        )                                                            # [B,S]

        # Constants
        const_same = ((V_f32 - 1.0) / V_f32) * ratio_f32 * (torch.log(ratio_safe) - 1.0)  # [B,1]
        const_diff = ((-torch.log(ratio_safe) - 1.0) / ratio_safe - (V_f32 - 2.0)) / V_f32 # [B,1]
        const = torch.where(same, const_same, const_diff)                                      # [B,S] via broadcast

        # Positive term in prob-space
        exp_all = torch.exp(ls32)                                  # [B,S,V]
        exp_mean = exp_all.mean(dim=-1)                            # [B,S]
        exp_x = torch.gather(exp_all, -1, x.unsqueeze(-1)).squeeze(-1)  # [B,S]
        pos_term = exp_mean - (exp_x / V_f32)                      # [B,S]

        return pos_term - neg_term + const                         # float32


class _AbsorbingGraph:
    """Absorbing graph with a MASK token id (last index)."""

    def __init__(self, dim: int, mask_token_id: int):
        self.dim = dim
        self.mask_id = int(mask_token_id)
        self.absorb = True

    # --- RATE MATRIX API (needed by Euler) --------------------------------
    def rate(self, i: torch.Tensor) -> torch.Tensor:
        """
        Forward generator row Q(x):
        if i != MASK: Q[i, MASK] = 1, Q[i,i] = -1
        if i == MASK: all zeros (absorbing)
        """
        B, S = i.shape
        V = self.dim
        m = self.mask_id

        i = i.long()
        out = torch.zeros(B, S, V, device=i.device, dtype=torch.float32)

        not_mask = (i != m)

        out[..., m] = not_mask.to(torch.float32)
        out.scatter_(-1, i.unsqueeze(-1), (-not_mask.to(torch.float32)).unsqueeze(-1))
        return out

    def transp_rate(self, i: torch.Tensor) -> torch.Tensor:
        """
        Row i of Q^T == column i of Q (off-diagonals only needed for reverse_rate).

        For absorbing diffusion:
        - If current state i == MASK: incoming rates from every non-mask token are 1
        - If current state i != MASK: no incoming off-diagonals (only diag), so zeros
        """
        B, S = i.shape
        V = self.dim
        m = self.mask_id

        i = i.long()
        is_mask = (i == m).unsqueeze(-1)  # (B,S,1)

        base = torch.ones((1, 1, V), device=i.device, dtype=torch.float32)
        base[..., m] = 0.0

        return is_mask.to(torch.float32) * base

    def reverse_rate(self, i: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """
        rr = Q^T(x) ⊙ score, then set diag so rows sum to 0.
        Do it in float32 for stability.
        """
        i = i.long()
        score_f32 = score.to(torch.float32)

        rr = self.transp_rate(i) * score_f32
        rr.scatter_(-1, i.unsqueeze(-1), 0.0)
        rr.scatter_(-1, i.unsqueeze(-1), -rr.sum(dim=-1, keepdim=True))
        return rr

    def sample_rate(self, i: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        V = self.dim
        w = F.one_hot(i, num_classes=V).to(rate.dtype) + rate
        w = w.clamp_min(0)
        B, S, Vp = w.shape
        idx = torch.multinomial(w.view(-1, Vp), 1).view(B, S)
        return idx

    # ---- row-stochastic forward transition for absorbing graph -----------
    # If i != MASK:  P(i->i) = e^{-ΔΣ},  P(i->MASK) = 1 - e^{-ΔΣ}
    # If i == MASK:  P(MASK->MASK) = 1
    def transition_row(self, i: torch.Tensor, sigma_delta: torch.Tensor) -> torch.Tensor:
        B, S = i.shape
        V = self.dim
        ds = sigma_delta.view(B, 1, 1).to(dtype=torch.float32)
        stay = torch.exp(-ds)

        row = torch.zeros(B, S, V, device=i.device, dtype=torch.float32)
        is_mask = (i == self.mask_id).unsqueeze(-1)

        row[..., self.mask_id:self.mask_id + 1] = is_mask.to(row.dtype)

        not_mask = (~is_mask).to(row.dtype)
        row.scatter_(2, i.unsqueeze(-1), (stay * not_mask).expand(B, S, 1))
        row[..., self.mask_id:self.mask_id + 1] += ((1.0 - stay) * not_mask).expand(B, S, 1)
        return row

    # transpose transition (used by sampler/denoiser/predictor)
    def transp_transition(self, i: torch.Tensor, sigma_delta: torch.Tensor) -> torch.Tensor:
        # i: (B,S), sigma_delta: (B,)
        B = i.shape[0]
        sigma = sigma_delta.view(B, 1, 1).to(dtype=torch.float32)

        edge = torch.exp(-sigma) * F.one_hot(i, num_classes=self.dim).to(dtype=torch.float32)

        edge += torch.where(
            i == self.mask_id,
            1.0 - torch.exp(-sigma.squeeze(-1)),
            0.0,
        ).unsqueeze(-1)
        return edge

    def sample_transition(self, x0: torch.Tensor, sigma_total: torch.Tensor) -> torch.Tensor:
        stay = torch.exp(-sigma_total).view(-1, 1)
        rand = torch.rand_like(x0, dtype=torch.float32)
        go_mask = (rand > stay).long()
        return torch.where(go_mask.bool(), torch.full_like(x0, self.mask_id), x0)

    def staggered_score(self, score: torch.Tensor, sigma_delta: torch.Tensor) -> torch.Tensor:
        # score: (B,S,V)
        B, S, V = score.shape
        ds = sigma_delta.view(B, 1, 1).to(score.dtype)
        epow = torch.exp(ds)
        out = score * epow

        sum_scores = score.sum(dim=-1, keepdim=True)
        extra = (1.0 - epow) * sum_scores

        out[..., self.mask_id] = out[..., self.mask_id] + extra.squeeze(-1)
        return out

    def score_entropy(self, log_score, sigma_total, x, x0):
        """
        Absorbing-graph score entropy.

        Returns float32 tensor of shape [B, S].
        Only positions with x == MASK contribute; others are zero.
        """
        B, S, V = log_score.shape

        rel = (x == self.mask_id)  # [B,S]
        if not bool(rel.any()):
            return torch.zeros(B, S, device=log_score.device, dtype=torch.float32)

        # Stable exp(sigma) - 1 and ratio
        esigm1_f32 = _stable_expm1_sigma_f32(sigma_total).view(B, 1)  # [B,1]
        ratio_f32 = 1.0 / esigm1_f32                                  # [B,1]
        ratio_safe = ratio_f32.clamp_min(_F32_TINY)

        # Single clamped fp32 view of logits
        ls32 = _clamped_log_scores_f32(log_score)                     # [B,S,V]

        # Positive term: sum over all non-mask states.
        # Fast path for the common case where MASK is the last index.
        if self.mask_id == (V - 1):
            pos_full = torch.exp(ls32[..., :V - 1]).sum(dim=-1)      # [B,S]
        else:
            exp_all = torch.exp(ls32)                                # [B,S,V]
            pos_full = exp_all.sum(dim=-1) - exp_all[..., self.mask_id]

        # Negative term
        ls_x0 = torch.gather(ls32, -1, x0.unsqueeze(-1)).squeeze(-1) # [B,S]
        neg_full = ratio_f32 * ls_x0                                 # [B,S]

        # Constant
        const_full = ratio_f32 * (torch.log(ratio_safe) - 1.0)       # [B,1]

        rhs = pos_full - neg_full + const_full                       # [B,S]
        return rhs.masked_fill(~rel, 0.0)                            # float32


class _LogLinearNoise:
    """Absorb default: Σ(t) = -log(1 - (1 - eps) t),  dΣ/dt = (1-eps)/(1-(1-eps)t)"""

    def __init__(self, eps: float = 1e-3):
        self.eps = float(eps)

    def total_and_rate(self, t: torch.Tensor):
        sigma_total = -torch.log1p(-(1.0 - self.eps) * t)
        sigma_rate = (1.0 - self.eps) / (1.0 - (1.0 - self.eps) * t)
        return sigma_total, sigma_rate


class _GeometricNoise:
    """Uniform default: Σ(t) = s0^{1-t} s1^t,  dΣ/dt = Σ * (log s1 - log s0)"""

    def __init__(self, sigma_min=1e-3, sigma_max=1.0):
        self.s0 = float(sigma_min)
        self.s1 = float(sigma_max)
        self._log_ratio = torch.log(torch.tensor(self.s1)) - torch.log(torch.tensor(self.s0))

    def total_and_rate(self, t: torch.Tensor):
        sigma_total = (self.s0 ** (1.0 - t)) * (self.s1 ** t)
        sigma_rate = sigma_total * self._log_ratio.to(t.device, t.dtype)
        return sigma_total, sigma_rate


class DiscreteForwardProcess:
    """
    API-compatible shim that hosts the analytic graph + schedules, so the trainer can keep using:
      - sample_xt(x0, t)
      - get_cumulative_noise(t), get_noise_rate(t)
      - score_entropy(), transition_row(), transp_transition(), staggered_score()
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.d_cfg = cfg.diffusion.discrete
        self.vocab_size = cfg.data.vocab_size
        self.mask_token_id = cfg.data.mask_token_id

        q = self.d_cfg.q_matrix_type.lower()
        if q == "absorb":
            self.graph = _AbsorbingGraph(self.vocab_size, self.mask_token_id)
            self.noise = _LogLinearNoise(getattr(self.d_cfg, "eps", 1e-3))
        elif q == "uniform":
            self.graph = _UniformGraph(self.vocab_size)
            self.noise = _GeometricNoise(
                getattr(self.d_cfg, "sigma_min", 1e-3),
                getattr(self.d_cfg, "sigma_max", 1.0),
            )
        else:
            raise ValueError(f"Unknown q_matrix_type: {self.d_cfg.q_matrix_type}")

    # --- schedule bridge (old names) ---
    def get_cumulative_noise(self, t: torch.Tensor) -> torch.Tensor:
        return self.noise_total_and_rate(t)[0]

    def get_noise_rate(self, t: torch.Tensor) -> torch.Tensor:
        return self.noise_total_and_rate(t)[1]

    def noise_total_and_rate(self, t: torch.Tensor):
        return self.noise.total_and_rate(t)

    # --- graph ops ---
    def sample_xt(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sigma, _ = self.noise_total_and_rate(t)
        return self.graph.sample_transition(x0, sigma)

    def transition_row(self, xt: torch.Tensor, sigma_delta: torch.Tensor) -> torch.Tensor:
        return self.graph.transition_row(xt, sigma_delta)

    def transp_transition(self, xt: torch.Tensor, sigma_delta: torch.Tensor) -> torch.Tensor:
        return self.graph.transp_transition(xt, sigma_delta)

    def staggered_score(self, scores: torch.Tensor, sigma_delta: torch.Tensor) -> torch.Tensor:
        return self.graph.staggered_score(scores, sigma_delta)

    def score_entropy(
        self,
        log_scores: torch.Tensor,
        sigma_total: torch.Tensor,
        xt: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        return self.graph.score_entropy(log_scores, sigma_total, xt, x0)

    @property
    def is_absorb(self) -> bool:
        return getattr(self.graph, "absorb", False)

    @property
    def mask_id(self) -> int:
        return getattr(self.graph, "mask_id", -1)