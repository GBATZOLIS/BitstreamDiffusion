# diffusion/continuous/samplers/lambda_profiles.py
"""
λ(σ) profiles for the entropy-gated reverse SDE samplers.

The Euler-Maruyama (Phase 3) and Predictor-Corrector (Phase 4) samplers
implement the reverse-SDE family

    dx = (1 + λ(σ)) · σ · s_θ(x, σ) · dr + sqrt(2 · λ(σ) · σ) · dW_r       (1)

where r = -σ, s_θ is the score, and λ(σ) ≥ 0 controls the Langevin
strength across noise levels. This module supplies λ.

The asymptotic identification between EDM-style churn (with churn rate
S_churn on an entropy-rate sampling grid) and the explicit reverse SDE
in (1) is, per `app:stochastic_sde_analysis`:

    λ_ent(σ) ≈ S_churn · π_α(log σ)                                       (2)

where π_α(log σ) is the entropy-rate density in log-σ space (training-time
sigma sampling distribution). `EntropyRateLambdaProfile(normalize='as_saved')`
implements λ(σ) = λ₀ · π_α(log σ) so the identity (2) matches **exactly**
under λ₀ = S_churn — this is the form used by the EM-vs-EDM-churn
equivalence test.

Conventions and table format
----------------------------
The on-disk tables `entropy_pdf.pt`, `entropy_sigmas.pt`, optionally
`entropy_edges.pt`, are written by
`utils.schedule_controller.EntropyScheduleController.save_entropy_tables`.
They store:

  - `entropy_sigmas.pt`: 1-D float32 [K] σ midpoints, ASCENDING (the
    production format). Bins are equispaced in log σ over [σ_min, σ_max].
  - `entropy_pdf.pt`:    1-D float32 [K] discrete distribution over bins,
    sums to 1 across the grid. Each entry ≈ π_α(log σ_i) · Δ(log σ).
  - `entropy_edges.pt`:  optional 1-D float32 [K+1] bin edges in σ-space.

Δ(log σ) = (log σ_max − log σ_min) / K is the (constant) bin width in
log-σ space; we prefer to read it from the edges file when present.

`normalize` selects how the saved table is interpreted as λ(σ):

  - `'peak'`:     table = pdf / pdf.max();  λ₀ is the **peak** Langevin
                  strength λ(σ⋆). Default for new code.
  - `'as_saved'`: table = pdf / Δ(log σ);  the table is interpreted as
                  the log-σ density π_α(log σ). Matches identity (2)
                  bit-for-bit.

Out-of-range queries are clamped to table edges (no extrapolation), as
spec'd in the task — entropy-rate density vanishes there in practice.

`LambdaProfile.evaluate` accepts `state` and `history` kwargs that are
currently ignored. They are extension hooks for future trajectory-adaptive
profiles (e.g. Hutchinson Tr J_D-modulated strength).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import torch


_VALID_NORMALIZE = ("peak", "as_saved")


def _load_entropy_pdf_table(
    run_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Load `entropy_pdf.pt`, `entropy_sigmas.pt`, and optionally
    `entropy_edges.pt` from `run_dir`. Tensors are returned on `device`.

    The σ grid is assumed ascending (production format).
    """
    run_dir = Path(run_dir)
    pdf_p = run_dir / "entropy_pdf.pt"
    sig_p = run_dir / "entropy_sigmas.pt"
    edg_p = run_dir / "entropy_edges.pt"

    if not pdf_p.exists() or not sig_p.exists():
        raise FileNotFoundError(
            f"Entropy tables not found under {run_dir!r}. "
            f"Expected entropy_pdf.pt and entropy_sigmas.pt."
        )

    pdf = torch.load(pdf_p, map_location=device, weights_only=True).to(
        device=device, dtype=torch.float32
    )
    sigmas = torch.load(sig_p, map_location=device, weights_only=True).to(
        device=device, dtype=torch.float32
    )

    edges: Optional[torch.Tensor] = None
    if edg_p.exists():
        edges = torch.load(edg_p, map_location=device, weights_only=True).to(
            device=device, dtype=torch.float32
        )

    if pdf.dim() != 1 or sigmas.dim() != 1 or pdf.numel() != sigmas.numel():
        raise ValueError(
            f"Bad entropy tables in {run_dir!r}: "
            f"pdf shape {tuple(pdf.shape)} vs sigmas shape {tuple(sigmas.shape)}"
        )
    if pdf.numel() < 2:
        raise ValueError(
            f"Entropy table in {run_dir!r} has K={pdf.numel()} bins; need >= 2."
        )
    if edges is not None and edges.numel() != sigmas.numel() + 1:
        raise ValueError(
            f"entropy_edges.pt has {edges.numel()} entries; expected K+1 = "
            f"{sigmas.numel() + 1} for K={sigmas.numel()} bins."
        )

    # Sanity: ascending in σ
    if not bool(torch.all(sigmas[1:] >= sigmas[:-1]).item()):
        raise ValueError(
            f"entropy_sigmas.pt in {run_dir!r} is not ascending in σ. "
            "Expected production (ascending) ordering."
        )

    return pdf, sigmas, edges


def _delta_log_sigma(
    sigmas: torch.Tensor,
    edges: Optional[torch.Tensor],
) -> float:
    """
    Constant bin width Δ(log σ) in log-σ space, preferred from `edges`
    when available and otherwise estimated from the midpoint grid.
    """
    if edges is not None and edges.numel() >= 2:
        log_lo = float(edges[0].clamp_min(1e-30).log().item())
        log_hi = float(edges[-1].clamp_min(1e-30).log().item())
        K = int(edges.numel() - 1)
        return (log_hi - log_lo) / max(1, K)

    log_sig = sigmas.clamp_min(1e-30).log()
    K_mid = int(log_sig.numel())
    if K_mid < 2:
        raise ValueError("Need at least 2 σ midpoints to estimate Δ(log σ).")
    # Equispaced midpoints => spacing is the bin width.
    return float(((log_sig[-1] - log_sig[0]) / float(K_mid - 1)).item())


class LambdaProfile:
    """
    Abstract λ(σ) profile.

    Subclasses must implement `evaluate(sigma, *, state, history) -> Tensor`
    returning a non-negative tensor with the same shape as `sigma`. `state`
    is the current x (for future state-dependent profiles); `history` is a
    dict carrying prior `(D_i, x_i, σ_i)` info (for future trajectory-adaptive
    profiles such as secant-Jacobian or Hutchinson Tr J_D modulation). Both
    are ignored by the current concrete implementations.

    `describe()` returns a small JSON-serializable dict for tags / logging.
    """

    def evaluate(
        self,
        sigma: torch.Tensor,
        *,
        state: Optional[torch.Tensor] = None,
        history: Optional[dict] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


class FlatLambdaProfile(LambdaProfile):
    """Constant Langevin strength λ(σ) = λ₀ for all σ."""

    def __init__(self, lambda_zero: float):
        self.lambda_zero = float(lambda_zero)
        if self.lambda_zero < 0.0:
            raise ValueError(f"lambda_zero must be >= 0, got {self.lambda_zero}")

    def evaluate(
        self,
        sigma: torch.Tensor,
        *,
        state: Optional[torch.Tensor] = None,
        history: Optional[dict] = None,
    ) -> torch.Tensor:
        if not isinstance(sigma, torch.Tensor):
            raise TypeError(f"sigma must be a Tensor, got {type(sigma).__name__}")
        out = torch.full_like(sigma.to(torch.float32), self.lambda_zero)
        return out.clamp_min(0.0)

    def describe(self) -> dict:
        return {
            "name": "flat",
            "lambda_zero": self.lambda_zero,
        }


class EntropyRateLambdaProfile(LambdaProfile):
    """
    λ(σ) = λ₀ · table(log σ), where `table` is the saved entropy-rate PDF
    interpreted as one of:

      - 'peak':     table = pdf / pdf.max()
                    Useful when you want λ₀ to be the peak Langevin strength.
      - 'as_saved': table = pdf / Δ(log σ)
                    The saved table is converted to a log-σ density
                    π_α(log σ). With this mode, the asymptotic identity
                    λ_ent(σ) ≈ S_churn · π_α(log σ) holds exactly under
                    λ₀ = S_churn — required by the EM-vs-EDM-churn
                    equivalence test.

    Out-of-range queries are clamped to the table edges (no extrapolation).
    """

    def __init__(
        self,
        lambda_zero: float,
        entropy_run_dir: Path,
        device: torch.device,
        *,
        normalize: str = "peak",
    ):
        if normalize not in _VALID_NORMALIZE:
            raise ValueError(
                f"normalize must be one of {_VALID_NORMALIZE}, got {normalize!r}"
            )
        if float(lambda_zero) < 0.0:
            raise ValueError(f"lambda_zero must be >= 0, got {lambda_zero}")
        if entropy_run_dir is None:
            raise ValueError(
                "EntropyRateLambdaProfile requires an explicit entropy_run_dir."
            )

        self.lambda_zero = float(lambda_zero)
        self.normalize = str(normalize)
        self.device = torch.device(device)
        self._entropy_run_dir = Path(entropy_run_dir)

        pdf, sigmas, edges = _load_entropy_pdf_table(self._entropy_run_dir, self.device)
        log_sigmas = sigmas.clamp_min(1e-30).log()

        if self.normalize == "peak":
            denom = float(pdf.max().clamp_min(1e-30).item())
            table = pdf / denom
            self._delta_log_sigma = _delta_log_sigma(sigmas, edges)
        elif self.normalize == "as_saved":
            d_log = _delta_log_sigma(sigmas, edges)
            if d_log <= 0.0:
                raise ValueError(
                    f"Δ(log σ) resolved to {d_log}; cannot normalize as_saved."
                )
            table = pdf / d_log
            self._delta_log_sigma = d_log
        else:
            raise AssertionError("unreachable")

        # Cache
        self._sigmas = sigmas
        self._log_sigmas = log_sigmas
        self._table = table.to(self.device)
        self._table_min = float(log_sigmas[0].item())
        self._table_max = float(log_sigmas[-1].item())
        self._K = int(pdf.numel())
        self._pdf_peak = float(pdf.max().clamp_min(0.0).item())

    def _interp(self, log_sigma_q: torch.Tensor) -> torch.Tensor:
        """
        Linear interp of `self._table` at `log_sigma_q` against
        `self._log_sigmas`. Inputs that fall outside the table are clamped
        to the table edges (no extrapolation).

        Works for any shape of `log_sigma_q`.
        """
        xq_clamped = log_sigma_q.clamp(min=self._table_min, max=self._table_max)
        orig_shape = xq_clamped.shape
        flat = xq_clamped.reshape(-1)

        idx = torch.searchsorted(self._log_sigmas, flat, right=False)
        idx = idx.clamp(min=1, max=self._log_sigmas.numel() - 1)
        x_lo = self._log_sigmas[idx - 1]
        x_hi = self._log_sigmas[idx]
        y_lo = self._table[idx - 1]
        y_hi = self._table[idx]
        denom = (x_hi - x_lo).clamp_min(1e-20)
        w = (flat - x_lo) / denom
        y = y_lo + w * (y_hi - y_lo)
        return y.reshape(orig_shape)

    def evaluate(
        self,
        sigma: torch.Tensor,
        *,
        state: Optional[torch.Tensor] = None,
        history: Optional[dict] = None,
    ) -> torch.Tensor:
        if not isinstance(sigma, torch.Tensor):
            raise TypeError(f"sigma must be a Tensor, got {type(sigma).__name__}")
        # Local-device float32 working tensor so that searchsorted and indexing
        # align with the cached self._log_sigmas / self._table.
        sigma_local = sigma.to(device=self.device, dtype=torch.float32)
        log_sigma_q = sigma_local.clamp_min(1e-30).log()
        interp = self._interp(log_sigma_q)
        result = (self.lambda_zero * interp).clamp_min(0.0)
        return result.to(device=sigma.device, dtype=torch.float32)

    def describe(self) -> dict:
        return {
            "name": "entropy_rate",
            "lambda_zero": self.lambda_zero,
            "normalize": self.normalize,
            "K_bins": self._K,
            "delta_log_sigma": self._delta_log_sigma,
            "pdf_peak": self._pdf_peak,
            "entropy_run_dir": str(self._entropy_run_dir),
        }


def make_lambda_profile(
    name: str,
    lambda_zero: float,
    entropy_run_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    *,
    normalize: str = "peak",
) -> LambdaProfile:
    """
    Factory dispatching on the profile name.

    Recognised names:
      - 'flat' / 'constant':       FlatLambdaProfile(lambda_zero)
      - 'entropy_rate' / 'er':     EntropyRateLambdaProfile(
                                       lambda_zero, entropy_run_dir, device,
                                       normalize=normalize,
                                   )

    `entropy_run_dir` and `device` are only required for the entropy-rate
    profile. They are passed through verbatim; for the flat profile they
    are ignored.
    """
    key = str(name).lower().strip()
    if key in {"flat", "constant"}:
        return FlatLambdaProfile(lambda_zero=lambda_zero)

    if key in {"entropy_rate", "entropy-rate", "er", "entropy"}:
        if device is None:
            raise ValueError(
                "make_lambda_profile: `device` is required for the "
                f"entropy_rate profile (name={name!r})."
            )
        if entropy_run_dir is None:
            raise ValueError(
                "make_lambda_profile: `entropy_run_dir` is required for the "
                f"entropy_rate profile (name={name!r})."
            )
        return EntropyRateLambdaProfile(
            lambda_zero=lambda_zero,
            entropy_run_dir=entropy_run_dir,
            device=device,
            normalize=normalize,
        )

    raise ValueError(
        f"Unknown lambda_profile name {name!r}. Valid: 'flat', 'entropy_rate'."
    )
