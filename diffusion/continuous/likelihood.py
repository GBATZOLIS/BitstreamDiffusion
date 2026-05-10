"""
diffusion/continuous/likelihood.py

Accurate BPC (Bits-Per-Character) estimation for continuous diffusion models
trained on binary/discrete data.

The model is interpreted as a latent-variable model:

  z ~ p_θ(z)        (continuous latent at noise level σ_min, defined via PF-ODE)
  x₀ ~ p_θ(x₀ | z)  (Bernoulli decoder parameterized by the diffusion model)

We approximate the discrete log-likelihood log p_θ(x₀) via the ELBO:

  log p_θ(x₀) >= E_{q(z|x₀)} [ log p_θ(x₀ | z) + log p_θ(z) - log q(z|x₀) ]

where:

  - q(z|x₀) = N(z; x₀, σ_min² I) is a Gaussian "bridge" (forward diffusion).
  - p_θ(z)   is defined via the Probability Flow ODE in σ-space.
  - p_θ(x₀|z) is a Bernoulli distribution (discrete bridge back to bits).

This file provides:

  - A flexible σ-schedule builder (Karras, entropic, log-uniform, log-normal).
  - PF-ODE log-density integration in σ with correct sign.
  - ELBO-based BPC computation for binary sequences, with optional σ_min override.

It also supports *per-example* Bits-Per-Character computation for semantic text
datasets (Text8, WikiText, etc.), by decoding bitstreams back to text and
normalizing by the true character length per sequence. Other datasets fall
back to a fixed bits_per_char rule.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from tqdm import tqdm
from contextlib import nullcontext

# Optional semantic decoders for text datasets (Text8, WikiText, etc.)
# These are soft dependencies; if the modules are missing, we simply
# won't use them and will fall back to the fixed bits_per_char logic.
try:
    from data.text8 import bits_to_text_semantic as text8_bits_to_text_semantic
except Exception:
    text8_bits_to_text_semantic = None  # type: ignore[assignment]

try:
    from data.wikitext import bits_to_text_semantic as wiki_bits_to_text_semantic
except Exception:
    wiki_bits_to_text_semantic = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# 1. Helper functions
# -----------------------------------------------------------------------------


def _score_from_logits(
    logits: torch.Tensor,
    x: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the score from model logits, consistent with the *NEW* training loss.

    NEW convention:
        probs = sigmoid(logits)          (NO division by σ² before sigmoid)
        score = (probs - x) / σ²
    """
    # logits: [B, S] or [B, S, 1]
    if logits.dim() == 3 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)  # [B, S, 1] -> [B, S]

    # sigma: [B] or scalar
    if sigma.dim() == 0:
        sigma = sigma.expand(x.size(0))  # [B]

    # fp32 for numerical stability (esp. under autocast)
    sigma2 = (sigma ** 2).view(-1, 1).to(torch.float32)  # [B, 1]
    logits_f32 = logits.to(torch.float32)
    x_f32 = x.to(torch.float32)

    # --- KEY FIX: no logits / sigma2 inside sigmoid ---
    probs = torch.sigmoid(logits_f32)  # [B, S]

    # score = (D - x) / σ²
    score = (probs - x_f32) / sigma2  # [B, S]
    return score


def _log_gaussian_prior(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Compute log N(x; 0, sigma^2 I) for each batch element.

    Args:
        x: [B, S]
        sigma: scalar

    Returns:
        log_prob: [B]
    """
    B, S = x.shape
    var = sigma ** 2
    log_2pi = math.log(2.0 * math.pi)
    quad = x.pow(2).sum(dim=1) / var  # [B]
    return -0.5 * (S * (log_2pi + math.log(var)) + quad)


def _hutchinson_divergence(
    f: torch.Tensor,
    x: torch.Tensor,
    num_probes: int = 1,
    eps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Estimate div_x f(x) using Hutchinson's trace estimator.

    div f(x) = E_v [ v^T (∂f/∂x) v ],
    with v ~ Rademacher({-1, +1}).

    Args:
        f: [B, S], vector field evaluated at x
        x: [B, S], requires_grad=True
        num_probes: number of Hutchinson probes (trade variance vs cost)
        eps: optional pre-sampled noise of shape [num_probes, B, S]

    Returns:
        div: [B]
    """
    assert x.requires_grad, "x must have requires_grad=True for divergence computation."
    B, S = x.shape
    device = x.device
    dtype = x.dtype

    if eps is None:
        # Sample integer Rademacher and cast to x.dtype
        eps = torch.randint(0, 2, (num_probes, B, S), device=device)  # int64
        eps = eps * 2 - 1                                             # {0,1} -> {-1, +1}
        eps = eps.to(dtype)

    div = torch.zeros(B, device=device, dtype=dtype)

    for k in range(num_probes):
        v = eps[k]  # [B, S]

        # Hutchinson trace estimator via Jacobian-vector product
        scalar = (f * v).sum()
        grad = torch.autograd.grad(
            scalar,
            x,
            create_graph=False,
            retain_graph=(k < num_probes - 1),
        )[0]  # [B, S]

        div = div + (grad * v).view(B, -1).sum(dim=1)  # [B]

    return div / float(num_probes)


# -----------------------------------------------------------------------------
# 2. σ-schedule builder
# -----------------------------------------------------------------------------


def _load_entropy_tables(cfg, device: torch.device):
    """
    Load precomputed entropy-based σ statistics if available.

    Expected files (saved under run directory):
        entropy_pdf.pt, entropy_cdf.pt, entropy_sigmas.pt
    """
    try:
        ckpt_path = Path(cfg.evaluation.checkpoint_path).expanduser().resolve()
        run_dir = ckpt_path.parent.parent
        pdf = torch.load(run_dir / "entropy_pdf.pt", map_location=device)
        cdf = torch.load(run_dir / "entropy_cdf.pt", map_location=device)
        sigs = torch.load(run_dir / "entropy_sigmas.pt", map_location=device)
        return pdf.to(device), cdf.to(device), sigs.to(device)
    except Exception:
        return None, None, None


def build_sigma_schedule(
    forward_process,
    cfg,
    steps: int,
    device: torch.device,
    schedule_override: Optional[str] = None,
    sigma_min_override: Optional[float] = None,
) -> torch.Tensor:
    """
    Build an ASCENDING σ schedule [σ_min, ..., σ_max] for PF-ODE integration.

    Supports:
      - karras / edm
      - entropic
      - loguniform
      - lognormal

    `sigma_min_override` lets you sweep σ_min at evaluation time while keeping
    σ_max fixed to the config value.
    """
    cont_cfg = cfg.diffusion.continuous

    # Use override if provided, else fall back to config
    if sigma_min_override is not None:
        sigma_min = float(sigma_min_override)
    else:
        sigma_min = float(cont_cfg.sigma_min)

    sigma_max = float(cont_cfg.sigma_max)
    rho = float(getattr(cont_cfg, "rho", 7.0))

    if sigma_min <= 0.0:
        raise ValueError(f"sigma_min must be > 0, got {sigma_min}")
    if sigma_min > sigma_max:
        raise ValueError(f"sigma_min ({sigma_min}) must be <= sigma_max ({sigma_max}).")

    # Decide schedule name
    if schedule_override is not None:
        schedule_name = str(schedule_override).lower()
    else:
        schedule_name = str(getattr(cfg.evaluation, "schedule", "karras")).lower()

    # Parameter u always spans [0, 1] exactly
    u = torch.linspace(0.0, 1.0, steps, device=device)

    # 1. Karras / EDM schedule
    if schedule_name in {"karras", "edm"}:
        inv_rho = 1.0 / rho
        start = sigma_min ** inv_rho
        end = sigma_max ** inv_rho
        sigmas = (start + u * (end - start)) ** rho

    # 2. Entropic schedule (data-driven)
    elif schedule_name == "entropic":
        pdf, cdf, sigmas_base = _load_entropy_tables(cfg, device)
        if cdf is None or sigmas_base is None:
            # Fallback to Karras if entropy tables are missing
            return build_sigma_schedule(
                forward_process,
                cfg,
                steps,
                device,
                schedule_override="karras",
                sigma_min_override=sigma_min_override,
            )

        # Ensure CDF ends at 1
        cdf = cdf.clone()
        cdf[-1] = 1.0

        # Invert CDF at quantiles u
        idx = torch.searchsorted(cdf, u, right=False)
        idx = torch.clamp(idx, 0, cdf.numel() - 1)
        sigmas = sigmas_base[idx]

        # Optional blending with Karras
        blend = float(getattr(cfg.evaluation, "entropic_blend_alpha", 0.0))
        if blend > 0.0:
            karras_sigmas = build_sigma_schedule(
                forward_process,
                cfg,
                steps,
                device,
                schedule_override="karras",
                sigma_min_override=sigma_min_override,
            )
            sigmas = (1.0 - blend) * sigmas + blend * karras_sigmas

    # 3. Log-uniform schedule
    elif schedule_name in {"loguniform", "log-uniform"}:
        log_min = math.log(sigma_min)
        log_max = math.log(sigma_max)
        sigmas = torch.exp(log_min + u * (log_max - log_min))

    # 4. Truncated log-normal schedule
    elif schedule_name in {"lognormal", "log-normal"}:
        # Base log-normal parameters (used in training)
        p_mean = float(getattr(cont_cfg, "p_mean", 0.0))
        p_std = float(getattr(cont_cfg, "p_std", 1.0))

        # Compute z_min, z_max such that:
        #   log(sigma_min) = p_mean + p_std * z_min
        #   log(sigma_max) = p_mean + p_std * z_max
        z_min = (math.log(sigma_min) - p_mean) / p_std
        z_max = (math.log(sigma_max) - p_mean) / p_std

        z = z_min + u * (z_max - z_min)
        log_sigma = p_mean + p_std * z
        sigmas = torch.exp(log_sigma)

    else:
        # Fallback to Karras if schedule name is unknown
        return build_sigma_schedule(
            forward_process,
            cfg,
            steps,
            device,
            schedule_override="karras",
            sigma_min_override=sigma_min_override,
        )

    # Enforce exact endpoints for precise definite integration
    sigmas[0] = sigma_min
    sigmas[-1] = sigma_max

    # Ensure ascending order (just in case)
    if sigmas[0] > sigmas[-1]:
        sigmas = torch.flip(sigmas, dims=[0])

    return sigmas


# -----------------------------------------------------------------------------
# 3. PF-ODE log-density integration in σ
# -----------------------------------------------------------------------------


def _pf_ode_logdensity_sigma(
    model,
    x_sigma_min: torch.Tensor,
    sigmas: torch.Tensor,
    num_probes: int = 1,
    progress: bool = False,
    method: str = "euler",
) -> torch.Tensor:
    """
    Approximate log p_θ(z) for z = x_{σ_min} by integrating the PF-ODE in σ.

        dx/dσ = f(x, σ) = -σ * s_θ(x, σ)

    from σ_min to σ_max, and use the log-density change:

        log p(z_min) = log p(z_max) + ∫ div_x f(x_σ, σ) dσ

    Returns:
        log p_θ(x_sigma_min): [B]
    """
    method = method.lower()
    if method not in {"euler", "heun"}:
        raise ValueError(f"Unknown PF-ODE method '{method}', expected 'euler' or 'heun'.")

    device = x_sigma_min.device
    B, S = x_sigma_min.shape

    # Ensure ascending σ
    if sigmas[0] > sigmas[-1]:
        sigmas = torch.flip(sigmas, dims=[0])

    # Step sizes: [K-1], all positive
    d_sigmas = sigmas[1:] - sigmas[:-1]

    # Work in float32 for stability
    x = x_sigma_min.detach().clone().to(torch.float32)
    logp_change = torch.zeros(B, device=device, dtype=torch.float32)

    iterator = range(len(sigmas) - 1)
    if progress:
        desc = "PF-ODE (σ, Euler)" if method == "euler" else "PF-ODE (σ, Heun)"
        iterator = tqdm(iterator, desc=desc, leave=False)

    # Ensure we are NOT under autocast (we want full precision here)
    if torch.cuda.is_available():
        amp_ctx = torch.cuda.amp.autocast(enabled=False)
    else:
        amp_ctx = nullcontext()

    with amp_ctx:
        if method == "euler":
            for i in iterator:
                sigma_i = sigmas[i]
                h = d_sigmas[i]

                x.requires_grad_(True)
                sigma_batch = sigma_i.expand(B)  # [B]

                logits = model(x, sigma_batch)
                score = _score_from_logits(logits, x, sigma_batch)

                f = -sigma_i * score  # dx/dσ
                div_f = _hutchinson_divergence(f, x, num_probes=num_probes)

                # Correct sign: log p(z_min) = log p(z_max) + ∫ div f dσ
                logp_change = logp_change + h * div_f

                with torch.no_grad():
                    x = x + h * f
                    x = x.detach()

        else:
            for i in iterator:
                sigma_i = sigmas[i]
                sigma_next = sigmas[i + 1]
                h = d_sigmas[i]

                x_i = x.detach()
                x_i.requires_grad_(True)
                sigma_batch_i = sigma_i.expand(B)

                logits_i = model(x_i, sigma_batch_i)
                score_i = _score_from_logits(logits_i, x_i, sigma_batch_i)
                f_i = -sigma_i * score_i
                div_i = _hutchinson_divergence(f_i, x_i, num_probes=num_probes)

                with torch.no_grad():
                    x_pred = x_i + h * f_i

                x_i = x_i.detach()
                f_i = f_i.detach()

                x_pred.requires_grad_(True)
                sigma_batch_next = sigma_next.expand(B)

                logits_next = model(x_pred, sigma_batch_next)
                score_next = _score_from_logits(logits_next, x_pred, sigma_batch_next)
                f_next = -sigma_next * score_next
                div_next = _hutchinson_divergence(f_next, x_pred, num_probes=num_probes)

                logp_change = logp_change + 0.5 * h * (div_i + div_next)

                with torch.no_grad():
                    x = x_i + 0.5 * h * (f_i + f_next)

                x = x.detach()
                x_pred = x_pred.detach()
                f_next = f_next.detach()

    sigma_max = float(sigmas[-1].item())
    logp_prior = _log_gaussian_prior(x, sigma=sigma_max)

    return logp_prior + logp_change


# -----------------------------------------------------------------------------
# 3.5 Dataset helpers for BPC normalization
# -----------------------------------------------------------------------------


def _unwrap_base_dataset(ds):
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds


def _to_binary_bits_cpu(batch: torch.Tensor) -> torch.Tensor:
    if batch.is_floating_point():
        bits = (batch > 0.5).to(torch.long)
    else:
        bits = (batch != 0).to(torch.long)
    return bits.cpu()


def _compute_chars_per_example_from_bits(
    cfg,
    dataset,
    bits_batch: torch.Tensor,
) -> Optional[torch.Tensor]:
    data_cfg = cfg.data
    binarization = str(getattr(data_cfg, "binarization", "ascii")).lower()
    if binarization != "semantic":
        return None

    base_ds = _unwrap_base_dataset(dataset)

    decode_fn = getattr(base_ds, "decode_bits_to_text", None)
    bits_cpu = _to_binary_bits_cpu(bits_batch)

    if callable(decode_fn):
        texts: List[str] = []
        for i in range(bits_cpu.size(0)):
            txt = decode_fn(bits_cpu[i])
            texts.append(txt)
        lengths = [len(t) for t in texts]
        return torch.tensor(lengths, dtype=torch.float32)

    tokenizer = getattr(base_ds, "tokenizer", None)
    new_to_old = getattr(base_ds, "new_to_old", None)
    if tokenizer is None or new_to_old is None:
        return None

    dataset_name = str(getattr(data_cfg, "dataset", "")).lower()

    decoder = None
    if "text8" in dataset_name and text8_bits_to_text_semantic is not None:
        decoder = text8_bits_to_text_semantic
    elif "wiki" in dataset_name and wiki_bits_to_text_semantic is not None:
        decoder = wiki_bits_to_text_semantic
    else:
        decoder = wiki_bits_to_text_semantic or text8_bits_to_text_semantic

    if decoder is None:
        return None

    texts: List[str] = []
    for i in range(bits_cpu.size(0)):
        try:
            txt = decoder(bits_cpu[i], tokenizer, new_to_old)
        except Exception:
            return None
        texts.append(txt)

    lengths = [len(t) for t in texts]
    return torch.tensor(lengths, dtype=torch.float32)


# -----------------------------------------------------------------------------
# 4. Public API: ELBO-based BPC computation
# -----------------------------------------------------------------------------


def bits_per_character_elbo_discrete(
    model,
    forward_process,
    cfg,
    x0_batch: torch.Tensor,
    steps: int = 128,
    num_probes: int = 1,
    mc_samples: int = 1,
    progress: bool = False,
    schedule: Optional[str] = None,
    sigma_min: Optional[float] = None,
    chars_per_example: Optional[torch.Tensor] = None,
    pf_method: str = "euler",
) -> Tuple[float, float]:
    """
    Compute Bits-Per-Character (BPC) via an ELBO on the discrete log-likelihood.

    NEW convention: recon uses UNSCALED logits (consistent with new training loss).
    """
    device = x0_batch.device
    B, S = x0_batch.shape

    cont_cfg = cfg.diffusion.continuous

    if sigma_min is None:
        sigma_min = float(cont_cfg.sigma_min)
    else:
        sigma_min = float(sigma_min)

    # Ensure targets are {0,1} floats
    if x0_batch.is_floating_point():
        x0_bits = (x0_batch > 0.5).to(torch.float32)
    else:
        x0_bits = (x0_batch != 0).to(torch.float32)

    sigmas = build_sigma_schedule(
        forward_process,
        cfg,
        steps=steps,
        device=device,
        schedule_override=schedule,
        sigma_min_override=sigma_min,
    )

    elbo_accum = torch.zeros(B, device=device, dtype=torch.float32)

    for _ in range(mc_samples):
        # q(z|x0) = N(x0, σ_min² I)
        noise = torch.randn_like(x0_bits)
        z = x0_bits + sigma_min * noise

        # Recon term: Bernoulli with UNSCALED logits at σ_min
        sigma_batch = torch.full((B,), sigma_min, device=device)
        logits = model(z, sigma_batch)
        if logits.dim() == 3 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)

        # --- KEY FIX: do NOT scale logits by σ_min² ---
        nll_recon = F.binary_cross_entropy_with_logits(
            logits.to(torch.float32),
            x0_bits.to(torch.float32),
            reduction="none",
        ).sum(dim=1)
        log_p_recon = -nll_recon

        # Prior term via PF-ODE likelihood
        with torch.enable_grad():
            log_p_z = _pf_ode_logdensity_sigma(
                model,
                x_sigma_min=z,
                sigmas=sigmas,
                num_probes=num_probes,
                progress=progress,
                method=pf_method,
            )

        # log q(z|x0)
        log_2pi = math.log(2.0 * math.pi)
        log_q = -0.5 * (
            S * (log_2pi + math.log(sigma_min ** 2))
            + noise.pow(2).sum(dim=1)
        )

        elbo_accum += (log_p_recon + log_p_z - log_q)

    elbo_mean = elbo_accum / float(mc_samples)
    nll_nats = -elbo_mean

    ln2 = math.log(2.0)

    if chars_per_example is None:
        bits_per_char = int(getattr(cfg.data, "bits_per_char", 8))
        num_chars_scalar = float(S) / float(bits_per_char)
        bpc = (nll_nats / ln2) / num_chars_scalar
    else:
        chars = chars_per_example.to(device=device, dtype=torch.float32).clamp_min(1.0)
        bpc = (nll_nats / ln2) / chars

    nll_nats_per_dim = nll_nats / float(S)

    return bpc.mean().item(), nll_nats_per_dim.mean().item()


@torch.no_grad()
def evaluate_bpc_over_loader(
    model,
    forward_process,
    cfg,
    data_loader,
    steps: int = 128,
    num_probes: int = 1,
    mc_samples: int = 1,
    max_batches: Optional[int] = None,
    progress: bool = True,
    schedule: Optional[str] = None,
    sigma_min: Optional[float] = None,
    pf_method: str = "euler",
) -> Tuple[float, float]:
    """
    Evaluate average BPC over a dataset loader.
    Supports overriding sigma_min via argument.
    """
    model.eval()
    total_bpc = 0.0
    total_nll = 0.0
    count = 0

    iterator = data_loader
    if progress:
        iterator = tqdm(iterator, desc="BPC Test Eval")

    device = next(model.parameters()).device
    dataset = data_loader.dataset

    for i, batch in enumerate(iterator):
        if max_batches is not None and i >= max_batches:
            break

        if isinstance(batch, (tuple, list)):
            batch = batch[0]

        x_cpu = batch

        chars_per_example = _compute_chars_per_example_from_bits(
            cfg=cfg,
            dataset=dataset,
            bits_batch=x_cpu,
        )

        x = x_cpu.to(device)

        if x.is_floating_point():
            x_bits = (x > 0.5).to(torch.float32)
        else:
            x_bits = (x != 0).to(torch.float32)

        bpc, nll = bits_per_character_elbo_discrete(
            model=model,
            forward_process=forward_process,
            cfg=cfg,
            x0_batch=x_bits,
            steps=steps,
            num_probes=num_probes,
            mc_samples=mc_samples,
            progress=False,
            schedule=schedule,
            sigma_min=sigma_min,
            chars_per_example=chars_per_example,
            pf_method=pf_method,
        )

        total_bpc += bpc
        total_nll += nll
        count += 1

        if progress:
            iterator.set_postfix({"bpc": f"{total_bpc / count:.4f}"})

    if count == 0:
        return float("nan"), float("nan")

    return total_bpc / count, total_nll / count
