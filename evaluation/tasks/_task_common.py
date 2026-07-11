"""Shared helpers for the task-driven evaluations (Sudoku, GSM8K).

Loads a trained CoBit checkpoint (applying EMA weights), builds the continuous
HeunSampler, and runs prompt-conditioned bitstream sampling. The prompt region
is clamped to the clean prefix at every solver step (handled inside the sampler
via cond_prefix_mask), so the generated suffix is conditioned on a fixed prompt
exactly as in S-FLM's _project_prefix.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Optional

import torch

from models import create_model
from utils.ema import EMA
from diffusion.continuous.processes import ContinuousForwardProcess
from diffusion.continuous.samplers import (
    HeunSampler, DDIMSampler, EulerMaruyamaSampler, PredictorCorrectorSampler,
    FeynmanKacEulerMaruyamaSampler,
)


def load_config(config_path: str):
    spec = importlib.util.spec_from_file_location("task_cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def resolve_sigma_data(cfg, run_dir, cli_override):
    """Resolve the EDM preconditioning sigma_data used at SAMPLING and set it on cfg.

    The denoiser uses c_in = 1/sqrt(sigma^2 + sigma_data^2) at every step, so eval
    must use the SAME sigma_data the model was trained with. Training overwrites it
    via SigmaDataEstimator and persists it to <run_dir>/sigma_data.json; the task
    configs hardcode 0.5, which usually does NOT match. Precedence:

        CLI override  >  trained sidecar (run_dir/sigma_data.json)  >  config default

    Returns (value, source_str). Falling back to the config default emits a loud
    warning so we never silently sample with the wrong (0.5) preconditioning.
    """
    sidecar = Path(run_dir) / "sigma_data.json"
    if cli_override is not None:
        val = float(cli_override)
        src = f"CLI --sigma_data={val:.4f}"
    elif sidecar.exists():
        val = float(json.loads(sidecar.read_text())["sigma_data"])
        src = f"trained value from {sidecar.name} ({val:.4f})"
    else:
        val = float(cfg.diffusion.continuous.sigma_data)
        src = (f"config default ({val:.4f})  ***WARNING***: no {sidecar.name} found and "
               f"no --sigma_data given; this is the hardcoded config value and likely "
               f"does NOT match training. Pass --sigma_data or write {sidecar}.")
        print("!" * 80, flush=True)
    cfg.diffusion.continuous.sigma_data = val
    print(f"[sigma_data] sampling with sigma_data={val:.4f}  (source: {src})", flush=True)
    return val, src


def _clean_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        k = k.replace("_orig_mod.", "")
        if k.startswith("module."):
            k = k[7:]
        out[k] = v
    return out


def load_model_and_sampler(cfg, ckpt_path: str, device, *, apply_ema: bool = True,
                           sampler_kind: str = "ddim",
                           lambda_zero: float = 0.0,
                           lambda_profile: str = "entropy_rate",
                           lambda_normalize: str = "as_saved",
                           guidance_mode: str = "predictor_only",
                           em_step_gamma_cap=None,
                           fkc_beta: float = 1.0,
                           fkc_num_particles: int = 8,
                           fkc_resampling_policy: str = "ess",
                           fkc_ess_threshold_fraction: float = 0.5,
                           fkc_final_resample: bool = True,
                           fkc_sc_policy: str = "inherit",
                           fkc_prior_mode: str = "sampler_gaussian"):
    """Return (model, sampler). Applies EMA shadow weights if present.

    sampler_kind='ddim' (default) -> DDIMSampler, the CoBit 'ddim_entropic'
    headline path (EDM-style stochastic churn on the entropy-rate sigma grid),
    matching evaluation.generation_driver.create_sampler. 'heun' is available
    for a 2nd-order ablation. 'em' -> EulerMaruyamaSampler, the explicit
    entropy-gated reverse-SDE sampler whose stochasticity is owned by a
    LambdaProfile (lambda_zero / lambda_profile / lambda_normalize); it refuses
    EDM churn and reduces to deterministic DDIM at lambda_zero=0.
    """
    model = create_model(cfg).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(_clean_state_dict(ckpt["model"]), strict=False)

    if apply_ema and ckpt.get("ema") is not None:
        ema = EMA(model, decay=float(getattr(cfg.train, "ema_decay", 0.9999)))
        try:
            ema.load_state_dict(ckpt["ema"])
            ema.to(device)
            ema.apply(model)
            print("[task_eval] applied EMA weights")
        except Exception as e:  # pragma: no cover
            print(f"[task_eval] WARNING: failed to apply EMA ({e}); using raw weights")

    proc = ContinuousForwardProcess(cfg)
    kind = str(sampler_kind).lower()
    if kind in {"ddim", "ddim_entropic", "entropic"}:
        sampler = DDIMSampler(model, proc, cfg)
    elif kind in {"heun", "karras"}:
        sampler = HeunSampler(model, proc, cfg)
    elif kind in {"em", "euler_maruyama"}:
        # em_step_gamma_cap=None -> use EulerMaruyamaSampler's own default (1.0).
        # The per-step churn cap bounds injected noise to <= sqrt(2*cap)*sigma; EDM's
        # own stability bound is gamma<=sqrt(2)-1~=0.41. See reports/EM_TINYGSM_COLLAPSE_ANALYSIS.md.
        em_kwargs = {}
        if em_step_gamma_cap is not None:
            em_kwargs["em_step_gamma_cap"] = float(em_step_gamma_cap)
        sampler = EulerMaruyamaSampler(
            model, proc, cfg,
            lambda_profile_name=str(lambda_profile),
            lambda_zero=float(lambda_zero),
            lambda_profile_normalize=str(lambda_normalize),
            **em_kwargs,
        )
    elif kind in {"pc", "predictor_corrector"}:
        sampler = PredictorCorrectorSampler(
            model, proc, cfg,
            lambda_profile_name=str(lambda_profile),
            lambda_zero=float(lambda_zero),
            lambda_profile_normalize=str(lambda_normalize),
            guidance_mode=str(guidance_mode),
        )
    elif kind in {"fkc_em", "fkc"}:
        # Feynman-Kac SMC sampler for the tempered target p^beta. Uses the same
        # entropy-gated lambda machinery as EM; beta / K / resampling are FKC.
        fkc_kwargs = {}
        if em_step_gamma_cap is not None:
            fkc_kwargs["em_step_gamma_cap"] = float(em_step_gamma_cap)
        sampler = FeynmanKacEulerMaruyamaSampler(
            model, proc, cfg,
            beta=float(fkc_beta),
            num_particles=int(fkc_num_particles),
            lambda_profile_name=str(lambda_profile),
            lambda_zero=float(lambda_zero),
            lambda_profile_normalize=str(lambda_normalize),
            resampling_policy=str(fkc_resampling_policy),
            ess_threshold_fraction=float(fkc_ess_threshold_fraction),
            final_resample=bool(fkc_final_resample),
            sc_policy=str(fkc_sc_policy),
            prior_mode=str(fkc_prior_mode),
            **fkc_kwargs,
        )
    else:
        raise ValueError(f"unknown sampler_kind={sampler_kind!r}")
    print(f"[task_eval] sampler = {sampler.__class__.__name__}")
    return model, sampler


def configure_stochastic(cfg, *, mode: str, gamma: float, num_steps: int, s_noise: float = 1.003,
                         qlo: float = 0.0, qhi: float = 1.0):
    """Set cfg.evaluation.stochastic in place (read by SigmaSchedule.resolve_stochastic_cfg).

    mode='deterministic' -> probability-flow ODE (no churn).
    mode='stochastic'    -> full-band entropy-CDF churn with s_churn = gamma*(NFE-1),
                            the CoBit entropy-rate headline operating point.
    """
    from ml_collections import config_dict

    st = config_dict.ConfigDict()
    if mode == "deterministic" or gamma <= 0.0:
        st.enabled = False
        st.s_churn = 0.0
        st.s_noise = 1.0
        st.window_mode = "deterministic"
    else:
        num_intervals = max(1, int(num_steps) - 1)
        st.enabled = True
        st.s_churn = float(gamma) * num_intervals
        st.s_noise = float(s_noise)
        st.window_mode = "entropy_cdf"
    st.entropy_quantile_lo = float(qlo)
    st.entropy_quantile_hi = float(qhi)
    st.entropy_fallback = "deterministic"
    st.s_tmin = None
    st.s_tmax = None
    cfg.evaluation.stochastic = st


@torch.no_grad()
def sample_bits(
    cfg,
    sampler: HeunSampler,
    *,
    prefix_full: torch.Tensor,   # [B, S] float in {0,1}
    prefix_mask: torch.Tensor,   # [B, S] bool
    num_steps: int,
    schedule: str = "entropic",
    entropy_run_dir: Optional[str] = None,
    sigma_min_override: Optional[float] = None,
    sigma_max_override: Optional[float] = None,
    seed: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    posterior_temp: float = 1.0,
    posterior_temp_target: str = "learned",
    posterior_temp_schedule: str = "const",
    posterior_temp_sigma_lo: float = 0.1,
    posterior_temp_sigma_hi: float = 4.0,
    posterior_temp_space: str = "bit",
    codeword_vocab_size: Optional[int] = None,
    codeword_topk: Optional[int] = None,
    score_temp_tau: float = 1.0,
    score_temp_clean_var: float = 0.25,
) -> torch.Tensor:
    """Run conditional sampling and return decoded bits [B, S] (long, 0/1).

    guidance_scale: classifier-free guidance weight w (probs = probs_u +
    w*(probs_c - probs_u)). None/0 => no guidance (plain conditional path).
    Requires a checkpoint trained with conditioning dropout (cfg.cond.p_uncond>0).

    posterior_temp: continuous analogue of MDLM/Duo low-T decoding. T<1 sharpens
    the per-bit Bernoulli posterior sigmoid(logit/T) toward 0/1 during the
    trajectory. target "learned" sharpens only the network logit (leaving the
    matched-filter data-consistency term untouched); schedule "sigma_ramp"
    applies it only across [sigma_lo, sigma_hi] (T=1 above sigma_hi). T=1 is a
    no-op (bit-identical to the untempered sampler).
    """
    if seed is not None:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))

    B, S = prefix_full.shape
    use_amp = bool(getattr(cfg.evaluation, "use_amp", True))
    amp_dtype = torch.bfloat16 if str(getattr(cfg.evaluation, "amp_dtype", "bf16")).startswith("bf") else torch.float16
    dev = prefix_full.device

    with torch.autocast(dev.type, enabled=use_amp, dtype=amp_dtype):
        x, probs = sampler.sample(
            num_samples=B,
            seq_len=S,
            conditioning_prefix_full=prefix_full,
            cond_prefix_mask=prefix_mask,
            num_steps=int(num_steps),
            schedule=schedule,
            entropy_run_dir=entropy_run_dir,
            sigma_min_override=sigma_min_override,
            sigma_max_override=sigma_max_override,
            guidance_scale=guidance_scale,
            sc_refresh_mode="carry",
            ati_eta=0.0,
            return_probs=True,
            progress=False,
            posterior_temp=posterior_temp,
            posterior_temp_target=posterior_temp_target,
            posterior_temp_schedule=posterior_temp_schedule,
            posterior_temp_sigma_lo=posterior_temp_sigma_lo,
            posterior_temp_sigma_hi=posterior_temp_sigma_hi,
            posterior_temp_space=posterior_temp_space,
            codeword_vocab_size=codeword_vocab_size,
            codeword_topk=codeword_topk,
            score_temp_tau=score_temp_tau,
            score_temp_clean_var=score_temp_clean_var,
        )
    bits = (probs.float() >= 0.5).long()
    return bits


@torch.no_grad()
def sample_bit_particles(
    cfg,
    sampler: "FeynmanKacEulerMaruyamaSampler",
    *,
    prefix_full: torch.Tensor,   # [B, S] float in {0,1}
    prefix_mask: torch.Tensor,   # [B, S] bool (True = prompt)
    num_steps: int,
    schedule: str = "entropic",
    entropy_run_dir: Optional[str] = None,
    sigma_min_override: Optional[float] = None,
    seed: Optional[int] = None,
):
    """Run the FKC particle sampler and return its FKCOutput.

    `out.bits` is [B, K, S] (long 0/1), the post-final-resample target
    population; `out.pre_resample_bits`, `out.log_weights_final`,
    `out.ancestors`, and `out.diagnostics` expose proposal coverage and SMC
    telemetry for maj@K / pass@K reporting and degeneracy diagnostics.
    """
    if seed is not None:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    B, S = prefix_full.shape
    use_amp = bool(getattr(cfg.evaluation, "use_amp", True))
    amp_dtype = torch.bfloat16 if str(getattr(cfg.evaluation, "amp_dtype", "bf16")).startswith("bf") else torch.float16
    dev = prefix_full.device
    with torch.autocast(dev.type, enabled=use_amp, dtype=amp_dtype):
        out = sampler.sample_particles(
            num_prompts=B, seq_len=S,
            conditioning_prefix_full=prefix_full, cond_prefix_mask=prefix_mask,
            num_steps=int(num_steps), schedule=schedule, entropy_run_dir=entropy_run_dir,
            sigma_min_override=sigma_min_override, seed=seed,
            guidance_scale=0.0, posterior_temp=1.0, ati_eta=0.0,
            return_diagnostics=True, progress=False,
        )
    return out
