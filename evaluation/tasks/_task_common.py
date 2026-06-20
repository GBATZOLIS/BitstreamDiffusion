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
from diffusion.continuous.samplers import HeunSampler, DDIMSampler


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
                           sampler_kind: str = "ddim"):
    """Return (model, sampler). Applies EMA shadow weights if present.

    sampler_kind='ddim' (default) -> DDIMSampler, the CoBit 'ddim_entropic'
    headline path (EDM-style stochastic churn on the entropy-rate sigma grid),
    matching evaluation.generation_driver.create_sampler. 'heun' is available
    for a 2nd-order ablation.
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
    seed: Optional[int] = None,
    guidance_scale: Optional[float] = None,
) -> torch.Tensor:
    """Run conditional sampling and return decoded bits [B, S] (long, 0/1).

    guidance_scale: classifier-free guidance weight w (probs = probs_u +
    w*(probs_c - probs_u)). None/0 => no guidance (plain conditional path).
    Requires a checkpoint trained with conditioning dropout (cfg.cond.p_uncond>0).
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
            guidance_scale=guidance_scale,
            sc_refresh_mode="carry",
            ati_eta=0.0,
            return_probs=True,
            progress=False,
        )
    bits = (probs.float() >= 0.5).long()
    return bits
