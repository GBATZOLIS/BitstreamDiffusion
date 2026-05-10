#!/usr/bin/env python3
"""
Profile training and inference efficiency for continuous token-space diffusion
versus patched bitstream diffusion, at both LM1B and OpenWebText scales.

The profiler is intentionally synthetic: it removes dataloading, decoding,
external PPL evaluation, cache writes, and host/device transfer effects. The
reported numbers isolate the model representation, output/loss boundary, and
iterative denoising cost.

Typical usage from repository root:

python scripts/profile_train_inference_efficiency.py \
  --lm1b-bit-config configs/lm1b/continuous/ablations/edm.py \
  --lm1b-token-config configs/lm1b/continuous/ablations/onehot_tokens_ce.py \
  --owt-bit-config configs/owt/continuous/raw_binary_bits.py \
  --outdir runs/efficiency_profile_train_infer \
  --train-batch-sizes 16 32 64 128 256 512 \
  --infer-micro-batches 4 8 16 32 64 128 256 512 \
  --infer-steps 256 \
  --warmup 3 \
  --steps 10

For OWT token-space profiling, you can either provide a config:

  --owt-token-config configs/owt/continuous/onehot_tokens_ce.py

or let this script synthesize a token-space config from the OWT bit config:

  --synthesize-missing-token-configs

Outputs:
  - train_results.jsonl / train_results.csv
  - inference_results.jsonl / inference_results.csv
  - boundary_results.jsonl / boundary_results.csv
  - analytic_scaling.jsonl / analytic_scaling.csv
  - latex_tables.tex

Notes:
  - Training profiling performs forward + backward + optimizer step.
  - Inference profiling performs iterative denoising with no gradients.
  - Inference throughput is reported in semantic tokens/s and, for diffusion,
    therefore already includes the full NFE budget. We also report denoiser
    calls/s and semantic-token denoiser-evals/s for diagnosing kernel efficiency.
  - Inference memory is measured for a single generation micro-batch on one GPU.
  - For stochastic profiling, pass --profile-stochastic. By default we profile
    deterministic DDIM-style updates because this isolates the representation
    and output boundary most cleanly.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Repository imports
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sdt import (  # noqa: E402
    OptimalSkipMLPHead,
    SequenceVDTContinuousModel,
    TokenFullHead,
)


# -----------------------------------------------------------------------------
# Progress utilities
# -----------------------------------------------------------------------------


@dataclass
class ProgressTracker:
    total: int
    label: str = "profiling"
    completed: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    def update(self, message: str = "") -> None:
        self.completed += 1
        elapsed = time.perf_counter() - self.start_time
        rate = self.completed / max(elapsed, 1e-12)
        remaining = max(self.total - self.completed, 0)
        eta = remaining / max(rate, 1e-12)
        pct = 100.0 * self.completed / max(self.total, 1)
        msg = f"[{self.label}] {self.completed}/{self.total} ({pct:.1f}%)"
        if message:
            msg += f" | {message}"
        msg += f" | elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}"
        print(msg, flush=True)


def format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def load_config_from_path(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import config file: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "get_config"):
        raise AttributeError(f"Config file {path} does not define get_config().")

    return module.get_config()


def ensure_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for meaningful memory/throughput profiling.")
    return torch.device("cuda")


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gib() -> float:
    return torch.cuda.max_memory_allocated() / (1024**3)


def current_device_name() -> str:
    return torch.cuda.get_device_name(torch.cuda.current_device())


def bf16_autocast(enabled: bool = True):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


def num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(x: Any, ndigits: int = 2) -> str:
    if x is None:
        return "--"
    if isinstance(x, str):
        return x
    return f"{float(x):.{ndigits}f}"


def fmt_int(x: Any) -> str:
    if x is None:
        return "--"
    if isinstance(x, str):
        return x
    return f"{int(round(float(x))):,}"


def latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_")


def maybe_get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def deep_copy_cfg(cfg: Any) -> Any:
    try:
        return copy.deepcopy(cfg)
    except Exception:
        # ml_collections ConfigDict normally supports deepcopy, but keep a
        # fallback for older versions.
        return cfg.copy_and_resolve_references()


# -----------------------------------------------------------------------------
# Config normalization / synthesis
# -----------------------------------------------------------------------------


def set_common_profile_flags(
    cfg: Any,
    *,
    self_condition: bool,
    use_flash_attn: Optional[bool],
    use_compile: bool,
) -> Any:
    """Mutate config for profiling."""
    if hasattr(cfg, "train"):
        cfg.train.use_compile = bool(use_compile)
        cfg.train.use_fp16 = True
        cfg.train.amp_dtype = "bf16"
        cfg.train.allow_tf32 = True

    if hasattr(cfg, "evaluation"):
        cfg.evaluation.use_compile = bool(use_compile)
        cfg.evaluation.amp_dtype = "bf16"
        cfg.evaluation.use_amp = True

    cfg.model.self_condition = bool(self_condition)

    if use_flash_attn is not None and hasattr(cfg.model, "use_flash_attn"):
        cfg.model.use_flash_attn = bool(use_flash_attn)

    return cfg


def synthesize_token_config_from_bit_config(
    bit_cfg: Any,
    *,
    dataset_name: str,
    vocab_size: int,
    sequence_len_tokens: int,
) -> Any:
    """
    Build a continuous one-hot token-space baseline config from a bitstream config.

    This is useful for OWT, where you may not have a separately saved token-space
    config. It keeps the same SDT trunk and switches only the representation,
    patching, input/output boundary, and token diffusion metadata.
    """
    cfg = deep_copy_cfg(bit_cfg)

    cfg.experiment = f"synthetic_profile/{dataset_name.lower()}_continuous_onehot_tokens_ce"

    cfg.data.representation = "tokens"
    cfg.data.sequence_len_tokens = int(sequence_len_tokens)
    cfg.data.sequence_len = int(sequence_len_tokens)
    cfg.data.vocab_size = int(vocab_size)
    cfg.data.bits_per_token = int(math.ceil(math.log2(int(vocab_size))))

    cfg.model.patch_size = 1
    cfg.model.head_type = "token_full"
    cfg.model.out_dim = int(vocab_size)
    cfg.model.continuous_logit_scaling = "none"
    cfg.model.matched_filter_center = 1.0 / float(vocab_size)
    cfg.model.matched_filter_scale = 1.0
    cfg.model.matched_filter_clip = None

    cfg.diffusion.continuous.data_center = 1.0 / float(vocab_size)
    cfg.diffusion.continuous.sigma_data = 0.5

    cfg.train.loss_type = "token_ce"
    cfg.train.loss_weighting = "edm"
    cfg.train.token_sm_chunk_size = 2048

    return cfg


@dataclass(frozen=True)
class DatasetPair:
    dataset: str
    representation_name: str
    cfg: Any
    source_config: str


def cfg_repr(cfg: Any) -> str:
    return str(cfg.data.representation).lower()


def cfg_is_tokens(cfg: Any) -> bool:
    return cfg_repr(cfg) == "tokens"


def cfg_is_bits(cfg: Any) -> bool:
    return cfg_repr(cfg) == "binary"


def cfg_semantic_len(cfg: Any) -> int:
    return int(getattr(cfg.data, "sequence_len_tokens", getattr(cfg.data, "sequence_len", 0)))


def cfg_model_len(cfg: Any) -> int:
    if cfg_is_tokens(cfg):
        return cfg_semantic_len(cfg)
    return int(getattr(cfg.data, "sequence_len", cfg_semantic_len(cfg) * int(cfg.data.bits_per_token)))


def cfg_bits_per_token(cfg: Any) -> int:
    return int(getattr(cfg.data, "bits_per_token", 1))


def cfg_vocab_for_table(cfg: Any, *, dataset: str) -> int:
    if cfg_is_tokens(cfg):
        return int(cfg.data.vocab_size)
    # For bit configs, cfg.data.vocab_size is 2. Use semantic/code vocabulary.
    if dataset.upper() == "LM1B":
        return 30522
    if dataset.upper() == "OWT":
        return 65536
    return int(2 ** cfg_bits_per_token(cfg))


# -----------------------------------------------------------------------------
# Synthetic data construction
# -----------------------------------------------------------------------------


def make_synthetic_train_batch(
    *,
    cfg: Any,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, int, int, int]:
    """
    Returns x, sigma, target, is_tokens, T_semantic, V, m.
    """
    is_tokens = cfg_is_tokens(cfg)
    T = cfg_semantic_len(cfg)
    m = cfg_bits_per_token(cfg)
    V = int(getattr(cfg.data, "vocab_size", 0))

    sigma = torch.ones(batch_size, device=device, dtype=dtype)

    if is_tokens:
        x = torch.randn(batch_size, T, V, device=device, dtype=dtype)
        target = torch.randint(0, V, (batch_size, T), device=device)
    else:
        S = cfg_model_len(cfg)
        x = torch.randn(batch_size, S, device=device, dtype=dtype)
        target = torch.randint(0, 2, (batch_size, S), device=device, dtype=torch.float32)

    return x, sigma, target, is_tokens, T, V, m


def make_synthetic_infer_state(
    *,
    cfg: Any,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    sigma_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], bool, int, int, int]:
    """
    Returns x, sigma, x0_hat, is_tokens, T_semantic, V, m.
    """
    is_tokens = cfg_is_tokens(cfg)
    T = cfg_semantic_len(cfg)
    m = cfg_bits_per_token(cfg)
    V = int(getattr(cfg.data, "vocab_size", 0))

    sigma = torch.full((batch_size,), float(sigma_max), device=device, dtype=dtype)

    if is_tokens:
        x = torch.randn(batch_size, T, V, device=device, dtype=dtype) * float(sigma_max)
        x0_hat = None
    else:
        S = cfg_model_len(cfg)
        x = torch.randn(batch_size, S, device=device, dtype=dtype) * float(sigma_max)
        x0_hat = None

    return x, sigma, x0_hat, is_tokens, T, V, m


# -----------------------------------------------------------------------------
# Losses and model steps
# -----------------------------------------------------------------------------


def chunked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    logits_2d = logits.reshape(-1, logits.shape[-1])
    target_1d = target.reshape(-1)

    if chunk_size is None or chunk_size <= 0 or logits_2d.shape[0] <= chunk_size:
        return F.cross_entropy(logits_2d.float(), target_1d)

    total_loss = logits_2d.new_zeros((), dtype=torch.float32)
    total_count = 0
    for start in range(0, logits_2d.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), logits_2d.shape[0])
        loss = F.cross_entropy(
            logits_2d[start:end].float(),
            target_1d[start:end],
            reduction="sum",
        )
        total_loss = total_loss + loss
        total_count += end - start
    return total_loss / max(total_count, 1)


def run_one_train_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    is_tokens: bool,
    ce_chunk_size: Optional[int],
    use_amp: bool,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)

    with bf16_autocast(enabled=use_amp):
        logits = model(x, sigma, None)
        if is_tokens:
            loss = chunked_cross_entropy(logits, target, chunk_size=ce_chunk_size)
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits.reshape(-1).float(),
                target.reshape(-1),
            )

    loss.backward()
    optimizer.step()
    return loss.detach()


def karras_sigmas(
    *,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """EDM/Karras grid from sigma_max down to sigma_min, length num_steps + 1."""
    ramp = torch.linspace(0, 1, int(num_steps) + 1, device=device, dtype=torch.float32)
    min_inv = float(sigma_min) ** (1.0 / float(rho))
    max_inv = float(sigma_max) ** (1.0 / float(rho))
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** float(rho)
    return sigmas.to(dtype=dtype)


@torch.no_grad()
def warmup_inference_denoiser_calls(
    *,
    model: nn.Module,
    cfg: Any,
    batch_size: int,
    num_calls: int,
    device: torch.device,
    use_amp: bool,
    sc_refresh_mode: str,
) -> None:
    """
    Warm up inference kernels using representative denoiser calls only.

    This avoids wasting time on complete diffusion trajectories before timing.
    The goal is just to trigger CUDA kernel loading/selection, allocator setup,
    autocast paths, and SDPA/FlashAttention steady-state behavior.
    """
    if int(num_calls) <= 0:
        return

    cont = cfg.diffusion.continuous
    sigma_min = float(getattr(cont, "sigma_min", 0.002))
    sigma_max = float(getattr(cont, "sigma_max", 80.0))
    sigma_mid = math.sqrt(max(sigma_min * sigma_max, 1e-12))
    sigma_values = [sigma_max, sigma_mid, sigma_min]

    x, _, x0_hat, is_tokens, _, _, _ = make_synthetic_infer_state(
        cfg=cfg,
        batch_size=batch_size,
        device=device,
        dtype=torch.bfloat16,
        sigma_max=sigma_max,
    )

    use_sc = bool(getattr(cfg.model, "self_condition", False))
    sc_refresh_mode = str(sc_refresh_mode).lower()

    for j in range(int(num_calls)):
        sigma_value = sigma_values[j % len(sigma_values)]
        sigma = torch.full((batch_size,), float(sigma_value), device=device, dtype=torch.bfloat16)
        model_sc = x0_hat if (use_sc and x0_hat is not None and sc_refresh_mode == "carry") else None

        with bf16_autocast(enabled=use_amp):
            logits = model(x, sigma, model_sc)
            if is_tokens:
                denoised = torch.softmax(logits.float(), dim=-1).to(dtype=x.dtype)
            else:
                denoised = torch.sigmoid(logits.reshape_as(x).float()).to(dtype=x.dtype)

        if use_sc and sc_refresh_mode == "carry":
            x0_hat = denoised.detach()

    synchronize()


@torch.no_grad()
def run_inference_loop(
    *,
    model: nn.Module,
    cfg: Any,
    batch_size: int,
    num_steps: int,
    device: torch.device,
    use_amp: bool,
    sc_refresh_mode: str,
    profile_stochastic: bool,
    s_churn: float,
    s_noise: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Minimal sampler-like loop for profiling.

    This deliberately mirrors the dominant sampling cost: repeated denoiser
    calls on the model state. It avoids decode/cache/eval overhead and does not
    require an entropy schedule object. For deterministic profiling, it performs
    DDIM/probability-flow Euler updates on a Karras grid. For stochastic
    profiling, it applies a full-band EDM churn perturbation before each step.

    Returns final x and final clean estimate/probabilities x0_hat.
    """
    cont = cfg.diffusion.continuous
    sigma_min = float(getattr(cont, "sigma_min", 0.002))
    sigma_max = float(getattr(cont, "sigma_max", 80.0))
    rho = float(getattr(cont, "rho", 7.0))

    x, sigma_vec, x0_hat, is_tokens, _, _, _ = make_synthetic_infer_state(
        cfg=cfg,
        batch_size=batch_size,
        device=device,
        dtype=torch.bfloat16,
        sigma_max=sigma_max,
    )

    sigmas = karras_sigmas(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        num_steps=num_steps,
        device=device,
        dtype=torch.bfloat16,
    )

    sc_refresh_mode = str(sc_refresh_mode).lower()
    use_sc = bool(getattr(cfg.model, "self_condition", False))

    for i in range(int(num_steps)):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_batch = torch.full((batch_size,), float(sigma_i), device=device, dtype=torch.bfloat16)

        x_in = x
        sigma_in = sigma_batch

        if profile_stochastic and float(s_churn) > 0.0:
            gamma = float(s_churn) / max(int(num_steps), 1)
            sigma_hat = sigma_i * (1.0 + gamma)
            sigma_hat_batch = torch.full((batch_size,), float(sigma_hat), device=device, dtype=torch.bfloat16)
            noise_scale = torch.sqrt(torch.clamp(sigma_hat**2 - sigma_i**2, min=0.0))
            x_in = x + float(s_noise) * noise_scale.to(dtype=x.dtype) * torch.randn_like(x)
            sigma_in = sigma_hat_batch
        else:
            sigma_hat = sigma_i

        model_sc = x0_hat if (use_sc and x0_hat is not None and sc_refresh_mode == "carry") else None

        with bf16_autocast(enabled=use_amp):
            logits = model(x_in, sigma_in, model_sc)
            if is_tokens:
                # Token-space model outputs [B,T,V]. For score-style continuous
                # sampling we need a clean-state proxy in the same shape as x.
                # Softmax is the natural probability simplex projection.
                denoised = torch.softmax(logits.float(), dim=-1).to(dtype=x.dtype)
            else:
                denoised = torch.sigmoid(logits.reshape_as(x_in).float()).to(dtype=x.dtype)

        # Probability-flow Euler/DDIM update:
        # dx/dsigma = (x - D(x,sigma)) / sigma.
        dt = sigma_next - sigma_hat
        x = x_in + dt.to(dtype=x.dtype) * (x_in - denoised) / sigma_hat.to(dtype=x.dtype).clamp_min(1e-8)

        if use_sc and sc_refresh_mode == "carry":
            x0_hat = denoised.detach()

    return x, x0_hat


# -----------------------------------------------------------------------------
# Training profiling
# -----------------------------------------------------------------------------


def profile_training_model(
    *,
    dataset: str,
    representation_name: str,
    cfg: Any,
    source_config: str,
    batch_sizes: Sequence[int],
    warmup: int,
    steps: int,
    device: torch.device,
    ce_chunk_size: Optional[int],
    lr: float,
    use_amp: bool,
    progress: Optional[ProgressTracker] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    print(f"\n[Train] Building {dataset} / {representation_name}")
    print(f"        source={source_config}")
    print(f"        representation={cfg_repr(cfg)}, head_type={cfg.model.head_type}")

    model = SequenceVDTContinuousModel(cfg).to(device).to(torch.bfloat16)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    params = num_params(model)
    print(f"        params={params:,}")

    for B in batch_sizes:
        print(f"\n[Train] {dataset} / {representation_name} | B={B}")
        x = sigma = target = None
        try:
            clear_cuda()
            reset_peak_memory()

            x, sigma, target, is_tokens, T, V, m = make_synthetic_train_batch(
                cfg=cfg,
                batch_size=int(B),
                device=device,
                dtype=torch.bfloat16,
            )

            for _ in range(int(warmup)):
                _ = run_one_train_step(
                    model=model,
                    optimizer=optimizer,
                    x=x,
                    sigma=sigma,
                    target=target,
                    is_tokens=is_tokens,
                    ce_chunk_size=ce_chunk_size,
                    use_amp=use_amp,
                )

            synchronize()
            reset_peak_memory()

            t0 = time.perf_counter()
            last_loss = None
            for _ in range(int(steps)):
                last_loss = run_one_train_step(
                    model=model,
                    optimizer=optimizer,
                    x=x,
                    sigma=sigma,
                    target=target,
                    is_tokens=is_tokens,
                    ce_chunk_size=ce_chunk_size,
                    use_amp=use_amp,
                )
            synchronize()
            t1 = time.perf_counter()

            elapsed = t1 - t0
            step_ms = 1000.0 * elapsed / max(int(steps), 1)
            semantic_tok_per_sec = (int(B) * int(T) * int(steps)) / max(elapsed, 1e-12)
            mem = peak_gib()

            row = dict(
                experiment="train",
                dataset=dataset,
                representation=representation_name,
                source_config=source_config,
                status="ok",
                batch_size=int(B),
                T_semantic=int(T),
                model_sequence_len=int(cfg_model_len(cfg)),
                V=int(V),
                semantic_vocab_size=int(cfg_vocab_for_table(cfg, dataset=dataset)),
                bits_per_token=int(m),
                head_type=str(cfg.model.head_type),
                params=int(params),
                peak_vram_gib=float(mem),
                step_ms=float(step_ms),
                semantic_tokens_per_sec=float(semantic_tok_per_sec),
                loss=float(last_loss.item()) if last_loss is not None else None,
            )
            rows.append(row)

            print(
                f"        OK | peak={mem:.2f} GiB | step={step_ms:.1f} ms | "
                f"semantic tok/s={semantic_tok_per_sec:,.0f}"
            )
            if progress is not None:
                progress.update(
                    f"train {dataset}/{representation_name} B={B}: "
                    f"{mem:.2f} GiB, {semantic_tok_per_sec:,.0f} tok/s"
                )

        except RuntimeError as e:
            msg = str(e)
            if "out of memory" in msg.lower() or "cuda out of memory" in msg.lower():
                clear_cuda()
                row = dict(
                    experiment="train",
                    dataset=dataset,
                    representation=representation_name,
                    source_config=source_config,
                    status="oom",
                    batch_size=int(B),
                    T_semantic=int(cfg_semantic_len(cfg)),
                    model_sequence_len=int(cfg_model_len(cfg)),
                    V=int(getattr(cfg.data, "vocab_size", 0)),
                    semantic_vocab_size=int(cfg_vocab_for_table(cfg, dataset=dataset)),
                    bits_per_token=int(cfg_bits_per_token(cfg)),
                    head_type=str(cfg.model.head_type),
                    params=int(params),
                    peak_vram_gib=None,
                    step_ms=None,
                    semantic_tokens_per_sec=None,
                    loss=None,
                )
                rows.append(row)
                print("        OOM")
                if progress is not None:
                    progress.update(f"train {dataset}/{representation_name} B={B}: OOM")
                # Continue to next representation/dataset, but larger batches for
                # this same model are unlikely to fit.
                break
            raise
        finally:
            del x, sigma, target
            clear_cuda()

    del model, optimizer
    clear_cuda()
    return rows


# -----------------------------------------------------------------------------
# Inference profiling
# -----------------------------------------------------------------------------


def profile_inference_model(
    *,
    dataset: str,
    representation_name: str,
    cfg: Any,
    source_config: str,
    micro_batches: Sequence[int],
    warmup_calls: int,
    repeats: int,
    num_steps: int,
    device: torch.device,
    use_amp: bool,
    sc_refresh_mode: str,
    profile_stochastic: bool,
    s_churn: float,
    s_noise: float,
    progress: Optional[ProgressTracker] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    print(f"\n[Infer] Building {dataset} / {representation_name}")
    print(f"        source={source_config}")
    print(f"        representation={cfg_repr(cfg)}, head_type={cfg.model.head_type}")

    model = SequenceVDTContinuousModel(cfg).to(device).to(torch.bfloat16)
    model.eval()
    params = num_params(model)
    print(f"        params={params:,}")

    for B in micro_batches:
        print(f"[Infer] {dataset} / {representation_name} | micro_batch={B} | NFE={num_steps}")
        try:
            clear_cuda()
            reset_peak_memory()

            warmup_inference_denoiser_calls(
                model=model,
                cfg=cfg,
                batch_size=int(B),
                num_calls=int(warmup_calls),
                device=device,
                use_amp=use_amp,
                sc_refresh_mode=sc_refresh_mode,
            )

            reset_peak_memory()

            t0 = time.perf_counter()
            for _ in range(int(repeats)):
                _ = run_inference_loop(
                    model=model,
                    cfg=cfg,
                    batch_size=int(B),
                    num_steps=int(num_steps),
                    device=device,
                    use_amp=use_amp,
                    sc_refresh_mode=sc_refresh_mode,
                    profile_stochastic=profile_stochastic,
                    s_churn=s_churn,
                    s_noise=s_noise,
                )
            synchronize()
            t1 = time.perf_counter()

            elapsed = t1 - t0
            per_generation_ms = 1000.0 * elapsed / max(int(repeats), 1)
            T = cfg_semantic_len(cfg)
            generated_semantic_tokens = int(B) * int(T) * int(repeats)
            semantic_tok_per_sec = generated_semantic_tokens / max(elapsed, 1e-12)
            denoiser_calls = int(B) * int(num_steps) * int(repeats)
            denoiser_calls_per_sec = denoiser_calls / max(elapsed, 1e-12)
            semantic_token_denoiser_evals_per_sec = (
                int(B) * int(T) * int(num_steps) * int(repeats) / max(elapsed, 1e-12)
            )
            mem = peak_gib()

            row = dict(
                experiment="inference",
                dataset=dataset,
                representation=representation_name,
                source_config=source_config,
                status="ok",
                micro_batch_size=int(B),
                nfe=int(num_steps),
                T_semantic=int(T),
                model_sequence_len=int(cfg_model_len(cfg)),
                V=int(getattr(cfg.data, "vocab_size", 0)),
                semantic_vocab_size=int(cfg_vocab_for_table(cfg, dataset=dataset)),
                bits_per_token=int(cfg_bits_per_token(cfg)),
                head_type=str(cfg.model.head_type),
                params=int(params),
                sc_refresh_mode=str(sc_refresh_mode),
                stochastic_enabled=bool(profile_stochastic),
                s_churn=float(s_churn if profile_stochastic else 0.0),
                s_noise=float(s_noise if profile_stochastic else 1.0),
                peak_vram_gib=float(mem),
                generation_ms=float(per_generation_ms),
                semantic_tokens_per_sec=float(semantic_tok_per_sec),
                denoiser_calls_per_sec=float(denoiser_calls_per_sec),
                semantic_token_denoiser_evals_per_sec=float(semantic_token_denoiser_evals_per_sec),
            )
            rows.append(row)

            print(
                f"        OK | peak={mem:.2f} GiB | gen={per_generation_ms:.1f} ms | "
                f"semantic tok/s={semantic_tok_per_sec:,.0f} | "
                f"calls/s={denoiser_calls_per_sec:,.1f}"
            )
            if progress is not None:
                progress.update(
                    f"infer {dataset}/{representation_name} B={B}, NFE={num_steps}: "
                    f"{mem:.2f} GiB, {semantic_tok_per_sec:,.0f} tok/s"
                )

        except RuntimeError as e:
            msg = str(e)
            if "out of memory" in msg.lower() or "cuda out of memory" in msg.lower():
                clear_cuda()
                row = dict(
                    experiment="inference",
                    dataset=dataset,
                    representation=representation_name,
                    source_config=source_config,
                    status="oom",
                    micro_batch_size=int(B),
                    nfe=int(num_steps),
                    T_semantic=int(cfg_semantic_len(cfg)),
                    model_sequence_len=int(cfg_model_len(cfg)),
                    V=int(getattr(cfg.data, "vocab_size", 0)),
                    semantic_vocab_size=int(cfg_vocab_for_table(cfg, dataset=dataset)),
                    bits_per_token=int(cfg_bits_per_token(cfg)),
                    head_type=str(cfg.model.head_type),
                    params=int(params),
                    sc_refresh_mode=str(sc_refresh_mode),
                    stochastic_enabled=bool(profile_stochastic),
                    s_churn=float(s_churn if profile_stochastic else 0.0),
                    s_noise=float(s_noise if profile_stochastic else 1.0),
                    peak_vram_gib=None,
                    generation_ms=None,
                    semantic_tokens_per_sec=None,
                    denoiser_calls_per_sec=None,
                    semantic_token_denoiser_evals_per_sec=None,
                )
                rows.append(row)
                print("        OOM")
                if progress is not None:
                    progress.update(f"infer {dataset}/{representation_name} B={B}, NFE={num_steps}: OOM")
                break
            raise
        finally:
            clear_cuda()

    del model
    clear_cuda()
    return rows


# -----------------------------------------------------------------------------
# Boundary-only profiling
# -----------------------------------------------------------------------------


class TokenBoundary(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.head = TokenFullHead(d_model=d_model, out_dim=vocab_size, dropout=0.0)

    def forward(
        self,
        h: torch.Tensor,
        t_emb: torch.Tensor,
        target: torch.Tensor,
        *,
        ce_chunk_size: Optional[int],
    ) -> torch.Tensor:
        logits = self.head(h, t_emb)
        return chunked_cross_entropy(logits, target, chunk_size=ce_chunk_size)


class BitstreamBoundary(nn.Module):
    def __init__(
        self,
        d_model: int,
        bits_per_token: int,
        *,
        content_dim: int = 64,
        noisy_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.head = OptimalSkipMLPHead(
            d_model=d_model,
            patch_size=bits_per_token,
            out_dim=1,
            content_dim=content_dim,
            noisy_dim=noisy_dim,
            hidden_dim=hidden_dim,
            dropout=0.0,
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        x_denoised: torch.Tensor,
        x_noisy: torch.Tensor,
        t_emb: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.head(
            x_denoised=x_denoised,
            x_noisy=x_noisy,
            patch_tokens=patch_tokens,
            t_emb=t_emb,
        )
        return F.binary_cross_entropy_with_logits(
            logits.reshape(-1).float(),
            target.reshape(-1),
        )


def default_boundary_settings() -> List[Dict[str, Any]]:
    return [
        {"dataset": "LM1B", "T": 128, "V": 30522, "d": 768},
        {"dataset": "OWT", "T": 1024, "V": 65536, "d": 768},
        {"dataset": "Long context", "T": 4096, "V": 128000, "d": 2048},
    ]


def run_boundary_one(
    *,
    boundary_name: str,
    dataset: str,
    T: int,
    V: int,
    d: int,
    B: int,
    warmup: int,
    steps: int,
    device: torch.device,
    ce_chunk_size: Optional[int],
    lr: float,
    bit_content_dim: int,
    bit_head_hidden: int,
    use_amp: bool,
) -> Dict[str, Any]:
    m = int(math.ceil(math.log2(int(V))))

    if boundary_name == "Token boundary":
        model: nn.Module = TokenBoundary(d_model=d, vocab_size=V).to(device).to(torch.bfloat16)
    elif boundary_name == "Bitstream boundary":
        model = BitstreamBoundary(
            d_model=d,
            bits_per_token=m,
            content_dim=bit_content_dim,
            noisy_dim=bit_content_dim,
            hidden_dim=bit_head_hidden,
        ).to(device).to(torch.bfloat16)
    else:
        raise ValueError(f"Unknown boundary name: {boundary_name}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    h = torch.randn(B, T, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    t_emb = torch.randn(B, d, device=device, dtype=torch.bfloat16)
    x_denoised = x_noisy = target = None

    try:
        if boundary_name == "Token boundary":
            target = torch.randint(0, V, (B, T), device=device)
        else:
            x_denoised = torch.randn(B, T * m, bit_content_dim, device=device, dtype=torch.bfloat16)
            x_noisy = torch.randn(B, T * m, bit_content_dim, device=device, dtype=torch.bfloat16)
            target = torch.randint(0, 2, (B, T * m), device=device, dtype=torch.float32)

        def step() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            if h.grad is not None:
                h.grad = None
            with bf16_autocast(enabled=use_amp):
                if boundary_name == "Token boundary":
                    assert target is not None
                    loss = model(h, t_emb, target, ce_chunk_size=ce_chunk_size)
                else:
                    assert x_denoised is not None and x_noisy is not None and target is not None
                    loss = model(h, x_denoised, x_noisy, t_emb, target)
            loss.backward()
            optimizer.step()
            return loss.detach()

        clear_cuda()
        reset_peak_memory()
        for _ in range(int(warmup)):
            _ = step()
        synchronize()
        reset_peak_memory()

        t0 = time.perf_counter()
        last_loss = None
        for _ in range(int(steps)):
            last_loss = step()
        synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        return dict(
            experiment="boundary",
            dataset=dataset,
            boundary=boundary_name,
            status="ok",
            B=int(B),
            T=int(T),
            V=int(V),
            d=int(d),
            bits_per_token=int(m),
            params=int(num_params(model)),
            peak_vram_gib=float(peak_gib()),
            step_ms=float(1000.0 * elapsed / max(int(steps), 1)),
            semantic_tokens_per_sec=float(B * T * steps / max(elapsed, 1e-12)),
            loss=float(last_loss.item()) if last_loss is not None else None,
        )
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "cuda out of memory" in msg.lower():
            clear_cuda()
            return dict(
                experiment="boundary",
                dataset=dataset,
                boundary=boundary_name,
                status="oom",
                B=int(B),
                T=int(T),
                V=int(V),
                d=int(d),
                bits_per_token=int(m),
                params=int(num_params(model)),
                peak_vram_gib=None,
                step_ms=None,
                semantic_tokens_per_sec=None,
                loss=None,
            )
        raise
    finally:
        del model, optimizer, h, t_emb
        if x_denoised is not None:
            del x_denoised
        if x_noisy is not None:
            del x_noisy
        if target is not None:
            del target
        clear_cuda()


def profile_boundaries(
    args: argparse.Namespace,
    device: torch.device,
    progress: Optional[ProgressTracker] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    settings = default_boundary_settings()
    if args.boundary_settings_json is not None:
        with open(args.boundary_settings_json, "r", encoding="utf-8") as f:
            settings = json.load(f)

    for s in settings:
        dataset = str(s.get("dataset", "custom"))
        T = int(s["T"])
        V = int(s["V"])
        d = int(s["d"])
        for boundary in ["Token boundary", "Bitstream boundary"]:
            print(f"\n[Boundary] {dataset} / {boundary} | B={args.boundary_batch} T={T} V={V} d={d}")
            row = run_boundary_one(
                boundary_name=boundary,
                dataset=dataset,
                T=T,
                V=V,
                d=d,
                B=int(args.boundary_batch),
                warmup=int(args.warmup),
                steps=int(args.steps),
                device=device,
                ce_chunk_size=args.ce_chunk_size,
                lr=float(args.lr),
                bit_content_dim=int(args.bit_content_dim),
                bit_head_hidden=int(args.bit_head_hidden),
                use_amp=not args.no_amp,
            )
            rows.append(row)
            if row["status"] == "ok":
                print(
                    f"        OK | peak={row['peak_vram_gib']:.2f} GiB | "
                    f"step={row['step_ms']:.2f} ms"
                )
                if progress is not None:
                    progress.update(
                        f"boundary {dataset}/{boundary}: "
                        f"{row['peak_vram_gib']:.2f} GiB, {row['step_ms']:.2f} ms"
                    )
            else:
                print("        OOM")
                if progress is not None:
                    progress.update(f"boundary {dataset}/{boundary}: OOM")
    return rows


# -----------------------------------------------------------------------------
# Analytic scaling
# -----------------------------------------------------------------------------


def run_analytic_scaling() -> List[Dict[str, Any]]:
    settings = [
        {"setting": "LM1B", "B": 512, "T": 128, "V": 30522, "d": 768},
        {"setting": "OWT", "B": 128, "T": 1024, "V": 65536, "d": 768},
        {"setting": "OWT global batch", "B": 512, "T": 1024, "V": 65536, "d": 768},
        {"setting": "Long context", "B": 16, "T": 8192, "V": 65536, "d": 1024},
        {"setting": "Large vocabulary", "B": 16, "T": 4096, "V": 128000, "d": 2048},
        {"setting": "Large model/vocab", "B": 8, "T": 4096, "V": 128000, "d": 4096},
    ]

    rows: List[Dict[str, Any]] = []
    for s in settings:
        B, T, V, d = int(s["B"]), int(s["T"]), int(s["V"]), int(s["d"])
        m = int(math.ceil(math.log2(V)))
        token_logits = B * T * V
        bit_logits = B * T * m
        token_params = d * V
        rows.append(
            dict(
                setting=s["setting"],
                B=B,
                T=T,
                V=V,
                d=d,
                bits_per_token=m,
                token_logits=token_logits,
                bit_logits=bit_logits,
                logit_reduction=float(token_logits / bit_logits),
                token_logits_bf16_gib=float(token_logits * 2 / (1024**3)),
                bit_logits_bf16_mib=float(bit_logits * 2 / (1024**2)),
                token_output_params=token_params,
                token_output_params_m=float(token_params / 1e6),
                token_output_bf16_gib=float(token_params * 2 / (1024**3)),
            )
        )
    return rows


# -----------------------------------------------------------------------------
# LaTeX generation
# -----------------------------------------------------------------------------


def ok_rows(rows: List[Dict[str, Any]], **where: Any) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        keep = True
        for k, v in where.items():
            if r.get(k) != v:
                keep = False
                break
        if keep:
            out.append(r)
    return out


def ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or float(b) == 0.0:
        return None
    return float(a) / float(b)


def build_pair_table_rows(
    rows: List[Dict[str, Any]],
    *,
    experiment: str,
    dataset: str,
    size_key: str,
) -> List[Dict[str, Any]]:
    ds_rows = [r for r in rows if r.get("experiment") == experiment and r.get("dataset") == dataset]
    token = {int(r[size_key]): r for r in ds_rows if r.get("representation") == "Tokens" and r.get("status") == "ok"}
    bits = {int(r[size_key]): r for r in ds_rows if r.get("representation") == "\\methodname{}" and r.get("status") == "ok"}
    keys = sorted(set(token.keys()) & set(bits.keys()))
    out = []
    for k in keys:
        tr, br = token[k], bits[k]
        out.append(dict(size=k, token=tr, bits=br))
    return out


def latex_train_table(rows: List[Dict[str, Any]], *, dataset: str, label: str) -> str:
    pairs = build_pair_table_rows(rows, experiment="train", dataset=dataset, size_key="batch_size")
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{rcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Batch} & \multicolumn{2}{c}{\textbf{Peak VRAM (GiB)}} & \textbf{Mem.} & \multicolumn{2}{c}{\textbf{Semantic tok/s}} & \textbf{Speed} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){5-6}")
    lines.append(r"\textbf{size} & \textbf{Tokens} & \textbf{\methodname{}} & $\boldsymbol{\downarrow}$ & \textbf{Tokens} & \textbf{\methodname{}} & $\boldsymbol{\uparrow}$ \\")
    lines.append(r"\midrule")
    for p in pairs:
        tr, br = p["token"], p["bits"]
        mem_ratio = ratio(tr["peak_vram_gib"], br["peak_vram_gib"])
        speed_ratio = ratio(br["semantic_tokens_per_sec"], tr["semantic_tokens_per_sec"])
        lines.append(
            f"{p['size']} & {tr['peak_vram_gib']:.2f} & {br['peak_vram_gib']:.2f} & "
            f"${mem_ratio:.2f}\\times$ & {fmt_int(tr['semantic_tokens_per_sec'])} & "
            f"{fmt_int(br['semantic_tokens_per_sec'])} & ${speed_ratio:.2f}\\times$ \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        rf"\caption{{\textbf{{End-to-end training-step efficiency on {dataset}.}} "
        rf"Both models use the same SDT trunk and semantic sequence length for {dataset}; "
        rf"they differ in the output/loss boundary. Throughput is reported in semantic tokens per second.}}"
    )
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_inference_table(rows: List[Dict[str, Any]], *, dataset: str, label: str) -> str:
    pairs = build_pair_table_rows(rows, experiment="inference", dataset=dataset, size_key="micro_batch_size")
    nfe = None
    if pairs:
        nfe = int(pairs[0]["bits"].get("nfe", 0))
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{rcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Micro} & \multicolumn{2}{c}{\textbf{Peak VRAM (GiB)}} & \textbf{Mem.} & \multicolumn{2}{c}{\textbf{Generated semantic tok/s}} & \textbf{Speed} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){5-6}")
    lines.append(r"\textbf{batch} & \textbf{Tokens} & \textbf{\methodname{}} & $\boldsymbol{\downarrow}$ & \textbf{Tokens} & \textbf{\methodname{}} & $\boldsymbol{\uparrow}$ \\")
    lines.append(r"\midrule")
    for p in pairs:
        tr, br = p["token"], p["bits"]
        mem_ratio = ratio(tr["peak_vram_gib"], br["peak_vram_gib"])
        speed_ratio = ratio(br["semantic_tokens_per_sec"], tr["semantic_tokens_per_sec"])
        lines.append(
            f"{p['size']} & {tr['peak_vram_gib']:.2f} & {br['peak_vram_gib']:.2f} & "
            f"${mem_ratio:.2f}\\times$ & {fmt_int(tr['semantic_tokens_per_sec'])} & "
            f"{fmt_int(br['semantic_tokens_per_sec'])} & ${speed_ratio:.2f}\\times$ \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    nfe_text = f" at {nfe} NFEs" if nfe else ""
    lines.append(
        rf"\caption{{\textbf{{Generation efficiency on {dataset}{nfe_text}.}} "
        rf"Inference throughput measures completed generated semantic tokens per second, "
        rf"therefore including the full iterative denoising budget. The benchmark excludes decoding and external evaluation.}}"
    )
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_compact_summary_table(train_rows: List[Dict[str, Any]], infer_rows: List[Dict[str, Any]]) -> str:
    """One compact table for the main text, selecting the largest common fitting batch."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{llrcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Mode} & \textbf{Batch} & \textbf{Memory reduction} & \textbf{Throughput speedup} \\")
    lines.append(r"\midrule")

    for dataset in ["LM1B", "OWT"]:
        train_pairs = build_pair_table_rows(train_rows, experiment="train", dataset=dataset, size_key="batch_size")
        infer_pairs = build_pair_table_rows(infer_rows, experiment="inference", dataset=dataset, size_key="micro_batch_size")
        for mode, pairs in [("Train", train_pairs), ("Generate", infer_pairs)]:
            if not pairs:
                continue
            p = pairs[-1]
            tr, br = p["token"], p["bits"]
            mem_ratio = ratio(tr["peak_vram_gib"], br["peak_vram_gib"])
            speed_ratio = ratio(br["semantic_tokens_per_sec"], tr["semantic_tokens_per_sec"])
            lines.append(
                f"{dataset} & {mode} & {p['size']} & "
                f"${mem_ratio:.2f}\\times$ & ${speed_ratio:.2f}\\times$ \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{\textbf{End-to-end systems impact of the bitstream boundary.} "
        r"For each dataset and mode we show the largest common batch size that fits for both the token-space and bitstream models on the profiling GPU.}"
    )
    lines.append(r"\label{tab:efficiency_summary_main}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_boundary_table(rows: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Dataset} & \textbf{Boundary} & \textbf{$T$} & \textbf{$V$} & \textbf{$d$} & \textbf{Peak VRAM (GiB)} & \textbf{Step time (ms)} \\")
    lines.append(r"\midrule")
    for i, r in enumerate(rows):
        peak = r"\textbf{OOM}" if r["status"] == "oom" else fmt_float(r["peak_vram_gib"], 2)
        step = "--" if r["status"] == "oom" else fmt_float(r["step_ms"], 2)
        lines.append(
            f"{latex_escape(r['dataset'])} & {latex_escape(r['boundary'])} & {r['T']} & {int(r['V']):,} & {r['d']} & {peak} & {step} \\\\"
        )
        if r["boundary"] == "Bitstream boundary" and i != len(rows) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{\textbf{Boundary-only head and loss profiling.} "
        r"This isolates the vocabulary-wide output/loss boundary from the Transformer trunk.}"
    )
    lines.append(r"\label{tab:boundary_profiling}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def sci_latex(n: float) -> str:
    if n == 0:
        return "$0$"
    exp = int(math.floor(math.log10(abs(n))))
    mant = n / (10**exp)
    return rf"${mant:.2f}{{\times}}10^{{{exp}}}$"


def latex_analytic_table(rows: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Setting} & \textbf{Token logits $BTV$} & \textbf{Bit logits $BT\lceil\log_2 V\rceil$} & \textbf{Reduction} \\")
    lines.append(r"\midrule")
    for r in rows:
        setting = f"{latex_escape(r['setting'])}: $B={r['B']},T={r['T']},V={int(r['V']):,}$"
        lines.append(
            f"{setting} & {sci_latex(r['token_logits'])} & {sci_latex(r['bit_logits'])} & "
            f"${r['logit_reduction']:.0f}\\times$ \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{\textbf{Vocabulary-boundary tensor sizes.} "
        r"Token-based diffusion models form vocabulary-wide logits of size $BTV$, whereas \methodname{} forms bit logits of size $BT\lceil\log_2 V\rceil$.}"
    )
    lines.append(r"\label{tab:logit_scaling}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def write_latex_tables(
    *,
    outdir: Path,
    train_rows: List[Dict[str, Any]],
    inference_rows: List[Dict[str, Any]],
    boundary_rows: List[Dict[str, Any]],
    analytic_rows: List[Dict[str, Any]],
) -> None:
    parts = [
        latex_compact_summary_table(train_rows, inference_rows),
    ]

    if any(r.get("dataset") == "LM1B" for r in train_rows):
        parts.append(latex_train_table(train_rows, dataset="LM1B", label="tab:efficiency_train_lm1b"))
    if any(r.get("dataset") == "OWT" for r in train_rows):
        parts.append(latex_train_table(train_rows, dataset="OWT", label="tab:efficiency_train_owt"))
    if any(r.get("dataset") == "LM1B" for r in inference_rows):
        parts.append(latex_inference_table(inference_rows, dataset="LM1B", label="tab:efficiency_infer_lm1b"))
    if any(r.get("dataset") == "OWT" for r in inference_rows):
        parts.append(latex_inference_table(inference_rows, dataset="OWT", label="tab:efficiency_infer_owt"))
    if boundary_rows:
        parts.append(latex_boundary_table(boundary_rows))
    if analytic_rows:
        parts.append(latex_analytic_table(analytic_rows))

    tex = "\n\n".join(parts)
    path = outdir / "latex_tables.tex"
    path.write_text(tex, encoding="utf-8")
    print(f"\nWrote LaTeX tables to: {path}")


# -----------------------------------------------------------------------------
# CLI and orchestration
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile train/inference efficiency for token vs bitstream continuous DLMs."
    )

    parser.add_argument("--lm1b-bit-config", type=str, required=True)
    parser.add_argument("--lm1b-token-config", type=str, required=True)
    parser.add_argument("--owt-bit-config", type=str, default=None)
    parser.add_argument("--owt-token-config", type=str, default=None)
    parser.add_argument(
        "--synthesize-missing-token-configs",
        action="store_true",
        help="Synthesize missing token configs from the corresponding bit configs.",
    )

    parser.add_argument("--outdir", type=str, default="runs/efficiency_profile_train_infer")

    parser.add_argument(
        "--train-batch-sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
    )
    parser.add_argument(
        "--infer-micro-batches",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512],
    )
    parser.add_argument("--infer-steps", type=int, default=256)
    parser.add_argument("--infer-repeats", type=int, default=None)
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Legacy/default warmup. Used for training unless --train-warmup is set. For inference, prefer --infer-warmup-calls.",
    )
    parser.add_argument("--train-warmup", type=int, default=None)
    parser.add_argument(
        "--infer-warmup-calls",
        type=int,
        default=30,
        help="Number of representative denoiser calls used to warm up inference. This is not full-trajectory warmup.",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument(
        "--ce-chunk-size",
        type=int,
        default=2048,
        help="Chunk size for token CE. Set <=0 to disable chunking.",
    )

    parser.add_argument(
        "--self-condition",
        action="store_true",
        help="Enable self-conditioning for profiling. Recommended for inference if you want the exact sampling path; off by default for cleaner single-call train profiling.",
    )
    parser.add_argument(
        "--sc-refresh-mode",
        type=str,
        default="carry",
        choices=["carry", "none"],
        help="Self-conditioning mode used by the synthetic inference loop.",
    )

    parser.add_argument(
        "--profile-stochastic",
        action="store_true",
        help="Include EDM-style full-band churn perturbation in the inference loop.",
    )
    parser.add_argument("--s-churn", type=float, default=0.0)
    parser.add_argument("--s-noise", type=float, default=1.003)

    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--use-compile", action="store_true")

    flash_group = parser.add_mutually_exclusive_group()
    flash_group.add_argument("--flash-attn", dest="use_flash_attn", action="store_true")
    flash_group.add_argument("--no-flash-attn", dest="use_flash_attn", action="store_false")
    parser.set_defaults(use_flash_attn=None)

    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-boundary", action="store_true")
    parser.add_argument("--skip-analytic", action="store_true")

    parser.add_argument("--boundary-batch", type=int, default=16)
    parser.add_argument("--boundary-settings-json", type=str, default=None)
    parser.add_argument("--bit-content-dim", type=int, default=64)
    parser.add_argument("--bit-head-hidden", type=int, default=128)

    return parser.parse_args()


def prepare_dataset_pairs(args: argparse.Namespace) -> List[DatasetPair]:
    pairs: List[DatasetPair] = []

    lm1b_bit = load_config_from_path(args.lm1b_bit_config)
    lm1b_tok = load_config_from_path(args.lm1b_token_config)
    pairs.append(DatasetPair("LM1B", r"\methodname{}", lm1b_bit, args.lm1b_bit_config))
    pairs.append(DatasetPair("LM1B", "Tokens", lm1b_tok, args.lm1b_token_config))

    if args.owt_bit_config is not None:
        owt_bit = load_config_from_path(args.owt_bit_config)
        pairs.append(DatasetPair("OWT", r"\methodname{}", owt_bit, args.owt_bit_config))

        if args.owt_token_config is not None:
            owt_tok = load_config_from_path(args.owt_token_config)
            pairs.append(DatasetPair("OWT", "Tokens", owt_tok, args.owt_token_config))
        elif args.synthesize_missing_token_configs:
            owt_tok = synthesize_token_config_from_bit_config(
                owt_bit,
                dataset_name="OWT",
                vocab_size=65536,
                sequence_len_tokens=int(getattr(owt_bit.data, "sequence_len_tokens", 1024)),
            )
            pairs.append(DatasetPair("OWT", "Tokens", owt_tok, "synthetic_from_owt_bit_config"))

    return pairs


def main() -> None:
    args = parse_args()

    if args.ce_chunk_size is not None and args.ce_chunk_size <= 0:
        args.ce_chunk_size = None

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = ensure_cuda()
    use_amp = not bool(args.no_amp)
    infer_repeats = int(args.infer_repeats) if args.infer_repeats is not None else int(args.steps)
    train_warmup = int(args.train_warmup) if args.train_warmup is not None else int(args.warmup)

    print("=" * 80)
    print("Train/inference efficiency profiler")
    print("=" * 80)
    print(f"project_root       : {PROJECT_ROOT}")
    print(f"device             : {current_device_name()}")
    print(f"outdir             : {outdir}")
    print(f"train_batch_sizes  : {args.train_batch_sizes}")
    print(f"infer_micro_batches: {args.infer_micro_batches}")
    print(f"infer_steps        : {args.infer_steps}")
    print(f"train_warmup/steps : {train_warmup}/{args.steps}")
    print(f"infer_warmup_calls : {args.infer_warmup_calls}")
    print(f"infer_repeats      : {infer_repeats}")
    print(f"ce_chunk_size      : {args.ce_chunk_size}")
    print(f"self_condition     : {args.self_condition}")
    print(f"use_amp            : {use_amp}")
    print(f"use_compile        : {args.use_compile}")
    print(f"use_flash_attn     : {args.use_flash_attn}")
    print(f"profile_stochastic : {args.profile_stochastic}")

    pairs = prepare_dataset_pairs(args)
    normalized_pairs: List[DatasetPair] = []
    for p in pairs:
        cfg = set_common_profile_flags(
            p.cfg,
            self_condition=bool(args.self_condition),
            use_flash_attn=args.use_flash_attn,
            use_compile=bool(args.use_compile),
        )
        normalized_pairs.append(DatasetPair(p.dataset, p.representation_name, cfg, p.source_config))

    train_rows: List[Dict[str, Any]] = []
    inference_rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    analytic_rows: List[Dict[str, Any]] = []

    total_units = 0
    if not args.skip_train:
        total_units += len(normalized_pairs) * len(args.train_batch_sizes)
    if not args.skip_inference:
        total_units += len(normalized_pairs) * len(args.infer_micro_batches)
    if not args.skip_boundary:
        boundary_settings = default_boundary_settings()
        if args.boundary_settings_json is not None:
            with open(args.boundary_settings_json, "r", encoding="utf-8") as f:
                boundary_settings = json.load(f)
        total_units += 2 * len(boundary_settings)
    progress = ProgressTracker(total=max(total_units, 1), label="efficiency-profile")

    if not args.skip_train:
        print("\n" + "=" * 80)
        print("TRAINING PROFILING")
        print("=" * 80)
        for p in normalized_pairs:
            train_rows.extend(
                profile_training_model(
                    dataset=p.dataset,
                    representation_name=p.representation_name,
                    cfg=p.cfg,
                    source_config=p.source_config,
                    batch_sizes=args.train_batch_sizes,
                    warmup=int(train_warmup),
                    steps=int(args.steps),
                    device=device,
                    ce_chunk_size=args.ce_chunk_size,
                    lr=float(args.lr),
                    use_amp=use_amp,
                    progress=progress,
                )
            )
        write_jsonl(outdir / "train_results.jsonl", train_rows)
        write_csv(outdir / "train_results.csv", train_rows)

    if not args.skip_inference:
        print("\n" + "=" * 80)
        print("INFERENCE PROFILING")
        print("=" * 80)
        for p in normalized_pairs:
            inference_rows.extend(
                profile_inference_model(
                    dataset=p.dataset,
                    representation_name=p.representation_name,
                    cfg=p.cfg,
                    source_config=p.source_config,
                    micro_batches=args.infer_micro_batches,
                    warmup_calls=int(args.infer_warmup_calls),
                    repeats=int(infer_repeats),
                    num_steps=int(args.infer_steps),
                    device=device,
                    use_amp=use_amp,
                    sc_refresh_mode=str(args.sc_refresh_mode),
                    profile_stochastic=bool(args.profile_stochastic),
                    s_churn=float(args.s_churn),
                    s_noise=float(args.s_noise),
                    progress=progress,
                )
            )
        write_jsonl(outdir / "inference_results.jsonl", inference_rows)
        write_csv(outdir / "inference_results.csv", inference_rows)

    if not args.skip_boundary:
        print("\n" + "=" * 80)
        print("BOUNDARY-ONLY PROFILING")
        print("=" * 80)
        boundary_rows = profile_boundaries(args, device, progress=progress)
        write_jsonl(outdir / "boundary_results.jsonl", boundary_rows)
        write_csv(outdir / "boundary_results.csv", boundary_rows)

    if not args.skip_analytic:
        analytic_rows = run_analytic_scaling()
        write_jsonl(outdir / "analytic_scaling.jsonl", analytic_rows)
        write_csv(outdir / "analytic_scaling.csv", analytic_rows)

    write_latex_tables(
        outdir=outdir,
        train_rows=train_rows,
        inference_rows=inference_rows,
        boundary_rows=boundary_rows,
        analytic_rows=analytic_rows,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
