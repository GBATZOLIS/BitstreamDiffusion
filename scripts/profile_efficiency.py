#!/usr/bin/env python3
"""
Profile memory/compute efficiency of token-space continuous diffusion vs.
patched bitstream diffusion.

Run from repository root:

python scripts/profile_efficiency.py \
  --bit-config configs/lm1b/continuous/ablations/edm.py \
  --token-config configs/lm1b/continuous/ablations/onehot_tokens_ce.py \
  --outdir runs/efficiency_profile_lm1b \
  --batch-sizes 64 128 256 512 \
  --steps 10 \
  --warmup 3

Outputs:
  - e2e_results.jsonl
  - e2e_results.csv
  - boundary_results.jsonl
  - boundary_results.csv
  - analytic_scaling.csv
  - latex_tables.tex
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Repository imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sdt import (  # noqa: E402
    SequenceVDTContinuousModel,
    TokenFullHead,
    OptimalSkipMLPHead,
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


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


def num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def set_common_profile_flags(cfg: Any, *, self_condition: bool, use_flash_attn: Optional[bool]) -> Any:
    """
    Mutate config for profiling. We deliberately disable compile because compile
    warmup/caching can obscure raw step timing. FlashAttention can be kept as in
    the config or overridden by CLI.
    """
    if hasattr(cfg, "train"):
        cfg.train.use_compile = False
        cfg.train.use_fp16 = True
        cfg.train.amp_dtype = "bf16"
        cfg.train.allow_tf32 = True

    if hasattr(cfg, "evaluation"):
        cfg.evaluation.use_compile = False

    cfg.model.self_condition = bool(self_condition)

    if use_flash_attn is not None and hasattr(cfg.model, "use_flash_attn"):
        cfg.model.use_flash_attn = bool(use_flash_attn)

    return cfg


def bf16_autocast():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def chunked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Cross-entropy over flattened [B*T, V] logits. If chunk_size is provided,
    computes a weighted mean over chunks. This reduces CE temporary memory but
    does not remove the full logits tensor already produced by the model/head.
    """
    logits_2d = logits.reshape(-1, logits.shape[-1])
    target_1d = target.reshape(-1)

    if chunk_size is None or chunk_size <= 0 or logits_2d.shape[0] <= chunk_size:
        return F.cross_entropy(logits_2d.float(), target_1d)

    total_loss = logits_2d.new_zeros((), dtype=torch.float32)
    total_count = 0

    for start in range(0, logits_2d.shape[0], chunk_size):
        end = min(start + chunk_size, logits_2d.shape[0])
        loss = F.cross_entropy(
            logits_2d[start:end].float(),
            target_1d[start:end],
            reduction="sum",
        )
        total_loss = total_loss + loss
        total_count += end - start

    return total_loss / max(total_count, 1)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape_model_name(name: str) -> str:
    return name.replace("_", r"\_")


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


# ---------------------------------------------------------------------
# Synthetic batch construction
# ---------------------------------------------------------------------


def make_synthetic_batch(
    *,
    cfg: Any,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, int, int, int]:
    """
    Returns:
      x, sigma, target, is_tokens, T_semantic, V, m
    """
    representation = str(cfg.data.representation).lower()
    is_tokens = representation == "tokens"

    T_semantic = int(getattr(cfg.data, "sequence_len_tokens", getattr(cfg.data, "sequence_len", 128)))
    V = int(getattr(cfg.data, "vocab_size", 0))
    m = int(getattr(cfg.data, "bits_per_token", 1))

    sigma = torch.ones(batch_size, device=device, dtype=dtype)

    if is_tokens:
        # Continuous token/one-hot state: [B, T, V].
        # Random dense state is enough for profiling projection/memory.
        x = torch.randn(batch_size, T_semantic, V, device=device, dtype=dtype)
        target = torch.randint(0, V, (batch_size, T_semantic), device=device)
    else:
        # Continuous bitstream state: [B, T*m].
        S_bits = T_semantic * m
        x = torch.randn(batch_size, S_bits, device=device, dtype=dtype)
        target = torch.randint(0, 2, (batch_size, S_bits), device=device, dtype=torch.float32)

    return x, sigma, target, is_tokens, T_semantic, V, m


# ---------------------------------------------------------------------
# Experiment 1: end-to-end model profiling
# ---------------------------------------------------------------------


def run_one_e2e_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    is_tokens: bool,
    vocab_size: int,
    ce_chunk_size: Optional[int],
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)

    with bf16_autocast():
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


def profile_e2e_model(
    *,
    name: str,
    cfg: Any,
    batch_sizes: List[int],
    warmup: int,
    steps: int,
    device: torch.device,
    ce_chunk_size: Optional[int],
    lr: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    representation = str(cfg.data.representation).lower()
    is_tokens_cfg = representation == "tokens"

    print(f"\n[E2E] Building model: {name}")
    print(f"      representation={representation}, head_type={cfg.model.head_type}")

    model = SequenceVDTContinuousModel(cfg).to(device).to(torch.bfloat16)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model_params = num_params(model)
    print(f"      params={model_params:,}")

    for B in batch_sizes:
        print(f"\n[E2E] {name} | batch={B}")

        x = sigma = target = None
        try:
            clear_cuda()
            reset_peak_memory()

            x, sigma, target, is_tokens, T_semantic, V, m = make_synthetic_batch(
                cfg=cfg,
                batch_size=B,
                device=device,
                dtype=torch.bfloat16,
            )

            assert is_tokens == is_tokens_cfg

            # Warmup
            for _ in range(warmup):
                _ = run_one_e2e_step(
                    model=model,
                    optimizer=optimizer,
                    x=x,
                    sigma=sigma,
                    target=target,
                    is_tokens=is_tokens,
                    vocab_size=V,
                    ce_chunk_size=ce_chunk_size,
                )

            synchronize()
            reset_peak_memory()

            # Timed loop
            t0 = time.perf_counter()
            last_loss = None
            for _ in range(steps):
                last_loss = run_one_e2e_step(
                    model=model,
                    optimizer=optimizer,
                    x=x,
                    sigma=sigma,
                    target=target,
                    is_tokens=is_tokens,
                    vocab_size=V,
                    ce_chunk_size=ce_chunk_size,
                )
            synchronize()
            t1 = time.perf_counter()

            elapsed = t1 - t0
            step_ms = elapsed * 1000.0 / steps
            semantic_tok_per_sec = (B * T_semantic * steps) / elapsed
            mem = peak_gib()

            row = {
                "experiment": "e2e",
                "model": name,
                "status": "ok",
                "batch_size": B,
                "T_semantic": T_semantic,
                "V": V,
                "bits_per_token": m,
                "representation": representation,
                "head_type": str(cfg.model.head_type),
                "params": model_params,
                "peak_vram_gib": mem,
                "step_ms": step_ms,
                "semantic_tokens_per_sec": semantic_tok_per_sec,
                "loss": float(last_loss.item()) if last_loss is not None else None,
            }
            rows.append(row)

            print(
                f"      OK | peak={mem:.2f} GiB | step={step_ms:.1f} ms | "
                f"semantic tok/s={semantic_tok_per_sec:,.0f}"
            )

        except RuntimeError as e:
            msg = str(e)
            if "out of memory" in msg.lower() or "cuda out of memory" in msg.lower():
                clear_cuda()
                row = {
                    "experiment": "e2e",
                    "model": name,
                    "status": "oom",
                    "batch_size": B,
                    "T_semantic": int(getattr(cfg.data, "sequence_len_tokens", 128)),
                    "V": int(getattr(cfg.data, "vocab_size", 0)),
                    "bits_per_token": int(getattr(cfg.data, "bits_per_token", 1)),
                    "representation": representation,
                    "head_type": str(cfg.model.head_type),
                    "params": model_params,
                    "peak_vram_gib": None,
                    "step_ms": None,
                    "semantic_tokens_per_sec": None,
                    "loss": None,
                }
                rows.append(row)
                print("      OOM")
                break
            raise

        finally:
            del x, sigma, target
            clear_cuda()

    del model, optimizer
    clear_cuda()
    return rows


def run_e2e_profiling(args: argparse.Namespace, device: torch.device) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: END-TO-END TRAINING-STEP PROFILING")
    print("=" * 80)

    bit_cfg = load_config_from_path(args.bit_config)
    token_cfg = load_config_from_path(args.token_config)

    bit_cfg = set_common_profile_flags(
        bit_cfg,
        self_condition=args.self_condition,
        use_flash_attn=args.use_flash_attn,
    )
    token_cfg = set_common_profile_flags(
        token_cfg,
        self_condition=args.self_condition,
        use_flash_attn=args.use_flash_attn,
    )

    rows: List[Dict[str, Any]] = []

    rows += profile_e2e_model(
        name="Continuous tokens",
        cfg=token_cfg,
        batch_sizes=args.batch_sizes,
        warmup=args.warmup,
        steps=args.steps,
        device=device,
        ce_chunk_size=args.ce_chunk_size,
        lr=args.lr,
    )

    rows += profile_e2e_model(
        name="Patched bitstream",
        cfg=bit_cfg,
        batch_sizes=args.batch_sizes,
        warmup=args.warmup,
        steps=args.steps,
        device=device,
        ce_chunk_size=None,
        lr=args.lr,
    )

    return rows


# ---------------------------------------------------------------------
# Experiment 2: boundary-only profiling
# ---------------------------------------------------------------------


class TokenBoundary(nn.Module):
    """
    Token output boundary: TokenFullHead + CE.

    This mirrors the vocabulary-wide output/loss boundary. The trunk activations
    are synthetic [B, T, d].
    """

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
    """
    Bitstream output boundary: OptimalSkipMLPHead + BCE.

    The patch tokens are synthetic [B, T, d]. The bit-level skip tensors are
    synthetic [B, T*m, C]. This matches the shape contract of the real head.
    """

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


def boundary_sweep_settings_from_args(args: argparse.Namespace) -> List[Dict[str, int]]:
    if args.boundary_settings_json is not None:
        with open(args.boundary_settings_json, "r") as f:
            settings = json.load(f)
        return settings

    # Minimal but informative defaults.
    return [
        {"T": 128, "V": 30522, "d": 768},
        {"T": 1024, "V": 65536, "d": 1024},
        {"T": 4096, "V": 128000, "d": 2048},
    ]


def run_boundary_one(
    *,
    name: str,
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
    bit_hidden_dim: int,
) -> Dict[str, Any]:
    m = math.ceil(math.log2(V))

    model: nn.Module
    if name == "Token boundary":
        model = TokenBoundary(d_model=d, vocab_size=V).to(device).to(torch.bfloat16)
    elif name == "Bitstream boundary":
        model = BitstreamBoundary(
            d_model=d,
            bits_per_token=m,
            content_dim=bit_content_dim,
            noisy_dim=bit_content_dim,
            hidden_dim=bit_hidden_dim,
        ).to(device).to(torch.bfloat16)
    else:
        raise ValueError(f"Unknown boundary name: {name}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Synthetic trunk activations.
    h = torch.randn(B, T, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    t_emb = torch.randn(B, d, device=device, dtype=torch.bfloat16)

    x_denoised = x_noisy = target = None

    try:
        if name == "Token boundary":
            target = torch.randint(0, V, (B, T), device=device)
        else:
            x_denoised = torch.randn(
                B,
                T * m,
                bit_content_dim,
                device=device,
                dtype=torch.bfloat16,
            )
            x_noisy = torch.randn(
                B,
                T * m,
                bit_content_dim,
                device=device,
                dtype=torch.bfloat16,
            )
            target = torch.randint(0, 2, (B, T * m), device=device, dtype=torch.float32)

        def step() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            if h.grad is not None:
                h.grad = None

            with bf16_autocast():
                if name == "Token boundary":
                    assert target is not None
                    loss = model(h, t_emb, target, ce_chunk_size=ce_chunk_size)
                else:
                    assert x_denoised is not None
                    assert x_noisy is not None
                    assert target is not None
                    loss = model(h, x_denoised, x_noisy, t_emb, target)

            loss.backward()
            optimizer.step()
            return loss.detach()

        clear_cuda()
        reset_peak_memory()

        for _ in range(warmup):
            _ = step()

        synchronize()
        reset_peak_memory()

        t0 = time.perf_counter()
        last_loss = None
        for _ in range(steps):
            last_loss = step()
        synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        step_ms = elapsed * 1000.0 / steps
        mem = peak_gib()

        return {
            "experiment": "boundary",
            "boundary": name,
            "status": "ok",
            "B": B,
            "T": T,
            "V": V,
            "d": d,
            "bits_per_token": m,
            "params": num_params(model),
            "peak_vram_gib": mem,
            "step_ms": step_ms,
            "semantic_tokens_per_sec": (B * T * steps) / elapsed,
            "loss": float(last_loss.item()) if last_loss is not None else None,
        }

    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "cuda out of memory" in msg.lower():
            clear_cuda()
            return {
                "experiment": "boundary",
                "boundary": name,
                "status": "oom",
                "B": B,
                "T": T,
                "V": V,
                "d": d,
                "bits_per_token": m,
                "params": num_params(model),
                "peak_vram_gib": None,
                "step_ms": None,
                "semantic_tokens_per_sec": None,
                "loss": None,
            }
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


def run_boundary_profiling(args: argparse.Namespace, device: torch.device) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: BOUNDARY-ONLY HEAD+LOSS PROFILING")
    print("=" * 80)

    settings = boundary_sweep_settings_from_args(args)
    rows: List[Dict[str, Any]] = []

    for setting in settings:
        T = int(setting["T"])
        V = int(setting["V"])
        d = int(setting["d"])

        for boundary_name in ["Token boundary", "Bitstream boundary"]:
            print(f"\n[Boundary] {boundary_name} | B={args.boundary_batch} T={T} V={V} d={d}")

            row = run_boundary_one(
                name=boundary_name,
                T=T,
                V=V,
                d=d,
                B=args.boundary_batch,
                warmup=args.warmup,
                steps=args.steps,
                device=device,
                ce_chunk_size=args.ce_chunk_size,
                lr=args.lr,
                bit_content_dim=args.bit_content_dim,
                bit_hidden_dim=args.bit_head_hidden,
            )
            rows.append(row)

            if row["status"] == "ok":
                print(
                    f"      OK | peak={row['peak_vram_gib']:.2f} GiB | "
                    f"step={row['step_ms']:.2f} ms | params={row['params']:,}"
                )
            else:
                print("      OOM")

    return rows


# ---------------------------------------------------------------------
# Analytic scaling tables
# ---------------------------------------------------------------------


def run_analytic_scaling() -> List[Dict[str, Any]]:
    settings = [
        {
            "setting": "LM1B",
            "B": 512,
            "T": 128,
            "V": 30522,
            "d": 768,
        },
        {
            "setting": "Long context",
            "B": 16,
            "T": 8192,
            "V": 65536,
            "d": 1024,
        },
        {
            "setting": "Large vocab",
            "B": 16,
            "T": 4096,
            "V": 128000,
            "d": 2048,
        },
        {
            "setting": "Large model/vocab",
            "B": 8,
            "T": 4096,
            "V": 128000,
            "d": 4096,
        },
    ]

    rows: List[Dict[str, Any]] = []

    for s in settings:
        B = int(s["B"])
        T = int(s["T"])
        V = int(s["V"])
        d = int(s["d"])
        m = math.ceil(math.log2(V))

        token_logits = B * T * V
        bit_logits = B * T * m
        reduction = token_logits / bit_logits

        token_logits_bf16_gib = token_logits * 2 / (1024**3)
        bit_logits_bf16_mib = bit_logits * 2 / (1024**2)

        token_output_params = d * V
        token_output_bf16_gib = token_output_params * 2 / (1024**3)

        rows.append(
            {
                "setting": s["setting"],
                "B": B,
                "T": T,
                "V": V,
                "d": d,
                "bits_per_token": m,
                "token_logits": token_logits,
                "bit_logits": bit_logits,
                "logit_reduction": reduction,
                "token_logits_bf16_gib": token_logits_bf16_gib,
                "bit_logits_bf16_mib": bit_logits_bf16_mib,
                "token_output_params": token_output_params,
                "token_output_bf16_gib": token_output_bf16_gib,
            }
        )

    return rows


# ---------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------


def sci_latex(n: float) -> str:
    if n == 0:
        return "$0$"
    exp = int(math.floor(math.log10(abs(n))))
    mant = n / (10**exp)
    return rf"${mant:.2f}{{\times}}10^{{{exp}}}$"


def latex_e2e_table(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Representation} & \textbf{Batch} & \textbf{Peak VRAM (GiB)} & "
        r"\textbf{Step time (ms)} & \textbf{Semantic tok/s} \\"
    )
    lines.append(r"\midrule")

    ordered_models = ["Continuous tokens", "Patched bitstream"]
    for mi, model_name in enumerate(ordered_models):
        model_rows = [r for r in rows if r["model"] == model_name]
        model_rows = sorted(model_rows, key=lambda r: int(r["batch_size"]))

        for r in model_rows:
            if r["status"] == "oom":
                peak = r"\textbf{OOM}"
                step = "--"
                toks = "--"
            else:
                peak = fmt_float(r["peak_vram_gib"], 2)
                step = fmt_float(r["step_ms"], 1)
                toks = fmt_int(r["semantic_tokens_per_sec"])

            lines.append(
                f"{latex_escape_model_name(model_name)} & {r['batch_size']} & "
                f"{peak} & {step} & {toks} \\\\"
            )

        if mi == 0:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{\textbf{End-to-end training-step efficiency.} "
        r"Both models use the same semantic sequence length $T=128$ and the same "
        r"12-layer Transformer trunk. The token-space model pays a vocabulary-wide "
        r"output/loss cost, whereas \methodname{} predicts $m=15$ bit logits per "
        r"semantic token. Throughput is reported in semantic tokens per second.}"
    )
    lines.append(r"\label{tab:efficiency_e2e}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_boundary_table(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Boundary} & \textbf{$T$} & \textbf{$V$} & \textbf{$d$} & "
        r"\textbf{Peak VRAM (GiB)} & \textbf{Step time (ms)} \\"
    )
    lines.append(r"\midrule")

    for i, r in enumerate(rows):
        if r["status"] == "oom":
            peak = r"\textbf{OOM}"
            step = "--"
        else:
            peak = fmt_float(r["peak_vram_gib"], 2)
            step = fmt_float(r["step_ms"], 2)

        lines.append(
            f"{latex_escape_model_name(r['boundary'])} & {r['T']} & {int(r['V']):,} & "
            f"{r['d']} & {peak} & {step} \\\\"
        )

        # Separator after each pair.
        if r["boundary"] == "Bitstream boundary" and i != len(rows) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{\textbf{Boundary-only head and loss profiling.} "
        r"This isolates the vocabulary-wide output/loss boundary from the "
        r"Transformer trunk. The token boundary uses a dense $d\to V$ head and "
        r"cross-entropy; the bitstream boundary uses the compact bit head and BCE.}"
    )
    lines.append(r"\label{tab:boundary_profiling}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_analytic_table(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Setting} & \textbf{Token logits $BTV$} & "
        r"\textbf{Bit logits $BT\lceil\log_2 V\rceil$} & \textbf{Reduction} \\"
    )
    lines.append(r"\midrule")

    for r in rows:
        setting = (
            f"{r['setting']}: "
            rf"$B={r['B']},T={r['T']},V={int(r['V']):,}$"
        )
        lines.append(
            f"{setting} & {sci_latex(r['token_logits'])} & "
            f"{sci_latex(r['bit_logits'])} & ${r['logit_reduction']:.0f}\\times$ \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{\textbf{Vocabulary-boundary tensor sizes.} "
        r"Token-based diffusion models form vocabulary-wide logits of size $BTV$, "
        r"whereas \methodname{} forms bit logits of size "
        r"$BT\lceil\log_2 V\rceil$.}"
    )
    lines.append(r"\label{tab:logit_scaling}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_head_param_table(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Setting} & \textbf{$d$} & \textbf{$V$} & "
        r"\textbf{Token output params $dV$} \\"
    )
    lines.append(r"\midrule")

    seen = set()
    for r in rows:
        key = (r["d"], r["V"])
        if key in seen:
            continue
        seen.add(key)

        params_m = r["token_output_params"] / 1e6
        mem = r["token_output_bf16_gib"]
        lines.append(
            f"{r['setting']} & {r['d']} & {int(r['V']):,} & "
            f"{params_m:.1f}M ({mem:.2f} GiB bf16) \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{\textbf{Vocabulary-head parameter scaling.} "
        r"A dense token output head scales linearly with vocabulary size. "
        r"The bitstream output boundary replaces this vocabulary-wide classifier "
        r"with a compact bitwise head whose size grows only with "
        r"$\lceil\log_2 V\rceil$.}"
    )
    lines.append(r"\label{tab:vocab_head_params}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def write_latex_tables(
    *,
    outdir: Path,
    e2e_rows: List[Dict[str, Any]],
    boundary_rows: List[Dict[str, Any]],
    analytic_rows: List[Dict[str, Any]],
) -> None:
    tex = "\n\n".join(
        [
            latex_e2e_table(e2e_rows),
            latex_boundary_table(boundary_rows),
            latex_analytic_table(analytic_rows),
            latex_head_param_table(analytic_rows),
        ]
    )

    path = outdir / "latex_tables.tex"
    path.write_text(tex)
    print(f"\nWrote LaTeX tables to: {path}")

    print("\n" + "=" * 80)
    print("LATEX TABLES")
    print("=" * 80)
    print(tex)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile bitstream vs token diffusion efficiency.")

    parser.add_argument(
        "--bit-config",
        type=str,
        default="configs/lm1b/continuous/ablations/edm.py",
        help="Path to patched bitstream config.",
    )
    parser.add_argument(
        "--token-config",
        type=str,
        default="configs/lm1b/continuous/ablations/onehot_tokens_ce.py",
        help="Path to token one-hot config.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="runs/efficiency_profile_lm1b",
        help="Output directory for JSONL/CSV/LaTeX results.",
    )

    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="Batch sizes for end-to-end profiling.",
    )
    parser.add_argument(
        "--boundary-batch",
        type=int,
        default=16,
        help="Batch size for boundary-only profiling.",
    )
    parser.add_argument(
        "--boundary-settings-json",
        type=str,
        default=None,
        help=(
            "Optional JSON file with boundary settings, e.g. "
            '[{"T":128,"V":30522,"d":768}, ...].'
        ),
    )

    parser.add_argument("--warmup", type=int, default=3, help="Warmup steps.")
    parser.add_argument("--steps", type=int, default=10, help="Timed profiling steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")

    parser.add_argument(
        "--ce-chunk-size",
        type=int,
        default=2048,
        help=(
            "Chunk size for token CE. Set <=0 to disable chunking. "
            "Chunking reduces CE temporary memory but not full-logit memory."
        ),
    )

    parser.add_argument(
        "--self-condition",
        action="store_true",
        help=(
            "Enable self-conditioning in the model during profiling. "
            "Default is off for cleaner single-pass profiling."
        ),
    )

    flash_group = parser.add_mutually_exclusive_group()
    flash_group.add_argument(
        "--flash-attn",
        dest="use_flash_attn",
        action="store_true",
        help="Force cfg.model.use_flash_attn=True.",
    )
    flash_group.add_argument(
        "--no-flash-attn",
        dest="use_flash_attn",
        action="store_false",
        help="Force cfg.model.use_flash_attn=False.",
    )
    parser.set_defaults(use_flash_attn=None)

    parser.add_argument(
        "--bit-content-dim",
        type=int,
        default=64,
        help="Content/noisy dim for boundary-only bitstream head.",
    )
    parser.add_argument(
        "--bit-head-hidden",
        type=int,
        default=128,
        help="Hidden dim for boundary-only bitstream head.",
    )

    parser.add_argument(
        "--skip-e2e",
        action="store_true",
        help="Skip end-to-end profiling.",
    )
    parser.add_argument(
        "--skip-boundary",
        action="store_true",
        help="Skip boundary-only profiling.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if args.ce_chunk_size is not None and args.ce_chunk_size <= 0:
        args.ce_chunk_size = None

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = ensure_cuda()

    print("=" * 80)
    print("Efficiency profiler")
    print("=" * 80)
    print(f"project_root       : {PROJECT_ROOT}")
    print(f"device             : {torch.cuda.get_device_name(device)}")
    print(f"bit_config         : {args.bit_config}")
    print(f"token_config       : {args.token_config}")
    print(f"outdir             : {outdir}")
    print(f"batch_sizes        : {args.batch_sizes}")
    print(f"boundary_batch     : {args.boundary_batch}")
    print(f"warmup/steps       : {args.warmup}/{args.steps}")
    print(f"ce_chunk_size      : {args.ce_chunk_size}")
    print(f"self_condition     : {args.self_condition}")
    print(f"use_flash_attn     : {args.use_flash_attn}")

    e2e_rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []

    if not args.skip_e2e:
        e2e_rows = run_e2e_profiling(args, device)
        write_jsonl(outdir / "e2e_results.jsonl", e2e_rows)
        write_csv(outdir / "e2e_results.csv", e2e_rows)
        print(f"\nWrote E2E results to {outdir / 'e2e_results.csv'}")

    if not args.skip_boundary:
        boundary_rows = run_boundary_profiling(args, device)
        write_jsonl(outdir / "boundary_results.jsonl", boundary_rows)
        write_csv(outdir / "boundary_results.csv", boundary_rows)
        print(f"\nWrote boundary results to {outdir / 'boundary_results.csv'}")

    analytic_rows = run_analytic_scaling()
    write_jsonl(outdir / "analytic_scaling.jsonl", analytic_rows)
    write_csv(outdir / "analytic_scaling.csv", analytic_rows)
    print(f"\nWrote analytic scaling to {outdir / 'analytic_scaling.csv'}")

    write_latex_tables(
        outdir=outdir,
        e2e_rows=e2e_rows,
        boundary_rows=boundary_rows,
        analytic_rows=analytic_rows,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()