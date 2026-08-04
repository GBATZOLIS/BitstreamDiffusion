from __future__ import annotations

import os
# Fix for some torch.compile interactions with CUDA Graphs
os.environ.setdefault("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", "1")

import contextlib
import math
import random
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from utils.tb_manager import TBManager
from tqdm import tqdm

from data import get_dataloaders
from data.proteins import get_dima_loader
from data.uniref50 import get_evodiff_uniref50_loader
from models import create_model
from utils.ema import EMA
from utils.optim import get_optimizer_and_scheduler
from utils.callbacks import (
    Callback,
    SigmaDataEstimator,
    SigmaGradNormCallback,
    EntropySchedulePlotCallback,
    OfflineEntropyProfileCallback,
    VLBBoundCallback,
    ExternalPPLCallback,
    MauveCallback,
    VisualizationCallback,
)


from utils.schedule_controller import EntropyScheduleController

from diffusion.continuous.logit_postprocess import _model_logits_continuous
from diffusion.continuous.processes import ContinuousForwardProcess
from diffusion.continuous.losses import binary_score_interpolation_loss, token_score_interpolation_loss
from diffusion.continuous.samplers import HeunSampler

from diffusion.discrete.processes import DiscreteForwardProcess
from diffusion.discrete.losses import dwdse_loss
from diffusion.discrete.samplers import TweedieTauLeapingSampler
from utils.ecc_secded import ecc_from_cfg, ecc_chunk_len

# Optional Weights & Biases
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional
    wandb = None
    WANDB_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Global numeric / matmul / attention settings
# ──────────────────────────────────────────────────────────────────────────────

# TF32 matmul on Ampere: good speedup with negligible impact for DL workloads.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    # Let PyTorch choose faster kernels for fp32 matmuls.
    torch.set_float32_matmul_precision("high")
except AttributeError:
    # Older PyTorch versions.
    pass

import torch._dynamo

torch._dynamo.config.cache_size_limit = 64  # or 128 if you want


def _enable_flash_sdp():
    """
    Enable PyTorch's flash / mem-efficient SDPA (scaled dot-product attention)
    backends. This does NOT change model architecture or checkpoints.
    """
    if not torch.cuda.is_available():
        print("[FlashSDP] CUDA not available; keeping default attention backends.")
        return

    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)  # keep math as a fallback
        # Only print on main process to avoid log spam (checked later)
    except AttributeError:
        print("[FlashSDP] SDPA backend toggles not found; PyTorch too old?")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ddp_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _atomic_torch_save(state, final_path) -> None:
    """Write a checkpoint to a temp sibling then atomically rename it into place.

    A preemption or crash mid-write then leaves the previous checkpoint intact
    rather than a truncated file, matching how last.pt is already written.
    """
    final_path = Path(final_path)
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, final_path)


def _mm_replay_task_weights(cfg) -> Dict[str, float]:
    """Task weights for the sequence-only replay loader.

    Replay rows carry a sequence but no structure (struct_mask all False), so a
    paired or structure-target task would leave the batch with no supervised bits.
    A sequence-producing mix keeps every replay step reinforcing the warm-started
    sequence model. Honours ``cfg.data.replay_task_weights`` when set, else uses a
    pure sequence-marginal draw.
    """
    override = getattr(cfg.data, "replay_task_weights", None)
    if override is not None:
        return {str(k): float(v) for k, v in dict(override).items()}
    return {"sequence_marginal": 1.0}


_MM_VAL_SIGMA_EDGES = (0.002, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 80.0)


def _ema_decay_for_step(start: float, end: float, ramp_steps: int, step: int) -> float:
    """Linear EMA-decay ramp, clamped at both endpoints."""
    if ramp_steps <= 0:
        return float(end)
    frac = min(max(float(step) / float(ramp_steps), 0.0), 1.0)
    return float(start + frac * (end - start))


def _mm_sigma_bin_label(lo: float, hi: float) -> str:
    def _part(value: float) -> str:
        return f"{value:g}".replace(".", "p")
    return f"{_part(lo)}_{_part(hi)}"


def _mm_validation_accumulator(device, tasks) -> Dict[str, torch.Tensor]:
    prefixes = [f"validation/modality/{m}" for m in ("seq", "struct")]
    prefixes += [
        f"validation/task/{task}/{modality}"
        for task in sorted(tasks)
        for modality in ("seq", "struct")
    ]
    prefixes += [
        f"validation/sigma/{modality}/{_mm_sigma_bin_label(lo, hi)}"
        for modality in ("seq", "struct")
        for lo, hi in zip(_MM_VAL_SIGMA_EDGES[:-1], _MM_VAL_SIGMA_EDGES[1:])
    ]
    # [EDM-weighted error sum, unweighted MSE sum, correct-bit sum, bit count]
    return {p: torch.zeros(4, device=device, dtype=torch.float64) for p in prefixes}


def _mm_accumulate_validation(
    stats: Dict[str, torch.Tensor], diagnostics: Dict[str, object], task_names
) -> None:
    device = next(iter(stats.values())).device
    task_names = list(task_names)

    def _add(prefix: str, modality: str, select: torch.Tensor) -> None:
        count = diagnostics[f"{modality}_count"].to(device=device, dtype=torch.float64)
        select = select.to(device=device, dtype=torch.bool) & (count > 0)
        if not bool(select.any()):
            return
        stats[prefix][0] += diagnostics[f"{modality}_edm_sum"].to(device, torch.float64)[select].sum()
        stats[prefix][1] += diagnostics[f"{modality}_mse_sum"].to(device, torch.float64)[select].sum()
        stats[prefix][2] += diagnostics[f"{modality}_correct_sum"].to(device, torch.float64)[select].sum()
        stats[prefix][3] += count[select].sum()

    batch_size = len(task_names)
    all_rows = torch.ones(batch_size, device=device, dtype=torch.bool)
    for modality in ("seq", "struct"):
        _add(f"validation/modality/{modality}", modality, all_rows)

    for task in sorted(set(task_names)):
        rows = torch.tensor([name == task for name in task_names], device=device)
        for modality in ("seq", "struct"):
            prefix = f"validation/task/{task}/{modality}"
            if prefix in stats:
                _add(prefix, modality, rows)

    for modality in ("seq", "struct"):
        sigma = diagnostics[f"sigma_{modality}"].to(device=device)
        for index, (lo, hi) in enumerate(zip(_MM_VAL_SIGMA_EDGES[:-1], _MM_VAL_SIGMA_EDGES[1:])):
            rows = (sigma >= lo) & (sigma <= hi if index == len(_MM_VAL_SIGMA_EDGES) - 2 else sigma < hi)
            prefix = f"validation/sigma/{modality}/{_mm_sigma_bin_label(lo, hi)}"
            _add(prefix, modality, rows)


def _mm_finalize_validation(stats: Dict[str, torch.Tensor]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for prefix, values in stats.items():
        count = float(values[3].item())
        if count <= 0.0:
            continue
        metrics[f"{prefix}/edm_loss"] = float(values[0].item() / count)
        metrics[f"{prefix}/mse"] = float(values[1].item() / count)
        metrics[f"{prefix}/bit_acc"] = float(values[2].item() / count)
        metrics[f"{prefix}/target_bits"] = count

    seq = metrics.get("validation/modality/seq/mse")
    struct = metrics.get("validation/modality/struct/mse")
    if seq is not None and struct is not None:
        metrics["validation/balanced_mse"] = 0.5 * (seq + struct)
    return metrics


def _maybe_set_seed(cfg):
    """Set Python/NumPy/Torch seeds for reproducibility if cfg.train.seed is provided."""
    seed = getattr(cfg.train, "seed", None)
    if seed is None:
        return

    # In DDP, we generally want the same seed for model initialization (so weights match),
    # but potentially different seeds for data sampling if not using DistributedSampler.
    # DistributedSampler handles the shuffling offset automatically using epoch+rank.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Allow turning determinism off for speed.
    # Default = True to preserve existing behaviour.
    deterministic = bool(getattr(cfg.train, "deterministic", True))
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def _human_bytes(n_params: int, dtype: torch.dtype = torch.float32) -> str:
    """Approximate parameter memory footprint given dtype."""
    bytes_per = {
        torch.float64: 8,
        torch.float32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.int64: 8,
        torch.int32: 4,
        torch.int16: 2,
        torch.int8: 1,
        torch.bool: 1,
    }.get(dtype, 4)
    total = n_params * bytes_per
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if total < 1024:
            return f"{total:.1f}{unit}"
        total /= 1024.0
    return f"{total:.1f}PB"


def _count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def _fmt_num(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)


def _cfg_to_dict(obj) -> Any:
    """
    Recursively convert ml_collections.ConfigDict / nested containers into
    plain JSON-serializable Python objects.

    Handles:
      - ConfigDict / objects with .to_dict()
      - dict
      - list / tuple
      - pathlib.Path
      - torch.device / torch.dtype
      - numpy scalars / arrays
      - torch tensors (best-effort: convert to Python scalars/lists)
    """
    # ConfigDict-like
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _cfg_to_dict(obj.to_dict())

    # Plain dict
    if isinstance(obj, dict):
        return {str(k): _cfg_to_dict(v) for k, v in obj.items()}

    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [_cfg_to_dict(v) for v in obj]

    # pathlib.Path
    if isinstance(obj, Path):
        return str(obj)

    # torch device / dtype
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)

    # numpy
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # torch tensors
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()

    # JSON-safe primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Fallback for simple objects with __dict__
    if hasattr(obj, "__dict__"):
        return {
            str(k): _cfg_to_dict(v)
            for k, v in obj.__dict__.items()
            if not str(k).startswith("_")
        }

    # Last resort: stringify
    return str(obj)


def _save_config_to_run_dir(cfg, run_dir: Path):
    """
    Save the effective config of this run into:
      - config.json        : machine-readable fully materialized config
      - original_config.py : copy of the Python config used (first run)
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = _cfg_to_dict(cfg)

    cfg_json_path = run_dir / "config.json"
    with open(cfg_json_path, "w") as f:
        json.dump(cfg_dict, f, indent=2, sort_keys=True)

    config_path = getattr(cfg, "_config_path", None)
    if config_path is not None and os.path.isfile(config_path):
        dst = run_dir / "original_config.py"
        if not dst.exists():
            shutil.copy2(config_path, dst)


class _NullWriter:
    """Drop-in TB writer that does nothing (safe for non-master ranks)."""

    def add_scalar(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_figure(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


def _unwrap_all(model: torch.nn.Module) -> torch.nn.Module:
    """Peel DDP and torch.compile wrappers until reaching the real module."""
    m = model
    while True:
        changed = False
        if hasattr(m, "module"):  # DDP wrapper usually outermost
            m = m.module
            changed = True
        if hasattr(m, "_orig_mod"):  # torch.compile wrapper
            m = m._orig_mod
            changed = True
        if not changed:
            break
    return m

def _gpu_mem_msg(device):
    if not torch.cuda.is_available():
        return "cpu"
    torch.cuda.synchronize(device)
    a = torch.cuda.memory_allocated(device) / (1024 ** 3)
    r = torch.cuda.memory_reserved(device) / (1024 ** 3)
    return f"alloc={a:.2f}GiB reserved={r:.2f}GiB"

# -----------------------------------------------------------------------------
# Conditioning helpers (shared with training)
# -----------------------------------------------------------------------------
def _bits_per_unit(cfg) -> int:
    data = getattr(cfg, "data", object())

    ecc_cfg = getattr(data, "ecc", None)
    if ecc_cfg is not None and bool(getattr(ecc_cfg, "enabled", False)):
        ecc = ecc_from_cfg(cfg)
        return int(ecc_chunk_len(ecc))  # e.g. 21

    bpt = getattr(data, "bits_per_token", None)
    if bpt is not None:
        return int(bpt)
    return int(getattr(data, "bits_per_char", 1))


def _cond_len_bits_fixed(cfg, seq_len_bits: int) -> int:
    """
    Fixed prefix length in bits (backward compatible).
    Prefers cfg.cond.cond_len_tokens (semantic/BPE), else cfg.cond.cond_len_chars.
    """
    cond_cfg = getattr(cfg, "cond", None)
    if cond_cfg is None or not bool(getattr(cond_cfg, "enabled", False)):
        return 0

    bits_per = _bits_per_unit(cfg)

    n_units = getattr(cond_cfg, "cond_len_tokens", None)
    if n_units is None:
        n_units = int(getattr(cond_cfg, "cond_len_chars", 0))
    else:
        n_units = int(n_units)

    cL = int(n_units * bits_per)
    return max(0, min(int(cL), int(seq_len_bits)))


def _sample_cond_len_bits_per_example(cfg, B: int, seq_len_bits: int, device) -> torch.Tensor:
    """
    Returns cL_bits per example: [B] int64, in [0, seq_len_bits].
    If cfg.cond.sample_prompt_len=False (default), returns fixed length repeated.
    """
    cond_cfg = getattr(cfg, "cond", None)
    if cond_cfg is None or not bool(getattr(cond_cfg, "enabled", False)):
        return torch.zeros(B, device=device, dtype=torch.long)

    sample_len = bool(getattr(cond_cfg, "sample_prompt_len", False))
    if not sample_len:
        cL = _cond_len_bits_fixed(cfg, seq_len_bits)
        return torch.full((B,), int(cL), device=device, dtype=torch.long)

    bits_per = _bits_per_unit(cfg)

    # min/max in units (tokens or chars)
    mn = getattr(cond_cfg, "cond_len_tokens_min", None)
    mx = getattr(cond_cfg, "cond_len_tokens_max", None)
    if mn is None or mx is None:
        # fallback to legacy names
        mn = int(getattr(cond_cfg, "cond_len_chars_min", 0))
        mx = int(getattr(cond_cfg, "cond_len_chars_max", 0))
    else:
        mn = int(mn)
        mx = int(mx)

    mn = max(0, mn)
    mx = max(mn, mx)

    # uniform integer in [mn, mx]
    if mx == mn:
        units = torch.full((B,), mn, device=device, dtype=torch.long)
    else:
        units = torch.randint(low=mn, high=mx + 1, size=(B,), device=device, dtype=torch.long)

    cL_bits = units * int(bits_per)
    cL_bits = torch.clamp(cL_bits, min=0, max=int(seq_len_bits)).to(torch.long)
    return cL_bits


def _make_prefix_mask_from_lengths(cL_bits: torch.Tensor, S: int) -> torch.Tensor:
    """
    cL_bits: [B] long
    returns prefix_mask: [B,S] bool where True indicates "prefix / conditioned" positions.
    """
    B = int(cL_bits.numel())
    ar = torch.arange(S, device=cL_bits.device).view(1, S).expand(B, S)
    return ar < cL_bits.view(B, 1)


def _make_null_value(cfg, device, dtype, *, is_tokens: bool = False, vocab_size: int | None = None) -> torch.Tensor:
    cond_cfg = getattr(cfg, "cond", None)
    strategy = str(getattr(cond_cfg, "null_strategy", "half")) if cond_cfg is not None else "half"

    if is_tokens:
        if strategy in {"half", "data_center"}:
            if vocab_size is None:
                raise ValueError("vocab_size required for token null value")
            dc = float(getattr(cfg.diffusion.continuous, "data_center", 1.0 / vocab_size))
            return torch.tensor(dc, device=device, dtype=dtype)
        if strategy == "zeros":
            return torch.tensor(0.0, device=device, dtype=dtype)
        if strategy == "random":
            return torch.tensor(float("nan"), device=device, dtype=dtype)
        raise ValueError(f"Unknown cfg.cond.null_strategy={strategy}")

    if strategy == "half":
        return torch.tensor(0.5, device=device, dtype=dtype)
    if strategy == "data_center":
        return torch.tensor(float(getattr(cfg.diffusion.continuous, "data_center", 0.5)), device=device, dtype=dtype)
    if strategy == "zeros":
        return torch.tensor(0.0, device=device, dtype=dtype)
    if strategy == "random":
        return torch.tensor(float("nan"), device=device, dtype=dtype)
    raise ValueError(f"Unknown cfg.cond.null_strategy={strategy}")

def _make_null_prefix_full(x0_full: torch.Tensor, prefix_mask: torch.Tensor, cfg) -> torch.Tensor:
    """
    x0_full:
      - binary mode: [B,S]
      - token mode:  [B,S,V]
    prefix_mask: [B,S] bool
    """
    is_tokens = (x0_full.dim() == 3)
    b = x0_full.size(0)
    s = x0_full.size(1)

    cond_cfg = getattr(cfg, "cond", None)
    strategy = str(getattr(cond_cfg, "null_strategy", "half")) if cond_cfg is not None else "half"

    out = x0_full.clone()
    if not prefix_mask.any():
        return out

    if is_tokens:
        v = x0_full.size(-1)
        pm = prefix_mask.unsqueeze(-1).expand_as(x0_full)

        if strategy == "random":
            rnd = torch.full((b, s, v), 1.0 / v, device=x0_full.device, dtype=x0_full.dtype)
            out[pm] = rnd[pm]
            return out

        null_val = _make_null_value(cfg, x0_full.device, x0_full.dtype, is_tokens=True, vocab_size=v)
        out[pm] = null_val
        return out

    if strategy == "random":
        rnd = torch.bernoulli(torch.full((b, s), 0.5, device=x0_full.device, dtype=x0_full.dtype))
        out[prefix_mask] = rnd[prefix_mask]
        return out

    null_val = _make_null_value(cfg, x0_full.device, x0_full.dtype, is_tokens=False)
    out[prefix_mask] = null_val
    return out

def _discrete_positions_per_token(cfg) -> int:
    """
    Number of model positions corresponding to one semantic/BPE token
    in the discrete branch.

    - binary bitstream discrete: one token = bits_per_token model positions
    - token discrete: one token = one model position
    """
    repr_mode = str(getattr(cfg.data, "representation", "tokens")).lower()
    if repr_mode == "binary":
        return _bits_per_unit(cfg)
    return 1


def _sample_cond_len_positions_per_example_continuous(cfg, B: int, seq_len_positions: int, device) -> torch.Tensor:
    cond_cfg = getattr(cfg, "cond", None)
    if cond_cfg is None or not bool(getattr(cond_cfg, "enabled", False)):
        return torch.zeros(B, device=device, dtype=torch.long)

    sample_len = bool(getattr(cond_cfg, "sample_prompt_len", False))
    repr_mode = str(getattr(getattr(cfg, "data", object()), "representation", "binary")).lower()

    if repr_mode == "tokens":
        pos_per_unit = 1
    else:
        pos_per_unit = _bits_per_unit(cfg)

    if not sample_len:
        n_units = getattr(cond_cfg, "cond_len_tokens", None)
        if n_units is None:
            n_units = int(getattr(cond_cfg, "cond_len_chars", 0))
        else:
            n_units = int(n_units)

        cL = max(0, min(int(n_units * pos_per_unit), int(seq_len_positions)))
        return torch.full((B,), cL, device=device, dtype=torch.long)

    mn = getattr(cond_cfg, "cond_len_tokens_min", None)
    mx = getattr(cond_cfg, "cond_len_tokens_max", None)
    if mn is None or mx is None:
        mn = int(getattr(cond_cfg, "cond_len_chars_min", 0))
        mx = int(getattr(cond_cfg, "cond_len_chars_max", 0))
    else:
        mn = int(mn)
        mx = int(mx)

    mn = max(0, mn)
    mx = max(mn, mx)

    if mx == mn:
        units = torch.full((B,), mn, device=device, dtype=torch.long)
    else:
        units = torch.randint(low=mn, high=mx + 1, size=(B,), device=device, dtype=torch.long)

    cL = units * int(pos_per_unit)
    return torch.clamp(cL, min=0, max=int(seq_len_positions)).to(torch.long)

def _sample_cond_len_positions_per_example_discrete(cfg, B: int, seq_len_positions: int, device) -> torch.Tensor:
    cond_cfg = getattr(cfg, "cond", None)
    if cond_cfg is None or not bool(getattr(cond_cfg, "enabled", False)):
        return torch.zeros(B, device=device, dtype=torch.long)

    sample_len = bool(getattr(cond_cfg, "sample_prompt_len", False))
    if not sample_len:
        n_units = getattr(cond_cfg, "cond_len_tokens", None)
        if n_units is None:
            n_units = int(getattr(cond_cfg, "cond_len_chars", 0))
        else:
            n_units = int(n_units)
        pos_per_tok = _discrete_positions_per_token(cfg)
        cL = max(0, min(int(n_units * pos_per_tok), int(seq_len_positions)))
        return torch.full((B,), cL, device=device, dtype=torch.long)

    mn = getattr(cond_cfg, "cond_len_tokens_min", None)
    mx = getattr(cond_cfg, "cond_len_tokens_max", None)
    if mn is None or mx is None:
        mn = int(getattr(cond_cfg, "cond_len_chars_min", 0))
        mx = int(getattr(cond_cfg, "cond_len_chars_max", 0))
    else:
        mn = int(mn)
        mx = int(mx)

    mn = max(0, mn)
    mx = max(mn, mx)

    if mx == mn:
        units = torch.full((B,), mn, device=device, dtype=torch.long)
    else:
        units = torch.randint(low=mn, high=mx + 1, size=(B,), device=device, dtype=torch.long)

    pos_per_tok = _discrete_positions_per_token(cfg)
    cL = units * int(pos_per_tok)
    return torch.clamp(cL, min=0, max=int(seq_len_positions)).to(torch.long)

def _dataloader_kwargs(cfg) -> dict:
    """
    DataLoader kwargs with version-safe handling of prefetch_factor and persistent_workers.

    Key rule: prefetch_factor is only valid when num_workers > 0.
    """
    nw = int(getattr(cfg.data, "num_workers", 0))
    pm = bool(getattr(cfg.data, "pin_memory", True))
    pf = int(getattr(cfg.data, "prefetch_factor", 2))

    kw = dict(
        num_workers=nw,
        pin_memory=pm,
    )
    if nw > 0:
        kw["prefetch_factor"] = pf
        kw["persistent_workers"] = True
    return kw

def _resolve_checkpointing_cfg(cfg) -> dict:
    """
    Backward-compatible checkpointing config resolver.

    Priority:
      1) cfg.train.checkpointing.* (new)
      2) legacy cfg.train.save_last / save_top_k / checkpoint_mode
      3) legacy cfg.train.ckpt_interval.* (if present from earlier experiments)
      4) defaults

    New concept:
      - interval: sparse archival checkpoints (step=....pt)
      - resume_interval: frequent rolling checkpoint (last.pt only)
    """
    train = getattr(cfg, "train", None)

    # --- defaults ---
    save_last = True
    save_top_k = 1
    mode = "min"

    # archival interval checkpoints
    interval_enabled = False
    interval_every_steps = 100_000
    interval_keep_last = 5  # None/0 means keep all

    # rolling resume checkpoint
    resume_interval_enabled = False
    resume_interval_every_steps = 5_000

    # --- legacy: cfg.train.* ---
    if train is not None:
        if hasattr(train, "save_last"):
            save_last = bool(train.save_last)
        if hasattr(train, "save_top_k"):
            save_top_k = int(train.save_top_k)
        if hasattr(train, "checkpoint_mode"):
            mode = str(train.checkpoint_mode)

        # legacy interval block
        if hasattr(train, "ckpt_interval"):
            ci = train.ckpt_interval
            interval_enabled = bool(getattr(ci, "enabled", interval_enabled))
            interval_every_steps = int(getattr(ci, "every_steps", interval_every_steps))
            interval_keep_last = getattr(ci, "keep_last", interval_keep_last)

    # --- new structured config overrides legacy ---
    if train is not None and hasattr(train, "checkpointing"):
        ck = train.checkpointing
        save_last = bool(getattr(ck, "save_last", save_last))
        save_top_k = int(getattr(ck, "save_top_k", save_top_k))
        mode = str(getattr(ck, "mode", mode))

        if hasattr(ck, "interval"):
            ci = ck.interval
            interval_enabled = bool(getattr(ci, "enabled", interval_enabled))
            interval_every_steps = int(getattr(ci, "every_steps", interval_every_steps))
            interval_keep_last = getattr(ci, "keep_last", interval_keep_last)

        if hasattr(ck, "resume_interval"):
            ri = ck.resume_interval
            resume_interval_enabled = bool(getattr(ri, "enabled", resume_interval_enabled))
            resume_interval_every_steps = int(getattr(ri, "every_steps", resume_interval_every_steps))

    if mode not in {"min", "max"}:
        raise ValueError(f"checkpointing.mode must be 'min' or 'max', got {mode}")

    # Interpret keep_last:
    #   None or 0 => keep all
    #   positive int => keep last N
    #   negative => treat as keep all
    if interval_keep_last is None:
        keep_last_norm = None
    else:
        try:
            keep_last_norm = int(interval_keep_last)
        except Exception:
            keep_last_norm = None
        if keep_last_norm is not None and keep_last_norm <= 0:
            keep_last_norm = None

    return {
        "save_last": save_last,
        "save_top_k": save_top_k,
        "mode": mode,

        "interval_enabled": interval_enabled,
        "interval_every_steps": interval_every_steps,
        "interval_keep_last": keep_last_norm,

        "resume_interval_enabled": resume_interval_enabled,
        "resume_interval_every_steps": resume_interval_every_steps,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        # ── Distributed Setup ────────────────────────────────────────────────
        # Distinguish "requested" vs "active" to avoid crashes/hangs when config
        # and launcher disagree.
        self.ddp_requested = bool(getattr(cfg.system, "distributed", False))
        self.ddp_active = bool(self.ddp_requested and _ddp_is_on())

        if self.ddp_active:
            self.rank = int(cfg.system.global_rank)
            self.local_rank = int(cfg.system.local_rank)
            self.world_size = int(cfg.system.world_size)
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            # Keep the old attribute name for minimal disruption / compatibility.
        self.is_distributed = self.ddp_active
        self.is_master = (self.rank == 0)

        # ── Device selection ─────────────────────────────────────────────────────
        # DDP: bind each process to cuda:<local_rank>
        # Single GPU / non-DDP: respect cfg.device (e.g. "cuda:1")
        if torch.cuda.is_available():
            if self.ddp_active:
                dev = torch.device(f"cuda:{self.local_rank}")
            else:
                dev_str = str(getattr(cfg, "device", "cuda:0"))
                # allow "cuda" as shorthand
                dev = torch.device("cuda:0" if dev_str == "cuda" else dev_str)

            if dev.type != "cuda":
                raise ValueError(f"cfg.device must be a CUDA device when CUDA is available, got: {dev}")

            torch.cuda.set_device(dev.index)
            self.device = dev
            self.cfg.device = str(dev)
        else:
            self.device = torch.device("cpu")
            self.cfg.device = "cpu"

        # Enable fast SDPA backends (flash/mem-efficient) on GPU.
        _enable_flash_sdp()
        if self.is_master:
            flash_avail = "Unknown (PyTorch < 2.3)"
            if torch.cuda.is_available() and hasattr(torch.backends.cuda, "is_flash_attention_available"):
                flash_avail = torch.backends.cuda.is_flash_attention_available()
            print(f"[FlashSDP] Enabled flash & mem-efficient attention backends. flash_available={flash_avail}")

        _maybe_set_seed(cfg)

        # ── dirs / writer (HPC-safe TB manager) ─────────────────────────────
        self.run_dir = Path("runs") / cfg.experiment
        self.ckpt_dir = self.run_dir / "checkpoints"

        if self.is_master:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            _save_config_to_run_dir(cfg, self.run_dir)

        # DDP filesystem sync (robust on network FS)
        if self.ddp_active:
            dist.barrier()

        # TensorBoard manager + writer
        self.tb = None
        self.writer = _NullWriter()

        tb_cfg = getattr(getattr(cfg, "logging", None), "tensorboard", None)
        tb_enabled = True if tb_cfg is None else bool(getattr(tb_cfg, "enabled", True))

        if self.is_master and tb_enabled:
            self.tb = TBManager(cfg=cfg, run_dir=self.run_dir, subdir="training_logs", is_master=True)

        if tb_cfg is None:
            self.tb_scalar_every_steps = 20
            self.tb_sync_every_steps = 0
            self.tb_sync_every_epochs = 1
        else:
            self.tb_scalar_every_steps = int(getattr(tb_cfg, "scalar_every_steps", 20))
            self.tb_sync_every_steps = int(getattr(tb_cfg, "sync_every_steps", 0))
            self.tb_sync_every_epochs = int(getattr(tb_cfg, "sync_every_epochs", 1))

        # ── W&B init (optional) ─────────────────────────────────────────────
        logging_cfg = getattr(cfg, "logging", None)
        self.use_wandb: bool = False
        # When True, every TensorBoard write (scalars, figures, images,
        # histograms, text) from the trainer AND all callbacks is mirrored to
        # W&B automatically, so a run is fully readable in W&B without a
        # per-metric _log_wandb call. In that mode the explicit _log_wandb calls
        # stand down (see _log_wandb) to avoid duplicate, step-conflicting series.
        self._wandb_sync_tb: bool = False
        # Running (EMA) smoothing of the noisy per-step train loss for logging.
        self._loss_ema: Optional[float] = None

        # WANDB_MODE=disabled/dryrun is a hard off, honoured regardless of config.
        _wandb_hard_off = os.environ.get("WANDB_MODE", "").lower() in (
            "disabled",
            "dryrun",
        )
        if (
            self.is_master
            and logging_cfg is not None
            and getattr(logging_cfg, "use_wandb", False)
            and not _wandb_hard_off
        ):
            if WANDB_AVAILABLE:
                self.use_wandb = True
                os.environ["WANDB_PYTORCH_DISABLE"] = "true"
                os.environ["WANDB_DISABLE_GRADIENTS"] = "true"

                cfg_dict = _cfg_to_dict(cfg)
                # Environment variables win over config so a run can be redirected
                # or switched to offline without editing the config.
                project = os.environ.get("WANDB_PROJECT") or getattr(logging_cfg, "project", "diffusion")
                entity = os.environ.get("WANDB_ENTITY") or getattr(logging_cfg, "entity", None)
                mode = os.environ.get("WANDB_MODE") or getattr(logging_cfg, "mode", "online")
                group = getattr(logging_cfg, "group", None)
                tags = getattr(logging_cfg, "tags", None)
                # W&B display name defaults to the experiment (= run dir), but can
                # be set independently via cfg.logging.run_name or WANDB_NAME.
                run_name = (
                    os.environ.get("WANDB_NAME")
                    or getattr(logging_cfg, "run_name", None)
                    or cfg.experiment
                )
                # Default on: mirror the full TensorBoard stream into W&B. The
                # SafeSummaryWriter is created after this init (in
                # prepare_for_run), so wandb's writer patch is active in time.
                self._wandb_sync_tb = bool(
                    getattr(logging_cfg, "sync_tensorboard", True)
                )

                wandb.init(
                    project=project,
                    entity=entity,
                    id=getattr(logging_cfg, "run_id", None),
                    resume="allow",
                    name=run_name,
                    group=group,
                    tags=list(tags) if tags else None,
                    config=cfg_dict,
                    dir=str(self.run_dir),
                    mode=mode,
                    sync_tensorboard=self._wandb_sync_tb,
                )
                if wandb.run is not None:
                    print(f"W&B run: {wandb.run.get_url()}")
                    if self._wandb_sync_tb:
                        print(
                            "W&B: mirroring all TensorBoard metrics/figures/images "
                            "into this run (sync_tensorboard=True)."
                        )
                    # Persist the run identity so a later eval process can resume
                    # this exact run and attach eval metrics to it (see
                    # utils/wandb_eval.log_eval_metrics).
                    try:
                        (self.run_dir / "wandb_run.json").write_text(
                            json.dumps(
                                {
                                    "id": wandb.run.id,
                                    "entity": wandb.run.entity,
                                    "project": wandb.run.project,
                                    "name": wandb.run.name,
                                    "url": wandb.run.get_url(),
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    except OSError:
                        pass
            else:
                print("⚠️  wandb not installed, skipping W&B logging.")

        # ── checkpoint config / bookkeeping (resolved + backward-compatible) ─────
        ck = _resolve_checkpointing_cfg(cfg)

        self.save_last = bool(ck["save_last"])
        self.save_top_k = int(ck["save_top_k"])
        self.checkpoint_mode = str(ck["mode"])

        self.best_metric = math.inf if self.checkpoint_mode == "min" else -math.inf
        self.best_ckpts: List[dict] = []
        self.early_stop_best = self.best_metric
        self.early_stop_bad_epochs = 0

        # periodic interval checkpoints (post-training analysis)
        self.ckpt_interval_enabled = bool(ck["interval_enabled"])
        self.ckpt_interval_every_steps = int(ck["interval_every_steps"])
        self.ckpt_interval_keep_last = ck["interval_keep_last"]  # None => keep all
        self.resume_interval_enabled = bool(ck["resume_interval_enabled"])
        self.resume_interval_every_steps = int(ck["resume_interval_every_steps"])

        # interval state (final threshold set after resume)
        self._next_interval_ckpt_step = None
        self._next_resume_ckpt_step = None
        self._interval_ckpt_paths: List[str] = []


        # ── data ────────────────────────────────────────────────────────────
        dl_kw = _dataloader_kwargs(cfg)

        dataset_name = str(getattr(cfg.data, "dataset", "")).lower()
        is_dima_length_bucketed = dataset_name in {
            "swissprotdima",
            "swissprot_dima",
            "dima_swissprot",
        }
        is_evodiff_length_bucketed = dataset_name in {
            "evodiffuniref50",
            "evodiff_uniref50",
            "uniref50_evodiff",
        }
        is_multimodal = dataset_name in {
            "proteinmultimodallfq",
            "protein_multimodal",
            "dplm_paired",
        }
        # Multimodal state (kept harmless / off for every non-multimodal path).
        self.is_multimodal = bool(is_multimodal)
        self.replay_fraction = 0.0
        self.replay_loader = None
        self._replay_iter = None
        self._last_mm_components = None
        self._last_mm_val_diagnostics = None

        if not (
            is_dima_length_bucketed or is_evodiff_length_bucketed or is_multimodal
        ):
            raw_train_loader, raw_val_loader, _ = get_dataloaders(cfg)

        if is_multimodal:
            # Keep the paired multimodal loaders exactly as built: the custom
            # source/length batch sampler and the task-aware collator (which
            # yields dict batches with x0/states/target masks) must survive, so
            # never rebuild a plain DataLoader from `.dataset` here.
            from data.protein_multimodal import get_multimodal_loader

            if self.ddp_active:
                assert cfg.train.batch_size % self.world_size == 0, (
                    f"Global batch_size ({cfg.train.batch_size}) must be divisible by "
                    f"world_size ({self.world_size})."
                )
                local_batch_size = cfg.train.batch_size // self.world_size
            else:
                local_batch_size = int(cfg.train.batch_size)
            mm_seed = int(getattr(cfg.train, "seed", 42))
            self.train_loader = get_multimodal_loader(
                cfg, split="train", batch_size=local_batch_size, shuffle=True, seed=mm_seed
            )
            self.val_loader = get_multimodal_loader(
                cfg, split="val", batch_size=local_batch_size, shuffle=False, seed=mm_seed
            )
            # Optional token-budgeted sequence-only UniRef50 replay (plan 6.6 step 3).
            # A separate paired-format shard set of sequence-only rows (struct_mask
            # all False) mixed by `sequence_replay_fraction`; batches are the same
            # dict format so the multimodal step consumes them transparently.
            self.replay_fraction = float(
                getattr(cfg.data, "sequence_replay_fraction", 0.0) or 0.0
            )
            replay_root = str(getattr(cfg.data, "sequence_replay_root", "") or "")
            replay_ready = bool(replay_root) and (
                Path(replay_root) / "manifest.json"
            ).exists()
            if self.replay_fraction > 0.0 and replay_ready:
                # Build the replay loader over the sequence-only corpus. The corpus
                # has a single source ("uniref50") that is absent from the paired
                # source_weights, so pass source_weights=None (equal weighting over
                # present sources) to avoid a zero-total-weight sampler error. Force
                # a sequence-producing task mix so every replay example supervises
                # its sequence bits (structure is ABSENT on these rows, so paired or
                # structure-target tasks would waste the replay batch).
                orig_shard = cfg.data.shard_dir
                orig_src_w = getattr(cfg.data, "source_weights", None)
                orig_task_w = getattr(cfg.data, "task_weights", None)
                cfg.data.shard_dir = replay_root
                cfg.data.source_weights = None
                cfg.data.task_weights = _mm_replay_task_weights(cfg)
                try:
                    self.replay_loader = get_multimodal_loader(
                        cfg,
                        split="train",
                        batch_size=local_batch_size,
                        shuffle=True,
                        seed=mm_seed + 101,
                    )
                finally:
                    cfg.data.shard_dir = orig_shard
                    cfg.data.source_weights = orig_src_w
                    cfg.data.task_weights = orig_task_w
            elif self.replay_fraction > 0.0 and self.is_master:
                reason = (
                    "no cfg.data.sequence_replay_root set"
                    if not replay_root
                    else f"no manifest.json under sequence_replay_root {replay_root!r} "
                    "(build it with scripts/proteins/setup/prepare_uniref50_replay.py)"
                )
                print(
                    f"[multimodal] sequence_replay_fraction>0 but {reason}; "
                    "replay disabled."
                )
                self.replay_fraction = 0.0
            elif self.replay_fraction > 0.0:
                # Non-master ranks must agree on the disabled state to stay in sync.
                self.replay_fraction = 0.0
        elif is_dima_length_bucketed or is_evodiff_length_bucketed:
            if self.ddp_active:
                assert cfg.train.batch_size % self.world_size == 0, (
                    f"Global batch_size ({cfg.train.batch_size}) must be divisible by "
                    f"world_size ({self.world_size})."
                )
                local_batch_size = cfg.train.batch_size // self.world_size
            else:
                local_batch_size = int(cfg.train.batch_size)
            protocol_loader = (
                get_evodiff_uniref50_loader
                if is_evodiff_length_bucketed
                else get_dima_loader
            )
            self.train_loader = protocol_loader(
                cfg,
                split="train",
                batch_size=local_batch_size,
                shuffle=True,
                seed=int(getattr(cfg.train, "seed", 42)),
            )
            self.val_loader = protocol_loader(
                cfg,
                split="val",
                batch_size=local_batch_size,
                shuffle=False,
                seed=int(getattr(cfg.train, "seed", 42)),
            )
        elif self.ddp_active:
            assert cfg.train.batch_size % self.world_size == 0, (
                f"Global batch_size ({cfg.train.batch_size}) must be divisible by world_size "
                f"({self.world_size}) for fixed-shape DDP training."
            )
            batch_size_per_gpu = cfg.train.batch_size // self.world_size

            train_sampler = DistributedSampler(
                raw_train_loader.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            self.train_loader = torch.utils.data.DataLoader(
                raw_train_loader.dataset,
                batch_size=batch_size_per_gpu,
                sampler=train_sampler,
                shuffle=False,
                drop_last=True,
                **dl_kw,
            )

            val_sampler = DistributedSampler(
                raw_val_loader.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=True,
            )
            self.val_loader = torch.utils.data.DataLoader(
                raw_val_loader.dataset,
                batch_size=batch_size_per_gpu,
                sampler=val_sampler,
                shuffle=False,
                drop_last=True,
                **dl_kw,
            )
        else:
            self.train_loader = torch.utils.data.DataLoader(
                raw_train_loader.dataset,
                batch_size=cfg.train.batch_size,
                shuffle=True,
                drop_last=True,
                **dl_kw,
            )
            self.val_loader = torch.utils.data.DataLoader(
                raw_val_loader.dataset,
                batch_size=cfg.train.batch_size,
                shuffle=False,
                drop_last=True,
                **dl_kw,
            )

        # ── model ───────────────────────────────────────────────────────────
        base = create_model(cfg).to(self.device)

        # IMPORTANT:
        # Precompute Flex block masks BEFORE DDP wrapping / torch.compile so
        # create_block_mask(...) is never reached inside compiled forward.
        raw_base_for_prep = _unwrap_all(base)
        if hasattr(raw_base_for_prep, "prepare_flex_masks"):
            if self.is_master:
                print("[flex] precomputing block masks before torch.compile ...")
            raw_base_for_prep.prepare_flex_masks(self.device)

        if self.ddp_active:
            try:
                base = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base)
            except Exception:
                pass
            # The multimodal warm-start curriculum freezes the trunk for the first
            # N steps (requires_grad=False) AFTER this DDP wrap, so those params
            # stop producing gradients mid-run. DDP's default reducer
            # (find_unused_parameters=False) would then error; enable unused-param
            # handling whenever a freeze window is configured.
            mm_freeze = self.is_multimodal and (
                int(getattr(cfg.model, "warm_start_freeze_trunk_steps", 0) or 0) > 0
            )
            base = DDP(
                base,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=bool(mm_freeze),
            )

        compile_enabled = bool(getattr(self.cfg.train, "use_compile", False))
        compile_mode = getattr(self.cfg.train, "compile_mode", "default")

        if compile_enabled and hasattr(torch, "compile"):
            try:
                if self.is_master:
                    print(f"[torch.compile] compiling model with mode={compile_mode!r}...")
                base = torch.compile(base, mode=compile_mode, fullgraph=False)
                if self.is_master:
                    print("[torch.compile] done.")
            except Exception as e:
                print(f"[torch.compile] WARNING: failed to compile model ({e}); using eager mode.")
        else:
            if not hasattr(torch, "compile") and self.is_master:
                print("[torch.compile] not available in this PyTorch; using eager mode.")

        self.model = base

        # ── gradient accumulation ──────────────────────────────────────────
        # Accumulate this many micro-batches into one optimizer step so the
        # effective global batch (in residue-patches) is held constant as the
        # node/GPU count changes. 1 == the previous single-step behavior.
        self.grad_accum_steps = max(
            1, int(getattr(cfg.train, "grad_accum_steps", 1) or 1)
        )
        self._accum_counter = 0
        self._did_optim_step = True

        # ── multimodal warm-start / curriculum knobs (gate 3) ──────────────
        self.freeze_trunk_steps = int(
            getattr(cfg.model, "warm_start_freeze_trunk_steps", 0) or 0
        )
        self.trunk_lr_mult = float(
            getattr(cfg.model, "warm_start_trunk_lr_mult", 1.0) or 1.0
        )
        self.warm_start_seq_checkpoint = str(
            getattr(cfg.model, "warm_start_seq_checkpoint", "") or ""
        )
        self._trunk_frozen = False
        # A dedicated lower-LR trunk group is only worth its resume complexity
        # when the multimodal path actually asks for a different trunk LR.
        self._trunk_group_active = bool(
            self.is_multimodal and abs(self.trunk_lr_mult - 1.0) > 1e-9
        )

        # ── opt / ema / amp ────────────────────────────────────────────────
        self.ema = EMA(self.model, decay=cfg.train.ema_decay)
        ema_ramp_cfg = getattr(cfg.train, "ema_ramp", None)
        self.ema_ramp_enabled = bool(
            getattr(ema_ramp_cfg, "enabled", False) if ema_ramp_cfg is not None else False
        )
        self.ema_ramp_start = float(
            getattr(ema_ramp_cfg, "start_decay", cfg.train.ema_decay)
            if ema_ramp_cfg is not None else cfg.train.ema_decay
        )
        self.ema_ramp_end = float(
            getattr(ema_ramp_cfg, "end_decay", cfg.train.ema_decay)
            if ema_ramp_cfg is not None else cfg.train.ema_decay
        )
        self.ema_ramp_steps = int(
            getattr(ema_ramp_cfg, "steps", 0) if ema_ramp_cfg is not None else 0
        )
        if self.ema_ramp_enabled:
            self.ema.decay = _ema_decay_for_step(
                self.ema_ramp_start, self.ema_ramp_end, self.ema_ramp_steps, 0
            )
        if self._trunk_group_active:
            # Group 0 = new multimodal adapters (base LR); group 1 = warm-started
            # trunk (base LR, scaled down each step by trunk_lr_mult). Built here
            # (before resume) so a resumed optimizer state matches the structure.
            self.opt, self.lr_sched = get_optimizer_and_scheduler(
                self.model, cfg, 0, params=self._mm_build_param_groups()
            )
        else:
            self.opt, self.lr_sched = get_optimizer_and_scheduler(self.model, cfg, 0)

        self.amp_enabled = bool(getattr(cfg.train, "use_fp16", False))
        amp_dtype_req = str(getattr(cfg.train, "amp_dtype", "auto")).lower()

        if not self.amp_enabled:
            self.amp_dtype = torch.float32
        else:
            if amp_dtype_req in {"bf16", "bfloat16"}:
                self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif amp_dtype_req in {"fp16", "float16"}:
                self.amp_dtype = torch.float16
            else:
                self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.use_scaler = (
            self.amp_enabled and (self.device.type == "cuda") and (self.amp_dtype == torch.float16)
        )
        self.scaler = GradScaler(enabled=self.use_scaler)
        self.grad_clip = float(getattr(cfg.optim, "grad_clip", 1.0))

        # ──────────────────────────────────────────────────────────────────
        # Entropy modes: none vs online(adaptive) vs offline
        # ──────────────────────────────────────────────────────────────────
        off = getattr(cfg.train, "entropy_offline", None)
        self.entropy_offline_enabled = bool(getattr(off, "enabled", False)) if off is not None else False

        legacy_entropy_online = bool(getattr(cfg.train, "entropy_online", False))
        entropy_compute_cfg = getattr(cfg.train, "entropy_compute", None)
        entropy_use_cfg = getattr(cfg.train, "entropy_use_for_sampling", None)

        if entropy_compute_cfg is None and entropy_use_cfg is None:
            self.entropy_compute = legacy_entropy_online
            self.entropy_use_for_sampling = legacy_entropy_online
        else:
            self.entropy_compute = (
                bool(entropy_compute_cfg) if entropy_compute_cfg is not None else legacy_entropy_online
            )
            self.entropy_use_for_sampling = (
                bool(entropy_use_cfg) if entropy_use_cfg is not None else legacy_entropy_online
            )

        if self.entropy_use_for_sampling and not self.entropy_compute and not self.entropy_offline_enabled:
            raise ValueError(
                "entropy_use_for_sampling=True but entropy_compute=False.\n"
                "Enable entropy_compute for online adaptive scheduling, or enable "
                "cfg.train.entropy_offline.enabled=True for offline scheduling."
            )

        # aliases
        self.entropy_online = self.entropy_use_for_sampling
        if self.entropy_offline_enabled:
            self.entropy_profile_source = "offline"
        elif self.entropy_compute:
            self.entropy_profile_source = "online"
        else:
            self.entropy_profile_source = "none"

        # hyperparams
        self.entropy_buffer_size = int(getattr(cfg.train, "entropy_buffer_size", 100_000))
        self.entropy_num_bins = int(getattr(cfg.train, "entropy_num_bins", 256))
        self.entropy_warmup_steps = int(getattr(cfg.train, "entropy_warmup_steps", 100_000))
        self.entropy_transition_steps = int(getattr(cfg.train, "entropy_transition_steps", 100_000))
        self.entropy_gamma_max = float(getattr(cfg.train, "entropy_gamma_max", 0.5))

        # ✅ Gabriel regularized schedule knobs
        self.entropy_mode = str(getattr(cfg.train, "entropy_mode", "regularized")).lower()
        self.entropy_regularizer_c = float(getattr(cfg.train, "entropy_regularizer_c", 0.1))
        self.entropy_regularizer_n = float(getattr(cfg.train, "entropy_regularizer_n", 3.0))

        # ✅ rate vs sqrt-rate via exponent p
        target = getattr(cfg.train, "entropy_target", "rate")
        if isinstance(target, str):
            t = target.lower()
            if t == "rate":
                self.entropy_rate_power = 1.0
            elif t in {"sqrt", "sqrt_rate", "sqrt-rate"}:
                self.entropy_rate_power = 0.5
            else:
                self.entropy_rate_power = float(t)  # allow "0.75" etc
        else:
            self.entropy_rate_power = float(target)

        self.entropy_target = target

        # ✅ FIFO ring buffer (CPU) — stable memory, no growth
        cap = self.entropy_buffer_size
        self._entropy_sig_buf = torch.empty(cap, dtype=torch.float32, device="cpu")
        self._entropy_metric_buf = torch.empty(cap, dtype=torch.float32, device="cpu")
        self._entropy_buf_ptr = 0
        self._entropy_buf_len = 0

        # entropy tables / state
        self._entropy_ready = False
        self._entropy_pdf = None
        self._entropy_cdf = None
        self._entropy_sigmas = None
        self._entropy_edges = None
        self._entropy_ln_mu = None
        self._entropy_ln_std = None

        self.callbacks: List[Callback] = []

        # ── framework-specific components ───────────────────────────────────
        if cfg.framework == "continuous_score":
            self.proc = ContinuousForwardProcess(cfg)
            self.entropy_ctrl = EntropyScheduleController(self, self.proc)
            repr_mode = str(getattr(self.cfg.data, "representation", "binary")).lower()
            if repr_mode == "tokens":
                self.loss_fn = token_score_interpolation_loss
            else:
                self.loss_fn = binary_score_interpolation_loss
            self.sampler = HeunSampler(self.model, self.proc, cfg)

            # ── callbacks ────────────────────────────────────────────────────────────

            # Offline entropy profile: if it uses dist collectives, keep it on all ranks.
            if self.entropy_offline_enabled:
                self.callbacks.append(OfflineEntropyProfileCallback(cfg))

            # SigmaDataEstimator / plotting are master-only (no collectives, pure
            # logging). They assume single-tensor sequence batches and a single
            # scalar-sigma schedule, so they are skipped on the multimodal path
            # (dict batches, independent modality noise; sigma_data is configured).
            if self.is_master and not self.is_multimodal:
                self.callbacks.append(SigmaDataEstimator(num_batches=10))
                self.callbacks.append(
                    EntropySchedulePlotCallback(
                        every_k_epochs=int(getattr(cfg.train, "entropy_plot_every_k_epochs", 20))
                    )
                )

            # ---- ALL-RANK callbacks that use dist collectives ----
            # External PPL
            ext_cfg = getattr(cfg.train, "external_perplexity", None)
            if ext_cfg is None:
                ext_cfg = getattr(cfg.train, "external_ppl", None)

            if ext_cfg is not None and bool(getattr(ext_cfg, "enabled", False)):
                self.callbacks.append(ExternalPPLCallback(cfg)) 


            # VLB (All ranks - critical for DDP synchronization). Skipped on the
            # multimodal path: VLBBoundCallback rebuilds the eval loader without the
            # MultimodalTaskCollator, so the dict rows are default-collated and
            # compute_vlb_over_loader crashes calling .to(device) on a dict. The
            # per-modality loss/accuracy are logged directly from the multimodal step
            # instead, so no VLB estimate is lost that the multimodal path produces.
            vlb_cfg = getattr(cfg.train, "vlb", None)
            if (
                vlb_cfg is not None
                and bool(getattr(vlb_cfg, "enabled", False))
                and not self.is_multimodal
            ):
                self.callbacks.append(
                    VLBBoundCallback(
                        every_k_epochs=int(getattr(vlb_cfg, "every_k_epochs", 10)),
                        sigma_min_eval=getattr(vlb_cfg, "sigma_min_eval", None),
                        sigma_max_eval=getattr(vlb_cfg, "sigma_max_eval", None),
                        sigma_sampling=getattr(vlb_cfg, "sigma_sampling", "log-uniform"),
                        num_mc_samples_per_batch=int(getattr(vlb_cfg, "num_mc_samples_per_batch", 1)),
                        include_prior=bool(getattr(vlb_cfg, "include_prior", False)),
                        use_amp=bool(getattr(vlb_cfg, "use_amp", True)),
                        progress=bool(getattr(vlb_cfg, "progress", False)),
                    )
                )
                
            # MAUVE
            mauve_cfg = getattr(cfg.train, "mauve", None)
            if mauve_cfg is not None and bool(getattr(mauve_cfg, "enabled", False)):
                self.callbacks.append(MauveCallback(cfg))  # pass FULL cfg


            # Visualization / sample dumps
            vis_cfg = getattr(cfg.train, "visualization", None)
            if vis_cfg is not None and bool(getattr(vis_cfg, "enabled", False)):
                self.callbacks.append(VisualizationCallback(cfg))


        elif cfg.framework == "discrete_sedd":
            self.proc = DiscreteForwardProcess(cfg)
            self.loss_fn = dwdse_loss

            repr_mode = str(getattr(cfg.data, "representation", "tokens")).lower()
            if repr_mode == "binary":
                seq_len = int(getattr(cfg.data, "sequence_len", 0))
            else:
                seq_len = int(
                    getattr(
                        cfg.data,
                        "sequence_len_tokens",
                        getattr(cfg.data, "sequence_len_chars", getattr(cfg.data, "sequence_len", 0)),
                    )
                )

            self.sampler = TweedieTauLeapingSampler(
                model=self.model,
                process=self.proc,
                device=self.device,
                vocab_size=int(cfg.data.vocab_size),
                is_absorb=bool(self.proc.is_absorb),
                mask_id=(int(self.proc.mask_id) if self.proc.is_absorb else None),
                seq_len=seq_len,
                num_steps=int(getattr(getattr(cfg, "evaluation", object()), "num_sampling_steps", 128)),
                t_eps=float(getattr(cfg.diffusion.discrete, "eps", 1e-3)),
            )

            self.callbacks = []

            mauve_cfg = getattr(cfg.train, "mauve", None)
            if mauve_cfg is not None and bool(getattr(mauve_cfg, "enabled", False)):
                self.callbacks.append(MauveCallback(cfg))

            vis_cfg = getattr(cfg.train, "visualization", None)
            if vis_cfg is not None and bool(getattr(vis_cfg, "enabled", False)):
                self.callbacks.append(VisualizationCallback(cfg))

        else:
            raise ValueError(f"Unknown framework: {cfg.framework}")

        # ── resume ──────────────────────────────────────────────────────────
        self.global_step = 0
        self.resume_mode = "scratch"  # one of: scratch | init_from | resume
        self.start_epoch = self._resume()

        # Multimodal warm start (sequence-checkpoint column surgery) + curriculum
        # trunk freeze. Warm start only applies to a fresh run; a true resume
        # already carries the trained (and possibly unfrozen) weights and its own
        # freeze bookkeeping is re-derived from the resumed global_step.
        if self.is_multimodal:
            if self.resume_mode == "scratch":
                self._mm_warm_start()
            self._mm_apply_initial_freeze()

        if self.is_master and self.tb is not None:
            self.tb.prepare_for_run(self.resume_mode)
            self.writer = self.tb.writer

        if self.ckpt_interval_enabled and self.ckpt_interval_every_steps > 0:
            gs = int(self.global_step)
            k = (gs // self.ckpt_interval_every_steps) + 1
            self._next_interval_ckpt_step = k * self.ckpt_interval_every_steps
        else:
            self._next_interval_ckpt_step = None

        if self.resume_interval_enabled and self.resume_interval_every_steps > 0:
            gs = int(self.global_step)
            k = (gs // self.resume_interval_every_steps) + 1
            self._next_resume_ckpt_step = k * self.resume_interval_every_steps
        else:
            self._next_resume_ckpt_step = None

        if self.cfg.framework == "continuous_score":
            self._load_entropy_tables_if_any()
            if getattr(self, "_entropy_ready", False) and self.entropy_profile_source == "none":
                self.entropy_profile_source = "disk"

        if self.is_master:
            self._print_model_summary()

    # ──────────────────────────────────────────────────────────────────────
    # EMA and logging helpers
    # ──────────────────────────────────────────────────────────────────────
    def _update_ema(self) -> None:
        if self.ema_ramp_enabled:
            self.ema.decay = _ema_decay_for_step(
                self.ema_ramp_start,
                self.ema_ramp_end,
                self.ema_ramp_steps,
                self.global_step + 1,
            )
        self.ema.update(self.model)

    def _log_wandb(self, data: dict):
        if not self.use_wandb or not self.is_master:
            return
        # When TensorBoard sync is on, the same metrics already flow to W&B via
        # the mirrored TB writes (with the TB global_step as the x-axis). Logging
        # them again here would duplicate the series and fight over the W&B step,
        # so stand down and let the mirror be the single source of truth.
        if getattr(self, "_wandb_sync_tb", False):
            return
        payload = dict(data)
        payload["global_step"] = int(self.global_step)
        wandb.log(payload)

    # ──────────────────────────────────────────────────────────────────────
    # Model summary
    # ──────────────────────────────────────────────────────────────────────
    def _print_model_summary(self):
        model_for_summary = _unwrap_all(self.model)
        total, trainable, non_trainable = _count_params(model_for_summary)
        first_param = next(model_for_summary.parameters(), None)
        p_dtype = first_param.dtype if first_param is not None else torch.float32
        amp_on = bool(self.cfg.train.use_fp16)
        amp_dtype = "bf16" if (amp_on and torch.cuda.is_bf16_supported()) else ("fp16" if amp_on else "fp32")

        sampler_name = "HeunSampler" if self.cfg.framework == "continuous_score" else "TweedieTauLeapingSampler"
        sampler_steps = getattr(self.cfg.evaluation, "num_sampling_steps", None)

        print("\n" + "─" * 80)
        print(f"Experiment: {self.cfg.experiment}")
        print(
            f"Framework : {self.cfg.framework}  |  Device: {self.device} (Rank {self.rank}/{self.world_size}) |  AMP: {amp_on} ({amp_dtype})"
        )
        print(f"Model     : {model_for_summary.__class__.__name__}")
        print(
            f"Params    : total={_fmt_num(total)} ({_human_bytes(total, p_dtype)}), "
            f"trainable={_fmt_num(trainable)}, frozen={_fmt_num(non_trainable)}"
        )
        print(f"Optimizer : {self.opt.__class__.__name__}  |  Scheduler: {self.lr_sched.__class__.__name__}")
        print(f"EMA       : decay={self.cfg.train.ema_decay}")
        print(f"Data      : dataset={self.cfg.data.dataset}, batch_size={self.cfg.train.batch_size} (Global)")
        if hasattr(self.cfg.data, "vocab_size"):
            extra = f"vocab_size={self.cfg.data.vocab_size}"
            if hasattr(self.cfg.data, "sequence_len"):
                extra += f", seq_len={self.cfg.data.sequence_len}"
            print(f"Discrete  : {extra}")
        if self.cfg.framework == "discrete_sedd":
            print(
                f"Diffusion : Q={self.cfg.diffusion.discrete.q_matrix_type}, "
                f"schedule={self.cfg.diffusion.discrete.schedule}, "
                f"t_max={self.cfg.diffusion.discrete.t_max}"
            )
        print(f"Sampler   : {sampler_name}" + (f" (steps={sampler_steps})" if sampler_steps is not None else ""))
        print(
            f"Resume    : start_epoch={self.start_epoch}, total_epochs={self.cfg.train.epochs}, "
            f"global_step={self.global_step}"
        )
        print(f"Paths     : run_dir={self.run_dir}, ckpt={self._checkpoint_path()}")
        print(
            f"Checkpointing: save_last={self.save_last}, save_top_k={self.save_top_k}, mode={self.checkpoint_mode}"
        )
        print("─" * 80 + "\n")

    def _maybe_run_sanity(self):
        s = getattr(self.cfg.train, "sanity", None)
        if s is None or (not bool(getattr(s, "enabled", False))):
            return

        sanity_epoch = int(getattr(s, "run_epoch", -1))

        if self.is_master:
            print(f"[sanity] running pre-train callbacks once (epoch={sanity_epoch}) using ACTUAL config values...")

        # Run the same hooks your training uses, once, before the epoch loop.
        # This will execute ALL-RANK callbacks safely if they set run_on_all_ranks=True.
        self._run_callbacks("on_epoch_end", sanity_epoch)
        self._run_callbacks("on_train_epoch_end", sanity_epoch)

        if self.ddp_active:
            dist.barrier()

        if self.is_master:
            print("[sanity] done.")


    # ──────────────────────────────────────────────────────────────────────
    # Checkpointing / Resume
    # ──────────────────────────────────────────────────────────────────────
    def _checkpoint_path(self, name: str = "last") -> Path:
        return self.ckpt_dir / f"{name}.pt"

    def _rng_state(self):
        state = {
            "py_random": random.getstate(),
            "np_random": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _set_rng_state(self, state: dict):
        try:
            random.setstate(state["py_random"])
            np.random.set_state(state["np_random"])
            torch.set_rng_state(state["torch_cpu"])
            if torch.cuda.is_available() and "torch_cuda" in state:
                torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception as e:
            if self.is_master:
                print(f"⚠️  RNG state restore warning: {e}")

    def _load_checkpoint(self, path: Path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"]
        clean_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("_orig_mod.", "")
            if k.startswith("module."):
                k = k[7:]
            clean_state_dict[k] = v

        model_to_load = _unwrap_all(self.model)

        try:
            model_to_load.load_state_dict(clean_state_dict, strict=True)
        except RuntimeError:
            model_to_load.load_state_dict(clean_state_dict, strict=False)

        if "opt" in ckpt:
            self.opt.load_state_dict(ckpt["opt"])
        if "lr_sched" in ckpt:
            self.lr_sched.load_state_dict(ckpt["lr_sched"])
        if self.use_scaler and ("scaler" in ckpt) and (ckpt["scaler"] is not None):
            self.scaler.load_state_dict(ckpt["scaler"])

        if "ema" in ckpt and ckpt["ema"] is not None:
            self.ema.load_state_dict(ckpt["ema"])
            self.ema.to(self.device)

        if "rng_state" in ckpt and ckpt["rng_state"] is not None:
            self._set_rng_state(ckpt["rng_state"])

        self.global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", -1) + 1
        self.best_metric = ckpt.get("best_metric", self.best_metric)
        self.best_ckpts = ckpt.get("best_ckpts", self.best_ckpts)
        self.early_stop_best = ckpt.get("early_stop_best", self.best_metric)
        self.early_stop_bad_epochs = int(ckpt.get("early_stop_bad_epochs", 0))

        # Multimodal exact-resume: restore the batch-sampler cursor and collator
        # RNG/counter, and if the checkpoint was taken mid-epoch, re-enter that
        # same epoch at the saved batch cursor instead of skipping to the next
        # epoch (gate 5). Non-multimodal resume is unchanged.
        self._mm_resume_cursor = 0
        self._mm_resume_epoch = -1
        if self.is_multimodal:
            bs = getattr(self.train_loader, "batch_sampler", None)
            coll = getattr(self.train_loader, "collate_fn", None)
            if "mm_sampler" in ckpt and bs is not None and hasattr(bs, "load_state_dict"):
                bs.load_state_dict(ckpt["mm_sampler"])
            if "mm_collator" in ckpt and coll is not None and hasattr(coll, "load_state_dict"):
                coll.load_state_dict(ckpt["mm_collator"])
            cursor = int(ckpt.get("mm_batch_cursor", 0))
            steps_per_epoch = int(getattr(self.cfg.train, "steps_per_epoch", 0))
            # Only re-enter the same epoch when the batch sampler can actually seek
            # to a cursor (SourceLengthBatchSampler). A non-seekable fallback
            # sampler (e.g. DistributedLengthBucketBatchSampler when
            # steps_per_epoch<=0) would restart the epoch from batch 0, so fall
            # back to the standard next-epoch resume instead.
            sampler_seekable = bs is not None and hasattr(bs, "start_batch")
            if (
                cursor > 0
                and sampler_seekable
                and (steps_per_epoch <= 0 or cursor < steps_per_epoch)
            ):
                # Rolling checkpoint mid-epoch: continue the same epoch.
                self._mm_resume_cursor = cursor
                self._mm_resume_epoch = int(ckpt.get("epoch", -1))
                start_epoch = int(ckpt.get("epoch", -1))

        if self.is_master:
            print(f"Resumed from {path} (Epoch {start_epoch})")
        return start_epoch

    def _clean_state_dict_keys(self, state_dict: dict) -> dict:
        """
        Make checkpoint state_dict portable across:
          - torch.compile (_orig_mod.)
          - DDP (module.)
        """
        clean = {}
        for k, v in state_dict.items():
            k = k.replace("_orig_mod.", "")
            if k.startswith("module."):
                k = k[7:]
            clean[k] = v
        return clean

    def _init_from_checkpoint_weights_only(self, path: Path):
        """
        Initialize:
          - running model weights   <- ckpt["model"]
          - EMA shadow weights      <- ckpt["ema"] (if present)
        WITHOUT resuming optimizer/scheduler/scaler/global_step/epoch.

        This is the "fresh run from weights" mode.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "model" not in ckpt:
            raise KeyError(f"Checkpoint at {path} missing key 'model'.")

        # --- load running weights into the running model ---
        model_sd = self._clean_state_dict_keys(ckpt["model"])
        model_to_load = _unwrap_all(self.model)
        try:
            model_to_load.load_state_dict(model_sd, strict=True)
        except RuntimeError:
            model_to_load.load_state_dict(model_sd, strict=False)

        # --- load EMA weights into EMA object (shadow) if present & enabled ---
        use_ema = bool(getattr(self.cfg.train, "init_from_use_ema", True))
        if use_ema and ("ema" in ckpt) and (ckpt["ema"] is not None):
            try:
                # Your checkpoints store: {"decay": float, "shadow": {name: tensor(cpu), ...}}
                self.ema.load_state_dict(ckpt["ema"])
                self.ema.to(self.device)
            except Exception as e:
                if self.is_master:
                    print(
                        f"⚠️  init_from: failed to load EMA state ({type(e).__name__}: {e}). "
                        f"Continuing with EMA re-initialized from running weights."
                    )
                # fallback: re-init EMA shadow from current model weights
                self.ema = EMA(self.model, decay=self.cfg.train.ema_decay)
                self.ema.to(self.device)
        else:
            # If EMA load disabled/unavailable: re-init EMA from current model weights.
            self.ema = EMA(self.model, decay=self.cfg.train.ema_decay)
            self.ema.to(self.device)

        # --- reset training bookkeeping (fresh run) ---
        self.global_step = 0
        self.best_metric = math.inf if self.checkpoint_mode == "min" else -math.inf
        self.best_ckpts = []
        self.early_stop_best = self.best_metric
        self.early_stop_bad_epochs = 0

        if self.is_master:
            print(f"[init_from] Initialized running weights from: {path}")
            if use_ema and ("ema" in ckpt) and (ckpt["ema"] is not None):
                print("[init_from] Loaded EMA shadow from checkpoint as well.")
            else:
                print("[init_from] EMA shadow initialized from running weights (no EMA loaded).")

    def _resume(self) -> int:
        ckpt_path = self._checkpoint_path()  # runs/<experiment>/checkpoints/last.pt

        init_from = getattr(self.cfg.train, "init_from", None)
        init_force = bool(getattr(self.cfg.train, "init_from_force", False))

        # Opt-in: weights-only initialization from a specified checkpoint.
        # This is NOT a true resume of training history.
        if init_from is not None:
            init_path = Path(init_from)
            if init_force or (not ckpt_path.exists()):
                if not init_path.exists():
                    raise FileNotFoundError(f"cfg.train.init_from not found: {init_path}")
                if self.is_master:
                    print(f"[init_from] force={init_force} | last_exists={ckpt_path.exists()} | path={init_path}")
                self._init_from_checkpoint_weights_only(init_path)
                self.resume_mode = "init_from"
                return 0

        # True training resume from last.pt
        if ckpt_path.exists():
            self.resume_mode = "resume"
            return self._load_checkpoint(ckpt_path)

        # Fresh run from scratch
        self.resume_mode = "scratch"
        if self.is_master:
            print("🏁 Starting training from scratch.")
        return 0

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────
    def _run_callbacks(self, method_name: str, *args):
        for cb in self.callbacks:
            run_all = bool(getattr(cb, "run_on_all_ranks", False))
            if self.is_master or run_all:
                if hasattr(cb, method_name):
                    if self.is_master and method_name == "on_epoch_end":
                        print(f"[callback] before {cb.__class__.__name__}: {_gpu_mem_msg(self.device)}")
                    getattr(cb, method_name)(self, *args)
                    if self.is_master and method_name == "on_epoch_end":
                        print(f"[callback] after  {cb.__class__.__name__}: {_gpu_mem_msg(self.device)}")
    
    # ──────────────────────────────────────────────────────────────────────
    # Multimodal (18-bit paired) training path: dict batches, warm start,
    # trunk freeze/unfreeze, lower trunk LR, and sequence-only replay (gates 2/3).
    # Every method below is inert unless cfg.data.dataset selects the paired
    # multimodal corpus, so the sequence-only paths are byte-for-byte unchanged.
    # ──────────────────────────────────────────────────────────────────────
    _MM_ADAPTER_PREFIXES = (
        "patch_proj",
        "unpatch_proj_content",
        "head",
        "mm_embed",
    )

    def _mm_is_trunk_param(self, name: str) -> bool:
        """True for warm-started shared params; False for new multimodal adapters.

        Adapters (``patch_proj``, ``unpatch_proj_content``, ``head``, ``mm_embed``)
        carry the freshly initialized structure columns / slot embeddings and
        always train; the trunk (``blocks``, ``time_*``, ``cont_input_proj``) is the
        warm-started shared backbone the curriculum freezes first, then unfreezes
        at a lower learning rate.
        """
        n = name.replace("_orig_mod.", "")
        if n.startswith("module."):
            n = n[7:]
        top = n.split(".", 1)[0]
        return top not in self._MM_ADAPTER_PREFIXES

    def _mm_build_param_groups(self):
        raw = _unwrap_all(self.model)
        adapter, trunk = [], []
        for name, p in raw.named_parameters():
            (trunk if self._mm_is_trunk_param(name) else adapter).append(p)
        # Order matters: group 0 = adapters, group 1 = trunk (scaled each step).
        return [
            {"params": adapter, "mm_group": "adapter"},
            {"params": trunk, "mm_group": "trunk"},
        ]

    def _mm_set_trunk_requires_grad(self, flag: bool) -> None:
        raw = _unwrap_all(self.model)
        for name, p in raw.named_parameters():
            if self._mm_is_trunk_param(name):
                p.requires_grad_(bool(flag))

    def _mm_apply_initial_freeze(self) -> None:
        """Freeze the trunk iff still inside the warm-start freeze window."""
        if self.freeze_trunk_steps <= 0:
            self._trunk_frozen = False
            return
        should_freeze = int(self.global_step) < int(self.freeze_trunk_steps)
        self._mm_set_trunk_requires_grad(not should_freeze)
        self._trunk_frozen = should_freeze
        if should_freeze and self.is_master:
            print(
                f"[multimodal] trunk frozen for warm start until step "
                f"{self.freeze_trunk_steps} (adapters train first)."
            )

    def _mm_maybe_unfreeze(self) -> None:
        if self._trunk_frozen and int(self.global_step) >= int(
            self.freeze_trunk_steps
        ):
            self._mm_set_trunk_requires_grad(True)
            self._trunk_frozen = False
            if self.is_master:
                print(
                    f"[multimodal] unfroze trunk at step {self.global_step} "
                    f"(trunk_lr_mult={self.trunk_lr_mult})."
                )

    def _mm_warm_start(self) -> None:
        """Column-surgery warm start from a sequence-only checkpoint (plan 6.4)."""
        path = self.warm_start_seq_checkpoint
        if not path:
            return
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"cfg.model.warm_start_seq_checkpoint not found: {p}"
            )
        from utils.protein_warmstart import warm_start_multimodal_from_sequence

        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        seq_sd = None
        if isinstance(ckpt, dict):
            use_ema = bool(getattr(self.cfg.model, "warm_start_use_ema", True))
            ema_obj = ckpt.get("ema") if use_ema else None
            if isinstance(ema_obj, dict) and "shadow" in ema_obj:
                seq_sd = ema_obj["shadow"]
            if seq_sd is None:
                seq_sd = ckpt.get("model", ckpt)
        else:
            seq_sd = ckpt
        report = warm_start_multimodal_from_sequence(
            _unwrap_all(self.model), seq_sd, verbose=self.is_master
        )
        if report["exact"] == 0 and report["slot_copied"] == 0:
            raise RuntimeError(
                f"warm start from {p} copied no parameters "
                f"({report}); trunk width/depth likely mismatched."
            )
        # Re-seed the EMA shadow from the warm-started weights so evaluation does
        # not average toward the discarded random init.
        self.ema = EMA(self.model, decay=self.cfg.train.ema_decay)
        self.ema.to(self.device)
        if self.is_master:
            print(f"[multimodal] warm start from {p}: {report}")

    def _mm_next_replay_batch(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        try:
            return next(self._replay_iter)
        except StopIteration:
            self._replay_iter = iter(self.replay_loader)
            return next(self._replay_iter)

    def _mm_should_replay(self) -> bool:
        """Deterministic per-step replay decision (resumable, keyed by step)."""
        if (
            not self.is_multimodal
            or self.replay_fraction <= 0.0
            or self.replay_loader is None
        ):
            return False
        seed = int(getattr(self.cfg.train, "seed", 42))
        draw = np.random.default_rng((seed, 777, int(self.global_step))).random()
        return bool(draw < self.replay_fraction)

    def _step_multimodal(self, batch, is_train: bool):
        """One 18-bit paired micro-step with gradient accumulation.

        Accumulates ``grad_accum_steps`` micro-batches into one optimizer step so
        the effective global batch (measured in residue-patches) can be held
        constant across node counts. The optimizer/scheduler/EMA and the trunk-LR
        rescale fire only on the accumulation boundary; ``self._did_optim_step``
        tells the train loop when a real optimizer step happened so it advances
        ``global_step`` once per effective batch (not per micro-batch). With
        ``grad_accum_steps == 1`` every micro-step is a boundary, i.e. the previous
        single-step behavior. Under DDP the gradient all-reduce is suppressed on
        non-boundary micro-steps via ``no_sync`` so accumulation stays cheap.
        """
        from trainers.multimodal_step import multimodal_training_step

        if is_train:
            # Unfreeze before the forward so the boundary step trains the trunk.
            self._mm_maybe_unfreeze()

        accum = max(1, int(getattr(self, "grad_accum_steps", 1)))
        is_boundary = (not is_train) or ((self._accum_counter + 1) >= accum)
        # Suppress DDP gradient sync on the non-final micro-steps of a window.
        suppress_sync = (
            is_train
            and accum > 1
            and not is_boundary
            and self.ddp_active
            and hasattr(self.model, "no_sync")
        )
        sync_ctx = self.model.no_sync() if suppress_sync else contextlib.nullcontext()

        # Online entropy schedule for the paired path. draw_sigma() returns the
        # log-normal/EDM base until the schedule is warmed up + ready, then blends
        # toward the entropic schedule (gamma ramp), exactly like _step_continuous.
        # Only engage on training micro-steps and only when entropy is configured,
        # so every other multimodal config is byte-for-byte unchanged.
        entropy_on = is_train and self.cfg.framework == "continuous_score" and (
            self.entropy_compute or self.entropy_use_for_sampling
        )
        sigma_draw_fn = self.entropy_ctrl.draw_sigma if entropy_on else None
        entropy_sink = {} if (is_train and self.entropy_compute) else None
        diagnostics_sink = {} if not is_train else None

        amp_enabled = bool(self.cfg.train.use_fp16)
        with autocast(self.device.type, enabled=amp_enabled, dtype=self.amp_dtype):
            loss, components = multimodal_training_step(
                self.model,
                batch,
                self.proc,
                self.cfg,
                device=self.device,
                is_train=is_train,
                sigma_draw_fn=sigma_draw_fn,
                entropy_sink=entropy_sink,
                diagnostics_sink=diagnostics_sink,
            )

        # Feed the entropy FIFO buffer with this micro-batch's valid per-modality
        # (sigma, denoising-MSE) pairs (both modalities, independent sigmas). Runs
        # every micro-batch under grad accumulation, mirroring _step_continuous.
        if entropy_sink:
            vs = entropy_sink["valid_seq"]
            vt = entropy_sink["valid_struct"]
            sig = torch.cat([entropy_sink["sigma_seq"][vs], entropy_sink["sigma_struct"][vt]])
            met = torch.cat([entropy_sink["metric_seq"][vs], entropy_sink["metric_struct"][vt]])
            if sig.numel() > 0:
                self._update_entropy_buffer(sig, met)

        if is_train:
            if self._accum_counter == 0:
                self.opt.zero_grad(set_to_none=True)
            scaled = loss / accum  # so accumulated grads average the window
            with sync_ctx:
                if self.use_scaler:
                    self.scaler.scale(scaled).backward()
                else:
                    scaled.backward()

            if is_boundary:
                if self.use_scaler:
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.opt.step()

                self.lr_sched.step()
                # The scheduler forces one lr on every group; scale the trunk group
                # (group 1) back down so the warm-started backbone moves slower.
                if self._trunk_group_active:
                    self.opt.param_groups[1]["lr"] *= self.trunk_lr_mult
                self._update_ema()
                self._accum_counter = 0
                self._did_optim_step = True
            else:
                self._accum_counter += 1
                self._did_optim_step = False
        else:
            self._did_optim_step = True

        self._last_mm_components = {k: float(v) for k, v in components.items()}
        if diagnostics_sink is not None:
            diagnostics_sink["task_names"] = list(batch.get("task_names", []))
            self._last_mm_val_diagnostics = diagnostics_sink
        else:
            self._last_mm_val_diagnostics = None
        return loss.item()

    # -----------------------------------------------------------------------------
    # Trainer method: full step_continuous
    # -----------------------------------------------------------------------------
    def _step_continuous(self, x0, is_train: bool):
        """
        Continuous score training step with optional CFG-style prefix conditioning.

        Supports:
        - binary continuous diffusion: x_t [B,S], logits [B,S] or [B,S,1]
        - one-hot token continuous diffusion: x_t [B,S,V], logits [B,S,V]
        - optional prefix conditioning
        - optional self-conditioning

        Memory-oriented implementation:
        - token one-hot state uses AMP dtype when enabled
        - xt is built in-place (no standalone noise tensor)
        - x0_hat allocated only if actually needed
        - large temporaries freed as early as possible
        """
        B = x0.size(0)
        repr_mode = str(getattr(self.cfg.data, "representation", "binary")).lower()
        is_cont_tokens = (repr_mode == "tokens")

        amp_enabled = bool(getattr(self.cfg.train, "use_fp16", False))
        state_dtype = self.amp_dtype if amp_enabled else torch.float32

        # ------------------------------------------------------------------
        # Prepare clean target / clean state
        # ------------------------------------------------------------------
        if is_cont_tokens:
            V = int(self.cfg.data.vocab_size)
            x0_target = x0.to(self.device, non_blocking=True).long().view(B, -1)  # [B,S]
            S = x0_target.size(1)

            # Dense one-hot clean state in reduced precision when AMP is enabled.
            x0_clean = torch.nn.functional.one_hot(
                x0_target, num_classes=V
            ).to(dtype=state_dtype)  # [B,S,V]
        else:
            x0_clean = x0.to(self.device, non_blocking=True).view(B, -1).to(dtype=torch.float32)  # [B,S]
            S = x0_clean.size(1)
            x0_target = None

        # ------------------------------------------------------------------
        # Conditioning setup
        # ------------------------------------------------------------------
        cond_cfg = getattr(self.cfg, "cond", None)
        cond_enabled_cfg = (cond_cfg is not None) and bool(getattr(cond_cfg, "enabled", False))

        if not cond_enabled_cfg:
            cond_enabled = False
            prefix_mask = None
            cL_pos = None
        else:
            cL_pos = _sample_cond_len_positions_per_example_continuous(
                self.cfg, B, S, device=self.device
            )
            cond_enabled = bool((cL_pos.max().item() if B > 0 else 0) > 0)
            prefix_mask = _make_prefix_mask_from_lengths(cL_pos, S) if cond_enabled else None

        noise_prefix = True
        suffix_only_loss = False
        p_uncond = 0.0

        if cond_enabled:
            noise_prefix = bool(getattr(cond_cfg, "noise_prefix", False))
            suffix_only_loss = bool(getattr(cond_cfg, "loss_on_suffix_only", True))
            p_uncond = float(getattr(cond_cfg, "p_uncond", 0.0))
            p_uncond = max(0.0, min(1.0, p_uncond))

        # ------------------------------------------------------------------
        # Draw sigma
        # ------------------------------------------------------------------
        sigma = self._draw_sigma(B)
        sigma_exp = sigma.view(-1, 1, 1) if is_cont_tokens else sigma.view(-1, 1)

        # ------------------------------------------------------------------
        # Build xt
        # ------------------------------------------------------------------
        drop_mask = None
        prefix_used_full = None

        if cond_enabled and (not noise_prefix):
            drop_mask = (torch.rand(B, device=self.device) < p_uncond)

            # Keep an editable prefix tensor only when conditional clean-prefix mode is used.
            prefix_used_full = x0_clean.clone()

            null_full = _make_null_prefix_full(x0_clean, prefix_mask, self.cfg)

            if drop_mask.any():
                if is_cont_tokens:
                    replace = drop_mask.view(B, 1, 1) & prefix_mask.unsqueeze(-1)
                    prefix_used_full[replace] = null_full[replace]
                else:
                    replace = drop_mask.view(B, 1) & prefix_mask
                    prefix_used_full[replace] = null_full[replace]

            del null_full

            # Build xt in-place: xt = sigma * N(0, I) + x0_clean
            xt = torch.empty_like(x0_clean)
            xt.normal_()
            xt.mul_(sigma_exp)
            xt.add_(x0_clean)

            # Clamp prefix positions to chosen clean/null prefix.
            if is_cont_tokens:
                pm = prefix_mask.unsqueeze(-1)
                xt[pm] = prefix_used_full[pm]
            else:
                xt[prefix_mask] = prefix_used_full[prefix_mask]

        else:
            # Unconditional or noisy-prefix mode
            xt = torch.empty_like(x0_clean)
            xt.normal_()
            xt.mul_(sigma_exp)
            xt.add_(x0_clean)

        # In token mode, loss target is x0_target, not x0_clean.
        # If we do not need x0_clean anymore, free it early.
        if is_cont_tokens:
            need_x0_clean_later = cond_enabled and (not noise_prefix)
            if not need_x0_clean_later:
                del x0_clean

        # ------------------------------------------------------------------
        # Self-conditioning
        # ------------------------------------------------------------------
        sc_enabled = bool(getattr(self.cfg.model, "self_condition", False))
        p_sc = float(getattr(self.cfg.train, "self_condition_prob", 0.5))

        x0_hat = None

        if sc_enabled and p_sc > 0.0:
            sc_mask = (torch.rand(B, device=self.device) < p_sc)
            needs_sc_injection = cond_enabled and (not noise_prefix)

            if sc_mask.any() or needs_sc_injection:
                x0_hat = torch.zeros_like(xt)

            if sc_mask.any():
                with autocast(self.device.type, enabled=amp_enabled, dtype=self.amp_dtype):
                    logits_sc = _model_logits_continuous(
                        self.model,
                        self.cfg,
                        xt,
                        sigma,
                        None,
                    )

                if is_cont_tokens:
                    # Keep detached SC state in xt dtype to avoid fp32 bloat.
                    x0_hat_all = torch.softmax(logits_sc.float(), dim=-1).detach().to(dtype=xt.dtype)

                    # Avoid torch.where over full tensor when possible.
                    if sc_mask.all():
                        x0_hat.copy_(x0_hat_all)
                    else:
                        x0_hat[sc_mask] = x0_hat_all[sc_mask]
                else:
                    x0_hat_all = torch.sigmoid(logits_sc.float()).detach().to(dtype=xt.dtype)
                    if sc_mask.all():
                        x0_hat.copy_(x0_hat_all)
                    else:
                        x0_hat[sc_mask] = x0_hat_all[sc_mask]

                del logits_sc, x0_hat_all

            if needs_sc_injection:
                if is_cont_tokens:
                    pm = prefix_mask.unsqueeze(-1)
                    x0_hat[pm] = prefix_used_full[pm]
                else:
                    x0_hat[prefix_mask] = prefix_used_full[prefix_mask]

        # prefix_used_full no longer needed after SC injection.
        if prefix_used_full is not None:
            del prefix_used_full

        # ------------------------------------------------------------------
        # Main forward + loss
        # ------------------------------------------------------------------
        with autocast(self.device.type, enabled=amp_enabled, dtype=self.amp_dtype):
            logits = _model_logits_continuous(
                self.model,
                self.cfg,
                xt,
                sigma,
                x0_hat,
            )

            loss_mask = None
            if suffix_only_loss and cond_enabled and (not noise_prefix):
                loss_mask = (~prefix_mask).to(dtype=torch.float32)

            loss_target = x0_target if is_cont_tokens else x0_clean

            if self.entropy_compute:
                loss, entropy_metric = self.loss_fn(
                    logits,
                    loss_target,
                    sigma,
                    self.cfg,
                    return_entropy_metric=True,
                    mask=loss_mask,
                )
            else:
                loss = self.loss_fn(
                    logits,
                    loss_target,
                    sigma,
                    self.cfg,
                    return_entropy_metric=False,
                    mask=loss_mask,
                )
                entropy_metric = None

        # Large intermediates no longer needed before backward bookkeeping.
        del logits, xt
        if x0_hat is not None:
            del x0_hat
        if (not is_cont_tokens) and (x0_clean is not None):
            # binary mode still uses x0_clean as loss target until here
            del x0_clean

        # ------------------------------------------------------------------
        # Optim step
        # ------------------------------------------------------------------
        if is_train:
            self.opt.zero_grad(set_to_none=True)

            if self.use_scaler:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()

            self.lr_sched.step()
            self._update_ema()

        # ------------------------------------------------------------------
        # Entropy buffer update
        # ------------------------------------------------------------------
        if self.entropy_compute and (entropy_metric is not None):
            if cond_enabled and (drop_mask is not None):
                keep = ~drop_mask
                if keep.any():
                    self._update_entropy_buffer(sigma[keep], entropy_metric[keep])
            else:
                self._update_entropy_buffer(sigma, entropy_metric)

        return loss.item()

    def _step_discrete(self, x0, is_train: bool):
        B, S = x0.shape
        x0_flat = x0.to(self.device, non_blocking=True).view(B, -1).long()

        cond_cfg = getattr(self.cfg, "cond", None)
        cond_enabled_cfg = (cond_cfg is not None) and bool(getattr(cond_cfg, "enabled", False))

        if cond_enabled_cfg:
            cL_pos = _sample_cond_len_positions_per_example_discrete(self.cfg, B, S, self.device)
            cond_enabled = bool((cL_pos.max().item() if B > 0 else 0) > 0)
            prefix_mask = _make_prefix_mask_from_lengths(cL_pos, S) if cond_enabled else None
        else:
            cond_enabled = False
            prefix_mask = None

        noise_prefix = bool(getattr(cond_cfg, "noise_prefix", False)) if cond_enabled else True
        suffix_only_loss = bool(getattr(cond_cfg, "loss_on_suffix_only", True)) if cond_enabled else False
        p_uncond = float(getattr(cond_cfg, "p_uncond", 0.0)) if cond_enabled else 0.0
        p_uncond = max(0.0, min(1.0, p_uncond))

        eps = getattr(self.cfg.diffusion.discrete, "eps", 1e-3)
        t = (1 - eps) * torch.rand(B, device=self.device) + eps
        sigma = self.proc.get_cumulative_noise(t)

        xt = self.proc.sample_xt(x0_flat, t)

        effective_prefix_mask = None
        if cond_enabled and (not noise_prefix):
            drop_mask = (torch.rand(B, device=self.device) < p_uncond)
            effective_prefix_mask = prefix_mask & (~drop_mask.view(B, 1))
            xt[effective_prefix_mask] = x0_flat[effective_prefix_mask]

        with autocast(self.device.type, enabled=self.cfg.train.use_fp16, dtype=self.amp_dtype):
            model_scores = self.model(xt, sigma)

            loss_mask = None
            if cond_enabled and (not noise_prefix) and suffix_only_loss:
                loss_mask = (~effective_prefix_mask).to(torch.float32)

            loss = self.loss_fn(
                model_scores,
                x0_flat,
                xt,
                self.proc,
                t,
                self.cfg,
                mask=loss_mask,
            )

        if is_train:
            self.opt.zero_grad(set_to_none=True)

            if self.use_scaler:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()

            self.lr_sched.step()
            self._update_ema()

        return loss.item()

    @torch.compiler.disable
    @torch.no_grad()
    def _validate_epoch(self, step_fn):
        self.model.eval()
        self.ema.apply(self.model)

        # Deterministic validation (opt-in via cfg.train.deterministic_validation):
        # the validation loss is a weighted denoising loss at a *random* sigma and a
        # random noise draw per batch, so with a small validation_max_batches it is a
        # high-variance estimator (it can swing several-fold epoch to epoch). Fixing
        # the sigma / Gaussian-noise stream to a constant seed makes val loss
        # comparable across epochs, so early-stopping and best-checkpoint selection
        # act on real generalization changes rather than sampling noise. The RNG is
        # snapshotted and restored so the training stream is completely unaffected.
        deterministic = bool(getattr(self.cfg.train, "deterministic_validation", False))
        rng_snapshot = None
        if deterministic:
            rng_snapshot = self._rng_state()
            vseed = int(getattr(self.cfg.train, "val_seed", 1234))
            random.seed(vseed)
            np.random.seed(vseed)
            torch.manual_seed(vseed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(vseed)

        local_loss = torch.tensor(0.0, device=self.device)
        local_count = torch.tensor(0.0, device=self.device)
        mm_stats = None
        if self.is_multimodal:
            task_weights = getattr(self.cfg.data, "task_weights", {})
            mm_stats = _mm_validation_accumulator(self.device, dict(task_weights).keys())

        pbar = tqdm(self.val_loader, desc="Validating", leave=False, disable=not self.is_master)

        for batch in pbar:
            if 0 < int(getattr(self.cfg.train, "validation_max_batches", 0)) <= int(local_count.item()):
                break
            x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
            loss = step_fn(x0, is_train=False)
            local_loss += loss
            local_count += 1.0
            if mm_stats is not None and self._last_mm_val_diagnostics is not None:
                diagnostics = self._last_mm_val_diagnostics
                _mm_accumulate_validation(
                    mm_stats, diagnostics, diagnostics.get("task_names", [])
                )

        if self.ddp_active:
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
            if mm_stats is not None:
                for values in mm_stats.values():
                    dist.all_reduce(values, op=dist.ReduceOp.SUM)

        avg_loss = (local_loss / local_count).item()
        val_metrics = _mm_finalize_validation(mm_stats) if mm_stats is not None else {}

        self.ema.restore(self.model)
        if rng_snapshot is not None:
            self._set_rng_state(rng_snapshot)
        return avg_loss, val_metrics

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ──────────────────────────────────────────────────────────────────────
    def _is_better(self, metric: float) -> bool:
        if self.checkpoint_mode == "min":
            return metric < self.best_metric
        else:
            return metric > self.best_metric

    def _build_ckpt_state(self, epoch: int) -> dict:
        raw_model = _unwrap_all(self.model)

        # --- NEW: make EMA checkpoint portable (save shadows on CPU) ---
        ema_sd = self.ema.state_dict()
        ema_sd_cpu = {
            "decay": float(ema_sd["decay"]),
            "shadow": {k: v.detach().cpu() for k, v in ema_sd["shadow"].items()},
        }
        # If you ever store backups (you currently don't persist them), ignore them:
        # ema_sd_cpu has only decay + shadow, which is all you need.

        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": raw_model.state_dict(),
            # --- CHANGED LINE HERE ---
            "ema": ema_sd_cpu,
            "opt": self.opt.state_dict(),
            "lr_sched": self.lr_sched.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_scaler else None,
            "rng_state": self._rng_state(),
            "best_metric": self.best_metric,
            "best_ckpts": self.best_ckpts,
            "early_stop_best": self.early_stop_best,
            "early_stop_bad_epochs": self.early_stop_bad_epochs,
        }

        # Multimodal resume payload: batch-sampler cursor + collator RNG/counter
        # so a preempted paired run resumes at the exact next batch (gate 5).
        if self.is_multimodal:
            bs = getattr(self.train_loader, "batch_sampler", None)
            coll = getattr(self.train_loader, "collate_fn", None)
            if bs is not None and hasattr(bs, "state_dict"):
                state["mm_sampler"] = bs.state_dict()
            if coll is not None and hasattr(coll, "state_dict"):
                state["mm_collator"] = coll.state_dict()
            state["mm_batch_cursor"] = int(getattr(self, "_epoch_batches_done", 0))

        return state

    def _save_ckpt(self, epoch: int, val_metric: float):
        # Only master saves
        if not self.is_master:
            return

        new_best = False
        new_best_path = None

        if self.save_top_k > 0 and self._is_better(val_metric):
            new_best = True
            self.best_metric = float(val_metric)

            name = f"epoch={epoch:04d}-val={val_metric:.4f}"
            new_best_path = self._checkpoint_path(name)
            self.best_ckpts.append({"path": new_best_path.name, "metric": float(val_metric), "epoch": int(epoch)})

            reverse = self.checkpoint_mode == "max"
            self.best_ckpts.sort(key=lambda d: d["metric"], reverse=reverse)

            while len(self.best_ckpts) > self.save_top_k:
                worst = self.best_ckpts.pop(-1)
                try:
                    os.remove(self.ckpt_dir / worst["path"])
                except FileNotFoundError:
                    pass

            print(f"✨ New best model at epoch {epoch} (val_metric={val_metric:.4f})")

        state = self._build_ckpt_state(epoch)

        if self.save_last:
            tmp_path = self._checkpoint_path("last.tmp")
            final_path = self._checkpoint_path("last")
            torch.save(state, tmp_path)
            os.replace(tmp_path, final_path)

        if self.save_top_k > 0 and new_best and new_best_path is not None:
            _atomic_torch_save(state, new_best_path)
            _atomic_torch_save(state, self._checkpoint_path("best"))

    def _maybe_save_resume_ckpt(self, epoch: int) -> None:
        """
        Save rolling resume checkpoint (last.pt) every N steps.

        This does NOT create archival step=...pt files.
        It only overwrites last.pt so Slurm-chained resumes waste minimal compute.
        """
        if (not self.is_master) or (not self.resume_interval_enabled):
            return
        if self._next_resume_ckpt_step is None:
            return
        if self.resume_interval_every_steps <= 0:
            return

        if int(self.global_step) < int(self._next_resume_ckpt_step):
            return

        state = self._build_ckpt_state(epoch)

        if self.save_last:
            tmp_path = self._checkpoint_path("last.tmp")
            final_path = self._checkpoint_path("last")
            torch.save(state, tmp_path)
            os.replace(tmp_path, final_path)

        while int(self._next_resume_ckpt_step) <= int(self.global_step):
            self._next_resume_ckpt_step += int(self.resume_interval_every_steps)
            
    def _maybe_save_interval_ckpt(self, epoch: int) -> None:
        """
        Save a checkpoint every N steps if enabled.

        Naming: step=000123456.pt
        Safe: master-only, no collectives.
        Pruning: if keep_last is None => keep ALL interval checkpoints.
        """
        if (not self.is_master) or (not self.ckpt_interval_enabled):
            return
        if self._next_interval_ckpt_step is None:
            return
        if self.ckpt_interval_every_steps <= 0:
            return

        if int(self.global_step) < int(self._next_interval_ckpt_step):
            return

        # Build checkpoint state (same as others)
        state = self._build_ckpt_state(epoch)

        # Save
        name = f"step={int(self.global_step):09d}"
        path = self._checkpoint_path(name)
        _atomic_torch_save(state, path)

        # Track interval ckpts for optional pruning (only interval ckpts)
        self._interval_ckpt_paths.append(path.name)

        keep_last = self.ckpt_interval_keep_last  # None => keep all
        if keep_last is not None and keep_last > 0:
            while len(self._interval_ckpt_paths) > keep_last:
                old = self._interval_ckpt_paths.pop(0)
                try:
                    os.remove(self.ckpt_dir / old)
                except FileNotFoundError:
                    pass

        # Advance threshold robustly even if steps jump
        while int(self._next_interval_ckpt_step) <= int(self.global_step):
            self._next_interval_ckpt_step += int(self.ckpt_interval_every_steps)


    # ──────────────────────────────────────────────────────────────────────
    def _update_early_stopping(self, val_metric: float) -> bool:
        cfg = getattr(self.cfg.train, "early_stopping", None)
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return False
        if self.global_step < int(getattr(cfg, "warmup_steps", 0)):
            return False

        min_delta = float(getattr(cfg, "min_delta", 0.0))
        if self.checkpoint_mode == "min":
            improved = val_metric < (self.early_stop_best - min_delta)
        else:
            improved = val_metric > (self.early_stop_best + min_delta)

        if improved:
            self.early_stop_best = float(val_metric)
            self.early_stop_bad_epochs = 0
        else:
            self.early_stop_bad_epochs += 1

        patience = max(1, int(getattr(cfg, "patience_epochs", 1)))
        should_stop = self.early_stop_bad_epochs >= patience
        if self.is_master:
            print(
                f"[early-stop] best={self.early_stop_best:.6f} "
                f"bad_epochs={self.early_stop_bad_epochs}/{patience} "
                f"min_delta={min_delta:g}"
            )
        return should_stop


    # Training Loop
    # ──────────────────────────────────────────────────────────────────────
    def train(self):
        if self.is_multimodal:
            step_fn = self._step_multimodal
        elif self.cfg.framework == "continuous_score":
            step_fn = self._step_continuous
        else:
            step_fn = self._step_discrete

        # IMPORTANT: callbacks may include run_on_all_ranks=True (e.g. offline entropy)
        self._run_callbacks("on_train_begin")

        # Lightning-style sanity: run callbacks once before training
        self._maybe_run_sanity()

        target_total_steps = int(getattr(self.cfg.optim, "total_steps", 0))
        stop_training = False

        try:
            for epoch in range(self.start_epoch, self.cfg.train.epochs):
                # Critical for DDP: shuffle data differently each epoch
                if self.ddp_active and hasattr(self.train_loader.sampler, "set_epoch"):
                    self.train_loader.sampler.set_epoch(epoch)
                if hasattr(self.train_loader.batch_sampler, "set_epoch"):
                    self.train_loader.batch_sampler.set_epoch(epoch)

                # Multimodal: advance the collator epoch too, and honour a
                # mid-epoch resume cursor for exact next-batch continuation.
                resume_cursor = 0
                if self.is_multimodal:
                    if getattr(self, "_mm_resume_cursor", 0) and epoch == getattr(
                        self, "_mm_resume_epoch", -1
                    ):
                        resume_cursor = int(self._mm_resume_cursor)
                        self._mm_resume_cursor = 0  # consume once
                    coll = getattr(self.train_loader, "collate_fn", None)
                    if coll is not None and hasattr(coll, "set_epoch"):
                        coll.set_epoch(epoch)
                    bs = getattr(self.train_loader, "batch_sampler", None)
                    if resume_cursor:
                        if bs is not None and hasattr(bs, "start_batch"):
                            bs.start_batch = resume_cursor
                        if coll is not None and hasattr(coll, "_counter"):
                            coll._counter = resume_cursor
                    # Advance the replay loader's sampler/collator epoch too, so
                    # replay does not draw the same frozen batch cycle every epoch.
                    if self.replay_loader is not None:
                        rbs = getattr(self.replay_loader, "batch_sampler", None)
                        if rbs is not None and hasattr(rbs, "set_epoch"):
                            rbs.set_epoch(epoch)
                        rcoll = getattr(self.replay_loader, "collate_fn", None)
                        if rcoll is not None and hasattr(rcoll, "set_epoch"):
                            rcoll.set_epoch(epoch)
                        self._replay_iter = None  # rebuild iterator for the new epoch
                self._epoch_batches_done = resume_cursor

                self.model.train()
                train_loss = 0.0
                num_train_batches = 0

                # Disable progress bar on workers
                pbar = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch+1}/{self.cfg.train.epochs}",
                    leave=True,
                    disable=not self.is_master,
                )

                for batch in pbar:
                    # Safety guard in case we resumed exactly at target_total_steps
                    if target_total_steps > 0 and self.global_step >= target_total_steps:
                        stop_training = True
                        break

                    if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                        torch.compiler.cudagraph_mark_step_begin()

                    # Token-budgeted sequence-only replay: with probability
                    # replay_fraction pull a sequence-only batch (same dict format)
                    # so paired training does not forget the warm-started sequence
                    # model (plan 6.6 step 3). Decision is deterministic per step
                    # for exact resume.
                    did_replay = False
                    if self._mm_should_replay():
                        batch = self._mm_next_replay_batch()
                        did_replay = True

                    x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
                    loss = step_fn(x0, is_train=True)

                    # global_step counts OPTIMIZER steps (effective batches), so it
                    # only advances on an accumulation boundary. Non-multimodal
                    # step fns leave _did_optim_step True, so they advance every
                    # batch as before. The sampler cursor (_epoch_batches_done)
                    # advances every micro-batch for exact-next-batch resume.
                    if self._did_optim_step:
                        self.global_step += 1
                    train_loss += loss
                    num_train_batches += 1
                    # Per-epoch batch cursor (resume-aware; may start > 0).
                    self._epoch_batches_done += 1

                    # ----------------------------------------------------------
                    # PATCH: refresh online entropy schedule during training
                    # ----------------------------------------------------------
                    if (
                        self.cfg.framework == "continuous_score"
                        and self.entropy_compute
                        and (not self.entropy_offline_enabled)
                        # The multimodal step now fills the entropy buffer too, so
                        # both paths refresh the schedule. _did_optim_step keeps this
                        # to one recompute per optimizer step under grad accumulation
                        # (and stays True on the continuous path, unchanged there).
                        and self._did_optim_step
                    ):
                        update_every = int(getattr(self.cfg.train, "entropy_update_every_steps", 2000))
                        if update_every > 0 and (self.global_step % update_every == 0):
                            self._recompute_entropy_from_buffer()

                    # Interval/Resume checkpointing
                    self._maybe_save_resume_ckpt(epoch)
                    self._maybe_save_interval_ckpt(epoch)

                    if self.is_master:
                        pbar.set_postfix(loss=f"{loss:.4f}")

                        # TB logging throttled (HPC-friendly)
                        # Running (EMA) train loss. The per-step diffusion loss is
                        # very noisy (a fresh random sigma each step swings the
                        # EDM-weighted value by an order of magnitude), so this
                        # smoothed curve is what to read for the training trend;
                        # loss/iter_train keeps the raw per-step value.
                        self._loss_ema = (
                            loss
                            if self._loss_ema is None
                            else 0.98 * self._loss_ema + 0.02 * loss
                        )

                        if self.tb_scalar_every_steps > 0 and (self.global_step % self.tb_scalar_every_steps == 0):
                            self.writer.add_scalar("loss/iter_train", loss, self.global_step)
                            self.writer.add_scalar("loss/iter_train_smooth", self._loss_ema, self.global_step)
                            lr = self.opt.param_groups[0]["lr"]
                            self.writer.add_scalar("learning_rate", lr, self.global_step)
                            self.writer.add_scalar("ema/decay", self.ema.decay, self.global_step)

                            self._log_wandb(
                                {
                                    "loss/iter_train": loss,
                                    "learning_rate": lr,
                                }
                            )

                            # Per-modality diagnostics for the paired path.
                            if self.is_multimodal and self._last_mm_components:
                                mm = {
                                    f"multimodal/{k}": v
                                    for k, v in self._last_mm_components.items()
                                }
                                mm["multimodal/replay_step"] = float(did_replay)
                                if self._trunk_group_active:
                                    mm["multimodal/trunk_lr"] = float(
                                        self.opt.param_groups[1]["lr"]
                                    )
                                for k, v in mm.items():
                                    self.writer.add_scalar(k, v, self.global_step)
                                self._log_wandb(mm)

                        # Optional step-based sync (usually keep 0 on HPC)
                        if (
                            self.tb is not None
                            and self.tb_sync_every_steps > 0
                            and (self.global_step % self.tb_sync_every_steps == 0)
                        ):
                            self.tb.maybe_sync(step=self.global_step, epoch=epoch)

                    if target_total_steps > 0 and self.global_step >= target_total_steps:
                        stop_training = True
                        break
                    steps_per_epoch = int(getattr(self.cfg.train, "steps_per_epoch", 0))
                    # Use the resume-aware cursor so a mid-epoch resume finishes
                    # the epoch at the original boundary instead of overshooting.
                    if steps_per_epoch > 0 and self._epoch_batches_done >= steps_per_epoch:
                        break

                # If we did not process any batch in this epoch, stop cleanly
                if num_train_batches == 0:
                    if self.is_master:
                        print(f"Reached target total_steps={target_total_steps}. Stopping training.")
                    break

                # Average losses using the actual number of processed batches
                avg_train_loss = train_loss / max(1, num_train_batches)

                # Validation returns the legacy EDM loss plus optional multimodal
                # diagnostics. Configured selection metrics drive best.pt without
                # changing the historical loss/epoch_val series.
                avg_val_loss, val_metrics = self._validate_epoch(step_fn)
                checkpoint_cfg = getattr(self.cfg.train, "checkpointing", None)
                checkpoint_metric = str(
                    getattr(checkpoint_cfg, "metric", "edm_loss")
                    if checkpoint_cfg is not None else "edm_loss"
                ).lower()
                if checkpoint_metric == "balanced_mse":
                    if "validation/balanced_mse" not in val_metrics:
                        raise RuntimeError(
                            "checkpointing.metric=balanced_mse requires multimodal validation diagnostics"
                        )
                    selection_metric = val_metrics["validation/balanced_mse"]
                elif checkpoint_metric in {"edm", "edm_loss", "loss/epoch_val"}:
                    selection_metric = avg_val_loss
                else:
                    raise ValueError(f"Unknown checkpointing.metric={checkpoint_metric!r}")
                early_stop_requested = self._update_early_stopping(selection_metric)

                if self.is_master:
                    self.writer.add_scalar("loss/epoch_train", avg_train_loss, self.global_step)
                    self.writer.add_scalar("loss/epoch_val", avg_val_loss, self.global_step)
                    self.writer.add_scalar("training/epoch_index", epoch, self.global_step)
                    self.writer.add_scalar("validation/selection_score", selection_metric, self.global_step)
                    if self.device.type == "cuda":
                        peak_gib = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                        self.writer.add_scalar("system/peak_cuda_memory_gib", peak_gib, self.global_step)
                        print(f"Peak CUDA memory: {peak_gib:.2f} GiB")
                    for key, value in sorted(val_metrics.items()):
                        self.writer.add_scalar(key, value, self.global_step)

                    print(
                        f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
                        f"Val Loss = {avg_val_loss:.4f}, "
                        f"Selection ({checkpoint_metric}) = {selection_metric:.6f}"
                    )

                    self._log_wandb(
                        {
                            "loss/epoch_train": avg_train_loss,
                            "loss/epoch_val": avg_val_loss,
                            "validation/selection_score": selection_metric,
                            **val_metrics,
                            "epoch": epoch,
                        }
                    )

                # Run callbacks (some may request running on all ranks)
                self._run_callbacks("on_epoch_end", epoch)

                if self.is_master:
                    self._save_ckpt(epoch, selection_metric)

                # Flush TB buffers and sync staged logs -> run_dir (master only)
                if self.is_master and self.tb is not None:
                    self.tb.flush()
                    if self.tb_sync_every_epochs > 0 and ((epoch + 1) % self.tb_sync_every_epochs == 0):
                        self.tb.maybe_sync(step=self.global_step, epoch=epoch)

                # NOTE:
                # Old epoch-end entropy recomputation removed on purpose.
                # The schedule is now updated online inside the batch loop.

                # Barrier to keep epochs roughly synced (good practice)
                if self.ddp_active:
                    dist.barrier()

                if stop_training or early_stop_requested:
                    if self.is_master:
                        if early_stop_requested:
                            print(
                                "Early stopping: validation did not improve beyond "
                                "min_delta for the configured patience."
                            )
                        else:
                            print(
                                f"Reached target total_steps={target_total_steps}. "
                                "Stopping training."
                            )
                    break

        except KeyboardInterrupt:
            if self.is_master:
                print("\n⛔ Training interrupted by user (KeyboardInterrupt). Attempting clean shutdown...")
            raise

        finally:
            # Always attempt to flush/sync/close TB even on exceptions / preemption
            if self.is_master and self.tb is not None:
                try:
                    # (optional) best-effort final flush/sync before TBManager.close()
                    self.tb.flush()
                    self.tb.maybe_sync(step=self.global_step, epoch=max(self.start_epoch, 0))
                except Exception:
                    pass
                try:
                    self.tb.close()
                except Exception:
                    pass

            # Always finish wandb on master if it was enabled
            if self.use_wandb and wandb is not None and self.is_master:
                try:
                    wandb.finish()
                except Exception:
                    pass

    # ──────────────────────────────────────────────────────────────────────
    # Entropy schedule helpers (continuous) — wrappers
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def entropy_fifo_push(self, sigmas: torch.Tensor, metric: torch.Tensor) -> None:
        """
        Push (sigma, metric) pairs into the FIFO ring buffer on CPU.
        sigmas: [B] or [B,1]
        metric: [B] or [B,1]
        """
        s = sigmas.detach().flatten().to(dtype=torch.float32, device="cpu")
        m = metric.detach().flatten().to(dtype=torch.float32, device="cpu")

        n = s.numel()
        if n == 0:
            return

        cap = self.entropy_buffer_size
        ptr = self._entropy_buf_ptr

        if n >= cap:
            # keep only last cap items
            s = s[-cap:]
            m = m[-cap:]
            n = cap

        end = ptr + n
        if end <= cap:
            self._entropy_sig_buf[ptr:end] = s
            self._entropy_metric_buf[ptr:end] = m
        else:
            k = cap - ptr
            self._entropy_sig_buf[ptr:cap] = s[:k]
            self._entropy_metric_buf[ptr:cap] = m[:k]
            r = n - k
            self._entropy_sig_buf[0:r] = s[k:]
            self._entropy_metric_buf[0:r] = m[k:]

        self._entropy_buf_ptr = (ptr + n) % cap
        self._entropy_buf_len = min(cap, self._entropy_buf_len + n)

    def _entropy_metric_from_logits(
        self,
        logits_2d: torch.Tensor,  # [B, S] or [B, S']
        target_2d: torch.Tensor,  # [B, S] or [B, S']
        mask: torch.Tensor | None = None,  # [B, S] float/bool or None
    ) -> torch.Tensor:
        """
        Per-example mean squared error in probability space (unweighted),
        optionally masked (suffix-only etc).

        Returns: [B]
        """
        # logits -> probs
        probs = torch.sigmoid(logits_2d.float())
        sq_err = (probs - target_2d.float()) ** 2  # [B,S]

        if mask is None:
            return sq_err.mean(dim=1)

        # accept bool or float masks
        if mask.dtype == torch.bool:
            w = mask.to(dtype=sq_err.dtype)
        else:
            w = mask.to(dtype=sq_err.dtype)

        # weighted mean per example; avoid div-by-zero if a row is fully masked
        num = (sq_err * w).sum(dim=1)               # [B]
        den = w.sum(dim=1).clamp_min(1.0)           # [B]
        return num / den


    def _draw_sigma(self, bsz: int) -> torch.Tensor:
        return self.entropy_ctrl.draw_sigma(bsz)

    def _entropy_paths(self):
        return self.entropy_ctrl.entropy_paths()

    def _save_entropy_tables(self, pdf, cdf, sigmas, edges=None):
        return self.entropy_ctrl.save_entropy_tables(pdf, cdf, sigmas, edges)

    def _load_entropy_tables_if_any(self):
        return self.entropy_ctrl.load_entropy_tables_if_any()

    def _entropy_gamma(self) -> float:
        return self.entropy_ctrl.entropy_gamma()

    def _update_entropy_buffer(self, sigma: torch.Tensor, entropy_metric: torch.Tensor):
        return self.entropy_ctrl.update_entropy_buffer(sigma, entropy_metric)

    def _fit_lognormal_to_entropy_profile(self):
        return self.entropy_ctrl.fit_lognormal_to_entropy_profile()

    def _recompute_entropy_from_buffer(self):
        return self.entropy_ctrl.recompute_entropy_from_buffer()

    def _sample_entropy_sigma(self, bsz: int) -> torch.Tensor:
        return self.entropy_ctrl.sample_entropy_sigma(bsz)

    def _sample_entropy_sigma_lognormal(self, bsz: int) -> torch.Tensor:
        return self.entropy_ctrl.sample_entropy_sigma_lognormal(bsz)
