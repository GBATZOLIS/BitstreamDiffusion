from __future__ import annotations

import os
# Fix for some torch.compile interactions with CUDA Graphs
os.environ.setdefault("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", "1")

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

        if self.is_master and logging_cfg is not None and getattr(logging_cfg, "use_wandb", False):
            if WANDB_AVAILABLE:
                self.use_wandb = True
                os.environ["WANDB_PYTORCH_DISABLE"] = "true"
                os.environ["WANDB_DISABLE_GRADIENTS"] = "true"

                cfg_dict = _cfg_to_dict(cfg)
                project = getattr(logging_cfg, "project", "diffusion")
                entity = getattr(logging_cfg, "entity", None)
                mode = getattr(logging_cfg, "mode", "online")
                run_name = cfg.experiment

                wandb.init(
                    project=project,
                    entity=entity,
                    id=logging_cfg.run_id,
                    resume="allow",
                    name=run_name,
                    config=cfg_dict,
                    dir=str(self.run_dir),
                    mode=mode,
                )
            else:
                print("⚠️  wandb not installed, skipping W&B logging.")

        # ── checkpoint config / bookkeeping (resolved + backward-compatible) ─────
        ck = _resolve_checkpointing_cfg(cfg)

        self.save_last = bool(ck["save_last"])
        self.save_top_k = int(ck["save_top_k"])
        self.checkpoint_mode = str(ck["mode"])

        self.best_metric = math.inf if self.checkpoint_mode == "min" else -math.inf
        self.best_ckpts: List[dict] = []

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
        raw_train_loader, raw_val_loader, _ = get_dataloaders(cfg)
        dl_kw = _dataloader_kwargs(cfg)

        if self.ddp_active:
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
            base = DDP(
                base,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
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

        # ── opt / ema / amp ────────────────────────────────────────────────
        self.ema = EMA(self.model, decay=cfg.train.ema_decay)
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

            # SigmaDataEstimator / plotting are master-only (no collectives, pure logging)
            if self.is_master:
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


            # VLB (All ranks - critical for DDP synchronization)
            vlb_cfg = getattr(cfg.train, "vlb", None)
            if vlb_cfg is not None and bool(getattr(vlb_cfg, "enabled", False)):
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
    # Basic W&B logging helper
    # ──────────────────────────────────────────────────────────────────────
    def _log_wandb(self, data: dict):
        if not self.use_wandb or not self.is_master:
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
        ckpt = torch.load(path, map_location="cpu")
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
        ckpt = torch.load(path, map_location="cpu")
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
    
    # -----------------------------------------------------------------------------
    # Trainer method: full step_continuous
    # -----------------------------------------------------------------------------
    def _step_continuous(self, x0, is_train: bool, batch_prefix_mask=None):
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

        # Per-example prefix mask supplied by the batch (task datasets: Sudoku /
        # TinyGSM). It takes precedence over the config-derived fixed/random
        # prompt length, enabling variable-length per-example prompts.
        batch_pm = None
        if batch_prefix_mask is not None:
            batch_pm = batch_prefix_mask.to(self.device, non_blocking=True).view(B, -1).bool()
            if batch_pm.shape[1] != S:
                raise ValueError(
                    f"batch_prefix_mask has {batch_pm.shape[1]} positions but x0 has S={S}"
                )

        if batch_pm is not None:
            prefix_mask = batch_pm
            cL_pos = None
            cond_enabled = True
        elif not cond_enabled_cfg:
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
            # Defaults are clean-prefix + suffix-only loss, which is exactly the
            # task (prompt/solution) regime; cfg.cond may still override.
            noise_prefix = bool(getattr(cond_cfg, "noise_prefix", False)) if cond_cfg is not None else False
            suffix_only_loss = bool(getattr(cond_cfg, "loss_on_suffix_only", True)) if cond_cfg is not None else True
            p_uncond = float(getattr(cond_cfg, "p_uncond", 0.0)) if cond_cfg is not None else 0.0
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
            self.ema.update(self.model)

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

    def _step_discrete(self, x0, is_train: bool, batch_prefix_mask=None):
        # batch_prefix_mask is accepted for interface parity with _step_continuous.
        # The task experiments use the continuous-score framework; discrete
        # conditioning still flows through the config-derived path below.
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
            self.ema.update(self.model)

        return loss.item()

    @torch.compiler.disable
    @torch.no_grad()
    def _unpack_batch(self, batch):
        """Return (x0, batch_prefix_mask).

        Supports three batch formats:
          - plain tensor                         -> (tensor, None)        [LM1B / OWT]
          - (list/tuple)                         -> (batch[0], None)
          - dict with keys 'x0' [, 'prefix_mask']-> (x0, prefix_mask)     [task datasets]

        batch_prefix_mask, when present, is a per-example [B, S] bool mask marking
        the conditioned prompt region (clamped clean, excluded from the loss).
        """
        if isinstance(batch, dict):
            return batch["x0"], batch.get("prefix_mask", None)
        if isinstance(batch, (list, tuple)):
            return batch[0], None
        return batch, None

    def _validate_epoch(self, step_fn):
        self.model.eval()
        self.ema.apply(self.model)

        local_loss = torch.tensor(0.0, device=self.device)
        local_count = torch.tensor(0.0, device=self.device)

        pbar = tqdm(self.val_loader, desc="Validating", leave=False, disable=not self.is_master)

        for batch in pbar:
            x0, batch_prefix_mask = self._unpack_batch(batch)
            loss = step_fn(x0, is_train=False, batch_prefix_mask=batch_prefix_mask)
            local_loss += loss
            local_count += 1.0

        if self.ddp_active:
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

        avg_loss = (local_loss / local_count).item()

        self.ema.restore(self.model)
        return avg_loss

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

        return {
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
        }

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
            torch.save(state, new_best_path)
            torch.save(state, self._checkpoint_path("best"))

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
        torch.save(state, path)

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
    # Training Loop
    # ──────────────────────────────────────────────────────────────────────
    def train(self):
        step_fn = self._step_continuous if self.cfg.framework == "continuous_score" else self._step_discrete

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

                    x0, batch_prefix_mask = self._unpack_batch(batch)
                    loss = step_fn(x0, is_train=True, batch_prefix_mask=batch_prefix_mask)

                    self.global_step += 1
                    train_loss += loss
                    num_train_batches += 1

                    # ----------------------------------------------------------
                    # PATCH: refresh online entropy schedule during training
                    # ----------------------------------------------------------
                    if (
                        self.cfg.framework == "continuous_score"
                        and self.entropy_compute
                        and (not self.entropy_offline_enabled)
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
                        if self.tb_scalar_every_steps > 0 and (self.global_step % self.tb_scalar_every_steps == 0):
                            self.writer.add_scalar("loss/iter_train", loss, self.global_step)
                            lr = self.opt.param_groups[0]["lr"]
                            self.writer.add_scalar("learning_rate", lr, self.global_step)

                            self._log_wandb(
                                {
                                    "loss/iter_train": loss,
                                    "learning_rate": lr,
                                }
                            )

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

                # If we did not process any batch in this epoch, stop cleanly
                if num_train_batches == 0:
                    if self.is_master:
                        print(f"Reached target total_steps={target_total_steps}. Stopping training.")
                    break

                # Average losses using the actual number of processed batches
                avg_train_loss = train_loss / max(1, num_train_batches)

                # Validated loss (synchronized)
                avg_val_loss = self._validate_epoch(step_fn)

                if self.is_master:
                    self.writer.add_scalar("loss/epoch_train", avg_train_loss, self.global_step)
                    self.writer.add_scalar("loss/epoch_val", avg_val_loss, self.global_step)
                    self.writer.add_scalar("training/epoch_index", epoch, self.global_step)

                    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

                    self._log_wandb(
                        {
                            "loss/epoch_train": avg_train_loss,
                            "loss/epoch_val": avg_val_loss,
                            "epoch": epoch,
                        }
                    )

                # Run callbacks (some may request running on all ranks)
                self._run_callbacks("on_epoch_end", epoch)

                if self.is_master:
                    self._save_ckpt(epoch, avg_val_loss)

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

                if stop_training:
                    if self.is_master:
                        print(f"Reached target total_steps={target_total_steps}. Stopping training.")
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
