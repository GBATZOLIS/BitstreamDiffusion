# evaluation/utils.py
from __future__ import annotations

import copy
import importlib.util
import math
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, ChainDataset

from data import get_loader
from utils.ema import EMA
from torch.cuda.amp import autocast

# Continuous
from diffusion.continuous.processes import ContinuousForwardProcess
from diffusion.continuous.samplers import HeunSampler, DDIMSampler

# Discrete
from diffusion.discrete.processes import DiscreteForwardProcess
from diffusion.discrete.samplers import TweedieTauLeapingSampler, EulerRateSampler

from evaluation.distributed import (
    get_rank_world_size,
    shard_count,
    all_gather_int,
    broadcast_optional_tensor,
    gather_varlen_firstdim_to_rank0,
)

import torch.distributed as dist

def resolve_eval_amp(device: torch.device, cfg: Optional[Any] = None) -> Tuple[bool, torch.dtype]:
    """
    Resolve AMP settings for eval sampling.
    Priority:
      1) cfg.evaluation.use_amp + cfg.evaluation.amp_dtype
      2) cfg.train.use_fp16 (legacy) + cfg.train.amp_dtype
      3) default: auto bf16-if-available else fp16
    """
    if device.type != "cuda":
        return False, torch.float32

    # defaults
    use_amp = True
    amp_dtype_str = "auto"

    # ---- 1) evaluation overrides ----
    if cfg is not None and hasattr(cfg, "evaluation"):
        ev = cfg.evaluation
        if hasattr(ev, "use_amp"):
            use_amp = bool(getattr(ev, "use_amp"))
        if hasattr(ev, "amp_dtype"):
            amp_dtype_str = str(getattr(ev, "amp_dtype")).lower()

    # ---- 2) training fallback ----
    if cfg is not None and hasattr(cfg, "train"):
        tr = cfg.train
        # only use train flags if eval didn't specify
        if (cfg is None) or (not hasattr(cfg, "evaluation")) or (not hasattr(cfg.evaluation, "use_amp")):
            if hasattr(tr, "use_fp16"):
                use_amp = bool(getattr(tr, "use_fp16"))
            elif hasattr(tr, "use_amp"):
                use_amp = bool(getattr(tr, "use_amp"))
        if (cfg is None) or (not hasattr(cfg, "evaluation")) or (not hasattr(cfg.evaluation, "amp_dtype")):
            if hasattr(tr, "amp_dtype"):
                amp_dtype_str = str(getattr(tr, "amp_dtype")).lower()

    if not use_amp:
        return False, torch.float32

    if amp_dtype_str in {"bf16", "bfloat16"}:
        return True, torch.bfloat16
    if amp_dtype_str in {"fp16", "float16"}:
        return True, torch.float16
    if amp_dtype_str in {"fp32", "float32", "none", "no"}:
        return False, torch.float32

    # auto
    return True, (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)


@torch.no_grad()
def warmup_continuous_sampler(
    sampler: Any,
    *,
    B: int,
    seq_len: int,
    num_steps: int,
    schedule: str,
    entropy_run_dir: Optional[Path],
    entropic_blend_alpha: float,
    sigma_min_override: float,
    sigma_max_override: Optional[float] = None,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    cond_kwargs: Optional[dict] = None,
) -> None:
    """
    Small-step warmup for torch.compile / kernel caching:
      - Runs sampler.sample with a small batch and few steps.
      - No decoding / logging; return_probs=False.
      - Passes through sigma_min_override AND sigma_max_override so warmup matches real sampling.
    """
    B = int(B)
    if B <= 0:
        return

    steps = max(1, int(num_steps))
    cond_kwargs = cond_kwargs or {}

    # Normalize entropy_run_dir
    if entropy_run_dir is not None:
        entropy_run_dir = Path(str(entropy_run_dir))

    # Synchronize for cleaner timing / to avoid overlap surprises
    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, enabled=bool(use_amp), dtype=amp_dtype):
            _ = sampler.sample(
                B,
                seq_len,
                schedule=str(schedule),
                num_steps=steps,
                entropic_blend_alpha=float(entropic_blend_alpha),
                entropy_run_dir=entropy_run_dir,
                sigma_min_override=float(sigma_min_override),
                sigma_max_override=(float(sigma_max_override) if sigma_max_override is not None else None),
                return_probs=False,
                progress=False,  # safe if your sampler supports it; remove if it doesn't
                **cond_kwargs,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

def _tile_rows_to_len(x: torch.Tensor, target: int) -> torch.Tensor:
    """
    If x has fewer than target rows, repeat rows to reach target.
    Keeps x[:target] if longer.
    """
    target = int(target)
    if x.size(0) == target:
        return x
    if x.size(0) > target:
        return x[:target].contiguous()
    if x.size(0) == 0:
        raise ValueError("Cannot tile an empty prefix tensor.")
    reps = (target + x.size(0) - 1) // x.size(0)
    return x.repeat(reps, 1)[:target].contiguous()

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
@contextmanager
def temp_cfg_overrides(cfg_obj, updates: dict):
    """
    Temporarily override attributes on a (ml_collections) ConfigDict-like object.
    Restores old values after the context. If an attribute didn't exist, it is deleted.
    """
    sentinel = object()
    old = {}
    for k, v in updates.items():
        old[k] = getattr(cfg_obj, k, sentinel)
        setattr(cfg_obj, k, v)
    try:
        yield
    finally:
        for k, prev in old.items():
            if prev is sentinel:
                try:
                    delattr(cfg_obj, k)
                except Exception:
                    pass
            else:
                setattr(cfg_obj, k, prev)


def unwrap_all(model: torch.nn.Module) -> torch.nn.Module:
    """Peel DDP and torch.compile wrappers until reaching the real module."""
    m = model
    while True:
        changed = False
        if hasattr(m, "module"):     # DDP
            m = m.module
            changed = True
        if hasattr(m, "_orig_mod"):  # torch.compile
            m = m._orig_mod
            changed = True
        if not changed:
            break
    return m


def load_config(path: str):
    spec = importlib.util.spec_from_file_location("config", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.get_config()


def get_fid_cfg(cfg):
    return getattr(getattr(cfg, "evaluation", None), "fid", None)


# -----------------------------------------------------------------------------
# Checkpoint loading (identical logic)
# -----------------------------------------------------------------------------
def load_checkpoint(
    model: torch.nn.Module,
    ema: EMA,
    ckpt_path: Path,
    device: torch.device,
    *,
    apply_ema: bool = True,
):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading checkpoint: {ckpt_path}")

    # Our checkpoints contain non-tensor objects (EMA state, optimizer, etc.) so
    # weights_only=True would refuse to load them. Released artefacts are
    # produced by this repo only — we explicitly opt in to full unpickling.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_to_load = unwrap_all(model)

    state = ckpt["model"]
    clean_state = {}
    for k, v in state.items():
        k = k.replace("_orig_mod.", "")
        if k.startswith("module."):
            k = k[7:]
        clean_state[k] = v

    try:
        model_to_load.load_state_dict(clean_state, strict=True)
    except RuntimeError:
        model_to_load.load_state_dict(clean_state, strict=False)

    if apply_ema and ("ema" in ckpt) and (ckpt["ema"] is not None):
        ema.load_state_dict(ckpt["ema"])
        ema.to(device)
        ema.apply(model)
        print("✓ EMA weights applied.")
    else:
        print("⚠️ No EMA found in checkpoint, using raw weights.")


# -----------------------------------------------------------------------------
# FID real-loader helpers (identical logic)
# -----------------------------------------------------------------------------
def resolve_entropy_run_dir(cfg, fid_cfg) -> Path:
    """
    Decide where the entropy profile (bin edges / pdf) lives for entropic schedule sampling.
    Priority:
      1) cfg.evaluation.fid.entropy_run_dir (if set)
      2) inferred from cfg.evaluation.checkpoint_path
    """
    if fid_cfg is not None:
        maybe = getattr(fid_cfg, "entropy_run_dir", None)
        if maybe not in (None, "", ".", "None"):
            return Path(str(maybe)).expanduser().resolve()

    ckpt_path = Path(cfg.evaluation.checkpoint_path).expanduser().resolve()
    return ckpt_path.parent.parent


def make_concat_real_loader_from_splits(cfg_real, splits, *, batch_size=None):
    loaders = [get_loader(cfg_real, split=s) for s in splits]
    datasets = [ld.dataset for ld in loaders]
    ds = ConcatDataset(datasets)

    bs = int(batch_size or getattr(cfg_real.evaluation, "num_samples", cfg_real.train.batch_size))
    num_workers = int(getattr(cfg_real.data, "num_workers", 4))
    pin_memory = bool(getattr(cfg_real.data, "pin_memory", True))
    prefetch_factor = getattr(cfg_real.data, "prefetch_factor", None)

    dl_kwargs = dict(
        dataset=ds,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0 and prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(**dl_kwargs)


def real_train_loader_for_fid(cfg_real, fid_cfg):
    """
    Returns a loader for the "real TRAIN reference" used in FID.

    Fixes CIFAR-10Bits 'train_size=50000 clamps to 49999' issue without touching dataset:
      - if you ask for 50000, use train+val concat as real reference.
    """
    real_train_split = None
    if fid_cfg is not None:
        real_train_split = getattr(fid_cfg, "real_train_split", None)
        if isinstance(real_train_split, str):
            real_train_split = real_train_split.lower().strip()

    dataset_name = str(getattr(cfg_real.data, "dataset", "")).lower()
    want_n = int(getattr(fid_cfg, "real_train_size", 0) or 0)

    auto_need_concat = (dataset_name in {"cifar10bits", "cifar10"} and want_n >= 50_000)

    if real_train_split in {"train+val", "train_val", "trainval"}:
        return make_concat_real_loader_from_splits(cfg_real, ["train", "val"])
    if auto_need_concat:
        return make_concat_real_loader_from_splits(cfg_real, ["train", "val"])

    return get_loader(cfg_real, split="train")


def unwrap_dataset(ds):
    """
    Peel common dataset wrappers to reach a representative base dataset.

    Handles:
      - Subset -> .dataset
      - ConcatDataset -> .datasets[0]
      - ChainDataset -> first dataset
      - Generic wrappers exposing .dataset
    """
    while True:
        if isinstance(ds, Subset):
            ds = ds.dataset
            continue
        if isinstance(ds, ConcatDataset):
            # assume homogeneous datasets; pick a representative
            ds = ds.datasets[0]
            continue
        if isinstance(ds, ChainDataset):
            # ChainDataset stores an iterable of datasets
            ds = list(ds.datasets)[0]
            continue
        if hasattr(ds, "dataset"):  # many wrappers use this
            ds = ds.dataset
            continue
        break
    return ds

# -----------------------------------------------------------------------------
# Flatten-order helpers (identical logic)
# -----------------------------------------------------------------------------
def extract_dataset_attr(ds, name):
    base = unwrap_dataset(ds)
    return getattr(base, name, None)


def collect_real_bit_seqs_for_external_ppl(
    loader,
    *,
    num_samples: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Collect num_samples REAL sequences from `loader` and return [N, seq_len] bits as torch.long in {0,1}.

    Supports:
      - Text8 semantic: long bits already in {0,1}
      - WikiText semantic: float bits in {0.0,1.0}
      - Any other: thresholds floats, (x!=0) for ints.

    Enforces exact seq_len by truncating/padding on the right.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0 (got {seq_len})")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0 (got {num_samples})")

    seqs = []
    got = 0

    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch

        # Flatten to [B,S]
        if x.dim() == 1:
            x = x.view(1, -1)
        elif x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Convert to bits long {0,1}
        if x.is_floating_point():
            x_bits = (x > 0.5).to(torch.long)
        else:
            x_bits = (x != 0).to(torch.long)

        # Enforce exact seq_len
        if x_bits.size(1) > seq_len:
            x_bits = x_bits[:, :seq_len]
        elif x_bits.size(1) < seq_len:
            x_bits = F.pad(x_bits, (0, seq_len - x_bits.size(1)), value=0)

        seqs.append(x_bits.cpu())
        got += x_bits.size(0)
        if got >= num_samples:
            break

    if not seqs:
        raise RuntimeError("Loader yielded no batches; cannot compute real baseline external ppl.")

    return torch.cat(seqs, dim=0)[:num_samples]  # [N, seq_len]

@torch.no_grad()
def seqs_to_images_for_eval(seqs: torch.Tensor, dataset) -> torch.Tensor:
    """
    Convert flattened bit-sequences into images for FID evaluation.

    Priority:
      1) If the underlying dataset implements `reconstruct_from_bits(bits)`,
         use that (most correct / supports arbitrary preprocessing).
      2) Otherwise try to reconstruct from generic metadata:
         - shape_hw (H,W)
         - channels
         - bits_per_pixel (bpp)
         - invperm (pixel permutation inverse, length H*W)

    Accepts `dataset` possibly wrapped in Subset/ConcatDataset/ChainDataset/etc.
    """
    if not isinstance(seqs, torch.Tensor):
        raise TypeError(f"seqs must be a torch.Tensor, got {type(seqs)}")
    if seqs.dim() != 2:
        raise ValueError(f"seqs must be 2D [B,S], got shape {tuple(seqs.shape)}")

    B, S = seqs.shape  # IMPORTANT: define early, used in shape inference.

    # --- Best path: dataset-provided reconstruction ---
    base_ds = unwrap_dataset(dataset)
    reconstruct_fn = getattr(base_ds, "reconstruct_from_bits", None)
    if callable(reconstruct_fn):
        imgs = []
        for i in range(B):
            bits_i = seqs[i]
            if bits_i.is_floating_point():
                bits_i = (bits_i > 0.5).to(torch.float32)
            else:
                bits_i = (bits_i != 0).to(torch.float32)

            img_chw = reconstruct_fn(bits_i)

            if not isinstance(img_chw, torch.Tensor):
                raise TypeError(
                    "reconstruct_from_bits must return a torch.Tensor, "
                    f"got {type(img_chw)}"
                )
            if img_chw.dim() != 3:
                raise ValueError(
                    "reconstruct_from_bits must return [C,H,W], "
                    f"got shape {tuple(img_chw.shape)}"
                )

            imgs.append(img_chw.unsqueeze(0))

        out = torch.cat(imgs, dim=0).to(torch.float32)  # [B,C,H,W]

        # Keep within [0,1] for FID preprocessing
        return out.clamp(0.0, 1.0)

    # --- Generic fallback path ---
    invperm = extract_dataset_attr(dataset, "invperm")
    shape_hw = extract_dataset_attr(dataset, "shape_hw")
    bits_per_pixel = extract_dataset_attr(dataset, "bits_per_pixel")
    channels = extract_dataset_attr(dataset, "channels") or 1
    channels = int(channels)

    # If shape is unknown, infer it.
    if shape_hw is None:
        # If bits_per_pixel is known, infer pixels = S/bpp then H=W=sqrt(pixels).
        if bits_per_pixel is not None and int(bits_per_pixel) > 1:
            bpp = int(bits_per_pixel)
            if S % bpp != 0:
                raise ValueError(
                    f"Cannot infer image: sequence_len S={S} not divisible by bits_per_pixel={bpp}"
                )
            P = S // bpp  # number of pixels
            H = int(round(math.sqrt(P)))
            if H * H != P:
                raise ValueError(
                    f"Cannot infer square image from pixels={P} "
                    f"(S={S}, bits_per_pixel={bpp})."
                )
            shape_hw = (H, H)  # fall through to bpp decoding below
        else:
            # true 1-bit-per-pixel square fallback
            H = int(round(math.sqrt(S)))
            if H * H != S:
                raise ValueError(f"Cannot infer square image from flattened size S={S}.")
            inv = torch.arange(S, device=seqs.device, dtype=torch.long)
            imgs = seqs[:, inv].view(B, 1, H, H).to(torch.float32)
            return imgs.clamp(0.0, 1.0)

    H, W = int(shape_hw[0]), int(shape_hw[1])
    num_pixels = H * W

    # pixel inverse permutation (if provided)
    if invperm is None:
        inv = torch.arange(num_pixels, device=seqs.device, dtype=torch.long)
    else:
        inv = invperm.to(seqs.device).view(-1)
        if inv.numel() != num_pixels:
            raise ValueError(
                f"invperm has length {inv.numel()} but expected H*W={num_pixels} "
                f"for shape_hw=({H},{W})."
            )

    # If bits_per_pixel is unknown or 1, treat as 1-bit grayscale image.
    if bits_per_pixel is None or int(bits_per_pixel) == 1:
        if S != num_pixels:
            raise ValueError(
                f"Expected S==H*W for 1-bit images, got S={S}, H*W={num_pixels}."
            )
        seqs_reordered = seqs[:, inv]
        imgs = seqs_reordered.view(B, 1, H, W).to(torch.float32)
        return imgs.clamp(0.0, 1.0)

    # Otherwise decode packed pixel values from bits.
    bpp = int(bits_per_pixel)
    if S != num_pixels * bpp:
        raise ValueError(
            f"Expected S==H*W*bits_per_pixel, got S={S}, "
            f"H*W*bits_per_pixel={num_pixels*bpp}."
        )

    if bpp % channels != 0:
        raise ValueError(f"bits_per_pixel={bpp} is not divisible by channels={channels}.")

    bits_per_channel = bpp // channels

    # Binarize to uint8 bits
    if seqs.is_floating_point():
        bits = (seqs > 0.5).to(torch.uint8)
    else:
        bits = (seqs != 0).to(torch.uint8)

    # [B, H*W, bpp]
    bits = bits.view(B, num_pixels, bpp)

    # undo pixel permutation
    bits = bits[:, inv, :]

    # [B, H*W, C, bits_per_channel]
    bits = bits.view(B, num_pixels, channels, bits_per_channel)

    # pack bits (big-endian within channel)
    vals = torch.zeros(B, num_pixels, channels, dtype=torch.uint8, device=seqs.device)
    for k in range(bits_per_channel):
        shift = bits_per_channel - 1 - k
        vals |= (bits[..., k] & 1) << shift

    # [B, C, H, W] in [0,1]
    imgs = vals.view(B, H, W, channels).permute(0, 3, 1, 2).contiguous()
    imgs = imgs.to(torch.float32) / 255.0
    return imgs.clamp(0.0, 1.0)


class SeqToImageDataset(Dataset):
    """Wraps flattened sequence dataset -> [C,H,W] images for FID."""
    def __init__(self, base_ds):
        super().__init__()
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if isinstance(item, (tuple, list)):
            x = item[0]
        else:
            x = item

        if x.dim() > 1:
            return x.to(torch.float32)

        seq = x.view(1, -1)
        imgs = seqs_to_images_for_eval(seq, self.base)
        return imgs[0]


def make_image_loader_from_loader(loader):
    ds = SeqToImageDataset(loader.dataset)
    return DataLoader(
        ds,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=getattr(loader, "num_workers", 4),
        pin_memory=getattr(loader, "pin_memory", True),
        drop_last=False,
        persistent_workers=getattr(loader, "persistent_workers", False),
    )


# -----------------------------------------------------------------------------
# External-perplexity sampling helpers (identical logic)
# -----------------------------------------------------------------------------
def bits_preview(bits_row: torch.Tensor, *, max_bits: int = 128) -> str:
    b = bits_row.detach().cpu().to(torch.long).view(-1)
    s = "".join(str(int(x)) for x in b[:max_bits].tolist())
    if b.numel() > max_bits:
        s += "..."
    return f"{s} (len={b.numel()})"


def semantic_token_bit_chunks(bits_row: torch.Tensor, *, bits_per_token: int, max_tokens: int = 8) -> str:
    b = bits_row.detach().cpu().to(torch.long).view(-1)
    T = min(max_tokens, b.numel() // bits_per_token)
    chunks = []
    for t in range(T):
        chunk = b[t * bits_per_token : (t + 1) * bits_per_token]
        chunks.append("".join(str(int(x)) for x in chunk.tolist()))
    suffix = " ..." if (b.numel() // bits_per_token) > T else ""
    return " | ".join(chunks) + suffix


def infer_discrete_seq_len(cfg) -> int:
    if hasattr(cfg.data, "sequence_len"):
        return int(cfg.data.sequence_len)

    repr_mode = str(getattr(cfg.data, "representation", "tokens")).lower()
    if repr_mode == "tokens" and hasattr(cfg.data, "sequence_len_chars"):
        return int(cfg.data.sequence_len_chars)

    if hasattr(cfg.data, "sequence_len_tokens"):
        return int(cfg.data.sequence_len_tokens)

    raise ValueError(
        "Could not infer discrete sequence length. Please set cfg.data.sequence_len "
        "(or sequence_len_chars / sequence_len_tokens)."
    )

# -----------------------------------------------------------------------------
# FID warnings helper (kept here so run_eval.py stays short)
# -----------------------------------------------------------------------------
def warn_if_small_fid_n(fid_num_samples: int):
    if fid_num_samples < 50_000:
        warnings.warn(
            f"[FID] Using fid_num_samples={fid_num_samples}. "
            "Many papers report FID with 50k generated samples; "
            "smaller N is noisier / not directly comparable."
        )


def maybe_override_real_train_size(cfg, fid_cfg):
    cfg_real = cfg
    if fid_cfg is not None and getattr(fid_cfg, "real_train_size", None) is not None:
        cfg_real = copy.deepcopy(cfg)
        cfg_real.data.train_size = int(getattr(fid_cfg, "real_train_size"))
    return cfg_real

# -----------------------------------------------------------------------------
# Conditioning helpers (MUST MATCH callbacks/generation.py)
# -----------------------------------------------------------------------------
def _eval_cond_enabled(cfg) -> bool:
    cc = getattr(cfg, "cond", None)
    return bool(cc is not None and getattr(cc, "enabled", False))


def _eval_cond_len_bits(cfg, seq_len: int) -> int:
    if not _eval_cond_enabled(cfg):
        return 0
    bits_per_char = int(getattr(getattr(cfg, "data", object()), "bits_per_char", 1))
    cond_len_chars = int(getattr(getattr(cfg, "cond", object()), "cond_len_chars", 0))
    cL = int(cond_len_chars * bits_per_char)
    return max(0, min(cL, int(seq_len)))


@torch.no_grad()
def _eval_sample_prefixes_from_loader(
    loader: Any,
    *,
    num_samples: int,
    cL: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Identical logic to GenerationCallback._sample_prefixes_from_data(...)
    Returns float32 [B,cL] on device, or None.
    """
    B = int(num_samples)
    if loader is None or B <= 0 or cL <= 0:
        return None

    prefixes: List[torch.Tensor] = []
    n_collected = 0

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]

        # Flatten to [B,S]
        if batch.dim() == 3:
            if batch.size(1) == 1:
                batch = batch.squeeze(1)
            else:
                batch = batch.view(batch.size(0), -1)
        else:
            batch = batch.view(batch.size(0), -1)

        if batch.size(1) < cL:
            continue

        p = batch[:, :cL].to(device=device, dtype=torch.float32, non_blocking=True)
        prefixes.append(p)
        n_collected += p.size(0)
        if n_collected >= B:
            break

    if not prefixes:
        return None
    return torch.cat(prefixes, dim=0)[:B].contiguous()


def _eval_entropy_run_dir(cfg) -> Path:
    """
    Exactly matches GenerationCallback._entropy_run_dir():
      - prefer cfg.train.generation.entropy_ckpt_path
      - else fall back to cfg.evaluation.checkpoint_path
    """
    g = getattr(getattr(cfg, "train", object()), "generation", None)
    ckpt = getattr(g, "entropy_ckpt_path", None) if g is not None else None
    if ckpt is None:
        ckpt = getattr(getattr(cfg, "evaluation", object()), "checkpoint_path", None)
    if ckpt is None:
        raise ValueError("Cannot resolve entropy run dir: missing cfg.train.generation.entropy_ckpt_path and cfg.evaluation.checkpoint_path")

    ckpt_path = Path(str(ckpt)).expanduser().resolve()
    return ckpt_path.parent.parent


def _eval_guidance_scales(cfg) -> List[float]:
    """
    Eval-time guidance scales preference:
      1) cfg.evaluation.external_ppl.guidance_scales
      2) cfg.train.generation.guidance_scales
      3) cfg.evaluation.guidance_scale
    Always returns a list[float], deduped order-preserving.
    """
    # 1) evaluation override
    ev = getattr(cfg, "evaluation", None)
    if ev is not None:
        ext = getattr(ev, "external_ppl", None)
        if ext is not None and getattr(ext, "guidance_scales", None) is not None:
            gs = getattr(ext, "guidance_scales")
        else:
            gs = None
    else:
        gs = None

    # 2) fall back to train.generation.guidance_scales
    if gs is None:
        g = getattr(getattr(cfg, "train", object()), "generation", None)
        gs = getattr(g, "guidance_scales", None) if g is not None else None

    # 3) last fallback to scalar evaluation.guidance_scale
    if gs is None:
        v = float(getattr(getattr(cfg, "evaluation", object()), "guidance_scale", 0.0))
        return [v]

    # normalize to list[float]
    if isinstance(gs, (list, tuple)):
        out = [float(x) for x in gs]
    else:
        out = [float(gs)]

    # dedup
    seen = set()
    uniq: List[float] = []
    for v in out:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq


def _eval_terminal_sigmas(cfg) -> List[float]:
    """
    Terminal sigma preference:
      1) cfg.evaluation.external_ppl.terminal_sigmas   (best: external-PPL specific)
      2) cfg.train.generation.terminal_sigmas          (callback default)
      3) cfg.diffusion.continuous.sigma_min
    """
    # 1) evaluation override (external ppl specific)
    ev = getattr(cfg, "evaluation", None)
    if ev is not None:
        ext = getattr(ev, "external_ppl", None)
        terms = getattr(ext, "terminal_sigmas", None) if ext is not None else None
        if terms is not None:
            if isinstance(terms, (list, tuple)):
                return [float(s) for s in terms]
            return [float(terms)]

    # 2) generation callback config
    g = getattr(getattr(cfg, "train", object()), "generation", None)
    terms = getattr(g, "terminal_sigmas", None) if g is not None else None
    if terms is not None:
        if isinstance(terms, (list, tuple)):
            return [float(s) for s in terms]
        return [float(terms)]

    # 3) fallback
    s0 = float(getattr(getattr(cfg.diffusion, "continuous", object()), "sigma_min", 1e-3))
    return [s0]



def _eval_num_steps(cfg) -> int:
    """
    For external perplexity generation:
      1) cfg.evaluation.external_ppl.num_sampling_steps   (most specific)
      2) cfg.train.generation.num_sampling_steps          (matches callback)
      3) cfg.evaluation.num_sampling_steps                (legacy)
      4) fallback 400
    """
    ext = getattr(getattr(cfg, "evaluation", object()), "external_ppl", None)
    if ext is not None and hasattr(ext, "num_sampling_steps"):
        return int(getattr(ext, "num_sampling_steps"))

    g = getattr(getattr(cfg, "train", object()), "generation", None)
    if g is not None and hasattr(g, "num_sampling_steps"):
        return int(getattr(g, "num_sampling_steps"))

    return int(getattr(getattr(cfg, "evaluation", object()), "num_sampling_steps", 400))


def _eval_entropic_blend_alpha(cfg) -> float:
    g = getattr(getattr(cfg, "train", object()), "generation", None)
    if g is not None and hasattr(g, "entropic_blend_alpha"):
        return float(getattr(g, "entropic_blend_alpha"))
    return float(getattr(getattr(cfg, "evaluation", object()), "entropic_blend_alpha", 0.0))


def _bits_from_probs(
    probs: torch.Tensor,
    *,
    prefix_bits: Optional[torch.Tensor],
    cL_bits: int,
    decode_strategy: str = "threshold",
) -> torch.Tensor:
    """
    EXACT match to GenerationCallback logic:
      - default is threshold at 0.5
      - prefix is clamped from prefix_bits, not from the generated sample
    """
    decode_strategy = str(decode_strategy).lower()
    if decode_strategy == "threshold":
        bits = (probs > 0.5).to(torch.long)
    elif decode_strategy == "bernoulli":
        bits = torch.bernoulli(probs.clamp(0, 1)).to(torch.long)
    else:
        raise ValueError(f"Unknown decode_strategy='{decode_strategy}' (use 'threshold' or 'bernoulli')")

    if prefix_bits is not None and cL_bits > 0:
        bits[:, :cL_bits] = (prefix_bits[:, :cL_bits] > 0.5).to(torch.long)
    return bits


# -----------------------------------------------------------------------------
# Continuous denoiser decoding helper (used by FID decoding for continuous models)
# -----------------------------------------------------------------------------
@torch.no_grad()
def continuous_state_to_bits_via_denoiser(
    *,
    cfg,
    model: torch.nn.Module,
    x: torch.Tensor,             # [B,S] continuous state
    sigma_decode: float,         # scalar sigma at which to decode
    conditioning_prefix: Optional[torch.Tensor] = None,  # [B,cL] float32
    cond_len_bits: int = 0,
    guidance_scale: float = 0.0,
    decode: str = "threshold",
    use_amp: Optional[bool] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Decode bits from a continuous state by ONE denoiser call.

    - Supports conditional completion + CFG (prob-space guidance)
    - Enforces prefix clamping in output bits
    - decode ∈ {"threshold","bernoulli"}
    """
    device = x.device
    B, S = x.shape

    # amp defaults
    if use_amp is None:
        use_amp = (device.type == "cuda")
    if amp_dtype is None:
        amp_dtype = torch.float16

    # sigma -> [B]
    sigma = torch.full((B,), float(sigma_decode), device=device, dtype=torch.float32)

    cL = int(cond_len_bits)
    cond_enabled = (conditioning_prefix is not None) and (cL > 0)
    use_cfg = cond_enabled and float(guidance_scale) > 0.0

    # Self-conditioning flag (match your training/model API)
    sc_enabled = bool(getattr(getattr(cfg, "model", object()), "self_condition", False))

    # Eval-only: we can pass zeros as SC input (samplers may refine internally, but for decoding this is fine)
    x0_hat = torch.zeros_like(x)

    decode = str(decode).lower()
    if decode not in {"threshold", "bernoulli"}:
        raise ValueError(f"decode must be 'threshold' or 'bernoulli' (got {decode})")

    with autocast(enabled=bool(use_amp) and device.type == "cuda", dtype=amp_dtype):
        if use_cfg:
            # Build unconditional "null prefix"
            null_strategy = str(getattr(getattr(cfg, "cond", object()), "null_strategy", "half")).lower()
            if null_strategy == "half":
                null_prefix = torch.full_like(conditioning_prefix, 0.5)
            elif null_strategy == "zeros":
                null_prefix = torch.zeros_like(conditioning_prefix)
            elif null_strategy == "random":
                null_prefix = torch.bernoulli(torch.full_like(conditioning_prefix, 0.5))
            else:
                raise ValueError(f"Unknown cfg.cond.null_strategy={null_strategy}")

            # Duplicate inputs (cond + uncond)
            x_cat = torch.cat([x, x], dim=0)                 # [2B,S]
            sig_cat = sigma.expand(2 * B)                    # [2B]

            # Clamp prefixes into x_cat
            x_cat[:B, :cL] = conditioning_prefix[:, :cL]
            x_cat[B:, :cL] = null_prefix[:, :cL]

            # SC conditioning input
            if sc_enabled:
                cond_cat = torch.cat([x0_hat, x0_hat], dim=0)
                cond_cat[:B, :cL] = conditioning_prefix[:, :cL]
                cond_cat[B:, :cL] = null_prefix[:, :cL]
            else:
                cond_cat = torch.zeros_like(x_cat)

            logits_cat = model(x_cat, sig_cat, cond_cat)
            if logits_cat.dim() == 3 and logits_cat.size(-1) == 1:
                logits_cat = logits_cat.squeeze(-1)

            logits_c = logits_cat[:B]
            logits_u = logits_cat[B:]

            probs_c = torch.sigmoid(logits_c.float()).to(dtype=torch.float32)
            probs_u = torch.sigmoid(logits_u.float()).to(dtype=torch.float32)

            # Prob-space CFG
            probs_g = probs_u + float(guidance_scale) * (probs_c - probs_u)

            # Enforce prefix from true conditioning_prefix
            probs_g[:, :cL] = conditioning_prefix[:, :cL]

            return _bits_from_probs(
                probs_g,
                prefix_bits=conditioning_prefix,
                cL_bits=cL,
                decode_strategy=decode,
            )

        else:
            # No CFG
            x_in = x
            if cond_enabled:
                x_in = x.clone()
                x_in[:, :cL] = conditioning_prefix[:, :cL]

            cond_in = x0_hat if sc_enabled else torch.zeros_like(x_in)

            logits = model(x_in, sigma, cond_in)
            if logits.dim() == 3 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)

            probs = torch.sigmoid(logits.float()).to(dtype=torch.float32)
            if cond_enabled:
                probs[:, :cL] = conditioning_prefix[:, :cL]

            return _bits_from_probs(
                probs,
                prefix_bits=conditioning_prefix,
                cL_bits=cL,
                decode_strategy=decode,
            )


# -----------------------------------------------------------------------------
# Main unified sampler-based text generation for external perplexity
# -----------------------------------------------------------------------------
@torch.no_grad()
def sample_text_sequences_for_external(
    *,
    cfg,
    model: torch.nn.Module,
    device: torch.device,
    num_samples: int,
    sampler_name: str,
    data_loader: Optional[Any] = None,
    return_dict: bool = False,
    decode_strategy: str = "threshold",
    guidance_scales: Optional[List[float]] = None,
    guidance_scale: Optional[float] = None,
    warmup: bool = True,
    ddp: bool = False,   # ✅ NEW: shard generation + gather to rank0
) -> Any:
    """
    External-PPL generation (continuous_score text bitstreams).

    IMPORTANT: preserves config semantics:
      - num_steps from cfg.evaluation.external_ppl.num_sampling_steps (if present)
      - terminal_sigmas from cfg.evaluation.external_ppl.terminal_sigmas (if present)
      - guidance_scales from cfg.evaluation.external_ppl.guidance_scales (if present)
    Falls back only if those fields are missing.

    DDP mode (ddp=True + torchrun):
      - rank0 samples prefix bits for ALL samples, broadcasts to all ranks
      - each rank samples its shard of sequences
      - per tag: gather bits to rank0 (rank0 returns full dict), other ranks return empty dict
    """
    sampler_name = str(sampler_name).lower()

    # aliases
    if sampler_name in {"heun", "heun_karras", "heunkarras"}:
        sampler_name = "heun_karras"
    if sampler_name in {"ddim", "ddim_entropic", "ddimentropic"}:
        sampler_name = "ddim_entropic"

    want_all = sampler_name in {"all", "sweep", "callback", "both"}
    sampler_list = ["heun_karras", "ddim_entropic"] if want_all else [sampler_name]

    if str(getattr(cfg, "framework", "")).lower() != "continuous_score":
        raise ValueError("sample_text_sequences_for_external supports cfg.framework='continuous_score' only.")

    seq_len = int(getattr(getattr(cfg, "data", object()), "sequence_len", 0))
    if seq_len <= 0:
        raise ValueError("cfg.data.sequence_len must be set for text generation.")

    # AMP
    use_amp, amp_dtype = resolve_eval_amp(device, cfg=cfg)


    # Build samplers
    proc = ContinuousForwardProcess(cfg)
    heun = HeunSampler(model, proc, cfg)
    ddim = DDIMSampler(model, proc, cfg)

    # --- eval config knobs (YOUR SEMANTICS) ---
    B_total = int(num_samples)

    # these are your existing helpers (as in your file)
    num_steps = _eval_num_steps(cfg)                 # external_ppl priority
    entropic_blend_alpha = _eval_entropic_blend_alpha(cfg)
    entropy_run_dir = _eval_entropy_run_dir(cfg)     # matches callback
    terminal_sigmas = _eval_terminal_sigmas(cfg)     # external_ppl priority

    # --- DDP bookkeeping ---
    ddp_active = bool(
        ddp
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
        and device.type == "cuda"
    )
    rank, world_size = get_rank_world_size()

    if ddp_active:
        B_local = shard_count(B_total, world_size, rank)
        counts = all_gather_int(B_local, device=device)
        offset = sum(counts[:rank])
    else:
        B_local = B_total
        offset = 0

    # ------------------------------------------------------------
    # Conditional prefix (DDP-safe)
    #   - rank0 samples prefix_total for ALL B_total, tiles if short
    #   - broadcast prefix_total or None to all ranks
    #   - each rank slices its local shard
    # ------------------------------------------------------------
    prefix_total: Optional[torch.Tensor] = None
    prefix_local: Optional[torch.Tensor] = None
    cL = 0

    if _eval_cond_enabled(cfg):
        cL = _eval_cond_len_bits(cfg, seq_len)

    if cL > 0 and data_loader is not None:
        if ddp_active:
            if rank == 0:
                # sample on rank0 only, then tile to B_total for robustness
                p = _eval_sample_prefixes_from_loader(data_loader, num_samples=B_total, cL=cL, device=device)
                if p is not None:
                    if p.dim() != 2 or p.size(1) != cL:
                        raise ValueError(f"prefix tensor must be [B,cL], got {tuple(p.shape)} (cL={cL})")
                    if p.size(0) != B_total:
                        p = _tile_rows_to_len(p, B_total)
                prefix_total = p

            # broadcast optional prefix_total
            prefix_total = broadcast_optional_tensor(
                prefix_total,
                shape=(B_total, cL),
                dtype=torch.float32,
                device=device,
                src=0,
            )

            if prefix_total is not None and B_local > 0:
                prefix_local = prefix_total[offset : offset + B_local].contiguous()
            else:
                prefix_local = None
                cL = 0  # ensures unconditional path consistent
        else:
            # non-DDP: preserve original semantics (NO tiling unless you want it)
            prefix_local = _eval_sample_prefixes_from_loader(data_loader, num_samples=B_local, cL=cL, device=device)
            if prefix_local is None:
                cL = 0
    else:
        cL = 0
        prefix_local = None

    # ------------------------------------------------------------
    # Resolve guidance scales (only meaningful if conditional completion)
    # Tag set determinism in DDP depends on cL/prefix being identical across ranks.
    # We ensured that above.
    # ------------------------------------------------------------
    if prefix_local is not None and cL > 0:
        if guidance_scales is not None:
            gss = [float(x) for x in list(guidance_scales)]
        elif guidance_scale is not None:
            gss = [float(guidance_scale)]
        else:
            gss = _eval_guidance_scales(cfg)

        if len(gss) == 0:
            gss = [1.0]
        guidance_scales_final = gss
    else:
        guidance_scales_final = [0.0]

    def _select_sampler(tag_sampler: str):
        if tag_sampler == "heun_karras":
            return heun, "karras"
        if tag_sampler == "ddim_entropic":
            return ddim, "entropic"
        raise ValueError(f"Unknown sampler '{tag_sampler}'")

    def _one_sample_local(tag_sampler: str, s_term: float, gs: float) -> torch.Tensor:
        """
        Returns bits for this rank only: [B_local, seq_len] long on GPU.
        """
        if B_local <= 0:
            return torch.empty((0, seq_len), device=device, dtype=torch.long)

        sampler, schedule = _select_sampler(tag_sampler)

        cond_kwargs = {}
        if prefix_local is not None and cL > 0:
            cond_kwargs = dict(
                conditioning_prefix=prefix_local,
                cond_len_bits=cL,
                guidance_scale=float(gs),
            )

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                _x, probs = sampler.sample(
                    B_local,
                    seq_len,
                    schedule=schedule,
                    num_steps=int(num_steps),
                    entropic_blend_alpha=float(entropic_blend_alpha),
                    entropy_run_dir=Path(str(entropy_run_dir)) if entropy_run_dir is not None else None,
                    sigma_min_override=float(s_term),
                    return_probs=True,
                    **cond_kwargs,
                )

        bits = _bits_from_probs(
            probs.to(device=device),
            prefix_bits=prefix_local,
            cL_bits=cL,
            decode_strategy=decode_strategy,
        )
        return bits

    # ------------------------------------------------------------
    # Warmup (true small-step sampler call)
    # ------------------------------------------------------------
    if warmup and B_local > 0:
        Bw = min(8, B_local)
        steps_w = min(10, int(num_steps))
        s_term0 = float(terminal_sigmas[0])
        gs0 = float(guidance_scales_final[0])

        cond_kwargs_w = {}
        if prefix_local is not None and cL > 0:
            cond_kwargs_w = dict(
                conditioning_prefix=prefix_local[:Bw].contiguous(),
                cond_len_bits=cL,
                guidance_scale=float(gs0),
            )

        for smp in sampler_list:
            sampler_obj, schedule = _select_sampler(smp)
            warmup_continuous_sampler(
                sampler_obj,
                B=Bw,
                seq_len=seq_len,
                num_steps=steps_w,
                schedule=schedule,
                entropy_run_dir=entropy_run_dir,
                entropic_blend_alpha=float(entropic_blend_alpha),
                sigma_min_override=s_term0,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                cond_kwargs=cond_kwargs_w if (prefix_local is not None and cL > 0) else None,
            )

    # ------------------------------------------------------------
    # Return modes (with DDP gather per tag)
    # ------------------------------------------------------------
    if not return_dict:
        s_term = float(terminal_sigmas[0])
        gs = float(guidance_scales_final[0])

        bits_local = _one_sample_local(sampler_list[0], s_term, gs)

        if ddp_active:
            bits_all = gather_varlen_firstdim_to_rank0(bits_local, dst=0)
            if rank == 0:
                return bits_all, (prefix_total if prefix_total is not None else None), cL
            return None, None, cL

        return bits_local, prefix_local, cL

    out: Dict[str, torch.Tensor] = {}
    for s_term in terminal_sigmas:
        for gs in guidance_scales_final:
            for smp in sampler_list:
                tag = f"{smp}_term{math.log10(float(s_term)):.2f}_gs{float(gs):g}"
                bits_local = _one_sample_local(smp, float(s_term), float(gs))

                if ddp_active:
                    bits_all = gather_varlen_firstdim_to_rank0(bits_local, dst=0)
                    if rank == 0:
                        out[tag] = bits_all.detach().cpu()
                else:
                    out[tag] = bits_local.detach().cpu()

    if ddp_active:
        if rank == 0:
            return out, (prefix_total if prefix_total is not None else None), cL
        return {}, None, cL

    return out, prefix_local, cL