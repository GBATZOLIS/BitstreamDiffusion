# evaluation/fid.py
import math
from typing import Optional

import numpy as np
from scipy import linalg

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_fid.inception import InceptionV3

__all__ = [
    "InceptionV3Pool3",
    "compute_fid_inception",
]

# ============================================================
# Preprocessing helpers
# ============================================================

@torch.no_grad()
def _ensure_float01(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure input is float32 in [0, 1]. If values look like 0..255, normalize.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")

    if x.dtype.is_floating_point:
        if x.max() > 1.0:
            x = x / 255.0
    else:
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0

    return x.clamp(0, 1)


@torch.no_grad()
def _unflatten_if_needed(x: torch.Tensor) -> torch.Tensor:
    """
    Accepts shapes:
      - (B, 784) -> assumes square -> (B,1,28,28)
      - (B, H, W) -> (B,1,H,W)
      - (B, C, H, W) -> passthrough
      - (B, H, W, C) -> (B, C, H, W)  (channels-last)
    Raises if it cannot infer a valid image shape.
    """
    if x.dim() == 2:
        # Flattened images (B, S)
        B, S = x.shape
        H = int(round(math.sqrt(S)))
        if H * H != S:
            raise ValueError(f"Cannot infer square image from flattened size {S}.")
        return x.view(B, 1, H, H)

    elif x.dim() == 3:
        # (B, H, W)
        B, H, W = x.shape
        return x.view(B, 1, H, W)

    elif x.dim() == 4:
        # Either (B, C, H, W) or (B, H, W, C)
        if x.shape[1] in (1, 3):
            return x
        if x.shape[-1] in (1, 3):
            # channels-last -> channels-first
            return x.permute(0, 3, 1, 2).contiguous()
        # Ambiguous 4D
        raise ValueError(
            f"Ambiguous 4D shape {tuple(x.shape)}: cannot determine channels dimension."
        )

    else:
        raise ValueError(f"Expected tensor with 2-4 dims, got shape {tuple(x.shape)}")


@torch.no_grad()
def _to_inception_input(x: torch.Tensor) -> torch.Tensor:
    """
    Bring x to shape (B,3,H,W) with values in [0,1].

    Resizing to 299x299 and normalization to [-1, 1] are handled internally
    by the FID InceptionV3 model (pytorch-fid style).
    """
    x = _unflatten_if_needed(x)
    x = _ensure_float01(x)
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    return x.clamp(0, 1)


# ============================================================
# Robust batch image extraction
# ============================================================

@torch.no_grad()
def _get_images_from_batch(batch):
    """
    Accepts many dataloader formats:
      - (imgs,), (imgs, y), (imgs, y, extra...)
      - dicts with keys 'image'/'images'/'img'/'x'
      - a bare tensor
    Returns the image tensor.
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 0:
            raise ValueError("Empty batch tuple/list encountered.")
        return batch[0]
    if isinstance(batch, dict):
        for key in ("image", "images", "img", "x"):
            if key in batch:
                return batch[key]
        # fallback to first value
        return next(iter(batch.values()))
    if isinstance(batch, torch.Tensor):
        return batch
    raise TypeError(f"Unsupported batch type: {type(batch)}")


# ============================================================
# FID InceptionV3 (pool3) feature extractor
# ============================================================

class InceptionV3Pool3(nn.Module):
    """
    FID Inception-V3 (pool3) features (2048-D), matching pytorch-fid / TTUR.

    This wraps pytorch-fid's InceptionV3 so that FID values are comparable
    to the standard implementation.
    """
    def __init__(self):
        super().__init__()
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        # resize_input=True -> internally resize to 299x299
        # normalize_input=True -> internally normalize to [-1,1]
        self.inception = InceptionV3(
            [block_idx],
            resize_input=True,
            normalize_input=True
        )
        self.inception.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is assumed (B,3,H,W) in [0,1]
        out = self.inception(x)[0]  # list -> take pool3 block
        return out.view(out.size(0), -1)  # (B, 2048)


# ============================================================
# Feature extraction utilities
# ============================================================

@torch.no_grad()
def _activations_from_loader(
    model_feat: nn.Module, loader, device: torch.device
) -> torch.Tensor:
    feats = []
    for batch in loader:
        xb = _get_images_from_batch(batch).to(device)
        xb = _to_inception_input(xb)
        f = model_feat(xb)
        feats.append(f.detach().cpu())
    return torch.cat(feats, 0)


@torch.no_grad()
def _activations_from_tensor(
    model_feat: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    bs: int = 256,
) -> torch.Tensor:
    feats = []
    for i in range(0, x.size(0), bs):
        xb = x[i : i + bs].to(device)
        xb = _to_inception_input(xb)
        f = model_feat(xb)
        feats.append(f.detach().cpu())
    return torch.cat(feats, 0)


# ============================================================
# FID math (NumPy + SciPy, matching pytorch-fid)
# ============================================================

def _stats_np(feats: np.ndarray):
    """
    Return mean and covariance of features.

    This matches numpy.cov(rowvar=False, bias=False) behavior (unbiased).
    """
    mu = np.mean(feats, axis=0)
    cov = np.cov(feats, rowvar=False, bias=False)
    return mu, cov


def _frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Frechet distance between two Gaussians, using SciPy's sqrtm.

    This closely follows the pytorch-fid implementation so that numbers
    are comparable across codebases.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # sqrtm may return complex numbers due to numerical error
    if np.iscomplexobj(covmean):
        if not np.allclose(np.imag(covmean), 0, atol=1e-3):
            raise ValueError("Imaginary component found in covmean with large magnitude.")
        covmean = np.real(covmean)

    tr_covmean = np.trace(covmean)

    fid = (
        diff.dot(diff)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2.0 * tr_covmean
    )
    return float(fid)


# ============================================================
# Public API
# ============================================================

@torch.no_grad()
def compute_fid_inception(
    real_loader,
    gen_images: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    featnet: Optional[InceptionV3Pool3] = None,
) -> float:
    """
    Compute FID using FID Inception-V3 pool3 features (2048-D),
    matching pytorch-fid / TTUR.

    Args:
        real_loader: Dataloader yielding real images in any of the supported batch formats.
                     Images can be flattened (B,S), (B,H,W), (B,C,H,W), or (B,H,W,C),
                     values in [0,1] (float) or [0,255] (float/uint8).
        gen_images:  Tensor of generated images with shape (N,1,H,W), (N,3,H,W),
                     (N,H,W), or (N,S) flattened; values in [0,1] or [0,255].
        device:      torch.device.
        batch_size:  Chunk size for processing gen_images (kept for API parity; loader uses its own batch size).
        featnet:     Optional pre-instantiated InceptionV3Pool3 on `device`. If None, one is created.

    Returns:
        FID (float, clamped to be non-negative).
    """
    if featnet is None:
        featnet = InceptionV3Pool3().to(device).eval()

    # Extract features
    fr = _activations_from_loader(featnet, real_loader, device)
    fg = _activations_from_tensor(featnet, gen_images, device, batch_size)

    # Convert to float64 NumPy for stable statistics
    fr_np = fr.numpy().astype(np.float64)
    fg_np = fg.numpy().astype(np.float64)

    mu_r, cov_r = _stats_np(fr_np)
    mu_g, cov_g = _stats_np(fg_np)

    fid = _frechet_distance(mu_r, cov_r, mu_g, cov_g)

    # Theoretically FID >= 0; clamp small negatives from numerical issues.
    fid = max(fid, 0.0)
    return fid
