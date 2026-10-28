"""Small PyTorch fallback for the two torch_scatter calls imported by DPLM."""

from __future__ import annotations

import torch


def _expanded_index(
    index: torch.Tensor, src: torch.Tensor, dim: int
) -> torch.Tensor:
    index = index.to(device=src.device, dtype=torch.long)
    if index.ndim == 1:
        shape = [1] * src.ndim
        shape[dim] = index.numel()
        index = index.reshape(shape)
    return index.expand_as(src)


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
    reduce: str = "sum",
) -> torch.Tensor:
    dim %= src.ndim
    expanded = _expanded_index(index, src, dim)
    if out is None:
        size = list(src.shape)
        inferred = int(index.max().item()) + 1 if index.numel() else 0
        size[dim] = inferred if dim_size is None else int(dim_size)
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
    if reduce in {"sum", "add"}:
        return out.scatter_add_(dim, expanded, src)
    if reduce == "mean":
        result = out.scatter_add_(dim, expanded, src)
        counts = torch.zeros_like(out).scatter_add_(
            dim, expanded, torch.ones_like(src)
        )
        return result / counts.clamp_min(1)
    name = {"min": "amin", "max": "amax"}.get(reduce, reduce)
    return out.scatter_reduce_(
        dim, expanded, src, reduce=name, include_self=False
    )


def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    return scatter(
        src, index, dim=dim, out=out, dim_size=dim_size, reduce="sum"
    )
