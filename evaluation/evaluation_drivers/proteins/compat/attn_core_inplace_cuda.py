"""Inference-only fallback for OpenFold's optional fused attention kernel."""

from __future__ import annotations

import torch


def forward_(attention_logits: torch.Tensor, *_shape) -> None:
    attention_logits.copy_(torch.softmax(attention_logits, dim=-1))


def backward_(
    *_args, **_kwargs
) -> None:  # pragma: no cover - evaluation is no-grad
    raise RuntimeError("The M0 evaluation attention shim is inference-only")
