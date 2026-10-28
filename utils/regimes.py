"""Shared conditioning-regime representation and adapter registry.

The three experiment families describe conditioning at different resolutions:
text/audio and MNIST-Sum produce bit masks directly, while proteins first build
a per-residue modality-state grid.  ``ConditioningPlan`` is the common boundary
between those descriptions and the code that applies noise, clamping, CFG
dropout, and loss masking.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from importlib import import_module
from typing import Callable

import torch


class ConditioningState(IntEnum):
    """Meaning of one modality/position in a conditioning regime."""

    ABSENT = 0
    OBSERVED = 1
    NOISY = 2
    DROPPED = 3


@dataclass(frozen=True)
class ConditioningPlan:
    """Materialized conditioning decisions for one batch.

    All masks are bit-level ``[B, S]`` boolean tensors. ``state_ids`` is
    optional because only models with an explicit state embedding (currently
    proteins) need the richer state grid after the masks have been derived.
    """

    regime_ids: torch.Tensor | None
    clamp_mask: torch.Tensor
    loss_mask: torch.Tensor
    protect_mask: torch.Tensor
    null_mask: torch.Tensor
    state_ids: torch.Tensor | None = None

    def validate(self, *, batch_size: int | None = None, seq_len: int | None = None):
        shape = self.clamp_mask.shape
        if len(shape) != 2:
            raise ValueError(f"conditioning masks must be [B,S], got {tuple(shape)}")
        if batch_size is not None and shape[0] != int(batch_size):
            raise ValueError(f"conditioning batch is {shape[0]}, expected {batch_size}")
        if seq_len is not None and shape[1] != int(seq_len):
            raise ValueError(f"conditioning length is {shape[1]}, expected {seq_len}")
        for name in ("loss_mask", "protect_mask", "null_mask"):
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(
                    f"{name} has shape {tuple(value.shape)}, expected {tuple(shape)}"
                )
            if value.dtype != torch.bool:
                raise TypeError(f"{name} must be boolean, got {value.dtype}")
        if self.clamp_mask.dtype != torch.bool:
            raise TypeError(f"clamp_mask must be boolean, got {self.clamp_mask.dtype}")
        if bool((self.protect_mask & ~self.clamp_mask).any()):
            raise ValueError("protect_mask must be a subset of clamp_mask")
        if bool((self.clamp_mask & self.loss_mask).any()):
            raise ValueError("clamped positions cannot also be loss targets")
        if self.regime_ids is not None:
            if self.regime_ids.shape != (shape[0],):
                raise ValueError(
                    f"regime_ids must be [B], got {tuple(self.regime_ids.shape)}"
                )
        return self


def plan_from_masks(
    regime_ids: torch.Tensor | None,
    clamp_mask: torch.Tensor,
    *,
    protect_mask: torch.Tensor | None = None,
    loss_mask: torch.Tensor | None = None,
    null_mask: torch.Tensor | None = None,
    state_ids: torch.Tensor | None = None,
) -> ConditioningPlan:
    """Build and validate a plan from already-materialized bit masks."""

    clamp = clamp_mask.to(dtype=torch.bool)
    protect = torch.zeros_like(clamp) if protect_mask is None else protect_mask.bool()
    loss = ~clamp if loss_mask is None else loss_mask.bool()
    null = torch.zeros_like(clamp) if null_mask is None else null_mask.bool()
    return ConditioningPlan(
        regime_ids=regime_ids,
        clamp_mask=clamp,
        loss_mask=loss,
        protect_mask=protect,
        null_mask=null,
        state_ids=state_ids,
    ).validate()


def plan_from_bit_states(
    state_ids: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    protect_mask: torch.Tensor | None = None,
    regime_ids: torch.Tensor | None = None,
) -> ConditioningPlan:
    """Derive the common masks from a bit-level state tensor ``[B,S]``."""

    if state_ids.ndim != 2:
        raise ValueError(f"bit state_ids must be [B,S], got {tuple(state_ids.shape)}")
    valid = torch.ones_like(state_ids, dtype=torch.bool)
    if valid_mask is not None:
        if valid_mask.shape != state_ids.shape:
            raise ValueError(
                f"valid_mask has shape {tuple(valid_mask.shape)}, "
                f"expected {tuple(state_ids.shape)}"
            )
        valid = valid_mask.bool()
    clamp = (state_ids == int(ConditioningState.OBSERVED)) & valid
    loss = (state_ids == int(ConditioningState.NOISY)) & valid
    null = (
        (state_ids == int(ConditioningState.ABSENT))
        | (state_ids == int(ConditioningState.DROPPED))
    ) & valid
    return plan_from_masks(
        regime_ids,
        clamp,
        protect_mask=protect_mask,
        loss_mask=loss,
        null_mask=null,
        state_ids=state_ids,
    )


def apply_condition_dropout_states(
    state_ids: torch.Tensor, drop_rows: torch.Tensor
) -> torch.Tensor:
    """Rewrite OBSERVED to DROPPED for selected examples."""

    if drop_rows.ndim != 1 or drop_rows.shape[0] != state_ids.shape[0]:
        raise ValueError(
            f"drop_rows must be [B]={state_ids.shape[0]}, got {tuple(drop_rows.shape)}"
        )
    out = state_ids.clone()
    selected = out[drop_rows.bool()]
    selected[selected == int(ConditioningState.OBSERVED)] = int(
        ConditioningState.DROPPED
    )
    out[drop_rows.bool()] = selected
    return out


RegimeAdapter = Callable[..., ConditioningPlan]
_ADAPTERS: dict[str, RegimeAdapter] = {}
_BUILTIN_ADAPTER_MODULES = {
    "mnist_sum": "utils.mnist_sum.utils",
    "text_audio": "utils.text_audio.utils",
}


def register_regime_adapter(name: str):
    """Register an experiment layout adapter without coupling it to Trainer."""

    key = str(name).strip().lower()

    def decorator(fn: RegimeAdapter):
        if key in _ADAPTERS and _ADAPTERS[key] is not fn:
            raise ValueError(f"regime adapter {key!r} is already registered")
        _ADAPTERS[key] = fn
        return fn

    return decorator


def sample_conditioning_plan(
    name: str,
    cfg,
    B: int,
    S: int,
    device: torch.device,
    **kwargs,
) -> ConditioningPlan:
    """Materialize a plan through a named layout adapter."""

    key = str(name).strip().lower()
    if key not in _ADAPTERS:
        module = _BUILTIN_ADAPTER_MODULES.get(key)
        if module is None:
            known = ", ".join(sorted(set(_ADAPTERS) | set(_BUILTIN_ADAPTER_MODULES)))
            raise ValueError(f"unknown regime adapter {name!r}; expected one of: {known}")
        import_module(module)
    plan = _ADAPTERS[key](cfg, B, S, device, **kwargs)
    return plan.validate(batch_size=B, seq_len=S)
