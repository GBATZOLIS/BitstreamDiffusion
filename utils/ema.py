# utils/ema.py
from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn


def _unwrap_model(m: nn.Module) -> nn.Module:
    # DDP -> .module
    if hasattr(m, "module"):
        m = m.module
    # torch.compile wrapper -> ._orig_mod
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def _clean_name(name: str) -> str:
    # Be aggressive and repeat until stable (handles "module._orig_mod." etc.)
    prev = None
    while prev != name:
        prev = name
        name = name.replace("_orig_mod.", "")
        if name.startswith("module."):
            name = name[7:]
    return name


class EMA:
    """Exponential moving average over trainable parameters (DDP/compile safe)."""

    def __init__(self, model: nn.Module, decay: float = 0.9999, *, verbose: bool = False):
        self.decay = float(decay)
        self.verbose = bool(verbose)

        model = _unwrap_model(model)

        # shadow keyed by CLEANED names
        self.shadow: Dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            k = _clean_name(n)
            self.shadow[k] = p.detach().clone()

        self._backup: Optional[Dict[str, torch.Tensor]] = None

        if self.verbose:
            print(f"[EMA] init: {len(self.shadow)} params shadowed")

    def _named_params(self, model: nn.Module):
        model = _unwrap_model(model)
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            yield _clean_name(n), p

    @torch.no_grad()
    def update(self, model: nn.Module):
        """shadow = decay * shadow + (1-decay) * param"""
        misses = 0
        total = 0
        for k, p in self._named_params(model):
            total += 1
            s = self.shadow.get(k, None)
            if s is None:
                misses += 1
                continue
            # Ensure dtype/device match (should already, but be defensive)
            if s.device != p.device or s.dtype != p.dtype:
                self.shadow[k] = s.to(device=p.device, dtype=p.dtype)
                s = self.shadow[k]
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

        if self.verbose and total > 0 and misses > 0:
            print(f"[EMA] update: matched={total-misses}/{total}, misses={misses}")

    @torch.no_grad()
    def apply(self, model: nn.Module):
        """Copy EMA params into model, backing up current params."""
        self._backup = {}
        misses = 0
        total = 0

        for k, p in self._named_params(model):
            total += 1
            s = self.shadow.get(k, None)
            if s is None:
                misses += 1
                continue
            self._backup[k] = p.detach().clone()
            if s.device != p.device or s.dtype != p.dtype:
                s = s.to(device=p.device, dtype=p.dtype)
            p.data.copy_(s)

        if self.verbose:
            print(f"[EMA] apply: matched={total-misses}/{total}, misses={misses}")

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """Restore pre-apply params."""
        if self._backup is None:
            return

        for k, p in self._named_params(model):
            b = self._backup.get(k, None)
            if b is None:
                continue
            if b.device != p.device or b.dtype != p.dtype:
                b = b.to(device=p.device, dtype=p.dtype)
            p.data.copy_(b)

        self._backup = None

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None):
        """Move EMA shadow tensors to a device (and optionally dtype)."""
        for k, v in list(self.shadow.items()):
            self.shadow[k] = v.to(device=device, dtype=(dtype or v.dtype))
        if self._backup is not None:
            for k, v in list(self._backup.items()):
                self._backup[k] = v.to(device=device, dtype=(dtype or v.dtype))

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict):
        self.decay = float(state_dict["decay"])
        # Keep as-is; caller can ema.to(device) after loading
        self.shadow = state_dict["shadow"]
        # Reset backup
        self._backup = None
