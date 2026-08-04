"""Warm-start column surgery from the 5-bit sequence model into the 18-bit model.

The patch size changes from 5 to 18, so ``patch_proj`` and the bit output heads
change shape and a plain ``load_state_dict`` cannot be used (plan section 6.4).
This module copies the trunk, time layers, and normalization verbatim, copies
only the five sequence-bit input and output columns from the sequence model, and
leaves the thirteen structure-bit columns freshly initialized.

Slot layout: the flattened patch places sequence bit slots first (0-4) then
structure slots (5-17), so the sequence slots occupy the leading columns/rows of
each per-patch projection and can be copied block-wise.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

SEQ_SLOTS = 5
STRUCT_SLOTS = 13
PATCH_SLOTS = SEQ_SLOTS + STRUCT_SLOTS


def _strip(name: str) -> str:
    prev = None
    while prev != name:
        prev = name
        name = name.replace("_orig_mod.", "")
        if name.startswith("module."):
            name = name[7:]
    return name


def warm_start_multimodal_from_sequence(
    mm_model: nn.Module,
    seq_state_dict: Dict[str, torch.Tensor],
    *,
    seq_patch_size: int = SEQ_SLOTS,
    verbose: bool = True,
) -> Dict[str, int]:
    """Copy compatible sequence-model weights into ``mm_model`` in place.

    Returns a report with counts of exact copies, slot-column copies, and skipped
    (freshly initialized) tensors. Requires the two models to share trunk width,
    depth, content dim, and positional feature counts so per-slot blocks align.
    """
    src = {_strip(k): v for k, v in seq_state_dict.items()}
    report = {
        "exact": 0,
        "slot_copied": 0,
        "skipped": 0,
        "missing_in_source": 0,
    }

    with torch.no_grad():
        for name, param in mm_model.named_parameters():
            key = _strip(name)
            if key not in src:
                report["missing_in_source"] += 1
                continue
            s = src[key]
            if s.shape == param.shape:
                param.copy_(s.to(param.dtype))
                report["exact"] += 1
                continue

            copied = _try_slot_copy(key, param, s, seq_patch_size)
            if copied:
                report["slot_copied"] += 1
            else:
                report["skipped"] += 1

    if verbose:
        print(
            f"[warm_start] exact={report['exact']} slot_copied={report['slot_copied']} "
            f"skipped={report['skipped']} missing_in_source={report['missing_in_source']}"
        )
    return report


def _try_slot_copy(
    key: str, tgt: torch.Tensor, src: torch.Tensor, seq_patch_size: int
) -> bool:
    """Copy the leading sequence-slot block for the known patch/head projections."""
    # patch_proj.weight: [E, PATCH_SLOTS * d]. Copy first seq_patch_size slot blocks
    # along the input (column) dimension.
    if key == "patch_proj.weight":
        e_t, in_t = tgt.shape
        e_s, in_s = src.shape
        if e_t != e_s or in_t % PATCH_SLOTS != 0 or in_s % seq_patch_size != 0:
            return False
        d_t = in_t // PATCH_SLOTS
        d_s = in_s // seq_patch_size
        if d_t != d_s:
            return False
        cols = seq_patch_size * d_t
        tgt[:, :cols].copy_(src[:, :cols].to(tgt.dtype))
        return True

    # unpatch_proj_content.weight: [C*PATCH_SLOTS, E]; bias: [C*PATCH_SLOTS].
    if key in ("unpatch_proj_content.weight", "unpatch_proj_content.bias"):
        out_t = tgt.shape[0]
        out_s = src.shape[0]
        if out_t % PATCH_SLOTS != 0 or out_s % seq_patch_size != 0:
            return False
        c_t = out_t // PATCH_SLOTS
        c_s = out_s // seq_patch_size
        if c_t != c_s:
            return False
        rows = seq_patch_size * c_t
        tgt[:rows].copy_(src[:rows].to(tgt.dtype))
        return True

    # optimal_skip_mlp head patch_adapter: weight [PATCH_SLOTS*hidden, E], bias [..].
    if key in ("head.patch_adapter.weight", "head.patch_adapter.bias"):
        out_t = tgt.shape[0]
        out_s = src.shape[0]
        if out_t % PATCH_SLOTS != 0 or out_s % seq_patch_size != 0:
            return False
        h_t = out_t // PATCH_SLOTS
        h_s = out_s // seq_patch_size
        if h_t != h_s:
            return False
        rows = seq_patch_size * h_t
        tgt[:rows].copy_(src[:rows].to(tgt.dtype))
        return True

    return False
