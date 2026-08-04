"""Task modes, per-modality noise, and conditioning masks for the 18-bit model.

Every user-facing protein task falls out of one model by changing which modality
is noised, observed, or absent. This module is the single definition of that
mapping. It produces, from a per-residue per-modality state grid:

  - per-bit noise (sigma) maps for the two modalities;
  - target-loss masks (loss only on noised target bits of real residues);
  - observed-clamp masks (bits held clean and reapplied every sampler step).

Three states are distinct per residue per modality (plan section 6.2 / 7.7):
  ABSENT   the modality has no valid value here (masked from loss and clamp);
  OBSERVED the modality is known and clean (sigma 0, clamped, not supervised);
  NOISY    the modality is a generation target (sigma sampled, supervised).

A zero-noise observed modality and a missing modality are not the same state.

Layout note: the 18-bit residue patch is ``[seq_0..seq_4 | struct_0..struct_12]``
(see ``data.protein_structure_codec``). Modality 0 is sequence (slots 0-4),
modality 1 is structure (slots 5-17).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

from .protein_structure_codec import (
    PATCH_BITS_PER_RESIDUE,
    SEQ_BITS_PER_RESIDUE,
    STRUCT_BITS_PER_RESIDUE,
)

# Modality-state codes.
ABSENT = 0
OBSERVED = 1
NOISY = 2

# Modality indices.
SEQ = 0
STRUCT = 1
NUM_MODALITIES = 2

# Slot -> modality index for the 18-bit patch.
SLOT_MODALITY = np.array(
    [SEQ] * SEQ_BITS_PER_RESIDUE + [STRUCT] * STRUCT_BITS_PER_RESIDUE,
    dtype=np.int64,
)
assert SLOT_MODALITY.shape[0] == PATCH_BITS_PER_RESIDUE


# Whole-example task modes: (seq_state, struct_state). Motif is per-residue and
# is built separately by ``build_motif_states``.
TASK_MODES: Dict[str, Tuple[int, int]] = {
    "joint": (NOISY, NOISY),  # p(s, z) co-generation
    "inverse_folding": (NOISY, OBSERVED),  # p(s | z)
    "forward_folding": (OBSERVED, NOISY),  # p(z | s)
    "sequence_marginal": (NOISY, ABSENT),  # p(s)
    "structure_marginal": (ABSENT, NOISY),  # p(z)
}

# A reasonable default paired-training mix (plan phase 4: start roughly balanced
# across the three core joint/folding/inverse tasks, small marginal/inpainting
# mass). Weights are tuned on validation, not fixed blindly.
DEFAULT_TASK_WEIGHTS: Dict[str, float] = {
    "joint": 0.30,
    "forward_folding": 0.20,
    "inverse_folding": 0.20,
    "structure_marginal": 0.10,
    "sequence_marginal": 0.10,
    "motif": 0.10,
}


def task_modality_states(task: str) -> Tuple[int, int]:
    """Return ``(seq_state, struct_state)`` for a whole-example task mode."""
    if task not in TASK_MODES:
        raise KeyError(
            f"unknown whole-example task {task!r}; use one of {sorted(TASK_MODES)}"
        )
    return TASK_MODES[task]


def build_motif_states(
    length: int,
    motif_spans: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Per-residue states for motif scaffolding: motif residues observed, rest noisy.

    ``motif_spans`` are half-open ``[start, end)`` residue ranges whose sequence
    AND structure are preserved (observed); every other residue is a generation
    target (noisy) in both modalities. Returns ``[length, 2]`` int8.
    """
    states = np.full((length, NUM_MODALITIES), NOISY, dtype=np.int8)
    for start, end in motif_spans:
        s = max(0, int(start))
        e = min(int(length), int(end))
        if e > s:
            states[s:e, SEQ] = OBSERVED
            states[s:e, STRUCT] = OBSERVED
    return states


def build_example_states(
    task: str,
    length: int,
    *,
    seq_available: bool = True,
    struct_available: bool = True,
    motif_spans: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Build a ``[length, 2]`` state grid for one example under a task.

    ``seq_available`` / ``struct_available`` downgrade a requested state to
    ABSENT when the example simply lacks that modality (e.g. a sequence-only
    replay row has no structure). This keeps presence honest per row.
    """
    if task == "motif":
        spans = (
            motif_spans
            if motif_spans is not None
            else _default_motif_span(length)
        )
        states = build_motif_states(length, spans).astype(np.int8)
    else:
        seq_state, struct_state = task_modality_states(task)
        states = np.empty((length, NUM_MODALITIES), dtype=np.int8)
        states[:, SEQ] = seq_state
        states[:, STRUCT] = struct_state

    if not seq_available:
        states[:, SEQ] = ABSENT
    if not struct_available:
        states[:, STRUCT] = ABSENT
    return states


def _default_motif_span(length: int) -> List[Tuple[int, int]]:
    """A single centered motif covering the middle third, as a training default."""
    if length < 3:
        return [(0, length)]
    third = max(1, length // 3)
    start = (length - third) // 2
    return [(start, start + third)]


def sample_task(rng: np.random.Generator, weights: Dict[str, float]) -> str:
    """Draw a task name from a weight dict (weights need not be normalized)."""
    names = list(weights.keys())
    w = np.asarray(
        [max(0.0, float(weights[n])) for n in names], dtype=np.float64
    )
    total = w.sum()
    if total <= 0:
        raise ValueError("task weights sum to zero")
    return str(rng.choice(names, p=w / total))


# -----------------------------------------------------------------------------
# Derived maps (numpy). Torch equivalents below share the same semantics.
# -----------------------------------------------------------------------------


def per_bit_sigma_map_np(
    states: np.ndarray,  # [B, L, 2] int
    sigma_seq: np.ndarray,  # [B]
    sigma_struct: np.ndarray,  # [B]
) -> np.ndarray:
    """Per-bit sigma ``[B, L*18]``: modality sigma where NOISY, else 0.

    Observed and absent bits get sigma 0 so the matched-filter input scaling
    treats them as clean; the loss and clamp masks keep the two states distinct.
    """
    states = np.asarray(states)
    b, l, m = states.shape
    if m != NUM_MODALITIES:
        raise ValueError("states last axis must be 2 (seq, struct)")
    sigma_mod = np.stack(
        [
            np.broadcast_to(np.asarray(sigma_seq)[:, None], (b, l)),
            np.broadcast_to(np.asarray(sigma_struct)[:, None], (b, l)),
        ],
        axis=-1,
    ).astype(np.float32)  # [B, L, 2]
    noisy = states == NOISY
    sigma_res = np.where(noisy, sigma_mod, 0.0)  # [B, L, 2]
    # Expand modality sigma across that modality's bit slots.
    per_slot = sigma_res[:, :, SLOT_MODALITY]  # [B, L, 18]
    return per_slot.reshape(b, l * PATCH_BITS_PER_RESIDUE)


def target_loss_masks_np(
    states: np.ndarray,  # [B, L, 2]
    residue_mask: np.ndarray,  # [B, L] bool, real residues
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(seq_target, struct_target)`` bit masks ``[B, L*5], [B, L*13]``.

    A bit is a supervised target iff its modality state is NOISY and the residue
    is real. Sequence and structure are reduced separately for equal-modality
    loss weighting.
    """
    states = np.asarray(states)
    residue_mask = np.asarray(residue_mask).astype(bool)
    b, l, _ = states.shape
    seq_res = (states[:, :, SEQ] == NOISY) & residue_mask  # [B, L]
    struct_res = (states[:, :, STRUCT] == NOISY) & residue_mask
    seq_bits = np.repeat(seq_res, SEQ_BITS_PER_RESIDUE, axis=1)  # [B, L*5]
    struct_bits = np.repeat(
        struct_res, STRUCT_BITS_PER_RESIDUE, axis=1
    )  # [B, L*13]
    return seq_bits, struct_bits


def full_modality_target_masks_np(
    states: np.ndarray,  # [B, L, 2]
    residue_mask: np.ndarray,  # [B, L] bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Full-patch target masks ``[B, L*18]`` for each modality separately.

    ``seq_full`` is True only at sequence slots (0-4) of residues whose sequence
    state is NOISY; ``struct_full`` is True only at structure slots (5-17) of
    residues whose structure state is NOISY. These align with the ``[B, L*18]``
    logits so the loss can reduce the two modalities independently.
    """
    states = np.asarray(states)
    residue_mask = np.asarray(residue_mask).astype(bool)
    b, l, _ = states.shape
    seq_res = (states[:, :, SEQ] == NOISY) & residue_mask  # [B, L]
    struct_res = (states[:, :, STRUCT] == NOISY) & residue_mask
    seq_full = np.zeros((b, l, PATCH_BITS_PER_RESIDUE), dtype=bool)
    struct_full = np.zeros((b, l, PATCH_BITS_PER_RESIDUE), dtype=bool)
    seq_full[:, :, :SEQ_BITS_PER_RESIDUE] = seq_res[:, :, None]
    struct_full[:, :, SEQ_BITS_PER_RESIDUE:] = struct_res[:, :, None]
    return (
        seq_full.reshape(b, l * PATCH_BITS_PER_RESIDUE),
        struct_full.reshape(b, l * PATCH_BITS_PER_RESIDUE),
    )


def per_bit_state_ids_np(states: np.ndarray) -> np.ndarray:
    """Per-bit modality-state id ``[B, L*18]`` (each bit inherits its modality state)."""
    states = np.asarray(states)
    b, l, _ = states.shape
    per_slot = states[:, :, SLOT_MODALITY]  # [B, L, 18]
    return per_slot.reshape(b, l * PATCH_BITS_PER_RESIDUE)


def observed_clamp_mask_np(
    states: np.ndarray,  # [B, L, 2]
    residue_mask: np.ndarray,  # [B, L] bool
) -> np.ndarray:
    """Per-bit clamp mask ``[B, L*18]``: True where a real residue's modality is OBSERVED."""
    states = np.asarray(states)
    residue_mask = np.asarray(residue_mask).astype(bool)
    b, l, _ = states.shape
    obs = (states == OBSERVED) & residue_mask[:, :, None]  # [B, L, 2]
    per_slot = obs[:, :, SLOT_MODALITY]  # [B, L, 18]
    return per_slot.reshape(b, l * PATCH_BITS_PER_RESIDUE)


# -----------------------------------------------------------------------------
# Torch equivalents (data-loading / GPU path)
# -----------------------------------------------------------------------------


def per_bit_sigma_map_torch(states, sigma_seq, sigma_struct):
    if not _HAS_TORCH:
        raise RuntimeError("torch is not available")
    b, l, m = states.shape
    slot_mod = torch.as_tensor(SLOT_MODALITY, device=states.device)
    sigma_mod = torch.stack(
        [
            sigma_seq.view(b, 1).expand(b, l),
            sigma_struct.view(b, 1).expand(b, l),
        ],
        dim=-1,
    ).to(torch.float32)  # [B, L, 2]
    sigma_res = torch.where(
        states == NOISY, sigma_mod, torch.zeros_like(sigma_mod)
    )
    per_slot = sigma_res[:, :, slot_mod]  # [B, L, 18]
    return per_slot.reshape(b, l * PATCH_BITS_PER_RESIDUE)


def target_loss_masks_torch(states, residue_mask):
    if not _HAS_TORCH:
        raise RuntimeError("torch is not available")
    residue_mask = residue_mask.bool()
    seq_res = (states[:, :, SEQ] == NOISY) & residue_mask
    struct_res = (states[:, :, STRUCT] == NOISY) & residue_mask
    seq_bits = seq_res.repeat_interleave(SEQ_BITS_PER_RESIDUE, dim=1)
    struct_bits = struct_res.repeat_interleave(STRUCT_BITS_PER_RESIDUE, dim=1)
    return seq_bits, struct_bits


def observed_clamp_mask_torch(states, residue_mask):
    if not _HAS_TORCH:
        raise RuntimeError("torch is not available")
    slot_mod = torch.as_tensor(SLOT_MODALITY, device=states.device)
    obs = (states == OBSERVED) & residue_mask.bool().unsqueeze(-1)
    per_slot = obs[:, :, slot_mod]
    return per_slot.reshape(
        states.shape[0], states.shape[1] * PATCH_BITS_PER_RESIDUE
    )
