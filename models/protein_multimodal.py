"""Additive multimodal components for the 18-bit protein model.

These modules are the only new learned pieces the multimodal path adds on top of
the shared SDT trunk (plan section 6.1/6.2). They are deliberately small and
composable so the trunk, time layers, and normalization can be warm-started from
the sequence-only checkpoint unchanged.

Provided:
  - ``MultimodalBitEmbeddings``: learned intra-patch slot embeddings (which of the
    18 bit positions) and modality-state embeddings (absent/observed/noisy),
    added to the per-bit content so the trunk can tell sequence bits from
    structure bits and clean from generated modalities;
  - ``two_modality_time_sigma``: exposes both modality noise levels to the time
    conditioning by summing their sinusoidal embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn

SEQ_BITS = 5
STRUCT_BITS = 13
PATCH_BITS = SEQ_BITS + STRUCT_BITS  # 18
NUM_STATES = 3  # absent, observed, noisy

# Slot -> modality (0 sequence, 1 structure), for building per-bit modality ids.
SLOT_MODALITY = tuple([0] * SEQ_BITS + [1] * STRUCT_BITS)


class MultimodalBitEmbeddings(nn.Module):
    """Learned per-bit slot and modality-state embeddings added to content.

    ``content`` has shape ``[B, S, C]`` with ``S = L * 18``. Slot ids are
    ``position % 18`` and state ids come from the collated modality states. Both
    embeddings are zero-initialized so a warm-started model starts as the
    identity and learns the distinctions gradually.
    """

    def __init__(self, content_dim: int, patch_size: int = PATCH_BITS):
        super().__init__()
        if patch_size != PATCH_BITS:
            raise ValueError(
                f"multimodal patch size must be {PATCH_BITS}, got {patch_size}"
            )
        self.content_dim = int(content_dim)
        self.patch_size = int(patch_size)
        self.slot_embed = nn.Embedding(self.patch_size, self.content_dim)
        self.state_embed = nn.Embedding(NUM_STATES, self.content_dim)
        nn.init.zeros_(self.slot_embed.weight)
        nn.init.zeros_(self.state_embed.weight)
        self.register_buffer(
            "_slot_ids_cache",
            torch.zeros(0, dtype=torch.long),
            persistent=False,
        )

    def slot_ids(self, seq_len: int, device) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device) % self.patch_size
        return idx

    def forward(
        self, content: torch.Tensor, state_ids: torch.Tensor
    ) -> torch.Tensor:
        b, s, c = content.shape
        if c != self.content_dim:
            raise ValueError(f"content dim {c} != {self.content_dim}")
        slot = self.slot_ids(s, content.device)  # [S]
        slot_e = self.slot_embed(slot)[None, :, :]  # [1, S, C]
        out = content + slot_e
        if state_ids is not None:
            if state_ids.shape != (b, s):
                raise ValueError(
                    f"state_ids shape {tuple(state_ids.shape)} != {(b, s)}"
                )
            out = out + self.state_embed(state_ids.long())
        return out


def two_modality_time_sigma(
    time_fn: nn.Module, sigma_seq: torch.Tensor, sigma_struct: torch.Tensor
) -> torch.Tensor:
    """Combine the two modality noise embeddings for the trunk time conditioning.

    ``time_fn`` maps a ``[B]`` sigma to an ``[B, E]`` sinusoidal embedding. Summing
    the two exposes both noise levels while keeping the downstream projection
    shape identical to the sequence-only path.
    """
    return time_fn(sigma_seq) + time_fn(sigma_struct)


def build_state_ids_from_states(states: torch.Tensor) -> torch.Tensor:
    """Map per-residue modality states ``[B, L, 2]`` to per-bit ids ``[B, L*18]``."""
    b, l, m = states.shape
    slot_mod = torch.as_tensor(
        SLOT_MODALITY, device=states.device, dtype=torch.long
    )
    per_slot = states[:, :, slot_mod]  # [B, L, 18]
    return per_slot.reshape(b, l * PATCH_BITS).long()
