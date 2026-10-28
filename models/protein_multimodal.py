"""Add sequence and structure inputs to the diffusion model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

SEQ_BITS = 5
STRUCT_BITS = 13
PATCH_BITS = SEQ_BITS + STRUCT_BITS  # 18

# Modality-state codes. These MUST match data.protein_tasks.
#   0 ABSENT    the modality has no valid value here
#   1 OBSERVED  the modality is known and clean
#   2 NOISY     the modality is a generation target
#   3 DROPPED   the modality was observed but the condition was dropped for
#               classifier-free guidance. A distinct state is
#               required: without it the "unconditional" half of a CFG batch is
#               still told those residues are OBSERVED and still receives sigma
#               0 for them, so the guidance direction is not a true
#               conditional-minus-unconditional difference.
NUM_STATES = 4

# Slot -> modality (0 sequence, 1 structure), for building per-bit
# modality ids.
SLOT_MODALITY = tuple([0] * SEQ_BITS + [1] * STRUCT_BITS)


class MultimodalBitEmbeddings(nn.Module):
    """Learned per-bit slot and modality-state embeddings added to content."""

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


class ModalityTimeFusion(nn.Module):
    """Asymmetric fusion of the two modality noise embeddings (P0-2)."""

    def __init__(self, dim: int, init_noise: float = 0.02):
        super().__init__()
        self.dim = int(dim)
        self.proj = nn.Linear(2 * self.dim, self.dim, bias=False)
        with torch.no_grad():
            eye = torch.eye(self.dim)
            weight = torch.cat([eye, eye], dim=1)
            if init_noise > 0.0:
                # Deterministic-scale perturbation: large enough that
                # the map is genuinely asymmetric at init (so the P0-2
                # regression test is meaningful from step 0), small
                # enough that the fused embedding is numerically the old
                # sum to ~1%.
                weight = weight + torch.randn_like(weight) * (
                    float(init_noise) / math.sqrt(2.0 * self.dim)
                )
            self.proj.weight.copy_(weight)

    def forward(
        self, emb_seq: torch.Tensor, emb_struct: torch.Tensor
    ) -> torch.Tensor:
        return self.proj(torch.cat([emb_seq, emb_struct], dim=-1))


def two_modality_time_sigma(
    time_fn: nn.Module,
    sigma_seq: torch.Tensor,
    sigma_struct: torch.Tensor,
    fusion: ModalityTimeFusion | None = None,
) -> torch.Tensor:
    """
    Combine the two modality noise embeddings for the trunk time
    conditioning.
    """
    emb_seq = time_fn(sigma_seq)
    emb_struct = time_fn(sigma_struct)
    if fusion is not None:
        return fusion(emb_seq, emb_struct)
    return emb_seq + emb_struct


class GlobalTaskConditioning(nn.Module):
    """
    Chain length and task shape, injected into the AdaLN conditioning
    vector.
    """

    # 4 states x 2 modalities = 8 occupancy fractions.
    _N_STATE_FEATS = NUM_STATES * 2

    def __init__(self, dim: int, n_length_freq: int = 16):
        super().__init__()
        self.dim = int(dim)
        self.n_length_freq = int(n_length_freq)
        # Fourier features of log-length, plus the raw normalised
        # length.
        self.n_feats = 2 * self.n_length_freq + 1 + self._N_STATE_FEATS
        self.mlp = nn.Sequential(
            nn.Linear(self.n_feats, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        nn.init.normal_(self.mlp[0].weight, 0.0, 0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        # Log-spaced frequencies over log-length. Registered as a buffer
        # so the device follows the module and the values are pinned in
        # the checkpoint.
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(64.0), self.n_length_freq)
        )
        self.register_buffer("_length_freqs", freqs, persistent=True)

    def features(
        self, lengths: torch.Tensor, states: torch.Tensor | None
    ) -> torch.Tensor:
        """[B, n_feats] from chain lengths [B] and states [B, L, 2]."""
        lengths = lengths.to(torch.float32).clamp_min(1.0)
        log_len = torch.log(lengths)[:, None]  # [B, 1]
        phase = log_len * self._length_freqs.to(log_len.device)[None, :]
        feats = [phase.sin(), phase.cos(), log_len / math.log(1000.0)]
        if states is None:
            feats.append(
                torch.zeros(
                    lengths.shape[0],
                    self._N_STATE_FEATS,
                    device=lengths.device,
                    dtype=torch.float32,
                )
            )
        else:
            b, _l, m = states.shape
            occ = []
            for modality in range(m):
                col = states[:, :, modality]
                for state in range(NUM_STATES):
                    occ.append((col == state).to(torch.float32).mean(dim=1))
            feats.append(torch.stack(occ, dim=-1))  # [B, 8]
        return torch.cat(feats, dim=-1).to(torch.float32)

    def forward(
        self, lengths: torch.Tensor, states: torch.Tensor | None, dtype
    ) -> torch.Tensor:
        return self.mlp(self.features(lengths, states)).to(dtype)


class ResiduePositionFeatures(nn.Module):
    """
    Absolute index, terminus markers and fractional position, per
    residue.
    """

    def __init__(
        self,
        dim: int,
        n_freq: int = 32,
        use_fraction: bool = True,
        max_period: float = 2048.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.n_freq = int(n_freq)
        self.use_fraction = bool(use_fraction)
        # sin/cos of absolute index, 2 log-distances, 2 terminus flags,
        # and optionally the fractional position and its complement.
        self.n_feats = 2 * self.n_freq + 4 + (2 if self.use_fraction else 0)
        self.proj = nn.Linear(self.n_feats, self.dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        inv = torch.exp(
            -math.log(float(max_period))
            * torch.arange(self.n_freq, dtype=torch.float32)
            / max(1, self.n_freq - 1)
        )
        self.register_buffer("_inv_freq", inv, persistent=True)

    def features(
        self, n_residues: int, lengths: torch.Tensor, device
    ) -> torch.Tensor:
        """[B, N, n_feats] for N residue slots and true lengths [B]."""
        b = int(lengths.shape[0])
        idx = torch.arange(n_residues, device=device, dtype=torch.float32)
        idx_b = idx[None, :].expand(b, n_residues)  # [B, N]
        lens = lengths.to(device=device, dtype=torch.float32).clamp_min(1.0)[
            :, None
        ]  # [B, 1]

        phase = idx_b[:, :, None] * self._inv_freq.to(device)[None, None, :]
        feats = [phase.sin(), phase.cos()]

        from_n = idx_b
        from_c = (lens - 1.0 - idx_b).clamp_min(0.0)
        feats.append(torch.log1p(from_n)[:, :, None] / math.log(1000.0))
        feats.append(torch.log1p(from_c)[:, :, None] / math.log(1000.0))
        feats.append((idx_b < 1.0).to(torch.float32)[:, :, None])
        feats.append((from_c < 1.0).to(torch.float32)[:, :, None])
        if self.use_fraction:
            frac = idx_b / (lens - 1.0).clamp_min(1.0)
            feats.append(frac[:, :, None])
            feats.append((1.0 - frac)[:, :, None])
        return torch.cat(feats, dim=-1)

    def forward(
        self, n_residues: int, lengths: torch.Tensor, device, dtype
    ) -> torch.Tensor:
        return self.proj(self.features(n_residues, lengths, device)).to(dtype)


def build_state_ids_from_states(states: torch.Tensor) -> torch.Tensor:
    """Map per-residue modality states [B, L, 2] to per-bit ids [B, L*18]."""
    b, l, m = states.shape
    slot_mod = torch.as_tensor(
        SLOT_MODALITY, device=states.device, dtype=torch.long
    )
    per_slot = states[:, :, slot_mod]  # [B, L, 18]
    return per_slot.reshape(b, l * PATCH_BITS).long()

