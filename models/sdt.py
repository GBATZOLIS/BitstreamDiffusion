from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import config_dict

# ============================================================
#  RoPE utilities (SDPA-friendly, compile/AMP safe)
# ============================================================


class RotaryEmbedding(nn.Module):
    """Stateless RoPE. Computes cos/sin for a given seq_len (float32 internally)."""

    def __init__(self, dim: int, base: float = 10_000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires even dim, got dim={dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("l,d->ld", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


# ============================================================
#  Attention wrapper (FLASH when possible, safe under compile/AMP)
# ============================================================


class SelfAttention(nn.Module):
    """
    Principled attention: one parameterization, multiple kernels.

    - Uses PyTorch SDPA (FlashAttention kernels when possible).
    - Supports self-attn and cross-attn.
    - Supports key_padding_mask (True = pad/ignore) and attn_bias.
    - Optional RoPE for SELF-attn.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        *,
        cross_attn: bool = False,
        use_flash_attn: bool = False,
        use_rope: bool = False,
        rope_base: float = 10_000.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        assert self.d_model % self.nhead == 0, "d_model must be divisible by nhead"
        self.head_dim = self.d_model // self.nhead
        self.cross_attn = bool(cross_attn)
        self.dropout = float(dropout)

        if self.cross_attn:
            self.q_proj = nn.Linear(self.d_model, self.d_model, bias=True)
            self.kv_proj = nn.Linear(self.d_model, 2 * self.d_model, bias=True)
            self.qkv_proj = None
        else:
            self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=True)
            self.q_proj = None
            self.kv_proj = None

        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        self.use_rope = bool(use_rope) and (not self.cross_attn)
        if self.use_rope:
            if self.head_dim % 2 != 0:
                raise ValueError(
                    f"use_rope=True requires even head_dim, got head_dim={self.head_dim} "
                    f"(d_model={self.d_model}, nhead={self.nhead})."
                )
            self.rope = RotaryEmbedding(self.head_dim, base=float(rope_base))
        else:
            self.rope = None

        for m in (self.qkv_proj, self.q_proj, self.kv_proj, self.out_proj):
            if m is None:
                continue
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.zeros_(m.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, e = x.shape
        return x.view(b, n, self.nhead, self.head_dim).transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, n, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, h * dh)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        x_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kv_src = x_kv if (self.cross_attn and x_kv is not None) else x

        if self.cross_attn:
            q = self.q_proj(x)
            kv = self.kv_proj(kv_src)
            k, v = kv.chunk(2, dim=-1)
        else:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if self.rope is not None:
            l = q.shape[-2]
            cos, sin = self.rope(l, device=q.device, dtype=q.dtype)
            q, k = _apply_rope(q, k, cos, sin)

        attn_mask = None
        if key_padding_mask is not None:
            kpm = key_padding_mask[:, None, None, :].to(torch.bool)
            attn_mask = kpm

        if attn_bias is not None:
            bias = attn_bias.to(device=q.device, dtype=q.dtype)
            if bias.dim() == 2:
                bias = bias[None, None, :, :]
            if attn_mask is None:
                attn_mask = bias
            else:
                if attn_mask.dtype == torch.bool:
                    neg_inf = torch.finfo(q.dtype).min
                    attn_mask = attn_mask.to(q.dtype) * neg_inf + bias
                else:
                    attn_mask = attn_mask + bias

        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        y = self._merge_heads(y)
        y = self.out_proj(y)
        return y


# ============================================================
#  Helpers: time embedding, blocks, local mixers, heads
# ============================================================


def _build_sinusoidal_table(e: int, device: str):
    inv = torch.exp(-math.log(10_000) * torch.arange(0, e, 2, device=device) / e)
    table = torch.zeros(1, e, device=device)
    table[:, 0::2] = inv
    table[:, 1::2] = inv
    return table


class _SinTimeSigma(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("_freq", _build_sinusoidal_table(dim, "cpu"))

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        freq = self._freq.to(sigma.device)
        phases = sigma.log()[:, None] * freq
        emb = torch.empty_like(phases)
        emb[:, 0::2] = phases[:, 0::2].cos()
        emb[:, 1::2] = phases[:, 1::2].sin()
        return emb


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.Wa = nn.Linear(dim, hidden, bias=True)
        self.Wb = nn.Linear(dim, hidden, bias=True)
        self.Wo = nn.Linear(hidden, dim, bias=True)
        self.drop = nn.Dropout(dropout)
        for m in (self.Wa, self.Wb, self.Wo):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.Wa(x)
        b = F.silu(self.Wb(x))
        y = a * b
        y = self.drop(y)
        return self.Wo(y)


class AdaLNZero(nn.Module):
    """Produces (scale, shift, gate) from time embedding; zero-init => identity start."""

    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(d, 3 * d))
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor):
        orig_dtype = h.dtype
        s, b, g = self.mlp(t_emb).chunk(3, dim=-1)
        h_norm = F.layer_norm(h, h.shape[-1:]).to(orig_dtype)
        h_mod = (1 + s).unsqueeze(1) * h_norm + b.unsqueeze(1)
        return h_mod, g


class PreNormBlockAda(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: Optional[int] = None,
        dropout: float = 0.0,
        use_swiglu: bool = False,
        *,
        use_flash_attn: bool = False,
        use_rope: bool = False,
        rope_base: float = 10_000.0,
    ):
        super().__init__()
        self.dim_ff = int(dim_ff or (4 * d_model))
        self.attn = SelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            cross_attn=False,
            use_flash_attn=use_flash_attn,
            use_rope=use_rope,
            rope_base=rope_base,
        )
        if use_swiglu:
            hidden = int(2 * self.dim_ff / 3)
            self.ff = SwiGLU(d_model, hidden, dropout=dropout)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, self.dim_ff),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.dim_ff, d_model),
            )
        self.drop = nn.Dropout(dropout)
        self.adaln1 = AdaLNZero(d_model)
        self.adaln2 = AdaLNZero(d_model)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h, gate = self.adaln1(x, t_emb)
        y = self.attn(h, key_padding_mask=key_padding_mask, attn_bias=attn_bias)
        x = x + self.drop(y) * gate.unsqueeze(1)

        h, gate = self.adaln2(x, t_emb)
        y = self.ff(h)
        x = x + self.drop(y) * gate.unsqueeze(1)
        return x


class PreNormBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: Optional[int] = None,
        dropout: float = 0.0,
        *,
        use_flash_attn: bool = False,
        use_rope: bool = False,
        rope_base: float = 10_000.0,
    ):
        super().__init__()
        self.dim_ff = int(dim_ff or (4 * d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            cross_attn=False,
            use_flash_attn=use_flash_attn,
            use_rope=use_rope,
            rope_base=rope_base,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, self.dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        y = self.attn(h, key_padding_mask=key_padding_mask, attn_bias=attn_bias)
        x = x + self.drop(y)
        y = self.ff(self.norm2(x))
        return x + self.drop(y)


class RelPosBias1D(nn.Module):
    """Shared-head 1D relative positional bias for patch tokens."""

    def __init__(self, max_distance: int = 64):
        super().__init__()
        assert max_distance >= 1
        self.max_distance = int(max_distance)
        self.num_buckets = 2 * self.max_distance - 1
        self.emb = nn.Embedding(self.num_buckets, 1)
        nn.init.normal_(self.emb.weight, mean=0.0, std=1e-4)

    def forward(self, n: int, device=None, dtype=None) -> torch.Tensor:
        if n <= 1:
            return torch.zeros((n, n), device=device, dtype=dtype)
        idx = torch.arange(n, device=device)
        rel = idx[:, None] - idx[None, :]
        rel = rel.clamp(-self.max_distance + 1, self.max_distance - 1)
        rel_bucket = rel + (self.max_distance - 1)
        bias = self.emb(rel_bucket)
        return bias.squeeze(-1).to(dtype)


class LocalSequenceMixer(nn.Module):
    """Local mixer in sequence space (depthwise + pointwise 1D conv)."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 9,
        dropout: float = 0.0,
        mixer_type: str = "conv",
    ):
        super().__init__()
        self.mixer_type = mixer_type
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        if mixer_type == "conv":
            pad = kernel_size // 2
            self.net = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=pad, groups=d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1),
            )
        else:
            self.net = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mixer_type == "conv":
            y = x.transpose(1, 2)
            y = self.net(y)
            y = y.transpose(1, 2)
            return x + self.dropout(y)
        return x


class HybridSequenceHeadV2(nn.Module):
    """Hybrid head for sequence denoising."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        out_dim: int,
        pos_dim: int,
        content_dim: int,
        noisy_dim: int = 1,
        kernel_size: int = 9,
        dropout: float = 0.0,
        use_cross_attn: bool = True,
        use_local_mixer: bool = False,
        use_self_attn: bool = True,
        *,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.content_dim = int(content_dim)
        self.noisy_dim = int(noisy_dim)
        in_channels = self.content_dim + self.noisy_dim
        self.seq_proj = nn.Linear(in_channels + pos_dim, d_model)

        if use_cross_attn:
            self.adaln_cross = AdaLNZero(d_model)
            self.cross_attn = SelfAttention(
                d_model,
                nhead,
                dropout=dropout,
                cross_attn=True,
                use_flash_attn=use_flash_attn,
                use_rope=False,
            )
        else:
            self.adaln_cross = None
            self.cross_attn = None

        if use_self_attn:
            self.adaln_self = AdaLNZero(d_model)
            self.self_attn = SelfAttention(
                d_model,
                nhead,
                dropout=dropout,
                cross_attn=False,
                use_flash_attn=use_flash_attn,
                use_rope=False,
            )
        else:
            self.adaln_self = None
            self.self_attn = None

        self.local_mixer = (
            LocalSequenceMixer(d_model=d_model, kernel_size=kernel_size, dropout=dropout, mixer_type="conv")
            if use_local_mixer
            else None
        )

        self.adaln_ff = AdaLNZero(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, out_dim)

    def forward(
        self,
        x_denoised: torch.Tensor,
        x_noisy: torch.Tensor,
        pos_feats: torch.Tensor,
        patch_tokens: Optional[torch.Tensor],
        t_emb: torch.Tensor,
        pad_mask_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, s, _ = x_denoised.shape
        content = torch.cat([x_denoised, x_noisy], dim=-1)
        x_aug = torch.cat([content, pos_feats.expand(b, -1, -1)], dim=-1)
        seq = self.seq_proj(x_aug)

        if self.cross_attn is not None and patch_tokens is not None:
            h, _ = self.adaln_cross(seq, t_emb)
            y = self.cross_attn(h, x_kv=patch_tokens, key_padding_mask=pad_mask_tokens)
            seq = seq + self.drop(y)

        if self.self_attn is not None:
            h, _ = self.adaln_self(seq, t_emb)
            y = self.self_attn(h)
            seq = seq + self.drop(y)

        if self.local_mixer is not None:
            seq = self.local_mixer(seq)

        h, gate_ff = self.adaln_ff(seq, t_emb)
        y = self.ff(h)
        seq = seq + self.drop(y) * gate_ff.unsqueeze(1)
        logits = self.out(seq)
        return logits


class OptimalSkipMLPHead(nn.Module):
    """
    Compact head used for bit-space models.

    Supports arbitrary noisy_dim, so continuous-bit self-conditioning can pass
    concatenated [noisy_embed || selfcond_embed] while keeping x_denoised at content_dim.
    """

    def __init__(
        self,
        d_model: int,
        patch_size: int,
        out_dim: int,
        content_dim: int,
        noisy_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.P = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.content_dim = int(content_dim)
        self.noisy_dim = int(noisy_dim)

        self.patch_adapter = nn.Linear(d_model, self.P * self.hidden_dim)
        self.input_adapter = nn.Linear(self.content_dim + self.noisy_dim, self.hidden_dim)
        self.time_proj = nn.Linear(d_model, self.hidden_dim)
        self.adaln = AdaLNZero(self.hidden_dim)
        self.mixer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

        nn.init.xavier_uniform_(self.patch_adapter.weight)
        nn.init.zeros_(self.patch_adapter.bias)
        nn.init.xavier_uniform_(self.input_adapter.weight)
        nn.init.zeros_(self.input_adapter.bias)
        nn.init.normal_(self.time_proj.weight, 0.0, 0.02)
        nn.init.zeros_(self.time_proj.bias)
        for m in self.mixer:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x_denoised: torch.Tensor,
        x_noisy: torch.Tensor,
        patch_tokens: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        b, s, _ = x_denoised.shape
        b2, n, _ = patch_tokens.shape
        if b != b2:
            raise RuntimeError("Batch size mismatch between x_denoised and patch_tokens")

        global_flat = self.patch_adapter(patch_tokens)
        global_feat = global_flat.view(b, n * self.P, self.hidden_dim)
        if global_feat.shape[1] > s:
            global_feat = global_feat[:, :s, :]
        elif global_feat.shape[1] < s:
            global_feat = F.pad(global_feat, (0, 0, 0, s - global_feat.shape[1]))

        local_input = torch.cat([x_denoised, x_noisy], dim=-1)
        local_feat = self.input_adapter(local_input)
        h = global_feat + local_feat

        t_emb_proj = self.time_proj(t_emb)
        h_norm, gate = self.adaln(h, t_emb_proj)
        h_gated = h_norm * gate.unsqueeze(1)
        logits = self.mixer(h_gated)
        return logits


class TokenFullHead(nn.Module):
    """Token-native full output head: no narrow bottleneck before the vocab projection."""

    def __init__(self, d_model: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.time_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, out_dim)
        nn.init.normal_(self.time_proj.weight, 0.0, 0.02)
        nn.init.zeros_(self.time_proj.bias)
        nn.init.normal_(self.out.weight, 0.0, 0.02)
        nn.init.zeros_(self.out.bias)
        for m in self.ff:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(tokens)
        h = self.ff(h)
        h = h + self.time_proj(t_emb).unsqueeze(1)
        return self.out(h)


# ============================================================
#   MAIN MODEL: SequenceVDTContinuousModel
# ============================================================


class SequenceVDTContinuousModel(nn.Module):
    """
    VDT-style diffusion denoiser generalized to 1D sequences.

    Design:
      - Shared trunk across all methods.
      - Token-space models are token-native: full vocab input/output adapters.
      - Bit-space models are compact: low-dimensional side paths and compact heads.
      - Continuous one-hot tokens use additive self-conditioning in trunk width E.
      - Continuous bits use concatenative self-conditioning in width 2C.
    """

    def __init__(self, cfg: config_dict.ConfigDict):
        super().__init__()
        self.cfg = cfg
        assert cfg.framework in ("continuous_score", "discrete_sedd")
        self.is_discrete = cfg.framework == "discrete_sedd"

        # ---- read config ----
        p = int(cfg.model.patch_size)
        e = int(cfg.model.embed_dim)
        l = int(cfg.model.n_blocks)
        h = int(cfg.model.n_heads)
        od = int(getattr(cfg.model, "out_dim", 1))
        dropout = float(getattr(cfg.model, "dropout", 0.0))
        head_type = str(getattr(cfg.model, "head_type", "optimal_skip_mlp")).lower()
        use_adaln = bool(getattr(cfg.model, "use_adaln", True))
        use_swiglu = bool(getattr(cfg.model, "use_swiglu", False))
        use_flash_attn = bool(getattr(cfg.model, "use_flash_attn", False))

        self.representation = str(getattr(cfg.data, "representation", "binary")).lower()
        self.is_token_rep = self.representation == "tokens"
        self.is_binary_rep = self.representation == "binary"
        self.is_discrete_tokens = self.is_discrete and self.is_token_rep
        self.is_discrete_bits = self.is_discrete and self.is_binary_rep
        self.is_continuous_tokens = (not self.is_discrete) and self.is_token_rep
        self.is_continuous_bits = (not self.is_discrete) and self.is_binary_rep

        self.self_condition = bool(getattr(cfg.model, "self_condition", False)) and (not self.is_discrete)
        self.center_inputs = bool(getattr(cfg.model, "center_inputs", True)) and (not self.is_discrete)

        self.use_rope_trunk = bool(getattr(cfg.model, "use_rope_trunk", False))
        self.rope_base = float(getattr(cfg.model, "rope_base", 10_000.0))
        self.abs_pos_mode = str(getattr(cfg.model, "abs_pos_mode", "full")).lower()
        if self.abs_pos_mode not in {"full", "local_only"}:
            raise ValueError(
                f"cfg.model.abs_pos_mode must be 'full' or 'local_only', got '{self.abs_pos_mode}'"
            )

        self.continuous_logit_scaling = str(
            getattr(cfg.model, "continuous_logit_scaling", "none")
        ).lower()
        if self.continuous_logit_scaling == "matched_filter":
            self.continuous_logit_scaling = "matched_filter_residual"
        if self.continuous_logit_scaling not in {
            "none",
            "inv_sigma2",
            "matched_filter_residual",
            "matched_filter_only",
        }:
            raise ValueError(
                f"Unknown cfg.model.continuous_logit_scaling={self.continuous_logit_scaling}"
            )

        self.matched_filter_scale = float(getattr(cfg.model, "matched_filter_scale", 1.0))
        self.matched_filter_clip = getattr(cfg.model, "matched_filter_clip", None)
        self.matched_filter_center = float(
            getattr(
                cfg.model,
                "matched_filter_center",
                getattr(getattr(cfg, "diffusion", {}), "continuous", {}).get("data_center", 0.5)
                if isinstance(getattr(cfg, "diffusion", None), dict)
                else getattr(getattr(getattr(cfg, "diffusion", None), "continuous", None), "data_center", 0.5),
            )
        )
        try:
            self.data_center = float(getattr(cfg.diffusion.continuous, "data_center", 0.5))
        except AttributeError:
            self.data_center = 0.5

        allowed_head_types = {
            "patch_bits_sedd",
            "hybrid_attn_v2",
            "optimal_skip_mlp",
            "token_full",
        }
        if head_type not in allowed_head_types:
            raise ValueError(f"Unsupported head_type: {head_type}")

        self.P, self.E, self.out_dim = p, e, od
        self.head_type = head_type
        self.vocab_size = int(getattr(cfg.data, "vocab_size", 0))

        if self.is_token_rep and self.P != 1:
            raise ValueError(
                "Token-space runs should use patch_size=1 so one LM token maps to one trunk token."
            )
        if self.is_binary_rep and self.is_discrete:
            bits_per_tok = int(getattr(cfg.data, "bits_per_token", 1))
            if self.P not in {1, bits_per_tok}:
                print(
                    f"[warn] discrete binary model uses patch_size={self.P} while bits_per_token={bits_per_tok}."
                )

        # ---- representation-specific channels / adapters ----
        cd_disc = int(getattr(cfg.model, "content_dim_discrete", 1))
        cd_cont = int(getattr(cfg.model, "content_dim_continuous", 1))

        self.token_embed = None
        self.cont_input_proj = None
        self.cont_token_input_proj = None
        self.cont_token_sc_proj = None
        self.q_matrix_type = "none"
        self.scale_by_sigma = False

        if self.is_discrete_tokens:
            self.C = self.E
            self.C_trunk_in = self.E
            self.token_embed = nn.Embedding(self.vocab_size, self.E)
            self.q_matrix_type = cfg.diffusion.discrete.q_matrix_type.lower()
            self.scale_by_sigma = bool(getattr(cfg.model, "scale_by_sigma", True))
        elif self.is_discrete_bits:
            self.C = cd_disc
            self.C_trunk_in = self.C
            self.token_embed = nn.Embedding(self.vocab_size, self.C)
            self.q_matrix_type = cfg.diffusion.discrete.q_matrix_type.lower()
            self.scale_by_sigma = bool(getattr(cfg.model, "scale_by_sigma", True))
        elif self.is_continuous_tokens:
            self.C = self.E
            self.C_trunk_in = self.E
            self.cont_token_input_proj = nn.Linear(self.vocab_size, self.E)
            self.cont_token_sc_proj = nn.Linear(self.vocab_size, self.E) if self.self_condition else None
        elif self.is_continuous_bits:
            self.C = cd_cont
            self.C_trunk_in = self.C * (2 if self.self_condition else 1)
            self.cont_input_proj = None if self.C == 1 else nn.Linear(1, self.C)
        else:
            raise ValueError(f"Unsupported representation/framework combination: {cfg.framework}, {self.representation}")

        # ---- positional features ----
        self.n_fourier_global = int(getattr(cfg.model, "n_fourier_global", 8))
        self.n_fourier_local = int(getattr(cfg.model, "n_fourier_local", 4))
        self.pos_dim = (2 * self.n_fourier_global) + (2 * self.n_fourier_local) + 1

        self.D_trunk_in = self.C_trunk_in + self.pos_dim
        self.patch_dim_in = self.D_trunk_in * self.P
        self.patch_dim_out_content = self.C * self.P

        md = int(getattr(cfg.model, "rpb_max_distance", 64))
        self.rpb = None if md <= 1 else RelPosBias1D(md)
        if use_flash_attn and self.rpb is not None:
            print(
                "[FlashAttn] rpb_max_distance > 1 -> encoder blocks need attn_bias; disabling flash-attn in encoder blocks."
            )
            use_flash_attn_blocks = False
        else:
            use_flash_attn_blocks = use_flash_attn

        # ---- sigma embedding ----
        self.time_fn = _SinTimeSigma(self.E)
        self.time_proj = nn.Linear(self.E, self.E, bias=False)
        self.time_cond = nn.Linear(self.E, self.E)

        # ---- patch / trunk ----
        self.patch_proj = nn.Linear(self.patch_dim_in, self.E)
        self.unpatch_proj_content = None if self.head_type == "token_full" else nn.Linear(self.E, self.patch_dim_out_content)

        dim_ff = int(getattr(cfg.model, "dim_ff", 4 * self.E))
        if use_adaln:
            self.blocks = nn.ModuleList(
                [
                    PreNormBlockAda(
                        self.E,
                        h,
                        dim_ff=dim_ff,
                        dropout=dropout,
                        use_swiglu=use_swiglu,
                        use_flash_attn=use_flash_attn_blocks,
                        use_rope=self.use_rope_trunk,
                        rope_base=self.rope_base,
                    )
                    for _ in range(l)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    PreNormBlock(
                        self.E,
                        h,
                        dim_ff=dim_ff,
                        dropout=dropout,
                        use_flash_attn=use_flash_attn_blocks,
                        use_rope=self.use_rope_trunk,
                        rope_base=self.rope_base,
                    )
                    for _ in range(l)
                ]
            )

        # ---- heads ----
        self.head = None
        if self.head_type == "patch_bits_sedd":
            self.E_head = int(getattr(cfg.model, "head_embed_dim", self.E))
            self.patch_to_bits = nn.Linear(self.E, self.P * self.E_head)
            self.bit_norm = nn.LayerNorm(self.E_head)
            self.bit_out = nn.Linear(self.E_head, self.out_dim)
            nn.init.normal_(self.patch_to_bits.weight, 0.0, 0.02)
            nn.init.zeros_(self.patch_to_bits.bias)
            nn.init.normal_(self.bit_out.weight, 0.0, 0.02)
            nn.init.zeros_(self.bit_out.bias)

        elif self.head_type == "hybrid_attn_v2":
            local_kernel = int(getattr(cfg.model, "head_kernel", 3))
            if local_kernel % 2 == 0:
                local_kernel += 1
            use_local_mixer = bool(getattr(cfg.model, "head_use_local_mixer", False))
            use_cross_attn = bool(getattr(cfg.model, "head_use_cross_attn", True))
            use_self_attn = bool(getattr(cfg.model, "head_use_self_attn", True))
            head_noisy_dim = self.C_trunk_in if (not self.is_discrete) else self.C
            self.head = HybridSequenceHeadV2(
                d_model=self.E,
                nhead=h,
                out_dim=self.out_dim,
                pos_dim=self.pos_dim,
                content_dim=self.C,
                noisy_dim=head_noisy_dim,
                kernel_size=local_kernel,
                dropout=dropout,
                use_cross_attn=use_cross_attn,
                use_local_mixer=use_local_mixer,
                use_self_attn=use_self_attn,
                use_flash_attn=use_flash_attn,
            )

        elif self.head_type == "optimal_skip_mlp":
            hidden_dim = int(getattr(cfg.model, "head_hidden", 256))
            head_noisy_dim = self.C_trunk_in if (not self.is_discrete) else self.C
            self.head = OptimalSkipMLPHead(
                d_model=self.E,
                patch_size=self.P,
                out_dim=self.out_dim,
                content_dim=self.C,
                noisy_dim=head_noisy_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

        elif self.head_type == "token_full":
            self.head = TokenFullHead(d_model=self.E, out_dim=self.out_dim, dropout=dropout)

        # ---- initialization ----
        init_linear_modules = [self.patch_proj, self.time_cond]
        if self.unpatch_proj_content is not None:
            init_linear_modules.append(self.unpatch_proj_content)
        for m in init_linear_modules:
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.time_proj.weight, 0.0, 0.02)

        if self.token_embed is not None:
            nn.init.normal_(self.token_embed.weight, 0.0, 0.02)
        if self.cont_input_proj is not None:
            nn.init.normal_(self.cont_input_proj.weight, 0.0, 0.02)
            nn.init.zeros_(self.cont_input_proj.bias)
        if self.cont_token_input_proj is not None:
            nn.init.normal_(self.cont_token_input_proj.weight, 0.0, 0.02)
            nn.init.zeros_(self.cont_token_input_proj.bias)
        if self.cont_token_sc_proj is not None:
            nn.init.normal_(self.cont_token_sc_proj.weight, 0.0, 0.02)
            nn.init.zeros_(self.cont_token_sc_proj.bias)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fourier_pos_feats(self, s: int, device, dtype):
        """Returns positional features [1, S, pos_dim]."""
        t32 = torch.float32
        pi2 = 2.0 * math.pi
        idx = torch.arange(s, device=device, dtype=t32)

        if self.n_fourier_global > 0:
            if self.abs_pos_mode == "local_only":
                global_feats = torch.zeros(s, 2 * self.n_fourier_global, device=device, dtype=t32)
            else:
                denom_g = max(s - 1, 1)
                k_g = torch.arange(1, self.n_fourier_global + 1, device=device, dtype=t32)
                omega_g = pi2 * k_g / denom_g
                ss = idx[:, None] * omega_g[None, :]
                global_feats = torch.cat([ss.sin(), ss.cos()], dim=-1)
        else:
            global_feats = torch.zeros(s, 0, device=device, dtype=t32)

        if self.n_fourier_local > 0:
            l = (torch.arange(s, device=device, dtype=t32) % self.P)
            denom_l = max(self.P - 1, 1)
            k_l = torch.arange(1, self.n_fourier_local + 1, device=device, dtype=t32)
            omega_l = pi2 * k_l / denom_l
            ll = l[:, None] * omega_l[None, :]
            local_feats = torch.cat([ll.sin(), ll.cos()], dim=-1)
        else:
            local_feats = torch.zeros(s, 0, device=device, dtype=t32)

        center = (self.P - 1) / 2.0
        dist_center = (
            ((torch.arange(s, device=device, dtype=t32) % self.P) - center).abs() / max(center, 1e-6)
        ).unsqueeze(-1)

        pos = torch.cat([global_feats, local_feats, dist_center], dim=-1)
        return pos.unsqueeze(0).to(dtype)

    def _append_pos(self, content: torch.Tensor) -> torch.Tensor:
        b, s, _ = content.shape
        pos = self._fourier_pos_feats(s, content.device, content.dtype)
        return torch.cat([content, pos.expand(b, -1, -1)], dim=-1)

    def _pad_to_multiple(self, x: torch.Tensor, multiple: int):
        b, s, d = x.shape
        r = s % multiple
        pad_len = 0 if r == 0 else (multiple - r)
        if pad_len == 0:
            return x, 0
        pad = torch.zeros(b, pad_len, d, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=1), pad_len

    def _patchify(self, x: torch.Tensor):
        b, s, d = x.shape
        if s % self.P != 0:
            raise RuntimeError("Sequence length must be padded to a multiple of patch size")
        n = s // self.P
        return x.view(b, n, self.P * d), n

    def _unpatchify_content(self, patches_content: torch.Tensor, s_pad: int):
        b, n, pc = patches_content.shape
        if pc != self.C * self.P or n * self.P != s_pad:
            raise RuntimeError("Unpatchify-content shape mismatch")
        return patches_content.view(b, s_pad, self.C)

    def _postprocess_logits(
        self,
        logits: torch.Tensor,
        sigma: torch.Tensor,
        x_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.q_matrix_type == "absorb":
            if self.scale_by_sigma:
                sigma_f32 = sigma.to(torch.float32)
                esigm1 = torch.where(
                    sigma_f32 < 0.5,
                    torch.expm1(sigma_f32),
                    torch.exp(sigma_f32) - 1.0,
                ).clamp_min(torch.finfo(torch.float32).tiny)
                logits = logits - esigm1.log().to(logits.dtype).view(-1, 1, 1)
                logits = logits - math.log(logits.size(-1) - 1)

            if x_t is not None:
                logits = torch.scatter(logits, -1, x_t.long().unsqueeze(-1), 0.0)

            logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
            logits = logits.clamp(-30.0, 30.0)
        return logits

    def _build_continuous_embed_1ch(self, x: torch.Tensor, c_in: torch.Tensor) -> torch.Tensor:
        """x: [B,S] -> [B,S,C]"""
        x_scaled = x * c_in.view(-1, 1)
        if self.cont_input_proj is None:
            return x_scaled.unsqueeze(-1)
        return self.cont_input_proj(x_scaled.unsqueeze(-1))

    def _build_content_continuous_bits(
        self,
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        x0_hat: Optional[torch.Tensor],
    ):
        sigma_data = float(self.cfg.diffusion.continuous.sigma_data)
        c_in = 1.0 / (sigma.pow(2) + sigma_data**2).sqrt()

        x_noisy = x_t.float()
        if self.center_inputs:
            x_noisy = x_noisy - self.data_center
        noisy_emb = self._build_continuous_embed_1ch(x_noisy, c_in)

        if not self.self_condition:
            content = noisy_emb
            return content, content

        if x0_hat is None:
            x_sc = torch.zeros_like(x_t, dtype=torch.float32, device=x_t.device)
            sc_emb = self._build_continuous_embed_1ch(x_sc, c_in)
        else:
            if x0_hat.shape != x_t.shape:
                raise RuntimeError("x0_hat must have same shape as x_t when provided")
            x_sc = x0_hat.float()
            if self.center_inputs:
                x_sc = x_sc - self.data_center
            sc_emb = self._build_continuous_embed_1ch(x_sc, c_in)

        content = torch.cat([noisy_emb, sc_emb], dim=-1)
        return content, content

    def _build_content_continuous_tokens(
        self,
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        x0_hat: Optional[torch.Tensor],
    ):
        if x_t.dim() != 3:
            raise RuntimeError("Continuous token representation expects x_t with shape [B,S,V]")

        sigma_data = float(self.cfg.diffusion.continuous.sigma_data)
        c_in = 1.0 / (sigma.pow(2) + sigma_data**2).sqrt()

        x_noisy = x_t.float()
        c_in = c_in.view(-1, 1, 1).to(device=x_noisy.device, dtype=x_noisy.dtype)

        if self.center_inputs:
            x_noisy = x_noisy - self.data_center

        x_noisy = x_noisy * c_in
        noisy_emb = self.cont_token_input_proj(x_noisy)

        if not self.self_condition:
            return noisy_emb, noisy_emb

        if x0_hat is None:
            sc_emb = torch.zeros_like(noisy_emb)
        else:
            if x0_hat.shape != x_t.shape:
                raise RuntimeError("x0_hat must have same shape as x_t when provided")
            x_sc = x0_hat.float()
            if self.center_inputs:
                x_sc = x_sc - self.data_center
            x_sc = x_sc * c_in
            sc_emb = self.cont_token_sc_proj(x_sc)

        content = noisy_emb + sc_emb
        return content, content

    def _build_content_discrete(self, x_t: torch.Tensor):
        if x_t.dim() != 2:
            raise RuntimeError("Discrete representation expects x_t with shape [B,S]")
        x_tokens = x_t.long()
        content = self.token_embed(x_tokens)
        return content, content

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        x0_hat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Shapes:
          - discrete tokens/bits:   x_t [B,S]
          - continuous bits:        x_t [B,S]
          - continuous one-hot:     x_t [B,S,V]
        """
        if sigma.dim() != 1 or x_t.size(0) != sigma.size(0):
            raise RuntimeError("Expected x_t batch dimension to match sigma [B]")

        if self.is_discrete_tokens or self.is_discrete_bits:
            if x_t.dim() != 2:
                raise RuntimeError("Discrete runs expect x_t with shape [B,S]")
            b, s_orig = x_t.shape
            content, x_skip = self._build_content_discrete(x_t)
        elif self.is_continuous_bits:
            if x_t.dim() != 2:
                raise RuntimeError("Continuous bit runs expect x_t with shape [B,S]")
            b, s_orig = x_t.shape
            content, x_skip = self._build_content_continuous_bits(x_t, sigma, x0_hat)
        elif self.is_continuous_tokens:
            if x_t.dim() != 3:
                raise RuntimeError("Continuous token runs expect x_t with shape [B,S,V]")
            b, s_orig, _ = x_t.shape
            content, x_skip = self._build_content_continuous_tokens(x_t, sigma, x0_hat)
        else:
            raise RuntimeError("Unsupported framework/representation combination")

        x_aug = self._append_pos(content)
        x_pad, pad_len = self._pad_to_multiple(x_aug, self.P)
        s_pad = x_pad.size(1)

        pad_mask_tokens = None
        if pad_len > 0:
            pad_mask_seq = torch.zeros(b, s_pad, dtype=torch.bool, device=x_pad.device)
            pad_mask_seq[:, s_pad - pad_len :] = True
            n = s_pad // self.P
            pad_mask_tokens = pad_mask_seq.view(b, n, self.P).all(dim=-1)
        else:
            n = s_pad // self.P

        tokens_in, n = self._patchify(x_pad)
        tokens = self.patch_proj(tokens_in)

        t_emb = self.time_cond(self.time_proj(self.time_fn(sigma)))
        attn_bias = None if self.rpb is None else self.rpb(n, device=tokens.device, dtype=tokens.dtype)

        for blk in self.blocks:
            if isinstance(blk, PreNormBlockAda):
                tokens = blk(tokens, t_emb, key_padding_mask=pad_mask_tokens, attn_bias=attn_bias)
            else:
                tokens = blk(tokens, key_padding_mask=pad_mask_tokens, attn_bias=attn_bias)

        if self.head_type == "patch_bits_sedd":
            b2, n2, _ = tokens.shape
            bits_flat = self.patch_to_bits(tokens)
            bits = bits_flat.view(b2, n2 * self.P, self.E_head)
            if pad_len > 0:
                bits = bits[:, :s_orig, :]
            logits = self.bit_out(self.bit_norm(bits))

        elif self.head_type == "token_full":
            if self.P != 1:
                raise RuntimeError("token_full head requires patch_size=1")
            tok = tokens[:, :s_orig, :] if pad_len > 0 else tokens
            logits = self.head(tok.contiguous(), t_emb)

        elif self.head_type in ("hybrid_attn_v2", "optimal_skip_mlp"):
            patches_out = self.unpatch_proj_content(tokens)
            x_recon_pad = self._unpatchify_content(patches_out, s_pad)
            x_recon = x_recon_pad[:, :s_orig, :].contiguous() if pad_len > 0 else x_recon_pad.contiguous()
            x_noisy = x_skip[:, :s_orig, :].to(x_recon.dtype)

            if self.head_type == "hybrid_attn_v2":
                pos = self._fourier_pos_feats(s_orig, x_recon.device, x_recon.dtype)
                logits = self.head(
                    x_denoised=x_recon,
                    x_noisy=x_noisy,
                    pos_feats=pos,
                    patch_tokens=tokens,
                    t_emb=t_emb,
                    pad_mask_tokens=pad_mask_tokens,
                )
            else:
                logits = self.head(
                    x_denoised=x_recon,
                    x_noisy=x_noisy,
                    patch_tokens=tokens,
                    t_emb=t_emb,
                )
        else:
            raise RuntimeError(f"Unknown head_type: {self.head_type}")

        if self.is_discrete:
            return self._postprocess_logits(logits, sigma, x_t=x_t)

        return logits
