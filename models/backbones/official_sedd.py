# models/backbones/official_sedd.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# FlashAttention is optional. If unavailable we transparently fall back to
# torch.nn.functional.scaled_dot_product_attention (SDPA), so the model
# still runs on machines without a working flash-attn build.
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    _HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_varlen_qkvpacked_func = None
    _HAS_FLASH_ATTN = False

# Local fused helpers and rotary embeddings
from . import sedd_helpers as helpers


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight LN that matches the original numerics (fp32 math + cast back)
# ──────────────────────────────────────────────────────────────────────────────
class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            y = F.layer_norm(x.float(), [self.dim])
        y = y.to(in_dtype)
        w = self.weight.to(in_dtype)
        return y * w[None, None, :]


# ──────────────────────────────────────────────────────────────────────────────
# Sinusoidal time embedding → MLP (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, silu: bool = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU() if silu else nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# ──────────────────────────────────────────────────────────────────────────────
# Transformer block with AdaLN modulation (unchanged, but factored helpers)
# ──────────────────────────────────────────────────────────────────────────────
class DDiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, cond_dim: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: torch.Tensor, rotary_cos_sin, c: torch.Tensor) -> torch.Tensor:
        B, S = x.shape[0], x.shape[1]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        # Mode-aware fused dropout-add-scale
        bdas = (
            helpers.bias_dropout_add_scale_fused_train
            if self.training
            else helpers.bias_dropout_add_scale_fused_inference
        )

        x_skip = x
        x_mod = helpers.modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x_mod)  # [B,S,3D]
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads)

        cos, sin = rotary_cos_sin
        qkv = helpers.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        if _HAS_FLASH_ATTN:
            # FlashAttention expects packed varlen; we pack (B,S) as a single chunk
            qkv_packed = rearrange(qkv, "b s ... -> (b s) ...")
            cu_seqlens = torch.arange(
                0, (B + 1) * S, step=S, dtype=torch.int32, device=qkv_packed.device
            )

            # Ensure FA dtype
            if torch.is_autocast_enabled():
                target = torch.get_autocast_gpu_dtype()
                if qkv_packed.dtype != target:
                    qkv_packed = qkv_packed.to(target)
            if qkv_packed.dtype not in (torch.float16, torch.bfloat16):
                raise RuntimeError(
                    f"FlashAttention expects fp16/bf16, got {qkv_packed.dtype}. "
                    "Check autocast dtype selection."
                )

            x_attn = flash_attn_varlen_qkvpacked_func(qkv_packed, cu_seqlens, S, 0.0, causal=False)
            x_attn = rearrange(x_attn, "(b s) h d -> b s (h d)", b=B)
        else:
            # SDPA fallback: split [B, S, 3, H, D] into (q, k, v) of [B, H, S, D]
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            x_attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            x_attn = rearrange(x_attn, "b h s d -> b s (h d)").contiguous()

        x = bdas(self.attn_out(x_attn), None, gate_msa, x_skip, self.dropout)

        x_mlp = self.mlp(helpers.modulate_fused(self.norm2(x), shift_mlp, scale_mlp))
        x = bdas(x_mlp, None, gate_mlp, x, self.dropout)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Input embeddings for the two regimes
# ──────────────────────────────────────────────────────────────────────────────
class EmbeddingLayer(nn.Module):
    """Discrete: vocabulary lookup → hidden."""
    def __init__(self, dim: int, vocab_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x_idx: torch.Tensor) -> torch.Tensor:
        return self.embedding[x_idx]


class ContinuousInputEmbed(nn.Module):
    """
    OLD (backwards-compatible) continuous embed:
      per-token scalar (or small vector) → hidden via a compact MLP.

    Accepts [B,S] or [B,S,Cin]; default Cin=1 for grayscale MNIST tokens.
    """
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:             # [B,S] → [B,S,1]
            x = x.unsqueeze(-1)
        return self.proj(x)           # [B,S,D]


class SimpleContextContinuousEmbed(nn.Module):
    """
    NEW minimal contextual embed for continuous diffusion on 1D sequences.
    For each position i, take a radius-r window (length W=2r+1) from x_t and
    (optionally) two sigma features, then a 2-layer MLP.

    Input:
      x:     [B, S] or [B, S, 1]   noisy sequence at time t
      sigma: [B] or [B, 1]         per-sample noise level

    Output:
      [B, S, D] (D = hidden_size)
    """
    def __init__(self, hidden_size: int, radius: int = 3, include_sigma: bool = True):
        super().__init__()
        self.r = int(radius)
        self.W = 2 * self.r + 1
        self.include_sigma = include_sigma

        in_dim = self.W
        if include_sigma:
            in_dim += 2  # [log(sigma), sigma]

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def _to_BS(x: torch.Tensor) -> torch.Tensor:
        # Accept [B,S] or [B,S,1]; return [B,S]
        if x.dim() == 3:
            assert x.size(-1) == 1, f"Expected last dim==1, got {x.shape}"
            x = x[..., 0]
        elif x.dim() != 2:
            raise ValueError(f"Expected x shape [B,S] or [B,S,1], got {x.shape}")
        return x

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x = self._to_BS(x).to(torch.float32)     # [B,S]
        B, S = x.shape

        # Build indices for a radius-r window at every position (clamped at edges)
        device = x.device
        offsets = torch.arange(-self.r, self.r + 1, device=device)            # [W]
        centers = torch.arange(S, device=device).unsqueeze(1)                 # [S,1]
        idx = centers + offsets.unsqueeze(0)                                  # [S,W]
        idx = idx.clamp_(0, S - 1)                                            # [S,W]
        idx = idx.unsqueeze(0).expand(B, -1, -1)                              # [B,S,W]

        # Gather windows
        w = x.gather(dim=1, index=idx)                                        # [B,S,W]

        # Optional sigma features (constant per sequence position)
        if self.include_sigma:
            if sigma.dim() == 1:
                sigma = sigma[:, None]                                        # [B,1]
            s = sigma.expand(B, 1).to(x.dtype)                                 # [B,1]
            s_log = s.clamp_min(1e-12).log().clamp_min(-30.0)                 # [B,1]
            # shape to [B,S,2]
            s_feats = torch.stack([s_log.squeeze(1), s.squeeze(1)], dim=1)    # [B,2]
            s_feats = s_feats[:, None, :].expand(B, S, 2)                     # [B,S,2]
            feats = torch.cat([w, s_feats], dim=-1)                            # [B,S,W+2]
        else:
            feats = w                                                          # [B,S,W]

        return self.proj(feats)                                                # [B,S,D]


# ──────────────────────────────────────────────────────────────────────────────
# Final projection with AdaLN modulation (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        nn.init.zeros_(self.linear.weight); nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # 1) run modulation MLP in its native dtype, then cast shift/scale to x.dtype
        wdtype_mod = self.adaLN_modulation.weight.dtype
        shift_scale = self.adaLN_modulation(c.to(wdtype_mod))
        shift, scale = shift_scale[:, None].chunk(2, dim=2)
        shift = shift.to(x.dtype); scale = scale.to(x.dtype)

        # 2) modulate normalized x in the same dtype as x
        x = helpers.modulate_fused(self.norm_final(x), shift, scale)

        # 3) linear expects input dtype == weight dtype when autocast isn't active;
        #    make it explicit to avoid bfloat16/float mismatches.
        wdtype_lin = self.linear.weight.dtype
        out = self.linear(x.to(wdtype_lin))

        return out


# ──────────────────────────────────────────────────────────────────────────────
# SEDD backbone: supports both discrete and continuous inputs
# Legacy head names restored: out_discrete / out_continuous
# ──────────────────────────────────────────────────────────────────────────────
class SEDD(nn.Module):
    """
    Versioned, mode-aware SEDD backbone.

    io_mode:
      - "discrete": legacy-discrete build; names preserved (out_discrete.*)
      - "continuous": continuous-only build (out_continuous.*)
      - "dual": both heads exist (discrete+continuous)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        mcfg = config.model
        self.io_mode = str(getattr(mcfg, "io_mode", "discrete")).lower()

        # Graph type flags
        graph_type = getattr(getattr(config, "graph", {}), "type", "absorb")
        self.absorb = (graph_type == "absorb")

        # Core hyperparams
        hidden_size = mcfg.hidden_size
        cond_dim    = mcfg.cond_dim
        n_heads     = mcfg.n_heads
        n_blocks    = mcfg.n_blocks
        dropout     = mcfg.dropout
        self.scale_by_sigma = bool(getattr(mcfg, "scale_by_sigma", False))

        # Vocab (discrete path)
        tokens = int(getattr(config, "tokens", 256))
        vocab_size_discrete = tokens + (1 if self.absorb else 0)

        # ---- Shared core (always) ----
        self.sigma_map  = TimestepEmbedder(cond_dim)
        self.rotary_emb = helpers.Rotary(hidden_size // n_heads)
        self.blocks = nn.ModuleList([
            DDiTBlock(hidden_size, n_heads, cond_dim, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # ---- Build I/O by mode ----
        build_discrete   = (self.io_mode in {"discrete", "dual"})
        build_continuous = (self.io_mode in {"continuous", "dual"})

        if build_discrete:
            # LEGACY name for compatibility with old checkpoints
            self.vocab_embed  = EmbeddingLayer(hidden_size, vocab_size_discrete)
            self.out_discrete = DDitFinalLayer(hidden_size, vocab_size_discrete, cond_dim)

        if build_continuous:
            # Backwards-compatible choice of continuous embedding
            cont_embed_type = str(getattr(mcfg, "cont_embed_type", "scalar")).lower()
            if cont_embed_type == "context":
                ctx_radius = int(getattr(mcfg, "ctx_radius", 3))
                include_sigma = bool(getattr(mcfg, "include_sigma_in_embed", True))
                self.cont_embed = SimpleContextContinuousEmbed(
                    hidden_size=hidden_size,
                    radius=ctx_radius,
                    include_sigma=include_sigma,
                )
                self._cont_embed_takes_sigma = True
            elif cont_embed_type == "scalar":
                self.cont_embed = ContinuousInputEmbed(in_channels=1, hidden_size=hidden_size)
                self._cont_embed_takes_sigma = False
            else:
                raise ValueError(f"Unknown cont_embed_type: {cont_embed_type!r} (use 'scalar' or 'context')")

            self.out_continuous  = DDitFinalLayer(hidden_size, 1, cond_dim)

    # --- helpers ---
    def _is_discrete_input(self, x: torch.Tensor) -> bool:
        return x.dtype in (torch.int64, torch.int32)

    def _forward_core(self, x: torch.Tensor, sigma: torch.Tensor, cond_dim: int):
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
            c = F.silu(self.sigma_map(sigma)).to(amp_dtype)
            rotary = self.rotary_emb(x)
            for block in self.blocks:
                x = block(x, rotary, c)
        return x, c

    def forward(self, x_in: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        discrete_in = self._is_discrete_input(x_in)

        # Route by input dtype + available heads
        if discrete_in:
            if not hasattr(self, "vocab_embed") or not hasattr(self, "out_discrete"):
                raise RuntimeError("SEDD built without discrete path (io_mode != 'discrete'/'dual').")
            x = self.vocab_embed(x_in)
            x, c = self._forward_core(x, sigma, self.sigma_map.mlp[-1].out_features)
            x = self.out_discrete(x, c)

            if self.scale_by_sigma:
                # Optional original masking/scaling if absorb
                assert self.absorb, "scale_by_sigma only valid for 'absorb' graph."
                esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
                esigm1_log = esigm1.log().to(x.dtype)[:, None, None]
                x = x - esigm1_log - math.log(x.shape[-1] - 1)

            # Zero-out the observed channel (legacy behavior)
            x = torch.scatter(x, -1, x_in[..., None], 0.0)
            return x

        # continuous input
        if not hasattr(self, "cont_embed") or not hasattr(self, "out_continuous"):
            raise RuntimeError("SEDD built without continuous path (io_mode != 'continuous'/'dual').")

        # Accept [B,S] or [B,S,1]
        xin = x_in if x_in.dim() == 3 else x_in.unsqueeze(-1)

        # Backwards-compatible embed call (some variants need sigma, others don't)
        if getattr(self, "_cont_embed_takes_sigma", False):
            x = self.cont_embed(xin.to(torch.float32), sigma)  # [B,S,D]
        else:
            x = self.cont_embed(xin.to(torch.float32))         # [B,S,D]

        x, c = self._forward_core(x, sigma, self.sigma_map.mlp[-1].out_features)
        x = self.out_continuous(x, c)                         # [B,S,1]
        return x
