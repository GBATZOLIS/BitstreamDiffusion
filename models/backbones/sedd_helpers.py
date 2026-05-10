# models/backbones/sedd_helpers.py
import torch
import torch.nn.functional as F
from typing import Optional
from torch import Tensor

# --- Fused Kernels (leave as-is if you rely on TorchScript fusion paths) ---
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


# --- Bias-Dropout-Add-Scale (scripted + typed) ---
@torch.jit.script
def bias_dropout_add_scale(
    x: Tensor,
    bias: Optional[Tensor],
    scale: Tensor,                 
    residual: Optional[Tensor],
    prob: float,
    training: bool,
) -> Tensor:
    # Combine x and bias
    out = x + bias if bias is not None else x
    # Apply dropout with explicit boolean 'training'
    out = F.dropout(out, p=prob, training=training)
    # Scale and add residual (if any)
    out = scale * out
    if residual is not None:
        out = residual + out
    return out


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: Tensor,
    bias: Optional[Tensor],
    scale: Tensor,
    residual: Optional[Tensor],
    prob: float,
) -> Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: Tensor,
    bias: Optional[Tensor],
    scale: Tensor,
    residual: Optional[Tensor],
    prob: float,
) -> Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


# --- Modulation (scripted + typed) ---
@torch.jit.script
def modulate_fused(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    # x * (1 + scale) + shift is standard FiLM-style modulation
    return x * (1 + scale) + shift


# --- Rotary Embeddings utils (EAGER, not scripted) ---
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = int(x.size(-1))
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_eager(qkv: torch.Tensor,
                                cos: torch.Tensor,
                                sin: torch.Tensor) -> torch.Tensor:
    # cos/sin are expected to broadcast to qkv: (B?, S, 3, H?, D)
    return (qkv * cos) + (rotate_half(qkv) * sin)


def apply_rotary_pos_emb(qkv: torch.Tensor,
                         cos: torch.Tensor,
                         sin: torch.Tensor) -> torch.Tensor:
    # Keep the callsite unchanged; use eager rotary to avoid TS bugs.
    return _apply_rotary_pos_emb_eager(qkv, cos, sin)



# --- Rotary Embedding Module (unchanged, no scripting needed here) ---
class Rotary(torch.nn.Module):
    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached: Optional[int] = None
        self.cos_cached: Optional[Tensor] = None
        self.sin_cached: Optional[Tensor] = None

    def forward(self, x: Tensor, seq_dim: int = 1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # cache shapes to match usage in your backbone (B=1, S, groups, ?, D)
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # third "group" acts as identity (no rotation)
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)
        return self.cos_cached, self.sin_cached
