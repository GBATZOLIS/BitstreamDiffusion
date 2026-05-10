# models/autoregressive/gpt.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- NEW: compile-disable decorator (torch 2.1.x compatible) ---
try:
    # Newer API (may or may not exist in your exact build)
    from torch.compiler import disable as _compile_disable  # type: ignore
except Exception:
    try:
        import torch._dynamo  # type: ignore
        _compile_disable = torch._dynamo.disable  # type: ignore
    except Exception:
        # Final fallback: no-op (shouldn't happen on torch 2.1.2)
        def _compile_disable(fn):
            return fn
        
@dataclass
class ARGPTConfig:
    vocab_size: int = 27
    max_seq_len: int = 512
    n_layer: int = 16
    n_head: int = 16
    d_model: int = 1024
    mlp_mult: float = 4.0
    dropout: float = 0.0
    rope_base: float = 10000.0
    # Prefer flash/mem-efficient SDPA (configured globally in train.py).
    use_flash_attn: bool = True


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, T, H, Dh], cos/sin: [T, 1, Dh]
    return (x * cos) + (_rotate_half(x) * sin)


class RoPECache(nn.Module):
    """
    Dynamic RoPE cache:
      - stores inv_freq in float32
      - lazily materializes cos/sin up to needed length per (device, dtype)
      - supports positions beyond cfg.max_seq_len (useful for long generation w/ sliding KV cache)
    """
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE head_dim must be even."
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq_f32", inv_freq, persistent=False)  # [Dh/2] float32
        self._cache: Dict[Tuple[torch.device, torch.dtype], Tuple[int, torch.Tensor, torch.Tensor]] = {}
        # maps (device, dtype) -> (cached_len, cos[T,1,Dh], sin[T,1,Dh])

        self.head_dim = head_dim

    def _build(self, length: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute in float32 then cast for numerical stability.
        t = torch.arange(length, device=device, dtype=torch.float32)  # [T]
        freqs = torch.einsum("t,d->td", t, self.inv_freq_f32.to(device=device))  # [T, Dh/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, Dh]
        cos = emb.cos().unsqueeze(1).to(dtype=dtype)  # [T,1,Dh]
        sin = emb.sin().unsqueeze(1).to(dtype=dtype)  # [T,1,Dh]
        return cos, sin

    # --- NEW: ensure cache growth happens outside torch.compile/cudagraph capture ---
    @_compile_disable
    def _ensure_cache(self, need: int, device: torch.device, dtype: torch.dtype) -> None:
        key = (device, dtype)
        cached = self._cache.get(key, None)

        if cached is None or cached[0] < need:
            # grow cache geometrically
            new_len = need if cached is None else max(need, int(cached[0] * 1.5))
            cos, sin = self._build(new_len, device=device, dtype=dtype)
            self._cache[key] = (new_len, cos, sin)

    def get(self, start: int, length: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos/sin slices for positions [start : start+length], shape [length,1,Dh].
        """
        if length <= 0:
            raise ValueError("RoPECache.get: length must be > 0")
        need = start + length
        key = (device, dtype)
        cached = self._cache.get(key, None)

        if cached is None or cached[0] < need:
            self._ensure_cache(need, device=device, dtype=dtype)

        _, cos_full, sin_full = self._cache[key]
        return cos_full[start:need], sin_full[start:need]



# KV cache type: per layer tuple(k, v) where
# k,v: [B, H, T_cache, Dh]
KV = Tuple[torch.Tensor, torch.Tensor]
KVCache = List[Optional[KV]]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ARGPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0

        self.d_model = cfg.d_model
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.max_seq_len = int(cfg.max_seq_len)

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.dropout = float(cfg.dropout)
        self.rope = RoPECache(head_dim=self.d_head, base=cfg.rope_base)

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: Optional[KV] = None,
        start_pos: int = 0,
        return_kv: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KV]]:
        """
        x: [B, T, D]
        kv_cache: (k_cache, v_cache) or None
          - k_cache/v_cache: [B, H, T_cache, Dh]
        start_pos: absolute position index of x[:, 0] in the sequence
        return_kv: if True, returns updated kv (for caching)

        Notes:
          - Training path uses kv_cache=None, return_kv=False.
          - Prefill path uses kv_cache=None, return_kv=True (builds cache for prompt).
          - Decode path uses kv_cache!=None and (typically) T=1 (fast incremental).
        """
        B, T, D = x.shape

        qkv = self.qkv(x)                 # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)    # each [B, T, D]

        # [B, T, H, Dh]
        q = q.view(B, T, self.n_head, self.d_head)
        k = k.view(B, T, self.n_head, self.d_head)
        v = v.view(B, T, self.n_head, self.d_head)

        # RoPE for positions [start_pos .. start_pos+T-1]
        # IMPORTANT: In KV-cache mode with a sliding window, positions must be relative
        # to the current cache window length, not absolute global time. Otherwise RoPE
        # becomes inconsistent after cropping.
        if kv_cache is not None:
            k_cache, _ = kv_cache
            start_pos_eff = int(k_cache.size(2))  # position within the current window
        else:
            start_pos_eff = int(start_pos)

        cos, sin = self.rope.get(start=start_pos_eff, length=T, device=x.device, dtype=q.dtype)  # [T,1,Dh]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)


        # to SDPA shape: [B, H, T, Dh]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        updated_kv: Optional[KV] = None

        if kv_cache is None:
            # Standard causal attention over the current chunk
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )  # [B, H, T, Dh]

            if return_kv:
                updated_kv = (k, v)

        else:
            # Incremental decoding: append new keys/values to cache, attend over cache.
            # For correctness + simplicity, we enforce T == 1 in cached mode.
            if T != 1:
                raise ValueError(
                    "Cached attention currently supports T==1 for incremental decoding. "
                    "Use prefill (kv_cache=None, return_kv=True) for prompts."
                )

            k_cache, v_cache = kv_cache  # [B,H,Tc,Dh]
            k_cat = torch.cat([k_cache, k], dim=2)
            v_cat = torch.cat([v_cache, v], dim=2)

            # Sliding window to cap memory
            if k_cat.size(2) > self.max_seq_len:
                k_cat = k_cat[:, :, -self.max_seq_len :, :].contiguous()
                v_cat = v_cat[:, :, -self.max_seq_len :, :].contiguous()

            # No causal mask needed for T==1 since there are no future keys in k_cat.
            y = F.scaled_dot_product_attention(
                q, k_cat, v_cat,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )  # [B, H, 1, Dh]

            if return_kv:
                updated_kv = (k_cat, v_cat)

        y = y.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        return self.proj(y), updated_kv


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_mult: float, dropout: float):
        super().__init__()
        d_hidden = int(hidden_mult * d_model)
        self.w1 = nn.Linear(d_model, d_hidden, bias=False)
        self.w2 = nn.Linear(d_model, d_hidden, bias=False)
        self.w3 = nn.Linear(d_hidden, d_model, bias=False)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.drop(self.w1(x) * F.silu(self.w2(x))))


class Block(nn.Module):
    def __init__(self, cfg: ARGPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, hidden_mult=cfg.mlp_mult, dropout=cfg.dropout)
        self.drop = nn.Dropout(float(cfg.dropout))

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: Optional[KV] = None,
        start_pos: int = 0,
        return_kv: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KV]]:
        a, new_kv = self.attn(self.norm1(x), kv_cache=kv_cache, start_pos=start_pos, return_kv=return_kv)
        x = x + self.drop(a)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x, new_kv


class AutoregressiveGPT(nn.Module):
    def __init__(self, cfg: ARGPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(float(cfg.dropout))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)

        # weight tying
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Training/eval forward (no KV caching):
          idx: [B, T] -> logits: [B, T, V]
        """
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len={self.cfg.max_seq_len}")

        x = self.tok_emb(idx)  # [B,T,D]
        x = self.drop(x)
        for blk in self.blocks:
            x, _ = blk(x, kv_cache=None, start_pos=0, return_kv=False)
        x = self.norm_f(x)
        return self.lm_head(x)  # [B,T,V]

    def _init_empty_kv_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> KVCache:
        """
        Create an empty KV cache for each layer: [B,H,0,Dh].
        """
        H = self.cfg.n_head
        Dh = self.cfg.d_model // self.cfg.n_head
        empty_k = torch.empty((batch_size, H, 0, Dh), device=device, dtype=dtype)
        empty_v = torch.empty((batch_size, H, 0, Dh), device=device, dtype=dtype)
        return [(empty_k, empty_v) for _ in range(self.cfg.n_layer)]

    @torch.no_grad()
    def forward_with_kv_cache(
        self,
        idx: torch.Tensor,
        *,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
        return_kv: bool = True,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Forward that can build/update a KV cache.

        - Prefill: kv_cache=None, idx=[B,T] (T<=max_seq_len), start_pos=0
          returns logits + full per-layer kv for the prompt.

        - Decode: kv_cache!=None, idx=[B,1], start_pos=pos_of_this_token
          returns logits + updated cache.

        Returns:
          logits: [B, T, V]
          new_cache: list length n_layer of (k,v) if return_kv else None
        """
        B, T = idx.shape
        if kv_cache is None:
            if T > self.cfg.max_seq_len:
                raise ValueError(
                    f"Prefill length {T} exceeds max_seq_len={self.cfg.max_seq_len}. "
                    "Truncate the prompt before calling forward_with_kv_cache."
                )
        else:
            if T != 1:
                raise ValueError("Decode with kv_cache expects idx shape [B,1].")

        x = self.tok_emb(idx)  # [B,T,D]
        x = self.drop(x)

        new_cache: Optional[KVCache] = None
        if return_kv:
            if kv_cache is None:
                # create cache container; layers will fill it
                new_cache = [None for _ in range(self.cfg.n_layer)]
            else:
                new_cache = [None for _ in range(self.cfg.n_layer)]

        # propagate through blocks
        for li, blk in enumerate(self.blocks):
            layer_kv = None if kv_cache is None else kv_cache[li]
            x, updated = blk(
                x,
                kv_cache=layer_kv,
                start_pos=start_pos,
                return_kv=return_kv,
            )
            if return_kv and new_cache is not None:
                new_cache[li] = updated  # type: ignore[assignment]

        x = self.norm_f(x)
        logits = self.lm_head(x)  # [B,T,V]
        return logits, new_cache

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        """
        KV-cache-enabled autoregressive generation.

        Behavior matches your old generate() semantics (top-k, temperature), but is much faster.
        """
        if max_new_tokens <= 0:
            return idx

        B, T0 = idx.shape
        if T0 <= 0:
            raise ValueError("generate() requires a non-empty prompt idx of shape [B,T].")

        # Match old behavior: only keep last max_seq_len tokens as context.
        if T0 > self.cfg.max_seq_len:
            idx = idx[:, -self.cfg.max_seq_len :]
            T0 = idx.size(1)

        # If user disables KV cache, fall back to slow path (original semantics).
        if not use_kv_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.cfg.max_seq_len :]
                logits = self(idx_cond)[:, -1, :]
                logits = logits / max(temperature, 1e-6)

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                    logits = torch.where(
                        logits < v[:, [-1]],
                        torch.full_like(logits, -1e10),
                        logits,
                    )

                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_id], dim=1)
            return idx

        # ---- KV cache path ----
        # Prefill prompt to build cache and get logits for last prompt token.
        logits, cache = self.forward_with_kv_cache(idx, kv_cache=None, start_pos=0, return_kv=True)
        assert cache is not None
        last_logits = logits[:, -1, :]  # [B,V]
        pos = T0  # next token position

        for _ in range(max_new_tokens):
            # sample next token from last_logits
            logits_step = last_logits / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits_step, k=min(top_k, logits_step.size(-1)))
                logits_step = torch.where(
                    logits_step < v[:, [-1]],
                    torch.full_like(logits_step, -1e10),
                    logits_step,
                )

            probs = F.softmax(logits_step, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [B,1]
            idx = torch.cat([idx, next_id], dim=1)

            # decode step: run one-token forward using cache
            step_logits, cache = self.forward_with_kv_cache(
                next_id,
                kv_cache=cache,
                start_pos=pos,
                return_kv=True,
            )
            assert cache is not None
            last_logits = step_logits[:, -1, :]  # [B,V]
            pos += 1

        return idx
