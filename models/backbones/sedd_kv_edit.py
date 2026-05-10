# models/backbones/sedd_kv_edit.py
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_func

# Reuse your helper kernels (modulation + fused residual path)
from . import sedd_helpers as helpers


@torch.jit.script
def _modulate_fused(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


def _split_ada(blk, c_slice: torch.Tensor):
    # returns: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    return blk.adaLN_modulation(c_slice)[:, None].chunk(6, dim=2)


class SEDDKVSingleEditEngineLogR:
    """
    Single-token edit engine for the Official SEDD backbone that returns
    LOG-RATIOS L_{·|j} (diagonal zeroed at the conditioning token j).
    No changes to the backbone are required.

    Usage:
      engine = SEDDKVSingleEditEngineLogR(wrapper.model)
      cache  = engine.build_cache(indices, sigma)  # base pass (one forward)
      edited = indices.clone(); edited[..., s] = j_prime  # supply any per-site plan
      L_all  = engine.logratios_for_all_site_edits(cache, edited)  # (B,S,V)
      L_site = L_all[b, s, :]
    """

    def __init__(self, sedd_backbone):
        """
        Args:
            sedd_backbone: models/backbones/official_sedd.SEDD instance
                           (i.e., `OfficialSEDDWrapper(...).model`)
        """
        self.m = sedd_backbone.eval()
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    @torch.no_grad()
    def build_cache(self, indices: torch.Tensor, sigma: torch.Tensor) -> Dict:
        """
        One normal forward pass that also saves per-block base K,V for all sites.

        Returns:
            dict with:
                indices, sigma, c, rotary
                KV: list[(K_base, V_base)] at each block with shape (B,S,H,D)
                log_r_full: base log-ratios with L_{j|j}=0 (B,S,V)
        """
        m = self.m

        with torch.cuda.amp.autocast(enabled=True, dtype=self.amp_dtype):
            x = m.vocab_embed(indices).to(self.amp_dtype)   # (B,S,D)
            c = F.silu(m.sigma_map(sigma)).to(self.amp_dtype)  # (B, Dcond)
            rotary = m.rotary_emb(x)

            KV: List[Tuple[torch.Tensor, torch.Tensor]] = []
            h = x
            # We reproduce the per-block pre-attn path to capture K,V_base
            for blk in m.blocks:
                # Modulate LN1(h)
                shift_msa, scale_msa, _, _, _, _ = _split_ada(blk, c)
                h_mod = _modulate_fused(blk.norm1(h), shift_msa, scale_msa)  # (B,S,D)

                # QKV and rotary
                qkv = blk.attn_qkv(h_mod)  # (B,S,3D)
                qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=blk.n_heads)
                cos, sin = rotary
                qkv = helpers.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

                # Extract base K,V for all tokens in this block
                K_base = qkv[:, :, 1].contiguous()  # (B,S,H,D)
                V_base = qkv[:, :, 2].contiguous()
                KV.append((K_base, V_base))

                # Full block forward (same as original)
                h = blk(h, rotary, c)

            # Head → log-ratios for the unedited sequence
            log_r_full = m.output_layer(h, c)  # (B,S,V)

            # Optional absorb scaling (kept identical to your backbone)
            if getattr(m, "scale_by_sigma", False):
                assert m.absorb, "scale_by_sigma=True is only valid for absorb graph."
                esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(log_r_full.dtype)
                log_r_full = log_r_full - esigm1_log[:, None, None] - torch.log(
                    torch.tensor(log_r_full.shape[-1] - 1.0, device=log_r_full.device, dtype=log_r_full.dtype)
                )

            # Zero L_{j|j}
            log_r_full = torch.scatter(log_r_full, -1, indices[..., None], 0.0)

        return {
            "indices": indices,
            "sigma": sigma,
            "c": c,
            "rotary": rotary,
            "KV": KV,
            "log_r_full": log_r_full,
        }

    @torch.no_grad()
    def logratios_for_all_site_edits(
        self,
        cache: Dict,
        edited_tokens: torch.Tensor,               # (B,S) — per-site token j' we want to condition on
        *,
        chunk_size: Optional[int] = 2048,
        zero_diagonal: bool = True,
    ) -> torch.Tensor:
        """
        Compute L_{·| edited_tokens[b,s]} at every site (b,s) in one pass by
        reusing base K,V everywhere except the edited slot per site.

        Returns:
            log_r_all: (B,S,V) with diagonal zeroed at the edited token if requested.
        """
        m = self.m
        device = cache["indices"].device
        B, S = cache["indices"].shape
        V = m.output_layer.linear.out_features
        amp = self.amp_dtype

        # Flatten selection of (b,s)
        b_idx = torch.arange(B, device=device).repeat_interleave(S)  # (B*S,)
        s_idx = torch.arange(S, device=device).repeat(B)             # (B*S,)
        t_new = edited_tokens.reshape(-1).to(device)                 # (B*S,)

        out = torch.empty((B * S, V), device=device, dtype=m.output_layer.linear.weight.dtype)

        total = b_idx.numel()
        if not chunk_size or chunk_size <= 0:
            chunk_size = total

        with torch.cuda.amp.autocast(enabled=True, dtype=amp):
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                E = end - start

                b = b_idx[start:end]      # (E,)
                s = s_idx[start:end]      # (E,)
                t = t_new[start:end]      # (E,)

                x_s = m.vocab_embed.embedding[t].view(E, 1, -1).to(amp)   # (E,1,D)
                c_slice = cache["c"][b]                                   # (E,Dc)
                cos, sin = cache["rotary"]                                # cos/sin: (1,S,3,1,D)

                # --- gather rotary at the edited positions (no broadcast over S!) ---
                # (1,S,3,1,D) -> index_select over S -> (1,E,3,1,D) -> (E,1,3,1,D)
                cos_pos = cos.index_select(1, s).squeeze(0).unsqueeze(1)   # (E,1,3,1,D)
                sin_pos = sin.index_select(1, s).squeeze(0).unsqueeze(1)   # (E,1,3,1,D)

                for ell, blk in enumerate(m.blocks):
                    Kb, Vb = cache["KV"][ell]          # (B,S,H,D)
                    K = Kb[b].contiguous()             # (E,S,H,D)
                    Vv = Vb[b].contiguous()            # (E,S,H,D)

                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _split_ada(blk, c_slice)
                    x_mod = _modulate_fused(blk.norm1(x_s), shift_msa[:, :1], scale_msa[:, :1])  # (E,1,D)

                    qkv = blk.attn_qkv(x_mod)                                              # (E,1,3D)
                    qkv = rearrange(qkv, "e s (three h d) -> e s three h d", three=3, h=blk.n_heads)  # (E,1,3,H,D)
                    qkv = helpers.apply_rotary_pos_emb(qkv, cos_pos.to(qkv.dtype), sin_pos.to(qkv.dtype))
                    # shapes stay (E,1,3,H,D) — no expansion to S

                    q_s = qkv[:, :, 0].to(K.dtype)     # (E,1,H,D)
                    k_s = qkv[:, :, 1].squeeze(1)      # (E,H,D)
                    v_s = qkv[:, :, 2].squeeze(1)      # (E,H,D)

                    row = torch.arange(E, device=device)
                    K[row, s] = k_s                    # (E,S,H,D) <- (E,H,D)
                    Vv[row, s] = v_s

                    attn_out = flash_attn_func(q_s, K, Vv, dropout_p=0.0, causal=False)   # (E,1,H,D)
                    attn_out = rearrange(attn_out, "e s h d -> e s (h d)")

                    x_s = helpers.bias_dropout_add_scale_fused_inference(
                        blk.attn_out(attn_out), None, gate_msa[:, :1], x_s, blk.dropout
                    )
                    x_mlp = blk.mlp(_modulate_fused(blk.norm2(x_s), shift_mlp[:, :1], scale_mlp[:, :1]))
                    x_s = helpers.bias_dropout_add_scale_fused_inference(
                        x_mlp, None, gate_mlp[:, :1], x_s, blk.dropout
                    )

                # Final head at the edited site → LOG-RATIOS
                shift, scale = m.output_layer.adaLN_modulation(c_slice)[:, None].chunk(2, dim=2)
                x_s = _modulate_fused(m.output_layer.norm_final(x_s), shift, scale)
                log_r = m.output_layer.linear(x_s)[:, 0, :]  # (E,V)

                if getattr(m, "scale_by_sigma", False):
                    sigma_e = cache["sigma"][b]
                    esigm1 = torch.where(sigma_e < 0.5, torch.expm1(sigma_e), sigma_e.exp() - 1)
                    log_r = log_r - esigm1.log().to(log_r.dtype)[:, None] - torch.log(
                        torch.tensor(V - 1.0, device=log_r.device, dtype=log_r.dtype)
                    )

                if zero_diagonal:
                    # IMPORTANT: zero the edited token (L_{j'|j'} = 0)
                    log_r.scatter_(1, t[:, None], 0.0)

                out[start:end] = log_r

        return out.view(B, S, V)

    @torch.no_grad()
    def binary_flip_all_sites(self, indices: torch.Tensor, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience for V=2: returns (base_log_r, flipped_log_r), both with diagonals zeroed correctly.
        """
        cache = self.build_cache(indices, sigma)
        flipped = 1 - indices
        edited = self.logratios_for_all_site_edits(cache, flipped, zero_diagonal=True)
        return cache["log_r_full"], edited
