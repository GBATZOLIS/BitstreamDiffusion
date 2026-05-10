from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import torch


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def required_parity_bits(m: int) -> int:
    r = 0
    while (1 << r) < (m + r + 1):
        r += 1
    return r


@dataclass
class ECCConfig:
    enabled: bool = False
    data_bits: int = 15
    parity_bits: Optional[int] = None  # if None -> auto
    include_overall_parity: bool = True
    unk_token: str = "<unk>"


def ecc_from_cfg(cfg) -> ECCConfig:
    ecc_cfg = getattr(getattr(cfg, "data", None), "ecc", None)
    if ecc_cfg is None:
        return ECCConfig(enabled=False)

    enabled = bool(getattr(ecc_cfg, "enabled", False))
    m = int(getattr(ecc_cfg, "data_bits", 15))
    r = getattr(ecc_cfg, "parity_bits", None)
    if r is None:
        r = required_parity_bits(m)
    else:
        r = int(r)
    include_p0 = bool(getattr(ecc_cfg, "include_overall_parity", True))
    unk_token = str(getattr(ecc_cfg, "unk_token", "<unk>"))
    return ECCConfig(enabled=enabled, data_bits=m, parity_bits=r, include_overall_parity=include_p0, unk_token=unk_token)


def ecc_chunk_len(ecc: ECCConfig) -> int:
    if not ecc.enabled:
        return ecc.data_bits
    return int(ecc.data_bits + int(ecc.parity_bits or 0) + (1 if ecc.include_overall_parity else 0))


def _build_pos_maps(m: int, r: int) -> Tuple[List[int], List[int]]:
    """
    Return:
      data_positions: positions in 1..n that hold data (len=m)
      parity_positions: powers of 2 positions (len=r)
    """
    n = m + r
    parity_positions = [1 << i for i in range(r)]  # 1,2,4,8,...
    data_positions = [pos for pos in range(1, n + 1) if not _is_pow2(pos)]
    if len(data_positions) != m:
        raise RuntimeError(f"Bad mapping: got {len(data_positions)} data positions for m={m}, r={r}, n={n}")
    return data_positions, parity_positions


def ecc_encode_data_bits(data_bits: torch.Tensor, ecc: ECCConfig) -> torch.Tensor:
    """
    data_bits: [..., m] in {0,1}
    returns:   [..., m+r(+1)] as [data, parity, p0] if enabled else data_bits
    """
    if not ecc.enabled:
        return data_bits

    m = int(ecc.data_bits)
    r = int(ecc.parity_bits or required_parity_bits(m))
    n = m + r

    if data_bits.shape[-1] != m:
        raise ValueError(f"ecc_encode_data_bits expects last dim m={m}, got {data_bits.shape[-1]}")

    device = data_bits.device
    dtype = data_bits.dtype

    data_positions, parity_positions = _build_pos_maps(m, r)

    # positions 1..n (we keep a 1-indexed conceptual mapping)
    code = torch.zeros((*data_bits.shape[:-1], n), device=device, dtype=dtype)

    # place data into non-parity positions (in increasing order)
    for i, pos in enumerate(data_positions):
        code[..., pos - 1] = data_bits[..., i]

    # compute parity bits: for each parity position p=2^k, parity = XOR of positions where (idx & p) != 0
    for k, p in enumerate(parity_positions):
        mask_positions = [idx for idx in range(1, n + 1) if (idx & p) != 0]
        if not mask_positions:
            continue
        # XOR over those positions (excluding the parity position itself is OK either way if parity bit is currently 0)
        vals = code[..., torch.tensor([i - 1 for i in mask_positions], device=device)]
        parity = vals.sum(dim=-1) % 2
        # set parity bit at position p
        code[..., p - 1] = parity.to(dtype)

    # extract parity bits in increasing parity position order
    parity_bits = torch.stack([code[..., p - 1] for p in parity_positions], dim=-1)  # [..., r]
    out = torch.cat([data_bits, parity_bits], dim=-1)

    if ecc.include_overall_parity:
        total = (code.sum(dim=-1) % 2).to(dtype)  # parity of n bits
        p0 = total  # choose p0 so that XOR(all n bits) XOR p0 == 0 (even)
        out = torch.cat([out, p0.unsqueeze(-1)], dim=-1)

    return out


def ecc_decode_token_bits(token_bits: torch.Tensor, ecc: ECCConfig) -> Tuple[torch.Tensor, int, bool]:
    """
    token_bits: [..., m+r(+1)] in {0,1}, stored as [data, parity, p0]
    returns:
      data_bits: [..., m] (corrected if possible)
      n_corrections: 0/1 (we count p0-only correction as 1 as well)
      uncorrectable: True if double error detected
    """
    if not ecc.enabled:
        return token_bits, 0, False

    m = int(ecc.data_bits)
    r = int(ecc.parity_bits or required_parity_bits(m))
    n = m + r
    L = m + r + (1 if ecc.include_overall_parity else 0)

    if token_bits.shape[-1] != L:
        raise ValueError(f"ecc_decode_token_bits expects chunk len {L}, got {token_bits.shape[-1]}")

    device = token_bits.device
    dtype = token_bits.dtype

    data = token_bits[..., :m]
    parity = token_bits[..., m:m + r]
    p0 = token_bits[..., m + r] if ecc.include_overall_parity else None

    data_positions, parity_positions = _build_pos_maps(m, r)

    # reconstruct code positions 1..n
    code = torch.zeros((*token_bits.shape[:-1], n), device=device, dtype=dtype)

    for i, pos in enumerate(data_positions):
        code[..., pos - 1] = data[..., i]
    for k, p in enumerate(parity_positions):
        code[..., p - 1] = parity[..., k]

    # syndrome
    syndrome = torch.zeros(token_bits.shape[:-1], device=device, dtype=torch.long)
    for k, p in enumerate(parity_positions):
        mask_positions = [idx for idx in range(1, n + 1) if (idx & p) != 0]
        vals = code[..., torch.tensor([i - 1 for i in mask_positions], device=device)]
        check = (vals.sum(dim=-1) % 2).to(torch.long)  # expected 0 if consistent
        syndrome = syndrome + (check * p)

    # overall parity check t
    if ecc.include_overall_parity:
        total = (code.sum(dim=-1) % 2).to(torch.long)
        t = (total + p0.to(torch.long)) % 2  # 0 means ok
    else:
        # without overall parity, we can't reliably detect double errors; treat syndrome!=0 as correctable
        t = (syndrome != 0).to(torch.long)

    # classify
    s_is_zero = (syndrome == 0)
    t_is_zero = (t == 0)

    # uncorrectable: syndrome != 0 and t == 0  (SECDED)
    uncorrectable = (~s_is_zero) & t_is_zero

    # correctable single-bit in code: syndrome != 0 and t == 1
    single_in_code = (~s_is_zero) & (~t_is_zero)

    # p0-only error: syndrome==0 and t==1
    p0_only = s_is_zero & (~t_is_zero)

    # apply correction in code for single_in_code
    if single_in_code.any():
        # flip bit at (syndrome-1) index for those elements
        idx = (syndrome - 1).clamp(min=0, max=n - 1)  # safe
        flat_code = code.view(-1, n)
        flat_single = single_in_code.view(-1)
        flat_idx = idx.view(-1)
        rows = torch.nonzero(flat_single, as_tuple=False).view(-1)
        cols = flat_idx[rows]
        flat_code[rows, cols] = 1 - flat_code[rows, cols]
        code = flat_code.view_as(code)

    # extract corrected data bits from code (non-parity positions)
    corrected_data = torch.stack([code[..., pos - 1] for pos in data_positions], dim=-1)  # [..., m]

    # count corrections: single bit in code OR p0-only
    n_corr = (single_in_code.to(torch.long) + p0_only.to(torch.long)).to(torch.long)

    return corrected_data, int(n_corr.max().item()) if n_corr.numel() == 1 else n_corr, bool(uncorrectable.max().item()) if uncorrectable.numel() == 1 else uncorrectable


@torch.no_grad()
def ecc_decode_batch_bitstream(bits: torch.Tensor, ecc: ECCConfig) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """
    bits: [B, S_bits] long/bool/float (0/1)
    returns:
      data_bits_stream: [B, T*m] long (concatenated corrected data bits)
      stats: dict with correction rates
      uncorrectable_mask_tokens: [B, T] bool
    """
    if bits.dim() != 2:
        raise ValueError("ecc_decode_batch_bitstream expects [B,S]")

    if bits.is_floating_point():
        bits01 = (bits > 0.5).to(torch.long)
    else:
        bits01 = (bits != 0).to(torch.long)

    if not ecc.enabled:
        # interpret as already "data bits"
        return bits01, {"ecc_enabled": 0.0}, torch.zeros((bits01.size(0), 0), dtype=torch.bool, device=bits01.device)

    m = int(ecc.data_bits)
    r = int(ecc.parity_bits or required_parity_bits(m))
    L = m + r + (1 if ecc.include_overall_parity else 0)

    B, S = bits01.shape
    T = S // L
    S_use = T * L
    bits01 = bits01[:, :S_use].contiguous()

    chunks = bits01.view(B, T, L)  # [B,T,L]

    # decode per token (vectorized-ish by flattening B*T)
    flat = chunks.view(B * T, L)
    data = flat[:, :m]
    parity = flat[:, m:m + r]
    p0 = flat[:, m + r:m + r + 1] if ecc.include_overall_parity else None

    # use the scalar decoder in a batched way by reusing its core logic
    # We call ecc_decode_token_bits on the whole tensor by keeping last dim = L.
    corrected_data, n_corr, uncorrectable = ecc_decode_token_bits(flat, ecc)

    if isinstance(n_corr, torch.Tensor):
        n_corr_t = n_corr.view(B, T)
    else:
        n_corr_t = torch.full((B, T), int(n_corr), device=bits01.device, dtype=torch.long)

    if isinstance(uncorrectable, torch.Tensor):
        uncor_t = uncorrectable.view(B, T)
    else:
        uncor_t = torch.full((B, T), bool(uncorrectable), device=bits01.device, dtype=torch.bool)

    corrected_data = corrected_data.view(B, T, m)
    data_stream = corrected_data.reshape(B, T * m).contiguous()

    # stats
    total_tokens = float(B * T) if T > 0 else 1.0
    n_corrected = float((n_corr_t > 0).sum().item())
    n_uncor = float(uncor_t.sum().item())

    stats = {
        "ecc_enabled": 1.0,
        "ecc_chunk_len": float(L),
        "ecc_tokens_per_sample": float(T),
        "ecc_frac_tokens_corrected": n_corrected / total_tokens,
        "ecc_frac_tokens_uncorrectable": n_uncor / total_tokens,
        "ecc_avg_corrections_per_token": float(n_corr_t.sum().item()) / total_tokens,
    }

    return data_stream, stats, uncor_t
