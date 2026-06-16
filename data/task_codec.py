"""Shared fixed-width MSB-first bit codec for the task-driven experiments.

This MUST match the bit-ordering convention used by the rest of
BitstreamDiffusion, so a model trained on task data is byte-compatible
with the existing raw-binary decode path.

Reference convention (data/openwebtext.py::_get_token_to_bits_table):

    data_table[tid, m - 1 - k] = (code >> k) & 1        # raw_binary

i.e. the token id is written MSB-first into `m = bits_per_token` slots:
index 0 holds the most-significant bit, index m-1 the least-significant.
The inverse used at eval time is utils.text_decode.bitstreams_to_token_ids_raw_binary,
which uses powers = 2 ** arange(m-1, -1, -1). The helpers here are the exact
same map, exposed for the Sudoku / TinyGSM datasets and their evaluators.
"""

from __future__ import annotations

import math

import torch


def bits_required(vocab_size: int) -> int:
    """Minimum number of bits to represent ids in [0, vocab_size)."""
    if vocab_size <= 1:
        return 1
    return int(math.ceil(math.log2(int(vocab_size))))


def token_ids_to_bits(token_ids: torch.Tensor, width: int) -> torch.Tensor:
    """token_ids [..., T] (long) -> bits [..., T * width] (long, 0/1), MSB-first.

    bit j of a token (j=0 is the MSB) = (id >> (width - 1 - j)) & 1, which is
    exactly the raw_binary table used by the OWT/LM1B codec.
    """
    if token_ids.dtype != torch.long:
        token_ids = token_ids.long()
    width = int(width)
    # shifts[j] = width-1-j  ==>  index 0 is the MSB.
    shifts = torch.arange(width - 1, -1, -1, device=token_ids.device, dtype=torch.long)
    bits = (token_ids.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(*token_ids.shape[:-1], token_ids.shape[-1] * width).long()


def bits_to_token_ids(bits: torch.Tensor, width: int) -> torch.Tensor:
    """bits [..., T * width] (binary / thresholded) -> token_ids [..., T] (long).

    Inverse of token_ids_to_bits; matches bitstreams_to_token_ids_raw_binary.
    """
    if bits.dtype != torch.long:
        bits = bits.long()
    width = int(width)
    assert bits.shape[-1] % width == 0, (
        f"bit length {bits.shape[-1]} not divisible by width {width}"
    )
    T = bits.shape[-1] // width
    b = bits.reshape(*bits.shape[:-1], T, width)
    powers = (2 ** torch.arange(width - 1, -1, -1, device=bits.device, dtype=torch.long))
    return (b * powers).sum(dim=-1).long()


def token_mask_to_bit_mask(token_mask: torch.Tensor, width: int) -> torch.Tensor:
    """Expand a per-token mask [..., T] to a per-bit mask [..., T * width].

    Preserves dtype (bool stays bool, float stays float).
    """
    width = int(width)
    expanded = token_mask.unsqueeze(-1).expand(*token_mask.shape, width)
    return expanded.reshape(*token_mask.shape[:-1], token_mask.shape[-1] * width).contiguous()
