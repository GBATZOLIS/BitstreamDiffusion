"""
data/mnist_sum_codec.py

Shared code space for the MNIST-Sum image+text corpus.

This mirrors `data/mm_codec.py` but is a separate module on purpose: mm_codec
hardcodes the 19-bit CC3M/CC12M geometry (18-bit LFQ payload + namespace bit),
and those constants back published checkpoints. Nothing here touches that path.

WHAT ONE POSITION IS
--------------------
The transformer patches the flat bitstream every `P` bits (`cfg.model.patch_size`),
so every trunk position is P bits regardless of modality. P is a property of the
*architecture*; how many semantic tokens are packed into those P bits is a layout
choice, exactly as the method permits ("modalities ... may share one position by
concatenating their codes into a single m-bit word; this too is only a layout
choice").

  image position : P raw binary pixels. There is NO image tokenizer and hence no
                   reconstruction floor -- generated and real images are exactly
                   comparable.
  text position  : P // TEXT_BITS tokens of TEXT_BITS=7 bits, packed LSB-first.
                   P=28 -> 4 tokens/position, P=14 -> 2. Both divide 28 evenly,
                   so the same vocabulary serves both and the two patch widths
                   are directly comparable.
  marker position: one reserved P-bit code.

Because the image is raw pixels, *every* P-bit value is a legal image patch, so
markers cannot be carved out of a shared namespace the way mm_codec does. They
are instead identified by position: the template is fixed, markers sit at known
indices and are never denoised, so their code value is only a convention.

TEXT VOCABULARY (7 bits, 128 slots, 40 used)
--------------------------------------------
A closed formal language -- 37 number words (0..36), "+", "=", PAD. No BPE
tokenizer is involved: training one would add a confound for no benefit, and a
hand-specified vocabulary is exact and auditable. Together with the absence of an
image tokenizer this means *neither* modality has a learned codec.

IMAGE ORDERING
--------------
Digit-major, then row-major within a digit: digit 0 (top-left) rows 0..27, then
digit 1 (top-right), 2 (bottom-left), 3 (bottom-right). Each digit is a
contiguous block of 784//P positions, so no patch ever straddles two digits and
a single digit can be masked or conditioned on by itself.

Hilbert ordering is deliberately NOT used here. With P=28 a patch is exactly one
digit row, so horizontal locality is already inside the patch and vertical
locality is sequential; a Hilbert curve over a 28x(784/P) arrangement would be
contrived. This is a documented deviation from the CC3M/CC12M image segments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

# --- text vocabulary -------------------------------------------------------
TEXT_BITS: int = 7
TEXT_VOCAB: int = 1 << TEXT_BITS            # 128 slots, 40 used

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = {20: "twenty", 30: "thirty"}

MAX_SUM: int = 36                            # 4 digits x 9


def num2words(n: int) -> str:
    """0..99 in words, hyphenated above twenty (36 -> 'thirty-six')."""
    if n < 20:
        return _ONES[n]
    t, r = (n // 10) * 10, n % 10
    return _TENS[t] if r == 0 else f"{_TENS[t]}-{_ONES[r]}"


# codes 0..36 are the number words; 37 "+", 38 "=", 39 PAD
NUMBER_WORDS: List[str] = [num2words(i) for i in range(MAX_SUM + 1)]
TOK_PLUS: int = MAX_SUM + 1
TOK_EQUALS: int = MAX_SUM + 2
TOK_PAD: int = MAX_SUM + 3
N_TEXT_CODES: int = TOK_PAD + 1              # 40

WORD_TO_CODE: Dict[str, int] = {w: i for i, w in enumerate(NUMBER_WORDS)}
WORD_TO_CODE["+"] = TOK_PLUS
WORD_TO_CODE["="] = TOK_EQUALS
CODE_TO_WORD: Dict[int, str] = {v: k for k, v in WORD_TO_CODE.items()}
CODE_TO_WORD[TOK_PAD] = ""

MARKER_NAMES = ("SOS", "SOT", "EOT", "SOI", "EOI", "EOS")

DIGIT_SLOT_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")
DIGIT_H = DIGIT_W = 28
PIXELS_PER_DIGIT = DIGIT_H * DIGIT_W         # 784
N_DIGITS = 4


def marker_codes(bits_per_token: int) -> Dict[str, int]:
    """Reserved codes at the top of the P-bit space (position identifies them)."""
    top = (1 << int(bits_per_token)) - 1
    return {name: top - i for i, name in enumerate(MARKER_NAMES)}


def tokens_per_text_position(bits_per_token: int) -> int:
    p = int(bits_per_token)
    if p % TEXT_BITS != 0:
        raise ValueError(
            f"patch_size={p} is not a multiple of TEXT_BITS={TEXT_BITS}; text "
            f"tokens would straddle position boundaries.")
    return p // TEXT_BITS


def image_positions(bits_per_token: int) -> int:
    p = int(bits_per_token)
    if PIXELS_PER_DIGIT % p != 0:
        raise ValueError(
            f"patch_size={p} does not divide {PIXELS_PER_DIGIT} pixels per digit; "
            f"a patch would straddle two digits.")
    return N_DIGITS * (PIXELS_PER_DIGIT // p)


# --- layout ----------------------------------------------------------------
@dataclass
class Segment:
    name: str
    kind: str          # "marker" | "text" | "image"
    pos_start: int
    pos_end: int
    code: Optional[int] = None


@dataclass
class Layout:
    bits_per_token: int
    caption_len_tokens: int      # text tokens (not positions)
    num_text_positions: int
    num_image_tokens: int        # image POSITIONS
    num_positions: int
    segments: List[Segment]

    @property
    def total_bits(self) -> int:
        return self.num_positions * self.bits_per_token

    def segment(self, name: str) -> Segment:
        for s in self.segments:
            if s.name == name:
                return s
        raise KeyError(name)

    def _bit_range(self, s: Segment):
        return s.pos_start * self.bits_per_token, s.pos_end * self.bits_per_token

    def _bit_mask(self, kinds: Sequence[str], device=None) -> torch.Tensor:
        m = torch.zeros(self.total_bits, dtype=torch.bool,
                        device=device if device is not None else "cpu")
        for s in self.segments:
            if s.kind in kinds:
                a, b = self._bit_range(s)
                m[a:b] = True
        return m

    def text_bit_mask(self, device=None):
        return self._bit_mask(("text",), device)

    def image_bit_mask(self, device=None):
        return self._bit_mask(("image",), device)

    def marker_bit_mask(self, device=None):
        return self._bit_mask(("marker",), device)

    def segment_ids_per_bit(self, device=None) -> torch.Tensor:
        """0 = marker/control, 1 = text, 2 = image -- matches mm_codec's ids."""
        kind_to_id = {"marker": 0, "text": 1, "image": 2}
        ids = torch.zeros(self.total_bits, dtype=torch.long,
                          device=device if device is not None else "cpu")
        for s in self.segments:
            a, b = self._bit_range(s)
            ids[a:b] = kind_to_id.get(s.kind, 0)
        return ids

    @property
    def num_segment_classes(self) -> int:
        return 4        # kept at 4 so the segment embedding matches mm_codec

    def marker_code_template(self) -> np.ndarray:
        row = np.zeros(self.num_positions, dtype=np.int64)
        for s in self.segments:
            if s.kind == "marker":
                row[s.pos_start:s.pos_end] = int(s.code)
        return row


def build_layout(bits_per_token: int, caption_len_tokens: int) -> Layout:
    """
    [SOS][SOT] text [EOT][SOI] image [EOI][EOS]  -- same shape as mm_codec.
    `caption_len_tokens` counts TEXT TOKENS; it must fill whole positions.
    """
    p = int(bits_per_token)
    tpp = tokens_per_text_position(p)
    Lt = int(caption_len_tokens)
    if Lt % tpp != 0:
        raise ValueError(
            f"caption_len_tokens={Lt} must be a multiple of {tpp} tokens/position "
            f"at patch_size={p}, otherwise a text position is half-used.")
    n_text_pos = Lt // tpp
    n_img_pos = image_positions(p)
    mk = marker_codes(p)

    pos = 0
    segs: List[Segment] = []

    def marker(name: str):
        nonlocal pos
        segs.append(Segment(name.lower(), "marker", pos, pos + 1, code=mk[name]))
        pos += 1

    marker("SOS")
    marker("SOT")
    segs.append(Segment("text", "text", pos, pos + n_text_pos)); pos += n_text_pos
    marker("EOT")
    marker("SOI")
    segs.append(Segment("image", "image", pos, pos + n_img_pos)); pos += n_img_pos
    marker("EOI")
    marker("EOS")

    return Layout(p, Lt, n_text_pos, n_img_pos, pos, segs)


def resolve_layout(cfg: Any) -> Layout:
    data = getattr(cfg, "data", None)
    p = int(getattr(getattr(cfg, "model", object()), "patch_size",
                    getattr(data, "bits_per_token", 28)))
    Lt = int(getattr(data, "caption_len_tokens", 12))
    return build_layout(p, Lt)


# --- equation <-> token codes ----------------------------------------------
def equation_words(digits: Sequence[int], total: Optional[int] = None) -> List[str]:
    d = [int(x) for x in digits]
    s = int(sum(d)) if total is None else int(total)
    out: List[str] = []
    for i, x in enumerate(d):
        if i:
            out.append("+")
        out.append(num2words(x))
    out += ["=", num2words(s)]
    return out


def equation_to_codes(digits: Sequence[int], total: Optional[int] = None) -> List[int]:
    return [WORD_TO_CODE[w] for w in equation_words(digits, total)]


def codes_to_words(codes: Sequence[int]) -> List[str]:
    return [CODE_TO_WORD.get(int(c), "<unk>") for c in codes if int(c) != TOK_PAD]


# --- packing ---------------------------------------------------------------
def pack_text_codes(layout: Layout, token_codes: Sequence[int]) -> np.ndarray:
    """[Lt] 7-bit token codes -> [num_text_positions] P-bit position codes."""
    tpp = tokens_per_text_position(layout.bits_per_token)
    ids = list(token_codes)[:layout.caption_len_tokens]
    ids += [TOK_PAD] * (layout.caption_len_tokens - len(ids))
    a = np.asarray(ids, dtype=np.int64).reshape(layout.num_text_positions, tpp)
    shifts = (TEXT_BITS * np.arange(tpp, dtype=np.int64))[None, :]
    return (a << shifts).sum(axis=1)


def unpack_text_codes(layout: Layout, pos_codes: torch.Tensor) -> torch.Tensor:
    """[B, num_text_positions] -> [B, Lt] 7-bit token codes."""
    tpp = tokens_per_text_position(layout.bits_per_token)
    shifts = torch.arange(tpp, device=pos_codes.device) * TEXT_BITS
    out = (pos_codes.unsqueeze(-1) >> shifts) & (TEXT_VOCAB - 1)
    return out.reshape(pos_codes.shape[0], -1)


def pack_image_bits(layout: Layout, digit_pixels: np.ndarray) -> np.ndarray:
    """
    [4, 28, 28] binary {0,1} (digit-major: TL, TR, BL, BR)
    -> [num_image_tokens] P-bit codes, digit-major then row-major within a digit.
    """
    p = layout.bits_per_token
    flat = np.asarray(digit_pixels, dtype=np.int64).reshape(N_DIGITS, PIXELS_PER_DIGIT)
    chunks = flat.reshape(-1, p)                       # [4*784/P, P]
    weights = (1 << np.arange(p, dtype=np.int64))[None, :]   # LSB-first
    return (chunks * weights).sum(axis=1)


def unpack_image_bits(layout: Layout, codes: torch.Tensor) -> torch.Tensor:
    """[B, num_image_tokens] -> [B, 4, 28, 28] float {0,1} (TL, TR, BL, BR)."""
    p = layout.bits_per_token
    shifts = torch.arange(p, device=codes.device)
    bits = (codes.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(codes.shape[0], N_DIGITS, DIGIT_H, DIGIT_W).float()


def quadrants_to_canvas(q: torch.Tensor) -> torch.Tensor:
    """[B, 4, 28, 28] -> [B, 1, 56, 56] laid out TL TR / BL BR."""
    top = torch.cat([q[:, 0], q[:, 1]], dim=-1)
    bot = torch.cat([q[:, 2], q[:, 3]], dim=-1)
    return torch.cat([top, bot], dim=-2).unsqueeze(1)


def assemble_code_row(layout: Layout, digit_pixels: np.ndarray,
                      token_codes: Sequence[int]) -> np.ndarray:
    """-> [num_positions] int64 row of P-bit codes, markers placed."""
    row = layout.marker_code_template()
    tseg, iseg = layout.segment("text"), layout.segment("image")
    row[tseg.pos_start:tseg.pos_end] = pack_text_codes(layout, token_codes)
    img = pack_image_bits(layout, digit_pixels)
    if img.shape[0] != (iseg.pos_end - iseg.pos_start):
        raise ValueError(f"image has {img.shape[0]} positions, layout wants "
                         f"{iseg.pos_end - iseg.pos_start}")
    row[iseg.pos_start:iseg.pos_end] = img
    return row


def split_codes(layout: Layout, codes: torch.Tensor) -> Dict[str, torch.Tensor]:
    tseg, iseg = layout.segment("text"), layout.segment("image")
    return {"text_pos": codes[:, tseg.pos_start:tseg.pos_end],
            "image": codes[:, iseg.pos_start:iseg.pos_end]}


# --- bits <-> codes (shared with mm_codec's convention: LSB-first) ----------
def ids_to_bits(codes: torch.Tensor, bits_per_token: int) -> torch.Tensor:
    shifts = torch.arange(int(bits_per_token), device=codes.device)
    return ((codes.unsqueeze(-1) >> shifts) & 1).to(torch.uint8)


def bits_to_ids(bits: torch.Tensor) -> torch.Tensor:
    p = bits.shape[-1]
    shifts = torch.arange(p, device=bits.device)
    return (bits.long() << shifts).sum(-1)
