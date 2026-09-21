"""
data/mm_codec.py

Single source of truth for the SHARED 19-BIT MULTIMODAL VOCABULARY used by the
joint image+text bitstream-diffusion experiment (CC3M -> zero-shot COCO-30K).

Why 19 bits
-----------
Open-MAGVIT2 (imagenet_256_L) is a Lookup-Free Quantizer: every one of the
2**18 codes is a valid image token, so the image modality fills the entire
18-bit space. To share ONE vocabulary across modalities we add a 19th
"namespace" bit (the MSB, bit index 18):

    bit 18 == 0  ->  IMAGE namespace : payload (bits 0..17) = LFQ code in [0, 2**18)
    bit 18 == 1  ->  TEXT+CONTROL    : payload (bits 0..17) = text token id  in [0, vocab)
                                        OR a reserved marker code in [MARKER_BASE, 2**18)

One transformer position == one 19-bit token of ANY modality, which is exactly
what the SDT backbone's uniform `patch_size` patchify wants (patch_size = 19).

Bit layout is LSB-first to match the existing LFQ canonical layout used by
data/imagenet_lfq_bits._ids_to_bits and utils/image_decode._bits_to_ids:
    bits[..., k] = (code >> k) & 1

Text tokenizer
--------------
Tokenizer-agnostic. Default `o200k_base` (OpenAI tiktoken, ~199,998 vocab ~= 18
bits -> fills the payload densely, killing the ~2.4-bit/token waste of gpt2).
Any tokenizer whose vocab fits below MARKER_BASE is a drop-in via
`cfg.data.text_tokenizer` (o200k_base / cl100k_base / gpt2 / a HF name).

Sequence layout (fixed length, per example)
--------------------------------------------
    [SOS][SOT] t_0 .. t_{Lt-1} [EOT][SOI] i_0 .. i_255 [EOI][EOS]
Marker positions are STRUCTURAL: always clamped clean during training/sampling.
Image tokens are stored in the SAME Hilbert token-grid order as the ImageNet
LFQ cache so utils/image_decode.decode_lfq_bits_to_images can be reused verbatim
on the image slice.

This module is import-cheap: tiktoken / transformers are imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Bit / namespace constants
# ---------------------------------------------------------------------------
BITS_PER_TOKEN: int = 19          # shared patch_size
PAYLOAD_BITS: int = 18            # bits 0..17
NAMESPACE_BIT: int = 18           # bit 18 (MSB): 0=image, 1=text/control
PAYLOAD_MASK: int = (1 << PAYLOAD_BITS) - 1     # 0x3FFFF
NS_FLAG: int = 1 << NAMESPACE_BIT               # 0x40000 — OR'd in for text/control

# Reserved marker codes live at the TOP of the text/control payload, safely
# above any supported text vocabulary (o200k ~200k, llama-3 128k, gpt2 50k).
# 2**18 - MARKER_BASE = 4096 reserved slots.
MARKER_BASE: int = (1 << PAYLOAD_BITS) - 4096   # 258048

# Marker payload offsets (added to MARKER_BASE), then OR'd with NS_FLAG.
_MARKER_OFFSETS: Dict[str, int] = {
    "PAD": 0,
    "SOS": 1,   # start of full sequence
    "EOS": 2,   # end of full sequence
    "SOT": 3,   # start of text
    "EOT": 4,   # end of text
    "SOI": 5,   # start of image
    "EOI": 6,   # end of image
    "SOA": 7,   # start of audio (reserved, future)
    "EOA": 8,   # end of audio  (reserved, future)
}

# Resolved marker CODES (full 19-bit values).
MARKERS: Dict[str, int] = {
    name: NS_FLAG | (MARKER_BASE + off) for name, off in _MARKER_OFFSETS.items()
}


def image_code(lfq_id: int | np.ndarray | torch.Tensor):
    """Image LFQ id (0..2**18-1) -> shared 19-bit code (namespace bit 0 == identity)."""
    return lfq_id  # namespace bit is 0 for image; payload IS the LFQ code


def text_code(token_id: int | np.ndarray | torch.Tensor):
    """Text token id (0..vocab-1) -> shared 19-bit code (namespace bit set)."""
    if isinstance(token_id, torch.Tensor):
        return token_id | NS_FLAG
    if isinstance(token_id, np.ndarray):
        return token_id | NS_FLAG
    return int(token_id) | NS_FLAG


def is_image_code(code: int) -> bool:
    return (int(code) >> NAMESPACE_BIT) == 0


# ---------------------------------------------------------------------------
# code <-> bits  (LSB-first, matches LFQ canonical layout)
# ---------------------------------------------------------------------------
def ids_to_bits(codes: torch.Tensor, n_bits: int = BITS_PER_TOKEN) -> torch.Tensor:
    """
    codes: long tensor [...] in [0, 2**n_bits) -> float bits [..., n_bits] in {0.,1.}.
    bits[..., k] = (code >> k) & 1   (LSB at index 0).
    """
    shifts = torch.arange(n_bits, device=codes.device, dtype=torch.long)
    return ((codes.long().unsqueeze(-1) >> shifts) & 1).to(torch.float32)


def bits_to_ids(bits: torch.Tensor) -> torch.Tensor:
    """
    bits: [..., n_bits] in {0,1}/{0.,1.} -> long codes [...]  = sum bit_k * 2**k.
    Floats are thresholded at 0.5 (continuous-diffusion output safe).
    """
    n_bits = int(bits.shape[-1])
    if bits.is_floating_point():
        b = (bits > 0.5).long()
    else:
        b = bits.long()
    weights = (1 << torch.arange(n_bits, device=bits.device, dtype=torch.long))
    return (b * weights).sum(dim=-1)


def ids_to_bits_np(codes: np.ndarray, n_bits: int = BITS_PER_TOKEN) -> np.ndarray:
    """Numpy LSB-first decomposition; codes [...] -> [..., n_bits] float32."""
    codes64 = codes.astype(np.int64, copy=False)
    shifts = np.arange(n_bits, dtype=np.int64)
    return ((codes64[..., None] >> shifts) & 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Text tokenizer (lazy, pluggable)
# ---------------------------------------------------------------------------
@dataclass
class TextTokenizer:
    name: str
    vocab_size: int
    _encode: Any
    _decode: Any
    backend: str  # "tiktoken" | "hf"
    # Highest id that is safe to decode. tiktoken reserves the top of n_vocab for
    # special tokens (o200k: 199999..200018) that .decode() REJECTS with a KeyError.
    # An untrained model (or the raw bitstream) emits ids anywhere in [0, 2^18), so
    # generated captions routinely contain these -> filter them out before decoding.
    # None -> fall back to vocab_size.
    decodable_vocab: int = None

    def encode(self, text: str) -> List[int]:
        return list(self._encode(text or ""))

    def decode(self, ids: Sequence[int]) -> str:
        lim = self.decodable_vocab or self.vocab_size
        ids = [int(i) for i in ids if 0 <= int(i) < lim]
        try:
            return self._decode(ids)
        except Exception:
            # Belt-and-suspenders: skip any residual undecodable id so a garbage
            # generated caption can never crash training/eval (esp. multi-node,
            # where a rank-0 viz exception cascades into NCCL failures).
            out: List[str] = []
            for t in ids:
                try:
                    out.append(self._decode([t]))
                except Exception:
                    continue
            return "".join(out)


_TOKENIZER_CACHE: Dict[str, TextTokenizer] = {}

# tiktoken encodings we support by name.
_TIKTOKEN_NAMES = {"o200k_base", "cl100k_base", "p50k_base", "r50k_base", "gpt2"}


def load_text_tokenizer(name: str = "o200k_base") -> TextTokenizer:
    """
    Load a text tokenizer by name (cached). tiktoken encodings preferred; any
    other name falls back to a HuggingFace AutoTokenizer.

    tiktoken note (offline HPC): set TIKTOKEN_CACHE_DIR and pre-download the BPE
    ranks during prep; tiktoken reads the cache file on subsequent (offline) use.
    """
    name = str(name)
    if name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[name]

    if name in _TIKTOKEN_NAMES:
        import tiktoken
        enc = tiktoken.get_encoding(name)
        # Highest safely-decodable id = lowest special-token id (o200k: 199999).
        try:
            special_ids = list(enc._special_tokens.values())
            decodable = min(special_ids) if special_ids else int(enc.n_vocab)
        except Exception:
            decodable = int(enc.n_vocab)
        # encode_ordinary ignores special tokens (we manage markers ourselves).
        tok = TextTokenizer(
            name=name,
            vocab_size=int(enc.n_vocab),
            _encode=enc.encode_ordinary,
            _decode=enc.decode,
            backend="tiktoken",
            decodable_vocab=int(decodable),
        )
    else:
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(name, use_fast=True)
        tok = TextTokenizer(
            name=name,
            vocab_size=int(hf.vocab_size),
            _encode=lambda s: hf.encode(s, add_special_tokens=False),
            _decode=lambda ids: hf.decode(ids, skip_special_tokens=True),
            backend="hf",
        )

    if tok.vocab_size > MARKER_BASE:
        raise ValueError(
            f"text tokenizer {name!r} vocab_size={tok.vocab_size} exceeds MARKER_BASE="
            f"{MARKER_BASE}; markers would collide. Use a smaller vocab or raise "
            f"PAYLOAD_BITS."
        )
    _TOKENIZER_CACHE[name] = tok
    return tok


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    name: str
    kind: str          # "marker" | "text" | "image"
    pos_start: int     # inclusive, in transformer positions
    pos_end: int       # exclusive
    code: Optional[int] = None   # fixed code for markers; None otherwise

    @property
    def bit_start(self) -> int:
        return self.pos_start * BITS_PER_TOKEN

    @property
    def bit_end(self) -> int:
        return self.pos_end * BITS_PER_TOKEN


@dataclass
class Layout:
    caption_len_tokens: int          # Lt
    num_image_tokens: int            # 256
    num_positions: int               # N
    segments: List[Segment]

    @property
    def total_bits(self) -> int:
        return self.num_positions * BITS_PER_TOKEN

    # ---- segment accessors -------------------------------------------------
    def segment(self, name: str) -> Segment:
        for s in self.segments:
            if s.name == name:
                return s
        raise KeyError(name)

    def _kind_pos_ranges(self, kind: str) -> List[Tuple[int, int]]:
        return [(s.pos_start, s.pos_end) for s in self.segments if s.kind == kind]

    # ---- boolean BIT masks over the flat sequence (length total_bits) ------
    def _bit_mask(self, kinds: Sequence[str], device=None) -> torch.Tensor:
        m = torch.zeros(self.total_bits, dtype=torch.bool,
                        device=device if device is not None else "cpu")
        for s in self.segments:
            if s.kind in kinds:
                m[s.bit_start:s.bit_end] = True
        return m

    def text_bit_mask(self, device=None) -> torch.Tensor:
        return self._bit_mask(("text",), device)

    def image_bit_mask(self, device=None) -> torch.Tensor:
        return self._bit_mask(("image",), device)

    def marker_bit_mask(self, device=None) -> torch.Tensor:
        return self._bit_mask(("marker",), device)

    # ---- per-bit segment ids (for the optional SDT segment embedding) ------
    # 0 = marker/control, 1 = text, 2 = image, 3 = audio(reserved)
    def segment_ids_per_bit(self, device=None) -> torch.Tensor:
        kind_to_id = {"marker": 0, "text": 1, "image": 2, "audio": 3}
        ids = torch.zeros(self.total_bits, dtype=torch.long,
                          device=device if device is not None else "cpu")
        for s in self.segments:
            ids[s.bit_start:s.bit_end] = kind_to_id.get(s.kind, 0)
        return ids

    @property
    def num_segment_classes(self) -> int:
        return 4

    # ---- a clean marker-code template row [N] (markers filled, rest 0) -----
    def marker_code_template(self) -> np.ndarray:
        row = np.zeros(self.num_positions, dtype=np.int64)
        for s in self.segments:
            if s.kind == "marker":
                row[s.pos_start:s.pos_end] = int(s.code)
        return row


def build_layout(caption_len_tokens: int = 64, num_image_tokens: int = 256) -> Layout:
    """
    Construct the fixed sequence layout:
        [SOS][SOT] text(Lt) [EOT][SOI] image(Ni) [EOI][EOS]
    Positions:
        0:SOS  1:SOT  [2 .. 2+Lt): text  2+Lt:EOT  3+Lt:SOI
        [4+Lt .. 4+Lt+Ni): image   4+Lt+Ni:EOI   5+Lt+Ni:EOS
    """
    Lt = int(caption_len_tokens)
    Ni = int(num_image_tokens)
    p = 0
    segs: List[Segment] = []

    def marker(name: str):
        nonlocal p
        segs.append(Segment(name.lower(), "marker", p, p + 1, code=MARKERS[name]))
        p += 1

    marker("SOS")
    marker("SOT")
    segs.append(Segment("text", "text", p, p + Lt)); p += Lt
    marker("EOT")
    marker("SOI")
    segs.append(Segment("image", "image", p, p + Ni)); p += Ni
    marker("EOI")
    marker("EOS")

    return Layout(Lt, Ni, p, segs)


def resolve_layout(cfg: Any) -> Layout:
    """Build the Layout from cfg.data scalars (no nested structures in ConfigDict)."""
    data = getattr(cfg, "data", None)
    Lt = int(getattr(data, "caption_len_tokens", 64))
    Ni = int(getattr(data, "num_image_tokens", 256))
    return build_layout(Lt, Ni)


# ---------------------------------------------------------------------------
# Assemble one example's code row (used by the prep script)
# ---------------------------------------------------------------------------
def assemble_code_row(
    layout: Layout,
    image_lfq_ids: np.ndarray,   # [num_image_tokens] in [0, 2**18), Hilbert order
    text_token_ids: Sequence[int],  # variable length; truncated/padded to Lt
) -> np.ndarray:
    """
    Returns a [num_positions] int64 row of 19-bit shared-vocab codes:
    markers placed, text truncated/padded with PAD marker, image codes inserted.
    """
    row = layout.marker_code_template()  # markers set, content slots = 0

    tseg = layout.segment("text")
    iseg = layout.segment("image")

    # Text: truncate to Lt, pad with PAD marker code.
    Lt = tseg.pos_end - tseg.pos_start
    ids = list(text_token_ids)[:Lt]
    text_codes = [text_code(int(t)) for t in ids]
    if len(text_codes) < Lt:
        text_codes += [MARKERS["PAD"]] * (Lt - len(text_codes))
    row[tseg.pos_start:tseg.pos_end] = np.asarray(text_codes, dtype=np.int64)

    # Image: namespace bit 0, payload = LFQ id. Expect Hilbert order already.
    img = np.asarray(image_lfq_ids, dtype=np.int64).reshape(-1)
    if img.shape[0] != (iseg.pos_end - iseg.pos_start):
        raise ValueError(
            f"image_lfq_ids has {img.shape[0]} tokens, layout expects "
            f"{iseg.pos_end - iseg.pos_start}"
        )
    if img.min() < 0 or img.max() >= (1 << PAYLOAD_BITS):
        raise ValueError("image LFQ ids out of [0, 2**18) range")
    row[iseg.pos_start:iseg.pos_end] = img  # namespace bit 0 == identity

    return row


# ---------------------------------------------------------------------------
# Split a decoded code row back into modality payloads (used by decode/eval)
# ---------------------------------------------------------------------------
def split_codes(layout: Layout, codes: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    codes: [B, num_positions] long (19-bit shared codes).
    Returns dict with:
        "image_lfq": [B, num_image_tokens] long  (low-18-bit LFQ ids, namespace stripped)
        "text_ids" : [B, Lt] long                (payload; may include marker/PAD codes)
    """
    iseg = layout.segment("image")
    tseg = layout.segment("text")
    image_lfq = (codes[:, iseg.pos_start:iseg.pos_end] & PAYLOAD_MASK)
    text_ids = (codes[:, tseg.pos_start:tseg.pos_end] & PAYLOAD_MASK)
    return {"image_lfq": image_lfq.long(), "text_ids": text_ids.long()}


def decode_text_ids_to_strings(
    text_ids: torch.Tensor,        # [B, Lt] payloads (post namespace-strip)
    tokenizer: TextTokenizer,
) -> List[str]:
    """Drop marker/PAD/out-of-vocab payloads, then tokenizer.decode each row."""
    out: List[str] = []
    vocab = tokenizer.vocab_size
    arr = text_ids.detach().cpu().tolist()
    for row in arr:
        ids = [int(t) for t in row if 0 <= int(t) < vocab]
        out.append(tokenizer.decode(ids))
    return out
