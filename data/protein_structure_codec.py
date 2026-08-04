"""Dedicated DPLM-2 LFQ structure-token codec and bit-order adapter.

This module is the single source of truth for converting between the DPLM-2
structure tokenizer's outputs and the 13 native bits per residue that the
Bitstream Diffusion model denoises. It is deliberately separate from the
amino-acid sequence codec in ``data/proteins.py``.

Why this file exists and must never be merged with the sequence codec: even
though the two mappings happen to coincide numerically today, the LFQ latent
dimensions carry their own physical meaning, sign convention, and tokenizer
provenance, which must not silently drift with an unrelated sequence codec. The
codec therefore carries a ``convention_hash`` that manifests pin, so any change
to the bit order is caught before cached structure bits can be misread.

Convention (default ``LFQBitCodec``): the released DPLM-2 tokenizer's forward
path (``LFQ.forward``) and its decode-from-id path (``LFQ.get_codebook_entry``)
BOTH multiply the 13 sign bits by descending powers of two,
``mask = 2 ** arange(12, -1, -1)``, so latent dimension 0 is the MOST significant
bit of the released decimal index and dimension 12 is the least. A released id
``n`` therefore satisfies ``sign_d = +1 iff bit (12 - d) of n is set``. The
ascending methods in DPLM's ``lfq.py`` (``decode`` / ``bits_to_indices`` /
``indices_to_bits``) are documented as "not utilized in all the experiments" and
are NOT on the release tokenize/detokenize path.

Last verified on 2026-07-18 by reading the released DPLM sources verbatim:
``src/byprot/utils/protein/tokenize_pdb.py`` (writes the token FASTA via
``VQModel.tokenize`` -> ``LFQ.forward``, serialized unchanged by
``struct_ids_to_seq``), ``src/byprot/models/structok/structok_lfq.py``
(``VQModel.detokenize`` -> ``LFQ.get_codebook_entry``), and
``src/byprot/models/structok/modules/lfq.py`` (both use the descending mask).

  - number of LFQ dimensions: 13 (codebook size 2**13 = 8192);
  - bit order: dimension 0 is the most significant bit (``msb_first``), matching
    the released decimal index; ``bit_d`` is the coefficient of ``2**(12 - d)``;
  - sign mapping: ``sign_d = +1`` when ``bit_d == 1`` else ``-1``;
  - token id (uint16) ``= sum_d bit_d * 2**(12 - d)`` in ``[0, 8191]``.

The 18-bit residue patch consumed by the multimodal model is
``[s_0..s_4 | z_0..z_12]``: five amino-acid sequence bits followed by the 13 LFQ
structure bits in ascending latent-dimension order (``z_i`` is native LFQ
dimension ``i``, i.e. ``z_0`` is the index's most significant bit).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

try:  # torch is available in the training environment; keep numpy-only paths usable without it.
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch is always present in this repo
    torch = None  # type: ignore
    _HAS_TORCH = False


NUM_LFQ_DIMS = 13
STRUCT_CODEBOOK_SIZE = 1 << NUM_LFQ_DIMS  # 8192
SEQ_BITS_PER_RESIDUE = 5
STRUCT_BITS_PER_RESIDUE = NUM_LFQ_DIMS
PATCH_BITS_PER_RESIDUE = SEQ_BITS_PER_RESIDUE + STRUCT_BITS_PER_RESIDUE  # 18


@dataclass(frozen=True)
class LFQBitCodec:
    """Pure, invertible conversions between LFQ token ids, bits, and signs.

    All conversions are vectorized and support numpy arrays and torch tensors of
    any shape; the trailing axis of size ``num_dims`` holds the per-dimension
    bits or signs. The codec carries no learned weights and no coordinate model;
    it is only the bit-order adapter. See ``DPLMStructureTokenizer`` for the
    frozen encoder/decoder that maps coordinates to ids.
    """

    num_dims: int = NUM_LFQ_DIMS
    bit_order: str = "msb_first"  # dim 0 is the MSB of the released DPLM index
    positive_bit_is_plus_one: bool = True  # bit==1 -> sign +1

    def __post_init__(self) -> None:
        if self.num_dims <= 0:
            raise ValueError("num_dims must be positive")
        if self.bit_order not in {"lsb_first", "msb_first"}:
            raise ValueError(
                f"bit_order must be 'lsb_first' or 'msb_first', got {self.bit_order!r}"
            )

    @property
    def codebook_size(self) -> int:
        return 1 << self.num_dims

    def _powers_np(self) -> np.ndarray:
        exps = np.arange(self.num_dims, dtype=np.int64)
        if self.bit_order == "msb_first":
            exps = exps[::-1].copy()
        return (np.int64(1) << exps).astype(np.int64)

    # ------------------------------------------------------------------
    # numpy paths
    # ------------------------------------------------------------------
    def index_to_bits_np(self, index: np.ndarray) -> np.ndarray:
        """Map integer ids ``[...]`` to bits ``[..., num_dims]`` (uint8 in {0,1})."""
        index = np.asarray(index)
        if index.size and (
            index.min() < 0 or index.max() >= self.codebook_size
        ):
            raise ValueError(
                f"structure token id out of range [0, {self.codebook_size - 1}]"
            )
        idx = index.astype(np.int64)[..., None]
        exps = np.arange(self.num_dims, dtype=np.int64)
        if self.bit_order == "msb_first":
            exps = exps[::-1].copy()
        bits = (idx >> exps) & 1
        return bits.astype(np.uint8)

    def bits_to_index_np(self, bits: np.ndarray) -> np.ndarray:
        """Map bits ``[..., num_dims]`` back to integer ids ``[...]`` (int64)."""
        bits = np.asarray(bits)
        if bits.shape[-1] != self.num_dims:
            raise ValueError(
                f"expected last axis {self.num_dims}, got {bits.shape[-1]}"
            )
        b01 = (bits != 0).astype(np.int64)
        return (b01 * self._powers_np()).sum(axis=-1)

    def bits_to_signs_np(self, bits: np.ndarray) -> np.ndarray:
        b01 = np.asarray(bits) != 0
        pos, neg = (
            (1.0, -1.0) if self.positive_bit_is_plus_one else (-1.0, 1.0)
        )
        return np.where(b01, pos, neg).astype(np.float32)

    def signs_to_bits_np(self, signs: np.ndarray) -> np.ndarray:
        s = np.asarray(signs)
        positive = s > 0 if self.positive_bit_is_plus_one else s < 0
        return positive.astype(np.uint8)

    def index_to_signs_np(self, index: np.ndarray) -> np.ndarray:
        return self.bits_to_signs_np(self.index_to_bits_np(index))

    def signs_to_index_np(self, signs: np.ndarray) -> np.ndarray:
        return self.bits_to_index_np(self.signs_to_bits_np(signs))

    # ------------------------------------------------------------------
    # torch paths (used on the data-loading / GPU collate path)
    # ------------------------------------------------------------------
    def index_to_bits_torch(self, index: "torch.Tensor") -> "torch.Tensor":
        """Map ids ``[...]`` to bits ``[..., num_dims]`` (uint8) on the id's device."""
        if not _HAS_TORCH:
            raise RuntimeError("torch is not available")
        idx = index.to(torch.int64)
        if idx.numel() and (
            int(idx.min()) < 0 or int(idx.max()) >= self.codebook_size
        ):
            raise ValueError(
                f"structure token id out of range [0, {self.codebook_size - 1}]"
            )
        exps = torch.arange(
            self.num_dims, device=idx.device, dtype=torch.int64
        )
        if self.bit_order == "msb_first":
            exps = torch.flip(exps, dims=[0])
        bits = (idx.unsqueeze(-1) >> exps) & 1
        return bits.to(torch.uint8)

    def bits_to_index_torch(self, bits: "torch.Tensor") -> "torch.Tensor":
        if not _HAS_TORCH:
            raise RuntimeError("torch is not available")
        if bits.shape[-1] != self.num_dims:
            raise ValueError(
                f"expected last axis {self.num_dims}, got {bits.shape[-1]}"
            )
        b01 = (bits != 0).to(torch.int64)
        exps = torch.arange(
            self.num_dims, device=bits.device, dtype=torch.int64
        )
        if self.bit_order == "msb_first":
            exps = torch.flip(exps, dims=[0])
        powers = torch.ones((), dtype=torch.int64, device=bits.device) << exps
        return (b01 * powers).sum(dim=-1)

    def bits_to_signs_torch(self, bits: "torch.Tensor") -> "torch.Tensor":
        if not _HAS_TORCH:
            raise RuntimeError("torch is not available")
        b01 = bits != 0
        pos, neg = (
            (1.0, -1.0) if self.positive_bit_is_plus_one else (-1.0, 1.0)
        )
        return torch.where(b01, torch.as_tensor(pos), torch.as_tensor(neg)).to(
            torch.float32
        )

    def signs_to_bits_torch(self, signs: "torch.Tensor") -> "torch.Tensor":
        if not _HAS_TORCH:
            raise RuntimeError("torch is not available")
        positive = signs > 0 if self.positive_bit_is_plus_one else signs < 0
        return positive.to(torch.uint8)

    # ------------------------------------------------------------------
    # provenance
    # ------------------------------------------------------------------
    def spec(self) -> Dict[str, object]:
        """Return the frozen convention as a JSON-serializable dict."""
        return {
            "codec": "dplm2_lfq",
            "num_dims": int(self.num_dims),
            "codebook_size": int(self.codebook_size),
            "bit_order": self.bit_order,
            "positive_bit_is_plus_one": bool(self.positive_bit_is_plus_one),
            "patch_layout": "[seq_0..seq_4 | lfq_0..lfq_12]",
            "seq_bits_per_residue": SEQ_BITS_PER_RESIDUE,
            "struct_bits_per_residue": int(self.num_dims),
        }

    def convention_hash(self) -> str:
        """Stable hash of the bit convention, recorded in cache manifests."""
        payload = json.dumps(self.spec(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


DEFAULT_STRUCT_CODEC = LFQBitCodec()


# -----------------------------------------------------------------------------
# 18-bit residue patch assembly
# -----------------------------------------------------------------------------


def assemble_patch_np(
    seq_bits: np.ndarray, struct_bits: np.ndarray
) -> np.ndarray:
    """Concatenate sequence bits ``[..., 5]`` and structure bits ``[..., 13]``.

    Returns the 18-bit residue patch ``[..., 18]`` laid out as
    ``[s_0..s_4 | z_0..z_12]``.
    """
    seq_bits = np.asarray(seq_bits)
    struct_bits = np.asarray(struct_bits)
    if seq_bits.shape[-1] != SEQ_BITS_PER_RESIDUE:
        raise ValueError(f"seq_bits last axis must be {SEQ_BITS_PER_RESIDUE}")
    if struct_bits.shape[-1] != STRUCT_BITS_PER_RESIDUE:
        raise ValueError(
            f"struct_bits last axis must be {STRUCT_BITS_PER_RESIDUE}"
        )
    return np.concatenate([seq_bits, struct_bits], axis=-1)


def split_patch_np(patch_bits: np.ndarray):
    """Inverse of :func:`assemble_patch_np`; returns ``(seq_bits, struct_bits)``."""
    patch_bits = np.asarray(patch_bits)
    if patch_bits.shape[-1] != PATCH_BITS_PER_RESIDUE:
        raise ValueError(f"patch last axis must be {PATCH_BITS_PER_RESIDUE}")
    return (
        patch_bits[..., :SEQ_BITS_PER_RESIDUE],
        patch_bits[..., SEQ_BITS_PER_RESIDUE:],
    )


# -----------------------------------------------------------------------------
# Frozen encoder/decoder wrapper (loads the released DPLM tokenizer)
# -----------------------------------------------------------------------------

DPLM_STRUCT_TOKENIZER_HF_REPO = "airkingbd/struct_tokenizer"
DPLM_GIT_REPO = "https://github.com/bytedance/dplm"


@dataclass
class TokenizerProvenance:
    """Pinned identity of the frozen structure tokenizer, recorded in manifests."""

    hf_repo: str = DPLM_STRUCT_TOKENIZER_HF_REPO
    hf_revision: Optional[str] = None
    dplm_git_commit: Optional[str] = None
    file_hashes: Dict[str, str] = field(default_factory=dict)
    decoder_atom_convention: str = "backbone N, CA, C, O"
    codec_spec: Dict[str, object] = field(
        default_factory=lambda: DEFAULT_STRUCT_CODEC.spec()
    )

    def to_dict(self) -> Dict[str, object]:
        return {
            "hf_repo": self.hf_repo,
            "hf_revision": self.hf_revision,
            "dplm_git_commit": self.dplm_git_commit,
            "file_hashes": dict(self.file_hashes),
            "decoder_atom_convention": self.decoder_atom_convention,
            "codec_spec": self.codec_spec,
        }


class DPLMStructureTokenizer:
    """Lazy wrapper around the frozen DPLM-2 LFQ structure tokenizer.

    The heavy model (encoder + decoder + its folding-model dependencies) lives in
    a separate environment; training GPUs never load it. This wrapper is used
    only offline during cache building and at evaluation/decoding time. It loads
    the released checkpoint on first use and exposes:
      - ``encode(coords) -> uint16 index[L]`` (coordinates to structure ids);
      - ``decode(index) -> coords`` (structure ids to backbone coordinates).

    When the checkpoint is not present it raises an actionable error rather than
    silently producing wrong outputs. The ``codec`` attribute performs the
    id<->bits conversion and is always available without the heavy weights.
    """

    def __init__(
        self,
        *,
        model_path: Optional[Union[str, Path]] = None,
        provenance: Optional[TokenizerProvenance] = None,
        codec: LFQBitCodec = DEFAULT_STRUCT_CODEC,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.provenance = provenance or TokenizerProvenance()
        self.codec = codec
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        if self.model_path is None or not Path(self.model_path).exists():
            raise RuntimeError(
                "DPLM structure tokenizer weights not found. Download them with\n"
                "  python scripts/proteins/setup/download_dplm_tokenizer.py\n"
                f"and point model_path at the local checkpoint (HF repo {self.provenance.hf_repo}). "
                "The encoder/decoder run in the separate DPLM-inference environment "
                "(the dplm-inference extra (uv sync --extra dplm-inference))."
            )
        # The concrete import lives in the DPLM-inference environment. Keep it lazy so
        # importing this module never pulls in the folding-model dependency stack.
        from evaluation.proteins.dplm_struct_tokenizer import (
            load_struct_tokenizer,  # noqa: WPS433
        )

        self._model = load_struct_tokenizer(
            self.model_path, device=self.device
        )
        return self._model

    def encode(self, coords) -> np.ndarray:
        """Encode backbone coordinates ``[L, 4, 3]`` to uint16 structure ids ``[L]``."""
        model = self._load()
        index = model.encode(coords)
        return np.asarray(index, dtype=np.uint16)

    def decode(self, index):
        """Decode uint16 structure ids ``[L]`` to backbone coordinates ``[L, 4, 3]``."""
        model = self._load()
        return model.decode(np.asarray(index, dtype=np.int64))

    def index_to_bits(self, index):
        return self.codec.index_to_bits_np(np.asarray(index))

    def bits_to_index(self, bits):
        return self.codec.bits_to_index_np(np.asarray(bits))


# -----------------------------------------------------------------------------
# Deterministic fixtures for the round-trip test suite
# -----------------------------------------------------------------------------


def build_full_roundtrip_fixture(
    codec: LFQBitCodec = DEFAULT_STRUCT_CODEC,
) -> Dict[str, object]:
    """Exhaustive id->bits->id fixture over the whole codebook, plus a hash.

    Proving the round trip on all 8192 ids guards the exact bit order. The hash
    lets a manifest pin the convention so a silent change is detected.
    """
    ids = np.arange(codec.codebook_size, dtype=np.int64)
    bits = codec.index_to_bits_np(ids)
    recovered = codec.bits_to_index_np(bits)
    exact = bool(np.array_equal(ids, recovered))
    digest = hashlib.sha256(bits.tobytes()).hexdigest()[:16]
    return {
        "codebook_size": int(codec.codebook_size),
        "exact_roundtrip": exact,
        "bits_hash": digest,
        "convention_hash": codec.convention_hash(),
    }
