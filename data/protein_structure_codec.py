"""Convert DPLM structure tokens to bits and back."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# torch is available in the training environment; keep numpy-only paths
# usable without it.
try:
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
    """Pure, invertible conversions between LFQ token ids, bits, and signs."""

    num_dims: int = NUM_LFQ_DIMS
    bit_order: str = "msb_first"  # dim 0 is the MSB of the released DPLM index
    positive_bit_is_plus_one: bool = True  # bit==1 -> sign +1

    def __post_init__(self) -> None:
        if self.num_dims <= 0:
            raise ValueError("num_dims must be positive")
        if self.bit_order not in {"lsb_first", "msb_first"}:
            raise ValueError(
                f"bit_order must be 'lsb_first' or 'msb_first', got "
                f"{self.bit_order!r}"
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
        """Map integer ids [...] to bits [..., num_dims] (uint8 in {0,1})."""
        index = np.asarray(index)
        if index.size and (
            index.min() < 0 or index.max() >= self.codebook_size
        ):
            raise ValueError(
                f"structure token id out of range [0, "
                f"{self.codebook_size - 1}]"
            )
        idx = index.astype(np.int64)[..., None]
        exps = np.arange(self.num_dims, dtype=np.int64)
        if self.bit_order == "msb_first":
            exps = exps[::-1].copy()
        bits = (idx >> exps) & 1
        return bits.astype(np.uint8)

    def bits_to_index_np(self, bits: np.ndarray) -> np.ndarray:
        """Map bits [..., num_dims] back to integer ids [...] (int64)."""
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
    def index_to_bits_torch(self, index: torch.Tensor) -> torch.Tensor:
        """Map ids [...] to bits [..., num_dims] (uint8) on the id's device."""
        if not _HAS_TORCH:
            raise RuntimeError("torch is not available")
        idx = index.to(torch.int64)
        if idx.numel() and (
            int(idx.min()) < 0 or int(idx.max()) >= self.codebook_size
        ):
            raise ValueError(
                f"structure token id out of range [0, "
                f"{self.codebook_size - 1}]"
            )
        exps = torch.arange(
            self.num_dims, device=idx.device, dtype=torch.int64
        )
        if self.bit_order == "msb_first":
            exps = torch.flip(exps, dims=[0])
        bits = (idx.unsqueeze(-1) >> exps) & 1
        return bits.to(torch.uint8)

    def bits_to_index_torch(self, bits: torch.Tensor) -> torch.Tensor:
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

    def bits_to_signs_torch(self, bits: torch.Tensor) -> torch.Tensor:
        if not _HAS_TORCH:
            raise RuntimeError("torch is not available")
        b01 = bits != 0
        pos, neg = (
            (1.0, -1.0) if self.positive_bit_is_plus_one else (-1.0, 1.0)
        )
        return torch.where(b01, torch.as_tensor(pos), torch.as_tensor(neg)).to(
            torch.float32
        )

    def signs_to_bits_torch(self, signs: torch.Tensor) -> torch.Tensor:
        if not _HAS_TORCH:
            raise RuntimeError("torch is not available")
        positive = signs > 0 if self.positive_bit_is_plus_one else signs < 0
        return positive.to(torch.uint8)

    # ------------------------------------------------------------------
    # provenance
    # ------------------------------------------------------------------
    def spec(self) -> dict[str, object]:
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
    """Concatenate sequence bits [..., 5] and structure bits [..., 13]."""
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
    """Inverse of assemble_patch_np; returns (seq_bits, struct_bits)."""
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
    """
    Pinned identity of the frozen structure tokenizer, recorded in
    manifests.
    """

    hf_repo: str = DPLM_STRUCT_TOKENIZER_HF_REPO
    hf_revision: str | None = None
    dplm_git_commit: str | None = None
    file_hashes: dict[str, str] = field(default_factory=dict)
    decoder_atom_convention: str = "backbone N, CA, C, O"
    codec_spec: dict[str, object] = field(
        default_factory=lambda: DEFAULT_STRUCT_CODEC.spec()
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "hf_repo": self.hf_repo,
            "hf_revision": self.hf_revision,
            "dplm_git_commit": self.dplm_git_commit,
            "file_hashes": dict(self.file_hashes),
            "decoder_atom_convention": self.decoder_atom_convention,
            "codec_spec": self.codec_spec,
        }


class DPLMStructureTokenizer:
    """Lazy wrapper around the frozen DPLM-2 LFQ structure tokenizer."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        provenance: TokenizerProvenance | None = None,
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
                f"DPLM structure tokenizer weights not found. Download "
                f"them with\n  python "
                f"scripts/proteins/setup/download_dplm_tokenizer.py\nand "
                f"point model_path at the local checkpoint (HF repo "
                f"{self.provenance.hf_repo}). The encoder/decoder run in "
                f"the separate DPLM-inference environment (the "
                f"dplm-inference extra (uv sync --extra dplm-inference))."
            )
        # The concrete import lives in the DPLM-inference environment.
        # Keep it lazy so importing this module never pulls in the
        # folding-model dependency stack.
        from evaluation.evaluation_drivers.proteins.dplm_struct_tokenizer import (
            load_struct_tokenizer,  # noqa: WPS433
        )

        self._model = load_struct_tokenizer(
            self.model_path, device=self.device
        )
        return self._model

    def encode(self, coords) -> np.ndarray:
        """
        Encode backbone coordinates [L, 4, 3] to uint16 structure ids
        [L].
        """
        model = self._load()
        index = model.encode(coords)
        return np.asarray(index, dtype=np.uint16)

    def decode(self, index):
        """
        Decode uint16 structure ids [L] to backbone coordinates [L, 4,
        3].
        """
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
) -> dict[str, object]:
    """Exhaustive id->bits->id fixture over the whole codebook, plus a hash."""
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
