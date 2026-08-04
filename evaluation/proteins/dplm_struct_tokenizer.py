"""Load the frozen DPLM-2 LFQ structure tokenizer for offline encode and decode.

This module is the concrete bridge between backbone coordinates and the pinned
13-bit LFQ structure code that the Bitstream Diffusion model denoises. The heavy
DPLM-2 tokenizer (a GVP encoder plus an ESMFold-style structure decoder, with
its folding-model dependency stack) lives in a separate DPLM-inference
environment described by ``the dplm-inference extra (uv sync --extra dplm-inference)``; training GPUs never
load it. Every DPLM import here is lazy so that this module imports cleanly with
only numpy and torch present.

The public entry point is :func:`load_struct_tokenizer`. It returns an object
exposing:
  - ``encode(coords)`` mapping backbone coordinates ``[L, 4, 3]`` (atoms N, CA,
    C, O) to uint16 structure ids ``[L]`` in ``[0, 8191]``;
  - ``decode(index)`` mapping structure ids ``[L]`` back to backbone coordinates
    ``[L, 4, 3]`` (atoms N, CA, C, O);
  - ``provenance`` describing the pinned tokenizer identity.

Bit order is pinned by ``DEFAULT_STRUCT_CODEC`` (``msb_first``), which matches the
released DPLM LFQ convention exactly: ``LFQ.forward`` (used to write the released
token ids) and ``LFQ.get_codebook_entry`` (used to detokenize an id) both use the
descending mask ``2 ** arange(12, -1, -1)`` (latent dim 0 = MSB of the id). This
wrapper never round-trips through DPLM's own integer index: ``encode`` runs the
DPLM encoder, sign quantizes the continuous LFQ latent, and maps the per-dimension
bits to an id with the codec; ``decode`` expands the id to codec bits, turns them
into per-dimension signs, and feeds the ``[1, L, 13]`` sign tensor straight into
the DPLM decoder (``detokenize``'s ndim-3 path). Because the codec now agrees with
DPLM's descending convention, the ids this wrapper produces are directly
comparable to the released structure-token ids, and an ``encode`` followed by
``decode`` round-trips the exact latent signs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

import numpy as np
import torch

from data.protein_structure_codec import (
    DEFAULT_STRUCT_CODEC,
    DPLM_GIT_REPO,
    DPLM_STRUCT_TOKENIZER_HF_REPO,
    NUM_LFQ_DIMS,
    STRUCT_CODEBOOK_SIZE,
    LFQBitCodec,
    TokenizerProvenance,
)

# Backbone atom slots inside the 37-atom AlphaFold/OpenFold atom convention that
# the DPLM tokenizer consumes and emits: N, CA, C occupy slots 0, 1, 2 and O
# occupies slot 4 (slot 3 is CB). These four slots are the backbone the codec
# round-trips.
ATOM37_BACKBONE_SLOTS = (0, 1, 2, 4)
NUM_ATOM37 = 37
NUM_BACKBONE_ATOMS = 4

_INSTALL_HINT = (
    "The frozen DPLM-2 structure tokenizer is unavailable. Download the "
    "checkpoint with\n"
    "  python scripts/proteins/setup/download_dplm_tokenizer.py\n"
    "and install the DPLM-inference environment listed in "
    "the dplm-inference extra (uv sync --extra dplm-inference) (see scripts/proteins/setup/setup_evaluation.sh). "
    f"The weights live in the Hugging Face repo {DPLM_STRUCT_TOKENIZER_HF_REPO} "
    f"and the encoder/decoder code is the official DPLM package at {DPLM_GIT_REPO}."
)


def _to_numpy(array) -> np.ndarray:
    """Return a numpy view of a numpy array or torch tensor without copying dtype."""
    if isinstance(array, np.ndarray):
        return array
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _looks_like_hf_repo(model_path: Union[str, Path]) -> bool:
    """Return True when the path is an unresolved Hugging Face repo id, not a local file."""
    text = str(model_path)
    if Path(text).exists():
        return False
    return "/" in text and not text.startswith(("/", "./", "../", "~"))


def _checkpoint_sha256(model_path: Union[str, Path]) -> dict:
    """Best-effort content hash of the local tokenizer checkpoint for provenance.

    Returns a mapping of file name to hex digest. Hashing is skipped silently
    when the checkpoint cannot be located or read so that provenance never
    blocks loading.
    """
    path = Path(model_path)
    candidates = []
    if path.is_file():
        candidates.append(path)
    elif path.is_dir():
        for name in ("dplm2_struct_tokenizer.ckpt", ".hydra/config.yaml"):
            candidate = path / name
            if candidate.is_file():
                candidates.append(candidate)
    hashes = {}
    for candidate in candidates:
        try:
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            hashes[candidate.name] = digest.hexdigest()
        except OSError:
            continue
    return hashes


class _DPLMStructTokenizerRunner:
    """Thin runtime wrapper around a loaded DPLM-2 LFQ structure tokenizer.

    Instances are produced by :func:`load_struct_tokenizer`; construct them
    through that function so the DPLM model and its dependencies are imported
    lazily and any failure raises an actionable error. The wrapper holds the
    frozen tokenizer module and reuses ``DEFAULT_STRUCT_CODEC`` for every
    id-to-bits conversion, keeping the bit order pinned.
    """

    def __init__(
        self,
        model,
        *,
        device: str,
        codec: LFQBitCodec,
        provenance: TokenizerProvenance,
    ) -> None:
        self._model = model
        self.device = device
        self.codec = codec
        self.provenance = provenance

    def encode(self, coords) -> np.ndarray:
        """Encode backbone coordinates ``[L, 4, 3]`` to uint16 structure ids ``[L]``.

        ``coords`` supplies the backbone atoms N, CA, C, O as a numpy array or
        torch tensor. The DPLM encoder produces the continuous LFQ latent, which
        is sign quantized and then mapped to a pinned id with the codec.
        """
        coords_np = _to_numpy(coords).astype(np.float32)
        if coords_np.ndim != 3 or coords_np.shape[1:] != (
            NUM_BACKBONE_ATOMS,
            3,
        ):
            raise ValueError(
                "coords must have shape [L, 4, 3] with backbone atoms N, CA, C, O; "
                f"got shape {tuple(coords_np.shape)}"
            )
        length = coords_np.shape[0]

        atom37 = np.zeros((length, NUM_ATOM37, 3), dtype=np.float32)
        for backbone_index, slot in enumerate(ATOM37_BACKBONE_SLOTS):
            atom37[:, slot, :] = coords_np[:, backbone_index, :]

        atom_positions = torch.from_numpy(atom37).unsqueeze(0).to(self.device)
        res_mask = torch.ones(
            (1, length), dtype=torch.float32, device=self.device
        )
        seq_length = torch.tensor(
            [length], dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            pre_quant, _ = self._model.encode(
                atom_positions=atom_positions,
                mask=res_mask,
                seq_length=seq_length,
            )
        latent = _to_numpy(pre_quant[0]).astype(np.float32)
        if latent.shape != (length, NUM_LFQ_DIMS):
            raise RuntimeError(
                "DPLM encoder returned an unexpected latent shape "
                f"{tuple(latent.shape)}, expected {(length, NUM_LFQ_DIMS)}"
            )

        # LFQ sign quantization: each latent dimension collapses to its sign, and
        # the codec maps the per-dimension bits to the pinned little-endian id.
        signs = np.where(latent > 0.0, 1.0, -1.0).astype(np.float32)
        bits = self.codec.signs_to_bits_np(signs)
        index = self.codec.bits_to_index_np(bits)
        return index.astype(np.uint16)

    def decode(self, index) -> np.ndarray:
        """Decode uint16 structure ids ``[L]`` to backbone coordinates ``[L, 4, 3]``.

        The id is expanded to codec bits, turned into per-dimension signs, and
        fed straight into the DPLM decoder. The returned coordinates are the
        backbone atoms N, CA, C, O in that order.
        """
        index_np = _to_numpy(index).astype(np.int64).reshape(-1)
        if index_np.size and (
            index_np.min() < 0 or index_np.max() >= STRUCT_CODEBOOK_SIZE
        ):
            raise ValueError(
                f"structure token id out of range [0, {STRUCT_CODEBOOK_SIZE - 1}]"
            )
        length = index_np.shape[0]

        # index -> codec bits -> per-dimension signs, matching the LFQ codebook
        # entries the DPLM decoder expects (bit 1 is sign +1, bit 0 is sign -1).
        signs = self.codec.index_to_signs_np(index_np).astype(np.float32)
        quant = torch.from_numpy(signs).unsqueeze(0).to(self.device)
        res_mask = torch.ones(
            (1, length), dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            decoder_out = self._model.detokenize(quant, res_mask=res_mask)
        atom37 = _to_numpy(decoder_out["atom37_positions"][0]).astype(
            np.float32
        )
        if atom37.shape != (length, NUM_ATOM37, 3):
            raise RuntimeError(
                "DPLM decoder returned an unexpected atom37 shape "
                f"{tuple(atom37.shape)}, expected {(length, NUM_ATOM37, 3)}"
            )
        coords = atom37[:, list(ATOM37_BACKBONE_SLOTS), :]
        return coords.astype(np.float32)


def _load_dplm_tokenizer(model_path: Union[str, Path], device: str):
    """Lazily import DPLM and load the frozen tokenizer, raising on any failure."""
    try:
        from byprot.models.utils import get_struct_tokenizer
    except Exception as exc:  # noqa: BLE001 - surface every DPLM import failure uniformly
        raise RuntimeError(_INSTALL_HINT) from exc

    try:
        model = get_struct_tokenizer(str(model_path), eval_mode=True)
        model = model.to(device).eval()
    except Exception as exc:  # noqa: BLE001 - checkpoint or config load failure
        raise RuntimeError(_INSTALL_HINT) from exc
    return model


def load_struct_tokenizer(
    model_path: Union[str, Path],
    device: str = "cpu",
) -> _DPLMStructTokenizerRunner:
    """Load the frozen DPLM-2 LFQ structure tokenizer from ``model_path``.

    ``model_path`` is the local checkpoint directory produced by
    ``scripts/proteins/setup/download_dplm_tokenizer.py`` (it may also be an
    unresolved Hugging Face repo id, which DPLM fetches on demand). The heavy
    DPLM package is imported lazily; if the checkpoint or the DPLM package is
    missing this raises a ``RuntimeError`` pointing at the download script and
    ``the dplm-inference extra (uv sync --extra dplm-inference)``.

    Returns an object exposing ``encode``, ``decode``, and ``provenance``.
    """
    if model_path is None:
        raise RuntimeError(_INSTALL_HINT)
    if not Path(model_path).exists() and not _looks_like_hf_repo(model_path):
        raise RuntimeError(_INSTALL_HINT)

    model = _load_dplm_tokenizer(model_path, device)

    provenance = TokenizerProvenance(
        hf_repo=DPLM_STRUCT_TOKENIZER_HF_REPO,
        file_hashes=_checkpoint_sha256(model_path),
    )
    return _DPLMStructTokenizerRunner(
        model,
        device=device,
        codec=DEFAULT_STRUCT_CODEC,
        provenance=provenance,
    )
