"""Encode and decode structures with the frozen DPLM tokenizer."""

from __future__ import annotations

import hashlib
from pathlib import Path

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

# Backbone atom slots inside the 37-atom AlphaFold/OpenFold atom
# convention that the DPLM tokenizer consumes and emits: N, CA, C occupy
# slots 0, 1, 2 and O occupies slot 4 (slot 3 is CB). These four slots
# are the backbone the codec round-trips.
ATOM37_BACKBONE_SLOTS = (0, 1, 2, 4)
NUM_ATOM37 = 37
NUM_BACKBONE_ATOMS = 4

_INSTALL_HINT = (
    f"The DPLM structure tokenizer is unavailable. Run ./setup.sh eval to "
    f"install its dependencies and download the code and weights. The "
    f"weights live in the Hugging Face repo "
    f"{DPLM_STRUCT_TOKENIZER_HF_REPO} and the encoder/decoder code is the "
    f"official DPLM package at {DPLM_GIT_REPO}."
)


def _to_numpy(array) -> np.ndarray:
    """
    Return a numpy view of a numpy array or torch tensor without copying
    dtype.
    """
    if isinstance(array, np.ndarray):
        return array
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _looks_like_hf_repo(model_path: str | Path) -> bool:
    """
    Return True when the path is an unresolved Hugging Face repo id, not
    a local file.
    """
    text = str(model_path)
    if Path(text).exists():
        return False
    return "/" in text and not text.startswith(("/", "./", "../", "~"))


def _checkpoint_sha256(model_path: str | Path) -> dict:
    """
    Best-effort content hash of the local tokenizer checkpoint for
    provenance.
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
    """Thin runtime wrapper around a loaded DPLM-2 LFQ structure tokenizer."""

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
        """
        Encode backbone coordinates [L, 4, 3] to uint16 structure ids
        [L].
        """
        coords_np = _to_numpy(coords).astype(np.float32)
        if coords_np.ndim != 3 or coords_np.shape[1:] != (
            NUM_BACKBONE_ATOMS,
            3,
        ):
            raise ValueError(
                f"coords must have shape [L, 4, 3] with backbone atoms N, "
                f"CA, C, O; got shape {tuple(coords_np.shape)}"
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
                f"DPLM encoder returned an unexpected latent shape "
                f"{tuple(latent.shape)}, expected {(length, NUM_LFQ_DIMS)}"
            )

        # LFQ sign quantization: each latent dimension collapses to its
        # sign, and the codec maps the per-dimension bits to the pinned
        # little-endian id.
        signs = np.where(latent > 0.0, 1.0, -1.0).astype(np.float32)
        bits = self.codec.signs_to_bits_np(signs)
        index = self.codec.bits_to_index_np(bits)
        return index.astype(np.uint16)

    def decode(self, index) -> np.ndarray:
        """
        Decode uint16 structure ids [L] to backbone coordinates [L, 4,
        3].
        """
        index_np = _to_numpy(index).astype(np.int64).reshape(-1)
        if index_np.size and (
            index_np.min() < 0 or index_np.max() >= STRUCT_CODEBOOK_SIZE
        ):
            raise ValueError(
                f"structure token id out of range [0, "
                f"{STRUCT_CODEBOOK_SIZE - 1}]"
            )
        length = index_np.shape[0]

        # index -> codec bits -> per-dimension signs, matching the LFQ
        # codebook entries the DPLM decoder expects (bit 1 is sign +1,
        # bit 0 is sign -1).
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
                f"DPLM decoder returned an unexpected atom37 shape "
                f"{tuple(atom37.shape)}, expected "
                f"{(length, NUM_ATOM37, 3)}"
            )
        coords = atom37[:, list(ATOM37_BACKBONE_SLOTS), :]
        return coords.astype(np.float32)


#: Where to find an OpenFold source tree. It is NOT pip-installable here: the
#: PyPI project called "openfold" is a different package, and the real one
#: (github.com/aqlaboratory/openfold) builds CUDA extensions that need nvcc,
#: which the EOS login node does not have. Only the pure-Python modules are
#: needed, so a plain source checkout on sys.path is enough.
_OPENFOLD_CANDIDATES = ("external/openfold",)

#: In-repo pure-PyTorch stand-ins for compiled deps that the tokenizer's import
#: graph touches but its inference path never calls (torch_scatter,
#: attn_core_inplace_cuda).
_SHIM_DIR = "evaluation/evaluation_drivers/proteins/compat"

_byprot_ready = False


def _install_byprot_compat() -> None:
    """
    Make the frozen DPLM-2 tokenizer importable without a full DPLM
    install.
    """
    global _byprot_ready
    if _byprot_ready:
        return

    import dataclasses
    import sys
    import types

    repo = Path(__file__).resolve().parents[3]

    for rel in _OPENFOLD_CANDIDATES:
        cand = Path(rel) if Path(rel).is_absolute() else repo / rel
        if (cand / "openfold").is_dir():
            sys.path.insert(0, str(cand))
            break
    shim = repo / _SHIM_DIR
    if shim.is_dir():
        sys.path.insert(0, str(shim))

    root = repo / "external" / "dplm" / "src" / "byprot"
    if not root.is_dir():
        raise RuntimeError(_INSTALL_HINT)

    def _pkg(name: str, path: Path):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod
        return mod

    byprot = _pkg("byprot", root)
    models = _pkg("byprot.models", root / "models")
    datamodules = _pkg("byprot.datamodules", root / "datamodules")
    models.MODEL_REGISTRY = {}
    datamodules.DATAMODULE_REGISTRY = {}

    def _registrar(registry):
        def register(name):
            def decorate(cls):
                registry[name] = cls
                return cls

            return decorate

        return register

    models.register_model = _registrar(models.MODEL_REGISTRY)
    datamodules.register_datamodule = _registrar(
        datamodules.DATAMODULE_REGISTRY
    )
    byprot.models = models
    byprot.datamodules = datamodules

    sys.modules.setdefault(
        "byprot.utils.strategies", types.ModuleType("byprot.utils.strategies")
    )

    try:
        import biotite.structure as _bs

        if not hasattr(_bs, "filter_backbone") and hasattr(
            _bs, "filter_peptide_backbone"
        ):
            _bs.filter_backbone = _bs.filter_peptide_backbone
    # biotite is optional for callers that never fold
    except Exception:  # noqa: BLE001
        pass

    _orig_dataclass = dataclasses.dataclass

    def _compat_dataclass(cls=None, **kwargs):
        def wrap(candidate):
            try:
                return _orig_dataclass(candidate, **kwargs)
            except ValueError as exc:
                if "mutable default" not in str(exc):
                    raise
                for name in getattr(candidate, "__annotations__", {}):
                    value = getattr(candidate, name, dataclasses.MISSING)
                    kind = getattr(value, "__class__", None)
                    if (
                        value is not dataclasses.MISSING
                        and kind is not None
                        and getattr(kind, "__hash__", object.__hash__) is None
                    ):
                        kind.__hash__ = object.__hash__
                return _orig_dataclass(candidate, **kwargs)

        return wrap if cls is None else wrap(cls)

    _compat_dataclass.__kwdefaults__ = _orig_dataclass.__kwdefaults__
    dataclasses.dataclass = _compat_dataclass
    try:
        import byprot.models.structok.structok_lfq  # noqa: F401
    finally:
        dataclasses.dataclass = _orig_dataclass

    _byprot_ready = True


def _load_dplm_tokenizer(model_path: str | Path, device: str):
    """
    Import DPLM leanly and load the frozen tokenizer, raising on any
    failure.
    """
    try:
        _install_byprot_compat()
        from byprot.models.structok.structok_lfq import VQModel
    # surface every DPLM import failure uniformly
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(_INSTALL_HINT) from exc

    try:
        from omegaconf import OmegaConf

        # Upstream get_struct_tokenizer expects
        # <path>/.hydra/config.yaml; download_dplm_tokenizer.py writes
        # config.yaml at the top level. Accept both. OmegaConf, not
        # yaml.safe_load: VQModel does attribute access on its
        # sub-configs and a plain dict fails on .embed_dim.
        base = Path(model_path)
        cfg_path = base / ".hydra" / "config.yaml"
        if not cfg_path.exists():
            cfg_path = base / "config.yaml"
        ckpt_path = base / "dplm2_struct_tokenizer.ckpt"
        if not ckpt_path.exists():
            ckpt_path = base / ".hydra" / "dplm2_struct_tokenizer.ckpt"

        cfg = OmegaConf.load(str(cfg_path))
        model = VQModel(**cfg)
        state = torch.load(str(ckpt_path), map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[struct-tokenizer] restored with {len(missing)} missing "
                f"and {len(unexpected)} unexpected keys"
            )
        model = model.requires_grad_(False).to(device).eval()
    # checkpoint or config load failure
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(_INSTALL_HINT) from exc
    return model


def load_struct_tokenizer(
    model_path: str | Path,
    device: str = "cpu",
) -> _DPLMStructTokenizerRunner:
    """Load the frozen DPLM-2 LFQ structure tokenizer from model_path."""
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
