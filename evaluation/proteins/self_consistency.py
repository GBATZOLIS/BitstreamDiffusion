"""ProteinMPNN plus ESMFold self-consistency for generated backbones.

This module implements the designability loop described in plan section 10.1.
A candidate backbone is redesigned into amino-acid sequences by ProteinMPNN
(the decoder), each sequence is folded back into a structure by ESMFold (the
folding model), and the predicted structure is compared against the original
backbone by an independent judge (TM-score and RMSD from
evaluation.proteins.structure_metrics). Plan section 10.2 requires these three
roles to stay distinct so the metric never scores a model against itself; the
SelfConsistency class enforces that the decoder and the folding model differ.

Designability follows the fixed thresholds in plan section 10.2: eight
ProteinMPNN sequences are folded by ESMFold, and a backbone counts as
designable when the self-consistency RMSD falls below 2.0 Angstrom or the
self-consistency TM-score rises above 0.5. The run method always returns the
continuous self-consistency values alongside the boolean pass flag, so callers
can rank candidates rather than only threshold them.

Only numpy and torch are imported eagerly. ProteinMPNN, ESMFold (transformers
or fair-esm) and the structure-metrics judge are imported lazily inside the
functions that use them, and a missing dependency raises a clear RuntimeError
that points at scripts/proteins/setup/setup_evaluation.sh and the requirements files.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import List, Tuple

import numpy as np
import torch

__all__ = ["proteinmpnn_sequences", "esmfold_plddt", "SelfConsistency"]


# ProteinMPNN maps sampled indices to amino acids through this fixed alphabet.
_PROTEINMPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

# Loaded models are cached per device so a self-consistency loop over many
# sequences does not reload multi-gigabyte weights on every fold or design call.
_ESMFOLD_CACHE: dict = {}
_PROTEINMPNN_CACHE: dict = {}


def _resolve_device(device=None) -> torch.device:
    """Return the requested torch device, defaulting to CUDA when available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ca_coords(coords) -> np.ndarray:
    """Return the alpha-carbon track from backbone or CA-only coordinates.

    Accepts an array of shape [L, 4, 3] in backbone atom order (N, CA, C, O),
    in which case the CA atom at index one is selected, or an array of shape
    [L, 3] that is already the CA track.
    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[1] >= 2:
        return arr[:, 1, :]
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return arr
    raise ValueError(
        "Coordinates must have shape [L, 4, 3] (backbone atoms N, CA, C, O) "
        "or [L, 3] (CA only)."
    )


def _load_structure_metrics():
    """Import the independent TM-score and RMSD judge, lazily and guarded."""
    try:
        from evaluation.proteins import structure_metrics
    except Exception as exc:  # noqa: BLE001 - re-raised with an actionable hint
        raise RuntimeError(
            "Self-consistency scoring needs evaluation.proteins.structure_metrics "
            "for tm_score and rmsd. Ensure the protein evaluation package is "
            "importable; see scripts/proteins/setup/setup_evaluation.sh."
        ) from exc
    for attribute in ("tm_score", "rmsd"):
        if not hasattr(structure_metrics, attribute):
            raise RuntimeError(
                f"evaluation.proteins.structure_metrics is missing '{attribute}', "
                "which self-consistency scoring requires."
            )
    return structure_metrics


def _extract_backbone_and_plddt(
    positions, plddt, length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce an ESMFold output to backbone coordinates and per-residue pLDDT.

    ESMFold returns atom14 positions whose first four atoms are N, CA, C and O
    for every residue type; leading recycle or structure-block dimensions are
    dropped and the batch axis is indexed. pLDDT is returned per residue on the
    zero to one hundred scale, averaging any per-atom axis.
    """
    pos = positions
    while pos.dim() > 4:
        pos = pos[-1]
    coords = pos[0, :, :4, :].float().cpu().numpy()

    values = plddt
    if hasattr(values, "dim"):
        values = values[0].float().cpu().numpy()
    else:
        values = np.asarray(values)[0]
    if values.ndim == 2:
        values = values.mean(axis=-1)

    coords = np.asarray(coords, dtype=np.float64)[:length]
    values = np.asarray(values, dtype=np.float64)[:length]
    return coords, values


def _try_esmfold_transformers(device: torch.device):
    """Build an ESMFold folding closure from transformers, or None if absent."""
    try:
        from transformers import AutoTokenizer, EsmForProteinFolding
    except ImportError:
        return None

    model_id = "facebook/esmfold_v1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = (
        EsmForProteinFolding.from_pretrained(model_id, low_cpu_mem_usage=True)
        .to(device)
        .eval()
    )
    if device.type == "cuda":
        model.esm = model.esm.half()

    def fold(sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        inputs = tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        ).to(device)
        with torch.no_grad():
            output = model(**inputs)
        return _extract_backbone_and_plddt(
            output["positions"], output["plddt"], len(sequence)
        )

    return fold


def _try_esmfold_fair_esm(device: torch.device):
    """Build an ESMFold folding closure from fair-esm, or None if absent."""
    try:
        import esm
    except ImportError:
        return None

    model = esm.pretrained.esmfold_v1().eval().to(device)

    def fold(sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            output = model.infer([sequence])
        return _extract_backbone_and_plddt(
            output["positions"], output["plddt"], len(sequence)
        )

    return fold


def _load_esmfold(device: torch.device):
    """Return a cached ESMFold folding closure, preferring transformers."""
    key = str(device)
    if key in _ESMFOLD_CACHE:
        return _ESMFOLD_CACHE[key]
    fold = _try_esmfold_transformers(device)
    if fold is None:
        fold = _try_esmfold_fair_esm(device)
    if fold is None:
        raise RuntimeError(
            "ESMFold is required to fold designed sequences, but neither "
            "transformers (facebook/esmfold_v1) nor fair-esm is installed. "
            "Install the protein inference stack with "
            "scripts/proteins/setup/setup_evaluation.sh dplm (it provides fair-esm "
            ">=2.0), or add transformers from the protein-eval extra (uv sync --extra protein-eval) / "
            "the dplm-inference extra (uv sync --extra dplm-inference)."
        )
    _ESMFOLD_CACHE[key] = fold
    return fold


def _import_proteinmpnn_module():
    """Import the ProteinMPNN reference module, honouring PROTEINMPNN_DIR."""
    directory = os.environ.get("PROTEINMPNN_DIR")
    if directory and directory not in sys.path:
        sys.path.insert(0, directory)
    for name in ("protein_mpnn_utils", "ProteinMPNN.protein_mpnn_utils"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise RuntimeError(
        "ProteinMPNN could not be imported. Clone the reference implementation "
        "from https://github.com/dauparas/ProteinMPNN and point PROTEINMPNN_DIR "
        "at the checkout so protein_mpnn_utils is importable, then set "
        "PROTEINMPNN_WEIGHTS to a weights file such as "
        "vanilla_model_weights/v_48_020.pt. See scripts/proteins/setup/setup_evaluation.sh "
        "and the dplm-inference extra (uv sync --extra dplm-inference)."
    )


def _load_proteinmpnn_weights(device: torch.device):
    """Locate and load ProteinMPNN weights, with an actionable error if absent."""
    path = os.environ.get("PROTEINMPNN_WEIGHTS")
    if not path:
        directory = os.environ.get("PROTEINMPNN_DIR")
        if directory:
            candidate = os.path.join(
                directory, "vanilla_model_weights", "v_48_020.pt"
            )
            if os.path.exists(candidate):
                path = candidate
    if not path or not os.path.exists(path):
        raise RuntimeError(
            "ProteinMPNN weights were not found. Set PROTEINMPNN_WEIGHTS to a "
            "checkpoint such as vanilla_model_weights/v_48_020.pt, or set "
            "PROTEINMPNN_DIR to a ProteinMPNN checkout that contains them. See "
            "scripts/proteins/setup/setup_evaluation.sh."
        )
    return torch.load(path, map_location=device)


def _load_proteinmpnn(device: torch.device):
    """Return a cached (module, model) pair for ProteinMPNN sampling."""
    key = str(device)
    if key in _PROTEINMPNN_CACHE:
        return _PROTEINMPNN_CACHE[key]
    module = _import_proteinmpnn_module()
    checkpoint = _load_proteinmpnn_weights(device)
    hidden_dim = 128
    model = module.ProteinMPNN(
        num_letters=len(_PROTEINMPNN_ALPHABET),
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=int(checkpoint.get("num_edges", 48)),
        augment_eps=0.0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    _PROTEINMPNN_CACHE[key] = (module, model)
    return module, model


def _indices_to_sequence(indices) -> str:
    """Map a tensor of ProteinMPNN token indices to an amino-acid string."""
    return "".join(_PROTEINMPNN_ALPHABET[int(i)] for i in indices.tolist())


def proteinmpnn_sequences(
    backbone_coords, n_seqs: int = 8, temperature: float = 0.1
) -> List[str]:
    """Design amino-acid sequences for a backbone with ProteinMPNN.

    backbone_coords is an array of shape [L, 4, 3] in backbone atom order
    (N, CA, C, O). ProteinMPNN samples n_seqs sequences at the given sampling
    temperature and returns them as strings of length L. ProteinMPNN is imported
    lazily; when it or its weights are missing a RuntimeError explains how to
    install them.
    """
    device = _resolve_device()
    coords = np.asarray(backbone_coords, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[1] < 4 or coords.shape[2] != 3:
        raise ValueError(
            "backbone_coords must have shape [L, 4, 3] with backbone atoms "
            "N, CA, C, O."
        )
    coords = coords[:, :4, :]
    length = coords.shape[0]
    module, model = _load_proteinmpnn(device)
    alphabet_size = len(_PROTEINMPNN_ALPHABET)

    coordinates = torch.from_numpy(coords[None]).float().to(device)
    mask = torch.ones(1, length, device=device)
    chain_mask = torch.ones(1, length, device=device)
    chain_mask_positions = torch.ones(1, length, device=device)
    chain_encoding = torch.ones(1, length, device=device)
    residue_index = torch.arange(length, device=device, dtype=torch.long)[None]
    reference_sequence = torch.zeros(
        1, length, dtype=torch.long, device=device
    )
    omit_aas = np.array(
        [1.0 if symbol == "X" else 0.0 for symbol in _PROTEINMPNN_ALPHABET],
        dtype=np.float32,
    )
    bias_aas = np.zeros(alphabet_size, dtype=np.float32)
    omit_aa_mask = torch.zeros(1, length, alphabet_size, device=device)
    pssm_coefficient = torch.zeros(1, length, device=device)
    pssm_bias = torch.zeros(1, length, alphabet_size, device=device)
    pssm_log_odds_mask = torch.zeros(1, length, alphabet_size, device=device)
    bias_by_residue = torch.zeros(1, length, alphabet_size, device=device)

    sequences: List[str] = []
    with torch.no_grad():
        for _ in range(int(n_seqs)):
            noise = torch.randn(1, length, device=device)
            try:
                sampled = model.sample(
                    coordinates,
                    noise,
                    reference_sequence,
                    chain_mask,
                    chain_encoding,
                    residue_index,
                    mask=mask,
                    temperature=float(temperature),
                    omit_AAs_np=omit_aas,
                    bias_AAs_np=bias_aas,
                    chain_M_pos=chain_mask_positions,
                    omit_AA_mask=omit_aa_mask,
                    pssm_coef=pssm_coefficient,
                    pssm_bias=pssm_bias,
                    pssm_multi=0.0,
                    pssm_log_odds_flag=0,
                    pssm_log_odds_mask=pssm_log_odds_mask,
                    pssm_bias_flag=0,
                    bias_by_res=bias_by_residue,
                )
            except TypeError:
                sampled = model.sample(
                    coordinates,
                    noise,
                    reference_sequence,
                    chain_mask,
                    chain_encoding,
                    residue_index,
                    mask=mask,
                    temperature=float(temperature),
                )
            tokens = sampled["S"][0]
            if hasattr(module, "_S_to_seq"):
                sequence = module._S_to_seq(tokens, chain_mask[0])
            else:
                sequence = _indices_to_sequence(tokens)
            sequences.append(str(sequence))
    return sequences


def esmfold_plddt(seq: str) -> Tuple[np.ndarray, np.ndarray]:
    """Fold a sequence with ESMFold and return backbone coordinates and pLDDT.

    Returns a tuple whose first element is the predicted backbone of shape
    [L, 4, 3] in atom order (N, CA, C, O) and whose second element is the
    per-residue pLDDT of shape [L] on the zero to one hundred scale. ESMFold is
    imported lazily from transformers or, if that is unavailable, fair-esm; when
    neither is present a RuntimeError explains how to install it.
    """
    fold = _load_esmfold(_resolve_device())
    coords, plddt = fold(seq)
    return np.asarray(coords, dtype=np.float64), np.asarray(
        plddt, dtype=np.float64
    )


class SelfConsistency:
    """Run the ProteinMPNN plus ESMFold self-consistency loop for a backbone.

    The designer redesigns the backbone into sequences, the fold model predicts
    a structure for each, and the structure-metrics judge compares each
    prediction against the original backbone. Plan section 10.2 requires the
    decoder, the folding model and the judge to be distinct; construction fails
    if the designer and fold model are the same, and the judge always comes from
    evaluation.proteins.structure_metrics. Only 'proteinmpnn' and 'esmfold' are
    implemented today.
    """

    def __init__(
        self,
        fold_model: str = "esmfold",
        designer: str = "proteinmpnn",
        temperature: float = 0.1,
    ):
        self.fold_model = str(fold_model)
        self.designer = str(designer)
        self.temperature = float(temperature)
        if self.designer == self.fold_model:
            raise ValueError(
                "Circularity discipline (plan section 10.2): the sequence "
                "designer and the folding model must be distinct."
            )
        if self.designer != "proteinmpnn":
            raise ValueError(
                f"Unsupported designer '{self.designer}'; only 'proteinmpnn' is "
                "implemented."
            )
        if self.fold_model != "esmfold":
            raise ValueError(
                f"Unsupported fold_model '{self.fold_model}'; only 'esmfold' is "
                "implemented."
            )

    def run(
        self,
        backbone_coords,
        n_seqs: int = 8,
        sc_rmsd_thresh: float = 2.0,
        sc_tm_thresh: float = 0.5,
    ) -> dict:
        """Design, fold and score a backbone, returning continuous metrics.

        Designs n_seqs sequences (eight by default, per plan section 10.2),
        folds each with ESMFold, and scores every prediction against the input
        backbone with TM-score and superposed RMSD. Returns a dictionary with
        the self-consistency RMSD (minimum over designs), the self-consistency
        TM-score (maximum over designs), the best per-sequence mean pLDDT, the
        designable flag (scRMSD below sc_rmsd_thresh or scTM above sc_tm_thresh),
        and a per-sequence breakdown. The continuous values are always present,
        not only the pass flag.
        """
        metrics = _load_structure_metrics()
        reference_ca = _ca_coords(backbone_coords)
        designed = proteinmpnn_sequences(
            backbone_coords, n_seqs=int(n_seqs), temperature=self.temperature
        )

        per_sequence = []
        for index, sequence in enumerate(designed):
            coords, plddt = esmfold_plddt(sequence)
            predicted_ca = _ca_coords(coords)
            usable = min(len(predicted_ca), len(reference_ca))
            predicted = predicted_ca[:usable]
            reference = reference_ca[:usable]
            tm = float(metrics.tm_score(predicted, reference))
            rmsd = float(metrics.rmsd(predicted, reference, superpose=True))
            per_sequence.append(
                {
                    "index": index,
                    "sequence": sequence,
                    "sc_rmsd": rmsd,
                    "sc_tm": tm,
                    "plddt": float(np.mean(plddt)),
                }
            )

        if not per_sequence:
            raise RuntimeError(
                "The sequence designer returned no sequences; cannot score "
                "self-consistency."
            )

        sc_rmsd = min(entry["sc_rmsd"] for entry in per_sequence)
        sc_tm = max(entry["sc_tm"] for entry in per_sequence)
        best_plddt = max(entry["plddt"] for entry in per_sequence)
        designable = bool(
            sc_rmsd < float(sc_rmsd_thresh) or sc_tm > float(sc_tm_thresh)
        )
        return {
            "sc_rmsd": sc_rmsd,
            "sc_tm": sc_tm,
            "plddt": best_plddt,
            "designable": designable,
            "n_seqs": len(per_sequence),
            "sc_rmsd_thresh": float(sc_rmsd_thresh),
            "sc_tm_thresh": float(sc_tm_thresh),
            "fold_model": self.fold_model,
            "designer": self.designer,
            "per_sequence": per_sequence,
        }
