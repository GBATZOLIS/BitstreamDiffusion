"""Inverse-folding evaluation for the multimodal bitstream diffusion model.

This module implements the inverse-folding protocol of plan section 10.7, the
conditional distribution p(s | z) that recovers an amino-acid sequence from a
fixed backbone. One 18-bit-per-residue model produces the whole family of
protein tasks by clamping the observed modality and diffusing the rest, so
inverse folding is obtained by clamping the thirteen structure bits per residue
(state OBSERVED) and sampling the five sequence bits with
evaluation.proteins.generate_multimodal.generate at task 'inverse_folding'.

The protocol runs on a held-out backbone corpus, intended to be CATH 4.3 test
domains together with a held-out PDB set, and reports for every backbone:

  - amino-acid recovery, the fraction of positions whose sampled residue matches
    the native residue;
  - sequence diversity per backbone, the spread across the designs sampled for a
    single fixed backbone;
  - self-consistency TM-score and RMSD, obtained by folding each generated
    sequence with an independent folding model (ESMFold via
    evaluation.proteins.self_consistency, distinct from both the bitstream
    generator and the ProteinMPNN baseline decoder) and comparing the predicted
    backbone against the native backbone with evaluation.proteins.structure_metrics;
  - the mean pLDDT of that independent fold;
  - sequence identity and novelty of the designs relative to the training set.

Two controls accompany the model. ProteinMPNN is the standard inverse-folding
baseline and is scored through the same measurements. The shuffled-structure
control conditions the model on a residue-permuted backbone whose structure no
longer matches the native fold; recovery there should collapse toward
background, which confirms that the model's recovery reflects genuine use of the
structure signal rather than the amino-acid prior.

The module imports cleanly with only numpy and torch present. Every heavy or
external dependency (the trained checkpoint and diffusion sampler, the DPLM-2
structure tokenizer, ESMFold, ProteinMPNN, and the sibling evaluation modules
that pull them in) is imported lazily inside the function that needs it, guarded
so a missing tool raises a clear, actionable RuntimeError. See
the protein-eval extra (uv sync --extra protein-eval), the dplm-inference extra (uv sync --extra dplm-inference), and
scripts/proteins/setup/setup_evaluation.sh for installation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

__all__ = [
    "BackboneRecord",
    "amino_acid_recovery",
    "sequence_identity",
    "sequence_diversity",
    "nearest_training_identity",
    "novelty_report",
    "shuffle_structure_index",
    "fold_and_score",
    "load_backbone_records",
    "run_inverse_folding",
    "main",
]

# Designability thresholds shared with evaluation.proteins.self_consistency
# (plan section 10.2): a fold counts as designable when its self-consistency
# RMSD is below two Angstroms or its self-consistency TM-score is above 0.5.
DESIGNABLE_SC_RMSD = 2.0
DESIGNABLE_SC_TM = 0.5

# Backbone atom order in the [L, 4, 3] layout used throughout the protein stack.
_ATOM_CA = 1


def _repo_root_on_path() -> None:
    """Ensure the repository root is importable when run as a bare script.

    The dotted-path import check and module use place the repository root on
    sys.path already. Running the file directly does not, so this inserts the
    root (three parents up from this file) before any sibling repo import.
    """
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)


# -----------------------------------------------------------------------------
# Sequence metrics. These are pure string operations and need only the standard
# library, so they stay importable and testable with no heavy dependency.
# -----------------------------------------------------------------------------


def amino_acid_recovery(prediction: str, native: str) -> float:
    """Fraction of positions whose predicted residue matches the native residue.

    Inverse folding never changes the residue count, so the two strings must
    have equal length. Returns a value in the range zero to one; an empty native
    sequence returns zero.
    """
    if len(prediction) != len(native):
        raise ValueError(
            "amino-acid recovery requires equal-length sequences, received "
            f"{len(prediction)} and {len(native)}."
        )
    if not native:
        return 0.0
    matches = sum(
        1
        for predicted, target in zip(prediction, native)
        if predicted == target
    )
    return matches / len(native)


def sequence_identity(a: str, b: str) -> float:
    """Positional identity over the aligned prefix of two sequences.

    Compares residues position by position over the shorter of the two lengths
    and returns the matching fraction. Designs for one backbone share the native
    length, so this is exact for the diversity computation; for training
    comparison of differing lengths use nearest_training_identity, which slides
    the shorter sequence to find the best gapless offset.
    """
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / n


def sequence_diversity(sequences: Sequence[str]) -> Dict[str, float]:
    """Diversity of a set of designs sampled for one fixed backbone.

    Returns the mean pairwise positional identity across the designs, the
    diversity as one minus that mean, the number of distinct sequences, and the
    count of designs considered. Fewer than two designs yield NaN identity and
    diversity because a spread is undefined for a single design.
    """
    seqs = [s for s in sequences if s]
    count = len(seqs)
    unique = float(len(set(seqs)))
    if count < 2:
        return {
            "mean_pairwise_identity": float("nan"),
            "diversity": float("nan"),
            "unique": unique,
            "count": float(count),
        }
    total = 0.0
    pairs = 0
    for i in range(count):
        for j in range(i + 1, count):
            total += sequence_identity(seqs[i], seqs[j])
            pairs += 1
    mean_identity = total / pairs
    return {
        "mean_pairwise_identity": mean_identity,
        "diversity": 1.0 - mean_identity,
        "unique": unique,
        "count": float(count),
    }


def _gapless_identity(query: str, reference: str) -> float:
    """Best gapless positional identity of query against reference.

    Slides the shorter sequence across the longer one and returns the maximum
    matching fraction, normalized by the shorter length. This is an
    alignment-free, dependency-free identity proxy; a true novelty screen should
    use mmseqs2 clustering from evaluation.proteins.structure_metrics, but this
    keeps the module usable with numpy only.
    """
    if not query or not reference:
        return 0.0
    short, long = (
        (query, reference)
        if len(query) <= len(reference)
        else (reference, query)
    )
    short_len = len(short)
    long_len = len(long)
    best = 0
    for offset in range(long_len - short_len + 1):
        matches = sum(
            1 for i in range(short_len) if short[i] == long[offset + i]
        )
        if matches > best:
            best = matches
        if best == short_len:
            break
    return best / short_len


def nearest_training_identity(
    sequence: str, training_sequences: Sequence[str]
) -> float:
    """Maximum gapless identity of a design to any training sequence.

    Returns NaN when the training set is empty. A high value means the design
    reproduces a sequence already seen in training and is therefore not novel.
    """
    if not training_sequences:
        return float("nan")
    return max(
        _gapless_identity(sequence, reference)
        for reference in training_sequences
    )


def novelty_report(
    sequence: str,
    training_sequences: Sequence[str],
    novelty_threshold: float = 0.5,
) -> Dict[str, float]:
    """Identity and novelty of one design relative to the training set.

    Reports the nearest training identity, the novelty as one minus that
    identity, and a boolean flag that is true when the nearest identity stays
    below novelty_threshold. With an empty training set the identity and novelty
    are NaN and the flag is false.
    """
    nearest = nearest_training_identity(sequence, training_sequences)
    if np.isnan(nearest):
        return {
            "nearest_training_identity": float("nan"),
            "novelty": float("nan"),
            "novel": 0.0,
        }
    return {
        "nearest_training_identity": float(nearest),
        "novelty": float(1.0 - nearest),
        "novel": 1.0 if nearest < float(novelty_threshold) else 0.0,
    }


def shuffle_structure_index(
    struct_index: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return a residue-permuted copy of the observed LFQ structure token ids.

    The shuffled-structure control breaks the residue-to-fold correspondence by
    permuting the observed structure tokens across residue positions while
    preserving their multiset. Conditioning on this scrambled backbone should
    push amino-acid recovery toward background, isolating how much of the
    model's recovery comes from the structure signal rather than the sequence
    prior.
    """
    index = np.asarray(struct_index)
    if index.ndim != 1:
        raise ValueError(
            "struct_index must be a one-dimensional array of LFQ token ids."
        )
    permutation = rng.permutation(index.shape[0])
    return index[permutation]


# -----------------------------------------------------------------------------
# Coordinate helpers and the independent-fold self-consistency score.
# -----------------------------------------------------------------------------


def _as_backbone(array) -> np.ndarray:
    """Coerce coordinates to [L, 4, 3] float32 in backbone atom order N, CA, C, O."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] >= 4 and arr.shape[2] == 3:
        return arr[:, :4, :]
    if arr.ndim == 2 and arr.shape[1] == 12:
        return arr.reshape(-1, 4, 3)
    raise ValueError(
        "coordinates must have shape [L, 4, 3] (backbone atoms N, CA, C, O) or "
        f"[L, 12]; received array of shape {arr.shape}."
    )


def _ca_track(coords) -> np.ndarray:
    """Return the [L, 3] alpha-carbon track from backbone or CA-only coordinates."""
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1] >= 2 and arr.shape[2] == 3:
        return arr[:, _ATOM_CA, :]
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    raise ValueError(
        "coordinates must have shape [L, 4, 3] or [L, 3]; received array of "
        f"shape {arr.shape}."
    )


def fold_and_score(sequence: str, native_backbone) -> Dict[str, float]:
    """Fold a design with the independent model and score it against the native.

    Folds the sequence with the independent ESMFold judge from
    evaluation.proteins.self_consistency, which is deliberately distinct from
    both the bitstream generator that produced the sequence and the ProteinMPNN
    baseline decoder. The predicted backbone is compared against the native
    backbone with the TM-score and superposed RMSD of
    evaluation.proteins.structure_metrics, and the mean pLDDT of the fold is
    reported. Both heavy tools are imported lazily; a missing tool raises a
    clear RuntimeError from the sibling module. Returns sc_tm, sc_rmsd, plddt,
    and the folded length.
    """
    _repo_root_on_path()
    try:
        from evaluation.proteins import structure_metrics
        from evaluation.proteins.self_consistency import esmfold_plddt
    except Exception as exc:  # noqa: BLE001 - re-raised with an actionable hint
        raise RuntimeError(
            "Self-consistency scoring needs evaluation.proteins.self_consistency "
            "and evaluation.proteins.structure_metrics. Ensure the protein "
            "evaluation package is importable; see scripts/proteins/setup/setup_evaluation.sh."
        ) from exc

    coords, plddt = esmfold_plddt(sequence)
    predicted_ca = _ca_track(coords)
    native_ca = _ca_track(native_backbone)
    usable = min(len(predicted_ca), len(native_ca))
    if usable == 0:
        raise RuntimeError(
            "The independent fold produced no residues to score."
        )
    predicted = predicted_ca[:usable]
    native = native_ca[:usable]
    return {
        "sc_tm": float(structure_metrics.tm_score(predicted, native)),
        "sc_rmsd": float(
            structure_metrics.rmsd(predicted, native, superpose=True)
        ),
        "plddt": float(np.mean(plddt)),
        "folded_length": int(len(predicted_ca)),
    }


# -----------------------------------------------------------------------------
# Backbone corpus loading.
# -----------------------------------------------------------------------------


@dataclass
class BackboneRecord:
    """One held-out backbone with its native sequence and optional structure ids.

    coords is the backbone of shape [L, 4, 3] in atom order N, CA, C, O.
    native_seq is the reference amino-acid string of length L. struct_index, when
    present, holds the precomputed uint16 LFQ structure token ids [L]; otherwise
    it is derived on demand from coords with the DPLM-2 structure tokenizer.
    """

    stable_id: str
    coords: np.ndarray
    native_seq: str
    split: str = "test"
    source: str = ""
    struct_index: Optional[np.ndarray] = None
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Number of residues in the backbone."""
        return int(self.coords.shape[0])


def _canonical_alphabet() -> str:
    """Return the twenty-letter canonical amino-acid alphabet from the data package."""
    _repo_root_on_path()
    from data.protein_multimodal import CANONICAL_AA

    return CANONICAL_AA


def _sequence_from_npz(npz: Dict[str, object], length: int) -> str:
    """Resolve a native amino-acid string of the given length from an npz payload.

    Prefers an explicit sequence string, then integer amino-acid ids mapped
    through the canonical alphabet. Raises when neither is present, since inverse
    folding recovery is undefined without a native sequence to compare against.
    """
    for key in ("sequence", "seq", "native_seq"):
        if key in npz:
            value = npz[key]
            text = (
                value.item()
                if isinstance(value, np.ndarray) and value.ndim == 0
                else value
            )
            text = str(text)
            if len(text) != length:
                raise ValueError(
                    f"{key} length {len(text)} does not match backbone length {length}."
                )
            return text
    for key in ("seq_ids", "aatype"):
        if key in npz:
            ids = np.asarray(npz[key]).reshape(-1).astype(np.int64)
            if ids.shape[0] != length:
                raise ValueError(
                    f"{key} length {ids.shape[0]} does not match backbone length {length}."
                )
            alphabet = _canonical_alphabet()
            if ids.size and (ids.min() < 0 or ids.max() >= len(alphabet)):
                raise ValueError(
                    f"{key} contains ids outside the canonical range [0, {len(alphabet) - 1}]."
                )
            return "".join(alphabet[i] for i in ids)
    raise ValueError(
        "backbone record has no native sequence; provide 'sequence' or 'seq_ids' "
        "so amino-acid recovery can be measured."
    )


def _struct_index_from_npz(
    npz: Dict[str, object], length: int
) -> Optional[np.ndarray]:
    """Return precomputed LFQ structure token ids [L] uint16 if the npz has them."""
    for key in ("struct_index", "struct_tokens", "tokens"):
        if key in npz:
            index = np.asarray(npz[key]).reshape(-1)
            if index.shape[0] != length:
                raise ValueError(
                    f"{key} length {index.shape[0]} does not match backbone length {length}."
                )
            return index.astype(np.uint16)
    return None


def _scalar_str(value: object, default: str) -> str:
    """Read a possibly zero-dimensional numpy scalar as a plain string."""
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def _records_from_npz(path: Path) -> List[BackboneRecord]:
    """Read every backbone chain stored in one npz file.

    Supports a single-chain file (a 'coords' array of shape [L, 4, 3] with a
    native sequence) and a ragged multi-chain file that adds an 'offsets' array
    delimiting consecutive chains inside a flat coordinate stack.
    """
    with np.load(path, allow_pickle=True) as loaded:
        npz = {key: loaded[key] for key in loaded.files}
    if "coords" not in npz:
        raise ValueError(f"{path}: missing required 'coords' array.")

    records: List[BackboneRecord] = []
    if "offsets" in npz:
        coords_flat = _as_backbone(npz["coords"])
        offsets = np.asarray(npz["offsets"]).astype(np.int64).reshape(-1)
        for row in range(int(offsets.shape[0] - 1)):
            start, end = int(offsets[row]), int(offsets[row + 1])
            coords = coords_flat[start:end]
            length = int(coords.shape[0])
            sub = {
                k: (
                    np.asarray(v)[start:end]
                    if k
                    in (
                        "seq_ids",
                        "aatype",
                        "struct_index",
                        "struct_tokens",
                        "tokens",
                    )
                    else v
                )
                for k, v in npz.items()
                if k not in ("coords", "offsets")
            }
            records.append(
                BackboneRecord(
                    stable_id=f"{path.stem}:{row}",
                    coords=np.ascontiguousarray(coords),
                    native_seq=_sequence_from_npz(sub, length),
                    split=_scalar_str(npz.get("split"), "test"),
                    source=_scalar_str(npz.get("source"), path.stem),
                    struct_index=_struct_index_from_npz(sub, length),
                )
            )
        return records

    coords = _as_backbone(npz["coords"])
    length = int(coords.shape[0])
    records.append(
        BackboneRecord(
            stable_id=_scalar_str(
                npz.get("stable_id") or npz.get("id"), path.stem
            ),
            coords=np.ascontiguousarray(coords),
            native_seq=_sequence_from_npz(npz, length),
            split=_scalar_str(npz.get("split"), "test"),
            source=_scalar_str(npz.get("source"), path.stem),
            struct_index=_struct_index_from_npz(npz, length),
        )
    )
    return records


def load_backbone_records(
    source,
    *,
    splits: Optional[Sequence[str]] = None,
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
    max_examples: Optional[int] = None,
) -> List[BackboneRecord]:
    """Load held-out backbones from an npz file or a directory of npz files.

    source is a single .npz file or a directory searched recursively for .npz
    files in sorted path order. Records are filtered by split label (for the CATH
    4.3 test and held-out PDB partitions), by residue-length window, and by a
    maximum count. Records whose backbone contains a non-finite coordinate are
    skipped. Returns the surviving BackboneRecord list.
    """
    path = Path(source)
    if path.is_dir():
        files = sorted(path.rglob("*.npz"))
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(
            f"backbone source {path} is not a file or directory."
        )
    if not files:
        raise FileNotFoundError(
            f"no .npz backbone files found under {path}. Each file must contain a "
            "'coords' array of shape [L, 4, 3] (backbone N, CA, C, O) and a native "
            "sequence via 'sequence' or 'seq_ids'."
        )

    wanted = set(splits) if splits else None
    records: List[BackboneRecord] = []
    for file_path in files:
        for record in _records_from_npz(file_path):
            if wanted is not None and record.split not in wanted:
                continue
            if min_len is not None and record.length < int(min_len):
                continue
            if max_len is not None and record.length > int(max_len):
                continue
            if not np.isfinite(record.coords).all():
                continue
            records.append(record)
            if max_examples is not None and len(records) >= int(max_examples):
                return records
    return records


# -----------------------------------------------------------------------------
# Model loading and structure-token resolution.
# -----------------------------------------------------------------------------


def _load_model(
    config_path: str, checkpoint: Path, device, num_steps: Optional[int]
):
    """Build the multimodal model from a config and checkpoint, EMA weights applied.

    Thin wrapper over ``evaluation.proteins.io.load_binary_protein_model``, shared
    with the other protein generation drivers. Returns the evaluated (model, cfg)
    pair.
    """
    _repo_root_on_path()
    from evaluation.proteins.io import load_binary_protein_model

    return load_binary_protein_model(
        config_path,
        checkpoint,
        device,
        num_steps=num_steps,
        context="inverse folding",
    )


class _StructureTokenResolver:
    """Resolve per-backbone LFQ structure token ids, caching the heavy tokenizer.

    Precomputed struct_index arrays on the records are used as is. When a record
    lacks them the backbone is encoded on demand with the DPLM-2 structure
    tokenizer, which is loaded once and reused. The tokenizer is imported lazily;
    if it or its checkpoint is missing, the sibling module raises a RuntimeError
    that points at scripts/proteins/setup/download_dplm_tokenizer.py.
    """

    def __init__(self, tokenizer_path: Optional[str], device: str = "cpu"):
        self._tokenizer_path = tokenizer_path
        self._device = device
        self._tokenizer = None

    def _tokenizer_instance(self):
        if self._tokenizer is None:
            if not self._tokenizer_path:
                raise RuntimeError(
                    "A backbone record has no precomputed structure tokens and no "
                    "structure tokenizer was provided. Pass --struct-tokenizer with "
                    "the DPLM-2 tokenizer checkpoint (see "
                    "scripts/proteins/setup/download_dplm_tokenizer.py) or supply records "
                    "that already carry a 'struct_index' array."
                )
            _repo_root_on_path()
            from evaluation.proteins.dplm_struct_tokenizer import (
                load_struct_tokenizer,
            )

            self._tokenizer = load_struct_tokenizer(
                self._tokenizer_path, device=self._device
            )
        return self._tokenizer

    def resolve(self, record: BackboneRecord) -> np.ndarray:
        """Return the uint16 LFQ token ids [L] for one backbone record."""
        if record.struct_index is not None:
            return np.asarray(record.struct_index, dtype=np.uint16)
        tokenizer = self._tokenizer_instance()
        index = tokenizer.encode(np.asarray(record.coords, dtype=np.float32))
        return np.asarray(index).reshape(-1).astype(np.uint16)


# -----------------------------------------------------------------------------
# Design generators for the model, the shuffled control, and the ProteinMPNN
# baseline.
# -----------------------------------------------------------------------------


def _design_with_model(
    model,
    cfg,
    device,
    record: BackboneRecord,
    struct_index: np.ndarray,
    num_samples: int,
) -> List[str]:
    """Sample sequences for a backbone by clamped inverse-folding diffusion."""
    _repo_root_on_path()
    from evaluation.proteins.generate_multimodal import generate

    result = generate(
        model,
        cfg,
        task="inverse_folding",
        length=record.length,
        num_samples=int(num_samples),
        device=device,
        observed={"struct_index": np.asarray(struct_index).reshape(-1)},
    )
    return list(result["seq_strings"])


def _design_with_proteinmpnn(
    record: BackboneRecord, num_samples: int, temperature: float
) -> List[str]:
    """Design sequences for a backbone with the ProteinMPNN baseline decoder."""
    _repo_root_on_path()
    from evaluation.proteins.self_consistency import proteinmpnn_sequences

    return list(
        proteinmpnn_sequences(
            np.asarray(record.coords, dtype=np.float32),
            n_seqs=int(num_samples),
            temperature=float(temperature),
        )
    )


# -----------------------------------------------------------------------------
# Per-method evaluation and aggregation.
# -----------------------------------------------------------------------------


def _summ(values: Sequence[float]) -> Dict[str, float]:
    """Mean, median, minimum, and maximum of a list of finite floats, or NaNs."""
    finite = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not finite:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "count": 0,
        }
    array = np.asarray(finite, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "count": int(array.size),
    }


def _evaluate_method(
    name: str,
    records: Sequence[BackboneRecord],
    struct_indices: Sequence[np.ndarray],
    designer: Callable[[BackboneRecord, np.ndarray], List[str]],
    *,
    training_sequences: Sequence[str],
    sc_designs: int,
    novelty_threshold: float,
) -> Dict[str, object]:
    """Design, measure, and aggregate one method over every backbone.

    For each backbone the designer returns a list of sequences. Amino-acid
    recovery is measured per design against the native sequence, sequence
    diversity is measured across the designs, and the highest-recovery designs
    (up to sc_designs of them) are folded by the independent model for
    self-consistency and pLDDT. Novelty is measured for the highest-recovery
    design. The per-backbone summaries and the pooled aggregates are returned.
    """
    per_backbone: List[Dict[str, object]] = []
    best_aar: List[float] = []
    mean_aar: List[float] = []
    diversity: List[float] = []
    sc_tm: List[float] = []
    sc_rmsd: List[float] = []
    plddt: List[float] = []
    designable: List[float] = []
    novelty: List[float] = []
    novel_flags: List[float] = []

    for record, struct_index in zip(records, struct_indices):
        designs = [str(seq) for seq in designer(record, struct_index)]
        recoveries = [
            amino_acid_recovery(seq, record.native_seq) for seq in designs
        ]
        order = sorted(
            range(len(designs)), key=lambda i: recoveries[i], reverse=True
        )

        entry: Dict[str, object] = {
            "stable_id": record.stable_id,
            "split": record.split,
            "length": record.length,
            "num_designs": len(designs),
            "aar_mean": float(np.mean(recoveries))
            if recoveries
            else float("nan"),
            "aar_best": float(np.max(recoveries))
            if recoveries
            else float("nan"),
            "diversity": sequence_diversity(designs),
        }
        if recoveries:
            best_aar.append(entry["aar_best"])
            mean_aar.append(entry["aar_mean"])
        if not np.isnan(entry["diversity"]["diversity"]):
            diversity.append(entry["diversity"]["diversity"])

        top_index = order[0] if order else None
        if top_index is not None and training_sequences:
            report = novelty_report(
                designs[top_index], training_sequences, novelty_threshold
            )
            entry["novelty"] = report
            if not np.isnan(report["novelty"]):
                novelty.append(report["novelty"])
                novel_flags.append(report["novel"])

        if sc_designs > 0 and designs:
            folded = []
            for index in order[: int(sc_designs)]:
                scored = fold_and_score(designs[index], record.coords)
                scored["design_index"] = int(index)
                scored["aar"] = float(recoveries[index])
                folded.append(scored)
            entry["self_consistency"] = folded
            record_sc_tm = max(item["sc_tm"] for item in folded)
            record_sc_rmsd = min(item["sc_rmsd"] for item in folded)
            record_plddt = max(item["plddt"] for item in folded)
            entry["sc_tm_best"] = record_sc_tm
            entry["sc_rmsd_best"] = record_sc_rmsd
            entry["plddt_best"] = record_plddt
            is_designable = bool(
                record_sc_rmsd < DESIGNABLE_SC_RMSD
                or record_sc_tm > DESIGNABLE_SC_TM
            )
            entry["designable"] = is_designable
            sc_tm.append(record_sc_tm)
            sc_rmsd.append(record_sc_rmsd)
            plddt.append(record_plddt)
            designable.append(1.0 if is_designable else 0.0)

        per_backbone.append(entry)

    aggregate: Dict[str, object] = {
        "num_backbones": len(per_backbone),
        "aar_best": _summ(best_aar),
        "aar_mean": _summ(mean_aar),
        "diversity": _summ(diversity),
    }
    if novelty:
        aggregate["novelty"] = _summ(novelty)
        aggregate["fraction_novel"] = float(np.mean(novel_flags))
    if sc_tm:
        aggregate["sc_tm"] = _summ(sc_tm)
        aggregate["sc_rmsd"] = _summ(sc_rmsd)
        aggregate["plddt"] = _summ(plddt)
        aggregate["designable_fraction"] = float(np.mean(designable))

    return {
        "method": name,
        "aggregate": aggregate,
        "per_backbone": per_backbone,
    }


def _read_training_sequences(
    path: Optional[str], limit: Optional[int]
) -> List[str]:
    """Read training sequences for novelty from a FASTA file, optionally subsampled."""
    if not path:
        return []
    _repo_root_on_path()
    from evaluation.proteins.io import read_fasta

    sequences = read_fasta(Path(path))
    if limit is not None and len(sequences) > int(limit):
        sequences = sequences[: int(limit)]
    return sequences


# -----------------------------------------------------------------------------
# Top-level protocol driver.
# -----------------------------------------------------------------------------


def run_inverse_folding(
    *,
    records: Sequence[BackboneRecord],
    methods: Sequence[str],
    training_sequences: Sequence[str],
    model=None,
    cfg=None,
    device=None,
    struct_resolver: Optional[_StructureTokenResolver] = None,
    num_samples: int = 8,
    sc_designs: int = 1,
    novelty_threshold: float = 0.5,
    proteinmpnn_temperature: float = 0.1,
    seed: int = 0,
) -> Dict[str, object]:
    """Run the inverse-folding protocol over a set of backbones and methods.

    methods may include 'model' (clamped bitstream inverse folding), 'shuffled'
    (the same model conditioned on a residue-permuted backbone, a control), and
    'proteinmpnn' (the baseline decoder). The model-driven methods require model,
    cfg, device, and a structure-token resolver. Returns a dict keyed by method
    name plus a shared summary; it is JSON-serializable with native scalars.
    """
    if not records:
        raise ValueError("no backbone records to evaluate.")
    rng = np.random.default_rng(int(seed))
    needs_model = any(method in ("model", "shuffled") for method in methods)
    if needs_model and (
        model is None or cfg is None or struct_resolver is None
    ):
        raise ValueError(
            "methods 'model' and 'shuffled' need a loaded model, cfg, and structure "
            "resolver; pass them or drop those methods."
        )

    struct_indices: List[Optional[np.ndarray]] = [None] * len(records)
    if needs_model:
        struct_indices = [
            struct_resolver.resolve(record) for record in records
        ]

    results: Dict[str, object] = {}
    for method in methods:
        if method == "model":

            def designer(record, struct_index):
                return _design_with_model(
                    model, cfg, device, record, struct_index, num_samples
                )
        elif method == "shuffled":

            def designer(record, struct_index):
                shuffled = shuffle_structure_index(struct_index, rng)
                return _design_with_model(
                    model, cfg, device, record, shuffled, num_samples
                )
        elif method == "proteinmpnn":

            def designer(record, struct_index):
                return _design_with_proteinmpnn(
                    record, num_samples, proteinmpnn_temperature
                )
        else:
            raise ValueError(
                f"unknown method {method!r}; choose from 'model', 'shuffled', 'proteinmpnn'."
            )
        results[method] = _evaluate_method(
            method,
            records,
            struct_indices,
            designer,
            training_sequences=training_sequences,
            sc_designs=int(sc_designs),
            novelty_threshold=float(novelty_threshold),
        )

    results["summary"] = {
        "num_backbones": len(records),
        "num_samples_per_backbone": int(num_samples),
        "sc_designs_folded": int(sc_designs),
        "methods": list(methods),
        "splits": sorted({record.split for record in records}),
        "mean_length": float(np.mean([record.length for record in records])),
    }
    return results


def _git(*args: str) -> Optional[str]:
    """Return the trimmed output of a git command, or None when it fails."""
    import subprocess

    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def _sha256(path: Optional[str]) -> Optional[str]:
    """Return the SHA-256 of a file, or None when the path is missing."""
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    _repo_root_on_path()
    from evaluation.proteins.io import sha256

    return sha256(candidate)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the inverse-folding protocol."""
    parser = argparse.ArgumentParser(
        description="Inverse-folding evaluation p(s | z) for the multimodal "
        "bitstream diffusion model on CATH 4.3 and held-out PDB backbones "
        "(plan section 10.7)."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="npz file or directory of backbones with coords and native sequences.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Path to write the JSON report.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["model", "proteinmpnn", "shuffled"],
        choices=["model", "proteinmpnn", "shuffled"],
        help="Which design methods to evaluate.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Restrict to these split labels (for example cath43_test pdb_heldout).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Model config for the model-driven methods.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="Model checkpoint (.pt)."
    )
    parser.add_argument(
        "--struct-tokenizer",
        default=None,
        help="DPLM-2 structure tokenizer checkpoint, used when records lack struct_index.",
    )
    parser.add_argument(
        "--training-fasta",
        default=None,
        help="FASTA of training sequences for the novelty and identity measurement.",
    )
    parser.add_argument(
        "--training-limit",
        type=int,
        default=20000,
        help="Cap on training sequences read for novelty.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Designs sampled per backbone.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override the number of sampling steps.",
    )
    parser.add_argument(
        "--sc-designs",
        type=int,
        default=1,
        help="Highest-recovery designs per backbone folded for self-consistency (0 disables).",
    )
    parser.add_argument(
        "--proteinmpnn-temperature",
        type=float,
        default=0.1,
        help="ProteinMPNN sampling temperature.",
    )
    parser.add_argument(
        "--novelty-threshold",
        type=float,
        default=0.5,
        help="Nearest-identity below this counts as novel.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=None,
        help="Skip backbones shorter than this.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Skip backbones longer than this.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Cap on the number of backbones.",
    )
    parser.add_argument(
        "--device", default="auto", help="Torch device: auto, cpu, or cuda."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling and sampling.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the inverse-folding protocol from the command line and write a JSON report.

    Loads the held-out backbones, loads the model when a model-driven method is
    requested, evaluates every requested method, and writes a JSON report with
    the per-method aggregates, the per-backbone measurements, and a provenance
    block recording the config, checkpoint, data source, and environment.
    """
    _repo_root_on_path()
    args = build_parser().parse_args(argv)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    records = load_backbone_records(
        args.data,
        splits=args.splits,
        min_len=args.min_len,
        max_len=args.max_len,
        max_examples=args.max_examples,
    )
    if not records:
        raise SystemExit("No backbone records matched the requested filters.")

    training_sequences = _read_training_sequences(
        args.training_fasta, args.training_limit
    )

    model = None
    cfg = None
    struct_resolver = None
    needs_model = any(
        method in ("model", "shuffled") for method in args.methods
    )
    if needs_model:
        if args.config is None or args.checkpoint is None:
            raise SystemExit(
                "methods 'model' and 'shuffled' require --config and --checkpoint."
            )
        model, cfg = _load_model(
            args.config, args.checkpoint, device, args.num_steps
        )
        struct_resolver = _StructureTokenResolver(
            args.struct_tokenizer,
            device="cuda" if device.type == "cuda" else "cpu",
        )

    results = run_inverse_folding(
        records=records,
        methods=args.methods,
        training_sequences=training_sequences,
        model=model,
        cfg=cfg,
        device=device,
        struct_resolver=struct_resolver,
        num_samples=args.num_samples,
        sc_designs=args.sc_designs,
        novelty_threshold=args.novelty_threshold,
        proteinmpnn_temperature=args.proteinmpnn_temperature,
        seed=args.seed,
    )

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "Inverse folding p(s | z) on CATH 4.3 test and held-out PDB backbones (plan 10.7).",
        "config": {"path": args.config, "sha256": _sha256(args.config)},
        "checkpoint": {
            "path": str(args.checkpoint) if args.checkpoint else None,
            "sha256": _sha256(
                str(args.checkpoint) if args.checkpoint else None
            ),
        },
        "data": {"path": str(args.data)},
        "arguments": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
        },
        "results": results,
    }

    from evaluation.proteins.io import atomic_json

    atomic_json(args.out, payload)
    for method in args.methods:
        aggregate = results[method]["aggregate"]
        print(
            f"{method}: {json.dumps(aggregate.get('aar_best', {}), sort_keys=True)}"
        )
    print(f"Saved inverse-folding report to {args.out}")
    return 0


if __name__ == "__main__":
    _repo_root_on_path()
    raise SystemExit(main())
