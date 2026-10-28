"""Load paired protein shards and build sequence and structure batches."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    _HAS_TORCH = False

from . import protein_tasks as T
from utils.regimes import plan_from_bit_states
from .protein_structure_codec import (
    DEFAULT_STRUCT_CODEC,
    PATCH_BITS_PER_RESIDUE,
    SEQ_BITS_PER_RESIDUE,
    STRUCT_CODEBOOK_SIZE,
    LFQBitCodec,
)
from .proteins import build_token_to_bits_table

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
SEQ_VOCAB_SIZE = len(CANONICAL_AA)  # 20
CANONICAL_AA_TO_ID = {aa: i for i, aa in enumerate(CANONICAL_AA)}


# -----------------------------------------------------------------------------
# Sequence bit codecs (raw binary default; permutation hook for
# ablations)
# -----------------------------------------------------------------------------


@dataclass
class SequenceBitCodec:
    """Amino-acid id -> 5 sequence bits."""

    name: str = "raw_binary"
    permutation: np.ndarray | None = None  # length-20 id permutation, or None

    def __post_init__(self):
        self._table = build_token_to_bits_table(
            SEQ_VOCAB_SIZE, SEQ_BITS_PER_RESIDUE
        )
        if self.permutation is not None:
            perm = np.asarray(self.permutation, dtype=np.int64)
            if perm.shape != (SEQ_VOCAB_SIZE,) or sorted(
                perm.tolist()
            ) != list(range(SEQ_VOCAB_SIZE)):
                raise ValueError(
                    "permutation must be a permutation of range(20)"
                )

    def bits(self, seq_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(seq_ids, dtype=np.int64)
        if self.permutation is not None:
            ids = self.permutation[ids]
        return self._table[torch.from_numpy(ids)].numpy().astype(np.uint8)

    def hash(self) -> str:
        payload = json.dumps(
            {
                "name": self.name,
                "perm": None
                if self.permutation is None
                else np.asarray(self.permutation).tolist(),
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class StructurePermutation:
    """
    Fixed random permutation of the 8192 LFQ indices for the
    native-vs-permuted ablation.
    """

    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.perm = rng.permutation(STRUCT_CODEBOOK_SIZE).astype(np.int64)

    def apply(self, index: np.ndarray) -> np.ndarray:
        return self.perm[np.asarray(index, dtype=np.int64)]


# -----------------------------------------------------------------------------
# Shard IO
# -----------------------------------------------------------------------------


@dataclass
class RowRecord:
    """One paired chain to be written into a shard."""

    stable_id: str
    seq_ids: np.ndarray  # [L] uint8, canonical-20 ids
    struct_index: np.ndarray  # [L] uint16, LFQ ids
    seq_mask: np.ndarray  # [L] bool
    struct_mask: np.ndarray  # [L] bool
    source: str = "unknown"
    cluster_id: str = ""
    split: str = "train"
    accession: str = ""
    chain: str = ""
    release: str = ""
    plddt: float = float("nan")
    resolution: float = float("nan")
    coverage: float = float("nan")
    deposited_length: int = 0
    chain_break: bool = False
    coverage_verified: bool = False
    chain_break_verified: bool = False
    pair_completeness_class: str = "unknown"
    deposition_date: str = ""

    def length(self) -> int:
        return int(self.seq_ids.shape[0])


def write_shard(
    shard_dir: Path,
    shard_name: str,
    rows: Sequence[RowRecord],
    *,
    tokenizer_revision: str = "unpinned",
) -> dict[str, object]:
    """Write one immutable ragged shard (.npz) plus a JSON metadata sidecar."""
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    seq_parts, struct_parts, seqm_parts, structm_parts = [], [], [], []
    offsets = [0]
    meta_rows = []
    for r in rows:
        L = r.length()
        if not (
            r.struct_index.shape[0] == L
            and r.seq_mask.shape[0] == L
            and r.struct_mask.shape[0] == L
        ):
            raise ValueError(f"row {r.stable_id}: modality lengths disagree")
        if (
            r.struct_index.size
            and int(np.asarray(r.struct_index).max()) >= STRUCT_CODEBOOK_SIZE
        ):
            raise ValueError(f"row {r.stable_id}: structure id out of range")
        seq_parts.append(np.asarray(r.seq_ids, dtype=np.uint8))
        struct_parts.append(np.asarray(r.struct_index, dtype=np.uint16))
        seqm_parts.append(np.asarray(r.seq_mask, dtype=bool))
        structm_parts.append(np.asarray(r.struct_mask, dtype=bool))
        offsets.append(offsets[-1] + L)
        seq_hash = hashlib.sha256(
            np.asarray(r.seq_ids, dtype=np.uint8).tobytes()
        ).hexdigest()[:16]
        coord_hash = hashlib.sha256(
            np.asarray(r.struct_index, dtype=np.uint16).tobytes()
        ).hexdigest()[:16]
        meta_rows.append(
            {
                "stable_id": r.stable_id,
                "source": r.source,
                "cluster_id": r.cluster_id,
                "split": r.split,
                "accession": r.accession,
                "chain": r.chain,
                "release": r.release,
                "length": L,
                "plddt": r.plddt,
                "resolution": r.resolution,
                "coverage": r.coverage,
                "deposited_length": r.deposited_length or L,
                "chain_break": r.chain_break,
                "deposition_date": r.deposition_date,
                "sequence_hash": seq_hash,
                "coordinate_hash": coord_hash,
                "tokenizer_revision": tokenizer_revision,
            }
        )
    npz_path = shard_dir / f"{shard_name}.npz"
    np.savez(
        npz_path,
        seq_ids=np.concatenate(seq_parts)
        if seq_parts
        else np.zeros(0, np.uint8),
        struct_index=np.concatenate(struct_parts)
        if struct_parts
        else np.zeros(0, np.uint16),
        seq_mask=np.concatenate(seqm_parts)
        if seqm_parts
        else np.zeros(0, bool),
        struct_mask=np.concatenate(structm_parts)
        if structm_parts
        else np.zeros(0, bool),
        offsets=np.asarray(offsets, dtype=np.int64),
    )
    sidecar = {
        "shard": shard_name,
        "n_rows": len(rows),
        "rows": meta_rows,
        "tokenizer_revision": tokenizer_revision,
    }
    with (shard_dir / f"{shard_name}.meta.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(sidecar, f)
    return sidecar


def write_manifest(
    shard_dir: Path,
    shard_names: Sequence[str],
    *,
    codec: LFQBitCodec = DEFAULT_STRUCT_CODEC,
    seq_codec: SequenceBitCodec | None = None,
    extra: dict[str, object] | None = None,
) -> Path:
    """
    Write the dataset manifest that pins codec conventions and shard
    list.
    """
    shard_dir = Path(shard_dir)
    artifact_sha256 = {}
    for shard_name in shard_names:
        for suffix in (".npz", ".meta.json"):
            artifact = shard_dir / f"{shard_name}{suffix}"
            digest = hashlib.sha256()
            with artifact.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            artifact_sha256[artifact.name] = digest.hexdigest()
    manifest = {
        "shards": list(shard_names),
        "struct_codec": codec.spec(),
        "struct_convention_hash": codec.convention_hash(),
        "seq_codec": (seq_codec or SequenceBitCodec()).name,
        "seq_codec_hash": (seq_codec or SequenceBitCodec()).hash(),
        "patch_bits": PATCH_BITS_PER_RESIDUE,
        "artifact_sha256": artifact_sha256,
        "release_content_sha256": hashlib.sha256(
            json.dumps(artifact_sha256, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    if extra:
        manifest.update(extra)
    path = shard_dir / "manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class ProteinMultimodalDataset(Dataset):
    """
    Reads paired shards for one split and returns per-row modality
    arrays.
    """

    is_text_dataset = False

    def __init__(
        self,
        shard_dir,
        *,
        split: str,
        struct_codec: LFQBitCodec = DEFAULT_STRUCT_CODEC,
        struct_permutation: StructurePermutation | None = None,
        min_len: int = 1,
        max_len: int = 512,
        long_chain_policy: str = "drop",
        crop_seed: int = 0,
    ):
        self.shard_dir = Path(shard_dir)
        self.split = split
        self.codec = struct_codec
        self.struct_permutation = struct_permutation
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        # drop is the ProteBit-1B primary policy.  A random fragment of
        # a folded chain can omit stabilising domains and long-range
        # contacts, so it must not be relabelled as a complete paired
        # protein.  The legacy random_crop option remains for
        # reproducibility of earlier 400M experiments, but the 1B
        # readiness gate rejects it.
        self.long_chain_policy = str(long_chain_policy)
        if self.long_chain_policy not in {"drop", "random_crop"}:
            raise ValueError(
                f"long_chain_policy must be 'drop' or 'random_crop', got "
                f"{long_chain_policy!r}"
            )
        self.crop_seed = int(crop_seed)
        self.crop_epoch = 0

        manifest_path = self.shard_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Build paired shards with "
                f"scripts/proteins/setup/prepare_dplm_paired.py"
            )
        with manifest_path.open() as f:
            self.manifest = json.load(f)
        # Fail loud if the stored structure convention differs from
        # ours.
        stored = self.manifest.get("struct_convention_hash")
        if stored is not None and stored != self.codec.convention_hash():
            raise ValueError(
                "structure codec convention hash mismatch between "
                "manifest and codec; the LFQ bit order changed and "
                "cached bits would be wrong"
            )

        self._shards = []
        self._index: list[tuple[int, int]] = []  # (shard_idx, row_idx)
        self.lengths: list[int] = []
        self.sources: list[str] = []
        self.cluster_ids: list[str] = []
        for shard_name in self.manifest["shards"]:
            npz = np.load(self.shard_dir / f"{shard_name}.npz")
            with (self.shard_dir / f"{shard_name}.meta.json").open() as f:
                sidecar = json.load(f)
            offsets = npz["offsets"]
            arrays = {
                "seq_ids": npz["seq_ids"],
                "struct_index": npz["struct_index"],
                "seq_mask": npz["seq_mask"],
                "struct_mask": npz["struct_mask"],
                "offsets": offsets,
            }
            shard_idx = len(self._shards)
            self._shards.append(arrays)
            for row_idx, meta in enumerate(sidecar["rows"]):
                if meta["split"] != split:
                    continue
                L = int(meta["length"])
                if L < self.min_len:
                    continue
                if L > self.max_len:
                    if self.long_chain_policy != "random_crop":
                        continue
                    # A cropped row joins the max_len length bucket,
                    # which is what keeps SourceLengthBatchSampler's
                    # same-length batching valid without any special
                    # case.
                    L = self.max_len
                self._index.append((shard_idx, row_idx))
                self.lengths.append(L)
                self.sources.append(meta["source"])
                self.cluster_ids.append(meta.get("cluster_id", ""))
        self.lengths = np.asarray(self.lengths, dtype=np.int64)

    def set_epoch(self, epoch: int) -> None:
        """
        Re-roll crop offsets each epoch, so a long chain is seen in
        windows.
        """
        self.crop_epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_weight = 1.0
        if isinstance(idx, (tuple, list)):
            idx, sample_weight = int(idx[0]), float(idx[1])
        shard_idx, row_idx = self._index[idx]
        arrays = self._shards[shard_idx]
        offsets = arrays["offsets"]
        start, end = int(offsets[row_idx]), int(offsets[row_idx + 1])

        if (end - start) > self.max_len:
            span = (end - start) - self.max_len
            if self.split == "train":
                # Keyed on (seed, epoch, row) rather than on a global
                # RNG, so the crop is reproducible on resume, identical
                # across ranks for the same row, and independent of
                # DataLoader worker count.
                offset = int(
                    np.random.default_rng(
                        (self.crop_seed, self.crop_epoch, idx)
                    ).integers(0, span + 1)
                )
            else:
                offset = span // 2  # deterministic centred crop for validation
            start += offset
            end = start + self.max_len

        seq_ids = np.asarray(arrays["seq_ids"][start:end], dtype=np.int64)
        struct_index = np.asarray(
            arrays["struct_index"][start:end], dtype=np.int64
        )
        if self.struct_permutation is not None:
            struct_index = self.struct_permutation.apply(struct_index)
        return {
            "seq_ids": seq_ids,
            "struct_index": struct_index,
            "seq_mask": np.asarray(arrays["seq_mask"][start:end], dtype=bool),
            "struct_mask": np.asarray(
                arrays["struct_mask"][start:end], dtype=bool
            ),
            "length": end - start,
            "source": self.sources[idx],
            "cluster_id": self.cluster_ids[idx],
            "row_index": idx,
            "sample_weight": sample_weight,
        }


class LazyEvoDiffUniRef50Dataset(Dataset):
    """Sequence-only view over frozen UniRef50 FASTA bytes and mmap indices."""

    is_text_dataset = False

    def __init__(self, index_dir, *, split: str, crop_seed: int = 0):
        self.shard_dir = Path(index_dir)
        self.split = str(split)
        self.crop_seed = int(crop_seed)
        self.crop_epoch = 0
        manifest_path = self.shard_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("storage_format") != "evodiff_fasta_offsets_v1":
            raise ValueError(
                f"unsupported lazy storage format in {manifest_path}"
            )
        if self.split not in self.manifest.get("splits", {}):
            raise ValueError(
                f"split {self.split!r} is not indexed in {manifest_path}"
            )

        source_root = Path(self.manifest["source_root"])
        if not source_root.exists():
            # Node-local staging places all requested dataset
            # directories next to one another and intentionally changes
            # their absolute root.
            staged = self.shard_dir.parent / source_root.name
            if staged.exists():
                source_root = staged
        data_dir = source_root / self.manifest.get("data_dir", "uniref50")
        self.fasta_path = data_dir / self.manifest.get(
            "fasta", "consensus.fasta"
        )
        self.source_offsets = np.load(
            self.shard_dir / self.manifest["source_offsets"], mmap_mode="r"
        )
        self.source_lengths = np.load(
            self.shard_dir / self.manifest["source_lengths"], mmap_mode="r"
        )

        files = self.manifest["splits"][self.split]
        self.source_indices = np.load(
            self.shard_dir / files["source_indices"], mmap_mode="r"
        )
        self.lengths = np.load(
            self.shard_dir / files["lengths"], mmap_mode="r"
        )
        self.length_order = np.load(
            self.shard_dir / files["length_order"], mmap_mode="r"
        )
        self.length_values = np.load(
            self.shard_dir / files["length_values"], mmap_mode="r"
        )
        self.length_boundaries = np.load(
            self.shard_dir / files["length_boundaries"], mmap_mode="r"
        )
        self.max_len = int(self.manifest["max_len"])
        # Training retains every UniRef row and crops only rows above
        # the model context. Validation is a complete-sequence
        # measurement: unsupported rows are excluded instead of
        # centre-cropped into a different target.
        self._row_positions = None
        if self.split != "train":
            full_lengths = self.source_lengths[self.source_indices]
            eligible = np.flatnonzero(full_lengths <= self.max_len).astype(
                np.int64
            )
            self._row_positions = eligible
            self.lengths = np.asarray(self.lengths[eligible], dtype=np.int64)
        self._fasta = None
        self._fasta_pid = None

    def __len__(self) -> int:
        if self._row_positions is not None:
            return int(self._row_positions.shape[0])
        return int(self.source_indices.shape[0])

    def set_epoch(self, epoch: int) -> None:
        self.crop_epoch = int(epoch)

    def _handle(self):
        pid = os.getpid()
        if self._fasta is None or self._fasta_pid != pid:
            if self._fasta is not None:
                self._fasta.close()
            self._fasta = self.fasta_path.open("rb")
            self._fasta_pid = pid
        return self._fasta

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_weight = 1.0
        if isinstance(idx, (tuple, list)):
            idx, sample_weight = int(idx[0]), float(idx[1])
        row_position = (
            int(self._row_positions[idx])
            if self._row_positions is not None
            else int(idx)
        )
        source_idx = int(self.source_indices[row_position])
        full_len = int(self.source_lengths[source_idx])
        handle = self._handle()
        handle.seek(int(self.source_offsets[source_idx]))
        raw = handle.readline().rstrip(b"\r\n")
        if len(raw) != full_len:
            raise RuntimeError(
                f"UniRef50 offset mismatch for source row {source_idx}: "
                f"expected {full_len}, read {len(raw)}"
            )
        if full_len > self.max_len:
            span = full_len - self.max_len
            if self.split == "train":
                start = int(
                    np.random.default_rng(
                        (self.crop_seed, self.crop_epoch, int(idx))
                    ).integers(0, span + 1)
                )
            else:
                raise RuntimeError(
                    "unsupported overlength UniRef validation row escaped the "
                    "complete-sequence filter"
                )
            raw = raw[start : start + self.max_len]

        seq = raw.decode("ascii", errors="replace")
        seq_ids = np.zeros(len(seq), dtype=np.int64)
        seq_mask = np.zeros(len(seq), dtype=bool)
        for pos, residue in enumerate(seq):
            token = CANONICAL_AA_TO_ID.get(residue)
            if token is not None:
                seq_ids[pos] = token
                seq_mask[pos] = True
        return {
            "seq_ids": seq_ids,
            "struct_index": np.zeros(len(seq), dtype=np.int64),
            "seq_mask": seq_mask,
            "struct_mask": np.zeros(len(seq), dtype=bool),
            "length": len(seq),
            "source": "uniref50",
            "cluster_id": f"uniref50_{source_idx}",
            "row_index": int(idx),
            "sample_weight": sample_weight,
        }


# -----------------------------------------------------------------------------
# Task-aware collation
# -----------------------------------------------------------------------------


class MultimodalTaskCollator:
    """
    Turns a list of same-length rows into an 18-bit batch under a drawn
    task.
    """

    def __init__(
        self,
        *,
        task_weights: dict[str, float] | None = None,
        seq_codec: SequenceBitCodec | None = None,
        struct_codec: LFQBitCodec = DEFAULT_STRUCT_CODEC,
        seed: int = 0,
        absent_fill: float = 0.5,
        span_coverage_range: tuple[float, float] = (0.10, 0.90),
    ):
        self.task_weights = dict(task_weights or T.DEFAULT_TASK_WEIGHTS)
        self.seq_codec = seq_codec or SequenceBitCodec()
        self.struct_codec = struct_codec
        self.seed = int(seed)
        # Fraction of residues given as context for the span tasks. The
        # rest are generation targets.
        self.span_coverage_range = (
            float(span_coverage_range[0]),
            float(span_coverage_range[1]),
        )
        # Motifs are a smaller fraction of the chain than a general
        # inpainting context: across the 24 benchmark problems the motif
        # is roughly 10-40% of the total design length.
        self.motif_coverage_range = (0.10, 0.40)
        # Value written into the bits of an ABSENT modality (P0-1). Must
        # match cfg.diffusion.continuous.data_center and the value the
        # sampler puts in unobserved slots, or training and generation
        # disagree.
        self.absent_fill = float(absent_fill)
        self.epoch = 0
        self._counter = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._counter = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "counter": self._counter,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.seed = int(state["seed"])
        self.epoch = int(state["epoch"])
        self._counter = int(state["counter"])

    def _sample_motif_spans(
        self, rng: np.random.Generator, length: int
    ) -> list[tuple[int, int]]:
        """Draw motif spans with the benchmark's own geometry."""
        mask = T.sample_span_mask(
            rng, length, coverage_range=self.motif_coverage_range
        )
        spans: list[tuple[int, int]] = []
        start = None
        for i, on in enumerate(mask):
            if on and start is None:
                start = i
            elif not on and start is not None:
                spans.append((start, i))
                start = None
        if start is not None:
            spans.append((start, length))
        return spans or [(0, max(1, length // 3))]

    def _batch_rng(
        self, batch: list[dict[str, object]]
    ) -> np.random.Generator:
        """Deterministic per-batch RNG keyed by the batch's row identities."""
        idxs = np.array(
            sorted(int(r.get("row_index", i)) for i, r in enumerate(batch)),
            dtype=np.int64,
        )
        digest = hashlib.blake2b(idxs.tobytes(), digest_size=8).digest()
        batch_key = int.from_bytes(digest, "little")
        self._counter += 1
        return np.random.default_rng([int(self.seed), batch_key])

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, object]:
        if not _HAS_TORCH:
            raise RuntimeError("torch is required for collation")
        # Training batches are exact-length. Validation may mix
        # neighbouring lengths to cover every row under DDP; pad only to
        # this batch's maximum and mask both padding and replicated
        # filler rows.
        L = max(int(r["length"]) for r in batch)
        B = len(batch)
        rng = self._batch_rng(batch)

        patch = np.zeros((B, L, PATCH_BITS_PER_RESIDUE), dtype=np.uint8)
        states = np.zeros((B, L, T.NUM_MODALITIES), dtype=np.int8)
        residue_mask = np.zeros((B, L), dtype=bool)
        seq_avail = np.zeros((B, L), dtype=bool)
        struct_avail = np.zeros((B, L), dtype=bool)
        task_names: list[str] = []
        sources: list[str] = []

        for b, r in enumerate(batch):
            row_len = int(r["length"])
            row_weight = float(r.get("sample_weight", 1.0))
            seq_bits = self.seq_codec.bits(r["seq_ids"])  # [Li, 5]
            struct_bits = self.struct_codec.index_to_bits_np(
                r["struct_index"]
            )  # [Li, 13]
            patch[b, :row_len, :SEQ_BITS_PER_RESIDUE] = seq_bits
            patch[b, :row_len, SEQ_BITS_PER_RESIDUE:] = struct_bits
            residue_mask[b, :row_len] = row_weight > 0.0
            seq_avail[b, :row_len] = np.asarray(r["seq_mask"], dtype=bool)
            struct_avail[b, :row_len] = np.asarray(
                r["struct_mask"], dtype=bool
            )
            task = T.sample_task(rng, self.task_weights)
            task_names.append(task)
            sources.append(str(r.get("source", "unknown")))
            # Span tasks (sequence_inpaint / structure_inpaint and the
            # two redesign variants) draw a fresh per-residue observed
            # mask for every example, so seq-to-seq and 3D-to-3D see
            # varied geometry rather than one fixed span. Motif spans
            # are drawn the same way instead of the old fixed centred
            # third, which meant every motif example in every run had
            # identical geometry.
            st = T.build_example_states(
                task,
                row_len,
                rng=rng,
                coverage_range=self.span_coverage_range,
                motif_spans=(
                    self._sample_motif_spans(rng, row_len)
                    if task == "motif"
                    else None
                ),
            )  # [L, 2]
            # Downgrade to ABSENT where a residue lacks that modality.
            st[~seq_avail[b, :row_len], T.SEQ] = T.ABSENT
            st[~struct_avail[b, :row_len], T.STRUCT] = T.ABSENT
            states[b, :row_len] = st

        seq_target, struct_target = T.target_loss_masks_np(
            states, residue_mask
        )
        seq_full, struct_full = T.full_modality_target_masks_np(
            states, residue_mask
        )

        state_grid = torch.from_numpy(states.astype(np.int64))
        bit_states = torch.from_numpy(T.per_bit_state_ids_np(states))
        conditioning = plan_from_bit_states(bit_states)

        # P0-1. Blank the content of ABSENT bits.
        #
        # ABSENT and OBSERVED both receive sigma 0, so without this the
        # noised input x_t is exactly x0 at ABSENT positions and the
        # true bits of a "missing" modality are visible in the content
        # channel. Overwriting them with a constant makes them carry no
        # information about x0, while the state embedding still tells
        # the model the modality is absent.
        #
        # The constant is the diffusion data centre (0.5): bits live in
        # [0, 1] and are centred at 0.5, so this is the neutral value
        # the sampler also uses for an unobserved slot. patch becomes
        # float32 because 0.5 is not representable in uint8.
        patch_f = patch.astype(np.float32)
        absent_bits = conditioning.null_mask.numpy().reshape(
            B, L, PATCH_BITS_PER_RESIDUE
        )
        patch_f[absent_bits] = self.absent_fill

        out = {
            "x0": torch.from_numpy(
                patch_f.reshape(B, L * PATCH_BITS_PER_RESIDUE)
            ).float(),
            "residue_mask": torch.from_numpy(residue_mask),
            "states": state_grid,
            "seq_target_mask": torch.from_numpy(seq_target),
            "struct_target_mask": torch.from_numpy(struct_target),
            "seq_target_full": torch.from_numpy(seq_full),
            "struct_target_full": torch.from_numpy(struct_full),
            "clamp_mask": conditioning.clamp_mask,
            "loss_mask": conditioning.loss_mask,
            "null_mask": conditioning.null_mask,
            "protect_mask": conditioning.protect_mask,
            "task_names": task_names,
            "sources": sources,
            "length": L,
            "row_lengths": torch.as_tensor(
                [int(r["length"]) for r in batch], dtype=torch.int64
            ),
            "sample_weight": torch.as_tensor(
                [float(r.get("sample_weight", 1.0)) for r in batch],
                dtype=torch.float32,
            ),
        }
        return out


# -----------------------------------------------------------------------------
# Loader factory (mirrors data/uniref50.py get_evodiff_uniref50_loader)
# -----------------------------------------------------------------------------


def _cfg_task_weights(config) -> dict[str, float] | None:
    tw = getattr(config.data, "task_weights", None)
    if tw is None:
        return None
    return {str(k): float(v) for k, v in dict(tw).items()}


def _cfg_source_weights(config) -> dict[str, float] | None:
    sw = getattr(config.data, "source_weights", None)
    if sw is None:
        return None
    return {str(k): float(v) for k, v in dict(sw).items()}


def get_multimodal_loader(
    config,
    *,
    split: str,
    batch_size: int | None = None,
    shuffle: bool | None = None,
    seed: int = 42,
):
    """Build a DataLoader for the paired multimodal dataset for one split."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is required")
    from torch.utils.data import DataLoader

    from .protein_multimodal_sampler import (
        PrecomputedLengthBatchSampler,
        SourceLengthBatchSampler,
        TokenBudgetValidationBatchSampler,
    )
    from .proteins import DistributedLengthBucketBatchSampler, _ddp_rank_world

    shard_dir = getattr(config.data, "shard_dir", "datasets/dplm_paired_m0")
    min_len = int(getattr(config.data, "min_len", 1))
    max_len = int(getattr(config.data, "max_len", 512))
    manifest_path = Path(shard_dir) / "manifest.json"
    expected_manifest_sha256 = str(
        getattr(config.data, "expected_manifest_sha256", "") or ""
    )
    require_manifest_binding = bool(
        getattr(config.data, "require_manifest_binding", False)
    )
    if require_manifest_binding and not expected_manifest_sha256:
        raise RuntimeError(
            "production data manifest is not bound; the launcher must set the "
            "expected SHA-256 before constructing the DataLoader"
        )
    if expected_manifest_sha256:
        actual_manifest_sha256 = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise RuntimeError(
                f"data manifest hash mismatch for {manifest_path}: "
                f"expected {expected_manifest_sha256}, got "
                f"{actual_manifest_sha256}"
            )
    manifest_probe = json.loads(manifest_path.read_text())
    is_lazy_uniref = (
        manifest_probe.get("storage_format") == "evodiff_fasta_offsets_v1"
    )
    if is_lazy_uniref:
        dataset = LazyEvoDiffUniRef50Dataset(
            shard_dir,
            split=split,
            crop_seed=int(getattr(config.train, "seed", 42)),
        )
    else:
        dataset = ProteinMultimodalDataset(
            shard_dir,
            split=split,
            min_len=min_len,
            max_len=max_len,
            long_chain_policy=str(
                getattr(config.data, "long_chain_policy", "drop")
            ),
            crop_seed=int(getattr(config.train, "seed", 42)),
        )
    if shuffle is None:
        shuffle = split == "train"
    rank, world_size = _ddp_rank_world()
    local_bs = int(batch_size or config.train.batch_size)
    steps_per_epoch = int(getattr(config.train, "steps_per_epoch", 0))
    token_budgeted = bool(
        getattr(config.data, "token_budgeted_batching", False)
    )
    residue_budget_per_rank = (
        int(getattr(config.data, "per_gpu_residue_budget", 0))
        if token_budgeted
        else None
    )
    if token_budgeted and residue_budget_per_rank <= 0:
        raise ValueError(
            "token_budgeted_batching requires data.per_gpu_residue_budget > 0"
        )
    force_train_length = getattr(config.data, "force_train_length", None)
    force_train_length = (
        None
        if force_train_length in (None, 0, "")
        else int(force_train_length)
    )
    min_budget_utilization = float(
        getattr(config.data, "minimum_residue_budget_utilization", 0.0) or 0.0
    )

    # The ABSENT fill value must equal the diffusion data centre, so
    # that a blanked modality sits at the same neutral value the sampler
    # uses for an unobserved slot (P0-1).
    absent_fill = 0.5
    diff_cfg = getattr(getattr(config, "diffusion", None), "continuous", None)
    if diff_cfg is not None:
        absent_fill = float(getattr(diff_cfg, "data_center", 0.5))

    collator = MultimodalTaskCollator(
        task_weights=_cfg_task_weights(config),
        seed=int(getattr(config.train, "seed", 42)),
        absent_fill=absent_fill,
    )

    if split == "train" and steps_per_epoch > 0:
        # steps_per_epoch counts OPTIMIZER steps (the P0-7 fix), but
        # this sampler yields MICRO-batches. Handing it steps_per_epoch
        # directly re-introduces the same defect P0-7 removed, one layer
        # down: with grad_accum_steps=2 the loader runs dry after 2,500
        # optimizer steps of a 5,000-step epoch, so the run reaches half
        # its configured schedule and exits normally with nothing
        # flagged.
        #
        # m1_small has grad_accum=1 so it is unaffected; the FLAGSHIP
        # has grad_accum=2 and would have silently trained half its
        # budget.
        accum = max(1, int(getattr(config.train, "grad_accum_steps", 1) or 1))
        micro_batches_per_epoch = steps_per_epoch * accum
        if is_lazy_uniref:
            sampler = PrecomputedLengthBatchSampler(
                length_order=dataset.length_order,
                length_values=dataset.length_values,
                length_boundaries=dataset.length_boundaries,
                batch_size=local_bs,
                num_batches=micro_batches_per_epoch,
                seed=int(seed),
                rank=rank,
                world_size=world_size,
                residue_budget_per_rank=residue_budget_per_rank,
                max_batch_size=local_bs,
                force_length=force_train_length,
                microbatches_per_update=accum,
                minimum_residue_budget_utilization=min_budget_utilization,
            )
        else:
            sampler = SourceLengthBatchSampler(
                dataset.lengths,
                dataset.sources,
                batch_size=local_bs,
                num_batches=micro_batches_per_epoch,
                seed=int(seed),
                rank=rank,
                world_size=world_size,
                source_weights=_cfg_source_weights(config),
                cluster_ids=dataset.cluster_ids,
                residue_budget_per_rank=residue_budget_per_rank,
                max_batch_size=local_bs,
                length_support_fraction=float(
                    getattr(config.data, "length_support_fraction", 0.0)
                ),
                length_support_boundaries=list(
                    getattr(
                        config.data,
                        "length_support_boundaries",
                        [128, 256, 512, 768, 1024],
                    )
                ),
                force_length=force_train_length,
                microbatches_per_update=accum,
                minimum_residue_budget_utilization=min_budget_utilization,
            )
    elif token_budgeted:
        sampler = TokenBudgetValidationBatchSampler(
            dataset.lengths,
            batch_size=local_bs,
            residue_budget_per_rank=int(residue_budget_per_rank),
            shuffle=bool(shuffle),
            seed=int(seed),
            rank=rank,
            world_size=world_size,
        )
    else:
        sampler = DistributedLengthBucketBatchSampler(
            dataset.lengths,
            batch_size=local_bs,
            shuffle=bool(shuffle),
            seed=int(seed),
            rank=rank,
            world_size=world_size,
        )
    num_workers = int(getattr(config.data, "num_workers", 4))

    # Persistent workers are incompatible with anything the trainer
    # updates on the collator or the dataset between epochs.
    #
    # The collate_fn and the dataset are pickled into each worker when
    # that worker starts. With persistent_workers=True that happens ONCE
    # for the whole run, so the scheduled (corpus, task) mixture and the
    # per-epoch long-chain crop re-roll -- both of which the trainer
    # applies by writing to the parent's objects at the epoch boundary
    # -- would never reach the process that actually collates. The run
    # would sit in mixture phase A for 300,000 steps while every log
    # reported the scheduled weights, and every long chain would be seen
    # through the same window forever.
    #
    # Without it, workers are re-forked at each iter(dataloader), i.e.
    # once per epoch, which is exactly where the trainer applies those
    # updates. The respawn cost amortises over 5,000 optimizer steps.
    #
    # This must be decided HERE rather than in the trainer's
    # _dataloader_kwargs: the multimodal path builds its loaders through this
    # factory and never passes through that helper, and the replay
    # loader is built through it too, under a temporary cfg.data swap.
    uses_epoch_state = (
        is_lazy_uniref
        or bool(getattr(config.data, "use_mixture_schedule", False))
        or str(getattr(config.data, "long_chain_policy", "drop")) != "drop"
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=bool(getattr(config.data, "pin_memory", True)),
        persistent_workers=(num_workers > 0) and (not uses_epoch_state),
        prefetch_factor=int(getattr(config.data, "prefetch_factor", 2))
        if num_workers > 0
        else None,
    )
