"""Sharded 18-bit paired sequence+structure dataset and task-aware collation.

This is the multimodal data path added alongside the sequence-only Swiss-Prot and
EvoDiff loaders (which are left untouched). It reads immutable, ragged shards of
paired rows, expands the compact ``uint16`` structure indices to 13 LFQ bits on
the collate path, assembles the 18-bit residue patch, and builds the per-task
noise/loss/clamp masks from ``data.protein_tasks``.

Row schema (plan section 7.7) stored per shard as a ragged concat plus a JSON
sidecar of per-row metadata:
  sequence_ids   uint8  concatenated, canonical-20 amino-acid ids
  structure_lfq_index uint16 concatenated, DPLM LFQ token ids
  seq_mask       bool   residue has a valid amino-acid identity
  struct_mask    bool   residue has a valid, above-threshold structure token
  offsets        int64  [N+1] row boundaries into the flat arrays
  (sidecar) stable_id, source, release, accession, chain, cluster_id, split,
            length, pLDDT/resolution/coverage, sequence_hash, coordinate_hash,
            tokenizer_revision

Structure indices, not unpacked bits, are stored; bits are expanded on collate.
A residue can have a sequence identity but no structure (a gap or unresolved
density), or the reverse, so ``seq_mask`` and ``struct_mask`` are separate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


# -----------------------------------------------------------------------------
# Sequence bit codecs (raw binary default; permutation hook for ablations)
# -----------------------------------------------------------------------------


@dataclass
class SequenceBitCodec:
    """Amino-acid id -> 5 sequence bits. Default is canonical-20 raw binary.

    ``permutation`` optionally relabels the 20 ids before encoding, which lets
    the balanced-semantic-code ablation (plan 5.1) freeze a different mapping
    without touching the storage format. The mapping and its hash are frozen.
    """

    name: str = "raw_binary"
    permutation: Optional[np.ndarray] = (
        None  # length-20 id permutation, or None
    )

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
    """Fixed random permutation of the 8192 LFQ indices for the native-vs-permuted
    ablation (plan 5.4). Applied to indices before bit encoding; identical
    vocabulary size and token frequencies, destroyed dimension-wise meaning."""

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

    def length(self) -> int:
        return int(self.seq_ids.shape[0])


def write_shard(
    shard_dir: Path,
    shard_name: str,
    rows: Sequence[RowRecord],
    *,
    tokenizer_revision: str = "unpinned",
) -> Dict[str, object]:
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
    seq_codec: Optional[SequenceBitCodec] = None,
    extra: Optional[Dict[str, object]] = None,
) -> Path:
    """Write the dataset manifest that pins codec conventions and shard list."""
    shard_dir = Path(shard_dir)
    manifest = {
        "shards": list(shard_names),
        "struct_codec": codec.spec(),
        "struct_convention_hash": codec.convention_hash(),
        "seq_codec": (seq_codec or SequenceBitCodec()).name,
        "seq_codec_hash": (seq_codec or SequenceBitCodec()).hash(),
        "patch_bits": PATCH_BITS_PER_RESIDUE,
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
    """Reads paired shards for one split and returns per-row modality arrays.

    ``__getitem__`` returns the raw per-residue ids and masks; the 18-bit patch
    and task masks are built in :class:`MultimodalTaskCollator` so the same rows
    can serve every task by redrawing the modality states.
    """

    is_text_dataset = False

    def __init__(
        self,
        shard_dir,
        *,
        split: str,
        struct_codec: LFQBitCodec = DEFAULT_STRUCT_CODEC,
        struct_permutation: Optional[StructurePermutation] = None,
        min_len: int = 1,
        max_len: int = 512,
    ):
        self.shard_dir = Path(shard_dir)
        self.split = split
        self.codec = struct_codec
        self.struct_permutation = struct_permutation
        self.min_len = int(min_len)
        self.max_len = int(max_len)

        manifest_path = self.shard_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Build paired shards with "
                "scripts/proteins/setup/prepare_dplm_paired.py"
            )
        with manifest_path.open() as f:
            self.manifest = json.load(f)
        # Fail loud if the stored structure convention differs from ours.
        stored = self.manifest.get("struct_convention_hash")
        if stored is not None and stored != self.codec.convention_hash():
            raise ValueError(
                "structure codec convention hash mismatch between manifest and codec; "
                "the LFQ bit order changed and cached bits would be wrong"
            )

        self._shards = []
        self._index: List[Tuple[int, int]] = []  # (shard_idx, row_idx)
        self.lengths: List[int] = []
        self.sources: List[str] = []
        self.cluster_ids: List[str] = []
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
                if L < self.min_len or L > self.max_len:
                    continue
                self._index.append((shard_idx, row_idx))
                self.lengths.append(L)
                self.sources.append(meta["source"])
                self.cluster_ids.append(meta.get("cluster_id", ""))
        self.lengths = np.asarray(self.lengths, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        shard_idx, row_idx = self._index[idx]
        arrays = self._shards[shard_idx]
        offsets = arrays["offsets"]
        start, end = int(offsets[row_idx]), int(offsets[row_idx + 1])
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
        }


# -----------------------------------------------------------------------------
# Task-aware collation
# -----------------------------------------------------------------------------


class MultimodalTaskCollator:
    """Turns a list of same-length rows into an 18-bit batch under a drawn task.

    Draws one task per example from ``task_weights`` with a per-batch RNG whose
    seed is derived deterministically from ``(self.seed, the batch's row ids)``.
    Deriving the seed from the batch *content* (not a mutable call counter) is
    what makes task sampling correct under DataLoader workers: the collate_fn is
    executed inside forked worker processes, so a counter incremented only in the
    main process would collide (every group of ``num_workers`` batches would share
    counter 0) and would never see the trainer's ``set_epoch`` / resume mutations.
    A content-derived seed instead gives the same batch the same tasks in every
    worker and after a resume (the batch-sampler reproduces the same rows), and
    different batches different tasks — independent of ``num_workers``.

    Builds the assembled ``x0`` bits, the per-residue modality state grid, and the
    sequence/structure target-loss and observed-clamp masks.
    """

    def __init__(
        self,
        *,
        task_weights: Optional[Dict[str, float]] = None,
        seq_codec: Optional[SequenceBitCodec] = None,
        struct_codec: LFQBitCodec = DEFAULT_STRUCT_CODEC,
        seed: int = 0,
    ):
        self.task_weights = dict(task_weights or T.DEFAULT_TASK_WEIGHTS)
        self.seq_codec = seq_codec or SequenceBitCodec()
        self.struct_codec = struct_codec
        self.seed = int(seed)
        self.epoch = 0
        self._counter = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._counter = 0

    def state_dict(self) -> Dict[str, int]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "counter": self._counter,
        }

    def load_state_dict(self, state: Dict[str, int]) -> None:
        self.seed = int(state["seed"])
        self.epoch = int(state["epoch"])
        self._counter = int(state["counter"])

    def _batch_rng(self, batch: List[Dict[str, object]]) -> np.random.Generator:
        """Deterministic per-batch RNG keyed by the batch's row identities.

        Worker-independent and resumable: the same set of rows always yields the
        same tasks (in any worker, on any resume), while different batches get
        different tasks. ``self.seed`` is fixed at construction so it survives the
        fork into DataLoader workers; ``_counter`` is advanced only to keep the
        ``state_dict`` API meaningful and no longer gates the draw.
        """
        idxs = np.array(
            sorted(int(r.get("row_index", i)) for i, r in enumerate(batch)),
            dtype=np.int64,
        )
        digest = hashlib.blake2b(idxs.tobytes(), digest_size=8).digest()
        batch_key = int.from_bytes(digest, "little")
        self._counter += 1
        return np.random.default_rng([int(self.seed), batch_key])

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, object]:
        if not _HAS_TORCH:
            raise RuntimeError("torch is required for collation")
        lengths = {int(r["length"]) for r in batch}
        if len(lengths) != 1:
            raise ValueError(
                "MultimodalTaskCollator requires same-length batches"
            )
        L = lengths.pop()
        B = len(batch)
        rng = self._batch_rng(batch)

        patch = np.zeros((B, L, PATCH_BITS_PER_RESIDUE), dtype=np.uint8)
        states = np.zeros((B, L, T.NUM_MODALITIES), dtype=np.int8)
        residue_mask = np.ones((B, L), dtype=bool)
        seq_avail = np.zeros((B, L), dtype=bool)
        struct_avail = np.zeros((B, L), dtype=bool)
        task_names: List[str] = []
        sources: List[str] = []

        for b, r in enumerate(batch):
            seq_bits = self.seq_codec.bits(r["seq_ids"])  # [L, 5]
            struct_bits = self.struct_codec.index_to_bits_np(
                r["struct_index"]
            )  # [L, 13]
            patch[b, :, :SEQ_BITS_PER_RESIDUE] = seq_bits
            patch[b, :, SEQ_BITS_PER_RESIDUE:] = struct_bits
            seq_avail[b] = np.asarray(r["seq_mask"], dtype=bool)
            struct_avail[b] = np.asarray(r["struct_mask"], dtype=bool)
            task = T.sample_task(rng, self.task_weights)
            task_names.append(task)
            sources.append(str(r.get("source", "unknown")))
            st = T.build_example_states(task, L)  # [L, 2] from task
            # Downgrade to ABSENT where a residue lacks that modality.
            st[~seq_avail[b], T.SEQ] = T.ABSENT
            st[~struct_avail[b], T.STRUCT] = T.ABSENT
            states[b] = st

        seq_target, struct_target = T.target_loss_masks_np(
            states, residue_mask
        )
        clamp_mask = T.observed_clamp_mask_np(states, residue_mask)

        seq_full, struct_full = T.full_modality_target_masks_np(
            states, residue_mask
        )
        out = {
            "x0": torch.from_numpy(
                patch.reshape(B, L * PATCH_BITS_PER_RESIDUE)
            ).float(),
            "residue_mask": torch.from_numpy(residue_mask),
            "states": torch.from_numpy(states.astype(np.int64)),
            "seq_target_mask": torch.from_numpy(seq_target),
            "struct_target_mask": torch.from_numpy(struct_target),
            "seq_target_full": torch.from_numpy(seq_full),
            "struct_target_full": torch.from_numpy(struct_full),
            "clamp_mask": torch.from_numpy(clamp_mask),
            "task_names": task_names,
            "sources": sources,
            "length": L,
        }
        return out


# -----------------------------------------------------------------------------
# Loader factory (mirrors data/uniref50.py get_evodiff_uniref50_loader)
# -----------------------------------------------------------------------------


def _cfg_task_weights(config) -> Optional[Dict[str, float]]:
    tw = getattr(config.data, "task_weights", None)
    if tw is None:
        return None
    return {str(k): float(v) for k, v in dict(tw).items()}


def _cfg_source_weights(config) -> Optional[Dict[str, float]]:
    sw = getattr(config.data, "source_weights", None)
    if sw is None:
        return None
    return {str(k): float(v) for k, v in dict(sw).items()}


def get_multimodal_loader(
    config,
    *,
    split: str,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    seed: int = 42,
):
    """Build a DataLoader for the paired multimodal dataset for one split.

    Train uses the resumable source/length-balanced sampler; validation uses a
    deterministic length-bucketed pass. The task-aware collator assembles the
    18-bit patch and the per-task masks.
    """
    if not _HAS_TORCH:
        raise RuntimeError("torch is required")
    from torch.utils.data import DataLoader

    from .protein_multimodal_sampler import SourceLengthBatchSampler
    from .proteins import DistributedLengthBucketBatchSampler, _ddp_rank_world

    shard_dir = getattr(config.data, "shard_dir", "datasets/dplm_paired_m0")
    min_len = int(getattr(config.data, "min_len", 1))
    max_len = int(getattr(config.data, "max_len", 512))
    dataset = ProteinMultimodalDataset(
        shard_dir, split=split, min_len=min_len, max_len=max_len
    )
    if shuffle is None:
        shuffle = split == "train"
    rank, world_size = _ddp_rank_world()
    local_bs = int(batch_size or config.train.batch_size)
    steps_per_epoch = int(getattr(config.train, "steps_per_epoch", 0))

    collator = MultimodalTaskCollator(
        task_weights=_cfg_task_weights(config),
        seed=int(getattr(config.train, "seed", 42)),
    )

    if split == "train" and steps_per_epoch > 0:
        sampler = SourceLengthBatchSampler(
            dataset.lengths,
            dataset.sources,
            batch_size=local_bs,
            num_batches=steps_per_epoch,
            seed=int(seed),
            rank=rank,
            world_size=world_size,
            source_weights=_cfg_source_weights(config),
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
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=bool(getattr(config.data, "pin_memory", True)),
        persistent_workers=num_workers > 0,
        prefetch_factor=int(getattr(config.data, "prefetch_factor", 2))
        if num_workers > 0
        else None,
    )
