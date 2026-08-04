"""Source/length-balanced, resumable, DDP-aware batch sampler for paired data.

Mirrors ``data/uniref50_sampler.py`` but adds two things the multimodal corpus
needs (plan 7.6 and section 11):

  - source balancing: experimental PDB and curated Swiss-Prot are oversampled
    relative to their raw counts so millions of synthetic AFDB/ESMAtlas rows do
    not swamp them; length buckets are drawn explicitly so short proteins do not
    dominate;
  - resumability: the sampler is deterministic given ``(seed, epoch, batch)`` and
    exposes ``state_dict``/``load_state_dict`` plus a ``start_batch`` cursor so a
    multi-day, multi-node run resumes at the exact next batch.

Batches are same-length (a single (source, length) bucket per step) so no padding
is needed and residue alignment between the two modalities is exact.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from torch.utils.data import Sampler

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    Sampler = object  # type: ignore
    _HAS_TORCH = False


class SourceLengthBatchSampler(Sampler):
    """Draw deterministic, source- and length-balanced, resumable DDP batches."""

    def __init__(
        self,
        lengths: Sequence[int],
        sources: Sequence[str],
        *,
        batch_size: int,
        num_batches: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        source_weights: Optional[Dict[str, float]] = None,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.sources = np.asarray([str(s) for s in sources])
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.start_batch = 0
        if (
            self.batch_size <= 0
            or self.num_batches <= 0
            or self.world_size <= 0
        ):
            raise ValueError(
                "batch_size, num_batches, world_size must be positive"
            )
        if len(self.lengths) != len(self.sources):
            raise ValueError("lengths and sources must have equal length")

        self.unique_sources = sorted(set(self.sources.tolist()))
        raw_counts = {
            s: int((self.sources == s).sum()) for s in self.unique_sources
        }
        if source_weights is None:
            # Equal per-source probability by default (full balancing across sources).
            source_weights = {s: 1.0 for s in self.unique_sources}
        w = np.asarray(
            [
                max(0.0, float(source_weights.get(s, 0.0)))
                for s in self.unique_sources
            ],
            dtype=np.float64,
        )
        present = np.asarray([raw_counts[s] > 0 for s in self.unique_sources])
        w = np.where(present, w, 0.0)
        if w.sum() <= 0:
            # Every present source is unweighted (e.g. a single-source corpus whose
            # label is absent from the configured paired source_weights). Fall back
            # to equal weighting over the present sources rather than crashing, so a
            # source-weight misconfiguration degrades gracefully instead of aborting
            # the whole run.
            if not present.any():
                raise ValueError("no source has any rows")
            import warnings

            warnings.warn(
                "SourceLengthBatchSampler: none of the present sources "
                f"{[s for s, p in zip(self.unique_sources, present) if p]} carry a "
                "positive source weight; falling back to equal weighting.",
                stacklevel=2,
            )
            w = present.astype(np.float64)
        self.source_probs = w / w.sum()

        # Per-source length buckets: {source: {length: indices}} and length prior.
        self._buckets: Dict[str, Dict[int, np.ndarray]] = {}
        self._len_values: Dict[str, np.ndarray] = {}
        self._len_probs: Dict[str, np.ndarray] = {}
        for s in self.unique_sources:
            sel = np.flatnonzero(self.sources == s)
            if sel.size == 0:
                continue
            src_lengths = self.lengths[sel]
            values, counts = np.unique(src_lengths, return_counts=True)
            self._len_values[s] = values
            self._len_probs[s] = counts.astype(np.float64) / counts.sum()
            self._buckets[s] = {int(v): sel[src_lengths == v] for v in values}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.start_batch = 0

    def state_dict(self) -> Dict[str, int]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "next_batch": self.start_batch,
        }

    def load_state_dict(self, state: Dict[str, int]) -> None:
        self.seed = int(state["seed"])
        self.epoch = int(state["epoch"])
        self.start_batch = int(state.get("next_batch", 0))

    def __len__(self) -> int:
        return self.num_batches

    def _draw_batch_indices(self, batch_id: int) -> List[int]:
        # A per-batch RNG keyed by (seed, epoch, batch_id) makes every batch
        # reproducible regardless of resume point.
        rng = np.random.default_rng((self.seed, self.epoch, batch_id))
        s = self.unique_sources[
            int(rng.choice(len(self.unique_sources), p=self.source_probs))
        ]
        values = self._len_values[s]
        length = int(rng.choice(values, p=self._len_probs[s]))
        bucket = self._buckets[s][length]
        global_batch = self.batch_size * self.world_size
        chosen = rng.choice(
            bucket, size=global_batch, replace=bucket.size < global_batch
        )
        start = self.rank * self.batch_size
        return chosen[start : start + self.batch_size].tolist()

    def __iter__(self):
        for batch_id in range(self.start_batch, self.num_batches):
            yield self._draw_batch_indices(batch_id)
