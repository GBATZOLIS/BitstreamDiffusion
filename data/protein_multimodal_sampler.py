"""Sample protein batches by source and sequence length."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    from torch.utils.data import Sampler

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    Sampler = object  # type: ignore
    _HAS_TORCH = False


class SourceLengthBatchSampler(Sampler):
    """
    Draw deterministic, source- and length-balanced, resumable DDP
    batches.
    """

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
        source_weights: dict[str, float] | None = None,
        cluster_ids: Sequence[str] | None = None,
        residue_budget_per_rank: int | None = None,
        max_batch_size: int | None = None,
        length_support_fraction: float = 0.0,
        length_support_boundaries: Sequence[int] | None = None,
        force_length: int | None = None,
        microbatches_per_update: int = 1,
        minimum_residue_budget_utilization: float = 0.0,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.sources = np.asarray([str(s) for s in sources])
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.residue_budget_per_rank = (
            None
            if residue_budget_per_rank is None
            else int(residue_budget_per_rank)
        )
        self.max_batch_size = int(max_batch_size or self.batch_size)
        self.length_support_fraction = float(length_support_fraction)
        self.length_support_boundaries = np.asarray(
            length_support_boundaries or (128, 256, 512, 768, 1024),
            dtype=np.int64,
        )
        self.force_length = None if force_length is None else int(force_length)
        self.microbatches_per_update = int(microbatches_per_update)
        self.minimum_residue_budget_utilization = float(
            minimum_residue_budget_utilization
        )
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
        if (
            self.residue_budget_per_rank is not None
            and self.residue_budget_per_rank <= 0
        ):
            raise ValueError("residue_budget_per_rank must be positive")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.microbatches_per_update <= 0:
            raise ValueError("microbatches_per_update must be positive")
        if not 0.0 <= self.minimum_residue_budget_utilization <= 1.0:
            raise ValueError(
                "minimum_residue_budget_utilization must be in [0, 1]"
            )
        if not 0.0 <= self.length_support_fraction <= 1.0:
            raise ValueError("length_support_fraction must be in [0, 1]")
        if (
            self.length_support_boundaries.ndim != 1
            or self.length_support_boundaries.size == 0
            or np.any(np.diff(self.length_support_boundaries) <= 0)
        ):
            raise ValueError(
                "length_support_boundaries must be strictly increasing"
            )
        if len(self.lengths) != len(self.sources):
            raise ValueError("lengths and sources must have equal length")

        if cluster_ids is None:
            self.cluster_ids = np.asarray(
                [f"row:{i}" for i in range(len(self.lengths))]
            )
        else:
            self.cluster_ids = np.asarray([str(c) for c in cluster_ids])
            if len(self.cluster_ids) != len(self.lengths):
                raise ValueError(
                    "cluster_ids and lengths must have equal length"
                )

        self.unique_sources = sorted(set(self.sources.tolist()))
        raw_counts = {
            s: int((self.sources == s).sum()) for s in self.unique_sources
        }
        if source_weights is None:
            # Equal per-source probability by default (full balancing
            # across sources).
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
            # Every present source is unweighted (e.g. a single-source
            # corpus whose label is absent from the configured paired
            # source_weights). Fall back to equal weighting over the
            # present sources rather than crashing, so a source-weight
            # misconfiguration degrades gracefully instead of aborting
            # the whole run.
            if not present.any():
                raise ValueError("no source has any rows")
            import warnings

            present_sources = [
                source
                for source, exists in zip(
                    self.unique_sources, present, strict=True
                )
                if exists
            ]
            warnings.warn(
                (
                    f"SourceLengthBatchSampler: none of the present "
                    f"sources {present_sources} carry a positive source "
                    f"weight; falling back to equal weighting."
                ),
                stacklevel=2,
            )
            w = present.astype(np.float64)
        self.source_probs = w / w.sum()

        # Per-source length buckets: {source: {length: indices}} and
        # length prior.
        self._buckets: dict[str, dict[int, np.ndarray]] = {}
        self._len_values: dict[str, np.ndarray] = {}
        self._len_probs: dict[str, np.ndarray] = {}
        self._row_probs: dict[str, dict[int, np.ndarray]] = {}
        self._support_bins: dict[str, list[np.ndarray]] = {}
        self._support_fraction_by_source: dict[str, float] = {}
        for s in self.unique_sources:
            sel = np.flatnonzero(self.sources == s)
            if sel.size == 0:
                continue
            src_lengths = self.lengths[sel]
            # Each cluster receives equal total mass inside a source,
            # and its members share that mass.  This prevents a large
            # homologous family from dominating merely because it has
            # many near-duplicate rows.
            src_clusters = self.cluster_ids[sel]
            _, inverse, cluster_counts = np.unique(
                src_clusters, return_inverse=True, return_counts=True
            )
            row_mass = 1.0 / cluster_counts[inverse].astype(np.float64)
            values = np.unique(src_lengths)
            if self.force_length is not None:
                values = values[values == self.force_length]
                if values.size == 0:
                    continue
            length_mass = np.asarray(
                [row_mass[src_lengths == value].sum() for value in values],
                dtype=np.float64,
            )
            self._len_values[s] = values
            self._len_probs[s] = length_mass / length_mass.sum()
            probs = self._len_probs[s]
            self._buckets[s] = {int(v): sel[src_lengths == v] for v in values}
            self._row_probs[s] = {}
            for value in values:
                weights = row_mass[src_lengths == value]
                self._row_probs[s][int(value)] = weights / weights.sum()
            support_bins: list[np.ndarray] = []
            lower = 0
            for upper in self.length_support_boundaries:
                support_bins.append(
                    values[(values > lower) & (values <= upper)]
                )
                lower = int(upper)
            self._support_bins[s] = [
                group for group in support_bins if group.size
            ]
            groups = self._support_bins[s]
            effective = self.length_support_fraction
            if groups and effective > 0.0:
                # Uniform broad-bin support is capped so no bin (and
                # therefore no equal-mass cluster inside it) receives
                # more than 2x its natural-stream probability. Any
                # unusable support mass falls back to the natural draw.
                q = 1.0 / len(groups)
                for group in groups:
                    p = float(probs[np.isin(values, group)].sum())
                    if q > p:
                        effective = min(effective, p / (q - p))
            self._support_fraction_by_source[s] = max(0.0, effective)

        available = np.asarray(
            [s in self._len_values for s in self.unique_sources], dtype=bool
        )
        self.source_probs = np.where(available, self.source_probs, 0.0)
        if self.source_probs.sum() <= 0:
            raise ValueError(
                f"no source contains forced length {self.force_length}"
            )
        self.source_probs /= self.source_probs.sum()

    def _local_batch_size(self, length: int) -> int:
        if self.residue_budget_per_rank is None:
            return self.batch_size
        local_batch = max(
            1,
            min(
                self.max_batch_size,
                self.residue_budget_per_rank // int(length),
            ),
        )
        realized = local_batch * int(length) / self.residue_budget_per_rank
        if realized + 1e-12 < self.minimum_residue_budget_utilization:
            raise RuntimeError(
                f"same-length token packing underfilled the registered "
                f"residue budget: length={length}, sequences={local_batch}"
                f", utilization={realized:.6f}, required>="
                f"{self.minimum_residue_budget_utilization:.6f}"
            )
        return local_batch

    def _draw_length(self, source: str, rng: np.random.Generator) -> int:
        values = self._len_values[source]
        probs = self._len_probs[source]
        if (
            self._support_fraction_by_source.get(source, 0.0) > 0
            and self._support_bins[source]
            and rng.random() < self._support_fraction_by_source[source]
        ):
            group = self._support_bins[source][
                int(rng.integers(0, len(self._support_bins[source])))
            ]
            mask = np.isin(values, group)
            group_probs = probs[mask]
            return int(rng.choice(group, p=group_probs / group_probs.sum()))
        return int(rng.choice(values, p=probs))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.start_batch = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "next_batch": self.start_batch,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.seed = int(state["seed"])
        self.epoch = int(state["epoch"])
        self.start_batch = int(state.get("next_batch", 0))

    def __len__(self) -> int:
        return self.num_batches

    def _draw_batch_indices(self, batch_id: int) -> list[int]:
        # Hold source and length fixed across an optimizer accumulation
        # window. This makes the equal microbatch average an exact
        # residue-normalized average within that update, while the row
        # draw remains independent.
        update_id = batch_id // self.microbatches_per_update
        rng = np.random.default_rng((self.seed, self.epoch, update_id, 0))
        s = self.unique_sources[
            int(rng.choice(len(self.unique_sources), p=self.source_probs))
        ]
        length = self._draw_length(s, rng)
        bucket = self._buckets[s][length]
        local_batch = self._local_batch_size(length)
        global_batch = local_batch * self.world_size
        row_rng = np.random.default_rng((self.seed, self.epoch, batch_id, 1))
        chosen = row_rng.choice(
            bucket,
            size=global_batch,
            replace=bucket.size < global_batch,
            p=self._row_probs[s][length],
        )
        start = self.rank * local_batch
        return chosen[start : start + local_batch].tolist()

    def __iter__(self):
        for batch_id in range(self.start_batch, self.num_batches):
            yield self._draw_batch_indices(batch_id)


class PrecomputedLengthBatchSampler(Sampler):
    """Exact-length sampler over a memory-mapped sorted row permutation."""

    def __init__(
        self,
        *,
        length_order,
        length_values,
        length_boundaries,
        batch_size: int,
        num_batches: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        residue_budget_per_rank: int | None = None,
        max_batch_size: int | None = None,
        force_length: int | None = None,
        microbatches_per_update: int = 1,
        minimum_residue_budget_utilization: float = 0.0,
    ):
        self.length_order = length_order
        self.length_values = np.asarray(length_values, dtype=np.int64)
        self.length_boundaries = np.asarray(length_boundaries, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.residue_budget_per_rank = (
            None
            if residue_budget_per_rank is None
            else int(residue_budget_per_rank)
        )
        self.max_batch_size = int(max_batch_size or self.batch_size)
        self.force_length = None if force_length is None else int(force_length)
        self.microbatches_per_update = int(microbatches_per_update)
        self.minimum_residue_budget_utilization = float(
            minimum_residue_budget_utilization
        )
        self.epoch = 0
        self.start_batch = 0
        counts = self.length_boundaries[:, 1] - self.length_boundaries[:, 0]
        if counts.size == 0 or int(counts.sum()) <= 0:
            raise ValueError("precomputed length index is empty")
        if (
            self.residue_budget_per_rank is not None
            and self.residue_budget_per_rank <= 0
        ):
            raise ValueError("residue_budget_per_rank must be positive")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.microbatches_per_update <= 0:
            raise ValueError("microbatches_per_update must be positive")
        if not 0.0 <= self.minimum_residue_budget_utilization <= 1.0:
            raise ValueError(
                "minimum_residue_budget_utilization must be in [0, 1]"
            )
        self.length_probs = counts.astype(np.float64) / counts.sum()
        if self.force_length is not None:
            mask = self.length_values == self.force_length
            if not bool(mask.any()):
                raise ValueError(
                    f"precomputed index has no length {self.force_length}"
                )
            self.length_probs = np.where(mask, self.length_probs, 0.0)
            self.length_probs /= self.length_probs.sum()

    def _local_batch_size(self, length: int) -> int:
        if self.residue_budget_per_rank is None:
            return self.batch_size
        local_batch = max(
            1,
            min(
                self.max_batch_size,
                self.residue_budget_per_rank // int(length),
            ),
        )
        realized = local_batch * int(length) / self.residue_budget_per_rank
        if realized + 1e-12 < self.minimum_residue_budget_utilization:
            raise RuntimeError(
                f"same-length token packing underfilled the registered "
                f"residue budget: length={length}, sequences={local_batch}"
                f", utilization={realized:.6f}, required>="
                f"{self.minimum_residue_budget_utilization:.6f}"
            )
        return local_batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.start_batch = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "next_batch": self.start_batch,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.seed = int(state["seed"])
        self.epoch = int(state["epoch"])
        self.start_batch = int(state.get("next_batch", 0))

    def __len__(self) -> int:
        return self.num_batches

    def _draw_batch_indices(self, batch_id: int) -> list[int]:
        update_id = batch_id // self.microbatches_per_update
        rng = np.random.default_rng((self.seed, self.epoch, update_id, 0))
        bucket_id = int(
            rng.choice(len(self.length_values), p=self.length_probs)
        )
        start, end = self.length_boundaries[bucket_id]
        bucket = self.length_order[int(start) : int(end)]
        local_batch = self._local_batch_size(
            int(self.length_values[bucket_id])
        )
        global_batch = local_batch * self.world_size
        row_rng = np.random.default_rng((self.seed, self.epoch, batch_id, 1))
        chosen = row_rng.choice(
            bucket, size=global_batch, replace=len(bucket) < global_batch
        )
        local_start = self.rank * local_batch
        return chosen[local_start : local_start + local_batch].tolist()

    def __iter__(self):
        for batch_id in range(self.start_batch, self.num_batches):
            yield self._draw_batch_indices(batch_id)


class TokenBudgetValidationBatchSampler(Sampler):
    """Deterministic complete validation pass with a per-rank residue cap."""

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        batch_size: int,
        residue_budget_per_rank: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = False,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.residue_budget_per_rank = int(residue_budget_per_rank)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        if (
            min(self.batch_size, self.residue_budget_per_rank, self.world_size)
            <= 0
        ):
            raise ValueError(
                "batch size, residue budget and world size must be positive"
            )
        self.dropped_rows = 0
        self.eligible_rows = int(len(self.lengths))
        self.replicated_fillers = sum(
            sum(1 for _idx, weight in batch if weight == 0.0)
            for batch in self._global_batches()
        )

    def _global_batches(self) -> list[list[tuple[int, float]]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.arange(len(self.lengths), dtype=np.int64)
        if self.shuffle:
            rng.shuffle(order)
            # Stable length sort preserves deterministic random order
            # within a length while keeping padding small across
            # neighbouring lengths.
            order = order[np.argsort(self.lengths[order], kind="stable")]
        else:
            order = order[np.argsort(self.lengths[order], kind="stable")]

        batches: list[list[tuple[int, float]]] = []
        cursor = 0
        while cursor < len(order):
            first = int(order[cursor])
            max_length = int(self.lengths[first])
            local_cap = self._local_batch_size(max_length)
            global_cap = local_cap * self.world_size
            end = min(len(order), cursor + global_cap)
            # Lengths are sorted, so the last row determines padded
            # memory. If it would violate the cap, shrink until it fits.
            # At least one row per rank is always allowed because max
            # supported L <= budget.
            while end > cursor:
                max_length = int(self.lengths[int(order[end - 1])])
                local_needed = (
                    end - cursor + self.world_size - 1
                ) // self.world_size
                if (
                    local_needed <= self.batch_size
                    and local_needed * max_length
                    <= self.residue_budget_per_rank
                ):
                    break
                end -= 1
            if end == cursor:
                raise ValueError(
                    f"validation row length {max_length} exceeds per-rank "
                    f"residue budget {self.residue_budget_per_rank}"
                )
            real = [int(i) for i in order[cursor:end]]
            local_size = (len(real) + self.world_size - 1) // self.world_size
            padded_size = local_size * self.world_size
            packed: list[tuple[int, float]] = [(i, 1.0) for i in real]
            packed.extend(
                (real[-1], 0.0) for _ in range(padded_size - len(real))
            )
            batches.append(packed)
            cursor = end
        return batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _local_batch_size(self, length: int) -> int:
        return max(
            1,
            min(self.batch_size, self.residue_budget_per_rank // int(length)),
        )

    def __len__(self) -> int:
        return len(self._global_batches())

    def __iter__(self):
        for group in self._global_batches():
            local_size = len(group) // self.world_size
            local_start = self.rank * local_size
            yield group[local_start : local_start + local_size]

