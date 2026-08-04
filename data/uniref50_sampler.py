"""Step-based exact-length sampling for the 41M-row UniRef50 corpus."""
from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class DistributedRandomLengthBatchSampler(Sampler):
    """Draw deterministic, length-frequency-weighted, fixed-shape DDP batches."""

    def __init__(
        self, lengths, *, batch_size: int, num_batches: int, seed: int,
        rank: int, world_size: int,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        if self.batch_size <= 0 or self.num_batches <= 0 or self.world_size <= 0:
            raise ValueError("batch_size, num_batches, and world_size must be positive")

        order = np.argsort(self.lengths, kind="stable")
        values, starts, counts = np.unique(
            self.lengths[order], return_index=True, return_counts=True
        )
        self.values = values
        self.probabilities = counts.astype(np.float64) / counts.sum()
        self.buckets = {
            int(value): order[start : start + count]
            for value, start, count in zip(values, starts, counts)
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        global_batch = self.batch_size * self.world_size
        for _ in range(self.num_batches):
            length = int(rng.choice(self.values, p=self.probabilities))
            bucket = self.buckets[length]
            selected = rng.choice(
                bucket, size=global_batch, replace=len(bucket) < global_batch
            )
            start = self.rank * self.batch_size
            yield selected[start : start + self.batch_size].tolist()
