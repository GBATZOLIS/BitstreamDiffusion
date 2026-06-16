"""DDP-safe dataset build guard.

Ensures only global rank 0 generates/caches a dataset while other ranks wait,
then everyone loads the finished cache. Avoids redundant generation and
non-atomic concurrent writes to the same cache file under torchrun.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch.distributed as dist


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


@contextmanager
def rank0_first():
    """Yield True on the builder rank (rank 0 / single process), False otherwise.

    Non-builder ranks block until rank 0 has finished the guarded build.
    """
    builder = (not _is_dist()) or _rank() == 0
    if not builder:
        dist.barrier()  # wait for rank 0 to finish building
    try:
        yield builder
    finally:
        if builder and _is_dist():
            dist.barrier()  # release the waiting ranks
