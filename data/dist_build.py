"""DDP-safe dataset build guard.

Ensures only global rank 0 generates/caches a dataset while other ranks wait,
then everyone loads the finished cache. Avoids redundant generation and
non-atomic concurrent writes to the same cache file under torchrun.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.distributed as dist


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _barrier() -> None:
    """NCCL-safe barrier.

    Binding device_ids avoids the "devices used by this process are currently
    unknown ... can cause a hang" failure when this is the first/early NCCL
    collective and the rank->GPU mapping has not been pinned via
    init_process_group(device_id=...). The training device is already set by the
    trainer before datasets are built (torch.cuda.set_device).
    """
    if not _is_dist():
        return
    backend = dist.get_backend()
    if backend == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


@contextmanager
def rank0_first():
    """Yield True on the builder rank (rank 0 / single process), False otherwise.

    Non-builder ranks block until rank 0 has finished the guarded build.
    """
    builder = (not _is_dist()) or _rank() == 0
    if not builder:
        _barrier()  # wait for rank 0 to finish building
    try:
        yield builder
    finally:
        if builder and _is_dist():
            _barrier()  # release the waiting ranks
