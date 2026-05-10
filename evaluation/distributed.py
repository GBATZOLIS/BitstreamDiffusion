from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int


def is_torchrun_env() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and ("LOCAL_RANK" in os.environ)


def init_distributed_if_needed(backend: str = "nccl") -> DistInfo:
    """
    Initialize torch.distributed if launched under torchrun.

    Notes:
    - We use the standard init_process_group(...) path after setting CUDA device.
    - We intentionally do not pass device_id=... here. The simpler, conventional
      initialization path is less brittle for debugging first-collective issues.
    - Timeout remains 20 minutes because rank 0 may spend substantial time in
      single-rank evaluation work (for example HF external PPL scoring).
    """
    if not is_torchrun_env():
        return DistInfo(enabled=False, rank=0, local_rank=0, world_size=1)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    if dist.is_available() and not dist.is_initialized():
        timeout = datetime.timedelta(minutes=20)
        dist.init_process_group(
            backend=backend,
            timeout=timeout,
        )

    return DistInfo(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def barrier() -> None:
    """
    Plain process-group barrier.

    We intentionally avoid dist.barrier(device_ids=[...]) here while debugging
    first-collective NCCL issues. A plain barrier is the simplest and most
    portable synchronization primitive.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    dist.barrier()


def get_rank_world_size() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def shard_count(total: int, world_size: int, rank: int) -> int:
    """
    Split total items across ranks as evenly as possible (first ranks get remainder).
    """
    total = int(total)
    ws = int(world_size)
    r = int(rank)
    base = total // ws
    rem = total % ws
    return base + (1 if r < rem else 0)


def all_gather_int(value: int, device: torch.device) -> List[int]:
    """
    All-gather a single integer from all ranks (returns list of ints).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [int(value)]
    t = torch.tensor([int(value)], device=device, dtype=torch.long)
    outs = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(outs, t)
    return [int(x.item()) for x in outs]


def broadcast_bool(flag: bool, *, src: int = 0, device: Optional[torch.device] = None) -> bool:
    if not (dist.is_available() and dist.is_initialized()):
        return bool(flag)
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
    t = torch.tensor([1 if flag else 0], device=device, dtype=torch.uint8)
    dist.broadcast(t, src=src)
    return bool(int(t.item()))


def broadcast_tensor_inplace(t: torch.Tensor, *, src: int = 0) -> torch.Tensor:
    """
    Broadcast an already-allocated tensor in-place from src to all ranks.
    """
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(t, src=src)
    return t


def broadcast_optional_tensor(
    x: Optional[torch.Tensor],
    *,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    src: int = 0,
) -> Optional[torch.Tensor]:
    """
    Broadcast Optional[tensor] with known shape:
      - src rank provides x (or None)
      - all ranks receive None or a tensor with given shape/dtype/device
    """
    if not (dist.is_available() and dist.is_initialized()):
        return x

    rank = dist.get_rank()
    has = (x is not None) if (rank == src) else False
    has = broadcast_bool(has, src=src, device=device)

    if not has:
        return None

    if rank != src:
        x = torch.empty(shape, device=device, dtype=dtype)
    assert x is not None
    broadcast_tensor_inplace(x, src=src)
    return x


def gather_varlen_firstdim_to_rank0(
    x: torch.Tensor,
    *,
    dst: int = 0,
) -> Optional[torch.Tensor]:
    """
    Gather variable-length tensors along dim0 to dst rank.
    - Pads each rank to max_len along dim0, gathers fixed-size tensors, then slices.
    - Works with NCCL (CUDA tensors). For speed we expect x to be on GPU.
    Returns:
      - rank==dst: concatenated tensor [sum_i len_i, ...]
      - else: None
    """
    if not (dist.is_available() and dist.is_initialized()):
        return x

    assert x.dim() >= 1, "Expected at least 1D tensor"
    device = x.device
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_n = int(x.size(0))
    counts = all_gather_int(local_n, device=device)
    max_n = max(counts) if counts else local_n

    if local_n < max_n:
        pad_shape = (max_n - local_n, *x.shape[1:])
        pad = torch.zeros(pad_shape, device=device, dtype=x.dtype)
        x_pad = torch.cat([x, pad], dim=0)
    else:
        x_pad = x

    if rank == dst:
        gather_list = [torch.empty_like(x_pad) for _ in range(world_size)]
        dist.gather(x_pad, gather_list=gather_list, dst=dst)
        parts = []
        for i, t in enumerate(gather_list):
            ni = counts[i]
            if ni > 0:
                parts.append(t[:ni].contiguous())
        return torch.cat(parts, dim=0) if parts else x_pad[:0]
    else:
        dist.gather(x_pad, gather_list=None, dst=dst)
        return None


def set_global_seed(seed: int, rank: int = 0) -> None:
    """
    Deterministic-ish seeding for eval generation to reduce duplicates across ranks.
    """
    s = int(seed) + 1000 * int(rank)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)