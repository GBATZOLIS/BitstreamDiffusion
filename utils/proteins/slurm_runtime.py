"""Stop training safely at a deadline or after a shutdown signal."""

from __future__ import annotations

import os
import signal
import time

import torch
import torch.distributed as dist

# Set by the signal handlers. Module-level because a handler must do as
# little as possible, and because Python delivers signals to the main
# thread only.
_SIGNAL_RECEIVED = False
_SIGNAL_NAME = ""


def _handler(signum, frame):  # noqa: ARG001
    global _SIGNAL_RECEIVED, _SIGNAL_NAME
    _SIGNAL_RECEIVED = True
    _SIGNAL_NAME = signal.Signals(signum).name


def install_signal_handlers() -> None:
    """Trap the signals Slurm uses for preemption and walltime warnings."""
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Not the main thread, or the platform disallows it. The
            # deadline path still works, so this is not fatal.
            pass


def resolve_job_end_time() -> float | None:
    """Absolute epoch second at which Slurm will kill this job, if known."""
    for key in ("PROTEBIT_JOB_END_TIME", "SLURM_JOB_END_TIME"):
        raw = os.environ.get(key)
        if not raw:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


class DeadlineGuard:
    """Decide when to stop training, identically on every rank."""

    def __init__(
        self,
        *,
        rank: int = 0,
        world_size: int = 1,
        device=None,
        margin_seconds: float | None = None,
        max_wall_seconds: float | None = None,
        check_every_steps: int = 20,
        verbose: bool = True,
    ):
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = device
        self.check_every_steps = max(1, int(check_every_steps))
        self.verbose = bool(verbose)
        self.reason = ""

        if margin_seconds is None:
            margin_seconds = float(
                os.environ.get("PROTEBIT_EXIT_MARGIN_SECONDS", 600)
            )
        self.margin_seconds = float(margin_seconds)

        self.started_at = time.time()
        self.job_end_time = resolve_job_end_time()

        # Fallback when Slurm's end time is unavailable (an interactive
        # run, or a site where scontrol is not reachable from the
        # compute node).
        self.max_wall_seconds = (
            float(max_wall_seconds) if max_wall_seconds else None
        )

        self._stopping = False
        install_signal_handlers()

        if self.verbose and self.rank == 0:
            if self.job_end_time is not None:
                remain = self.job_end_time - self.started_at
                print(
                    f"[deadline] job ends in {remain / 60:.1f} min; will "
                    f"save and exit at T-{self.margin_seconds / 60:.1f} "
                    f"min (after {(remain - self.margin_seconds) / 60:.1f}"
                    f" min of training)"
                )
            elif self.max_wall_seconds:
                print(
                    f"[deadline] no Slurm end time; using "
                    f"max_wall_seconds={self.max_wall_seconds / 60:.1f} "
                    f"min"
                )
            else:
                print("[deadline] No walltime limit configured.")

    # -- local decision ----------------------------------------------------

    def _local_should_stop(self) -> tuple[bool, str]:
        if _SIGNAL_RECEIVED:
            return True, f"signal {_SIGNAL_NAME}"

        now = time.time()
        if self.job_end_time is not None:
            if now >= self.job_end_time - self.margin_seconds:
                left = self.job_end_time - now
                return True, f"walltime deadline ({left:.0f}s left)"
        elif self.max_wall_seconds is not None:
            if now - self.started_at >= self.max_wall_seconds:
                return True, "max_wall_seconds reached"
        return False, ""

    # -- collective decision ----------------------------------------------

    def should_stop(
        self, global_step: int, *, force_check: bool = False
    ) -> bool:
        """True when every rank should save and exit."""
        if self._stopping:
            return True

        if (
            not force_check
            and (int(global_step) % self.check_every_steps) != 0
        ):
            return False

        local, reason = self._local_should_stop()

        if (
            self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
        ):
            flag = torch.tensor(
                [1.0 if local else 0.0],
                device=self.device if self.device is not None else "cpu",
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            agreed = bool(flag.item() > 0.5)
        else:
            agreed = local

        if agreed:
            self._stopping = True
            self.reason = reason or "another rank requested stop"
            if self.rank == 0:
                print(
                    f"\n[deadline] stopping at step {global_step}: "
                    f"{self.reason}. Saving a resume checkpoint before "
                    f"exit."
                )
        return agreed

    # -- reporting ---------------------------------------------------------

    def seconds_remaining(self) -> float:
        if self.job_end_time is None:
            if self.max_wall_seconds is None:
                return float("inf")
            return self.max_wall_seconds - (time.time() - self.started_at)
        return self.job_end_time - time.time()

    def elapsed(self) -> float:
        return time.time() - self.started_at
