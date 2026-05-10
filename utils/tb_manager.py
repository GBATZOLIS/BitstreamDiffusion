import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, Any

from utils.safe_tb import SafeSummaryWriter


def _first_writable_dir(candidates) -> Optional[Path]:
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
            test = p / ".tb_write_test"
            test.write_text("ok")
            test.unlink()
            return p
        except Exception:
            continue
    return None


def _detect_scratch_root() -> Optional[Path]:
    candidates = [
        os.environ.get("SLURM_TMPDIR"),
        os.environ.get("TMPDIR"),
        os.environ.get("LOCAL_SCRATCH"),
        os.environ.get("SCRATCH"),
        f"/scratch/{os.environ.get('USER', '')}",
        "/tmp",
    ]
    return _first_writable_dir(candidates)


def choose_tb_dirs(
    run_dir: Path,
    experiment: str,
    tb_cfg: dict,
) -> Tuple[Path, Path, bool]:
    """
    Returns: (tb_write_dir, tb_canonical_dir, using_scratch)
    - tb_write_dir is where SummaryWriter writes
    - tb_canonical_dir is run_dir/training_logs (persistent)
    """
    canonical = run_dir / "training_logs"
    canonical.mkdir(parents=True, exist_ok=True)

    env_override = os.environ.get("TB_LOG_DIR", "").strip()
    if env_override:
        p = Path(env_override).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p, canonical, False

    mode = str(tb_cfg.get("log_dir", "auto")).lower()

    if mode not in {"auto", "run_dir", "scratch"}:
        p = Path(mode).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p, canonical, False

    if mode == "run_dir":
        return canonical, canonical, False

    scratch = _detect_scratch_root()
    if scratch is None:
        if mode == "scratch":
            raise RuntimeError("TB log_dir='scratch' but no writable scratch dir found.")
        return canonical, canonical, False

    tb_write = scratch / "tb_runs" / experiment
    tb_write.mkdir(parents=True, exist_ok=True)
    return tb_write, canonical, True


def sync_tb(tb_write_dir: Path, tb_canonical_dir: Path):
    """
    Copy event files back to canonical dir using atomic replace.
    """
    tb_canonical_dir.mkdir(parents=True, exist_ok=True)

    for src in tb_write_dir.glob("events.out.tfevents.*"):
        dst = tb_canonical_dir / src.name
        tmp = tb_canonical_dir / (src.name + ".tmp")

        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        except Exception as e:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            print(f"[TB] sync warning: failed copying {src} -> {dst}: {e}")


def copy_existing_events_to_scratch(tb_canonical_dir: Path, tb_write_dir: Path):
    tb_write_dir.mkdir(parents=True, exist_ok=True)

    for src in tb_canonical_dir.glob("events.out.tfevents.*"):
        dst = tb_write_dir / src.name
        if dst.exists():
            continue

        tmp = tb_write_dir / (src.name + ".tmp")
        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            pass


def clear_event_files(tb_dir: Path):
    """
    Delete TensorBoard event files in a directory, but leave the directory itself.
    """
    if not tb_dir.exists():
        return

    for p in tb_dir.glob("events.out.tfevents.*"):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[TB] cleanup warning: failed deleting {p}: {e}")


def make_writer(tb_write_dir: Path, tb_cfg: dict) -> SafeSummaryWriter:
    return SafeSummaryWriter(
        str(tb_write_dir),
        max_queue=int(tb_cfg.get("max_queue", 1000)),
        flush_secs=int(tb_cfg.get("flush_secs", 30)),
        fail_silently=bool(tb_cfg.get("fail_silently", True)),
    )


def _tb_cfg_to_dict(tb_cfg: Any) -> dict:
    if tb_cfg is None:
        return {}
    if hasattr(tb_cfg, "to_dict"):
        return tb_cfg.to_dict()
    if isinstance(tb_cfg, dict):
        return tb_cfg

    out = {}
    for k in dir(tb_cfg):
        if k.startswith("_"):
            continue
        try:
            out[k] = getattr(tb_cfg, k)
        except Exception:
            pass
    return out


class TBManager:
    """
    Handles:
      - choosing TB write dir (scratch vs run_dir)
      - explicit preparation for fresh run vs true resume
      - SafeSummaryWriter creation AFTER preparation
      - syncing event files back to canonical run_dir/<subdir>
    """

    def __init__(self, cfg: Any, run_dir: Path, subdir: str = "training_logs", is_master: bool = True):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.subdir = str(subdir)
        self.is_master = bool(is_master)

        tb_cfg_obj = getattr(getattr(cfg, "logging", None), "tensorboard", None)
        self.tb_cfg = _tb_cfg_to_dict(tb_cfg_obj)

        self.enabled = bool(self.tb_cfg.get("enabled", True))
        self.sync_to_run_dir = bool(self.tb_cfg.get("sync_to_run_dir", True))
        self.copy_existing_to_scratch = bool(self.tb_cfg.get("copy_existing_to_scratch", True))

        experiment = str(getattr(cfg, "experiment", self.run_dir.name))

        tb_write_dir, _tb_canonical_dir_unused, using_scratch = choose_tb_dirs(
            run_dir=self.run_dir,
            experiment=experiment,
            tb_cfg=self.tb_cfg,
        )

        tb_canonical_dir = self.run_dir / self.subdir
        tb_canonical_dir.mkdir(parents=True, exist_ok=True)

        if using_scratch:
            tb_write_dir = tb_write_dir / self.subdir
            tb_write_dir.mkdir(parents=True, exist_ok=True)

        self.tb_write_dir = tb_write_dir
        self.tb_canonical_dir = tb_canonical_dir
        self.using_scratch = using_scratch

        try:
            self._same_dir = (self.tb_write_dir.resolve() == self.tb_canonical_dir.resolve())
        except Exception:
            self._same_dir = False

        self.writer = None
        self._prepared = False

        print(
            f"[TB] enabled={self.enabled}, using_scratch={self.using_scratch}, "
            f"write_dir={self.tb_write_dir}, canonical_dir={self.tb_canonical_dir}"
        )

    def prepare_for_run(self, resume_mode: str):
        """
        Prepare TensorBoard directories based on run mode, then create writer.

        resume_mode:
          - 'resume'    : true checkpoint resume, preserve ALL event files
          - 'init_from' : fresh TB history
          - 'scratch'   : fresh TB history
        """
        if not self.enabled or not self.is_master:
            return

        if self.writer is not None:
            raise RuntimeError("[TB] prepare_for_run called after writer creation.")

        if resume_mode == "resume":
            if self.using_scratch and self.copy_existing_to_scratch:
                copy_existing_events_to_scratch(self.tb_canonical_dir, self.tb_write_dir)
                print(
                    "[TB] resume mode: preserved existing event files; "
                    "copied canonical -> write dir where missing."
                )
            else:
                print("[TB] resume mode: preserved existing event files.")

        elif resume_mode in {"init_from", "scratch"}:
            clear_event_files(self.tb_write_dir)
            if not self._same_dir:
                clear_event_files(self.tb_canonical_dir)

            print(
                f"[TB] {resume_mode} mode: cleared TensorBoard event files for fresh history.\n"
                f"      write_dir={self.tb_write_dir}\n"
                f"      canonical_dir={self.tb_canonical_dir}"
            )

        else:
            print(f"[TB] unknown resume_mode={resume_mode!r}; leaving event files unchanged.")

        self.writer = make_writer(self.tb_write_dir, self.tb_cfg)
        self._prepared = True
        print(f"[TB] writer created at: {self.tb_write_dir}")

    def flush(self):
        if not self.enabled or not self.is_master or self.writer is None:
            return
        try:
            self.writer.flush()
        except Exception as e:
            print(f"[TB] flush warning: {e}")

    def maybe_sync(self, step: int = 0, epoch: int = 0, flush: bool = True):
        if not self.enabled or not self.is_master:
            return
        if self.writer is None:
            return
        if not self.sync_to_run_dir:
            return
        if self._same_dir:
            return

        if flush:
            self.flush()

        try:
            sync_tb(self.tb_write_dir, self.tb_canonical_dir)
        except Exception as e:
            print(f"[TB] sync warning: {e}")

    def close(self):
        if not self.enabled or not self.is_master:
            return
        if self.writer is None:
            return

        try:
            self.flush()
        except Exception:
            pass
        try:
            self.maybe_sync(flush=False)
        except Exception as e:
            print(f"[TB] pre-close sync warning: {e}")

        try:
            self.writer.close()
        except Exception as e:
            print(f"[TB] close warning: {e}")

        try:
            self.maybe_sync(flush=False)
        except Exception as e:
            print(f"[TB] final sync warning: {e}")