import csv
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _resolve_eval_dirs(cfg):
    """
    Base:
      runs/{experiment}/evaluation
    Samples:
      runs/{experiment}/evaluation/samples
    CSV:
      runs/{experiment}/evaluation/results.csv
    """
    out_dir = Path(getattr(cfg.evaluation, "out_dir", f"runs/{cfg.experiment}/evaluation")).expanduser()
    samples_dir = Path(getattr(cfg.evaluation, "samples_dir", str(out_dir / "samples"))).expanduser()
    results_csv = Path(getattr(cfg.evaluation, "results_csv", str(out_dir / "results.csv"))).expanduser()
    return out_dir, samples_dir, results_csv


def _resolve_results_paths(cfg) -> Tuple[Path, Path]:
    """
    Returns:
      results_csv, results_jsonl
    """
    _, _, results_csv = _resolve_eval_dirs(cfg)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    results_jsonl = results_csv.with_suffix(".jsonl")
    return results_csv, results_jsonl


def _resolve_fid_checkpoints(cfg, fid_cfg, cli_list: Optional[List[str]]) -> List[Path]:
    """
    Returns a non-empty list of checkpoint paths for FID evaluation.
    """
    items = cli_list
    if items is None:
        items = getattr(fid_cfg, "checkpoints", None)
    if not items:
        return [Path(cfg.evaluation.checkpoint_path).expanduser()]

    run_root = Path(f"runs/{cfg.experiment}").expanduser()
    out: List[Path] = []

    def _expand_one(s: str) -> List[Path]:
        s = str(s)
        cands: List[Path] = []

        p1 = (run_root / s).expanduser()
        p2 = Path(s).expanduser()

        if p1.is_dir():
            cands = sorted(p1.glob("*.pt"))
            return cands
        if p2.is_dir():
            cands = sorted(p2.glob("*.pt"))
            return cands

        g1 = sorted(Path(x) for x in glob.glob(str(p1)))
        if len(g1) > 0:
            return g1
        g2 = sorted(Path(x) for x in glob.glob(str(p2)))
        if len(g2) > 0:
            return g2

        if p1.is_file():
            return [p1]
        if p2.is_file():
            return [p2]

        return []

    for s in items:
        out.extend(_expand_one(s))

    seen = set()
    uniq: List[Path] = []
    for p in out:
        ps = str(p.resolve())
        if ps not in seen:
            seen.add(ps)
            uniq.append(p)

    if len(uniq) == 0:
        raise FileNotFoundError(
            f"FID checkpoints resolved to empty. items={items}. "
            f"Tried relative to {run_root} and cwd."
        )

    return uniq


def _append_results_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    base_fields = [
        "timestamp_utc",
        "experiment",
        "framework",
        "dataset",
        "checkpoint",
        "metric",
        "split",
        "value",
        "details",
    ]
    extra_fields = sorted({k for r in rows for k in r.keys() if k not in base_fields})
    fieldnames = base_fields + extra_fields

    write_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        f.flush()
        os.fsync(f.fileno())


def _append_results_jsonl(jsonl_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _append_results_online(cfg, rows: List[Dict[str, Any]]) -> None:
    """
    Append rows immediately to both CSV and JSONL so partial progress survives
    interruptions.
    """
    if not rows:
        return

    results_csv, results_jsonl = _resolve_results_paths(cfg)
    _append_results_csv(results_csv, rows)
    _append_results_jsonl(results_jsonl, rows)


def _normalize_splits(maybe_splits: Optional[List[str]]) -> Optional[List[str]]:
    if maybe_splits is None:
        return None
    out: List[str] = []
    for s in maybe_splits:
        s = str(s).lower()
        if s == "both":
            out.extend(["train", "test"])
        else:
            out.append(s)

    seen = set()
    out2 = []
    for s in out:
        if s not in seen:
            seen.add(s)
            out2.append(s)
    return out2