"""Paper protein config with a global score temperature, for temperature sweeps.

    SCORE_TEMP=0.5 python scripts/proteins/evaluate.py forward \
        --config configs/proteins/proteins_400m_score_temp.py --checkpoint <ckpt>

Everything (checkpoint, DDIM entropic grid, 50 steps, CFG 5, churn) is the paper
config; only cfg.evaluation.score_temp changes, and results go to
<run>/evaluation/score_temp_<tau>/ so different temperatures never overwrite each
other or the paper results.  SCORE_TEMP is required (no silent default).
Optional SCORE_TEMP_FINAL=sharpened tempers the last step too (ablation only).
"""
import importlib.util
import os
from pathlib import Path


def get_config():
    spec = importlib.util.spec_from_file_location(
        "_proteins_400m", Path(__file__).with_name("proteins_400m.py")
    )
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    cfg = base.get_config()
    if "SCORE_TEMP" not in os.environ:
        raise ValueError("Set SCORE_TEMP (e.g. 1, 0.7, 0.5, 0.3) for this config")
    tau = float(os.environ["SCORE_TEMP"])
    if not 0 < tau <= 1:
        raise ValueError("SCORE_TEMP must be in (0, 1]")
    final = os.environ.get("SCORE_TEMP_FINAL", "unsharpened")
    if final not in ("unsharpened", "sharpened"):
        raise ValueError("SCORE_TEMP_FINAL must be 'unsharpened' or 'sharpened'")
    cfg.evaluation.score_temp = tau
    cfg.evaluation.score_temp_final = final
    tag = f"score_temp_{tau:g}" + ("" if final == "unsharpened" else "_sharpened")
    cfg.evaluation.out_dir = str(Path(cfg.evaluation.out_dir) / tag)
    cfg.evaluation.samples_dir = str(Path(cfg.evaluation.out_dir) / "samples")
    return cfg
