#!/usr/bin/env python
"""Scan runs/proteins and emit a single machine-readable results bundle.

Produces two artefacts under ``.trentinium/results/`` (the gitignored results
area, next to ``results.html``):

* ``results_data.json`` -- the canonical bundle (for HTTP-served viewing / drag & drop);
* ``results_data.js``   -- ``window.RESULTS_DATA = {...}`` so ``results.html`` renders
                           standalone over ``file://`` (script tags are not CORS-blocked).

Re-run this whenever new runs appear under ``runs/proteins/`` and refresh the page.

    .venv/bin/python scripts/proteins/run/build_results_data.py
"""

from __future__ import annotations

import glob
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = ROOT / "runs" / "proteins"
RESULTS_DIR = ROOT / ".trentinium" / "results"
OUT_JSON = RESULTS_DIR / "results_data.json"
OUT_JS = RESULTS_DIR / "results_data.js"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def clean(value):
    """Recursively convert NaN / Inf to None so the payload is valid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def load_json(path: Path):
    """Tolerant JSON load: the eval files legitimately contain bare ``NaN``."""
    text = path.read_text(encoding="utf-8")
    return json.loads(text, parse_constant=lambda _c: float("nan"))


def read_scalars(logdir: Path) -> dict:
    """Merge every ``events.out.tfevents.*`` under ``logdir`` into step->value series."""
    merged: dict[str, dict[int, tuple[float, float]]] = {}
    for ev in sorted(glob.glob(str(logdir / "events.out.tfevents.*"))):
        acc = EventAccumulator(ev, size_guidance={"scalars": 0})
        try:
            acc.Reload()
        except Exception:  # pragma: no cover - corrupt tail records
            continue
        for tag in acc.Tags().get("scalars", []):
            bucket = merged.setdefault(tag, {})
            for e in acc.Scalars(tag):
                prev = bucket.get(e.step)
                # keep the most recently written value for a repeated step
                if prev is None or e.wall_time >= prev[0]:
                    bucket[e.step] = (e.wall_time, e.value)
    series = {}
    for tag, bucket in merged.items():
        steps = sorted(bucket)
        series[tag] = {
            "steps": steps,
            "values": [
                (bucket[s][1] if math.isfinite(bucket[s][1]) else None)
                for s in steps
            ],
        }
    return series


CKPT_RE = re.compile(r"(epoch|step)=0*(\d+)-val=([0-9.]+)\.pt$")
CKPT_STEP_RE = re.compile(r"step=0*(\d+)\.pt$")


def read_checkpoints(run_dir: Path) -> list:
    out = []
    for ckpt_dir in (run_dir / "checkpoints", run_dir / "checkpoints_ar"):
        if not ckpt_dir.is_dir():
            continue
        for p in sorted(ckpt_dir.glob("*.pt")):
            name = p.name
            m = CKPT_RE.search(name)
            if m:
                out.append(
                    {
                        "name": name,
                        "kind": m.group(1),
                        "index": int(m.group(2)),
                        "val": float(m.group(3)),
                        "bytes": p.stat().st_size,
                    }
                )
            elif CKPT_STEP_RE.search(name):
                out.append(
                    {
                        "name": name,
                        "kind": "step",
                        "index": int(CKPT_STEP_RE.search(name).group(1)),
                        "val": None,
                        "bytes": p.stat().st_size,
                    }
                )
            elif name in ("best.pt", "last.pt"):
                out.append(
                    {
                        "name": name,
                        "kind": name.split(".")[0],
                        "index": None,
                        "val": None,
                        "bytes": p.stat().st_size,
                    }
                )
    return out


def read_evals(run_dir: Path) -> list:
    """Collect per-evaluation generation-quality metric JSONs."""
    evals = []
    eval_dir = run_dir / "protein_eval"
    if eval_dir.is_dir():
        for p in sorted(eval_dir.glob("*.json")):
            try:
                payload = load_json(p)
            except Exception:
                continue
            payload = {k: v for k, v in payload.items() if k != "artifacts"}
            payload["_name"] = p.stem
            evals.append(clean(payload))
    # frozen generation smoke metrics live in a different shape
    for fm in run_dir.glob("*/frozen_metrics.json"):
        try:
            payload = load_json(fm)
        except Exception:
            continue
        results = payload.get("results", {})
        results["_name"] = "frozen_generation"
        results["protocol_note"] = results.get("protocol")
        evals.append(clean(results))
    return evals


# ---------------------------------------------------------------------------
# run classification
# ---------------------------------------------------------------------------

DATASET_LABELS = {
    "SwissProt": "SwissProt (char)",
    "SwissProtDiMA": "SwissProt (DiMA protocol)",
    "EvoDiffUniRef50": "EvoDiff UniRef50 (2020)",
}


def classify(config: dict, is_ar: bool) -> dict:
    data = config.get("data", {})
    model = config.get("model", {})
    dataset_key = data.get("dataset", "unknown")
    representation = data.get("representation", "tokens")

    if is_ar:
        family = "Autoregressive Transformer"
        method = "autoregressive"
        params = None
        size = _fmt_params(_ar_param_estimate(model))
    elif model.get("name") == "sdt" and representation == "binary":
        family = "BitStream (CoBit-SDT)"
        method = "bitstream_diffusion"
        params = model.get("expected_num_parameters")
        size = (
            _fmt_params(params)
            if params
            else _fmt_params(_sdt_estimate(model))
        )
    elif model.get("name") == "sdt" and representation == "tokens":
        family = "Categorical Diffusion (SDT)"
        method = "categorical_diffusion"
        params = model.get("expected_num_parameters")
        size = _fmt_params(params) if params else None
    else:
        family = model.get("name", "unknown")
        method = "unknown"
        params = model.get("expected_num_parameters")
        size = _fmt_params(params) if params else None

    return {
        "family": family,
        "method": method,
        "params": params,
        "size": size,
        "dataset_key": dataset_key,
        "dataset_label": DATASET_LABELS.get(dataset_key, dataset_key),
        "representation": representation,
        "tokenizer": data.get("tokenizer") or ("tokens" if is_ar else None),
    }


def _fmt_params(n):
    if not n:
        return None
    return f"{n / 1e6:.1f}M"


def _sdt_estimate(model: dict):
    return model.get("expected_num_parameters")


def _ar_param_estimate(model: dict):
    d = model.get("d_model")
    layers = model.get("n_layer")
    vocab = model.get("vocab_size")
    mlp = model.get("mlp_mult", 4.0)
    if not (d and layers):
        return None
    per_layer = 4 * d * d + 2 * d * d * mlp
    embed = (vocab or 0) * d
    return int(layers * per_layer + 2 * embed)


def run_kind(name: str) -> str:
    lname = name.lower()
    if "smoke" in lname:
        return "smoke"
    if "viability" in lname:
        return "viability"
    return "production"


DISPLAY_NAMES = {
    "evodiff_uniref50_bitstream_seed0": "BitStream · UniRef50 (production)",
    "swissprot_char5": "BitStream · SwissProt-char (pilot)",
    "dima35m_bitstream_viability_seed0": "BitStream-35.8M · DiMA (viability)",
    "dima35m_bitstream_ddp_smoke": "BitStream-35.8M · DiMA (smoke)",
    "dima35m_bitstream_vlb_ddp_smoke": "BitStream-35.8M · DiMA VLB (smoke)",
    "dima_small_ar_ddp_smoke": "AR-Transformer · DiMA (smoke)",
    "dima_small_ar_bos_ddp_smoke": "AR-Transformer+BOS · DiMA (smoke)",
    "dima_small_categorical_ddp_smoke": "Categorical-SDT · DiMA (smoke)",
    "swissprot_char5_smoke": "BitStream · SwissProt-char (smoke)",
}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_run(run_dir: Path) -> dict | None:
    name = run_dir.name
    cfg_path = run_dir / "config.json"
    is_ar = False
    if not cfg_path.exists():
        cfg_path = run_dir / "config_ar.json"
        is_ar = True
    if not cfg_path.exists():
        return None
    try:
        config = load_json(cfg_path)
    except Exception:
        return None

    curves = {}
    for logname in ("training_logs", "training_logs_ar"):
        logdir = run_dir / logname
        if logdir.is_dir():
            curves.update(read_scalars(logdir))

    checkpoints = read_checkpoints(run_dir)
    evals = read_evals(run_dir)
    meta = classify(config, is_ar)

    optim = config.get("optim", {})
    train = config.get("train", {})
    system = config.get("system", {})
    data = config.get("data", {})

    def last(tag):
        s = curves.get(tag)
        if not s or not s["values"]:
            return None
        vals = [v for v in s["values"] if v is not None]
        return vals[-1] if vals else None

    def best(tag, mode="min"):
        s = curves.get(tag)
        if not s:
            return None
        vals = [v for v in s["values"] if v is not None]
        if not vals:
            return None
        return min(vals) if mode == "min" else max(vals)

    summary = {
        "final_train_loss": last("loss/epoch_train")
        if not is_ar
        else last("ar/bpt_train_step"),
        "final_val_loss": last("loss/epoch_val"),
        "best_val_loss": best("loss/epoch_val"),
        "final_vlb_bpd": last("VLB/val_bpd"),
        "best_vlb_bpd": best("VLB/val_bpd"),
        "final_vlb_npt": last("VLB/val_nats_per_token"),
        "best_vlb_npt": best("VLB/val_nats_per_token"),
        "ar_bpt_val": last("ar/bpt_val"),
        "ar_bpt_train": last("ar/bpt_train_step"),
        "sigma_data": last("sigma_data/estimate"),
    }

    # observed training progress from the curves
    step_tag = "loss/iter_train" if not is_ar else "ar/loss_nats"
    steps_seen = curves.get(step_tag, {}).get("steps", [])
    last_step = steps_seen[-1] if steps_seen else None
    epoch_idx = curves.get("training/epoch_index", {}).get("values", [])
    epochs_seen = (
        (int(max(v for v in epoch_idx if v is not None)) + 1)
        if epoch_idx
        else None
    )

    return {
        "id": name,
        "display_name": DISPLAY_NAMES.get(name, name),
        "kind": run_kind(name),
        "framework": config.get(
            "framework", "autoregressive" if is_ar else "unknown"
        ),
        "meta": meta,
        "planned_steps": optim.get("total_steps"),
        "observed_steps": last_step,
        "planned_epochs": train.get("epochs"),
        "observed_epochs": epochs_seen,
        "batch_size": train.get("batch_size"),
        "global_batch_size": train.get(
            "global_batch_size", train.get("batch_size")
        ),
        "world_size": system.get("world_size"),
        "lr": optim.get("lr"),
        "warmup": optim.get("warmup"),
        "scheduler": optim.get("scheduler"),
        "seed": train.get("seed"),
        "loss_type": train.get(
            "loss_type", "cross_entropy" if is_ar else None
        ),
        "loss_weighting": train.get("loss_weighting"),
        "bits_per_token": data.get("bits_per_token"),
        "vocab_size": data.get("vocab_size"),
        "sequence_len_tokens": data.get("sequence_len_tokens"),
        "max_len": data.get("max_len"),
        "min_len": data.get("min_len"),
        "protocol": data.get("protocol"),
        "num_sampling_steps": config.get("evaluation", {}).get(
            "num_sampling_steps"
        ),
        "model_config": _pick(config.get("model", {}), MODEL_KEYS),
        "summary": summary,
        "curves": curves,
        "checkpoints": checkpoints,
        "evals": evals,
    }


MODEL_KEYS = [
    "name",
    "embed_dim",
    "n_blocks",
    "n_heads",
    "dim_ff",
    "d_model",
    "n_layer",
    "n_head",
    "patch_size",
    "head_type",
    "continuous_logit_scaling",
    "out_dim",
    "self_condition",
    "use_rope_trunk",
    "use_swiglu",
    "use_adaln",
    "dropout",
]


def _pick(d: dict, keys):
    return {k: d[k] for k in keys if k in d}


def main():
    runs = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        run = build_run(run_dir)
        if run:
            runs.append(run)

    bundle = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_count": len(runs),
        "runs": clean(runs),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(bundle, indent=1), encoding="utf-8")
    OUT_JS.write_text(
        "window.RESULTS_DATA = "
        + json.dumps(bundle, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT_JSON}  ({OUT_JSON.stat().st_size / 1024:.0f} KB)")
    print(f"wrote {OUT_JS}   ({OUT_JS.stat().st_size / 1024:.0f} KB)")
    print(f"runs: {len(runs)}")
    for r in runs:
        n_curves = len(r["curves"])
        n_eval = len(r["evals"])
        print(
            f"  - {r['id']:42s} {r['kind']:11s} curves={n_curves:2d} evals={n_eval}"
        )


if __name__ == "__main__":
    main()
