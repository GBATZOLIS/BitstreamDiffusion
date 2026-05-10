#evaluation/run_eval.py
import os
# Fix for some torch.compile interactions with CUDA Graphs
os.environ.setdefault("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", "1")

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import torch
import torch._dynamo
# Increase cache limit to match training
torch._dynamo.config.cache_size_limit = 64

from models import create_model
from utils.ema import EMA
from data import get_loader

from evaluation.utils import (
    load_config,
    unwrap_all,
    load_checkpoint,
    get_fid_cfg,
)

from evaluation.distributed import (
    init_distributed_if_needed,
    set_global_seed,
    barrier,
)

# ---- Import utility helpers ----
from evaluation.evaluation_drivers.utils import (
    _resolve_eval_dirs,
    _append_results_csv,
    _normalize_splits,
)

# ---- Import sub-drivers ----
from evaluation.evaluation_drivers.likelihood import evaluate_likelihood
from evaluation.evaluation_drivers.vlb import evaluate_vlb
from evaluation.evaluation_drivers.external_ppl import evaluate_external_ppl
from evaluation.evaluation_drivers.fid_sweep import evaluate_fid_sweep
from evaluation.evaluation_drivers.fid import evaluate_fid
from evaluation.evaluation_drivers.mauve import evaluate_mauve
from evaluation.evaluation_drivers.generate_samples import evaluate_generate_samples


# -----------------------------------------------------------------------------
# Main Orchestrator
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("Unified evaluation for discrete & continuous diffusion")
    ap.add_argument("--config", required=True, help="Path to config file")

    ap.add_argument(
        "--metrics", nargs="+", default=["bpc"],
        choices=["bpc", "bpd", "fid", "fid_sweep", "vlb", "external_ppl", "external_bpc", "external_bpt", "mauve", "generate_samples"],
        help="Which metrics to compute",
    )

    ap.add_argument("--compile", action="store_true", help="torch.compile the model for faster sampling")
    ap.add_argument("--compile_mode", type=str, default=None, help="Override torch.compile mode")
    ap.add_argument("--steps", type=int, default=100, help="ODE integration steps")
    ap.add_argument("--hutchinson", type=int, default=1, help="Hutchinson probes for PF-ODE divergence")
    ap.add_argument("--mc_samples", type=int, default=1, help="Monte Carlo samples for ELBO expectation")
    ap.add_argument("--schedule", type=str, default="entropic", choices=["karras", "entropic", "loguniform", "lognormal"])
    ap.add_argument("--sigma_min", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None, help="Override batch size")

    ap.add_argument("--fid_samples", type=int, default=10000)
    ap.add_argument("--fid_splits", nargs="+", default=None, choices=["train", "test", "both"])
    ap.add_argument("--fid_checkpoints", nargs="+", default=None)

    ap.add_argument("--max_batches", type=int, default=None, help="Limit evaluation batches (debug)")
    ap.add_argument("--sampler", type=str, default="tweedie", help="Discrete sampler for FID: tweedie|euler")
    ap.add_argument("--pf_method", type=str, default="heun", choices=["euler", "heun"])
    ap.add_argument("--no_ema", action="store_true")

    ap.add_argument("--vlb_splits", nargs="+", default=["test"], choices=["train", "test"])
    ap.add_argument("--vlb_sigma_sampling", type=str, default=None, choices=["log-uniform", "log-normal"])
    ap.add_argument("--vlb_sigma_min", type=float, default=None)
    ap.add_argument("--vlb_sigma_max", type=float, default=None)
    ap.add_argument("--vlb_mc", type=int, default=None)
    ap.add_argument("--vlb_include_prior", action="store_true")

    # MAUVE Arguments
    ap.add_argument("--mauve_checkpoints", nargs="+", default=None, help="Override MAUVE checkpoints list")
    ap.add_argument("--mauve_samples", type=int, default=None, help="Override MAUVE sample size")
    ap.add_argument("--mauve_micro_batch_size", type=int, default=None, help="Override MAUVE micro batch size")
    ap.add_argument("--mauve_samplers", nargs="+", default=None, help="Override MAUVE samplers")
    ap.add_argument("--mauve_terminal_sigmas", nargs="+", type=float, default=None, help="Override MAUVE terminal sigmas")
    ap.add_argument("--mauve_guidance_scales", nargs="+", type=float, default=None, help="Override MAUVE guidance scales")
    ap.add_argument("--mauve_steps", type=int, default=None, help="Override MAUVE sampling steps")

    # External-PPL checkpoint sweep / generation args
    ap.add_argument("--external_ppl_checkpoints", nargs="+", default=None, help="Override external-PPL checkpoints list")
    ap.add_argument("--external_ppl_num_samples", type=int, default=None, help="Override external-PPL sample size")
    ap.add_argument("--external_ppl_micro_batch_size", type=int, default=None, help="Override external-PPL generation micro batch size")
    ap.add_argument("--external_ppl_samplers", nargs="+", default=None, help="Override external-PPL samplers")
    ap.add_argument("--external_ppl_terminal_sigmas", nargs="+", type=float, default=None, help="Override external-PPL terminal sigmas")
    ap.add_argument("--external_ppl_guidance_scales", nargs="+", type=float, default=None, help="Override external-PPL guidance scales")
    ap.add_argument("--external_ppl_steps", type=int, default=None, help="Override external-PPL sampling steps")

    ap.add_argument("--generate_samples_checkpoints", nargs="+", default=None)
    ap.add_argument("--generate_samples_num_samples", type=int, default=None)
    ap.add_argument("--generate_samples_micro_batch_size", type=int, default=None)
    ap.add_argument("--generate_samples_prompt_len_tokens", type=int, default=None)
    ap.add_argument("--generate_samples_samplers", nargs="+", default=None)
    ap.add_argument("--generate_samples_terminal_sigmas", nargs="+", type=float, default=None)
    ap.add_argument("--generate_samples_guidance_scales", nargs="+", type=float, default=None)
    ap.add_argument("--generate_samples_steps", type=int, default=None)
    ap.add_argument("--generate_samples_seed", type=int, default=None)
    ap.add_argument("--generate_samples_sigma_max", type=str, default=None)  # pass "none" to unset

    args = ap.parse_args()
    cfg = load_config(args.config)

    # --- Compilation Settings Logic ---
    # Priority: 1. CLI args -> 2. cfg.evaluation.use_compile -> 3. cfg.train.use_compile
    if args.compile:
        compile_enabled = True
    else:
        eval_use_compile = getattr(cfg.evaluation, "use_compile", None)
        if eval_use_compile is not None:
            compile_enabled = bool(eval_use_compile)
        else:
            compile_enabled = bool(getattr(cfg.train, "use_compile", False))

    if args.compile_mode is not None:
        compile_mode = str(args.compile_mode)
    else:
        eval_compile_mode = getattr(cfg.evaluation, "compile_mode", None)
        if eval_compile_mode is not None:
            compile_mode = str(eval_compile_mode)
        else:
            compile_mode = str(getattr(cfg.train, "compile_mode", "default"))

    # Legacy compilation warmup args
    eval_compile_cfg = getattr(cfg.evaluation, "compile", None)
    compile_warmup = bool(getattr(eval_compile_cfg, "warmup", True))
    compile_warmup_steps = int(getattr(eval_compile_cfg, "warmup_steps", 8))

    fid_cfg = get_fid_cfg(cfg)
    cfg_fid_splits = getattr(fid_cfg, "splits", None) if fid_cfg is not None else None
    fid_splits = _normalize_splits(args.fid_splits) or _normalize_splits(cfg_fid_splits)
    if fid_splits is None:
        fid_splits = ["train", "test"]

    dist_info = init_distributed_if_needed()
    ddp_active = bool(dist_info.enabled and dist_info.world_size > 1)
    rank0 = (not ddp_active) or (dist_info.rank == 0)

    if ddp_active:
        device = torch.device(f"cuda:{dist_info.local_rank}")
    else:
        device = torch.device(cfg.device)

    # ✅ ALWAYS seed (DDP uses rank-specific offset, non-DDP uses rank=0)
    set_global_seed(int(getattr(cfg.train, "seed", 0) or 0), rank=(dist_info.rank if ddp_active else 0))

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.batch_size is not None:
        if rank0:
            print(f"⚠️ Overriding batch size: {cfg.train.batch_size} -> {args.batch_size}")
        cfg.train.batch_size = args.batch_size

    out_dir, samples_dir, results_csv = _resolve_eval_dirs(cfg)
    if rank0:
        out_dir.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(parents=True, exist_ok=True)

    cfg.evaluation.save_dir = str(samples_dir)

    run_meta = dict(
        timestamp_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        experiment=str(cfg.experiment),
        framework=str(cfg.framework),
        dataset=str(getattr(cfg.data, "dataset", "")),
        checkpoint=str(getattr(cfg.evaluation, "checkpoint_path", "")),
        world_size=int(dist_info.world_size if ddp_active else 1),
        compile_enabled=bool(compile_enabled),
        compile_mode=str(compile_mode),
    )
    if device.type == "cuda":
        run_meta["device_name"] = torch.cuda.get_device_name(device)

    results_rows: List[Dict[str, Any]] = []

    if rank0:
        print(f"🚀 Evaluation Mode: {cfg.experiment}")
        print(f"   Framework: {cfg.framework}")
        print(f"   Metrics:   {args.metrics}")
        if ddp_active:
            print(f"   DDP gen:   True (world_size={dist_info.world_size})")
        print(f"   Out dir:   {out_dir}")
        print(f"   Samples:   {samples_dir}")
        print(f"   CSV:       {results_csv}")
        print(f"   Compile:   {compile_enabled} (mode={compile_mode})")

    dataset_name = str(getattr(cfg.data, "dataset", "")).strip().lower()
    is_text8 = dataset_name == "text8"

    is_text_dataset = dataset_name in {
        "text8",
        "wikitext103",
        "wikitext-103",
        "wikitext",
        "openwebtext2",
        "openwebtext",
        "openwebtext",
        "openwebtext",
        "lm1b",
        "one-billion-word",
        "one_billion_word",
        "1bw",
    }

    need_test = any(
        m in args.metrics
        for m in ["bpc", "bpd", "external_ppl", "external_bpc", "external_bpt", "fid", "fid_sweep", "vlb", "mauve", "generate_samples"]
    )
    need_train = (
        ((("fid" in args.metrics) or ("fid_sweep" in args.metrics)) and ("train" in fid_splits))
        or (("vlb" in args.metrics) and ("train" in args.vlb_splits))
    )

    test_loader = get_loader(cfg, split="test") if need_test else None
    train_loader = get_loader(cfg, split="train") if need_train else None

    model = create_model(cfg).to(device)
    ema = EMA(unwrap_all(model), decay=0.0)
    use_ema = not args.no_ema
    load_checkpoint(model, ema, Path(cfg.evaluation.checkpoint_path), device, apply_ema=use_ema)
    if rank0:
        print(f"✓ Using {'EMA' if use_ema else 'RAW'} weights for evaluation.")

    with torch.no_grad():
        p = next(model.parameters())
        if rank0:
            print("param mean/std:", p.float().mean().item(), p.float().std().item())

    model.eval()

    # Apply compilation exactly as in training
    if compile_enabled and hasattr(torch, "compile"):
        model = torch.compile(model, mode=compile_mode, fullgraph=False)
        model.eval()
        if rank0:
            print(f"✓ torch.compile enabled for EVAL (mode={compile_mode}, fullgraph=False).")
    elif compile_enabled and not hasattr(torch, "compile"):
        if rank0:
            print("⚠️ torch.compile requested but not available in this PyTorch version.")

    # Metrics evaluation calls - routing directly to our modularized sub-drivers
    if rank0 and ("bpc" in args.metrics or "bpd" in args.metrics):
        evaluate_likelihood(args, cfg, model, test_loader, device, rank0, run_meta, results_rows, is_text_dataset)

    if "vlb" in args.metrics:
        evaluate_vlb(args, cfg, model, train_loader, test_loader, device, rank0, run_meta, results_rows)

    if any(m in args.metrics for m in ["external_ppl", "external_bpc", "external_bpt"]):
        evaluate_external_ppl(
            args, cfg, model, ema, use_ema, test_loader,
            device, rank0, ddp_active, run_meta, results_rows, is_text_dataset
        )

    if "fid_sweep" in args.metrics:
        evaluate_fid_sweep(args, cfg, model, test_loader, device, rank0, ddp_active, samples_dir, is_text_dataset)

    if "fid" in args.metrics:
        evaluate_fid(
            args, cfg, model, ema, use_ema, test_loader, train_loader, device,
            rank0, ddp_active, dist_info, samples_dir, fid_splits, run_meta,
            results_rows, compile_enabled, compile_warmup, compile_warmup_steps,
            is_text_dataset
        )

    if "mauve" in args.metrics:
        evaluate_mauve(args, cfg, model, ema, use_ema, test_loader, device, rank0, ddp_active, run_meta, results_rows, is_text_dataset)

    if "generate_samples" in args.metrics:
        evaluate_generate_samples(
            args, cfg, model, ema, use_ema, test_loader, device, rank0, ddp_active, run_meta, is_text_dataset
        )

    # Finalize writes
    barrier()
    if rank0:
        if len(results_rows) == 0:
            print("⚠️  No results were produced.")
        else:
            print(f"✓ Evaluation completed. Produced {len(results_rows)} in-memory rows.")
            print(f"✓ Results were appended online to: {results_csv}")
            print(f"✓ JSONL path: {results_csv.with_suffix('.jsonl')}")
    barrier()

    if ddp_active:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()