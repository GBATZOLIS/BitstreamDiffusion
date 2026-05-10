from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from diffusion.continuous.processes import ContinuousForwardProcess
from diffusion.discrete.processes import DiscreteForwardProcess
from evaluation.mauve import MauveEvaluator, MauveConfig
from evaluation.nfe import compute_nfe
from evaluation.sampling_specs import build_sampling_specs
from evaluation.text_generations import (
    SharedGenerationCache,
    resolve_checkpoints,
)
from evaluation.text_metrics import rep_n, distinct_n, diversity_234
from evaluation.utils import load_checkpoint
from evaluation.evaluation_drivers.utils import _append_results_online


def _shared_cache_root(cfg, metric_cfg) -> Path:
    root = getattr(metric_cfg, "shared_cache_dir", None)
    if root is None:
        root = getattr(getattr(cfg, "evaluation", object()), "shared_text_cache_dir", None)
    if root is None:
        root = Path(cfg.evaluation.out_dir) / "shared_text_cache"
    return Path(root)


def _generation_meta_fields(meta: Dict[str, Any]) -> Dict[str, Any]:
    rename_map = {
        "generation_wall_time_sec": "generation_wall_time_sec",
        "world_size": "world_size",
        "num_samples": "generated_num_samples",
        "sequence_len_tokens": "sequence_len_tokens",
        "sequence_len_model": "sequence_len_model",
        "full_lm_tokens": "full_lm_tokens",
        "generated_lm_tokens": "generated_lm_tokens",
        "full_model_positions": "full_model_positions",
        "generated_model_positions": "generated_model_positions",
        "samples_per_sec": "samples_per_sec",
        "full_lm_tokens_per_sec": "full_lm_tokens_per_sec",
        "generated_lm_tokens_per_sec": "generated_lm_tokens_per_sec",
        "lm_tokens_per_sec": "lm_tokens_per_sec",
        "full_model_positions_per_sec": "full_model_positions_per_sec",
        "generated_model_positions_per_sec": "generated_model_positions_per_sec",
        "model_positions_per_sec": "model_positions_per_sec",
    }
    out: Dict[str, Any] = {}
    for src_k, dst_k in rename_map.items():
        v = meta.get(src_k, None)
        if isinstance(v, (int, float)):
            out[dst_k] = v
    return out


def _append_metric_rows(
    *,
    rows_out: List[Dict[str, Any]],
    run_meta: Dict[str, Any],
    base_fields: Dict[str, Any],
    metric_prefix: str,
    metric_dict: Dict[str, float],
    details: str,
):
    for k, v in metric_dict.items():
        if isinstance(v, (int, float)) and not np.isnan(v):
            row: Dict[str, Any] = {}
            row.update(run_meta)
            row.update(base_fields)
            row["metric"] = f"{metric_prefix}_{k}"
            row["value"] = float(v)
            row["details"] = details
            rows_out.append(row)


def _resolve_mauve_device_id(cfg, m_cfg, device: torch.device) -> Optional[int]:
    # Highest priority: evaluation.mauve.device_id
    explicit = getattr(m_cfg, "device_id", None)
    if explicit is not None:
        return int(explicit)

    # Fallback: train.mauve.device_id
    train_mauve = getattr(getattr(cfg, "train", object()), "mauve", None)
    if train_mauve is not None:
        explicit_train = getattr(train_mauve, "device_id", None)
        if explicit_train is not None:
            return int(explicit_train)

    # Infer from torch.device
    if device.type != "cuda":
        return None

    # torch.device("cuda:0") -> index=0
    if device.index is not None:
        return int(device.index)

    # torch.device("cuda") in single-GPU eval often has index=None; use 0.
    return 0


def evaluate_mauve(
    args,
    cfg,
    model,
    ema,
    use_ema,
    test_loader,
    device,
    rank0,
    ddp_active,
    run_meta,
    results_rows,
    is_text_dataset,
):
    if not is_text_dataset:
        if rank0:
            print("\n⚠️  Skipping MAUVE for non-text dataset.")
        return

    if rank0:
        print("\n── Calculating MAUVE (shared generations; full + suffix) ──")

    eval_cfg = getattr(cfg, "evaluation", object())
    m_cfg = getattr(eval_cfg, "mauve", getattr(getattr(cfg, "train", object()), "mauve", None))
    if m_cfg is None:
        raise ValueError("Missing cfg.evaluation.mauve (or cfg.train.mauve).")

    n_samples = int(args.mauve_samples or getattr(m_cfg, "num_samples", 4096))
    micro_bs = int(args.mauve_micro_batch_size or getattr(m_cfg, "micro_batch_size", 512))
    samplers = list(args.mauve_samplers or getattr(m_cfg, "samplers", ["heun_karras"]))
    terminal_sigmas = list(args.mauve_terminal_sigmas or getattr(m_cfg, "terminal_sigmas", [0.08]))
    guidance_scales = list(args.mauve_guidance_scales or getattr(m_cfg, "guidance_scales", [0.0]))
    num_steps = int(
        args.mauve_steps
        or getattr(m_cfg, "num_sampling_steps", getattr(cfg.evaluation, "num_sampling_steps", 128))
    )
    feat_name = str(getattr(m_cfg, "featurizer_name", "gpt2-large"))
    max_len = int(getattr(m_cfg, "max_tokens", getattr(cfg.data, "sequence_len_tokens", 256)))
    sigma_max = getattr(m_cfg, "sigma_max", None)
    sigma_max = None if sigma_max is None else float(sigma_max)
    seed = int(getattr(m_cfg, "seed", 42))
    compute_full = bool(getattr(m_cfg, "compute_full_text", True))
    compute_suffix = bool(getattr(m_cfg, "compute_suffix_text", True))
    compute_repetition = bool(getattr(m_cfg, "compute_repetition", True))

    # Clean featurizer batch size control
    mauve_batch_size = int(getattr(m_cfg, "batch_size", 128))
    mauve_device_id = _resolve_mauve_device_id(cfg, m_cfg, device)

    ckpt_spec = args.mauve_checkpoints or getattr(m_cfg, "checkpoints", None)
    base_ckpt_dir = Path(cfg.evaluation.checkpoint_path).parent
    checkpoints_to_run = resolve_checkpoints(ckpt_spec, base_ckpt_dir)
    if ckpt_spec is None and len(checkpoints_to_run) == 0:
        checkpoints_to_run = [None]

    cache_root = _shared_cache_root(cfg, m_cfg)
    shared_cache = SharedGenerationCache(cfg, cache_root)

    if cfg.framework == "continuous_score":
        proc = ContinuousForwardProcess(cfg)
    elif cfg.framework == "discrete_sedd":
        proc = DiscreteForwardProcess(cfg)
    else:
        raise ValueError(f"Unknown framework: {cfg.framework}")

    mauve_evaluator = None
    if rank0:
        print(
            f"   MAUVE featurizer device_id={mauve_device_id}, "
            f"batch_size={mauve_batch_size}, model={feat_name}"
        )
        mauve_evaluator = MauveEvaluator(
            cfg=MauveConfig(
                featurize_model_name=feat_name,
                max_text_length=max_len,
                device_id=mauve_device_id,
                batch_size=mauve_batch_size,
            )
        )

    sampling_specs = build_sampling_specs(cfg=cfg, metric_cfg=m_cfg)

    if sampling_specs is None:
        tag_specs = []
        if str(cfg.framework).lower().startswith("discrete"):
            for sampler_name in samplers:
                tag_specs.append(
                    dict(
                        tag=str(sampler_name),
                        sampler_name=str(sampler_name),
                        terminal_sigma=0.0,
                        guidance_scale=0.0,
                        num_steps=num_steps,
                        sc_refresh_mode="refined",
                        target_nfe=None,
                        actual_nfe=compute_nfe(
                            framework=str(cfg.framework),
                            sampler_name=str(sampler_name),
                            num_steps=num_steps,
                            self_condition=bool(getattr(cfg.model, "self_condition", False)),
                            sc_refresh_mode="refined",
                            return_probs=True,
                            discrete_denoise=True,
                        ),
                    )
                )
        else:
            for sampler_name in samplers:
                for sigma in terminal_sigmas:
                    for gs in guidance_scales:
                        actual_nfe = compute_nfe(
                            framework=str(cfg.framework),
                            sampler_name=str(sampler_name),
                            num_steps=num_steps,
                            self_condition=bool(getattr(cfg.model, "self_condition", False)),
                            sc_refresh_mode="refined",
                            return_probs=True,
                            discrete_denoise=True,
                        )
                        tag_specs.append(
                            dict(
                                tag=f"{sampler_name}_term{math.log10(float(sigma)):.2f}_gs{float(gs):.1f}",
                                sampler_name=str(sampler_name),
                                terminal_sigma=float(sigma),
                                guidance_scale=float(gs),
                                num_steps=num_steps,
                                sc_refresh_mode="refined",
                                target_nfe=None,
                                actual_nfe=actual_nfe,
                            )
                        )
    else:
        tag_specs = sampling_specs

    for ckpt_path in checkpoints_to_run:
        curr_run_meta = copy.deepcopy(run_meta)

        if ckpt_path is not None:
            ckpt_name = ckpt_path.name
            if rank0:
                print(f"\n   -> Checkpoint: {ckpt_name}")
            load_checkpoint(model, ema, ckpt_path, device, apply_ema=use_ema)
            model.eval()
            curr_run_meta["checkpoint"] = ckpt_name
        else:
            ckpt_name = Path(cfg.evaluation.checkpoint_path).name
            curr_run_meta["checkpoint"] = ckpt_name

        for spec in tag_specs:
            tag = spec["tag"]
            sampler_name = str(spec["sampler_name"])
            spec_steps = int(spec["num_steps"])
            terminal_sigma = float(spec["terminal_sigma"])
            guidance_scale = float(spec["guidance_scale"])
            sc_refresh_mode = str(spec.get("sc_refresh_mode", "refined"))
            target_nfe = spec.get("target_nfe", None)
            actual_nfe = int(spec["actual_nfe"])

            cache_key = shared_cache.make_cache_key(
                checkpoint_name=ckpt_name,
                split="test",
                num_samples=n_samples,
                num_steps=spec_steps,
                micro_batch_size=micro_bs,
                sigma_max=sigma_max,
                use_ema=use_ema,
                seed=seed,
            )

            texts = shared_cache.get_or_create(
                checkpoint_name=ckpt_name,
                split="test",
                cache_key=cache_key,
                tag=tag,
                model=model,
                proc=proc,
                device=device,
                data_loader=test_loader,
                num_samples=n_samples,
                sampler_names=[sampler_name],
                terminal_sigmas=[terminal_sigma],
                guidance_scales=[guidance_scale],
                num_steps=spec_steps,
                seed=seed,
                use_amp=True,
                amp_dtype="auto",
                micro_batch_size=micro_bs,
                sigma_max=sigma_max,
                meta=dict(
                    tag=tag,
                    sampler_name=sampler_name,
                    terminal_sigma=terminal_sigma,
                    guidance_scale=guidance_scale,
                    num_samples=n_samples,
                    num_steps=spec_steps,
                    nfe=actual_nfe,
                    target_nfe=target_nfe,
                    sc_refresh_mode=sc_refresh_mode,
                    sigma_max=sigma_max,
                    use_ema=bool(use_ema),
                ),
                sampling_specs=[spec],
            )

            if not rank0:
                continue

            gen_meta_fields = _generation_meta_fields(texts.meta)

            base_fields = dict(
                split="test",
                tag=tag,
                sampler_name=sampler_name,
                terminal_sigma=terminal_sigma,
                guidance_scale=guidance_scale,
                num_sampling_steps=spec_steps,
                micro_batch_size=micro_bs,
                num_samples=n_samples,
                featurizer_name=feat_name,
                max_text_length=max_len,
                sigma_max=sigma_max,
                use_ema=bool(use_ema),
                nfe=actual_nfe,
                target_nfe=target_nfe,
                sc_refresh_mode=sc_refresh_mode,
                mauve_device_id=mauve_device_id,
                mauve_batch_size=mauve_batch_size,
                **gen_meta_fields,
            )

            rows_out: List[Dict[str, Any]] = []

            if compute_full:
                with torch.amp.autocast("cuda", enabled=False):
                    res_full = mauve_evaluator.score(
                        p_text=texts.full,
                        q_text=texts.ref_full,
                    )

                _append_metric_rows(
                    rows_out=rows_out,
                    run_meta=curr_run_meta,
                    base_fields=base_fields,
                    metric_prefix="full",
                    metric_dict=res_full,
                    details=f"view=full,tag={tag},N={n_samples},steps={spec_steps},nfe={actual_nfe}",
                )
                print(f"      MAUVE full   [{tag}]: {res_full.get('mauve', float('nan')):.4f}")

            if compute_suffix:
                with torch.amp.autocast("cuda", enabled=False):
                    res_suffix = mauve_evaluator.score(
                        p_text=texts.suffix,
                        q_text=texts.ref_suffix,
                    )

                _append_metric_rows(
                    rows_out=rows_out,
                    run_meta=curr_run_meta,
                    base_fields=base_fields,
                    metric_prefix="suffix",
                    metric_dict=res_suffix,
                    details=f"view=suffix,tag={tag},N={n_samples},steps={spec_steps},nfe={actual_nfe}",
                )
                print(f"      MAUVE suffix [{tag}]: {res_suffix.get('mauve', float('nan')):.4f}")

            if compute_repetition:
                rep_metrics = {
                    "rep2_suffix": rep_n(texts.suffix, 2),
                    "rep3_suffix": rep_n(texts.suffix, 3),
                    "rep4_suffix": rep_n(texts.suffix, 4),
                    "distinct2_suffix": distinct_n(texts.suffix, 2),
                    "distinct3_suffix": distinct_n(texts.suffix, 3),
                    "distinct4_suffix": distinct_n(texts.suffix, 4),
                    "diversity_234_suffix": diversity_234(texts.suffix),
                }
                _append_metric_rows(
                    rows_out=rows_out,
                    run_meta=curr_run_meta,
                    base_fields=base_fields,
                    metric_prefix="rep",
                    metric_dict=rep_metrics,
                    details=f"view=suffix,tag={tag},N={n_samples},steps={spec_steps},nfe={actual_nfe}",
                )

            results_rows.extend(rows_out)
            _append_results_online(cfg, rows_out)