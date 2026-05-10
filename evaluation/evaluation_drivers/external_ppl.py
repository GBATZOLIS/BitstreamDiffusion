# evaluation/evaluation_drivers/external_ppl.py
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from diffusion.continuous.processes import ContinuousForwardProcess
from diffusion.discrete.processes import DiscreteForwardProcess
from evaluation.distributed import barrier as ddp_barrier
from evaluation.external_perplexity import HFExternalPerplexityEvaluator
from evaluation.nfe import compute_nfe
from evaluation.sampling_specs import build_sampling_specs
from evaluation.text_generations import (
    SharedGenerationCache,
    resolve_checkpoints,
)
from evaluation.utils import load_checkpoint
from evaluation.evaluation_drivers.utils import _append_results_online
from utils.text_decode import (
    debug_gpt2id_bpe16_decode_batch,
    load_openwebtext_gpt2id_bpe16_assets,
)


def _shared_cache_root(cfg, metric_cfg) -> Path:
    root = getattr(metric_cfg, "shared_cache_dir", None)
    if root is None:
        root = getattr(getattr(cfg, "evaluation", object()), "shared_text_cache_dir", None)
    if root is None:
        root = Path(cfg.evaluation.out_dir) / "shared_text_cache"
    return Path(root)


def _resolve_effective_stochastic_fields(cfg, spec: Dict[str, Any]) -> Dict[str, Any]:
    st_cfg = getattr(getattr(cfg, "evaluation", object()), "stochastic", None)

    def _cfg_get(name: str, default=None):
        if st_cfg is None:
            return default
        return getattr(st_cfg, name, default)

    enabled = bool(spec.get("stochastic_enabled", _cfg_get("enabled", False)))

    out: Dict[str, Any] = {
        "stochastic_enabled": enabled,
    }

    if not enabled:
        return out

    out.update(
        s_churn=float(spec.get("s_churn", _cfg_get("s_churn", 0.0))),
        s_noise=float(spec.get("s_noise", _cfg_get("s_noise", 1.0))),
        window_mode=str(spec.get("window_mode", _cfg_get("window_mode", "entropy_cdf"))),
        entropy_quantile_lo=spec.get("entropy_quantile_lo", _cfg_get("entropy_quantile_lo", None)),
        entropy_quantile_hi=spec.get("entropy_quantile_hi", _cfg_get("entropy_quantile_hi", None)),
        s_tmin=spec.get("s_tmin", _cfg_get("s_tmin", None)),
        s_tmax=spec.get("s_tmax", _cfg_get("s_tmax", None)),
        entropy_fallback=str(spec.get("entropy_fallback", _cfg_get("entropy_fallback", "deterministic"))),
    )
    return out


def _generation_meta_fields(meta: Dict[str, object]) -> Dict[str, object]:
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
    out: Dict[str, object] = {}
    for src_k, dst_k in rename_map.items():
        v = meta.get(src_k, None)
        if isinstance(v, (int, float)):
            out[dst_k] = v
    return out


def _append_result_rows(
    *,
    rows_out: List[Dict[str, Any]],
    run_meta: Dict[str, Any],
    base_fields: Dict[str, Any],
    metric_dict: Dict[str, float],
    metric_prefix: str,
    details: str,
):
    for k, v in metric_dict.items():
        row: Dict[str, Any] = {}
        row.update(run_meta)
        row.update(base_fields)
        row["metric"] = f"{metric_prefix}_{k}"
        row["value"] = float(v)
        row["details"] = details
        rows_out.append(row)


def _debug_text_batch_stats(name: str, texts: List[str], *, max_preview: int = 2) -> None:
    if texts is None:
        print(f"[eval external_ppl][debug] {name}: None", flush=True)
        return

    n = len(texts)
    lengths_chars = [len(t) for t in texts]
    empty_count = sum(1 for t in texts if len(t.strip()) == 0)
    eos_literal_count = sum(t.count("<|endoftext|>") for t in texts)

    lengths_words: List[int] = []
    for t in texts:
        s = t.strip()
        lengths_words.append(0 if not s else len(s.split()))

    def _safe_stats(xs: List[int]) -> str:
        if not xs:
            return "min=NA mean=NA max=NA"
        return f"min={min(xs)} mean={sum(xs) / len(xs):.2f} max={max(xs)}"

    print(
        f"[eval external_ppl][debug] {name}: "
        f"N={n} empty={empty_count} eos_literal_total={eos_literal_count} "
        f"chars({_safe_stats(lengths_chars)}) "
        f"words({_safe_stats(lengths_words)})",
        flush=True,
    )

    for i, t in enumerate(texts[:max_preview]):
        preview = t[:500].replace("\n", "\\n")
        print(
            f"[eval external_ppl][debug] {name} preview[{i}] = {preview}",
            flush=True,
        )


def _debug_metric_dict(name: str, d: Dict[str, float]) -> None:
    if d is None:
        print(f"[eval external_ppl][debug] {name}: None", flush=True)
        return
    items = ", ".join(f"{k}={v}" for k, v in d.items())
    print(f"[eval external_ppl][debug] {name}: {items}", flush=True)


def _maybe_debug_owt_gpt2id_bpe16_eval(cfg, ext_cfg, texts, *, checkpoint_name: str, tag: str, rank0: bool) -> None:
    """
    Optional debugging hook for the double-tokenized OWT setup.

    It prints a decode dump for the generated code-token sequences stored in the
    shared cache object, but only when:
      - dataset == OpenWebText
      - sequence_codec == gpt2id_bpe16
      - evaluation.external_ppl.debug_owt_gpt2id_bpe16_decode == True
    """
    if not rank0:
        return

    ds_name = str(getattr(cfg.data, "dataset", "")).strip().lower()
    seq_codec = str(getattr(cfg.data, "sequence_codec", "base")).strip().lower()

    if ds_name != "openwebtext":
        return
    if seq_codec != "gpt2id_bpe16":
        return

    debug_enabled = bool(getattr(ext_cfg, "debug_owt_gpt2id_bpe16_decode", False))
    if not debug_enabled:
        return

    debug_once = bool(getattr(ext_cfg, "debug_owt_gpt2id_bpe16_once", True))
    max_rows = int(getattr(ext_cfg, "debug_owt_gpt2id_bpe16_max_rows", 8))

    latch_name = "_eval_owt_gpt2id_bpe16_debug_printed"
    if debug_once and bool(getattr(cfg.evaluation, latch_name, False)):
        return

    gen_tokens = getattr(texts, "gen_tokens", None)
    if gen_tokens is None:
        print(
            f"[eval external_ppl][debug][{checkpoint_name}/{tag}] "
            "shared cache object has no gen_tokens; skipping decode dump.",
            flush=True,
        )
        return

    root = Path(getattr(cfg.data, "root", "./datasets/openwebtext"))
    tokenizer_name = str(getattr(cfg.data, "tokenizer_name", "gpt2"))
    code_tokenizer_path = str(getattr(cfg.data, "code_tokenizer_path"))
    code_tokenizer_meta_path = getattr(cfg.data, "code_tokenizer_meta_path", None)

    gpt2_tok, code_tok, code_meta = load_openwebtext_gpt2id_bpe16_assets(
        root=root,
        code_tokenizer_path=code_tokenizer_path,
        code_tokenizer_meta_path=code_tokenizer_meta_path,
        tokenizer_name=tokenizer_name,
    )

    dbg_text = debug_gpt2id_bpe16_decode_batch(
        gen_tokens[:max_rows],
        gpt2_tokenizer=gpt2_tok,
        code_tokenizer=code_tok,
        code_meta=code_meta,
        max_rows=max_rows,
    )

    print(
        f"\n[eval external_ppl][debug][{checkpoint_name}/{tag}] gpt2id_bpe16 decode dump\n{dbg_text}\n",
        flush=True,
    )

    setattr(cfg.evaluation, latch_name, True)


def _paths_point_to_same_checkpoint(a: Optional[Path], b: Optional[Path]) -> bool:
    if a is None or b is None:
        return False
    try:
        return a.expanduser().resolve() == b.expanduser().resolve()
    except Exception:
        return a.name == b.name


def evaluate_external_ppl(
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
            print("\n⚠️  Skipping external PPL for non-text dataset.")
        return

    ext_cfg = getattr(getattr(cfg, "evaluation", object()), "external_ppl", None)
    if ext_cfg is None:
        raise ValueError("Missing cfg.evaluation.external_ppl")

    backend = str(getattr(ext_cfg, "backend", "hf_causal_lm")).lower()
    if backend != "hf_causal_lm":
        raise ValueError("This implementation expects cfg.evaluation.external_ppl.backend='hf_causal_lm'")

    model_name = str(getattr(ext_cfg, "hf_model_name", "openai-community/gpt2-large"))
    revision = getattr(ext_cfg, "hf_revision", None)
    hf_dtype = str(getattr(ext_cfg, "hf_dtype", "bfloat16"))
    attn_impl = getattr(ext_cfg, "attn_implementation", "sdpa")

    n_samples = int(args.external_ppl_num_samples or getattr(ext_cfg, "num_samples", 4096))
    micro_bs = int(args.external_ppl_micro_batch_size or getattr(ext_cfg, "micro_batch_size", 512))
    samplers = list(
        args.external_ppl_samplers
        or getattr(
            ext_cfg,
            "samplers",
            ["heun_karras" if cfg.framework == "continuous_score" else "tweedie"],
        )
    )
    terminal_sigmas = list(args.external_ppl_terminal_sigmas or getattr(ext_cfg, "terminal_sigmas", [0.08]))
    guidance_scales = list(args.external_ppl_guidance_scales or getattr(ext_cfg, "guidance_scales", [0.0]))
    num_steps = int(
        args.external_ppl_steps
        or getattr(ext_cfg, "num_sampling_steps", getattr(cfg.evaluation, "num_sampling_steps", 128))
    )
    sigma_max = getattr(ext_cfg, "sigma_max", None)
    sigma_max = None if sigma_max is None else float(sigma_max)
    seed = int(getattr(ext_cfg, "seed", 42))
    score_mode = str(getattr(ext_cfg, "score_mode", "completion_only_conditional")).lower()
    compute_real = bool(getattr(ext_cfg, "compute_real_reference", True))
    default_sc_refresh_mode = str(getattr(ext_cfg, "sc_refresh_mode", "refined")).lower()

    if rank0:
        print("\n── Calculating external Gen-PPL (shared generations) ──")
        print(f"   Evaluator:   {model_name}")
        print(f"   Score mode:  {score_mode}")
        print(f"   N samples:   {n_samples}")
        print(f"   SC mode:     {default_sc_refresh_mode}")

    ckpt_spec = args.external_ppl_checkpoints or getattr(ext_cfg, "checkpoints", None)
    base_ckpt_dir = Path(cfg.evaluation.checkpoint_path).parent
    checkpoints_to_run = resolve_checkpoints(ckpt_spec, base_ckpt_dir)
    if ckpt_spec is None and len(checkpoints_to_run) == 0:
        checkpoints_to_run = [None]

    initial_ckpt_path = Path(cfg.evaluation.checkpoint_path).expanduser()
    initial_ckpt_name = initial_ckpt_path.name

    # IMPORTANT:
    # run_eval.py has already loaded cfg.evaluation.checkpoint_path before this
    # function is called. During a checkpoint sweep, however, we may subsequently
    # load 100K, 200K, ..., 800K. Therefore, when we later reach the original
    # checkpoint again, e.g. 900K, it is NOT necessarily still loaded.
    #
    # The old code compared every requested checkpoint against initial_ckpt_path.
    # That incorrectly skipped reloading the final checkpoint when it matched the
    # initial path, causing the final checkpoint to reuse the previously loaded
    # weights. We instead track the currently loaded checkpoint and update it
    # after every successful load_checkpoint call.
    try:
        current_loaded_ckpt_path = initial_ckpt_path.resolve()
    except Exception:
        current_loaded_ckpt_path = initial_ckpt_path

    cache_root = _shared_cache_root(cfg, ext_cfg)
    shared_cache = SharedGenerationCache(cfg, cache_root)

    if cfg.framework == "continuous_score":
        proc = ContinuousForwardProcess(cfg)
    elif cfg.framework == "discrete_sedd":
        proc = DiscreteForwardProcess(cfg)
    else:
        raise ValueError(f"Unknown framework: {cfg.framework}")

    # Important: initialize lazily on rank 0 only, after distributed generation
    # for the first spec has completed. This avoids rank asymmetry during DDP generation.
    ppl_evaluator = None

    sampling_specs = build_sampling_specs(cfg=cfg, metric_cfg=ext_cfg)

    if sampling_specs is None:
        tag_specs = []
        if str(cfg.framework).lower().startswith("discrete"):
            for sampler_name in samplers:
                sc_refresh_mode = "refined"
                tag_specs.append(
                    dict(
                        tag=str(sampler_name),
                        sampler_name=str(sampler_name),
                        terminal_sigma=0.0,
                        guidance_scale=0.0,
                        num_steps=num_steps,
                        sc_refresh_mode=sc_refresh_mode,
                        target_nfe=None,
                        actual_nfe=compute_nfe(
                            framework=str(cfg.framework),
                            sampler_name=str(sampler_name),
                            num_steps=num_steps,
                            self_condition=bool(getattr(cfg.model, "self_condition", False)),
                            sc_refresh_mode=sc_refresh_mode,
                            return_probs=True,
                            discrete_denoise=True,
                        ),
                    )
                )
        else:
            for sampler_name in samplers:
                for sigma in terminal_sigmas:
                    for gs in guidance_scales:
                        sc_refresh_mode = default_sc_refresh_mode
                        actual_nfe = compute_nfe(
                            framework=str(cfg.framework),
                            sampler_name=str(sampler_name),
                            num_steps=num_steps,
                            self_condition=bool(getattr(cfg.model, "self_condition", False)),
                            sc_refresh_mode=sc_refresh_mode,
                            return_probs=True,
                            discrete_denoise=True,
                        )
                        tag_specs.append(
                            dict(
                                tag=(
                                    f"{sampler_name}"
                                    f"_scr-{sc_refresh_mode}"
                                    f"_term{math.log10(float(sigma)):.2f}"
                                    f"_gs{float(gs):.1f}"
                                ),
                                sampler_name=str(sampler_name),
                                terminal_sigma=float(sigma),
                                guidance_scale=float(gs),
                                num_steps=num_steps,
                                sc_refresh_mode=sc_refresh_mode,
                                target_nfe=None,
                                actual_nfe=actual_nfe,
                                ati_eta=0.0,
                            )
                        )
    else:
        tag_specs = sampling_specs

    for ckpt_path in checkpoints_to_run:
        curr_run_meta = copy.deepcopy(run_meta)

        if ckpt_path is None:
            ckpt_name = initial_ckpt_name
            curr_run_meta["checkpoint"] = ckpt_name
            if rank0:
                print(f"\n   -> Checkpoint: {ckpt_name} (already loaded)")

        else:
            ckpt_name = ckpt_path.name
            curr_run_meta["checkpoint"] = ckpt_name

            try:
                requested_ckpt_path = ckpt_path.expanduser().resolve()
            except Exception:
                requested_ckpt_path = ckpt_path.expanduser()

            already_loaded = _paths_point_to_same_checkpoint(
                requested_ckpt_path,
                current_loaded_ckpt_path,
            )

            if already_loaded:
                if rank0:
                    print(f"\n   -> Checkpoint: {ckpt_name} (already loaded)")
            else:
                if rank0:
                    print(f"\n   -> Checkpoint: {ckpt_name}")

                load_checkpoint(model, ema, ckpt_path, device, apply_ema=use_ema)
                model.eval()

                current_loaded_ckpt_path = requested_ckpt_path

        # Intentionally no barrier here.
        # Distributed synchronization already happens naturally inside generation
        # and again at the end of each spec.

        for spec in tag_specs:
            tag = spec["tag"]
            sampler_name = str(spec["sampler_name"])
            spec_steps = int(spec["num_steps"])
            terminal_sigma = float(spec["terminal_sigma"])
            guidance_scale = float(spec["guidance_scale"])
            sc_refresh_mode = str(spec.get("sc_refresh_mode", default_sc_refresh_mode))
            target_nfe = spec.get("target_nfe", None)
            actual_nfe = int(spec["actual_nfe"])
            ati_eta = float(spec.get("ati_eta", 0.0))

            stochastic_fields = _resolve_effective_stochastic_fields(cfg, spec)

            cache_key = shared_cache.make_cache_key(
                checkpoint_name=ckpt_name,
                split="test",
                num_samples=n_samples,
                num_steps=spec_steps,
                micro_batch_size=micro_bs,
                sigma_max=sigma_max,
                use_ema=use_ema,
                seed=seed,
                sampler_name=sampler_name,
                terminal_sigma=terminal_sigma,
                guidance_scale=guidance_scale,
                sc_refresh_mode=sc_refresh_mode,
                ati_eta=ati_eta,
                **stochastic_fields,
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
                    ati_eta=ati_eta,
                    **stochastic_fields,
                ),
                sampling_specs=[spec],
            )

            if rank0:
                if ppl_evaluator is None:
                    ppl_evaluator = HFExternalPerplexityEvaluator(
                        model_name=model_name,
                        revision=revision,
                        device=device,
                        torch_dtype=hf_dtype,
                        attn_implementation=attn_impl,
                        use_amp=True,
                    )

                _maybe_debug_owt_gpt2id_bpe16_eval(
                    cfg,
                    ext_cfg,
                    texts,
                    checkpoint_name=ckpt_name,
                    tag=tag,
                    rank0=rank0,
                )

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
                    sigma_max=sigma_max,
                    use_ema=bool(use_ema),
                    nfe=actual_nfe,
                    target_nfe=target_nfe,
                    sc_refresh_mode=sc_refresh_mode,
                    ati_eta=ati_eta,
                    external_lm_name=model_name,
                    **stochastic_fields,
                    **gen_meta_fields,
                )

                rows_out: List[Dict[str, Any]] = []
                real_scores = None

                if score_mode == "completion_only_conditional":
                    _debug_text_batch_stats("gen_prompt", texts.prompt)
                    _debug_text_batch_stats("gen_suffix", texts.suffix)
                    if compute_real:
                        _debug_text_batch_stats("real_prompt", texts.ref_prompt)
                        _debug_text_batch_stats("real_suffix", texts.ref_suffix)

                    gen_scores = ppl_evaluator.score_prompt_completion_pairs(texts.prompt, texts.suffix)
                    _debug_metric_dict("gen_scores", gen_scores)

                    if compute_real:
                        real_scores = ppl_evaluator.score_prompt_completion_pairs(texts.ref_prompt, texts.ref_suffix)
                        _debug_metric_dict("real_scores", real_scores)

                    view = "suffix"

                elif score_mode == "full":
                    _debug_text_batch_stats("gen_full", texts.full)
                    if compute_real:
                        _debug_text_batch_stats("real_full", texts.ref_full)

                    gen_scores = ppl_evaluator.score_texts(texts.full)
                    _debug_metric_dict("gen_scores", gen_scores)

                    if compute_real:
                        real_scores = ppl_evaluator.score_texts(texts.ref_full)
                        _debug_metric_dict("real_scores", real_scores)

                    view = "full"

                else:
                    raise ValueError(f"Unknown external-PPL score_mode: {score_mode}")

                _append_result_rows(
                    rows_out=rows_out,
                    run_meta=curr_run_meta,
                    base_fields=base_fields,
                    metric_dict=gen_scores,
                    metric_prefix=f"gen_{view}",
                    details=(
                        f"view={view},tag={tag},N={n_samples},steps={spec_steps},"
                        f"nfe={actual_nfe},lm={model_name},sc_refresh_mode={sc_refresh_mode}"
                    ),
                )

                if compute_real and real_scores is not None:
                    real_base_fields: Dict[str, Any] = dict(base_fields)
                    real_base_fields["tag"] = "REAL"
                    _append_result_rows(
                        rows_out=rows_out,
                        run_meta=curr_run_meta,
                        base_fields=real_base_fields,
                        metric_dict=real_scores,
                        metric_prefix=f"real_{view}",
                        details=(
                            f"view={view},tag=REAL,N={n_samples},steps={spec_steps},"
                            f"nfe={actual_nfe},lm={model_name},sc_refresh_mode={sc_refresh_mode}"
                        ),
                    )

                results_rows.extend(rows_out)
                _append_results_online(cfg, rows_out)

                print(
                    f"      External scores [{tag} | {view}]: "
                    f"external_bpt={gen_scores.get('external_bpt', None)} "
                    f"external_ppl={gen_scores.get('external_ppl', None)}",
                    flush=True,
                )

            # Keep all ranks aligned before moving to the next spec.
            ddp_barrier()

        # Optional end-of-checkpoint sync; harmless and keeps phases aligned
        # if multiple checkpoints are evaluated.
        ddp_barrier()