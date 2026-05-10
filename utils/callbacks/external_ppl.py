from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import torch

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None

from evaluation.generation_driver import GenerationDriver, GenerationBatch
from evaluation.external_perplexity import HFExternalPerplexityEvaluator
from utils.text_decode import (
    decode_bitstreams_for_eval,
    decode_token_sequences_for_eval,
    debug_gpt2id_bpe16_decode_batch,
    bitstreams_to_token_ids_raw_binary,
    extract_dataset_attr,
)
from utils.model_utils import unwrap_model


# -----------------------------------------------------------------------------
# DDP helpers
# -----------------------------------------------------------------------------
def _ddp_is_on() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return int(dist.get_rank()) if _ddp_is_on() else 0


def _world() -> int:
    return int(dist.get_world_size()) if _ddp_is_on() else 1


def _rank0() -> bool:
    return (not _ddp_is_on()) or (_rank() == 0)


def _dbg(msg: str, *, all_ranks: bool = False) -> None:
    if all_ranks:
        print(f"[ExternalPPL][rank{_rank()}/{_world()}] {msg}", flush=True)
    elif _rank0():
        print(f"[ExternalPPL][rank{_rank()}/{_world()}] {msg}", flush=True)


def _tb(trainer) -> Optional[Any]:
    tb = getattr(trainer, "tb", None)
    if tb is not None:
        return tb
    tb = getattr(trainer, "tb_manager", None)
    if tb is not None:
        return tb
    return None


def _global_step(trainer, epoch: int) -> int:
    gs = getattr(trainer, "global_step", None)
    if gs is None:
        return int(epoch)
    try:
        return int(gs)
    except Exception:
        return int(epoch)


def _get_dataset_obj_from_loader(loader) -> Optional[Any]:
    return getattr(loader, "dataset", None)


def _normalize_text8_flag(cfg: Any) -> bool:
    ds = str(getattr(getattr(cfg, "data", None), "dataset", "")).strip().lower()
    ds = ds.replace("_", "").replace("-", "").replace(" ", "")
    return ds == "text8"


# -----------------------------------------------------------------------------
# Config resolution
# -----------------------------------------------------------------------------
@dataclass
class _ResolvedExternalPPL:
    enabled: bool
    every_k_epochs: int
    run_on_sanity: bool
    splits: List[str]

    # generation
    num_samples: int
    sampler: str
    terminal_sigma: float
    guidance_scale: float
    num_steps: int
    entropic_blend_alpha: float
    entropy_run_dir: Optional[str]
    seed: int
    micro_batch_size: Optional[int]
    sigma_max: Optional[float]
    sc_refresh_mode: str
    decode_mode: str
    score_mode: str

    # evaluator behaviour
    compute_real_reference: bool

    # HF evaluator
    backend: str
    hf_model_name: str
    hf_revision: Optional[str]
    hf_dtype: str
    attn_implementation: Optional[str]
    device: Optional[str]
    use_amp: bool

    # logging
    log_prefix: str


def _coerce_splits(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return [str(s) for s in list(x)]


def _first_scalar(x, default=None):
    if x is None:
        return default
    if isinstance(x, (list, tuple)):
        return x[0] if len(x) > 0 else default
    return x


def _sanitize_sampler_name(x, default: str = "heun_karras") -> str:
    if x is None:
        return default

    if isinstance(x, (list, tuple)):
        x = x[0] if len(x) > 0 else default

    s = str(x).strip()

    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if "," in inner:
            inner = inner.split(",", 1)[0].strip()
        inner = inner.strip().strip("'").strip('"')
        if inner:
            s = inner

    return s or default


def _resolve_ext_block(cfg: Any):
    train = getattr(cfg, "train", None)
    ext = getattr(train, "external_perplexity", None) if train is not None else None
    if ext is None and train is not None:
        ext = getattr(train, "external_ppl", None)
    return ext


def _resolve_generation_defaults(cfg: Any):
    train = getattr(cfg, "train", None)
    return getattr(train, "generation", None) if train is not None else None


def _resolve_cfg(cfg: Any) -> _ResolvedExternalPPL:
    ext = _resolve_ext_block(cfg)
    gen = _resolve_generation_defaults(cfg)

    enabled = bool(getattr(ext, "enabled", False)) if ext is not None else False

    run_on_sanity = False
    if ext is not None and getattr(ext, "run_on_sanity", None) is not None:
        run_on_sanity = bool(getattr(ext, "run_on_sanity"))

    k = None
    if ext is not None:
        k = getattr(ext, "every_k_epochs", None)
        if k is None:
            k = getattr(ext, "every_epochs", None)
        if k is None:
            k = getattr(ext, "every", None)
    if k is None:
        k = 10 if enabled else 1
    every_k_epochs = max(1, int(k))

    splits: List[str] = []
    if ext is not None:
        splits = _coerce_splits(getattr(ext, "splits", None))
        if not splits:
            s = getattr(ext, "split", None)
            if s is not None:
                splits = [str(s)]
    if not splits:
        splits = ["val"]
    splits = [str(s).lower() for s in splits]

    def pick(name: str, gen_name: str, default):
        if ext is not None and getattr(ext, name, None) is not None:
            return getattr(ext, name)
        if gen is not None and getattr(gen, gen_name, None) is not None:
            return getattr(gen, gen_name)
        return default

    num_samples = int(pick("num_samples", "num_samples", 1024))

    raw_sampler = None
    if ext is not None:
        raw_sampler = getattr(ext, "sampler", None)
        if raw_sampler is None:
            raw_sampler = getattr(ext, "samplers", None)
    if raw_sampler is None and gen is not None:
        raw_sampler = getattr(gen, "sampler", None)
        if raw_sampler is None:
            raw_sampler = getattr(gen, "samplers", None)
    sampler = _sanitize_sampler_name(raw_sampler, default="heun_karras")

    terminal_sigma = getattr(ext, "terminal_sigma", None) if ext is not None else None
    if terminal_sigma is None and ext is not None:
        terminal_sigma = getattr(ext, "terminal_sigmas", None)
    if terminal_sigma is None and gen is not None:
        terminal_sigma = getattr(gen, "terminal_sigmas", None)
    terminal_sigma = float(_first_scalar(terminal_sigma, 0.08))

    guidance_scale = getattr(ext, "guidance_scale", None) if ext is not None else None
    if guidance_scale is None and ext is not None:
        guidance_scale = getattr(ext, "guidance_scales", None)
    if guidance_scale is None and gen is not None:
        guidance_scale = getattr(gen, "guidance_scales", None)
    guidance_scale = float(_first_scalar(guidance_scale, 0.0))

    num_steps = int(
        (getattr(ext, "num_steps", None) if ext is not None else None)
        or (getattr(ext, "num_sampling_steps", None) if ext is not None else None)
        or (getattr(gen, "num_sampling_steps", None) if gen is not None else None)
        or 256
    )

    entropic_blend_alpha = float(pick("entropic_blend_alpha", "entropic_blend_alpha", 0.0))

    entropy_run_dir = None
    if ext is not None:
        entropy_run_dir = getattr(ext, "entropy_run_dir", None)
        if entropy_run_dir is None:
            entropy_run_dir = getattr(ext, "entropy_ckpt_path", None)
    if entropy_run_dir is None and gen is not None:
        entropy_run_dir = getattr(gen, "entropy_ckpt_path", None)

    seed = int(pick("seed", "seed", 42))

    micro_batch_size = None
    if ext is not None:
        micro_batch_size = getattr(ext, "micro_batch_size", None)
    if micro_batch_size is None and gen is not None:
        micro_batch_size = getattr(gen, "micro_batch_size", None)
    micro_batch_size = int(micro_batch_size) if micro_batch_size is not None else None

    sigma_max = None
    if ext is not None and getattr(ext, "sigma_max", None) is not None:
        sigma_max = float(getattr(ext, "sigma_max"))
    elif gen is not None and getattr(gen, "sigma_max", None) is not None:
        sigma_max = float(getattr(gen, "sigma_max"))

    sc_refresh_mode = "refined"
    if ext is not None and getattr(ext, "sc_refresh_mode", None) is not None:
        sc_refresh_mode = str(getattr(ext, "sc_refresh_mode")).lower()
    elif gen is not None and getattr(gen, "sc_refresh_mode", None) is not None:
        sc_refresh_mode = str(getattr(gen, "sc_refresh_mode")).lower()

    decode_mode = "suffix"
    if ext is not None and getattr(ext, "decode_mode", None) is not None:
        decode_mode = str(getattr(ext, "decode_mode"))
    decode_mode = decode_mode.lower().strip()

    score_mode = "completion_only_conditional"
    if ext is not None and getattr(ext, "score_mode", None) is not None:
        score_mode = str(getattr(ext, "score_mode"))
    score_mode = score_mode.lower().strip()

    compute_real_reference = bool(getattr(ext, "compute_real_reference", False)) if ext is not None else False

    backend = "hf_causal_lm"
    if ext is not None and getattr(ext, "backend", None) is not None:
        backend = str(getattr(ext, "backend")).lower()

    hf_model_name = "openai-community/gpt2-large"
    if ext is not None and getattr(ext, "hf_model_name", None) is not None:
        hf_model_name = str(getattr(ext, "hf_model_name"))

    hf_revision = getattr(ext, "hf_revision", None) if ext is not None else None
    hf_dtype = str(getattr(ext, "hf_dtype", "bfloat16")) if ext is not None else "bfloat16"
    attn_implementation = getattr(ext, "attn_implementation", "sdpa") if ext is not None else "sdpa"

    device = getattr(ext, "device", None) if ext is not None else None
    if device is not None:
        device = str(device)

    use_amp = bool(getattr(ext, "use_amp", True)) if ext is not None else True

    log_prefix = "external_perplexity"
    if ext is not None and getattr(ext, "log_prefix", None) is not None:
        log_prefix = str(getattr(ext, "log_prefix"))

    return _ResolvedExternalPPL(
        enabled=enabled,
        every_k_epochs=every_k_epochs,
        run_on_sanity=run_on_sanity,
        splits=splits,
        num_samples=num_samples,
        sampler=sampler,
        terminal_sigma=terminal_sigma,
        guidance_scale=guidance_scale,
        num_steps=num_steps,
        entropic_blend_alpha=entropic_blend_alpha,
        entropy_run_dir=entropy_run_dir,
        seed=seed,
        micro_batch_size=micro_batch_size,
        sigma_max=sigma_max,
        sc_refresh_mode=sc_refresh_mode,
        decode_mode=decode_mode,
        score_mode=score_mode,
        compute_real_reference=compute_real_reference,
        backend=backend,
        hf_model_name=hf_model_name,
        hf_revision=hf_revision,
        hf_dtype=hf_dtype,
        attn_implementation=attn_implementation,
        device=device,
        use_amp=use_amp,
        log_prefix=log_prefix,
    )


# -----------------------------------------------------------------------------
# Generation wrapper
# -----------------------------------------------------------------------------
def _driver_generate(
    driver: GenerationDriver,
    trainer: Any,
    loader: Any,
    *,
    num_samples: int,
    sampler: str,
    terminal_sigma: float,
    guidance_scale: float,
    num_steps: int,
    entropic_blend_alpha: float,
    entropy_run_dir: Optional[str],
    seed: int,
    micro_batch_size: Optional[int],
    sigma_max: Optional[float],
    sc_refresh_mode: str,
) -> Dict[str, GenerationBatch]:
    sigma_f = float(terminal_sigma)
    sigma_tag = f"{math.log10(sigma_f):.2f}" if sigma_f > 0.0 else "none"

    sampling_specs = [
        dict(
            tag=f"{str(sampler).lower()}_term{sigma_tag}_gs{float(guidance_scale):.1f}_scr-{str(sc_refresh_mode).lower()}",
            sampler_name=str(sampler).lower(),
            terminal_sigma=float(terminal_sigma),
            guidance_scale=float(guidance_scale),
            num_steps=int(num_steps),
            sc_refresh_mode=str(sc_refresh_mode).lower(),
        )
    ]

    eval_model = unwrap_model(trainer.model)
    eval_model.eval()

    return driver.generate_prompt_completion(
        model=eval_model,
        proc=trainer.proc,
        device=trainer.device,
        loader=loader,
        num_samples=int(num_samples),
        sampler_names=[str(sampler)],
        terminal_sigmas=[float(terminal_sigma)],
        guidance_scales=[float(guidance_scale)],
        num_steps=int(num_steps),
        sampling_specs=sampling_specs,
        entropic_blend_alpha=float(entropic_blend_alpha),
        entropy_run_dir=entropy_run_dir,
        seed=int(seed),
        use_amp=True,
        amp_dtype="auto",
        micro_batch_size=micro_batch_size,
        sigma_max=sigma_max,
        gather_to_rank0=True,
    )


# -----------------------------------------------------------------------------
# Callback
# -----------------------------------------------------------------------------
class ExternalPPLCallback:
    run_on_all_ranks = True

    def __init__(self, cfg_full: Any):
        self.cfg_full = cfg_full
        self._driver: Optional[GenerationDriver] = None
        self._evaluator: Optional[HFExternalPerplexityEvaluator] = None
        self._last_run_key: Optional[Tuple[int, int]] = None
        self._owt_gpt2id_bpe16_debug_printed = False

    def _ensure_driver(self, trainer) -> None:
        if self._driver is None:
            self._driver = GenerationDriver(self.cfg_full)

    def _ensure_evaluator(self, trainer, r: _ResolvedExternalPPL) -> None:
        if self._evaluator is not None:
            return

        if r.backend != "hf_causal_lm":
            raise ValueError(
                f"ExternalPPLCallback expects backend='hf_causal_lm', got backend='{r.backend}'"
            )

        dev = torch.device(str(r.device)) if r.device is not None else trainer.device

        self._evaluator = HFExternalPerplexityEvaluator(
            model_name=r.hf_model_name,
            revision=r.hf_revision,
            device=dev,
            torch_dtype=r.hf_dtype,
            attn_implementation=r.attn_implementation,
            use_amp=bool(r.use_amp),
        )

    def _should_run(self, epoch: int, r: _ResolvedExternalPPL) -> bool:
        if not r.enabled:
            return False
        if int(epoch) < 0:
            return bool(r.run_on_sanity)

        k = max(1, int(r.every_k_epochs))
        return ((int(epoch) + 1) % k) == 0

    def _log_scalars(self, trainer, scalars: Dict[str, float], step: int, tag: str, log_prefix: str) -> None:
        tb = _tb(trainer)
        if tb is None:
            return
        for k, v in scalars.items():
            name = f"{log_prefix}/{tag}/{k}"
            if hasattr(tb, "add_scalar"):
                try:
                    tb.add_scalar(name, float(v), step)
                except TypeError:
                    tb.add_scalar(name, float(v))
            elif hasattr(tb, "writer") and hasattr(tb.writer, "add_scalar"):
                tb.writer.add_scalar(name, float(v), step)

    def _resolve_loader(self, trainer, split_name: str):
        split_name = str(split_name).lower()
        if split_name in {"val", "valid", "validation"}:
            return getattr(trainer, "val_loader", None), "val"
        if split_name == "test":
            return getattr(trainer, "test_loader", None), "test"
        if split_name == "train":
            return getattr(trainer, "train_loader", None), "train"
        return None, split_name

    def _maybe_debug_owt_gpt2id_bpe16_decode(
        self,
        *,
        trainer,
        batch: GenerationBatch,
        ds_obj: Optional[Any],
        split_norm: str,
        tag: str,
    ) -> None:
        cfg = self.cfg_full

        ds_name = str(getattr(cfg.data, "dataset", "")).strip().lower()
        seq_codec = str(getattr(cfg.data, "sequence_codec", "base")).strip().lower()

        if ds_name != "openwebtext":
            return
        if seq_codec != "gpt2id_bpe16":
            return

        ext = _resolve_ext_block(cfg)
        debug_enabled = bool(
            getattr(ext, "debug_owt_gpt2id_bpe16_decode", False)
        ) if ext is not None else False
        if not debug_enabled:
            return

        debug_once = bool(
            getattr(ext, "debug_owt_gpt2id_bpe16_once", True)
        ) if ext is not None else True

        max_rows = int(
            getattr(ext, "debug_owt_gpt2id_bpe16_max_rows", 8)
        ) if ext is not None else 8

        if debug_once and self._owt_gpt2id_bpe16_debug_printed:
            return

        if ds_obj is None:
            print(
                f"[ExternalPPL][debug][{split_norm}/{tag}] ds_obj is None, skipping decode debug.",
                flush=True,
            )
            return

        gpt2_tok = extract_dataset_attr(ds_obj, "tokenizer")
        code_tok = extract_dataset_attr(ds_obj, "code_tokenizer")
        code_meta = extract_dataset_attr(ds_obj, "code_meta")

        if gpt2_tok is None or code_tok is None or code_meta is None:
            print(
                f"[ExternalPPL][debug][{split_norm}/{tag}] missing tokenizer/code_meta "
                f"(gpt2_tok={gpt2_tok is not None}, code_tok={code_tok is not None}, code_meta={code_meta is not None}), "
                "skipping decode debug.",
                flush=True,
            )
            return

        if batch.gen_tokens is not None:
            code_ids = batch.gen_tokens[:max_rows]
        elif batch.gen_bits is not None:
            code_ids = bitstreams_to_token_ids_raw_binary(
                batch.gen_bits[:max_rows],
                bits_per_token=int(getattr(cfg.data, "bits_per_token", 16)),
                cfg=cfg,
            )
        else:
            print(
                f"[ExternalPPL][debug][{split_norm}/{tag}] no gen_tokens/gen_bits found.",
                flush=True,
            )
            return

        dbg_text = debug_gpt2id_bpe16_decode_batch(
            code_ids,
            gpt2_tokenizer=gpt2_tok,
            code_tokenizer=code_tok,
            code_meta=code_meta,
            max_rows=max_rows,
        )

        print(
            f"\n[ExternalPPL][debug][{split_norm}/{tag}] gpt2id_bpe16 decode dump\n{dbg_text}\n",
            flush=True,
        )

        self._owt_gpt2id_bpe16_debug_printed = True

    def _decode_for_score_mode(
        self,
        *,
        trainer,
        batch: GenerationBatch,
        ds_obj: Optional[Any],
        score_mode: str,
        decode_mode: str,
    ):
        has_bits = batch.gen_bits is not None and batch.ref_bits is not None
        has_tokens = batch.gen_tokens is not None and batch.ref_tokens is not None

        if not has_bits and not has_tokens:
            return None

        normalize_text8 = _normalize_text8_flag(trainer.cfg)

        if score_mode == "full":
            if has_bits:
                gen_full = decode_bitstreams_for_eval(
                    self.cfg_full,
                    batch.gen_bits,
                    mode="full",
                    dataset_obj=ds_obj,
                    normalize_text8=normalize_text8,
                )
                ref_full = None
                if batch.ref_bits is not None:
                    ref_full = decode_bitstreams_for_eval(
                        self.cfg_full,
                        batch.ref_bits,
                        mode="full",
                        dataset_obj=ds_obj,
                        normalize_text8=normalize_text8,
                    )
                return {
                    "gen_full": gen_full,
                    "ref_full": ref_full,
                }

            gen_full = decode_token_sequences_for_eval(
                self.cfg_full,
                batch.gen_tokens,
                mode="full",
                dataset_obj=ds_obj,
                normalize_text8=normalize_text8,
            )
            ref_full = None
            if batch.ref_tokens is not None:
                ref_full = decode_token_sequences_for_eval(
                    self.cfg_full,
                    batch.ref_tokens,
                    mode="full",
                    dataset_obj=ds_obj,
                    normalize_text8=normalize_text8,
                )
            return {
                "gen_full": gen_full,
                "ref_full": ref_full,
            }

        if score_mode == "completion_only_conditional":
            if has_bits:
                if batch.prompt_len_bits is None:
                    raise RuntimeError("Bitstream conditional ExternalPPL requires prompt_len_bits.")

                prompt_texts = decode_bitstreams_for_eval(
                    self.cfg_full,
                    batch.ref_bits,
                    prompt_len_bits=batch.prompt_len_bits,
                    mode="prompt",
                    dataset_obj=ds_obj,
                    normalize_text8=normalize_text8,
                )

                gen_suffix = decode_bitstreams_for_eval(
                    self.cfg_full,
                    batch.gen_bits,
                    prompt_len_bits=batch.prompt_len_bits,
                    mode="suffix",
                    dataset_obj=ds_obj,
                    normalize_text8=normalize_text8,
                )

                ref_suffix = None
                if batch.ref_bits is not None:
                    ref_suffix = decode_bitstreams_for_eval(
                        self.cfg_full,
                        batch.ref_bits,
                        prompt_len_bits=batch.prompt_len_bits,
                        mode="suffix",
                        dataset_obj=ds_obj,
                        normalize_text8=normalize_text8,
                    )

                gen_for_display = gen_suffix
                if decode_mode == "full":
                    gen_for_display = decode_bitstreams_for_eval(
                        self.cfg_full,
                        batch.gen_bits,
                        mode="full",
                        dataset_obj=ds_obj,
                        normalize_text8=normalize_text8,
                    )

                return {
                    "prompt": prompt_texts,
                    "gen_suffix": gen_suffix,
                    "ref_suffix": ref_suffix,
                    "gen_display": gen_for_display,
                }

            if batch.prompt_len_tokens is None:
                raise RuntimeError("Token conditional ExternalPPL requires prompt_len_tokens.")

            prompt_texts = decode_token_sequences_for_eval(
                self.cfg_full,
                batch.ref_tokens,
                prompt_len_tokens=batch.prompt_len_tokens,
                mode="prompt",
                dataset_obj=ds_obj,
                normalize_text8=normalize_text8,
            )

            gen_suffix = decode_token_sequences_for_eval(
                self.cfg_full,
                batch.gen_tokens,
                prompt_len_tokens=batch.prompt_len_tokens,
                mode="suffix",
                dataset_obj=ds_obj,
                normalize_text8=normalize_text8,
            )

            ref_suffix = None
            if batch.ref_tokens is not None:
                ref_suffix = decode_token_sequences_for_eval(
                    self.cfg_full,
                    batch.ref_tokens,
                    prompt_len_tokens=batch.prompt_len_tokens,
                    mode="suffix",
                    dataset_obj=ds_obj,
                    normalize_text8=normalize_text8,
                )

            gen_for_display = gen_suffix
            if decode_mode == "full":
                gen_for_display = decode_token_sequences_for_eval(
                    self.cfg_full,
                    batch.gen_tokens,
                    mode="full",
                    dataset_obj=ds_obj,
                    normalize_text8=normalize_text8,
                )

            return {
                "prompt": prompt_texts,
                "gen_suffix": gen_suffix,
                "ref_suffix": ref_suffix,
                "gen_display": gen_for_display,
            }

        raise ValueError(
            f"ExternalPPLCallback: score_mode must be 'full' or 'completion_only_conditional' "
            f"(got {score_mode})"
        )

    @torch.no_grad()
    def _run(self, trainer, epoch: int) -> None:
        r = _resolve_cfg(self.cfg_full)

        if not self._should_run(epoch, r):
            return

        run_key = (int(epoch), int(_global_step(trainer, epoch)))
        if self._last_run_key == run_key:
            return
        self._last_run_key = run_key

        self._ensure_driver(trainer)

        _dbg(
            f"START epoch={epoch} splits={r.splits} "
            f"num_samples={r.num_samples} sampler={r.sampler} steps={r.num_steps} "
            f"terminal_sigma={r.terminal_sigma} gs={r.guidance_scale} "
            f"sc_refresh={r.sc_refresh_mode} sigma_max={r.sigma_max} "
            f"micro_bs={r.micro_batch_size} score_mode={r.score_mode} "
            f"decode_mode={r.decode_mode}"
        )

        t0 = time.perf_counter()

        for split_name in r.splits:
            loader, split_norm = self._resolve_loader(trainer, split_name)
            if loader is None:
                _dbg(f"[{split_norm}] loader missing, skipping.")
                continue

            ds_obj = _get_dataset_obj_from_loader(loader)

            _dbg(f"[{split_norm}] entering generation.", all_ranks=True)
            batches = _driver_generate(
                self._driver,
                trainer,
                loader,
                num_samples=r.num_samples,
                sampler=r.sampler,
                terminal_sigma=r.terminal_sigma,
                guidance_scale=r.guidance_scale,
                num_steps=r.num_steps,
                entropic_blend_alpha=r.entropic_blend_alpha,
                entropy_run_dir=r.entropy_run_dir,
                seed=r.seed,
                micro_batch_size=r.micro_batch_size,
                sigma_max=r.sigma_max,
                sc_refresh_mode=r.sc_refresh_mode,
            )
            _dbg(f"[{split_norm}] exited generation.", all_ranks=True)

            if not _rank0():
                continue

            _dbg(f"[{split_norm}] generation done. tags={list(batches.keys())}")

            self._ensure_evaluator(trainer, r)
            step = _global_step(trainer, epoch)

            for tag, batch in batches.items():
                decoded = self._decode_for_score_mode(
                    trainer=trainer,
                    batch=batch,
                    ds_obj=ds_obj,
                    score_mode=r.score_mode,
                    decode_mode=r.decode_mode,
                )
                if decoded is None:
                    continue

                self._maybe_debug_owt_gpt2id_bpe16_decode(
                    trainer=trainer,
                    batch=batch,
                    ds_obj=ds_obj,
                    split_norm=split_norm,
                    tag=tag,
                )

                if r.score_mode == "full":
                    gen_full = decoded["gen_full"]
                    if not gen_full:
                        continue

                    metrics = self._evaluator.score_texts(gen_full)
                    self._log_scalars(
                        trainer,
                        metrics,
                        step,
                        f"{split_norm}/{tag}/gen",
                        r.log_prefix,
                    )

                    ref_full = decoded["ref_full"]
                    if r.compute_real_reference and ref_full:
                        real_metrics = self._evaluator.score_texts(ref_full)
                        self._log_scalars(
                            trainer,
                            real_metrics,
                            step,
                            f"{split_norm}/{tag}/real",
                            r.log_prefix,
                        )

                elif r.score_mode == "completion_only_conditional":
                    prompt_texts = decoded["prompt"]
                    gen_suffix = decoded["gen_suffix"]
                    if not prompt_texts or not gen_suffix:
                        continue

                    metrics = self._evaluator.score_prompt_completion_pairs(prompt_texts, gen_suffix)
                    self._log_scalars(
                        trainer,
                        metrics,
                        step,
                        f"{split_norm}/{tag}/gen",
                        r.log_prefix,
                    )

                    ref_suffix = decoded["ref_suffix"]
                    if r.compute_real_reference and ref_suffix:
                        real_metrics = self._evaluator.score_prompt_completion_pairs(prompt_texts, ref_suffix)
                        self._log_scalars(
                            trainer,
                            real_metrics,
                            step,
                            f"{split_norm}/{tag}/real",
                            r.log_prefix,
                        )

            _dbg(f"[{split_norm}] scoring done.")

        if _rank0():
            duration = time.perf_counter() - t0
            tb = _tb(trainer)
            if tb:
                name = f"{r.log_prefix}/timing_sec"
                if hasattr(tb, "add_scalar"):
                    tb.add_scalar(name, duration, _global_step(trainer, epoch))
                elif hasattr(tb, "writer"):
                    tb.writer.add_scalar(name, duration, _global_step(trainer, epoch))

        _dbg("END")

    def on_epoch_end(self, trainer, epoch: int) -> None:
        self._run(trainer, epoch)

    def on_train_epoch_end(self, trainer, epoch: int) -> None:
        self._run(trainer, epoch)