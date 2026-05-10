#evaluation/text_generations.py
from __future__ import annotations

import json
import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from evaluation.generation_driver import GenerationDriver, GenerationBatch
from utils.text_decode import (
    decode_bitstreams_for_eval,
    decode_token_sequences_for_eval,
    bitstreams_to_token_ids_raw_binary,
)


def _ddp_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank_world() -> tuple[int, int]:
    if not _ddp_is_on():
        return 0, 1
    return int(dist.get_rank()), int(dist.get_world_size())


def barrier() -> None:
    if not _ddp_is_on():
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()
        
def _sanitize_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(tag))


def _short_cache_key_dir(cache_key: str, *, prefix_chars: int = 64) -> str:
    """
    Convert an arbitrary long/human-readable cache key into a filesystem-safe,
    short directory component.

    The full cache key is still stored in _cache_key.json next to the cache file.
    """
    key = str(cache_key)
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()

    prefix = re.sub(r"[^A-Za-z0-9_.=-]+", "_", key)
    prefix = prefix[:prefix_chars].strip("._-")
    if not prefix:
        prefix = "cache"

    return f"{prefix}__h={digest}"


def _write_cache_key_manifest(cache_dir: Path, cache_key: str) -> None:
    """
    Store the full unhashed cache key for debugging/auditing.

    Only rank 0 writes in DDP to avoid races. Non-DDP also writes.
    This is best-effort; cache correctness does not depend on this file.
    """
    rank, _ = _rank_world()
    if rank != 0:
        return

    manifest = cache_dir / "_cache_key.json"
    if manifest.exists():
        return

    tmp = manifest.with_suffix(".json.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump({"cache_key": str(cache_key)}, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(manifest)
    except FileExistsError:
        pass
    except OSError:
        # Manifest is only for debugging. Do not fail generation because of it.
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _tensor_to_nested_list(x: Optional[torch.Tensor]) -> Optional[List[List[int]]]:
    if x is None:
        return None
    x = x.detach().to("cpu")
    if x.is_floating_point():
        x = (x > 0.5).to(torch.long)
    else:
        x = x.to(torch.long)
    return x.tolist()


@dataclass
class CachedTexts:
    prompt: List[str]
    suffix: List[str]
    full: List[str]
    ref_prompt: List[str]
    ref_suffix: List[str]
    ref_full: List[str]

    # raw payloads for exact posthoc reconstruction
    gen_bits: Optional[List[List[int]]]
    ref_bits: Optional[List[List[int]]]
    gen_tokens: Optional[List[List[int]]]
    ref_tokens: Optional[List[List[int]]]

    meta: Dict[str, Any]

    @staticmethod
    def empty() -> "CachedTexts":
        return CachedTexts([], [], [], [], [], [], None, None, None, None, {})


def parse_generation_tag(tag: str) -> dict:
    out = {"tag": str(tag)}
    s = str(tag)

    # New format:
    #   sampler_scr-carry_term-1.10_gs0.0
    # Old format:
    #   sampler_term-1.10_gs0.0
    sampler_name = s
    sc_refresh_mode = None
    terminal_sigma = None
    terminal_sigma_log10 = None
    guidance_scale = None

    if "_scr-" in s:
        sampler_name, rest = s.split("_scr-", 1)
        if "_term" in rest and "_gs" in rest:
            sc_refresh_mode, rest2 = rest.split("_term", 1)
            log10sigma_str, gs_str = rest2.split("_gs", 1)
            try:
                terminal_sigma_log10 = float(log10sigma_str)
                terminal_sigma = float(10 ** terminal_sigma_log10)
            except Exception:
                terminal_sigma = None
                terminal_sigma_log10 = None
            try:
                guidance_scale = float(gs_str)
            except Exception:
                guidance_scale = None
    elif "_term" in s and "_gs" in s:
        sampler_name = s.split("_term", 1)[0]
        rest = s.split("_term", 1)[1]
        log10sigma_str, gs_str = rest.split("_gs", 1)
        try:
            terminal_sigma_log10 = float(log10sigma_str)
            terminal_sigma = float(10 ** terminal_sigma_log10)
        except Exception:
            terminal_sigma = None
            terminal_sigma_log10 = None
        try:
            guidance_scale = float(gs_str)
        except Exception:
            guidance_scale = None

    out["sampler_name"] = sampler_name
    out["sc_refresh_mode"] = sc_refresh_mode
    out["terminal_sigma"] = terminal_sigma
    out["terminal_sigma_log10"] = terminal_sigma_log10
    out["guidance_scale"] = guidance_scale
    return out


def resolve_checkpoints(spec, base_ckpt_dir: Path) -> list[Path]:
    from glob import glob

    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return [str(v) for v in x]
        return [str(x)]

    def _is_glob_pattern(s: str) -> bool:
        return any(ch in s for ch in ["*", "?", "["])

    out = []
    for s in _as_list(spec):
        p = Path(s)
        if p.is_absolute():
            out.append(p)
        elif _is_glob_pattern(s):
            out.extend(Path(x) for x in glob(str(base_ckpt_dir / s)))
        else:
            out.append(base_ckpt_dir / p)

    out = [p for p in out if p.exists() and p.is_file()]
    uniq = {}
    for p in out:
        uniq[str(p.resolve())] = p
    return list(sorted(uniq.values(), key=lambda p: p.name))


class SharedGenerationCache:
    """
    Generates once, decodes once on rank 0, and stores prompt/suffix/full text views
    plus raw bit/token payloads so multiple metrics can reuse them.

    Important DDP safety properties:
      - rank 0 decides cache-hit / cache-miss and broadcasts that decision
      - tensor gathering happens inside GenerationDriver
      - only rank 0 decodes and writes the cache file
      - all ranks synchronize before advancing to the next spec
    """

    def __init__(self, cfg: Any, cache_root: Path):
        self.cfg = cfg
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.driver = GenerationDriver(cfg)

    @staticmethod
    def make_cache_key(
        *,
        checkpoint_name: str,
        split: str,
        num_samples: int,
        num_steps: int,
        micro_batch_size: int,
        sigma_max: Optional[float],
        use_ema: bool,
        seed: int,
        sampler_name: Optional[str] = None,
        terminal_sigma: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        sc_refresh_mode: Optional[str] = None,
        ati_eta: Optional[float] = None,
        stochastic_enabled: Optional[bool] = None,
        s_churn: Optional[float] = None,
        s_noise: Optional[float] = None,
        window_mode: Optional[str] = None,
        entropy_quantile_lo: Optional[float] = None,
        entropy_quantile_hi: Optional[float] = None,
        s_tmin: Optional[float] = None,
        s_tmax: Optional[float] = None,
        entropy_fallback: Optional[str] = None,
    ) -> str:
        def _fmt_float(x):
            return "none" if x is None else f"{float(x):.8g}"

        def _fmt_str(x):
            return "none" if x is None else str(x)

        sigma_max_s = _fmt_float(sigma_max)
        sampler_s = _fmt_str(sampler_name)
        term_s = _fmt_float(terminal_sigma)
        gs_s = _fmt_float(guidance_scale)
        sc_s = _fmt_str(sc_refresh_mode)
        ati_s = _fmt_float(ati_eta)

        st_enabled_s = "none" if stochastic_enabled is None else str(int(bool(stochastic_enabled)))
        s_churn_s = _fmt_float(s_churn)
        s_noise_s = _fmt_float(s_noise)
        window_mode_s = _fmt_str(window_mode)
        q_lo_s = _fmt_float(entropy_quantile_lo)
        q_hi_s = _fmt_float(entropy_quantile_hi)
        s_tmin_s = _fmt_float(s_tmin)
        s_tmax_s = _fmt_float(s_tmax)
        fallback_s = _fmt_str(entropy_fallback)

        return (
            f"ckpt={checkpoint_name}|split={split}|N={int(num_samples)}|"
            f"steps={int(num_steps)}|mb={int(micro_batch_size)}|"
            f"sigma_max={sigma_max_s}|ema={int(bool(use_ema))}|seed={int(seed)}|"
            f"sampler={sampler_s}|term={term_s}|gs={gs_s}|scr={sc_s}|ati={ati_s}|"
            f"stoch={st_enabled_s}|s_churn={s_churn_s}|s_noise={s_noise_s}|"
            f"window={window_mode_s}|qlo={q_lo_s}|qhi={q_hi_s}|"
            f"s_tmin={s_tmin_s}|s_tmax={s_tmax_s}|fallback={fallback_s}"
        )

    def _legacy_path_no_mkdir(
        self,
        *,
        checkpoint_name: str,
        split: str,
        cache_key: str,
        tag: str,
    ) -> Path:
        """
        Old cache layout:

            cache_root / checkpoint_name / split / full_cache_key / tag.jsonl

        We keep this only for backwards-compatible lookup. Do not mkdir this path,
        because long cache keys can raise ENAMETOOLONG.
        """
        return (
            self.cache_root
            / checkpoint_name
            / split
            / str(cache_key)
            / f"{_sanitize_tag(tag)}.jsonl"
        )


    def _path(self, *, checkpoint_name: str, split: str, cache_key: str, tag: str) -> Path:
        """
        Filesystem-safe cache path.

        New layout:

            cache_root / checkpoint_name / split / short_hash_key / tag.jsonl

        If an old-style cache already exists, reuse it. Otherwise create/use the
        hashed directory.
        """
        # Backwards compatibility: reuse already-created old cache files.
        try:
            legacy = self._legacy_path_no_mkdir(
                checkpoint_name=checkpoint_name,
                split=split,
                cache_key=cache_key,
                tag=tag,
            )
            if legacy.exists():
                return legacy
        except OSError:
            # Long full-key paths may fail even on .exists().
            pass

        key_dir = _short_cache_key_dir(cache_key)
        p = self.cache_root / checkpoint_name / split / key_dir
        p.mkdir(parents=True, exist_ok=True)
        _write_cache_key_manifest(p, cache_key)

        return p / f"{_sanitize_tag(tag)}.jsonl"

    def has(self, *, checkpoint_name: str, split: str, cache_key: str, tag: str) -> bool:
        return self._path(
            checkpoint_name=checkpoint_name,
            split=split,
            cache_key=cache_key,
            tag=tag,
        ).exists()

    def load(self, *, checkpoint_name: str, split: str, cache_key: str, tag: str) -> CachedTexts:
        path = self._path(
            checkpoint_name=checkpoint_name,
            split=split,
            cache_key=cache_key,
            tag=tag,
        )
        prompt, suffix, full = [], [], []
        ref_prompt, ref_suffix, ref_full = [], [], []
        gen_bits, ref_bits, gen_tokens, ref_tokens = [], [], [], []
        meta: Dict[str, Any] = {}

        saw_gen_bits = False
        saw_ref_bits = False
        saw_gen_tokens = False
        saw_ref_tokens = False

        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                if i == 0 and "_meta" in row:
                    meta = row["_meta"]
                    continue

                prompt.append(row["prompt"])
                suffix.append(row["suffix"])
                full.append(row["full"])
                ref_prompt.append(row["ref_prompt"])
                ref_suffix.append(row["ref_suffix"])
                ref_full.append(row["ref_full"])

                if "gen_bits" in row and row["gen_bits"] is not None:
                    gen_bits.append(row["gen_bits"])
                    saw_gen_bits = True
                if "ref_bits" in row and row["ref_bits"] is not None:
                    ref_bits.append(row["ref_bits"])
                    saw_ref_bits = True
                if "gen_tokens" in row and row["gen_tokens"] is not None:
                    gen_tokens.append(row["gen_tokens"])
                    saw_gen_tokens = True
                if "ref_tokens" in row and row["ref_tokens"] is not None:
                    ref_tokens.append(row["ref_tokens"])
                    saw_ref_tokens = True

        return CachedTexts(
            prompt=prompt,
            suffix=suffix,
            full=full,
            ref_prompt=ref_prompt,
            ref_suffix=ref_suffix,
            ref_full=ref_full,
            gen_bits=gen_bits if saw_gen_bits else None,
            ref_bits=ref_bits if saw_ref_bits else None,
            gen_tokens=gen_tokens if saw_gen_tokens else None,
            ref_tokens=ref_tokens if saw_ref_tokens else None,
            meta=meta,
        )

    def save(
        self,
        *,
        checkpoint_name: str,
        split: str,
        cache_key: str,
        tag: str,
        texts: CachedTexts,
    ) -> Path:
        path = self._path(
            checkpoint_name=checkpoint_name,
            split=split,
            cache_key=cache_key,
            tag=tag,
        )
        tmp = path.with_suffix(path.suffix + ".tmp")

        n = len(texts.full)

        with tmp.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": texts.meta}, ensure_ascii=False) + "\n")
            for i in range(n):
                row = {
                    "prompt": texts.prompt[i],
                    "suffix": texts.suffix[i],
                    "full": texts.full[i],
                    "ref_prompt": texts.ref_prompt[i],
                    "ref_suffix": texts.ref_suffix[i],
                    "ref_full": texts.ref_full[i],
                    "gen_bits": None if texts.gen_bits is None else texts.gen_bits[i],
                    "ref_bits": None if texts.ref_bits is None else texts.ref_bits[i],
                    "gen_tokens": None if texts.gen_tokens is None else texts.gen_tokens[i],
                    "ref_tokens": None if texts.ref_tokens is None else texts.ref_tokens[i],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
        return path

    def _decode_batch(self, batch: GenerationBatch, dataset_obj) -> CachedTexts:
        has_bits = batch.gen_bits is not None and batch.ref_bits is not None
        has_tokens = batch.gen_tokens is not None and batch.ref_tokens is not None

        if not has_bits and not has_tokens:
            raise ValueError("GenerationBatch has neither bits nor tokens.")

        if has_bits:
            prompt = decode_bitstreams_for_eval(
                self.cfg,
                batch.gen_bits,
                prompt_len_bits=batch.prompt_len_bits,
                mode="prompt",
                dataset_obj=dataset_obj,
            )
            suffix = decode_bitstreams_for_eval(
                self.cfg,
                batch.gen_bits,
                prompt_len_bits=batch.prompt_len_bits,
                mode="suffix",
                dataset_obj=dataset_obj,
            )
            full = decode_bitstreams_for_eval(
                self.cfg,
                batch.gen_bits,
                prompt_len_bits=batch.prompt_len_bits,
                mode="full",
                dataset_obj=dataset_obj,
            )

            ref_prompt = decode_bitstreams_for_eval(
                self.cfg,
                batch.ref_bits,
                prompt_len_bits=batch.prompt_len_bits,
                mode="prompt",
                dataset_obj=dataset_obj,
            )
            ref_suffix = decode_bitstreams_for_eval(
                self.cfg,
                batch.ref_bits,
                prompt_len_bits=batch.prompt_len_bits,
                mode="suffix",
                dataset_obj=dataset_obj,
            )
            ref_full = decode_bitstreams_for_eval(
                self.cfg,
                batch.ref_bits,
                prompt_len_bits=batch.prompt_len_bits,
                mode="full",
                dataset_obj=dataset_obj,
            )

            gen_tokens = None
            ref_tokens = None

            ds_name = str(getattr(self.cfg.data, "dataset", "")).strip().lower()
            seq_codec = str(getattr(self.cfg.data, "sequence_codec", "base")).strip().lower()
            repr_mode = str(getattr(self.cfg.data, "representation", "binary")).strip().lower()
            binarization = str(getattr(self.cfg.data, "binarization", "raw_binary")).strip().lower()

            if (
                ds_name == "openwebtext"
                and seq_codec == "gpt2id_bpe16"
                and repr_mode == "binary"
                and binarization == "raw_binary"
            ):
                gen_tokens = _tensor_to_nested_list(
                    bitstreams_to_token_ids_raw_binary(
                        batch.gen_bits,
                        bits_per_token=int(getattr(self.cfg.data, "bits_per_token", 16)),
                        cfg=self.cfg,
                    )
                )
                ref_tokens = _tensor_to_nested_list(
                    bitstreams_to_token_ids_raw_binary(
                        batch.ref_bits,
                        bits_per_token=int(getattr(self.cfg.data, "bits_per_token", 16)),
                        cfg=self.cfg,
                    )
                )

            return CachedTexts(
                prompt=prompt,
                suffix=suffix,
                full=full,
                ref_prompt=ref_prompt,
                ref_suffix=ref_suffix,
                ref_full=ref_full,
                gen_bits=_tensor_to_nested_list(batch.gen_bits),
                ref_bits=_tensor_to_nested_list(batch.ref_bits),
                gen_tokens=gen_tokens,
                ref_tokens=ref_tokens,
                meta={},
            )

        prompt = decode_token_sequences_for_eval(
            self.cfg,
            batch.gen_tokens,
            prompt_len_tokens=batch.prompt_len_tokens,
            mode="prompt",
            dataset_obj=dataset_obj,
        )
        suffix = decode_token_sequences_for_eval(
            self.cfg,
            batch.gen_tokens,
            prompt_len_tokens=batch.prompt_len_tokens,
            mode="suffix",
            dataset_obj=dataset_obj,
        )
        full = decode_token_sequences_for_eval(
            self.cfg,
            batch.gen_tokens,
            prompt_len_tokens=batch.prompt_len_tokens,
            mode="full",
            dataset_obj=dataset_obj,
        )

        ref_prompt = decode_token_sequences_for_eval(
            self.cfg,
            batch.ref_tokens,
            prompt_len_tokens=batch.prompt_len_tokens,
            mode="prompt",
            dataset_obj=dataset_obj,
        )
        ref_suffix = decode_token_sequences_for_eval(
            self.cfg,
            batch.ref_tokens,
            prompt_len_tokens=batch.prompt_len_tokens,
            mode="suffix",
            dataset_obj=dataset_obj,
        )
        ref_full = decode_token_sequences_for_eval(
            self.cfg,
            batch.ref_tokens,
            prompt_len_tokens=batch.prompt_len_tokens,
            mode="full",
            dataset_obj=dataset_obj,
        )

        return CachedTexts(
            prompt=prompt,
            suffix=suffix,
            full=full,
            ref_prompt=ref_prompt,
            ref_suffix=ref_suffix,
            ref_full=ref_full,
            gen_bits=None,
            ref_bits=None,
            gen_tokens=_tensor_to_nested_list(batch.gen_tokens),
            ref_tokens=_tensor_to_nested_list(batch.ref_tokens),
            meta={},
        )

    @torch.no_grad()
    def get_or_create(
        self,
        *,
        checkpoint_name: str,
        split: str,
        cache_key: str,
        tag: str,
        model,
        proc,
        device: torch.device,
        data_loader,
        num_samples: int,
        sampler_names: List[str],
        terminal_sigmas: List[float],
        guidance_scales: List[float],
        num_steps: int,
        seed: int,
        use_amp: bool,
        amp_dtype: str,
        micro_batch_size: int,
        sigma_max: Optional[float],
        meta: Dict[str, Any],
        sampling_specs: Optional[List[Dict[str, Any]]] = None,
    ) -> CachedTexts:
        path = self._path(
            checkpoint_name=checkpoint_name,
            split=split,
            cache_key=cache_key,
            tag=tag,
        )
        rank, _world = _rank_world()
        rank0 = rank == 0

        # Rank 0 decides cache hit / miss, then broadcasts.
        cache_exists = bool(path.exists()) if rank0 else False
        if _ddp_is_on():
            exists_obj = [cache_exists]
            dist.broadcast_object_list(exists_obj, src=0)
            cache_exists = bool(exists_obj[0])

        if cache_exists:
            barrier()
            return (
                self.load(
                    checkpoint_name=checkpoint_name,
                    split=split,
                    cache_key=cache_key,
                    tag=tag,
                )
                if rank0
                else CachedTexts.empty()
            )

        entropy_dir_override = getattr(getattr(self.cfg, "evaluation", object()), "entropy_run_dir", None)

        # Safe path: GenerationDriver gathers tensors to rank 0.
        batches = self.driver.generate_prompt_completion(
            model=model,
            proc=proc,
            device=device,
            loader=data_loader,
            num_samples=num_samples,
            sampler_names=sampler_names,
            terminal_sigmas=terminal_sigmas,
            guidance_scales=guidance_scales,
            num_steps=num_steps,
            sampling_specs=sampling_specs,
            seed=seed,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            micro_batch_size=micro_batch_size,
            sigma_max=sigma_max,
            gather_to_rank0=True,
            entropy_run_dir=entropy_dir_override,
        )

        if not rank0:
            barrier()
            return CachedTexts.empty()

        if tag not in batches:
            raise KeyError(
                f"Rank 0 expected generation result for tag='{tag}', "
                f"but GenerationDriver returned keys={list(batches.keys())}"
            )

        batch = batches[tag]
        batch_stats = dict(getattr(batch, "stats", {}) or {})
        merged = self._decode_batch(batch, getattr(data_loader, "dataset", None))

        merged_meta = dict(meta)
        merged_meta.update(batch_stats)
        merged.meta = merged_meta

        self.save(
            checkpoint_name=checkpoint_name,
            split=split,
            cache_key=cache_key,
            tag=tag,
            texts=merged,
        )
        barrier()
        return merged