from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import math
import time
import sys
import functools
import numpy as np
from typing import Any, Dict, Optional, List, Tuple

import torch

try:
    import torch.distributed as dist
except Exception:
    dist = None

import mauve

from evaluation.generation_driver import GenerationDriver
from evaluation.mauve import MauveEvaluator, MauveConfig
from evaluation.text_metrics import avg_token_unigram_entropy_from_token_ids
from utils.text_decode import (
    decode_bitstreams_for_eval,
    decode_token_sequences_for_eval,
    decode_bitstreams_to_token_ids_for_eval,
    decode_token_sequences_to_token_ids_for_eval,
)
from utils.ecc_secded import ecc_from_cfg, ecc_decode_batch_bitstream
from utils.model_utils import unwrap_model, free_vram


def _ddp_on() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return int(dist.get_rank()) if _ddp_on() else 0


def _world() -> int:
    return int(dist.get_world_size()) if _ddp_on() else 1


def _rank0() -> bool:
    return (not _ddp_on()) or _rank() == 0


def _is_discrete_framework(cfg: Any) -> bool:
    return str(getattr(cfg, "framework", "")).lower().startswith("discrete")


def _normalize_text8_flag(cfg: Any) -> bool:
    ds = str(getattr(getattr(cfg, "data", None), "dataset", "")).strip().lower()
    ds = ds.replace("_", "").replace("-", "").replace(" ", "")
    return ds == "text8"


class MauveCallback:
    run_on_all_ranks = True

    def __init__(self, cfg: Any):
        self.cfg_full = cfg
        self._driver: Optional[GenerationDriver] = None
        self._mauve: Optional[MauveEvaluator] = None
        self._last_epoch_ran: Optional[int] = None

    def _resolve_param(self, name: str, default: Any) -> Any:
        train_cfg = getattr(self.cfg_full, "train", None)
        if train_cfg is None:
            return default

        mauve_cfg = getattr(train_cfg, "mauve", None)
        if mauve_cfg is not None:
            v = getattr(mauve_cfg, name, None)
            if v is not None:
                return v

        gen_cfg = getattr(train_cfg, "generation", None)
        if gen_cfg is not None:
            v = getattr(gen_cfg, name, None)
            if v is not None:
                return v

        return default

    def _sc_refresh_mode(self) -> str:
        train_cfg = getattr(self.cfg_full, "train", None)
        if train_cfg is None:
            return "refined"

        mauve_cfg = getattr(train_cfg, "mauve", None)
        if mauve_cfg is not None:
            v = getattr(mauve_cfg, "sc_refresh_mode", None)
            if v is not None:
                return str(v).lower()

        gen_cfg = getattr(train_cfg, "generation", None)
        if gen_cfg is not None:
            v = getattr(gen_cfg, "sc_refresh_mode", None)
            if v is not None:
                return str(v).lower()

        return "refined"

    def _build_sampling_specs(
        self,
        *,
        samplers: List[str],
        terminal_sigmas: List[float],
        guidance_scales: List[float],
        num_steps: int,
        sc_refresh_mode: str,
    ) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for sampler in samplers:
            sampler_l = str(sampler).lower()
            for sigma in terminal_sigmas:
                sigma_f = float(sigma)
                sigma_tag = f"{math.log10(sigma_f):.2f}" if sigma_f > 0.0 else "none"
                for gs in guidance_scales:
                    gs_f = float(gs)
                    tag = f"{sampler_l}_term{sigma_tag}_gs{gs_f:.1f}_scr-{sc_refresh_mode}"
                    specs.append(
                        dict(
                            tag=tag,
                            sampler_name=sampler_l,
                            terminal_sigma=sigma_f,
                            guidance_scale=gs_f,
                            num_steps=int(num_steps),
                            sc_refresh_mode=str(sc_refresh_mode),
                        )
                    )
        return specs

    def _log_scalar(self, trainer, name: str, value: float, step: int) -> None:
        w = getattr(trainer, "writer", None)
        if w is None:
            return
        w.add_scalar(name, float(value), step)
        try:
            w.flush()
        except Exception:
            pass

    def _ensure_mauve(self, trainer) -> None:
        if self._mauve is not None:
            return

        free_vram()

        feat_name = str(self._resolve_param("featurizer_name", "gpt2-large"))
        max_len = int(self._resolve_param("max_tokens", 256))
        device_id = trainer.device.index if trainer.device.type == "cuda" else None

        if _rank0():
            print(f"\n[Mauve] Initializing {feat_name} on cuda:{device_id}...", file=sys.stderr)

        self._mauve = MauveEvaluator(
            cfg=MauveConfig(
                featurize_model_name=feat_name,
                max_text_length=max_len,
                device_id=device_id,
            )
        )

    def _cleanup_mauve(self) -> None:
        if self._mauve is None:
            return
        if _rank0():
            print("[Mauve] Tearing down featurizer and clearing VRAM...", file=sys.stderr)
        del self._mauve
        self._mauve = None
        free_vram()

    def _ecc_local_stats_and_weight(self, trainer, bits: torch.Tensor) -> Tuple[Dict[str, float], int]:
        ecc = ecc_from_cfg(trainer.cfg)
        if ecc is None or not bool(getattr(ecc, "enabled", False)) or bits is None or bits.numel() == 0:
            return {}, 0

        if bits.is_floating_point():
            bits = (bits > 0.5).to(torch.long)
        elif bits.dtype != torch.long:
            bits = bits.to(torch.long)

        _, stats, _ = ecc_decode_batch_bitstream(bits, ecc)

        seq_tok = int(getattr(getattr(trainer.cfg, "data", None), "sequence_len_tokens", 0) or 0)
        B = int(bits.size(0))
        w = B * seq_tok if seq_tok > 0 else B

        out: Dict[str, float] = {}
        for k, v in (stats or {}).items():
            if isinstance(v, (int, float, np.number)) and np.isfinite(float(v)):
                out[str(k)] = float(v)
        return out, int(w)

    def _log_ecc_stats_rank0(self, trainer, bits: torch.Tensor, *, step: int, tag: str) -> None:
        """
        `bits` here are already gathered to rank 0 by GenerationDriver, so there is
        no cross-rank communication in this method.
        """
        if not _rank0() or bits is None or bits.numel() == 0:
            return

        stats, _ = self._ecc_local_stats_and_weight(trainer, bits)
        if not stats:
            return

        for k, v in stats.items():
            self._log_scalar(trainer, f"mauve/val/{tag}/ecc/{k}", float(v), step)

    def _score_mauve_rank0(self, gen_texts: List[str], ref_texts: List[str]) -> Dict[str, float]:
        original_compute = mauve.compute_mauve
        for bs in (128, 64, 32, 16, 8, 4, 2, 1):
            try:
                @functools.wraps(original_compute)
                def fast_compute(*args, **kwargs):
                    kwargs["batch_size"] = int(bs)
                    return original_compute(*args, **kwargs)

                mauve.compute_mauve = fast_compute
                with torch.amp.autocast("cuda", enabled=False):
                    return self._mauve.score(p_text=gen_texts, q_text=ref_texts)
            except torch.cuda.OutOfMemoryError:
                free_vram()
                continue
            finally:
                mauve.compute_mauve = original_compute

        return {"mauve": float("nan")}

    def _debug_texts(self, tag: str, gen_texts: List[str], ref_texts: List[str]) -> None:
        if not _rank0():
            return
        n_empty_gen = sum(len((x or "").strip()) == 0 for x in gen_texts)
        n_empty_ref = sum(len((x or "").strip()) == 0 for x in ref_texts)
        print(
            f"[Mauve debug] {tag}: gen={len(gen_texts)} ref={len(ref_texts)} "
            f"empty_gen={n_empty_gen} empty_ref={n_empty_ref}",
            file=sys.stderr,
        )
        if len(gen_texts) > 0:
            print(f"[Mauve debug] first gen: {repr(gen_texts[0][:200])}", file=sys.stderr)
            print(f"[Mauve debug] first ref: {repr(ref_texts[0][:200])}", file=sys.stderr)

    @torch.no_grad()
    def _run(self, trainer, epoch: int) -> None:
        ep = int(epoch)
        k = int(self._resolve_param("every_k_epochs", 1))
        if ep >= 0 and ((ep + 1) % k) != 0:
            return
        if self._last_epoch_ran == ep:
            return
        self._last_epoch_ran = ep

        if self._driver is None:
            self._driver = GenerationDriver(self.cfg_full)

        samplers = list(self._resolve_param("samplers", ["heun_karras"]))
        terminal_sigmas = list(self._resolve_param("terminal_sigmas", [0.08]))
        num_steps = int(self._resolve_param("num_sampling_steps", 32))
        sigma_max = self._resolve_param("sigma_max", None)
        sigma_max = float(sigma_max) if sigma_max is not None else None
        n_samples = int(self._resolve_param("num_samples", 2048))
        micro_bs = self._resolve_param("micro_batch_size", None)
        if micro_bs is None:
            micro_bs = self._resolve_param("gen_chunk_size", None)

        sc_refresh_mode = self._sc_refresh_mode()

        if _is_discrete_framework(self.cfg_full):
            guidance_scales = [0.0]
        else:
            guidance_scales = list(self._resolve_param("guidance_scales", [0.0, 2.0]))

        sampling_specs = self._build_sampling_specs(
            samplers=samplers,
            terminal_sigmas=terminal_sigmas,
            guidance_scales=guidance_scales,
            num_steps=int(num_steps),
            sc_refresh_mode=sc_refresh_mode,
        )

        if _rank0():
            print(
                f"\n[Mauve] Epoch {ep}: distributed generation "
                f"N={n_samples} steps={num_steps} samplers={samplers} "
                f"sc_refresh={sc_refresh_mode} "
                f"gs={guidance_scales if not _is_discrete_framework(self.cfg_full) else '[discrete:no-CFG]'} "
                f"micro_bs={micro_bs}",
                file=sys.stderr,
            )

        t_gen_start = time.time()

        eval_model = unwrap_model(trainer.model)
        normalize_text8 = _normalize_text8_flag(trainer.cfg)

        batches = self._driver.generate_prompt_completion(
            model=eval_model,
            proc=trainer.proc,
            device=trainer.device,
            loader=trainer.val_loader,
            num_samples=n_samples,
            sampler_names=samplers,
            terminal_sigmas=terminal_sigmas,
            guidance_scales=guidance_scales,
            num_steps=num_steps,
            sampling_specs=sampling_specs,
            use_amp=True,
            amp_dtype="auto",
            micro_batch_size=micro_bs,
            sigma_max=sigma_max,
            gather_to_rank0=True,
        )
        t_gen_end = time.time()

        ds_obj = getattr(trainer.val_loader, "dataset", None)
        step = int(getattr(trainer, "global_step", ep))
        mode = "full"

        try:
            if not _rank0():
                return

            for tag, batch in batches.items():
                has_bits = batch.gen_bits is not None and batch.ref_bits is not None
                has_tokens = batch.gen_tokens is not None and batch.ref_tokens is not None

                if not has_bits and not has_tokens:
                    continue

                if has_bits:
                    gen_bits = batch.gen_bits[:n_samples]
                    ref_bits = batch.ref_bits[:n_samples]

                    self._log_ecc_stats_rank0(trainer, gen_bits, step=step, tag=tag)

                    t_dec_start = time.time()
                    gen_texts = decode_bitstreams_for_eval(
                        trainer.cfg,
                        gen_bits,
                        mode=mode,
                        dataset_obj=ds_obj,
                        normalize_text8=normalize_text8,
                    )
                    ref_texts = decode_bitstreams_for_eval(
                        trainer.cfg,
                        ref_bits,
                        mode=mode,
                        dataset_obj=ds_obj,
                        normalize_text8=normalize_text8,
                    )
                    gen_token_ids = decode_bitstreams_to_token_ids_for_eval(
                        trainer.cfg,
                        gen_bits,
                        dataset_obj=ds_obj,
                    )
                    ref_token_ids = decode_bitstreams_to_token_ids_for_eval(
                        trainer.cfg,
                        ref_bits,
                        dataset_obj=ds_obj,
                    )
                    t_dec_end = time.time()

                else:
                    gen_tokens = batch.gen_tokens[:n_samples]
                    ref_tokens = batch.ref_tokens[:n_samples]

                    t_dec_start = time.time()
                    gen_texts = decode_token_sequences_for_eval(
                        trainer.cfg,
                        gen_tokens,
                        mode=mode,
                        dataset_obj=ds_obj,
                        normalize_text8=normalize_text8,
                    )
                    ref_texts = decode_token_sequences_for_eval(
                        trainer.cfg,
                        ref_tokens,
                        mode=mode,
                        dataset_obj=ds_obj,
                        normalize_text8=normalize_text8,
                    )
                    gen_token_ids = decode_token_sequences_to_token_ids_for_eval(
                        trainer.cfg,
                        gen_tokens,
                        dataset_obj=ds_obj,
                    )
                    ref_token_ids = decode_token_sequences_to_token_ids_for_eval(
                        trainer.cfg,
                        ref_tokens,
                        dataset_obj=ds_obj,
                    )
                    t_dec_end = time.time()

                free_vram()

                self._debug_texts(tag, gen_texts, ref_texts)

                try:
                    H_gen = float(avg_token_unigram_entropy_from_token_ids(gen_token_ids))
                    H_ref = float(avg_token_unigram_entropy_from_token_ids(ref_token_ids))

                    self._log_scalar(trainer, f"mauve/val/{tag}/sample_entropy_gen", H_gen, step)
                    self._log_scalar(trainer, f"mauve/val/{tag}/sample_entropy_ref", H_ref, step)
                except Exception as e:
                    print(f"[Mauve] Warning: failed to compute entropy for {tag}: {e}", file=sys.stderr)

                self._ensure_mauve(trainer)

                t_score_start = time.time()
                res = self._score_mauve_rank0(gen_texts, ref_texts)
                t_score_end = time.time()

                mauve_val = float(res.get("mauve", float("nan")))
                self._log_scalar(trainer, f"mauve/val/{tag}/score", mauve_val, step)

                print(
                    f"[Mauve] {tag}: gen={t_gen_end - t_gen_start:.2f}s "
                    f"dec={t_dec_end - t_dec_start:.2f}s "
                    f"score={t_score_end - t_score_start:.2f}s "
                    f"mauve={mauve_val:.4f}",
                    file=sys.stderr,
                )

                del gen_texts
                del ref_texts
                del gen_token_ids
                del ref_token_ids
                free_vram()

        finally:
            del batches
            free_vram()

            if _rank0():
                self._cleanup_mauve()
            if _ddp_on():
                dist.barrier()

    def on_epoch_end(self, trainer, epoch: int) -> None:
        self._run(trainer, epoch)