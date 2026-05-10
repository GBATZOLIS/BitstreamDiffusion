from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torchvision.utils import make_grid, save_image

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None

from evaluation.generation_driver import GenerationDriver, GenerationBatch
from utils.text_decode import (
    decode_bitstreams_for_eval,
    decode_token_sequences_for_eval,
)
from .base import Callback
from .dataset_utils import is_text_dataset, is_text8, seqs_to_images
from .text_logging import log_text_sequences
from utils.ecc_secded import ecc_from_cfg, ecc_decode_batch_bitstream
from utils.model_utils import unwrap_model, free_vram


def _ddp_is_on() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()


def _rank_world() -> Tuple[int, int]:
    if not _ddp_is_on():
        return 0, 1
    return int(dist.get_rank()), int(dist.get_world_size())


def _rank0() -> bool:
    return _rank_world()[0] == 0


def _barrier() -> None:
    if _ddp_is_on():
        dist.barrier()


def _writer(trainer) -> Optional[Any]:
    w = getattr(trainer, "writer", None)
    if w is not None:
        return w
    tbm = getattr(trainer, "tb_manager", None)
    if tbm is not None and hasattr(tbm, "writer"):
        return tbm.writer
    tb = getattr(trainer, "tb", None)
    if tb is not None:
        return tb
    return None


def _resolve_save_dir(cfg: Any) -> Path:
    base = getattr(cfg, "base_log_dir", None)
    exp = getattr(cfg, "experiment", None)
    if base and exp:
        return Path(str(base)) / str(exp) / "eval_logs" / "viz"
    ev = getattr(cfg, "evaluation", None)
    out_dir = getattr(ev, "out_dir", None) if ev is not None else None
    if out_dir:
        return Path(str(out_dir)) / "viz"
    return Path("eval_logs") / "viz"


def _get_dataset_obj(trainer, *, prefer: str = "val") -> Optional[Any]:
    loader = getattr(trainer, f"{prefer}_loader", None)
    if loader is not None and hasattr(loader, "dataset"):
        return loader.dataset
    loader = getattr(trainer, "train_loader", None)
    if loader is not None and hasattr(loader, "dataset"):
        return loader.dataset
    return None


class VisualizationCallback(Callback):
    run_on_all_ranks = True

    def __init__(self, cfg=None):
        self._driver: Optional[GenerationDriver] = None
        self.cfg = cfg

    def _viz_cfg(self, cfg: Any):
        return getattr(getattr(cfg, "train", None), "visualization", None)

    def _gen_cfg(self, cfg: Any):
        return getattr(getattr(cfg, "train", None), "generation", None)

    def _enabled(self, cfg: Any) -> bool:
        v = self._viz_cfg(cfg)
        if v is not None and getattr(v, "enabled", False):
            return True
        g = self._gen_cfg(cfg)
        return bool(g is not None and getattr(g, "enabled", False))

    def _every_epochs(self, cfg: Any) -> int:
        v = self._viz_cfg(cfg)
        if v is not None:
            if hasattr(v, "every_k_epochs"):
                return int(v.every_k_epochs)
            if hasattr(v, "every_epochs"):
                return int(v.every_epochs)
        g = self._gen_cfg(cfg)
        return int(getattr(g, "every_epochs", 1)) if g is not None else 0

    def _num_samples(self, cfg: Any) -> int:
        v = self._viz_cfg(cfg)
        if v is not None and hasattr(v, "num_samples"):
            return int(v.num_samples)
        g = self._gen_cfg(cfg)
        return int(getattr(g, "num_samples", 64)) if g is not None else 64

    def _num_steps(self, cfg: Any) -> int:
        v = self._viz_cfg(cfg)
        if v is not None and hasattr(v, "num_sampling_steps"):
            return int(v.num_sampling_steps)
        g = self._gen_cfg(cfg)
        return int(getattr(g, "num_sampling_steps", 256)) if g is not None else 256

    def _samplers(self, cfg: Any) -> List[str]:
        v = self._viz_cfg(cfg)
        if v is not None:
            names = getattr(v, "samplers", None)
            if names is None:
                names = getattr(v, "sampler", None)
            if names is not None:
                if isinstance(names, str):
                    return [names.lower()]
                return [str(x).lower() for x in names]

        g = self._gen_cfg(cfg)
        if g is None:
            return ["heun_karras", "ddim_entropic"]
        names = getattr(g, "samplers", None)
        if names is None:
            names = getattr(g, "sampler", None)
        if names is None:
            return ["heun_karras", "ddim_entropic"]
        if isinstance(names, str):
            return [names.lower()]
        return [str(x).lower() for x in names]

    def _terminal_sigmas(self, cfg: Any) -> List[float]:
        v = self._viz_cfg(cfg)
        if v is not None:
            ts = getattr(v, "terminal_sigmas", None)
            if ts is None:
                t = getattr(v, "terminal_sigma", None)
                if t is not None:
                    ts = [float(t)]
            if ts is not None:
                return [float(x) for x in ts]

        g = self._gen_cfg(cfg)
        if g is None:
            return [float(getattr(getattr(cfg, "diffusion", object()).continuous, "sigma_min", 0.01))]
        ts = getattr(g, "terminal_sigmas", None)
        if ts is None:
            t = getattr(g, "terminal_sigma", None)
            if t is None:
                t = float(getattr(getattr(cfg.diffusion, "continuous", object()), "sigma_min", 0.01))
            ts = [float(t)]
        return [float(x) for x in ts]

    def _guidance_scales(self, cfg: Any) -> List[float]:
        v = self._viz_cfg(cfg)
        if v is not None:
            gs = getattr(v, "guidance_scales", None)
            if gs is not None:
                if isinstance(gs, (list, tuple)):
                    out = [float(x) for x in gs]
                else:
                    out = [float(gs)]
                seen, uniq = set(), []
                for val in out:
                    if val not in seen:
                        uniq.append(val)
                        seen.add(val)
                return uniq

        g = self._gen_cfg(cfg)
        if g is None:
            return [0.0]
        gs = getattr(g, "guidance_scales", None)
        if gs is None:
            return [0.0]
        if isinstance(gs, (list, tuple)):
            out = [float(x) for x in gs]
        else:
            out = [float(gs)]
        seen, uniq = set(), []
        for val in out:
            if val not in seen:
                uniq.append(val)
                seen.add(val)
        return uniq

    def _entropic_blend_alpha(self, cfg: Any) -> float:
        v = self._viz_cfg(cfg)
        if v is not None and hasattr(v, "entropic_blend_alpha"):
            return float(v.entropic_blend_alpha)
        g = self._gen_cfg(cfg)
        return float(getattr(g, "entropic_blend_alpha", 0.0)) if g is not None else 0.0

    def _max_text_samples(self, cfg: Any) -> int:
        v = self._viz_cfg(cfg)
        if v is not None:
            if hasattr(v, "max_text_samples"):
                return int(v.max_text_samples)
            if hasattr(v, "num_samples"):
                return int(v.num_samples)
        g = self._gen_cfg(cfg)
        return int(getattr(g, "max_text_samples", 8)) if g is not None else 8

    def _grid_nrow(self, cfg: Any, B: int) -> int:
        g = self._gen_cfg(cfg)
        nrow = getattr(g, "grid_nrow", None) if g is not None else None
        if nrow is None:
            return max(1, int(math.sqrt(max(1, B))))
        return int(nrow)

    def _seed(self, cfg: Any) -> int:
        v = self._viz_cfg(cfg)
        if v is not None and getattr(v, "seed", None) is not None:
            return int(getattr(v, "seed"))
        g = self._gen_cfg(cfg)
        if g is not None and getattr(g, "seed", None) is not None:
            return int(getattr(g, "seed"))
        return int(getattr(getattr(cfg, "train", object()), "seed", 42))

    def _save_to_disk(self, cfg: Any) -> bool:
        v = self._viz_cfg(cfg)
        if v is not None:
            if getattr(v, "save_to_disk", False):
                return True
            if getattr(v, "save_txt", False):
                return True
        g = self._gen_cfg(cfg)
        return bool(getattr(g, "save_to_disk", False)) if g is not None else False

    def _entropy_run_dir(self, cfg: Any) -> Optional[str]:
        v = self._viz_cfg(cfg)
        if v is not None:
            direct = getattr(v, "entropy_run_dir", None)
            if direct:
                return str(direct)

        g = self._gen_cfg(cfg)
        if g is None:
            return None

        direct = getattr(g, "entropy_run_dir", None)
        if direct:
            return str(direct)

        ckpt = getattr(g, "entropy_ckpt_path", None)
        if ckpt is None:
            ev = getattr(cfg, "evaluation", None)
            ckpt = getattr(ev, "checkpoint_path", None) if ev is not None else None
        if ckpt is None:
            return None

        ckpt_path = Path(str(ckpt)).expanduser().resolve()
        run_root = ckpt_path.parent.parent
        return str(run_root)

    def _cond_enabled(self, cfg: Any) -> bool:
        cc = getattr(cfg, "cond", None)
        return bool(cc is not None and getattr(cc, "enabled", False))

    def _splits_to_run(self, cfg: Any) -> List[Optional[str]]:
        viz_cfg = getattr(getattr(cfg, "train", None), "visualization", None)
        if viz_cfg is not None:
            splits = getattr(viz_cfg, "splits", None)
            if splits is not None:
                return [splits] if isinstance(splits, str) else list(splits)

        gen_cfg = self._gen_cfg(cfg)
        if gen_cfg is not None:
            splits = getattr(gen_cfg, "splits", None)
            if splits is not None:
                return [splits] if isinstance(splits, str) else list(splits)

        if self._cond_enabled(cfg):
            return ["train", "val"]
        return [None]

    def _sigma_max(self, cfg: Any) -> Optional[float]:
        v = self._viz_cfg(cfg)
        if v is not None and hasattr(v, "sigma_max"):
            val = getattr(v, "sigma_max", None)
            return None if val is None else float(val)

        g = self._gen_cfg(cfg)
        if g is not None and hasattr(g, "sigma_max"):
            val = getattr(g, "sigma_max", None)
            return None if val is None else float(val)

        return None

    def _micro_batch_size(self, cfg: Any) -> Optional[int]:
        v = self._viz_cfg(cfg)
        if v is not None and hasattr(v, "micro_batch_size"):
            mb = getattr(v, "micro_batch_size", None)
            if mb is not None:
                return int(mb)

        g = self._gen_cfg(cfg)
        if g is not None and hasattr(g, "micro_batch_size"):
            mb = getattr(g, "micro_batch_size", None)
            if mb is not None:
                return int(mb)

        return None

    def _sc_refresh_mode(self, cfg: Any) -> str:
        v = self._viz_cfg(cfg)
        if v is not None and hasattr(v, "sc_refresh_mode"):
            mode = getattr(v, "sc_refresh_mode", None)
            if mode is not None:
                return str(mode).lower()

        g = self._gen_cfg(cfg)
        if g is not None and hasattr(g, "sc_refresh_mode"):
            mode = getattr(g, "sc_refresh_mode", None)
            if mode is not None:
                return str(mode).lower()

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

    def _ensure_driver(self, trainer: Any) -> None:
        if self._driver is None:
            self._driver = GenerationDriver(trainer.cfg)

    def _should_run(self, cfg: Any, epoch: int) -> bool:
        if not self._enabled(cfg):
            return False
        k = self._every_epochs(cfg)
        if k <= 0:
            return False
        return ((int(epoch) + 1) % int(k)) == 0

    def _maybe_log_ecc_stats(
        self,
        trainer,
        bits: torch.Tensor,
        *,
        epoch: int,
        tag: str,
        split: Optional[str] = None,
    ):
        ecc = ecc_from_cfg(trainer.cfg)
        if ecc is None or not bool(getattr(ecc, "enabled", False)):
            return

        if bits.is_floating_point():
            bits = (bits > 0.5).long()

        _, stats, _ = ecc_decode_batch_bitstream(bits, ecc)

        split = str(split).lower().strip() if split is not None else None
        if split in {"train", "val", "test"}:
            prefix = f"{split}/ecc/{tag}"
        else:
            prefix = f"ecc/{tag}"

        w = _writer(trainer)
        if w is not None:
            for k, v in stats.items():
                w.add_scalar(f"{prefix}/{k}", float(v), epoch)
            w.flush()

    def _log_images(
        self,
        trainer: Any,
        *,
        tag: str,
        bits_u8: torch.Tensor,
        epoch: int,
        save_dir: Optional[Path],
    ) -> None:
        w = _writer(trainer)
        if w is None:
            return
        bits_f = bits_u8.to(dtype=torch.float32)
        imgs = seqs_to_images(bits_f, trainer)
        grid = make_grid(
            imgs.detach().cpu().clamp(0, 1),
            nrow=self._grid_nrow(trainer.cfg, imgs.size(0)),
        )
        w.add_image(tag, grid, epoch)
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            out = save_dir / f"{tag.replace('/', '__')}.png"
            save_image(grid, out)

    def _save_texts(
        self,
        save_dir: Path,
        *,
        tag: str,
        full: List[str],
        suffix: List[str],
        prompt: Optional[List[str]] = None,
    ) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = tag.replace("/", "__")
        p = save_dir / f"{safe_tag}.txt"

        n = min(len(full), len(suffix))
        if prompt is not None:
            n = min(n, len(prompt))

        lines: List[str] = []
        for i in range(n):
            lines.append(f"=== sample {i} ===")
            if prompt is not None:
                lines.append("[prompt]")
                lines.append(prompt[i])
                lines.append("")
            lines.append("[full]")
            lines.append(full[i])
            lines.append("")
            lines.append("[suffix]")
            lines.append(suffix[i])
            lines.append("")
        p.write_text("\n".join(lines), encoding="utf-8")

    def _log_text_preview(
        self,
        trainer: Any,
        *,
        tag: str,
        epoch: int,
        full: List[str],
        suffix: Optional[List[str]] = None,
        prompt: Optional[List[str]] = None,
    ) -> None:
        w = _writer(trainer)
        if w is None:
            return

        n = len(full)
        lines: List[str] = []
        for i in range(n):
            lines.append(f"Sample {i}:")
            if prompt is not None:
                lines.append(f"  prompt: {repr(prompt[i])}")
            if suffix is not None:
                lines.append(f"  suffix: {repr(suffix[i])}")
            lines.append(f"  text: {repr(full[i])}")
            lines.append("")

        body = "\n".join(lines)
        w.add_text(tag, f"```\n{body}\n```", epoch)
        try:
            w.flush()
        except Exception:
            pass

    def _decode_text_batch(
        self,
        cfg: Any,
        batch: GenerationBatch,
        *,
        ds_obj: Optional[Any],
        max_text: int,
        normalize_text8: bool,
    ) -> Optional[Dict[str, List[str]]]:
        has_bits = batch.gen_bits is not None and batch.ref_bits is not None
        has_tokens = batch.gen_tokens is not None and batch.ref_tokens is not None

        if not has_bits and not has_tokens:
            return None

        if has_bits:
            gen_bits = batch.gen_bits[:max_text]
            prompt_len_bits = batch.prompt_len_bits[:max_text] if batch.prompt_len_bits is not None else None

            full = decode_bitstreams_for_eval(
                cfg,
                gen_bits,
                mode="full",
                dataset_obj=ds_obj,
                normalize_text8=normalize_text8,
            )
            suffix = decode_bitstreams_for_eval(
                cfg,
                gen_bits,
                prompt_len_bits=prompt_len_bits,
                mode="suffix",
                dataset_obj=ds_obj,
                normalize_text8=normalize_text8,
            )
            prompt = decode_bitstreams_for_eval(
                cfg,
                batch.ref_bits[:max_text],
                prompt_len_bits=prompt_len_bits,
                mode="prompt",
                dataset_obj=ds_obj,
                normalize_text8=normalize_text8,
            ) if prompt_len_bits is not None else [""] * len(full)

            return dict(full=full, suffix=suffix, prompt=prompt)

        gen_tokens = batch.gen_tokens[:max_text]
        prompt_len_tokens = batch.prompt_len_tokens[:max_text] if batch.prompt_len_tokens is not None else None

        full = decode_token_sequences_for_eval(
            cfg,
            gen_tokens,
            mode="full",
            dataset_obj=ds_obj,
            normalize_text8=normalize_text8,
        )
        suffix = decode_token_sequences_for_eval(
            cfg,
            gen_tokens,
            prompt_len_tokens=prompt_len_tokens,
            mode="suffix",
            dataset_obj=ds_obj,
            normalize_text8=normalize_text8,
        )
        prompt = decode_token_sequences_for_eval(
            cfg,
            batch.ref_tokens[:max_text],
            prompt_len_tokens=prompt_len_tokens,
            mode="prompt",
            dataset_obj=ds_obj,
            normalize_text8=normalize_text8,
        ) if prompt_len_tokens is not None else [""] * len(full)

        return dict(full=full, suffix=suffix, prompt=prompt)

    @torch.no_grad()
    def on_epoch_end(self, trainer: Any, epoch: int) -> None:
        cfg = trainer.cfg
        if not self._should_run(cfg, epoch):
            return

        self._ensure_driver(trainer)
        is_text = is_text_dataset(trainer)
        ds_obj = _get_dataset_obj(trainer, prefer="val")

        num_samples = self._num_samples(cfg)
        samplers = self._samplers(cfg)
        terminal_sigmas = self._terminal_sigmas(cfg)
        guidance_scales = self._guidance_scales(cfg)
        num_steps = self._num_steps(cfg)
        entropic_blend_alpha = self._entropic_blend_alpha(cfg)
        entropy_run_dir = self._entropy_run_dir(cfg)
        seed = self._seed(cfg)
        use_amp = bool(getattr(getattr(cfg, "train", object()), "use_fp16", False))
        sigma_max = self._sigma_max(cfg)
        micro_batch_size = self._micro_batch_size(cfg)
        sc_refresh_mode = self._sc_refresh_mode(cfg)

        sampling_specs = self._build_sampling_specs(
            samplers=samplers,
            terminal_sigmas=terminal_sigmas,
            guidance_scales=guidance_scales,
            num_steps=int(num_steps),
            sc_refresh_mode=sc_refresh_mode,
        )

        trainer.ema.apply(trainer.model)
        trainer.model.eval()

        try:
            for split in self._splits_to_run(cfg):
                if split == "train":
                    loader = trainer.train_loader
                else:
                    loader = trainer.val_loader

                eval_model = unwrap_model(trainer.model)
                eval_model.eval()

                with torch.inference_mode():
                    out = self._driver.generate_prompt_completion(
                        model=eval_model,
                        proc=trainer.proc,
                        device=trainer.device,
                        loader=loader,
                        num_samples=int(num_samples),
                        sampler_names=[str(s) for s in samplers],
                        terminal_sigmas=[float(x) for x in terminal_sigmas],
                        guidance_scales=[float(g) for g in guidance_scales],
                        num_steps=int(num_steps),
                        sampling_specs=sampling_specs,
                        entropic_blend_alpha=float(entropic_blend_alpha),
                        entropy_run_dir=entropy_run_dir,
                        seed=int(seed),
                        use_amp=use_amp,
                        amp_dtype="auto",
                        micro_batch_size=micro_batch_size,
                        sigma_max=sigma_max,
                    )

                if _rank0():
                    batches_by_tag: Dict[str, GenerationBatch] = dict(out)

                    save_root = _resolve_save_dir(cfg) / f"epoch_{int(epoch):04d}"
                    save_dir = save_root if self._save_to_disk(cfg) else None
                    max_text = self._max_text_samples(cfg)
                    normalize = is_text8(trainer)

                    for base_tag, batch in batches_by_tag.items():
                        tag = f"generated_samples/{base_tag}"
                        if split in {"train", "val"}:
                            tag = f"{split}/{tag}"

                        has_bits = batch.gen_bits is not None and batch.ref_bits is not None
                        has_tokens = batch.gen_tokens is not None and batch.ref_tokens is not None

                        if is_text:
                            if has_bits:
                                self._maybe_log_ecc_stats(
                                    trainer,
                                    batch.gen_bits,
                                    epoch=epoch,
                                    tag=base_tag,
                                    split=split,
                                )

                                prefix_bits = batch.ref_bits if batch.ref_bits is not None else None
                                prefix_mask = None
                                if prefix_bits is not None and batch.prompt_len_bits is not None:
                                    B = int(prefix_bits.size(0))
                                    S = int(prefix_bits.size(1))
                                    ar = torch.arange(S).view(1, S)
                                    pl = batch.prompt_len_bits.view(B, 1)
                                    prefix_mask = ar < pl

                                log_text_sequences(
                                    trainer,
                                    batch.gen_bits[:max_text],
                                    tag,
                                    epoch,
                                    max_samples=max_text,
                                    prefix_bits=(prefix_bits[:max_text] if prefix_bits is not None else None),
                                    prefix_mask=(prefix_mask[:max_text] if prefix_mask is not None else None),
                                )

                            elif has_tokens:
                                decoded = self._decode_text_batch(
                                    cfg,
                                    batch,
                                    ds_obj=ds_obj,
                                    max_text=max_text,
                                    normalize_text8=normalize,
                                )
                                if decoded is not None:
                                    self._log_text_preview(
                                        trainer,
                                        tag=tag,
                                        epoch=epoch,
                                        full=decoded["full"],
                                        suffix=decoded["suffix"],
                                        prompt=decoded["prompt"],
                                    )

                                    if save_dir is not None:
                                        self._save_texts(
                                            save_dir,
                                            tag=tag,
                                            full=decoded["full"],
                                            suffix=decoded["suffix"],
                                            prompt=decoded["prompt"],
                                        )

                                    continue

                            decoded = self._decode_text_batch(
                                cfg,
                                batch,
                                ds_obj=ds_obj,
                                max_text=max_text,
                                normalize_text8=normalize,
                            )
                            if decoded is not None and save_dir is not None:
                                self._save_texts(
                                    save_dir,
                                    tag=tag,
                                    full=decoded["full"],
                                    suffix=decoded["suffix"],
                                    prompt=decoded["prompt"],
                                )

                        else:
                            self._log_images(
                                trainer,
                                tag=tag,
                                bits_u8=batch.gen_bits,
                                epoch=epoch,
                                save_dir=save_dir,
                            )

                    del batches_by_tag

                del out
                free_vram()

                _barrier()

        finally:
            free_vram()
            trainer.ema.restore(trainer.model)
            _barrier()