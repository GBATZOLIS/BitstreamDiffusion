from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple

import torch
from torch.cuda.amp import autocast
from torchvision.utils import make_grid

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from utils.text_decode import decode_samples_to_text
from .base import Callback
from .dataset_utils import is_text_dataset, is_text8, seqs_to_images
from .text_logging import log_text_sequences
from utils.ecc_secded import ecc_from_cfg, ecc_chunk_len, ecc_decode_batch_bitstream


class GenerationCallback(Callback):
    """
    Generates and logs samples during training.

    CRITICAL RULE:
      - ALL generation parameters are read ONLY from self.cfg.train.generation

    Conditioning:
      - Conditional generation enabled if cfg.cond.enabled=True.
      - Prefixes are taken from train and val splits and passed into samplers.
      - Supports fixed prefix length OR per-example prefix length sampling.

    Logging hierarchy (VLB-like):
      - train/generated_samples/<tag>
      - val/generated_samples/<tag>
      - train/external_ppl/<tag>/<metric>
      - val/external_ppl/<tag>/<metric>
    """

    run_on_all_ranks = False

    def __init__(self, sampler: Any, cfg: Any, is_discrete: bool):
        self.sampler = sampler
        self.ddim_sampler = None
        self.cfg = cfg
        self.is_discrete = is_discrete
        self._ext_eval = None

    # ---------------------------------------------------------------------
    # cfg.train.generation access
    # ---------------------------------------------------------------------
    def _gen_cfg(self):
        return getattr(getattr(self.cfg, "train", None), "generation", None)

    def _require_gen_cfg(self):
        g = self._gen_cfg()
        if g is None:
            raise ValueError("GenerationCallback requires cfg.train.generation to exist.")
        return g

    def _gen_enabled(self) -> bool:
        g = self._gen_cfg()
        return bool(g is not None and getattr(g, "enabled", False))

    def _gen_every_epochs(self) -> int:
        g = self._require_gen_cfg()
        return int(getattr(g, "every_epochs", 1))

    def _gen_num_samples(self) -> int:
        g = self._require_gen_cfg()
        return int(getattr(g, "num_samples", 64))

    def _gen_num_steps(self) -> int:
        g = self._require_gen_cfg()
        return int(getattr(g, "num_sampling_steps", 400))

    def _gen_samplers(self) -> List[str]:
        g = self._require_gen_cfg()
        names = getattr(g, "samplers", None)
        if names is None:
            names = getattr(g, "sampler", None)
        if names is None:
            return ["tweedie"]
        if isinstance(names, str):
            return [names.lower()]
        return [str(n).lower() for n in names]

    def _gen_max_text_samples(self) -> int:
        g = self._require_gen_cfg()
        return int(getattr(g, "max_text_samples", 8))

    def _gen_grid_nrow(self, B_gen: int) -> int:
        g = self._require_gen_cfg()
        nrow = getattr(g, "grid_nrow", None)
        if nrow is None:
            return max(1, int(math.sqrt(B_gen)))
        return int(nrow)

    def _gen_progress_bar(self) -> bool:
        g = self._require_gen_cfg()
        return bool(getattr(g, "progress_bar", True))

    # Continuous-only knobs
    def _gen_entropic_blend_alpha(self) -> float:
        g = self._require_gen_cfg()
        return float(getattr(g, "entropic_blend_alpha", 0.0))

    def _gen_terminal_sigmas(self) -> List[float]:
        g = self._require_gen_cfg()
        terms = getattr(g, "terminal_sigmas", None)
        if terms is None:
            s0 = getattr(g, "terminal_sigma", None)
            if s0 is None:
                s0 = float(getattr(self.cfg.diffusion.continuous, "sigma_min", 1e-3))
            terms = [float(s0)]
        return [float(s) for s in terms]

    def _entropy_run_dir(self) -> Path:
        g = self._require_gen_cfg()
        ckpt = getattr(g, "entropy_ckpt_path", None)
        if ckpt is None:
            ckpt = getattr(getattr(self.cfg, "evaluation", None), "checkpoint_path", None)
            if ckpt is None:
                raise ValueError("Need entropy tables dir but neither cfg.train.generation.entropy_ckpt_path nor cfg.evaluation.checkpoint_path is set.")
        ckpt_path = Path(str(ckpt)).expanduser().resolve()
        return ckpt_path.parent.parent

    # ---------------------------------------------------------------------
    # Conditioning helpers (match training semantics)
    # ---------------------------------------------------------------------
    def _cond_cfg(self):
        return getattr(self.cfg, "cond", None)

    def _cond_enabled_cfg(self) -> bool:
        cc = self._cond_cfg()
        return bool(cc is not None and getattr(cc, "enabled", False))

    def _bits_per_unit(self) -> int:
        ecc = ecc_from_cfg(self.cfg)
        if ecc.enabled:
            return int(ecc_chunk_len(ecc))  # 21
        data = getattr(self.cfg, "data", object())
        bpt = getattr(data, "bits_per_token", None)
        if bpt is not None:
            return int(bpt)
        return int(getattr(data, "bits_per_char", 1))

    def _fixed_cond_len_bits(self, seq_len_bits: int) -> int:
        if not self._cond_enabled_cfg():
            return 0
        cond = self._cond_cfg()
        bits_per = self._bits_per_unit()

        n_units = getattr(cond, "cond_len_tokens", None)
        if n_units is None:
            n_units = int(getattr(cond, "cond_len_chars", 0))
        else:
            n_units = int(n_units)

        cL = int(n_units * bits_per)
        return max(0, min(int(cL), int(seq_len_bits)))

    def _sample_cond_len_bits_per_example(self, B: int, seq_len_bits: int, device) -> torch.Tensor:
        cond = self._cond_cfg()
        if cond is None or not bool(getattr(cond, "enabled", False)):
            return torch.zeros(B, device=device, dtype=torch.long)

        sample_len = bool(getattr(cond, "sample_prompt_len", False))
        if not sample_len:
            cL = self._fixed_cond_len_bits(seq_len_bits)
            return torch.full((B,), int(cL), device=device, dtype=torch.long)

        bits_per = self._bits_per_unit()
        mn = getattr(cond, "cond_len_tokens_min", None)
        mx = getattr(cond, "cond_len_tokens_max", None)
        if mn is None or mx is None:
            mn = int(getattr(cond, "cond_len_chars_min", 0))
            mx = int(getattr(cond, "cond_len_chars_max", 0))
        else:
            mn = int(mn)
            mx = int(mx)

        mn = max(0, mn)
        mx = max(mn, mx)

        if mx == mn:
            units = torch.full((B,), mn, device=device, dtype=torch.long)
        else:
            units = torch.randint(low=mn, high=mx + 1, size=(B,), device=device, dtype=torch.long)

        cL_bits = units * int(bits_per)
        cL_bits = torch.clamp(cL_bits, min=0, max=int(seq_len_bits)).to(torch.long)
        return cL_bits

    def _prefix_mask_from_lengths(self, cL_bits: torch.Tensor, S: int) -> torch.Tensor:
        B = int(cL_bits.numel())
        ar = torch.arange(S, device=cL_bits.device).view(1, S).expand(B, S)
        return ar < cL_bits.view(B, 1)

    def _gen_guidance_scales(self) -> List[float]:
        g = self._require_gen_cfg()
        gs = getattr(g, "guidance_scales", None)
        if gs is None:
            return [0.0]
        if isinstance(gs, (list, tuple)):
            out = [float(x) for x in gs]
        else:
            out = [float(gs)]
        # dedup keep order
        seen = set()
        uniq = []
        for v in out:
            if v not in seen:
                uniq.append(v)
                seen.add(v)
        return uniq

    def _get_split_iter_order(self) -> List[str]:
        g = self._require_gen_cfg()
        splits = getattr(g, "splits", None)
        if splits is None:
            return ["train", "val"]
        if isinstance(splits, str):
            splits = [splits]
        out = []
        for s in splits:
            s = str(s).lower().strip()
            if s in {"train", "val"}:
                out.append(s)
        return out or ["train", "val"]


    @torch.no_grad()
    def _sample_full_sequences_from_split(self, trainer: Any, B: int, *, split: str) -> Optional[torch.Tensor]:
        """
        Returns full sequences [B,S] float32 on trainer.device from split loader.
        """
        split = str(split).lower().strip()
        split_loader = getattr(trainer, f"{split}_loader", None)
        if split_loader is None:
            return None

        chunks = []
        n = 0
        for batch in split_loader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]

            if batch.dim() == 3:
                if batch.size(1) == 1:
                    batch = batch.squeeze(1)
                else:
                    batch = batch.view(batch.size(0), -1)
            else:
                batch = batch.view(batch.size(0), -1)

            batch = batch.to(device=trainer.device, dtype=torch.float32, non_blocking=True)
            chunks.append(batch)
            n += batch.size(0)
            if n >= B:
                break

        if not chunks:
            return None
        x = torch.cat(chunks, dim=0)[:B].contiguous()
        return x

    # ---------------------------------------------------------------------
    # External perplexity
    # ---------------------------------------------------------------------
    def _external_cfg(self):
        return getattr(getattr(self.cfg, "train", None), "external_perplexity", None)

    def _external_enabled(self) -> bool:
        cfg_ext = self._external_cfg()
        return bool(cfg_ext is not None and getattr(cfg_ext, "enabled", False))

    def _maybe_init_external(self, trainer: Any):
        if self._ext_eval is not None:
            return self._ext_eval

        cfg_ext = self._external_cfg()
        if cfg_ext is None:
            return None

        ar_config = getattr(cfg_ext, "ar_config", None)
        if ar_config is None:
            raise ValueError("cfg.train.external_perplexity.enabled=True but ar_config is not set.")

        from evaluation.external_perplexity import load_config as load_ar_config
        from evaluation.external_perplexity import ExternalPerplexityEvaluator

        cfg_ar = load_ar_config(str(ar_config))
        ar_ckpt = getattr(cfg_ext, "ar_ckpt", None)
        if ar_ckpt is None:
            if not hasattr(cfg_ar, "evaluation") or not hasattr(cfg_ar.evaluation, "checkpoint_path"):
                raise ValueError("AR config must define cfg_ar.evaluation.checkpoint_path (or set cfg.train.external_perplexity.ar_ckpt).")
            ar_ckpt = str(cfg_ar.evaluation.checkpoint_path)

        self._ext_eval = ExternalPerplexityEvaluator(
            ar_config_path=str(ar_config),
            ar_ckpt_path=str(ar_ckpt),
            device=trainer.device,
            use_amp=bool(getattr(cfg_ext, "use_amp", True)),
        )
        return self._ext_eval

    @torch.no_grad()
    def _run_external_perplexity(
        self,
        trainer: Any,
        *,
        epoch: int,
        sampler_tag: str,
        samples_for_decode: torch.Tensor,
        split: Optional[str] = None,
    ):
        if not self._external_enabled():
            return
        if not getattr(trainer, "is_master", True):
            return
        if not is_text_dataset(trainer):
            return

        ext = self._maybe_init_external(trainer)
        if ext is None:
            return

        ds_for_decode = (
            trainer.val_loader.dataset
            if getattr(trainer, "val_loader", None) is not None
            else trainer.train_loader.dataset
        )

        normalize = is_text8(trainer)

        texts_raw = decode_samples_to_text(
            self.cfg,
            samples_for_decode,
            dataset_obj=ds_for_decode,
            normalize_text8=normalize,
        )

        out = ext.score(texts_raw)

        cfg_ext = self._external_cfg()
        base = str(getattr(cfg_ext, "log_prefix", "external_ppl")).strip()
        split = None if split is None else str(split).lower().strip()

        if split in {"train", "val"}:
            prefix_root = f"{split}/{base}/{sampler_tag}"
        else:
            prefix_root = f"{base}/{sampler_tag}"

        for k, v in out.items():
            trainer.writer.add_scalar(f"{prefix_root}/{k}", float(v), epoch)

        if getattr(trainer, "use_wandb", False) and wandb is not None:
            trainer._log_wandb({f"{prefix_root}/{k}": float(v) for k, v in out.items()})

        split_str = f"{split}/" if split in {"train", "val"} else ""
        print(f"✓ External perplexity ({split_str}{sampler_tag}): " + ", ".join(f"{k}={v:.4f}" for k, v in out.items()))

    #ECC SECDED helper
    def _maybe_log_ecc_stats(self, trainer, bits: torch.Tensor, *, epoch: int, tag: str, split: Optional[str] = None):
        # 1. Check Config
        ecc = ecc_from_cfg(self.cfg)
        if not ecc.enabled:
            # Debug: Uncomment if you suspect config issues
            # print(f"[GenerationCallback] ECC disabled in config for epoch {epoch}, skipping stats.")
            return

        # 2. Compute Stats
        _, stats, _ = ecc_decode_batch_bitstream(bits, ecc)
        print(stats)

        # 3. Construct Tag
        # If split is None (unconditional), we use "ecc" as root.
        # If split is "train"/"val", we use "train/ecc" or "val/ecc".
        split = str(split).lower().strip() if split is not None else None
        if split in {"train", "val", "test"}:
            prefix = f"{split}/ecc/{tag}"
        else:
            prefix = f"ecc/{tag}"

        # 4. Log to TensorBoard (Strictly)
        if hasattr(trainer, "writer") and trainer.writer is not None:
            for k, v in stats.items():
                full_key = f"{prefix}/{k}"
                trainer.writer.add_scalar(full_key, float(v), epoch)
            
            # Force write to disk immediately to ensure it appears
            trainer.writer.flush()
            print(f"✓ ECC stats logged to TensorBoard: {prefix}/*")
        else:
            print("⚠️ Trainer has no 'writer' attribute; cannot log ECC stats to TensorBoard.")



    # ---------------------------------------------------------------------
    # Sanity log: real samples once
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def on_train_begin(self, trainer: Any):
        if not is_text_dataset(trainer):
            return
        if not getattr(trainer, "is_master", True):
            return

        print("\n— Logging Real Data Samples (Sanity Check) ...")
        try:
            val_split = getattr(trainer, "val_loader", None)
            if val_split is None:
                print("⚠️ No validation split found, skipping real sample log.")
                return

            batch = next(iter(val_split))
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(trainer.device)

            real_samples = batch[: self._gen_max_text_samples()]
            log_text_sequences(trainer, real_samples, "real_data", 0, max_samples=self._gen_max_text_samples())
            print("✓ Real samples logged to TensorBoard/WandB.")
        except Exception as e:
            print(f"❌ Failed to log real samples: {e}")

    # ---------------------------------------------------------------------
    # Main generation hook
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def on_epoch_end(self, trainer: Any, epoch: int):
        if not self._gen_enabled():
            return

        every = self._gen_every_epochs()
        if every <= 0:
            return
        if (epoch + 1) % every != 0:
            return
        if not getattr(trainer, "is_master", True):
            return

        print(f"\n— Running GenerationCallback (Epoch {epoch + 1})")

        trainer.ema.apply(trainer.model)
        trainer.model.eval()

        is_text = is_text_dataset(trainer)

        use_amp = bool(getattr(self.cfg.train, "use_fp16", False))
        amp_dtype = getattr(trainer, "amp_dtype", torch.float16)

        B_gen = self._gen_num_samples()
        num_steps = self._gen_num_steps()
        
        # Get the list of requested samplers from config (e.g. ["heun_karras", "ddim_entropic"])
        target_sampler_names = self._gen_samplers()

        with autocast(enabled=use_amp, dtype=amp_dtype):
            try:
                # ============================= DISCRETE =============================
                if self.is_discrete:
                    # (Discrete logic remains unchanged)
                    seq_len = int(getattr(self.cfg.data, "sequence_len", 0))
                    if seq_len <= 0:
                        raise ValueError(f"cfg.data.sequence_len must be > 0 for generation, got {seq_len}")

                    from diffusion.discrete.samplers import TweedieTauLeapingSampler, EulerRateSampler

                    proc = trainer.proc
                    is_absorb = bool(getattr(proc, "is_absorb", False))
                    mask_id_cfg = int(getattr(self.cfg.data, "mask_token_id", -1))
                    mask_id = mask_id_cfg if is_absorb else None

                    samplers: Dict[str, Any] = {}
                    for name in target_sampler_names:
                        key = str(name).lower()
                        if key in {"tweedie", "tweedie_tau", "tau"}:
                            samplers["tweedie"] = TweedieTauLeapingSampler(
                                model=trainer.model,
                                process=proc,
                                device=trainer.device,
                                vocab_size=int(self.cfg.data.vocab_size),
                                mask_id=mask_id,
                                is_absorb=is_absorb,
                                seq_len=seq_len,
                                num_steps=num_steps,
                                t_eps=float(getattr(self._require_gen_cfg(), "t_eps", 1e-3)),
                                denoise=bool(getattr(self._require_gen_cfg(), "denoise", True)),
                                log_score_clip=float(getattr(self._require_gen_cfg(), "log_score_clip_tweedie", 80.0)),
                                prob_floor=float(getattr(self._require_gen_cfg(), "prob_floor", 1e-12)),
                                disallow_mask_in_final=bool(getattr(self._require_gen_cfg(), "disallow_mask_in_final", True)),
                                save_trajectories=bool(getattr(self._require_gen_cfg(), "save_trajectories", False)),
                                trajectory_to_cpu=bool(getattr(self._require_gen_cfg(), "trajectory_to_cpu", True)),
                                progress_bar=bool(self._gen_progress_bar()),
                            )
                        elif key in {"euler", "euler_rate"}:
                            samplers["euler"] = EulerRateSampler(
                                model=trainer.model,
                                process=proc,
                                device=trainer.device,
                                vocab_size=int(self.cfg.data.vocab_size),
                                mask_id=mask_id,
                                is_absorb=is_absorb,
                                seq_len=seq_len,
                                num_steps=num_steps,
                                t_eps=float(getattr(self._require_gen_cfg(), "t_eps", 1e-3)),
                                log_score_clip=float(getattr(self._require_gen_cfg(), "log_score_clip_euler", 60.0)),
                                prob_floor=float(getattr(self._require_gen_cfg(), "prob_floor", 1e-12)),
                                disallow_mask_in_final=bool(getattr(self._require_gen_cfg(), "disallow_mask_in_final", True)),
                                save_trajectories=bool(getattr(self._require_gen_cfg(), "save_trajectories", False)),
                                trajectory_to_cpu=bool(getattr(self._require_gen_cfg(), "trajectory_to_cpu", True)),
                                progress_bar=bool(self._gen_progress_bar()),
                            )

                    if not samplers:
                        print("⚠️ No valid discrete samplers configured; skipping generation.")
                        return

                    for tag, sampler in samplers.items():
                        seq = sampler.sample(num_samples=B_gen)

                        if is_text:
                            tag_text = f"generated_samples/{tag}"
                            log_text_sequences(
                                trainer,
                                seq[: self._gen_max_text_samples()],
                                tag_text,
                                epoch,
                                max_samples=self._gen_max_text_samples(),
                            )
                            self._run_external_perplexity(
                                trainer,
                                epoch=epoch,
                                sampler_tag=tag,
                                samples_for_decode=seq,
                                split=None,
                            )
                        else:
                            seq_f = torch.zeros_like(seq, dtype=torch.float32)
                            seq_f[seq == 0] = 0.0
                            seq_f[seq == 1] = 1.0
                            if is_absorb and mask_id_cfg >= 0:
                                seq_f[seq == mask_id_cfg] = 0.5

                            imgs = seqs_to_images(seq_f, trainer)
                            grid = make_grid(imgs.cpu().clamp(0, 1), nrow=self._gen_grid_nrow(B_gen))
                            trainer.writer.add_image(f"generated_samples/{tag}", grid, epoch)
                    return

                # ============================= CONTINUOUS =============================
                seq_len = int(getattr(self.cfg.data, "sequence_len", 0))
                if seq_len <= 0:
                    raise ValueError(f"cfg.data.sequence_len must be > 0 for generation, got {seq_len}")

                entropic_blend_alpha = float(self._gen_entropic_blend_alpha())
                entropy_run_dir = self._entropy_run_dir()
                terminal_sigmas = self._gen_terminal_sigmas()

                # Conditioning setup
                split_specs = [(None, None, None, None)] # unconditional default
                if self._cond_enabled_cfg():
                    split_specs = []
                    for split in self._get_split_iter_order():
                        x_full = self._sample_full_sequences_from_split(trainer, B_gen, split=split)
                        if x_full is None:
                            print(f"⚠️ conditioning enabled but failed to fetch sequences from split='{split}', skipping.")
                            continue

                        # per-example prompt lengths
                        cL_bits = self._sample_cond_len_bits_per_example(B_gen, seq_len, device=trainer.device)
                        if int(cL_bits.max().item()) <= 0:
                            split_specs.append((split, None, None, None))
                            continue

                        prefix_mask = self._prefix_mask_from_lengths(cL_bits, seq_len)
                        prefix_full = x_full
                        fixed_cL = int(cL_bits[0].item())
                        all_equal = bool((cL_bits == fixed_cL).all().item())

                        split_specs.append((split, prefix_full, prefix_mask, fixed_cL if all_equal else None))

                    if not split_specs:
                        split_specs = [(None, None, None, None)]

                guidance_scales_default = self._gen_guidance_scales()

                # Helper to convert probs to bits (ECC aware)
                def _bits_from_probs(probs: torch.Tensor, prefix_full: Optional[torch.Tensor], prefix_mask: Optional[torch.Tensor]) -> torch.Tensor:
                    bits = (probs > 0.5).to(torch.long)
                    if prefix_full is not None and prefix_mask is not None:
                        bits[prefix_mask] = (prefix_full[prefix_mask] > 0.5).to(torch.long)
                    return bits

                def _float_bits_from_probs(probs: torch.Tensor, prefix_full: Optional[torch.Tensor], prefix_mask: Optional[torch.Tensor]) -> torch.Tensor:
                    bits_f = (probs > 0.5).to(torch.float32)
                    if prefix_full is not None and prefix_mask is not None:
                        bits_f[prefix_mask] = (prefix_full[prefix_mask] > 0.5).to(torch.float32)
                    return bits_f

                # --- Run generation per split / sigma / guidance / SAMPLER ---
                for split, prefix_full, prefix_mask, fixed_cL in split_specs:
                    do_cond = (prefix_full is not None and prefix_mask is not None)
                    guidance_scales = guidance_scales_default if do_cond else [0.0]

                    for s_term in terminal_sigmas:
                        for gs in guidance_scales:
                            
                            # Prepare conditioning kwargs
                            cond_kwargs = {}
                            if do_cond:
                                cond_kwargs = dict(
                                    conditioning_prefix_full=prefix_full,
                                    cond_prefix_mask=prefix_mask,
                                    guidance_scale=float(gs),
                                )
                                if fixed_cL is not None:
                                    cond_kwargs.update(
                                        conditioning_prefix=prefix_full[:, :fixed_cL],
                                        cond_len_bits=int(fixed_cL),
                                    )

                            # Iterate over configured samplers (Fixed: No longer hardcoded)
                            for sampler_name in target_sampler_names:
                                s_name = sampler_name.lower()
                                
                                # 1. Determine Implementation & Schedule
                                is_ddim = "ddim" in s_name
                                schedule_tag = "entropic" if "entropic" in s_name else "karras"
                                
                                if is_ddim:
                                    # Lazy init DDIM
                                    if self.ddim_sampler is None:
                                        from diffusion.continuous.samplers import DDIMSampler
                                        self.ddim_sampler = DDIMSampler(trainer.model, trainer.proc, self.cfg)
                                    sampler_obj = self.ddim_sampler
                                    algo_tag = "ddim"
                                else:
                                    # Default to Heun (trainer.sampler)
                                    sampler_obj = self.sampler
                                    algo_tag = "heun"

                                # 2. Run Sampling
                                _, probs = sampler_obj.sample(
                                    B_gen,
                                    seq_len,
                                    schedule=schedule_tag,
                                    num_steps=num_steps,
                                    entropic_blend_alpha=entropic_blend_alpha,
                                    entropy_run_dir=entropy_run_dir,
                                    sigma_min_override=s_term,
                                    return_probs=True,
                                    progress=bool(self._gen_progress_bar()),
                                    **cond_kwargs,
                                )

                                # 3. Logging
                                # Tag: e.g. "heun_karras_term-1.20_gs2.0"
                                base_tag = f"{algo_tag}_{schedule_tag}_term{math.log10(s_term):.2f}_gs{gs:g}"
                                
                                # Convert to bits
                                bits_out = _bits_from_probs(probs, prefix_full, prefix_mask)
                                
                                # Log ECC stats (if enabled)
                                self._maybe_log_ecc_stats(trainer, bits_out, epoch=epoch, tag=base_tag, split=split)

                                # Prefix for tensorboard
                                tag_full = f"{split}/generated_samples/{base_tag}" if split in {"train", "val"} else f"generated_samples/{base_tag}"

                                if is_text:
                                    log_text_sequences(
                                        trainer,
                                        bits_out[: self._gen_max_text_samples()],
                                        tag_full,
                                        epoch,
                                        max_samples=self._gen_max_text_samples(),
                                        prefix_bits=(prefix_full[: self._gen_max_text_samples()] if do_cond else None),
                                        prefix_mask=(prefix_mask[: self._gen_max_text_samples()] if do_cond else None),
                                        # prefix_len_bits is handled via mask logic inside format_decoded_block
                                    )
                                    self._run_external_perplexity(
                                        trainer,
                                        epoch=epoch,
                                        sampler_tag=base_tag,
                                        samples_for_decode=bits_out,
                                        split=split,
                                    )
                                else:
                                    # Image support
                                    bits_f = _float_bits_from_probs(probs, prefix_full, prefix_mask)
                                    imgs = seqs_to_images(bits_f, trainer)
                                    grid = make_grid(imgs.cpu().clamp(0, 1), nrow=self._gen_grid_nrow(B_gen))
                                    trainer.writer.add_image(tag_full, grid, epoch)

                if is_text:
                    print(f"✓ Logged continuous text samples ({', '.join(target_sampler_names)}).")

            finally:
                trainer.ema.restore(trainer.model)
