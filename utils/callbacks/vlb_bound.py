from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Any, Dict, Tuple
import math

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from evaluation.vlb import compute_vlb_over_loader
from utils.model_utils import unwrap_model


def _ddp_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_master(trainer) -> bool:
    return bool(getattr(trainer, "is_master", True))


class VLBBoundCallback:
    run_on_all_ranks: bool = True

    def __init__(
        self,
        *,
        every_k_epochs: int = 10,
        sigma_min_eval: Optional[float] = None,
        sigma_max_eval: Optional[float] = None,
        sigma_sampling: str = "log-uniform",
        sigma_mc_dist: Optional[str] = None,
        num_mc_samples_per_batch: int = 1,
        include_prior: bool = False,
        max_batches: Optional[int] = None,
        progress: bool = False,
        use_amp: bool = True,
        amp_dtype: str = "auto",
        allow_conditional_clean_prefix: bool = True,
        force_unconditional_path: bool = False,
        debug_integrand: bool = False,
        debug_first_n_batches: int = 1,
        debug_num_sigma_bins: int = 6,
        debug_compare_null_prefix: bool = True,
        debug_compare_noise_prefix: bool = True,
        null_prefix_value: float = 0.0,
        null_prefix_mode: str = "constant",
    ):
        self.every_k_epochs = int(every_k_epochs)
        self.sigma_min_eval = sigma_min_eval
        self.sigma_max_eval = sigma_max_eval

        if sigma_mc_dist is not None:
            sigma_sampling = sigma_mc_dist
        self.sigma_sampling = str(sigma_sampling)

        self.num_mc_samples_per_batch = int(num_mc_samples_per_batch)
        self.include_prior = bool(include_prior)
        self.max_batches = max_batches
        self.progress = bool(progress)

        self.use_amp = bool(use_amp)
        self.amp_dtype = str(amp_dtype).lower().strip()

        self.allow_conditional_clean_prefix = bool(allow_conditional_clean_prefix)
        self.force_unconditional_path = bool(force_unconditional_path)

        self.debug_integrand = bool(debug_integrand)
        self.debug_first_n_batches = int(debug_first_n_batches)
        self.debug_num_sigma_bins = int(debug_num_sigma_bins)
        self.debug_compare_null_prefix = bool(debug_compare_null_prefix)
        self.debug_compare_noise_prefix = bool(debug_compare_noise_prefix)
        self.null_prefix_value = float(null_prefix_value)
        self.null_prefix_mode = str(null_prefix_mode)

        self._cached_loaders: Dict[Tuple[Any, ...], DataLoader] = {}

    def _resolve_amp_dtype(self, trainer) -> torch.dtype:
        if self.amp_dtype in {"auto", ""}:
            return getattr(trainer, "amp_dtype", torch.float16)
        if self.amp_dtype in {"fp16", "float16"}:
            return torch.float16
        if self.amp_dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        return getattr(trainer, "amp_dtype", torch.float16)

    def _get_vlb_cfg(self, trainer):
        train_cfg = getattr(getattr(trainer, "cfg", None), "train", None)
        return getattr(train_cfg, "vlb", None) if train_cfg is not None else None

    def _get_splits(self, vlb_cfg) -> List[str]:
        if vlb_cfg is None:
            return ["train", "val"]
        splits = getattr(vlb_cfg, "splits", None)
        if splits is None:
            return ["train", "val"]
        return [str(s) for s in splits]

    def _get_split_loader(self, trainer, split: str):
        split = str(split)
        if split == "train":
            base_loader = trainer.train_loader
        elif split == "val":
            base_loader = trainer.val_loader
        else:
            raise ValueError(f"Unknown split: {split}")

        dataset = base_loader.dataset
        vlb_cfg = self._get_vlb_cfg(trainer)

        # Key patch: allow VLB to use its own much smaller eval batch size.
        vlb_batch_size = getattr(vlb_cfg, "batch_size", None) if vlb_cfg is not None else None
        if vlb_batch_size is None:
            batch_size = int(
                getattr(base_loader, "batch_size", getattr(trainer.cfg.train, "batch_size", 256))
            )
        else:
            batch_size = int(vlb_batch_size)

        num_workers = int(getattr(trainer.cfg.data, "num_workers", 0))
        pin_memory = bool(getattr(trainer.cfg.data, "pin_memory", False))
        prefetch_factor = int(getattr(trainer.cfg.data, "prefetch_factor", 2))
        persistent_workers = False

        if _ddp_is_on():
            world_size = int(getattr(trainer, "world_size", dist.get_world_size()))
            rank = int(getattr(trainer, "rank", dist.get_rank()))
        else:
            world_size = 1
            rank = 0

        cache_key = (
            "vlb_eval_loader",
            split,
            id(dataset),
            int(batch_size),
            int(num_workers),
            bool(pin_memory),
            int(prefetch_factor),
            bool(persistent_workers),
            bool(_ddp_is_on()),
            int(world_size),
            int(rank),
        )

        cached = self._cached_loaders.get(cache_key, None)
        if cached is not None:
            return cached

        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor

        if _ddp_is_on():
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=True,
            )
            loader = DataLoader(dataset, sampler=sampler, **loader_kwargs)
        else:
            loader = DataLoader(dataset, **loader_kwargs)

        self._cached_loaders[cache_key] = loader
        return loader

    @torch.no_grad()
    def on_epoch_end(self, trainer, epoch: int):
        if self.every_k_epochs <= 0:
            return
        if ((epoch + 1) % self.every_k_epochs) != 0:
            return

        vlb_cfg = self._get_vlb_cfg(trainer)

        if vlb_cfg is not None:
            self.sigma_sampling = str(getattr(vlb_cfg, "sigma_sampling", self.sigma_sampling))
            self.sigma_min_eval = getattr(vlb_cfg, "sigma_min_eval", self.sigma_min_eval)
            self.sigma_max_eval = getattr(vlb_cfg, "sigma_max_eval", self.sigma_max_eval)
            self.num_mc_samples_per_batch = int(
                getattr(vlb_cfg, "num_mc_samples_per_batch", self.num_mc_samples_per_batch)
            )
            self.include_prior = bool(getattr(vlb_cfg, "include_prior", self.include_prior))
            self.use_amp = bool(getattr(vlb_cfg, "use_amp", self.use_amp))
            self.progress = bool(getattr(vlb_cfg, "progress", self.progress))
            self.force_unconditional_path = bool(
                getattr(vlb_cfg, "force_unconditional_path", self.force_unconditional_path)
            )
            self.allow_conditional_clean_prefix = bool(
                getattr(vlb_cfg, "allow_conditional_clean_prefix", self.allow_conditional_clean_prefix)
            )
            self.debug_integrand = bool(getattr(vlb_cfg, "debug_integrand", self.debug_integrand))
            self.debug_first_n_batches = int(
                getattr(vlb_cfg, "debug_first_n_batches", self.debug_first_n_batches)
            )
            self.debug_num_sigma_bins = int(
                getattr(vlb_cfg, "debug_num_sigma_bins", self.debug_num_sigma_bins)
            )
            self.debug_compare_null_prefix = bool(
                getattr(vlb_cfg, "debug_compare_null_prefix", self.debug_compare_null_prefix)
            )
            self.debug_compare_noise_prefix = bool(
                getattr(vlb_cfg, "debug_compare_noise_prefix", self.debug_compare_noise_prefix)
            )
            self.null_prefix_value = float(getattr(vlb_cfg, "null_prefix_value", self.null_prefix_value))
            self.null_prefix_mode = str(getattr(vlb_cfg, "null_prefix_mode", self.null_prefix_mode))

        amp_dtype = self._resolve_amp_dtype(trainer)
        splits = self._get_splits(vlb_cfg)

        ema_applied = False
        if hasattr(trainer, "ema") and getattr(trainer, "ema", None) is not None:
            trainer.ema.apply(trainer.model)
            ema_applied = True

        try:
            for split in splits:
                split = str(split)
                if split not in {"train", "val"}:
                    continue

                max_batches = self.max_batches
                if vlb_cfg is not None:
                    if split == "train":
                        max_batches = getattr(vlb_cfg, "max_batches_train", max_batches)
                    else:
                        max_batches = getattr(vlb_cfg, "max_batches_val", max_batches)

                loader = self._get_split_loader(trainer, split)
                eval_model = unwrap_model(trainer.model)

                res = compute_vlb_over_loader(
                    model=eval_model,
                    cfg=trainer.cfg,
                    data_loader=loader,
                    device=trainer.device,
                    sigma_min_eval=self.sigma_min_eval,
                    sigma_max_eval=self.sigma_max_eval,
                    sigma_sampling=self.sigma_sampling,
                    num_mc_samples_per_batch=self.num_mc_samples_per_batch,
                    include_prior=self.include_prior,
                    use_amp=self.use_amp,
                    amp_dtype=amp_dtype,
                    max_batches=max_batches,
                    progress=self.progress and _is_master(trainer),
                    allow_conditional_clean_prefix=self.allow_conditional_clean_prefix,
                    force_unconditional_path=self.force_unconditional_path,
                    debug_integrand=self.debug_integrand,
                    debug_first_n_batches=self.debug_first_n_batches,
                    debug_num_sigma_bins=self.debug_num_sigma_bins,
                    debug_compare_null_prefix=self.debug_compare_null_prefix,
                    debug_compare_noise_prefix=self.debug_compare_noise_prefix,
                    null_prefix_value=self.null_prefix_value,
                    null_prefix_mode=self.null_prefix_mode,
                )

                if not _is_master(trainer):
                    continue

                step = int(getattr(trainer, "global_step", 0))
                writer = getattr(trainer, "writer", None)

                data_cfg = getattr(trainer.cfg, "data", None)
                repr_mode = str(getattr(data_cfg, "representation", "binary")).lower()
                bits_per_token = int(getattr(data_cfg, "bits_per_token", 15))
                ln2 = math.log(2.0)

                units_per_token = bits_per_token if repr_mode == "binary" else 1

                vlb_nats_per_token = float(res.vlb_bpd) * ln2 * units_per_token
                recon_nats_per_token = float(res.recon_bpd) * ln2 * units_per_token
                diff_nats_per_token = float(res.diff_bpd) * ln2 * units_per_token
                prior_nats_per_token = float(res.prior_bpd) * ln2 * units_per_token

                if writer is not None:
                    writer.add_scalar(f"VLB/{split}_bpd", res.vlb_bpd, step)
                    writer.add_scalar(f"VLB/{split}_recon_bpd", res.recon_bpd, step)
                    writer.add_scalar(f"VLB/{split}_diff_bpd", res.diff_bpd, step)
                    writer.add_scalar(f"VLB/{split}_prior_bpd", res.prior_bpd, step)

                    writer.add_scalar(f"VLB/{split}_nats_per_token", vlb_nats_per_token, step)
                    writer.add_scalar(f"VLB/{split}_recon_nats_per_token", recon_nats_per_token, step)
                    writer.add_scalar(f"VLB/{split}_diff_nats_per_token", diff_nats_per_token, step)
                    writer.add_scalar(f"VLB/{split}_prior_nats_per_token", prior_nats_per_token, step)

                    writer.add_scalar("VLB/units_per_token", units_per_token, step)
                    writer.add_scalar("VLB/bits_per_token_cfg", bits_per_token, step)
                    writer.add_scalar(
                        "VLB/is_binary_representation",
                        1.0 if repr_mode == "binary" else 0.0,
                        step,
                    )

                if hasattr(trainer, "_log_wandb"):
                    try:
                        trainer._log_wandb(
                            {
                                f"VLB/{split}_bpd": res.vlb_bpd,
                                f"VLB/{split}_recon_bpd": res.recon_bpd,
                                f"VLB/{split}_diff_bpd": res.diff_bpd,
                                f"VLB/{split}_prior_bpd": res.prior_bpd,
                                f"VLB/{split}_nats_per_token": vlb_nats_per_token,
                                f"VLB/{split}_recon_nats_per_token": recon_nats_per_token,
                                f"VLB/{split}_diff_nats_per_token": diff_nats_per_token,
                                f"VLB/{split}_prior_nats_per_token": prior_nats_per_token,
                                "VLB/units_per_token": units_per_token,
                                "VLB/bits_per_token_cfg": bits_per_token,
                                "VLB/is_binary_representation": 1.0 if repr_mode == "binary" else 0.0,
                                "VLB/sigma_min_eval": res.sigma_min_eval,
                                "VLB/sigma_max_eval": res.sigma_max_eval,
                                "VLB/K": res.K,
                                "VLB/mode": res.mode,
                                "VLB/split": split,
                            }
                        )
                    except Exception:
                        pass

                print(
                    f"[VLB/{split}] epoch={epoch} mode={res.mode} dist={res.sigma_sampling} "
                    f"repr={repr_mode} units_per_token={units_per_token} "
                    f"sigma_min_eval={res.sigma_min_eval:g} sigma_max_eval={res.sigma_max_eval:g} "
                    f"K={res.K} max_batches={max_batches} "
                    f"vlb_bpd={res.vlb_bpd:.4f} (recon={res.recon_bpd:.4f}, diff={res.diff_bpd:.4f}, prior={res.prior_bpd:.4f}) "
                    f"S={res.S_dim} N={res.num_examples}"
                )
        finally:
            if ema_applied:
                trainer.ema.restore(trainer.model)