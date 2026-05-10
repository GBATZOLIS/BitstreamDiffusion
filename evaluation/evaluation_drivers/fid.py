import math
import torch
from pathlib import Path
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from typing import List

from data import get_loader
from evaluation.fid import compute_fid_inception
from evaluation.utils import (
    get_fid_cfg,
    resolve_entropy_run_dir,
    _eval_cond_enabled,
    _eval_cond_len_bits,
    resolve_eval_amp,
    _eval_sample_prefixes_from_loader,
    seqs_to_images_for_eval,
    warn_if_small_fid_n,
    maybe_override_real_train_size,
    real_train_loader_for_fid,
    make_image_loader_from_loader,
    warmup_continuous_sampler,
    load_checkpoint,
)
from diffusion.continuous.processes import ContinuousForwardProcess
from diffusion.discrete.processes import DiscreteForwardProcess
from evaluation.distributed import barrier, shard_count, gather_varlen_firstdim_to_rank0

# Clean, safe import from our new utils file
from evaluation.evaluation_drivers.utils import _resolve_fid_checkpoints


def evaluate_fid(
    args, 
    cfg, 
    model, 
    ema, 
    use_ema, 
    test_loader, 
    train_loader, 
    device, 
    rank0, 
    ddp_active, 
    dist_info, 
    samples_dir, 
    fid_splits, 
    run_meta, 
    results_rows, 
    compile_enabled, 
    compile_warmup, 
    compile_warmup_steps, 
    is_text_dataset
):
    if is_text_dataset:
        if rank0:
            print("\n⚠️  Skipping FID for text dataset.")
        return

    if rank0:
        print("\n── Calculating FID ───────────────────────────────────────")

    fid_cfg = get_fid_cfg(cfg)
    if fid_cfg is None:

        class _Empty:  # pragma: no cover
            pass

        fid_cfg = _Empty()

    # -----------------------------
    # FID config
    # -----------------------------
    fid_total = int(getattr(fid_cfg, "num_samples", args.fid_samples))
    fid_batch = int(getattr(fid_cfg, "batch_size", getattr(cfg.evaluation, "batch_size", 64)))
    fid_steps = int(getattr(fid_cfg, "num_sampling_steps", getattr(cfg.evaluation, "num_sampling_steps", 256)))
    fid_entropic_blend_alpha = float(
        getattr(fid_cfg, "entropic_blend_alpha", getattr(cfg.evaluation, "entropic_blend_alpha", 0.0))
    )
    fid_sampler_name = str(getattr(fid_cfg, "sampler", getattr(cfg.evaluation, "sampler", "heun"))).lower()
    fid_sigma_decode = float(getattr(fid_cfg, "sigma_decode", cfg.diffusion.continuous.sigma_min))
    fid_sigma_max_override = getattr(fid_cfg, "sigma_max_override", None)
    fid_sigma_max_override = float(fid_sigma_max_override) if fid_sigma_max_override is not None else None

    if rank0 and fid_sigma_max_override is not None:
        print(f"   FID sigma_max override: {fid_sigma_max_override}")

    fid_decode = str(getattr(fid_cfg, "decode", "threshold")).lower()
    if fid_decode not in {"threshold", "bernoulli"}:
        raise ValueError(f"cfg.evaluation.fid.decode must be 'threshold' or 'bernoulli' (got {fid_decode})")

    preview_both = bool(getattr(fid_cfg, "preview_both_decodes", True))
    entropy_run_dir = resolve_entropy_run_dir(cfg, fid_cfg)

    if rank0:
        warn_if_small_fid_n(fid_total)
        if ddp_active:
            print(f"   DDP gen: True (world_size={dist_info.world_size})")
        print(f"   FID splits: {fid_splits}")

    # -----------------------------
    # Need test_loader for dataset shape + prefixes (if conditional)
    # -----------------------------
    if test_loader is None:
        raise RuntimeError("FID requested but test_loader is None. Ensure need_test includes 'fid'.")

    ds = test_loader.dataset
    C = int(getattr(ds, "channels", getattr(cfg.data, "channels", 3)))
    shape_hw = getattr(ds, "shape_hw", None)
    if shape_hw is None:
        H = int(getattr(cfg.data, "height", 32))
        W = int(getattr(cfg.data, "width", 32))
    else:
        H, W = int(shape_hw[0]), int(shape_hw[1])

    seq_len = int(cfg.data.sequence_len)

    # Conditioning policy (kept consistent)
    cond_enabled = _eval_cond_enabled(cfg)
    cL_base = _eval_cond_len_bits(cfg, seq_len) if cond_enabled else 0
    fid_gs = float(getattr(fid_cfg, "guidance_scale", getattr(cfg.evaluation, "guidance_scale", 0.0)))

    if rank0:
        if cond_enabled and cL_base > 0:
            print(f"   Conditioning: True (cL={cL_base} bits), guidance_scale={fid_gs}")
        else:
            print("   Conditioning: False")

    # -----------------------------
    # Build sampler once (it uses model reference; weights will change per-ckpt)
    # -----------------------------
    forced_schedule = None
    if cfg.framework == "continuous_score":
        proc = ContinuousForwardProcess(cfg)
        if fid_sampler_name in {"heun", "heun_karras", "karras"}:
            from diffusion.continuous.samplers import HeunSampler

            sampler = HeunSampler(model, proc, cfg)
            forced_schedule = "karras"
        elif fid_sampler_name in {"ddim", "ddim_entropic", "entropic"}:
            from diffusion.continuous.samplers import DDIMSampler

            sampler = DDIMSampler(model, proc, cfg)
            forced_schedule = "entropic"
        else:
            raise ValueError(
                f"Unknown FID sampler '{fid_sampler_name}'. Expected: heun_karras | ddim_entropic (aliases: heun/ddim)."
            )
    elif cfg.framework == "discrete_sedd":
        proc = DiscreteForwardProcess(cfg)
        from diffusion.discrete.samplers import TweedieTauLeapingSampler, EulerRateSampler

        if args.sampler.lower() in {"euler", "euler_rate"}:
            sampler = EulerRateSampler(model, proc, cfg)
        else:
            sampler = TweedieTauLeapingSampler(model, proc, cfg)
        forced_schedule = None
    else:
        raise ValueError(f"Unknown framework for FID: {cfg.framework}")

    save_dir = samples_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    use_amp, amp_dtype = resolve_eval_amp(device)
    if (not ddp_active) or rank0:
        print(f"[eval amp] use_amp={use_amp} dtype={amp_dtype}")

    def _decode_probs_to_bits(probs: torch.Tensor, *, decode: str, prefix_bits, cL_bits: int) -> torch.Tensor:
        if decode == "threshold":
            bits = (probs > 0.5).to(torch.long)
        elif decode == "bernoulli":
            bits = torch.bernoulli(probs.clamp(0, 1)).to(torch.long)
        else:
            raise ValueError(f"decode must be 'threshold' or 'bernoulli' (got {decode})")
        if prefix_bits is not None and cL_bits > 0:
            bits[:, :cL_bits] = (prefix_bits[:, :cL_bits] > 0.5).to(torch.long)
        return bits

    # -----------------------------
    # DDP sharding
    # -----------------------------
    fid_local = shard_count(fid_total, dist_info.world_size, dist_info.rank) if ddp_active else fid_total

    if ddp_active:
        import torch.distributed as dist

        t = torch.tensor([fid_local], device=device, dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        fid_local_max = int(t.item())
    else:
        fid_local_max = fid_local

    max_iters = int(math.ceil(fid_local_max / max(1, fid_batch))) if fid_local_max > 0 else 0

    # -----------------------------
    # Resolve checkpoints to evaluate
    # -----------------------------
    fid_ckpts = _resolve_fid_checkpoints(cfg, fid_cfg, args.fid_checkpoints)
    if rank0:
        print("   FID checkpoints to evaluate:")
        for p in fid_ckpts:
            print(f"     - {p}")

    # -----------------------------
    # Build real-data loaders once
    # -----------------------------
    cfg_real = maybe_override_real_train_size(cfg, fid_cfg)

    train_img_loader = None
    test_img_loader = None

    if rank0 and ("train" in fid_splits):
        real_train_loader = real_train_loader_for_fid(cfg_real, fid_cfg)
        train_img_loader = make_image_loader_from_loader(real_train_loader)

    if rank0 and ("test" in fid_splits):
        real_test_loader = get_loader(cfg_real, split="test") if (cfg_real is not cfg) else test_loader
        test_img_loader = make_image_loader_from_loader(real_test_loader)

    # -----------------------------
    # Optional compile warmup
    # -----------------------------
    if cfg.framework == "continuous_score" and fid_local_max > 0:
        Bw = min(fid_batch, max(1, fid_local))
        steps_w = min(int(compile_warmup_steps), fid_steps)
        if Bw > 0 and steps_w > 0 and compile_enabled and compile_warmup:
            if rank0:
                print(f"🔥 Warming up compiler (once) with B={Bw}, steps={steps_w}...")
            warmup_continuous_sampler(
                sampler,
                B=Bw,
                seq_len=seq_len,
                num_steps=steps_w,
                schedule=forced_schedule,
                entropy_run_dir=Path(str(entropy_run_dir)) if entropy_run_dir is not None else None,
                entropic_blend_alpha=float(fid_entropic_blend_alpha),
                sigma_min_override=float(fid_sigma_decode),
                sigma_max_override=float(fid_sigma_max_override) if fid_sigma_max_override is not None else None,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                cond_kwargs=None,
            )
        barrier()

    # -----------------------------
    # Evaluate each checkpoint
    # -----------------------------
    for ckpt_path in fid_ckpts:
        ckpt_path = Path(ckpt_path).expanduser()

        load_checkpoint(model, ema, ckpt_path, device, apply_ema=use_ema)
        model.eval()
        barrier()

        ckpt_tag = ckpt_path.stem.replace("=", "_").replace(" ", "_")
        if rank0:
            print(f"\n── FID @ checkpoint: {ckpt_path} ───────────────────────────────")

        preview_saved_main = False
        preview_saved_both = False
        gen_parts_cpu: List[torch.Tensor] = []
        remaining = int(fid_local)

        iterator = range(max_iters)
        if rank0:
            iterator = tqdm(
                iterator,
                desc=f"Generating FID Images [{ckpt_tag}]",
                total=max_iters,
                unit="batch",
                colour="green",
            )

        with torch.inference_mode():
            for _it in iterator:
                B = min(fid_batch, remaining) if remaining > 0 else 0
                remaining -= B

                if B > 0:
                    prefix = None
                    cL_eff = 0
                    if cfg.framework == "continuous_score" and cond_enabled and cL_base > 0:
                        prefix = _eval_sample_prefixes_from_loader(
                            test_loader, num_samples=B, cL=cL_base, device=device
                        )
                        if prefix is not None:
                            cL_eff = cL_base

                    cond_kwargs = {}
                    if prefix is not None and cL_eff > 0:
                        cond_kwargs = dict(
                            conditioning_prefix=prefix,
                            cond_len_bits=cL_eff,
                            guidance_scale=float(fid_gs),
                        )

                    if cfg.framework == "continuous_score":
                        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                            _x, probs = sampler.sample(
                                B,
                                seq_len,
                                entropy_run_dir=Path(str(entropy_run_dir)) if entropy_run_dir is not None else None,
                                schedule=forced_schedule,
                                num_steps=int(fid_steps),
                                entropic_blend_alpha=float(fid_entropic_blend_alpha),
                                sigma_min_override=float(fid_sigma_decode),
                                sigma_max_override=float(fid_sigma_max_override)
                                if fid_sigma_max_override is not None
                                else None,
                                return_probs=True,
                                progress=False,
                                **cond_kwargs,
                            )

                        probs = probs.to(device=device, dtype=torch.float32)
                        seq_bits = _decode_probs_to_bits(
                            probs, decode=fid_decode, prefix_bits=prefix, cL_bits=cL_eff
                        )

                        if rank0 and preview_both and (not preview_saved_both):
                            bits_thr = _decode_probs_to_bits(
                                probs, decode="threshold", prefix_bits=prefix, cL_bits=cL_eff
                            )
                            imgs_thr = seqs_to_images_for_eval(bits_thr, ds)

                            bits_ber = _decode_probs_to_bits(
                                probs, decode="bernoulli", prefix_bits=prefix, cL_bits=cL_eff
                            )
                            imgs_ber = seqs_to_images_for_eval(bits_ber, ds)

                            nrow = int(math.sqrt(imgs_thr.size(0))) or 8
                            save_image(
                                make_grid(imgs_thr.cpu().clamp(0, 1), nrow=nrow),
                                save_dir / f"fid_preview_{ckpt_tag}_threshold_{fid_sampler_name}.png",
                            )
                            save_image(
                                make_grid(imgs_ber.cpu().clamp(0, 1), nrow=nrow),
                                save_dir / f"fid_preview_{ckpt_tag}_bernoulli_{fid_sampler_name}.png",
                            )
                            preview_saved_both = True
                    else:
                        seq = sampler.sample(B, seq_len)
                        seq_bits = (seq == 1).to(torch.float32)

                    imgs = seqs_to_images_for_eval(seq_bits, ds)
                    imgs = imgs.to(device=device, dtype=torch.float32, non_blocking=True)

                    if rank0 and (not preview_saved_main):
                        nrow = int(math.sqrt(imgs.size(0))) or 8
                        out_path = save_dir / f"fid_preview_{ckpt_tag}_{fid_sampler_name}_{fid_decode}.png"
                        save_image(make_grid(imgs.detach().cpu().clamp(0, 1), nrow=nrow), out_path)
                        preview_saved_main = True

                    local_u8 = (imgs.clamp(0, 1) * 255.0).to(torch.uint8)
                else:
                    local_u8 = torch.empty((0, C, H, W), device=device, dtype=torch.uint8)

                if ddp_active:
                    gathered_u8 = gather_varlen_firstdim_to_rank0(local_u8, dst=0)
                    if rank0 and (gathered_u8 is not None) and gathered_u8.numel() > 0:
                        gen_parts_cpu.append(gathered_u8.detach().cpu())
                else:
                    if rank0 and local_u8.numel() > 0:
                        gen_parts_cpu.append(local_u8.detach().cpu())

            barrier()

        if rank0:
            if not gen_parts_cpu:
                raise RuntimeError(f"No images were generated for FID at checkpoint {ckpt_path}")

            gen_u8 = torch.cat(gen_parts_cpu, dim=0)
            if gen_u8.dim() != 4 or gen_u8.size(0) == 0:
                raise RuntimeError(f"Generated image tensor has invalid shape: {tuple(gen_u8.shape)}")

            gen_u8 = gen_u8[:fid_total].contiguous()
            gen_imgs = gen_u8.float() / 255.0

            run_meta_ckpt = {**run_meta, "checkpoint": str(ckpt_path)}

            if "train" in fid_splits:
                if train_img_loader is None:
                    raise RuntimeError("train split requested for FID but train_img_loader is None.")
                fid_train = compute_fid_inception(train_img_loader, gen_imgs, device)
                print(f"FID (Inception, vs TRAIN) [{ckpt_tag}]: {fid_train:.4f}")
                results_rows.append(
                    dict(
                        **run_meta_ckpt,
                        metric="fid",
                        split="train",
                        value=float(fid_train),
                        details=f"N={fid_total},batch={fid_batch},steps={fid_steps},sampler={fid_sampler_name},decode={fid_decode},sigma_decode={fid_sigma_decode},sigma_max_override={fid_sigma_max_override}",
                    )
                )

            if "test" in fid_splits:
                if test_img_loader is None:
                    raise RuntimeError("test split requested for FID but test_img_loader is None.")
                fid_test = compute_fid_inception(test_img_loader, gen_imgs, device)
                print(f"FID (Inception, vs TEST)  [{ckpt_tag}]: {fid_test:.4f}")
                results_rows.append(
                    dict(
                        **run_meta_ckpt,
                        metric="fid",
                        split="test",
                        value=float(fid_test),
                        details=f"N={fid_total},batch={fid_batch},steps={fid_steps},sampler={fid_sampler_name},decode={fid_decode},sigma_decode={fid_sigma_decode},sigma_max_override={fid_sigma_max_override}",
                    )
                )

        barrier()

        # Optional hygiene between checkpoints
        if device.type == "cuda":
            torch.cuda.empty_cache()