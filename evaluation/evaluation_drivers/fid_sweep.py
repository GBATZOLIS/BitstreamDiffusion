import math
import torch
from pathlib import Path
from torchvision.utils import make_grid, save_image

from evaluation.utils import get_fid_cfg, resolve_entropy_run_dir, _eval_cond_enabled, _eval_cond_len_bits, resolve_eval_amp, _eval_sample_prefixes_from_loader, seqs_to_images_for_eval
from diffusion.continuous.processes import ContinuousForwardProcess
from evaluation.distributed import barrier

def evaluate_fid_sweep(args, cfg, model, test_loader, device, rank0, ddp_active, samples_dir, is_text_dataset):
    if is_text_dataset:
        if rank0: print("\n⚠️  Skipping fid_sweep for text dataset.")
        return

    if rank0: print("\n── FID SWEEP (smoke test): one batch per sigma_decode ───────────")

    fid_cfg = get_fid_cfg(cfg)
    if fid_cfg is None:
        class _Empty: pass
        fid_cfg = _Empty()

    sweep_cfg = getattr(fid_cfg, "sweep", None)
    if sweep_cfg is None or not bool(getattr(sweep_cfg, "enabled", True)):
        if rank0: print("⚠️  fid_sweep requested but cfg.evaluation.fid.sweep is missing/disabled.")
        return
        
    sigmas = list(getattr(sweep_cfg, "sigmas", []))
    if len(sigmas) == 0: raise ValueError("fid_sweep: cfg.evaluation.fid.sweep.sigmas must be a non-empty list")

    fid_batch_default = int(getattr(fid_cfg, "batch_size", getattr(cfg.evaluation, "batch_size", 64)))
    sweep_batch = getattr(sweep_cfg, "batch_size", None)
    sweep_batch = int(sweep_batch) if sweep_batch is not None else fid_batch_default

    fid_steps = int(getattr(fid_cfg, "num_sampling_steps", getattr(cfg.evaluation, "num_sampling_steps", 256)))
    fid_entropic_blend_alpha = float(getattr(fid_cfg, "entropic_blend_alpha", getattr(cfg.evaluation, "entropic_blend_alpha", 0.0)))
    fid_sampler_name = str(getattr(fid_cfg, "sampler", getattr(cfg.evaluation, "sampler", "heun"))).lower()

    fid_decode_default = str(getattr(fid_cfg, "decode", "threshold")).lower()
    sweep_decode = getattr(sweep_cfg, "decode", None)
    sweep_decode = str(sweep_decode).lower() if sweep_decode is not None else fid_decode_default
    if sweep_decode not in {"threshold", "bernoulli"}:
        raise ValueError(f"fid_sweep decode must be 'threshold' or 'bernoulli' (got {sweep_decode})")

    preview_both = bool(getattr(sweep_cfg, "preview_both_decodes", False))
    sigma_max_override = float(getattr(sweep_cfg, "sigma_max_override", getattr(fid_cfg, "sigma_max_override", None) or 0.0))
    if sigma_max_override <= 0: raise ValueError("fid_sweep: sigma_max_override must be > 0 (e.g. 20.0)")

    entropy_run_dir = resolve_entropy_run_dir(cfg, fid_cfg)
    ds = test_loader.dataset
    C = int(getattr(ds, "channels", getattr(cfg.data, "channels", 3)))
    shape_hw = getattr(ds, "shape_hw", None)
    if shape_hw is None:
        H, W = int(getattr(cfg.data, "height", 32)), int(getattr(cfg.data, "width", 32))
    else:
        H, W = int(shape_hw[0]), int(shape_hw[1])

    seq_len = int(cfg.data.sequence_len)
    cond_enabled = _eval_cond_enabled(cfg)
    cL_base = _eval_cond_len_bits(cfg, seq_len) if cond_enabled else 0
    fid_gs = float(getattr(fid_cfg, "guidance_scale", getattr(cfg.evaluation, "guidance_scale", 0.0)))

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
        else: raise ValueError(f"Unknown FID sampler '{fid_sampler_name}' for fid_sweep.")
    else:
        raise ValueError("fid_sweep currently only supported for continuous_score (your use-case).")

    use_amp, amp_dtype = resolve_eval_amp(device)
    if rank0:
        print(f"   sampler={fid_sampler_name}  steps={fid_steps}  batch={sweep_batch}  sigma_max={sigma_max_override}")
        print(f"   decode={sweep_decode}  schedule={forced_schedule}  amp={use_amp} dtype={amp_dtype}")
        if cond_enabled and cL_base > 0: print(f"   conditioning=True cL={cL_base} gs={fid_gs}")
        else: print("   conditioning=False")

    sweep_dir = samples_dir / "fid_sweep"
    if rank0: sweep_dir.mkdir(parents=True, exist_ok=True)

    def _decode_probs_to_bits(probs: torch.Tensor, *, decode: str, prefix_bits, cL_bits: int) -> torch.Tensor:
        if decode == "threshold": bits = (probs > 0.5).to(torch.long)
        elif decode == "bernoulli": bits = torch.bernoulli(probs.clamp(0, 1)).to(torch.long)
        else: raise ValueError(f"decode must be 'threshold' or 'bernoulli' (got {decode})")
        if prefix_bits is not None and cL_bits > 0: bits[:, :cL_bits] = (prefix_bits[:, :cL_bits] > 0.5).to(torch.long)
        return bits

    save_bits = bool(getattr(sweep_cfg, "save_bits", False))

    if ddp_active and not rank0:
        barrier()
    else:
        for smin in sigmas:
            smin = float(smin)
            if rank0: print(f"\n[fid_sweep] sigma_decode={smin:g}, sigma_max_override={sigma_max_override:g}")

            B = int(sweep_batch)
            prefix, cL_eff = None, 0
            if cond_enabled and cL_base > 0:
                prefix = _eval_sample_prefixes_from_loader(test_loader, num_samples=B, cL=cL_base, device=device)
                if prefix is not None: cL_eff = cL_base

            cond_kwargs = {}
            if prefix is not None and cL_eff > 0:
                cond_kwargs = dict(conditioning_prefix=prefix, cond_len_bits=cL_eff, guidance_scale=float(fid_gs))

            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                _x, probs = sampler.sample(
                    B, seq_len, entropy_run_dir=Path(str(entropy_run_dir)) if entropy_run_dir is not None else None,
                    schedule=forced_schedule, num_steps=int(fid_steps), entropic_blend_alpha=float(fid_entropic_blend_alpha),
                    sigma_min_override=float(smin), sigma_max_override=float(sigma_max_override),
                    return_probs=True, progress=False, **cond_kwargs,
                )

            probs = probs.to(device=device, dtype=torch.float32)
            bits = _decode_probs_to_bits(probs, decode=sweep_decode, prefix_bits=prefix, cL_bits=cL_eff)
            imgs = seqs_to_images_for_eval(bits, ds).detach().cpu().clamp(0, 1)

            nrow = int(math.sqrt(imgs.size(0))) or 8
            out_png = sweep_dir / f"sigmaDecode_{smin:g}_sigmaMax_{sigma_max_override:g}_{fid_sampler_name}_{sweep_decode}.png"
            save_image(make_grid(imgs, nrow=nrow), out_png)
            if rank0: print(f"✓ saved: {out_png}")

            if preview_both:
                bits_thr = _decode_probs_to_bits(probs, decode="threshold", prefix_bits=prefix, cL_bits=cL_eff)
                imgs_thr = seqs_to_images_for_eval(bits_thr, ds).detach().cpu().clamp(0, 1)
                save_image(make_grid(imgs_thr, nrow=nrow), sweep_dir / f"sigmaDecode_{smin:g}_sigmaMax_{sigma_max_override:g}_{fid_sampler_name}_threshold.png")

                bits_ber = _decode_probs_to_bits(probs, decode="bernoulli", prefix_bits=prefix, cL_bits=cL_eff)
                imgs_ber = seqs_to_images_for_eval(bits_ber, ds).detach().cpu().clamp(0, 1)
                save_image(make_grid(imgs_ber, nrow=nrow), sweep_dir / f"sigmaDecode_{smin:g}_sigmaMax_{sigma_max_override:g}_{fid_sampler_name}_bernoulli.png")

            if save_bits:
                out_pt = sweep_dir / f"bits_sigmaDecode_{smin:g}_sigmaMax_{sigma_max_override:g}_{fid_sampler_name}_{sweep_decode}.pt"
                torch.save(bits.cpu(), out_pt)
                if rank0: print(f"✓ saved bits: {out_pt}")
        barrier()