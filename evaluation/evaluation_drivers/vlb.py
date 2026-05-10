from evaluation.vlb import compute_vlb_over_loader
from evaluation.utils import resolve_eval_amp

def evaluate_vlb(args, cfg, model, train_loader, test_loader, device, rank0, run_meta, results_rows):
    if cfg.framework != "continuous_score":
        if rank0: print("\n⚠️  Skipping VLB: only implemented for continuous_score.")
        return

    if rank0: print("\n── Calculating VLB bound (NLL upper bound) ─────────────────────")

    use_amp, amp_dtype = resolve_eval_amp(device)
    vlb_cfg_eval = getattr(getattr(cfg, "evaluation", object()), "vlb", None)
    vlb_cfg_train = getattr(getattr(cfg, "train", object()), "vlb", None)

    def _get(name, default):
        if vlb_cfg_eval is not None and getattr(vlb_cfg_eval, name, None) is not None: return getattr(vlb_cfg_eval, name)
        if vlb_cfg_train is not None and getattr(vlb_cfg_train, name, None) is not None: return getattr(vlb_cfg_train, name)
        return default

    sigma_sampling = args.vlb_sigma_sampling or str(_get("sigma_sampling", "log-uniform"))
    sigma_min_eval = args.vlb_sigma_min if args.vlb_sigma_min is not None else _get("sigma_min_eval", None)
    sigma_max_eval = args.vlb_sigma_max if args.vlb_sigma_max is not None else _get("sigma_max_eval", None)
    K = args.vlb_mc if args.vlb_mc is not None else int(_get("num_mc_samples_per_batch", 1))
    include_prior = bool(_get("include_prior", False)) or bool(args.vlb_include_prior)

    split_loaders = {"train": train_loader, "test": test_loader}

    for split in args.vlb_splits:
        loader = split_loaders.get(split)
        if loader is None:
            if rank0: print(f"⚠️  VLB[{split}] requested but loader is not loaded.")
            continue

        res = compute_vlb_over_loader(
            model=model, cfg=cfg, data_loader=loader, device=device,
            sigma_min_eval=sigma_min_eval, sigma_max_eval=sigma_max_eval, sigma_sampling=sigma_sampling,
            num_mc_samples_per_batch=K, include_prior=include_prior, use_amp=use_amp, amp_dtype=amp_dtype,
            max_batches=args.max_batches, progress=rank0, allow_conditional_clean_prefix=True,
        )

        if rank0:
            print(f"✅ VLB[{split}]  vlb_bpd={res.vlb_bpd:.4f}  (recon={res.recon_bpd:.4f}, diff={res.diff_bpd:.4f}, prior={res.prior_bpd:.4f})  "
                  f"S={res.S_dim}  K={res.K}  dist={res.sigma_sampling}  [σmin={res.sigma_min_eval:g}, σmax={res.sigma_max_eval:g}]")
            results_rows.append(
                dict(**run_meta, metric="vlb_bpd", split=split, value=float(res.vlb_bpd),
                     details=f"recon={res.recon_bpd:.4f},diff={res.diff_bpd:.4f},prior={res.prior_bpd:.4f},K={res.K},dist={res.sigma_sampling},sigma_min={res.sigma_min_eval},sigma_max={res.sigma_max_eval}")
            )