import math
import torch
from tqdm import tqdm

from diffusion.continuous.processes import ContinuousForwardProcess
from diffusion.continuous.likelihood import evaluate_bpc_over_loader
from diffusion.discrete.processes import DiscreteForwardProcess
from diffusion.discrete.likelihood import bits_per_dim_dwdse

def evaluate_likelihood(args, cfg, model, test_loader, device, rank0, run_meta, results_rows, is_text_dataset):
    print("\n── Calculating Likelihood ────────────────────────────────")

    if cfg.framework == "continuous_score":
        proc = ContinuousForwardProcess(cfg)
        if args.sigma_min is not None:
            print(f"⚡ Overriding sigma_min: {cfg.diffusion.continuous.sigma_min} -> {args.sigma_min}")

        mean_bpc, mean_nll_nats_dim = evaluate_bpc_over_loader(
            model=model,
            forward_process=proc,
            cfg=cfg,
            data_loader=test_loader,
            steps=args.steps,
            num_probes=args.hutchinson,
            mc_samples=args.mc_samples,
            max_batches=args.max_batches,
            progress=True,
            schedule=args.schedule,
            sigma_min=args.sigma_min,
        )

        bits_per_dim = mean_nll_nats_dim / math.log(2.0)

        if is_text_dataset:
            print(f"\n✅ Final Results (Text):")
            print(f"   Schedule:   {args.schedule}")
            print(f"   Steps:      {args.steps}")
            print(f"   Sigma Min:  {args.sigma_min if args.sigma_min is not None else 'Config'}")
            print(f"   BPC:        {mean_bpc:.4f}")
            results_rows.append(
                dict(**run_meta, metric="bpc", split="test", value=float(mean_bpc),
                     details=f"schedule={args.schedule},steps={args.steps},sigma_min={args.sigma_min}")
            )
        else:
            print(f"\n✅ Final Results (Image):")
            print(f"   bits/dim (from ELBO NLL): {bits_per_dim:.4f} [σ_min = {args.sigma_min if args.sigma_min is not None else cfg.diffusion.continuous.sigma_min}]")
            results_rows.append(
                dict(**run_meta, metric="bpd_elbo", split="test", value=float(bits_per_dim),
                     details=f"schedule={args.schedule},steps={args.steps},sigma_min={args.sigma_min}")
            )

    elif cfg.framework == "discrete_sedd":
        print("Running Discrete DWDSE evaluation...")
        proc = DiscreteForwardProcess(cfg)
        vals = []
        iterator = tqdm(test_loader, desc="BPD (DWDSE bound)", leave=True)
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                if args.max_batches is not None and i >= args.max_batches: break
                if isinstance(batch, (tuple, list)): batch = batch[0]
                x = batch.to(device).view(batch.size(0), -1).long()
                bpd_batch = bits_per_dim_dwdse(model, proc, cfg, x, time_samples=args.steps)
                vals.append(bpd_batch)
                iterator.set_postfix(avg_bpd=f"{sum(vals) / len(vals):.4f}")

        bpd = sum(vals) / len(vals)
        print(f"✅ Discrete bits/dim (DWDSE upper bound): {bpd:.4f}")
        results_rows.append(
            dict(**run_meta, metric="bpd_dwdse", split="test", value=float(bpd), details=f"time_samples={args.steps}")
        )
    else:
        raise ValueError(f"Unknown framework: {cfg.framework}")