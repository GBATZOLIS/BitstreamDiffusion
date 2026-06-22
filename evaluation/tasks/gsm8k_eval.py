"""GSM8K executable-code evaluation for CoBit (S-FLM parity).

For each GSM8K test problem: condition on the clean prompt ([BOS] question \\n),
sample the 512-token bitstream solution, decode the generated suffix back to
text with the SmolLM tokenizer, execute the Python program in the restricted
sandbox, and compare the returned number to the gold answer (#### N).

Primary metric: one-sample executable-code accuracy over the 1319 test
problems, with a percentile bootstrap 95% CI (matches S-FLM main.py).

Usage:
    python -m evaluation.tasks.gsm8k_eval \
        --config configs/tasks/tinygsm_bits.py \
        --checkpoint runs/.../checkpoints/step=000250000.pt \
        --sampler stochastic --gamma 0.0 --steps 1024 --limit 1319
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.tinygsm import GSM8KTestDataset
from data.task_codec import bits_to_token_ids
from evaluation.tasks._task_common import (
    load_config, load_model_and_sampler, configure_stochastic, sample_bits,
    resolve_sigma_data,
)
from evaluation.tasks.sandbox_gsm8k import evaluate_samples


def bootstrap_ci(correct: np.ndarray, n_boot: int, seed: int = 0):
    n = len(correct)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n_boot <= 1:
        acc = float(correct.mean())
        return acc, acc, acc
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = correct[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(means.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sampler", default="stochastic", choices=["stochastic", "deterministic"],
                    help="stochastic => EDM-style churn (needs gamma>0); deterministic => no churn")
    ap.add_argument("--sampler_kind", default="ddim", choices=["ddim", "heun", "em", "pc"],
                    help="ddim = CoBit ddim_entropic headline path (EDM churn, capped at gamma<=sqrt(2)-1); "
                         "heun = 2nd-order ablation; em = Euler-Maruyama entropy-gated reverse SDE "
                         "(stochasticity via lambda_zero, NO churn cap -> exceed the EDM ceiling); "
                         "pc = predictor-corrector entropy-gated SDE.")
    ap.add_argument("--schedule", default="entropic", choices=["entropic", "karras"])
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--guidance_scale", type=float, default=0.0,
                    help="Classifier-free guidance weight w (probs_u + w*(probs_c-probs_u)). "
                         "0 => no guidance. Requires a checkpoint trained with cond dropout.")
    ap.add_argument("--ema", type=int, default=1, help="1=EMA weights (headline), 0=raw weights")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma_min", type=float, default=None)
    ap.add_argument("--sigma_max", type=float, default=None,
                    help="Override the reverse-integration START sigma (default: cfg sigma_max=80). "
                         "Cap below the undertrained/collapsed high-sigma band to test that hypothesis.")
    ap.add_argument("--sigma_data", type=float, default=None,
                    help="Override EDM preconditioning sigma_data used at sampling "
                         "(feeds c_in=1/sqrt(sigma^2+sigma_data^2) in the denoiser). "
                         "Default: config value (0.5). The value the model was TRAINED "
                         "with is the SigmaDataEstimator estimate (~0.40); pass it here "
                         "to test train/eval-matched preconditioning.")
    ap.add_argument("--lambda_zero", type=float, default=0.0,
                    help="EM/PC entropy-gated SDE Langevin strength lambda_0 (>=0). 0 => deterministic "
                         "(EM bit-identical to DDIM det). With lambda_normalize=as_saved, lambda_0 is the "
                         "EDM-equivalent cumulative churn S_churn=gamma*(NFE-1); the EDM cap gamma<=sqrt(2)-1 "
                         "corresponds to lambda_0<=0.4142*(NFE-1) -- EM can go ABOVE this.")
    ap.add_argument("--lambda_profile", default="entropy_rate", choices=["entropy_rate", "flat"])
    ap.add_argument("--lambda_normalize", default="as_saved", choices=["as_saved", "peak"],
                    help="as_saved: lambda_0 = S_churn anchor (use for EDM-equivalence + above-cap sweeps).")
    ap.add_argument("--guidance_mode", default="predictor_only", choices=["predictor_only", "all"],
                    help="PC only: predictor_only guides PF predictor, corrector uses conditional score.")
    # Posterior temperature: continuous analogue of MDLM/Duo low-T / S-FLM top-k=1.
    # T<1 sharpens the per-bit Bernoulli posterior sigmoid(logit/T) toward 0/1.
    ap.add_argument("--posterior_temp", type=float, default=1.0,
                    help="Bit-posterior temperature T (<1 sharpens; 1.0 = no-op).")
    ap.add_argument("--posterior_temp_target", default="learned", choices=["learned", "full"],
                    help="learned: temper only the network logit, leave matched-filter at T=1 (recommended). "
                         "full: temper the whole postprocessed logit.")
    ap.add_argument("--posterior_temp_schedule", default="const", choices=["const", "sigma_ramp"],
                    help="const: T everywhere. sigma_ramp: T=1 above sigma_hi -> T at/below sigma_lo (log-interp).")
    ap.add_argument("--posterior_temp_sigma_lo", type=float, default=0.1,
                    help="sigma_ramp lower edge: at/below this sigma, full temperature T applies.")
    ap.add_argument("--posterior_temp_sigma_hi", type=float, default=4.0,
                    help="sigma_ramp upper edge: at/above this sigma, T=1 (untempered, protects diversity).")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    steps = int(args.steps or getattr(cfg.evaluation, "num_sampling_steps", 1024))
    timeout_s = float(getattr(getattr(cfg.evaluation, "gsm8k", object()), "timeout_s", 5.0))
    n_boot = int(getattr(getattr(cfg.evaluation, "gsm8k", object()), "bootstrap_size", 10000))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.checkpoint).resolve().parent.parent
    out_dir = Path(args.out_dir or (run_dir / "gsm8k_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the sigma_data the model was TRAINED with (sidecar) unless overridden.
    sigma_data_used, _ = resolve_sigma_data(cfg, run_dir, args.sigma_data)

    model, sampler = load_model_and_sampler(
        cfg, args.checkpoint, device, apply_ema=bool(args.ema), sampler_kind=args.sampler_kind,
        lambda_zero=args.lambda_zero, lambda_profile=args.lambda_profile,
        lambda_normalize=args.lambda_normalize, guidance_mode=args.guidance_mode)
    schedule = args.schedule
    configure_stochastic(cfg, mode=args.sampler, gamma=args.gamma, num_steps=steps)

    ds = GSM8KTestDataset(cfg)
    tok = ds.tok
    bpt = ds.bits_per_token
    tok_len = len(tok)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))

    per_correct = []
    n_invalid_tok = 0
    n_gen_tokens = 0
    records = []

    for start in range(0, n, args.batch_size):
        idxs = list(range(start, min(start + args.batch_size, n)))
        x0 = torch.stack([ds[i]["x0"] for i in idxs]).float().to(device)
        pm = torch.stack([ds[i]["prefix_mask"] for i in idxs]).to(device)
        plens = [int(ds[i]["prompt_len_tokens"]) for i in idxs]

        bits = sample_bits(
            cfg, sampler, prefix_full=x0, prefix_mask=pm, num_steps=steps,
            schedule=schedule, entropy_run_dir=str(run_dir),
            sigma_min_override=args.sigma_min, sigma_max_override=args.sigma_max,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            posterior_temp=args.posterior_temp,
            posterior_temp_target=args.posterior_temp_target,
            posterior_temp_schedule=args.posterior_temp_schedule,
            posterior_temp_sigma_lo=args.posterior_temp_sigma_lo,
            posterior_temp_sigma_hi=args.posterior_temp_sigma_hi,
        )
        gen_ids = bits_to_token_ids(bits, bpt)  # [B,512]

        for b, gi in enumerate(idxs):
            ids_row = gen_ids[b].tolist()
            suffix_ids = ids_row[plens[b]:]
            # Track decoded ids that fall outside the tokenizer vocab.
            n_gen_tokens += len(suffix_ids)
            n_invalid_tok += sum(1 for t in suffix_ids if t >= tok_len)
            # Clamp out-of-range ids so the tokenizer can decode.
            safe = [t if 0 <= t < tok_len else tok.eos_token_id for t in suffix_ids]
            text = tok.decode(safe, skip_special_tokens=True)

            rec = ds[gi]
            ok = evaluate_samples(text, rec["response_ground_truth"], timeout_s)
            per_correct.append(1 if ok else 0)
            if len(records) < 100:
                records.append({
                    "idx": gi,
                    "prompt": rec["prompt"][:200],
                    "response": text[:400],
                    "correct": bool(ok),
                })

        done = min(start + args.batch_size, n)
        acc_so_far = sum(per_correct) / max(1, done)
        print(f"[gsm8k] {done}/{n}  acc={100.0 * acc_so_far:.2f}%", flush=True)

    correct = np.asarray(per_correct, dtype=np.float64)
    acc, lo, hi = bootstrap_ci(correct, n_boot, seed=args.seed)

    result = {
        "task": "gsm8k",
        "checkpoint": str(args.checkpoint),
        "sampler": args.sampler,
        "gamma": args.gamma,
        "guidance_scale": args.guidance_scale,
        "sampler_kind": args.sampler_kind,
        "lambda_zero": args.lambda_zero,
        "lambda_profile": args.lambda_profile,
        "lambda_normalize": args.lambda_normalize,
        "guidance_mode": args.guidance_mode,
        "posterior_temp": args.posterior_temp,
        "posterior_temp_target": args.posterior_temp_target,
        "posterior_temp_schedule": args.posterior_temp_schedule,
        "posterior_temp_sigma_lo": args.posterior_temp_sigma_lo,
        "posterior_temp_sigma_hi": args.posterior_temp_sigma_hi,
        "steps": steps,
        "sigma_data": sigma_data_used,
        "num_examples": int(n),
        "accuracy": float(correct.mean()) if n else 0.0,
        "bootstrap_accuracy": acc,
        "ci95_low": lo,
        "ci95_high": hi,
        "num_correct": int(correct.sum()),
        "invalid_token_rate": n_invalid_tok / max(1, n_gen_tokens),
        "timeout_s": timeout_s,
        "sample_records": records,
    }
    tag = f"{args.sampler}_g{args.gamma}_w{args.guidance_scale}_s{steps}_sd{sigma_data_used:.4f}_ema{int(bool(args.ema))}"
    if abs(float(args.posterior_temp) - 1.0) > 1e-8:
        tag += f"_pt{args.posterior_temp:g}_{args.posterior_temp_target}_{args.posterior_temp_schedule}"
        if args.posterior_temp_schedule == "sigma_ramp":
            tag += f"_lo{args.posterior_temp_sigma_lo:g}_hi{args.posterior_temp_sigma_hi:g}"
    if args.sigma_max is not None:
        tag += f"_smax{args.sigma_max:g}"
    if args.sigma_min is not None:
        tag += f"_smin{args.sigma_min:g}"
    if args.sampler_kind != "ddim":
        tag += f"_kind{args.sampler_kind}_lz{args.lambda_zero:g}_{args.lambda_normalize}"
        if args.sampler_kind == "pc":
            tag += f"_{args.guidance_mode}"
    out_path = out_dir / f"gsm8k_results_{tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("\n=== GSM8K RESULT ===")
    print(json.dumps({k: v for k, v in result.items() if k != "sample_records"}, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
