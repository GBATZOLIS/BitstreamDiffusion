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
    ap.add_argument("--sampler_kind", default="ddim", choices=["ddim", "heun"],
                    help="ddim = CoBit ddim_entropic headline path (EDM churn); heun = 2nd-order ablation")
    ap.add_argument("--schedule", default="entropic", choices=["entropic", "karras"])
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--ema", type=int, default=1, help="1=EMA weights (headline), 0=raw weights")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma_min", type=float, default=None)
    ap.add_argument("--sigma_data", type=float, default=None,
                    help="Override EDM preconditioning sigma_data used at sampling "
                         "(feeds c_in=1/sqrt(sigma^2+sigma_data^2) in the denoiser). "
                         "Default: config value (0.5). The value the model was TRAINED "
                         "with is the SigmaDataEstimator estimate (~0.40); pass it here "
                         "to test train/eval-matched preconditioning.")
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
        cfg, args.checkpoint, device, apply_ema=bool(args.ema), sampler_kind=args.sampler_kind)
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
            sigma_min_override=args.sigma_min, seed=args.seed,
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
    tag = f"{args.sampler}_g{args.gamma}_s{steps}_sd{sigma_data_used:.4f}_ema{int(bool(args.ema))}"
    out_path = out_dir / f"gsm8k_results_{tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("\n=== GSM8K RESULT ===")
    print(json.dumps({k: v for k, v in result.items() if k != "sample_records"}, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
