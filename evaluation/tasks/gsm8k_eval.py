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
import os
from pathlib import Path

import numpy as np
import torch

from data.tinygsm import GSM8KTestDataset
from data.task_codec import bits_to_token_ids
from evaluation.tasks._task_common import (
    load_config, load_model_and_sampler, configure_stochastic, sample_bits,
    resolve_sigma_data,
)
from evaluation.tasks.sandbox_gsm8k import (
    evaluate_samples, predict_answer, _extract_gold_answer, _numbers_equal,
)


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


def _run_fkc_gsm8k(cfg, sampler, ds, n, bpt, tok, tok_len, args, run_dir, out_dir,
                   sigma_data_used, steps, timeout_s, n_boot):
    """FKC particle evaluation on GSM8K: per-particle acc / pass@K / maj@K + SMC telemetry.

    Mirrors _run_fkc_sudoku, with one deliberate difference: Sudoku votes over the
    exact token suffix (the solution IS the answer), whereas here we vote over the
    EXECUTED answer, since many distinct programs return the same number. Voting
    over program text would under-count agreement and understate maj@K.

    Reported metrics:
      * particle_mean_accuracy : mean exact-execution accuracy over all B*K samples
                                 -- the like-for-like comparator to the single-sample
                                 number in the paper (25.4 / 27.5 / 29.2%)
      * pass_at_k              : any particle returns the gold answer (oracle ceiling)
      * maj_at_k               : self-consistency vote over executed answers
      * weighted_vote_accuracy : FKC-weight-weighted vote (informative only for beta>1)
    """
    from evaluation.tasks._task_common import sample_bit_particles
    from collections import Counter

    K = int(args.num_particles)
    print(f"[gsm8k-fkc] K={K} particles x batch_size={args.batch_size} "
          f"=> effective forward batch {K * args.batch_size} "
          f"(reduce --batch_size if this OOMs)", flush=True)

    n_pass = n_maj = n_wvote = 0
    n_topw = n_botw = 0
    part_correct = part_total = 0
    n_answered = 0                 # particles that executed to a number at all
    distinct_sum = 0
    min_ess = float("inf")
    total_resamples = 0
    uniq_anc = []
    n_invalid_tok = n_gen_tokens = 0
    mean_w_correct = mean_w_incorrect = 0.0
    nw_correct = nw_incorrect = 0
    per_prompt_maj = []            # for bootstrap CI
    per_prompt_particle_mean = []
    records = []

    for start in range(0, n, args.batch_size):
        idxs = list(range(start, min(start + args.batch_size, n)))
        Bc = len(idxs)
        x0 = torch.stack([ds[i]["x0"] for i in idxs]).float().to(sampler.device)
        pm = torch.stack([ds[i]["prefix_mask"] for i in idxs]).to(sampler.device)
        plens = [int(ds[i]["prompt_len_tokens"]) for i in idxs]

        out = sample_bit_particles(
            cfg, sampler, prefix_full=x0, prefix_mask=pm, num_steps=steps,
            schedule=args.schedule, entropy_run_dir=str(run_dir),
            sigma_min_override=args.sigma_min, seed=args.seed,
            guidance_scale=args.guidance_scale,
        )
        S = x0.shape[1]
        gen_ids = bits_to_token_ids(out.bits.reshape(Bc * K, S), bpt).reshape(Bc, K, -1)
        gen_ids_pre = bits_to_token_ids(
            out.pre_resample_bits.reshape(Bc * K, S), bpt).reshape(Bc, K, -1)
        w_norm = torch.softmax(out.log_weights_final, dim=1).cpu()      # [B,K]

        summ = out.diagnostics.as_summary()
        if summ["min_ess"] is not None:
            min_ess = min(min_ess, summ["min_ess"])
        total_resamples += summ["num_resample_events"]
        if summ["final_unique_ancestors"] is not None:
            uniq_anc.extend(summ["final_unique_ancestors"])

        def _decode_exec(row_ids, plen):
            """token ids -> suffix text -> executed numeric answer (or None)."""
            suffix_ids = row_ids[plen:]
            n_bad = sum(1 for t in suffix_ids if t >= tok_len)
            safe = [t if 0 <= t < tok_len else tok.eos_token_id for t in suffix_ids]
            text = tok.decode(safe, skip_special_tokens=True)
            return predict_answer(text, timeout_s), text, len(suffix_ids), n_bad

        for b, gi in enumerate(idxs):
            rec = ds[gi]
            gold = _extract_gold_answer(rec["response_ground_truth"])

            preds, corrects = [], []
            for k in range(K):
                pred, text, ntok, nbad = _decode_exec(gen_ids[b, k].cpu().tolist(), plens[b])
                n_gen_tokens += ntok
                n_invalid_tok += nbad
                ok = bool(_numbers_equal(pred, gold))
                preds.append(pred)
                corrects.append(int(ok))
                part_correct += int(ok)
                part_total += 1
                n_answered += int(pred is not None)
                if len(records) < 50 and k == 0:
                    records.append({"idx": gi, "prompt": rec["prompt"][:200],
                                    "response": text[:400], "correct": ok})

            n_pass += int(any(corrects))
            per_prompt_particle_mean.append(sum(corrects) / K)
            answered = [p for p in preds if p is not None]
            distinct_sum += len(set(answered))

            # self-consistency: plurality over executed answers (None never wins;
            # deterministic tie-break by smallest value)
            maj_ok = False
            if answered:
                counts = Counter(answered)
                top = max(counts.items(), key=lambda kv: (kv[1], -float(kv[0])))
                maj_ok = bool(_numbers_equal(top[0], gold))
            n_maj += int(maj_ok)
            per_prompt_maj.append(int(maj_ok))

            # ---- weight-vs-correctness on the PRE-final-resample population ----
            wb = w_norm[b]
            correct_pre, wvote = [], {}
            for k in range(K):
                pred_p, _, _, _ = _decode_exec(gen_ids_pre[b, k].cpu().tolist(), plens[b])
                okp = int(bool(_numbers_equal(pred_p, gold)))
                correct_pre.append(okp)
                wk = float(wb[k])
                if pred_p is not None:
                    wvote[pred_p] = wvote.get(pred_p, 0.0) + wk
                if okp:
                    mean_w_correct += wk; nw_correct += 1
                else:
                    mean_w_incorrect += wk; nw_incorrect += 1
            n_topw += correct_pre[int(torch.argmax(wb).item())]
            n_botw += correct_pre[int(torch.argmin(wb).item())]
            if wvote:
                wv = max(wvote.items(), key=lambda kv: (kv[1], -float(kv[0])))
                n_wvote += int(bool(_numbers_equal(wv[0], gold)))

        done = min(start + args.batch_size, n)
        print(f"[gsm8k-fkc] {done}/{n}  particle_acc={100.0*part_correct/max(1,part_total):.2f}%  "
              f"maj@{K}={100.0*n_maj/max(1,done):.2f}%  pass@{K}={100.0*n_pass/max(1,done):.2f}%  "
              f"min_ess={min_ess:.2f}", flush=True)

    maj_arr = np.asarray(per_prompt_maj, dtype=np.float64)
    part_arr = np.asarray(per_prompt_particle_mean, dtype=np.float64)
    maj_acc, maj_lo, maj_hi = bootstrap_ci(maj_arr, n_boot, seed=args.seed)
    p_acc, p_lo, p_hi = bootstrap_ci(part_arr, n_boot, seed=args.seed)

    result = {
        "task": "gsm8k", "checkpoint": str(args.checkpoint),
        "sampler_kind": args.sampler_kind, "beta": args.beta, "num_particles": K,
        "steps": steps, "proposal": args.proposal, "churn_gamma": args.churn_gamma,
        "lambda_zero": args.lambda_zero, "lambda_profile": args.lambda_profile,
        "lambda_normalize": args.lambda_normalize,
        "resampling_policy": args.resampling_policy, "ess_threshold": args.ess_threshold,
        "sc_policy": args.sc_policy, "prior_mode": args.prior_mode,
        "final_resample": bool(args.final_resample),
        "resample_entropy_frac": args.resample_entropy_frac,
        "guidance_scale": args.guidance_scale, "ema": bool(args.ema),
        "sigma_data": sigma_data_used, "num_examples": int(n),
        "particle_mean_accuracy": part_correct / max(1, part_total),
        "particle_mean_accuracy_ci95": [p_lo, p_hi],
        "pass_at_k": n_pass / max(1, n),
        "maj_at_k": n_maj / max(1, n),
        "maj_at_k_ci95": [maj_lo, maj_hi],
        "weighted_vote_accuracy": n_wvote / max(1, n),
        "top_weight_accuracy": n_topw / max(1, n),
        "bottom_weight_accuracy": n_botw / max(1, n),
        "mean_weight_of_correct": (mean_w_correct / nw_correct) if nw_correct else None,
        "mean_weight_of_incorrect": (mean_w_incorrect / nw_incorrect) if nw_incorrect else None,
        "answered_rate": n_answered / max(1, part_total),
        "mean_distinct_answers": distinct_sum / max(1, n),
        "min_ess": (None if min_ess == float("inf") else min_ess),
        "total_resample_events": total_resamples,
        "mean_final_unique_ancestors": (sum(uniq_anc) / len(uniq_anc)) if uniq_anc else None,
        "invalid_token_rate": n_invalid_tok / max(1, n_gen_tokens),
        "timeout_s": timeout_s,
        "gsm8k_test_path": args.gsm8k_test_path,
        # Per-prompt vectors (index-aligned across cells, since every cell uses the
        # same problems and seed). Required for a PAIRED comparison between betas:
        # the between-prompt variance cancels in the per-prompt difference, which
        # aggregate means alone cannot recover.
        "per_prompt_particle_acc": per_prompt_particle_mean,
        "per_prompt_maj": per_prompt_maj,
        "sample_records": records,
    }
    prop_tag = (f"em_lz{args.lambda_zero:g}" if args.proposal == "em"
                else f"churn_g{args.churn_gamma:g}")
    tag = (f"fkc_{prop_tag}_beta{args.beta:g}_K{K}_s{steps}_{args.resampling_policy}"
           f"_sc{args.sc_policy}_ess{args.ess_threshold:g}_ema{int(bool(args.ema))}")
    if args.guidance_scale > 0.0:
        tag += f"_cfgw{args.guidance_scale:g}"
    if args.resample_entropy_frac is not None:
        tag += f"_rband{args.resample_entropy_frac:g}"
    out_path = out_dir / f"gsm8k_results_{tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("\n=== GSM8K FKC RESULT ===")
    print(json.dumps({k: v for k, v in result.items() if k != "sample_records"}, indent=2))
    print(f"saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sampler", default="stochastic", choices=["stochastic", "deterministic"],
                    help="stochastic => EDM-style churn (needs gamma>0); deterministic => no churn")
    ap.add_argument("--sampler_kind", default="ddim", choices=["ddim", "heun", "em", "pc", "fkc_em"],
                    help="ddim = CoBit ddim_entropic headline path (EDM churn, capped at gamma<=sqrt(2)-1); "
                         "heun = 2nd-order ablation; em = Euler-Maruyama entropy-gated reverse SDE "
                         "(stochasticity via lambda_zero, NO churn cap -> exceed the EDM ceiling); "
                         "pc = predictor-corrector entropy-gated SDE; "
                         "fkc_em = Feynman-Kac SMC sampler for the tempered target p^beta.")
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
    ap.add_argument("--em_step_gamma_cap", type=float, default=None,
                    help="EM only: per-step stability cap on effective churn gamma_step=lam*Delta/sigma. "
                         "Bounds injected noise to <= sqrt(2*cap)*sigma. Default (unset) = sampler default 1.0 "
                         "(=> up to sqrt(2)*sigma, too loose). RECOMMENDED 0.41 (~sqrt(2)-1, EDM's own bound): "
                         "removes the low-sigma tail blow-up while leaving the EDM-equivalent bulk untouched. "
                         "See reports/EM_TINYGSM_COLLAPSE_ANALYSIS.md.")
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
    ap.add_argument("--posterior_temp_space", default="bit", choices=["bit", "token"],
                    help="bit: per-bit sigmoid(logit/T) (factorized; collapses below T~0.25). "
                         "token: sharpen the joint posterior over VALID codewords (MDLM/Duo analogue, "
                         "no invalid-code cliff). target full=(raw+mf)/T, learned=raw/T+mf.")
    ap.add_argument("--codeword_topk", type=int, default=None,
                    help="token space: softmax over only the top-k valid tokens per position (speed/memory). "
                         "None = full vocab.")
    # ---- Track A1: local score-temperature (particle-free, 0 extra NFE) ----
    ap.add_argument("--score_temp_tau", type=float, default=1.0,
                    help="Track A1 local score-temperature tau (<1 sharpens). Rescales the PF-ODE "
                         "score by kappa(sigma)=(v+sigma^2)/(tau*v+sigma^2): ->1 at high sigma, ->1/tau "
                         "as sigma->0 (late sharpening only). tau=1.0 is a bit-identical no-op. Zero "
                         "extra NFE; base posterior is untouched for self-conditioning/decoding.")
    ap.add_argument("--score_temp_clean_var", type=float, default=0.25,
                    help="Track A1 clean-bit variance v (default 0.25 = Var of ideal 0/1 bits, mean 0.5). "
                         "This is NOT the EDM preconditioning sigma_data; keep it separate.")
    # ---- FKC (sampler_kind=fkc_em): Feynman-Kac SMC for the tempered target p^beta ----
    ap.add_argument("--beta", type=float, default=1.0,
                    help="FKC tempering exponent (>=1). beta=1 is the untempered base (K=1 == EM; "
                         "K>1 == K independent samples + voting). The log-weight is EXTENSIVE in the "
                         "number of free bits -- GSM8K has far more than Sudoku's 356, so keep beta-1 "
                         "very small and watch min_ess: sweep {1.0,1.001,1.002,1.005,1.01}.")
    ap.add_argument("--num_particles", type=int, default=8,
                    help="FKC particle count K per prompt. Effective forward batch is K*batch_size, "
                         "so drop --batch_size accordingly (e.g. K=16 -> --batch_size 4).")
    ap.add_argument("--ess_threshold", type=float, default=0.5,
                    help="FKC resample when ESS < ess_threshold * K (fraction).")
    ap.add_argument("--resampling_policy", default="ess", choices=["ess", "every_step_active", "never"])
    ap.add_argument("--sc_policy", default="inherit", choices=["inherit", "zero", "stateless_two_pass"],
                    help="FKC self-conditioning policy. inherit = carry D_k as particle state (headline).")
    ap.add_argument("--final_resample", type=int, default=1, help="FKC mandatory final resample (1/0).")
    ap.add_argument("--prior_mode", default="sampler_gaussian",
                    choices=["sampler_gaussian", "forward_marginal_diag"],
                    help="FKC tempered prior variance: sigma_max^2/beta (default) or (sigma_max^2+v)/beta.")
    ap.add_argument("--proposal", default="edm_churn", choices=["em", "edm_churn"],
                    help="FKC proposal: edm_churn = EDM-churn proposal (churn_gamma), far more stable "
                         "and much better per-particle quality on CoBit checkpoints; em = explicit "
                         "entropy-gated Euler-Maruyama (lambda_zero), a cleaner theoretical probe.")
    ap.add_argument("--churn_gamma", type=float, default=0.0,
                    help="FKC edm_churn proposal: per-step churn gamma (capped at sqrt(2)-1 ~ 0.4142). "
                         "Provides the stochasticity that lets duplicated ancestors branch. Use the "
                         "value that is best for plain sampling (0.41 for this checkpoint).")
    ap.add_argument("--resample_entropy_frac", type=float, default=None,
                    help="Confine RESAMPLING to the central entropy-rate band holding this fraction of "
                         "the log-sigma pdf mass (e.g. 0.8 => [q0.1, q0.9]). Weights still accumulate "
                         "everywhere, so the target is unchanged.")
    ap.add_argument("--gsm8k_test_path", default=None,
                    help="Override the GSM8K test JSON (list of {prompt, response_ground_truth}). "
                         "Used to evaluate a fixed SHARD of the test set so that runs on different "
                         "machines can be concatenated: shards must be disjoint and share every "
                         "other setting. Default: datasets/gsm8k/gsm8k_test.json (all 1319).")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--allow_cpu", action="store_true",
                    help="Permit running on CPU. By default the eval ASSERTS CUDA is available, "
                         "because a silent CPU fallback (e.g. CUDA failing to init on a bad node) "
                         "runs ~100x slower and silently invalidates throughput/results. Set "
                         "GSM8K_ALLOW_CPU=1 for the same effect.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    steps = int(args.steps or getattr(cfg.evaluation, "num_sampling_steps", 1024))
    timeout_s = float(getattr(getattr(cfg.evaluation, "gsm8k", object()), "timeout_s", 5.0))
    n_boot = int(getattr(getattr(cfg.evaluation, "gsm8k", object()), "bootstrap_size", 10000))
    # Guard against a silent CPU fallback when CUDA fails to init on a bad node:
    # such a run would (a) be ~100x slower and (b) silently produce results that
    # look real. Fail loudly unless CPU was explicitly requested.
    allow_cpu = bool(args.allow_cpu) or os.environ.get("GSM8K_ALLOW_CPU", "") not in ("", "0")
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError(
            "CUDA is not available — refusing to run the GSM8K eval on CPU.\n"
            "A silent CPU fallback (CUDA failing to init on a bad node) runs ~100x slower "
            "and silently invalidates results. If this is intentional, pass --allow_cpu "
            "(or set GSM8K_ALLOW_CPU=1)."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.checkpoint).resolve().parent.parent
    out_dir = Path(args.out_dir or (run_dir / "gsm8k_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the sigma_data the model was TRAINED with (sidecar) unless overridden.
    sigma_data_used, _ = resolve_sigma_data(cfg, run_dir, args.sigma_data)

    model, sampler = load_model_and_sampler(
        cfg, args.checkpoint, device, apply_ema=bool(args.ema), sampler_kind=args.sampler_kind,
        lambda_zero=args.lambda_zero, lambda_profile=args.lambda_profile,
        lambda_normalize=args.lambda_normalize, guidance_mode=args.guidance_mode,
        em_step_gamma_cap=args.em_step_gamma_cap,
        fkc_beta=args.beta, fkc_num_particles=args.num_particles,
        fkc_resampling_policy=args.resampling_policy,
        fkc_ess_threshold_fraction=args.ess_threshold,
        fkc_final_resample=bool(args.final_resample),
        fkc_sc_policy=args.sc_policy, fkc_prior_mode=args.prior_mode,
        fkc_proposal=args.proposal, fkc_churn_gamma=args.churn_gamma,
        fkc_resample_entropy_frac=args.resample_entropy_frac)
    schedule = args.schedule
    configure_stochastic(cfg, mode=args.sampler, gamma=args.gamma, num_steps=steps)

    if args.gsm8k_test_path is not None:
        cfg.data.gsm8k_test_path = args.gsm8k_test_path
        print(f"[gsm8k] test set: {args.gsm8k_test_path}", flush=True)
    ds = GSM8KTestDataset(cfg)
    tok = ds.tok
    bpt = ds.bits_per_token
    tok_len = len(tok)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))

    if args.sampler_kind in {"fkc_em", "fkc"}:
        return _run_fkc_gsm8k(cfg, sampler, ds, n, bpt, tok, tok_len, args, run_dir,
                              out_dir, sigma_data_used, steps, timeout_s, n_boot)

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
            posterior_temp_space=args.posterior_temp_space,
            codeword_vocab_size=tok_len,
            codeword_topk=args.codeword_topk,
            score_temp_tau=args.score_temp_tau,
            score_temp_clean_var=args.score_temp_clean_var,
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
        "em_step_gamma_cap": args.em_step_gamma_cap,
        "posterior_temp": args.posterior_temp,
        "posterior_temp_target": args.posterior_temp_target,
        "posterior_temp_schedule": args.posterior_temp_schedule,
        "posterior_temp_sigma_lo": args.posterior_temp_sigma_lo,
        "posterior_temp_sigma_hi": args.posterior_temp_sigma_hi,
        "posterior_temp_space": args.posterior_temp_space,
        "codeword_topk": args.codeword_topk,
        "score_temp_tau": args.score_temp_tau,
        "score_temp_clean_var": args.score_temp_clean_var,
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
    if abs(float(args.posterior_temp) - 1.0) > 1e-8 or args.posterior_temp_space != "bit":
        tag += f"_pt{args.posterior_temp:g}_{args.posterior_temp_target}_{args.posterior_temp_schedule}"
        if args.posterior_temp_schedule == "sigma_ramp":
            tag += f"_lo{args.posterior_temp_sigma_lo:g}_hi{args.posterior_temp_sigma_hi:g}"
        if args.posterior_temp_space != "bit":
            tag += f"_{args.posterior_temp_space}"
            if args.codeword_topk is not None:
                tag += f"k{args.codeword_topk}"
    if abs(float(args.score_temp_tau) - 1.0) > 1e-8:
        tag += f"_tau{args.score_temp_tau:g}"
    if args.sigma_max is not None:
        tag += f"_smax{args.sigma_max:g}"
    if args.sigma_min is not None:
        tag += f"_smin{args.sigma_min:g}"
    if args.sampler_kind != "ddim":
        tag += f"_kind{args.sampler_kind}_lz{args.lambda_zero:g}_{args.lambda_normalize}"
        if args.sampler_kind == "em" and args.em_step_gamma_cap is not None:
            tag += f"_cap{args.em_step_gamma_cap:g}"
        if args.sampler_kind == "pc":
            tag += f"_{args.guidance_mode}"
    out_path = out_dir / f"gsm8k_results_{tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("\n=== GSM8K RESULT ===")
    print(json.dumps({k: v for k, v in result.items() if k != "sample_records"}, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
