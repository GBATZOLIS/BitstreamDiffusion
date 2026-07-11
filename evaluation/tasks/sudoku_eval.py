"""Sudoku exact-match evaluation for CoBit (S-FLM parity).

Primary metric: exact-match accuracy of the full 89-token solution suffix
(positions 91..179) against the unique ground-truth solution, decoded from the
generated bitstream. Mirrors S-FLM main.py::_sudoku_eval:
    correct = (generated_suffix == ground_truth_suffix).all(dim=1)

Secondary diagnostics (not the headline): invalid-token rate, grid-cell exact
match (ignoring separators), separator-position accuracy, clue consistency,
and full Sudoku-rule validity of the completed grid.

Usage:
    python -m evaluation.tasks.sudoku_eval \
        --config configs/tasks/sudoku_bits.py \
        --checkpoint runs/.../checkpoints/step=000020000.pt \
        --sampler stochastic --gamma 0.0 --steps 180 --limit 2000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from data.sudoku import (
    SudokuDataset, PROMPT_LEN_TOKENS, TOTAL_LEN_TOKENS, BITS_PER_TOKEN,
    ROW_SEPARATOR_ID,
)
from data.task_codec import bits_to_token_ids
from evaluation.tasks._task_common import (
    load_config, load_model_and_sampler, configure_stochastic, sample_bits,
    resolve_sigma_data,
)

GRID_START_PUZZLE = 1
GRID_START_SOLUTION = PROMPT_LEN_TOKENS  # 91


def _grid_cells(ids_row, start):
    """Extract the 81 cell values from a 89-token grid starting at `start`.

    Returns (cells, sep_ok) where sep_ok asserts separators sit at expected slots.
    """
    cells = []
    sep_ok = True
    i = start
    for r in range(9):
        cells.extend(ids_row[i:i + 9]); i += 9
        if r < 8:
            sep_ok = sep_ok and (ids_row[i] == ROW_SEPARATOR_ID)
            i += 1
    return cells, sep_ok


def _valid_sudoku(cells):
    if any(c < 1 or c > 9 for c in cells):
        return False
    g = [cells[i * 9:(i + 1) * 9] for i in range(9)]
    full = set(range(1, 10))
    for i in range(9):
        if set(g[i]) != full:
            return False
        if {row[i] for row in g} != full:
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = [g[br + i][bc + j] for i in range(3) for j in range(3)]
            if set(box) != full:
                return False
    return True


def _run_fkc_sudoku(cfg, sampler, ds, n, bpt, args, run_dir, out_dir, sigma_data_used):
    """FKC particle evaluation: pass@K / maj@K / per-particle acc + SMC telemetry.

    For each prompt we decode all K particles, then report:
      * particle_mean_acc : mean over particles of exact-match (per-particle acc)
      * pass_at_k         : any particle exactly matches the unique solution
      * maj_at_k          : majority vote over VALID grids matches (invalid grids
                            never win; ties broken deterministically)
    plus valid rate, mean distinct solutions/prompt, and ESS/ancestry diagnostics.
    """
    from evaluation.tasks._task_common import sample_bit_particles

    K = int(args.num_particles)
    steps = int(args.steps or getattr(cfg.evaluation, "num_sampling_steps", 180))
    n_pass = n_maj = 0
    part_correct = 0
    part_total = 0
    n_valid_part = 0
    distinct_sum = 0
    min_ess = float("inf")
    total_resamples = 0
    uniq_anc = []
    records = []

    for start in range(0, n, args.batch_size):
        idxs = list(range(start, min(start + args.batch_size, n)))
        Bc = len(idxs)
        x0 = torch.stack([ds[i]["x0"] for i in idxs]).float().to(sampler.device)
        pm = torch.stack([ds[i]["prefix_mask"] for i in idxs]).to(sampler.device)
        gt_ids = torch.stack([ds[i]["input_ids"] for i in idxs]).to(sampler.device)

        out = sample_bit_particles(
            cfg, sampler, prefix_full=x0, prefix_mask=pm, num_steps=steps,
            schedule=args.schedule, entropy_run_dir=str(run_dir),
            sigma_min_override=args.sigma_min, seed=args.seed,
        )
        S = x0.shape[1]
        gen_ids = bits_to_token_ids(out.bits.reshape(Bc * K, S), bpt).reshape(Bc, K, -1)  # [B,K,180]
        gt_suffix = gt_ids[:, PROMPT_LEN_TOKENS:].cpu()

        d = out.diagnostics
        summ = d.as_summary()
        if summ["min_ess"] is not None:
            min_ess = min(min_ess, summ["min_ess"])
        total_resamples += summ["num_resample_events"]
        if summ["final_unique_ancestors"] is not None:
            uniq_anc.extend(summ["final_unique_ancestors"])

        for b, gi in enumerate(idxs):
            gt = gt_suffix[b].tolist()
            suffixes = []          # decoded solution suffix per particle
            valid_suffixes = []
            any_exact = False
            for k in range(K):
                row = gen_ids[b, k].cpu().tolist()
                suffix = row[PROMPT_LEN_TOKENS:]
                suffixes.append(tuple(suffix))
                sol_cells, _ = _grid_cells(row, GRID_START_SOLUTION)
                exact = (suffix == gt)
                any_exact = any_exact or exact
                part_correct += int(exact)
                part_total += 1
                if _valid_sudoku(sol_cells):
                    n_valid_part += 1
                    valid_suffixes.append(tuple(suffix))
            n_pass += int(any_exact)
            distinct_sum += len(set(suffixes))
            # majority vote among valid grids (invalid never win; deterministic tie-break)
            maj_ok = False
            if valid_suffixes:
                from collections import Counter
                counts = Counter(valid_suffixes)
                top = max(counts.items(), key=lambda kv: (kv[1], [-c for c in kv[0]]))
                maj_ok = (list(top[0]) == gt)
            n_maj += int(maj_ok)
            if len(records) < 50:
                records.append({"idx": gi, "pass": bool(any_exact), "maj": bool(maj_ok),
                                "distinct": len(set(suffixes))})

        done = min(start + args.batch_size, n)
        print(f"[sudoku-fkc] {done}/{n}  pass@{K}={n_pass} ({100.0*n_pass/max(1,done):.1f}%)  "
              f"maj@{K}={n_maj} ({100.0*n_maj/max(1,done):.1f}%)  min_ess={min_ess:.2f}", flush=True)

    result = {
        "task": "sudoku", "difficulty": cfg.data.difficulty,
        "checkpoint": str(args.checkpoint), "sampler_kind": args.sampler_kind,
        "beta": args.beta, "num_particles": K, "steps": steps,
        "proposal": args.proposal, "churn_gamma": args.churn_gamma,
        "lambda_zero": args.lambda_zero, "lambda_profile": args.lambda_profile,
        "lambda_normalize": args.lambda_normalize, "resampling_policy": args.resampling_policy,
        "ess_threshold": args.ess_threshold, "sc_policy": args.sc_policy,
        "prior_mode": args.prior_mode, "final_resample": bool(args.final_resample),
        "ema": bool(args.ema), "sigma_data": sigma_data_used, "num_examples": n,
        "particle_mean_accuracy": part_correct / max(1, part_total),
        "pass_at_k": n_pass / max(1, n),
        "maj_at_k": n_maj / max(1, n),
        "valid_sudoku_rate": n_valid_part / max(1, part_total),
        "mean_distinct_solutions": distinct_sum / max(1, n),
        "min_ess": (None if min_ess == float("inf") else min_ess),
        "total_resample_events": total_resamples,
        "mean_final_unique_ancestors": (sum(uniq_anc) / len(uniq_anc)) if uniq_anc else None,
        "sample_records": records,
    }
    prop_tag = (f"em_lz{args.lambda_zero:g}" if args.proposal == "em"
                else f"churn_g{args.churn_gamma:g}")
    tag = (f"{cfg.data.difficulty}_fkc_{prop_tag}_beta{args.beta:g}_K{K}_s{steps}"
           f"_{args.resampling_policy}_sc{args.sc_policy}_ess{args.ess_threshold:g}_ema{int(bool(args.ema))}")
    out_path = out_dir / f"sudoku_results_{tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("\n=== SUDOKU FKC RESULT ===")
    print(json.dumps({k: v for k, v in result.items() if k != "sample_records"}, indent=2))
    print(f"saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--difficulty", default=None, help="override (else from config/env)")
    ap.add_argument("--sampler", default="stochastic", choices=["stochastic", "deterministic"],
                    help="stochastic => EDM-style churn (needs gamma>0); deterministic => no churn")
    ap.add_argument("--sampler_kind", default="ddim", choices=["ddim", "heun", "em", "pc", "fkc_em"],
                    help="ddim = CoBit ddim_entropic headline path (EDM churn); heun = 2nd-order ablation; "
                         "fkc_em = Feynman-Kac SMC sampler for the tempered target p^beta")
    ap.add_argument("--schedule", default="entropic", choices=["entropic", "karras"],
                    help="sigma grid; entropic = trained entropy-rate schedule")
    ap.add_argument("--gamma", type=float, default=0.0, help="churn gamma; 0 => deterministic")
    ap.add_argument("--ema", type=int, default=1, help="1=EMA weights (headline), 0=raw weights")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="max validation puzzles")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma_min", type=float, default=None, help="terminal sigma override")
    ap.add_argument("--sigma_data", type=float, default=None,
                    help="Override EDM preconditioning sigma_data used at sampling "
                         "(feeds c_in=1/sqrt(sigma^2+sigma_data^2) in the denoiser). "
                         "Default: config value (0.5). The value the model was TRAINED "
                         "with is the SigmaDataEstimator estimate (see training log: "
                         "'sigma_data estimated: ...'); pass it here to test "
                         "train/eval-matched preconditioning.")
    ap.add_argument("--guidance_scale", type=float, default=0.0,
                    help="Classifier-free guidance scale w. 0 => no guidance (conditional). "
                         "probs_g = probs_u + w*(probs_c - probs_u), where probs_u uses the "
                         "cfg.cond.null_strategy null prefix. Requires a model trained with "
                         "p_uncond>0; otherwise the unconditional path is untrained.")
    ap.add_argument("--lambda_zero", type=float, default=0.0,
                    help="EM/PC entropy-gated SDE: Langevin strength lambda_0 (>=0). "
                         "0 => deterministic (bit-identical to DDIM det). Only used by --sampler_kind em.")
    ap.add_argument("--lambda_profile", default="entropy_rate", choices=["entropy_rate", "flat"],
                    help="EM/PC lambda(sigma) profile shape.")
    ap.add_argument("--lambda_normalize", default="as_saved", choices=["as_saved", "peak"],
                    help="EM/PC lambda table normalization: 'as_saved' (lambda_0=S_churn anchor) or 'peak'.")
    ap.add_argument("--guidance_mode", default="predictor_only", choices=["predictor_only", "all"],
                    help="PC only: 'predictor_only' guides the PF predictor and runs the Langevin "
                         "corrector at the conditional score (entropy-gated-SDE-correct CFG); "
                         "'all' guides both (naive CFG).")
    ap.add_argument("--em_step_gamma_cap", type=float, default=None,
                    help="EM only: per-step churn cap gamma_step=lam*Delta/sigma (bounds injected "
                         "noise to <= sqrt(2*cap)*sigma). Unset = sampler default 1.0; RECOMMENDED 0.41 "
                         "(~sqrt(2)-1). See reports/EM_TINYGSM_COLLAPSE_ANALYSIS.md.")
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
                    help="FKC tempering exponent (>=1). beta=1 is the untempered base (K=1 == EM). "
                         "log-weight is extensive in free bits, so keep beta-1 SMALL (Sudoku has 356 "
                         "free bits): sweep {1.0,1.02,1.05,1.1,1.25,1.5,2.0}.")
    ap.add_argument("--num_particles", type=int, default=8, help="FKC particle count K per prompt.")
    ap.add_argument("--ess_threshold", type=float, default=0.5,
                    help="FKC resample when ESS < ess_threshold * K (fraction).")
    ap.add_argument("--resampling_policy", default="ess", choices=["ess", "every_step_active", "never"])
    ap.add_argument("--sc_policy", default="inherit", choices=["inherit", "zero", "stateless_two_pass"],
                    help="FKC self-conditioning policy. inherit = carry D_k as particle state (headline).")
    ap.add_argument("--final_resample", type=int, default=1, help="FKC mandatory final resample (1/0).")
    ap.add_argument("--prior_mode", default="sampler_gaussian",
                    choices=["sampler_gaussian", "forward_marginal_diag"],
                    help="FKC tempered prior variance: sigma_max^2/beta (default) or (sigma_max^2+v)/beta.")
    ap.add_argument("--proposal", default="em", choices=["em", "edm_churn"],
                    help="FKC proposal: em = explicit entropy-gated reverse-SDE Euler-Maruyama "
                         "(lambda_zero); edm_churn = EDM-churn proposal (churn_gamma), the "
                         "asymptotically-exact churn analogue -- far more stable at low NFE.")
    ap.add_argument("--churn_gamma", type=float, default=0.0,
                    help="FKC edm_churn proposal: per-step churn gamma (capped at sqrt(2)-1). "
                         "Provides the stochasticity that lets duplicated ancestors branch.")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--allow_cpu", action="store_true",
                    help="Permit running on CPU. By default the eval ASSERTS CUDA is available, "
                         "because a silent CPU fallback (e.g. CUDA failing to init on a bad node) "
                         "runs far slower and silently invalidates results. Set SUDOKU_ALLOW_CPU=1 "
                         "for the same effect.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.difficulty:
        cfg.data.difficulty = args.difficulty
    steps = int(args.steps or getattr(cfg.evaluation, "num_sampling_steps", 180))
    # Guard against a silent CPU fallback when CUDA fails to init on a bad node.
    allow_cpu = bool(args.allow_cpu) or os.environ.get("SUDOKU_ALLOW_CPU", "") not in ("", "0")
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError(
            "CUDA is not available — refusing to run the Sudoku eval on CPU.\n"
            "A silent CPU fallback (CUDA failing to init on a bad node) runs far slower "
            "and silently invalidates results. If this is intentional, pass --allow_cpu "
            "(or set SUDOKU_ALLOW_CPU=1)."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.checkpoint).resolve().parent.parent
    out_dir = Path(args.out_dir or (run_dir / "sudoku_eval"))
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
        fkc_proposal=args.proposal, fkc_churn_gamma=args.churn_gamma)
    schedule = args.schedule
    configure_stochastic(cfg, mode=args.sampler, gamma=args.gamma, num_steps=steps)

    ds = SudokuDataset(cfg, split="val")
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    bpt = BITS_PER_TOKEN

    if args.sampler_kind in {"fkc_em", "fkc"}:
        return _run_fkc_sudoku(cfg, sampler, ds, n, bpt, args, run_dir, out_dir, sigma_data_used)

    n_exact = 0
    n_grid = 0
    n_valid = 0
    n_clue = 0
    n_sep = 0
    n_invalid_tok = 0
    n_sol_tokens = 0
    records = []

    for start in range(0, n, args.batch_size):
        idxs = list(range(start, min(start + args.batch_size, n)))
        x0 = torch.stack([ds[i]["x0"] for i in idxs]).float().to(device)         # [B,720]
        pm = torch.stack([ds[i]["prefix_mask"] for i in idxs]).to(device)        # [B,720] bool
        gt_ids = torch.stack([ds[i]["input_ids"] for i in idxs]).to(device)      # [B,180]

        bits = sample_bits(
            cfg, sampler, prefix_full=x0, prefix_mask=pm, num_steps=steps,
            schedule=schedule, entropy_run_dir=str(run_dir),
            sigma_min_override=args.sigma_min, seed=args.seed,
            guidance_scale=args.guidance_scale,
            score_temp_tau=args.score_temp_tau,
            score_temp_clean_var=args.score_temp_clean_var,
        )
        gen_ids = bits_to_token_ids(bits, bpt)                                   # [B,180]

        gen_suffix = gen_ids[:, PROMPT_LEN_TOKENS:].cpu()
        gt_suffix = gt_ids[:, PROMPT_LEN_TOKENS:].cpu()
        exact = (gen_suffix == gt_suffix).all(dim=1)
        n_exact += int(exact.sum())

        for b, gi in enumerate(idxs):
            row = gen_ids[b].cpu().tolist()
            gtrow = gt_ids[b].cpu().tolist()
            sol_cells, sep_ok = _grid_cells(row, GRID_START_SOLUTION)
            gt_cells, _ = _grid_cells(gtrow, GRID_START_SOLUTION)
            puz_cells, _ = _grid_cells(gtrow, GRID_START_PUZZLE)

            # invalid tokens in the solution suffix (ids outside [0,11]).
            suffix = row[PROMPT_LEN_TOKENS:]
            n_invalid_tok += sum(1 for t in suffix if t < 0 or t > 11)
            n_sol_tokens += len(suffix)

            n_grid += int(sol_cells == gt_cells)
            n_sep += int(sep_ok)
            valid = _valid_sudoku(sol_cells)
            n_valid += int(valid)
            clue_ok = valid and all(
                (p == 0) or (p == s) for p, s in zip(puz_cells, sol_cells)
            )
            n_clue += int(clue_ok)
            if len(records) < 50:
                records.append({"idx": gi, "exact": bool(exact[b]), "valid": valid})

        print(f"[sudoku] {min(start + args.batch_size, n)}/{n}  "
              f"exact={n_exact}  ({100.0 * n_exact / max(1, min(start + args.batch_size, n)):.1f}%)",
              flush=True)

    result = {
        "task": "sudoku",
        "difficulty": cfg.data.difficulty,
        "checkpoint": str(args.checkpoint),
        "sampler": args.sampler,
        "gamma": args.gamma,
        "ema": bool(args.ema),
        "steps": steps,
        "guidance_scale": args.guidance_scale,
        "sampler_kind": args.sampler_kind,
        "lambda_zero": args.lambda_zero,
        "lambda_profile": args.lambda_profile,
        "lambda_normalize": args.lambda_normalize,
        "guidance_mode": args.guidance_mode,
        "score_temp_tau": args.score_temp_tau,
        "score_temp_clean_var": args.score_temp_clean_var,
        "sigma_data": sigma_data_used,
        "num_examples": n,
        "exact_match_accuracy": n_exact / max(1, n),
        "grid_exact_match": n_grid / max(1, n),
        "valid_sudoku_rate": n_valid / max(1, n),
        "clue_consistency_rate": n_clue / max(1, n),
        "separator_accuracy": n_sep / max(1, n),
        "invalid_token_rate": n_invalid_tok / max(1, n_sol_tokens),
        "sample_records": records,
    }
    tag = f"{cfg.data.difficulty}_{args.sampler}_g{args.gamma}_s{steps}_sd{sigma_data_used:.4f}_w{args.guidance_scale:g}_ema{int(bool(args.ema))}"
    if abs(args.score_temp_tau - 1.0) > 1e-8:
        tag += f"_tau{args.score_temp_tau:g}"
    if args.sampler_kind != "ddim":
        tag += f"_kind{args.sampler_kind}_lz{args.lambda_zero:g}_{args.lambda_normalize}"
        if args.sampler_kind == "pc":
            tag += f"_{args.guidance_mode}"
    out_path = out_dir / f"sudoku_results_{tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("\n=== SUDOKU RESULT ===")
    print(json.dumps({k: v for k, v in result.items() if k != "sample_records"}, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
