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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--difficulty", default=None, help="override (else from config/env)")
    ap.add_argument("--sampler", default="stochastic", choices=["stochastic", "deterministic"],
                    help="stochastic => EDM-style churn (needs gamma>0); deterministic => no churn")
    ap.add_argument("--sampler_kind", default="ddim", choices=["ddim", "heun"],
                    help="ddim = CoBit ddim_entropic headline path (EDM churn); heun = 2nd-order ablation")
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
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.difficulty:
        cfg.data.difficulty = args.difficulty
    steps = int(args.steps or getattr(cfg.evaluation, "num_sampling_steps", 180))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.checkpoint).resolve().parent.parent
    out_dir = Path(args.out_dir or (run_dir / "sudoku_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the sigma_data the model was TRAINED with (sidecar) unless overridden.
    sigma_data_used, _ = resolve_sigma_data(cfg, run_dir, args.sigma_data)

    model, sampler = load_model_and_sampler(
        cfg, args.checkpoint, device, apply_ema=bool(args.ema), sampler_kind=args.sampler_kind)
    schedule = args.schedule
    configure_stochastic(cfg, mode=args.sampler, gamma=args.gamma, num_steps=steps)

    ds = SudokuDataset(cfg, split="val")
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    bpt = BITS_PER_TOKEN

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
    tag = f"{cfg.data.difficulty}_{args.sampler}_g{args.gamma}_s{steps}_sd{sigma_data_used:.4f}_ema{int(bool(args.ema))}"
    out_path = out_dir / f"sudoku_results_{tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("\n=== SUDOKU RESULT ===")
    print(json.dumps({k: v for k, v in result.items() if k != "sample_records"}, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
