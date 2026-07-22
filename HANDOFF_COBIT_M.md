# HANDOFF — open-sourcing the CoBit-M (462M) checkpoint

Read this top-to-bottom before doing anything. Context ran out mid-task; this is
the full state.

## Goal
Open-source the **CoBit-M (462M, OWT, step 750K)** model in the public repo
`/home/gb511/BitstreamDiffusion` (clone of github.com/GBATZOLIS/BitstreamDiffusion)
so downloaders reproduce the **Table 2** numbers. Also: migrate **all** released
checkpoints (CoBit-S LM1B, CoBit-S OWT, CoBit-M) from Google Drive to Hugging Face,
hosting **only the slim EMA-baked checkpoints**.

Working repo (private, source of truth for ckpts/caches): `/home/gb511/ContinuousDiffusionDiscreteSpace`.
Env: `conda activate pytorch` (csic47, 2× RTX A6000 48GB). gpt2-large is cached at `~/.cache/huggingface`.

## Table 2 targets (CoBit-M, 750K EMA)
| NFE | γ | s_churn=γ·(NFE−1) | GenPPL | Entropy |
|---|---|---|---|---|
| 256 | 0.21 | 53.55 | 19.48 | 5.40 |
| 256 | 0.13 | 33.15 | 18.47 | 5.378 (caption) |
| 384 | 0.24 | 91.92 | 13.06 | 5.33 |
| 512 | 0.26 | 132.86 | 9.87 | 5.25 |

## DONE (all committed to working tree, NOT git-committed yet)
- `configs/owt/rate_bits_edm_weight_medium_24x1024.py` — CoBit-M **training** config (ported, owt_flm→owt normalized, no multimodal leakage).
- `configs/owt/eval_cobit_m_750K.py` — CoBit-M **eval** config. Env knobs: `EVAL_CELLS` (`table2`|`low_ppl`|`all`|`"256:0.21,512:0.26"`), `EVAL_NUM_SAMPLES`, `EVAL_OUT_SUFFIX`, `EVAL_CKPT_STEP`, `EVAL_SEED`. `cfg.evaluation.entropy_run_dir="assets/entropy_tables/owt_medium"`.
- `scripts/owt/make_release_checkpoint.py` — bakes EMA shadow into weights, drops optimizer. **Proven identical** to eval-time `ema.apply()`.
- `scripts/owt/eval_cobit_m.sh` — eval launcher (**default NPROC=1**; see NCCL gotcha).
- `scripts/upload_to_hf.py`, `scripts/download_from_hf.py` — HF publish/fetch (place files in exact config paths).
- `scripts/_run_one_eval.sh` — flat single-GPU runner used for verification.
- `assets/entropy_tables/owt_medium/` — the 4 medium entropy tables (DIFFER from `owt/`!). Added to `assets/entropy_tables/SHA256SUMS` (all 12 verify OK).
- `README.md` — added CoBit-M Table 2 headline, layout, eval/training cmds, config index; **migrated download section Drive→HF** with placeholder `COBIT_HF_REPO`.
- `PUBLISHING_HF.md` — maintainer HF upload guide. `release/HF_MODEL_CARD.md` — ready Hub model card.
- `release/` slim EMA checkpoints (gitignored, for HF upload):
  - `cobit_s_lm1b_step001000000_ema.pt` (0.53 GB)
  - `cobit_s_owt_step000750000_ema.pt` (0.54 GB)
  - `cobit_m_owt_step000750000_ema.pt` (1.85 GB)

## VERIFIED ✅ (NFE=256, slim EMA ckpts, single seed, A6000 vs paper GH200)
- CoBit-M  256 γ0.21 = **19.33** (t 19.48), γ0.13 = 18.97 (t 18.47); H 5.39≈5.40
- CoBit-S OWT 256 γ0.13 = **26.71** (t 27.06), γ0.18 = 33.50 (t 34.35); H≈5.32
- CoBit-S LM1B 256 γ0.20 = **60.50** (t 59.76); H 4.31=4.31
- Real refs: OWT 14.87 (t 15.07), LM1B 53.13 (t 53.06). All within seed/hardware noise → **release artifacts are correct**.

## REMAINING WORK
### A. Hugging Face upload (BLOCKED on user credentials)
User must: (1) create HF account/org, (2) create PUBLIC model repo (e.g. `gbatzolis/CoBit`), (3) get a **Write** token.
Then either they `hf auth login` (token cached, preferred — avoids pasting token), or give you the token.
NOTE: `scripts/upload_to_hf.py` currently **requires** `--token`/`HF_TOKEN`. Consider relaxing it to allow `token=None` (use cached login) — small edit in `main()` (`HfApi(token=args.token or None)` and drop the hard `sys.exit(1)` when no token, since cached creds work).
Then:
```
python -m pip install "huggingface_hub>=0.23"
python scripts/upload_to_hf.py --repo-id <owner>/CoBit --dry-run
python scripts/upload_to_hf.py --repo-id <owner>/CoBit
```
Then `sed -i 's/COBIT_HF_REPO/<owner>\/CoBit/g' README.md` (it appears ~3× in the download section), paste `release/HF_MODEL_CARD.md` as the Hub model card.

### B. Commit + push GitHub (user asked to push, with updated README)
`git status` shows untracked: the new configs/scripts, `PUBLISHING_HF.md`, `assets/entropy_tables/owt_medium/`, modified `README.md`, `SHA256SUMS`. Also a stray `?? results/` (pre-existing, not ours — check before adding). Do NOT commit `runs/`, `datasets/`, `release/`, `logs/` (all gitignored — verify). **Confirm git remote/branch first** (`git remote -v`, `git branch`); ask user whether to push to `main` or a branch. Don't add Co-Authored-By trailer (user preference).

### C. Run 384/512 overnight (user wants this AFTER HF/push setup is ready)
The 256 cells already passed. Launch the two remaining CoBit-M cells durably, **single-GPU** (NOT 2-GPU — see gotcha):
```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pytorch   # pre-activate (required, see gotcha)
EVAL_CELLS="384:0.24,512:0.26" EVAL_OUT_SUFFIX=_hi \
  setsid bash scripts/_run_one_eval.sh 0 configs/owt/eval_cobit_m_750K.py > logs/v_hi.log 2>&1 < /dev/null &
```
Then parse `runs/paper/unconditional_text/owt/continuous_rate_raw_binary_bits_medium_24x1024/evaluation_cobit_m_table2_step000750000_hi/results.jsonl` for gen_full_external_ppl (targets 13.06, 9.87). 512 cell ~10 min gen.

## CRITICAL GOTCHAS
1. **2-GPU DDP eval HANGS on this box** (PCIe-only A6000s, no NVLink): NCCL all-gather busy-waits at 100% GPU forever after generation, even with NCCL_P2P_DISABLE=1. **Always use NPROC=1.** Run cells on separate GPUs as separate single-GPU processes (they each use ~7GB of 48GB; can even share a GPU).
2. **Pre-activate conda in the parent shell before `setsid`** — the runner's internal `conda activate` under strict bash flags failed when conda wasn't already active. The Bash tool starts a fresh shell each call, so always `source ...conda.sh && conda activate pytorch` first in the same command.
3. **EMA**: released ckpts have NO "ema" key (baked into "model"). `load_checkpoint` prints "No EMA found, using raw weights" — that's CORRECT (raw == baked EMA). Eval default `apply_ema=True` is a harmless no-op.
4. **Entropy tables are model-specific**: owt_medium ≠ owt. Eval resolves them via `cfg.evaluation.entropy_run_dir` (text_generations.py:659), so shipping in `assets/` is sufficient; run-dir copies not needed.
5. Verification scaffolding in the public repo (gitignored): `datasets/openwebtext_gpt2_trainm100k`→symlink to working-repo OWT cache; `datasets/lm1b`→symlink; `runs/.../checkpoints/step=*.pt`→symlinks to `release/*.pt`. These let evals run locally without rebuilding the 105GB cache.

## Verification logs (this session)
`logs/v256_m.log`, `logs/v256_lm1b.log`, `logs/v256_owt.log` (all finished, "DONE GPU" markers).
Result dirs under `runs/paper/unconditional_text/.../evaluation_*`.
