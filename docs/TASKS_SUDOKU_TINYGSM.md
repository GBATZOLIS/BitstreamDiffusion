# Task-driven experiments: Sudoku & TinyGSM→GSM8K (S-FLM parity)

This adds the two **verifiable, ground-truth-solution** tasks from
*S-FLM: Language Modeling with Hyperspherical Flows* (Deschenaux & Gulcehre)
to CoBit, so we can compare bitstream diffusion against S-FLM / MDLM / Duo /
AR on the **same** benchmarks under a fair protocol.

The motivation is S-FLM's own argument: GenPPL/entropy can be misleading in
verifiable domains, so correctness on Sudoku (exact-match) and GSM8K
(executable-code accuracy) is the headline signal.

## What is faithful to S-FLM (parity)

| Aspect | S-FLM | CoBit (here) |
|---|---|---|
| Sudoku vocab/layout | 12 ids, `[BOS] puzzle(89) [BOS] solution(89)` = 180 tok | identical |
| Sudoku prompt | first 91 tok, excluded from loss | identical (per-example prefix mask) |
| Sudoku difficulty | easy/medium/hard = 40/35/30 clues | identical (`sudoku_generator.py` vendored verbatim) |
| Sudoku split | 48k train / 2k val, seed 42, dedup, unique-solution | identical |
| Sudoku metric | exact-match of 89-token solution suffix | identical (`(gen==gt).all(dim=1)`) |
| Sudoku NFE | 180 | 180 |
| TinyGSM data | `TinyGSM/TinyGSM`, SmolLM-135M, `[BOS] q \n code [EOS]`, len 512 | identical |
| TinyGSM masking | `train_on_prompt=False, train_on_pad=True, filter_too_long=True` | identical (loss = ~prefix over answer+EOS+pad) |
| GSM8K test | `gsm8k_test.json` (1319, `prompt`/`response_ground_truth`) | identical (vendored) |
| GSM8K scoring | restricted sandbox, fn `simple_math_problem`, math-only import, 1e-3 tol, 5s | identical (`sandbox_gsm8k.py` vendored) |
| GSM8K CI | percentile bootstrap 95% | identical |
| Sudoku trunk | 8 blocks / 512-d / 8 heads, 20k steps, batch 256 | identical |
| TinyGSM trunk | 12 blocks / 768-d / 12 heads, 250k steps, batch 512 | identical |
| EMA / optim | EMA 0.9999, AdamW lr 3e-4, wd 0, constant+warmup | identical |

## What necessarily differs (the method under test)

CoBit replaces the hyperspherical token-embedding state with a **fixed-width
analog-bit** state:
- Sudoku: 4 bits/token → 720-bit bitstream.
- TinyGSM: 16 bits/token (SmolLM-135M, |V|=49152+PAD) → 8192-bit bitstream.

Training is CoBit's Gaussian bitstream diffusion (matched-filter residual,
binary score matching, EDM weighting, entropy-rate noise schedule). The prompt
is conditioned by clamping the clean prompt bits at every solver step (same
mechanism CoBit already uses for prefix conditioning; identical in spirit to
S-FLM's `_project_prefix`).

**Eval sampler.** The headline path is CoBit's `ddim_entropic` sampler — the
`DDIMSampler` integrating the probability-flow update on the entropy-rate sigma
grid with **EDM-style stochastic churn** interleaved (`s_churn = γ·(NFE−1)`,
`s_noise = 1.003`, full entropy band), exactly the path that produces the
paper's GenPPL numbers (`evaluation/generation_driver.py::create_sampler`).
`evaluation/tasks/_task_common.py` builds `DDIMSampler` by default
(`sampler_kind="ddim"`); `--gamma 0` collapses it to deterministic probability
flow on the same grid (reported as an ablation), and `sampler_kind="heun"` is a
2nd-order ablation. Sweep `γ` (the diffusion "temperature") at eval — globally
constrained Sudoku typically prefers small/zero churn, while GSM8K benefits from
the stochastic operating point.

A bitstream-specific failure mode that S-FLM does not have: decoded token ids
can fall outside the vocabulary (Sudoku ids 12–15; SmolLM ids ≥ 49153). Both
evaluators report an `invalid_token_rate` diagnostic.

## Files

```
data/task_codec.py            # MSB-first fixed-width bit codec (matches the repo raw_binary table)
data/sudoku_generator.py      # vendored verbatim from s-flm
data/sudoku.py                # SudokuDataset -> {x0, prefix_mask, input_ids}
data/tinygsm.py               # TinyGSMDataset + GSM8KTestDataset
configs/tasks/sudoku_bits.py  # 8L/512d, difficulty via SUDOKU_DIFFICULTY env
configs/tasks/tinygsm_bits.py # 12L/768d, TINYGSM_MAX_TRAIN env caps the corpus
evaluation/tasks/sudoku_eval.py   # exact-match + diagnostics
evaluation/tasks/gsm8k_eval.py    # executable-code accuracy + bootstrap CI
evaluation/tasks/sandbox_gsm8k.py # vendored sandbox
scripts/tasks/{train,eval}_{sudoku,gsm8k}.sh
tests/test_task_codec.py tests/test_sudoku_format.py tests/test_gsm8k_sandbox.py
```

Trainer change (`trainers/trainer.py`): batches may now be a dict
`{x0, prefix_mask}`; `_step_continuous` honours a per-example `batch_prefix_mask`
(needed for TinyGSM's variable prompt length). Plain-tensor batches (LM1B/OWT)
are unchanged.

## Running

### Sudoku (cheap; ~hours on a couple of GPUs)
```bash
for d in easy medium hard; do
  SUDOKU_DIFFICULTY=$d NPROC=2 bash scripts/tasks/train_sudoku.sh
done
# headline = entropy-rate stochastic; sweep gamma (the diffusion "temperature").
for d in easy medium hard; do
  for g in 0.0 0.05 0.1 0.2; do
    CKPT=runs/tasks/sudoku/$d/cobit_raw_binary_bits/checkpoints/step=000020000.pt \
      SUDOKU_DIFFICULTY=$d SAMPLER=stochastic GAMMA=$g STEPS=180 \
      bash scripts/tasks/eval_sudoku.sh
  done
done
```

### TinyGSM → GSM8K (Isambard: 2 nodes × 4 GH200)
```bash
# node 0
NNODES=2 NODE_RANK=0 NPROC=4 RDZV_ENDPOINT=<host0>:29502 bash scripts/tasks/train_tinygsm.sh
# node 1
NNODES=2 NODE_RANK=1 NPROC=4 RDZV_ENDPOINT=<host0>:29502 bash scripts/tasks/train_tinygsm.sh

CKPT=runs/tasks/tinygsm/cobit_raw_binary_bits/checkpoints/step=000250000.pt \
  SAMPLER=stochastic GAMMA=0.1 STEPS=1024 bash scripts/tasks/eval_gsm8k.sh
```

Local smoke (capped corpus): `TINYGSM_MAX_TRAIN=4000` streams a small subset.

> **NFE note.** S-FLM's paper text reports GSM8K at 1024 sampling steps, while
> their repo's sample scripts default to 32. We make NFE a flag and recommend
> reporting CoBit at the **same NFE S-FLM used for the number being compared**
> (and ideally a small NFE sweep), Sudoku at 180.

## Reference numbers to beat / match

Sudoku (exact-match %, 180 NFE): Duo 96.3/84.7/58.4, FLM 94.2/82.7/44.5,
S-FLM(+trunc+adaptive) 94.8/85.2/45.0, AR-greedy 14.6/5.1/1.0.

GSM8K (acc %, T=1): AR-sample 53.9, AR-greedy 63.3, MDLM 18.0, Duo 17.2,
FLM 0.3, S-FLM vanilla 1.2, S-FLM(+adaptive+S-arch) 12.4, S-FLM(+top-k=1) 18.0.
The fair first target for CoBit is the non-AR diffusion band (~17–18%).
