# Temperature scaling, Feynman–Kac correctors, and reward guidance on CoBit — technical report

**Scope.** A full account of what we implemented, what we ran, and what we found,
for principled low-temperature / annealed / guided sampling on CoBit (continuous
VE diffusion over analog bits), evaluated on Sudoku. Intended for team review to
decide next actions.

**Branch:** `tasks/fkc-temperature`.
**Commits:** `ead8f4b` (A1 + FKC EM), `90c8c30` (eval wiring), `324e1d5` (EDM-churn
proposal), `d2c15ad` (weight-vs-correctness diagnostics), `4df1872` (CFG+FKC Prop 3.1
+ entropy-band + this investigation's findings note).

---

## 0. TL;DR

- **What worked (a little):** *Track A1*, a local score-level temperature, gives a
  small, monotone, collapse-free gain (+2.2 pts exact-match, +4.0 pts valid on hard
  Sudoku, deterministic). It is the only lever that helped.
- **What did not work:** *Feynman–Kac correctors* — annealed (`p^β`), CFG (Prop 3.1
  geometric average), and with entropy-band-restricted resampling — **never beat
  plain β=1 multi-sample voting**, and usually hurt by collapsing particle diversity.
- **Why, demonstrated (not asserted):** the FKC importance weight is **not
  correlated with correctness** (`top_weight_accuracy ≈ mean_particle_accuracy`, and
  in one setting the *lowest*-weight particle was *more* accurate). Resampling under
  such a weight collapses K=16 particles to **~2 distinct samples**, and coverage
  (`pass@K`) is bottlenecked by that distinct count.
- **Headline numbers (hard Sudoku, raw weights, K=16, 180 steps):**

  | method | maj@16 | pass@16 | distinct/16 | note |
  |---|---|---|---|---|
  | β=1 voting (no reweighting) | **90.8%** | **93.6%** | 5.45 | the baseline to beat |
  | Annealed FKC, β=1.05 | 71.5% | 72.3% | 2.70 | tempering collapses coverage |
  | CFG-FKC, w=1.5 | ~64% | ~64% | ~2 | pass==maj (full collapse) |
  | + entropy band | ~73–76% | ~73–77% | ~2 | band does not rescue |

- **Actionable conclusion:** this is the *"confidence ≠ correctness"* regime.
  Closing the maj@K→oracle gap needs a **correctness-correlated reward/value
  signal**, not temperature. For Sudoku the rule-checker is an exact, label-free
  reward → **Track C (reward-guided sampling)** is the recommended next milestone.

---

## 1. Background and motivation

CoBit denoises analog bits `x_σ = x0 + σε` (VE process). The denoiser outputs a
per-bit posterior `D_θ(x,σ,h) = sigmoid(ℓ_MF + r_θ)` (h = carry-mode self-
conditioning), inducing the score `s_θ = (D_θ − x)/σ²`. Prompt coordinates (mask M)
are clamped at every step. We wanted a *principled* way to sharpen / temper this
continuous model, because the naive approach (per-bit posterior temperature,
`sigmoid(logit/T)`) is not principled for continuous diffusion and **collapses below
T≈0.25** (factorized rounding emits out-of-code bits).

We keep three targets strictly separate:
1. **Local score sharpening** (Track A1) — a per-step geometric sharpening of the
   score; particle-free, zero extra NFE.
2. **Global tempering** `π_β ∝ p_θ^β` (annealed FKC).
3. **Correctness tilting** `π_R ∝ p_θ · e^{λR}` (reward guidance; Track C/D — *not
   yet run*, see §8).

FKC (Skreta et al., arXiv:2503.02819) provides the exact machinery for target (2)
and for a CFG variant (Prop 3.1). This report covers (1) and (2) in full, and lays
out (3).

---

## 2. What we implemented

All code on `tasks/fkc-temperature`. File map:
- `diffusion/continuous/samplers.py` — `DDIMSampler` (A1 knob),
  `EulerMaruyamaSampler` (pre-existing entropy-gated SDE), and the new
  `FeynmanKacEulerMaruyamaSampler`.
- `diffusion/continuous/smc.py` — SMC utilities (ESS, systematic resampling, gather,
  diagnostics, `FKCOutput`).
- `evaluation/tasks/_task_common.py` — factory + `sample_bits` / `sample_bit_particles`.
- `evaluation/tasks/sudoku_eval.py` — CLI + `_run_fkc_sudoku` (maj@K / pass@K / SMC
  telemetry / weight-vs-correctness diagnostics).
- `tests/test_score_temp.py`, `tests/test_fkc.py`, `tests/_sampler_harness.py`.

### 2.1 Track A1 — local score-level temperature
Rescale the PF-ODE score by

    κ(σ) = (v + σ²) / (τ·v + σ²),   v = clean-bit variance = 0.25

used as `s_used = κ·s_base`, `d_used = −σ·s_used` (never re-passed through a sigmoid).
Under a locally-Gaussian clean model `N(μ, vI)`, sharpening the clean component to
variance `τv` (τ<1) rescales the *available* score by exactly κ. Properties: κ→1 at
high σ (mode allocation untouched), κ→1/τ as σ→0 (late sharpening only); τ=1 is a
**bit-identical no-op**. The base posterior `D_θ` is retained for self-conditioning,
decoding, and entropy. `v` is a separate constant — **not** the EDM preconditioning
`sigma_data` (which is the trained sidecar value ~0.46). CLI: `--score_temp_tau`.

### 2.2 FKC — Feynman–Kac Euler–Maruyama sampler
Runs K weighted particles per prompt (layout `[B,K,S]`), resamples within prompt
groups only, log-weights in float64. Prompt clamping, carry-mode self-conditioning
(treated as **particle state** — resampled with x), and the tempered prior
`x = c + (σ_max/√β)ε` are all handled. Two proposals:

- **`em`** — explicit entropy-gated reverse-SDE step (`lambda_zero`/`LambdaProfile`).
  Bit-identical to `EulerMaruyamaSampler` at K=1. Unstable at low NFE (matches the
  known EM tail-collapse).
- **`edm_churn`** — EDM churn (`churn_gamma`) before the denoiser, then a β-scaled
  PF-ODE step. The asymptotically-exact churn analogue; **far more stable at 180
  steps**, so used for all K>1 experiments.

**Two mutually-exclusive tempering modes** (combining them raises an error):

- **Annealed** (`beta>1`, no guidance): single-model target `p_c^β`.
  Proposal drift = β-scaled score; log-weight increment
  `Δlog w = ½·β(β−1)·(σ_k²−σ_{k+1}²)·‖s_k‖²_free`.
  This is the **target-score simulation (a=0)** of the FKC paper, generalised to
  CoBit's entropy-gated λ(σ); the noise is *not* divided by √β, so it is **not** the
  tempered-noise (a=1/2) variant. (Tempered-noise would be λ≡1/β; target-score is
  λ≡1; our default entropy-gated λ(σ) is a third member of the same family — the
  corrector weight is λ-independent, verified analytically, see gate 8.)

- **CFG (Prop 3.1)** (`guidance_scale = w > 0`, `beta = 1`): two-model geometric
  average `q_u^{1−w} q_c^w`. Per step it denoises the conditional path (prompt =
  clean prefix) *and* the unconditional path (prompt = null prefix), each carrying
  its own SC particle state. Drift = standard CFG at weight w: `(1−w)s_u + w·s_c`;
  log-weight `Δlog w = ½·w(w−1)·Δσ²·‖s_c − s_u‖²_free` (reweight toward large
  conditional/unconditional disagreement). Decodes from the geometric-average
  posterior mean.

**Resampling schedule.** Systematic resampling (isolated RNG generator so it never
perturbs the noise stream), fired when `ESS < ess_threshold·K` and only where the
proposal is stochastic. Optional **entropy band** (`resample_entropy_frac`, e.g.
0.8): confine resampling to the central entropy-rate mass band `[q_{(1-f)/2},
q_{(1+f)/2}]` in σ; **weights still accumulate at every step** so the target is
unchanged (only the estimator's resampling schedule changes).

**Diagnostics logged:** per-particle mean accuracy, `top_weight_accuracy` /
`bottom_weight_accuracy` (accuracy of the single highest/lowest-weight particle),
weighted-vote accuracy, maj@K, pass@K, min-ESS, mean distinct solutions, resample
events, mean final unique ancestors.

### 2.3 Correctness gates (16, all green)
`tests/test_fkc.py` + `tests/test_score_temp.py`, CPU, deterministic tiny denoiser
harness. Notable ones:
- β=1, K=1 **bit-identical** to `EulerMaruyamaSampler` (λ₀=0 and λ₀>0), and the
  `edm_churn` proposal bit-identical to production DDIM+EDM-churn.
- β=1: all weight increments 0, ESS≡K, no in-loop resampling.
- prompt invariance; ancestry gather consistency; duplicate-ancestor branching;
  fixed-u₀ systematic resample.
- **analytic 1-D Gaussian**: weighted/resampled `Var → (v+σ²)/β` for
  λ∈{0, 1, 1/β, λ(σ)} — validates the tempering math and the λ-independence of the
  weight.
- CFG w=1 reduces bit-identically to plain conditional; CFG w>1 reweights with
  prompt invariance; β>1 + guidance rejected.
- A1: τ=1 end-to-end bit-identical no-op; κ limits/monotonicity.

---

## 3. Experimental setup

- **Task/data:** Sudoku, hard difficulty (30 clues), 2000-puzzle val set; 720-bit
  bitstream (180 tokens × 4 bits), 91-token prompt clamped → **356 free bits**.
- **Checkpoints:** 20k-step `cobit_raw_binary_bits` (non-CFG) and
  `cobit_raw_binary_bits_cfg` (`p_uncond=0.1`). Both load the trained `sigma_data`
  sidecar (~0.46).
- **CRITICAL — weights:** EMA weights **collapse under deterministic sampling** at
  20k steps (hard, deterministic: EMA → 0.0% vs raw → 42.2%). All runs use **raw
  (non-EMA) weights**. This matches the intuition that 20k steps is too short for the
  EMA shadow to be useful (and, for the CFG model, that dropout splits an already-
  short budget).
- **Metrics:** exact-match of the 89-token solution suffix (headline); valid-Sudoku
  rate; and for particle runs: `maj@K` (majority over valid grids), `pass@K` (any
  particle correct), `distinct` (mean distinct decoded solutions among K), `min-ESS`.
- **Baselines (pre-existing, best stochastic churn, single sample, n=2000):** easy
  98.6%, medium 92.1%, hard 67.2%. Hard deterministic single sample: non-CFG 41%,
  CFG 55.5%. Hard is the informative regime (headroom).

---

## 4. Results

### 4.1 Track A1 — small, monotone, no collapse (the one positive)
Deterministic PF-ODE, non-CFG, raw weights, n=2000:

| τ | exact-match | valid |
|---|---|---|
| 1.0 (no-op) | 41.20% | 43.35% |
| 0.85 | 41.85% | 44.40% |
| 0.70 | 42.05% | 44.95% |
| 0.50 | 42.75% | 46.15% |
| 0.30 | 43.40% | 47.40% |

Monotone with sharpening, **no cliff** (contrast the old bit-space posterior-temp
which collapsed below T≈0.25). Gains are modest — expected for a particle-free local
approximation (τ = 1/β is the weightless limit of the annealed target).

### 4.2 β=1 multi-sample voting — the reference (and the winner)
`edm_churn` g=0.4, β=1 (⇒ no reweighting, no resampling; min-ESS = K):

| model | K | distinct/K | pass@K | maj@K | n |
|---|---|---|---|---|---|
| non-CFG | 16 | 5.45 | 93.6% | 90.8% | 512 |
| CFG (interim) | 16 | – | ~76% | ~72% | ~1.7k |
| CFG (interim) | 32 | – | ~80% | ~75% | ~0.9k |

The CFG-trained model **votes worse** than the non-CFG one — consistent with a
weaker conditional path from `p_uncond` dropout at 20k steps.

### 4.3 Annealed FKC — tempering monotonically hurts, ESS collapses
non-CFG, `edm_churn` g=0.4, K=16, n=512:

| β | pass@16 | maj@16 | min-ESS |
|---|---|---|---|
| 1.0 (=voting) | 93.6% | 90.8% | 16.0 |
| 1.005 | 91.2% | 88.9% | 7.4 |
| 1.01 | 89.7% | 87.3% | 6.4 |
| 1.02 | 86.5% | 84.4% | 2.8 |
| 1.05 | 72.3% | 71.5% | 1.8 |

Every increase in β lowers pass@K and maj@K while ESS collapses toward 1. There is
no operating point where tempering beats β=1.

### 4.4 CFG-FKC (Prop 3.1) — full collapse
CFG checkpoint, `edm_churn` g=0.4, K=16, w∈{1.25,1.5,2,3}. All variants collapse:
`pass@K == maj@K` (all particles identical), min-ESS = 1, maj ≈ 60–66%. Plain CFG
guidance is itself flat on this task (guidance w=0→3 leaves single-sample accuracy
~61%), and the under-trained unconditional branch makes `∇log q_u` noisy, inflating
the `‖s_c − s_u‖²` weight → immediate collapse even at w=1.25. (n=128 band-test row:
w=1.5 none/0.8/0.5 → maj 64.1 / 63.3 / 60.9.)

### 4.5 Entropy-band resampling — does not rescue FKC
Anneal β=1.05, non-CFG, `edm_churn` g=0.4, **n=1000**:

| band | p_mean | top_w | bot_w | wvote | maj@16 | pass@16 | min-ESS | distinct |
|---|---|---|---|---|---|---|---|---|
| none | 66.9 | 67.2 | 66.9 | 72.5 | 72.2 | 73.2 | 1.73 | 2.70 |
| 0.8 | 67.0 | 67.1 | 66.6 | 75.6 | 72.6 | 73.4 | 1.00 | 2.09 |
| 0.5 | 64.7 | 64.9 | **67.5** | 83.5 | 75.8 | 76.5 | 1.00 | 1.91 |

The band does **not** lift ESS (it is ≤ the no-band case) and does **not** reliably
improve maj@K. Mechanism: the ESS-collapsing weight dispersion lives in the **low-σ
tail, outside the central band** (where the matched-filter score `(D−x)/σ²`
explodes), so band-gating removes resampling that was partially fighting the
collapse. An apparent band-0.5 uptick (maj 75.8) is **not significant** (~1.8σ at
n=1000, single seed), comes with *lower* per-particle accuracy (64.7) and *less*
diversity (1.91), and the large `wvote=83.5` is a **validity-selection artifact**
(`wvote` restricts to valid grids, and valid ≈ correct for unique-solution Sudoku).

### 4.6 The two decisive diagnostics
1. **The FKC weight does not rank samples by correctness.** Across all cells
   `top_weight_accuracy ≈ mean_particle_accuracy`, and at band 0.5 the lowest-weight
   particle was *more* accurate than the highest (67.5 > 64.9). A correctness-
   correlated weight would give `top_w ≫ bot_w`; we see equality-to-inversion.
2. **Coverage is bottlenecked by the distinct-sample count, which resampling
   destroys.** β=1 keeps 5.45 distinct grids of 16 → the correct one is present 93.6%
   of the time. Tempering collapses this to ~2 distinct → coverage falls to ~73%.
   The `pass@16` label is misleading — effective coverage is `pass@~2`. (Note even
   β=1 yields only 5.45/16 distinct: the base model's stochastic diversity is already
   limited, so K has diminishing returns — K=32 barely beat K=16.)

---

## 5. Interpretation

The results are internally consistent and point to one conclusion: **on hard Sudoku,
the model's likelihood (and its CFG conditional/unconditional disagreement) is not
aligned with task correctness.** Therefore any scheme that concentrates probability
mass toward high-likelihood regions — annealing, CFG guidance, importance-weighted
SMC — cannot improve accuracy, and by collapsing particle diversity it destroys the
one thing that *was* helping (coverage from independent samples / voting). This is
exactly the plan's predicted *"confidence ≠ correctness"* regime.

Two mechanisms compound the failure: (i) the FKC weight `∝ Δσ²‖s‖²` is **extensive**
over 356 free bits and low-σ-dominated, so importance weights are hopelessly
degenerate (ESS→1) regardless of *where* we resample; (ii) the corrector concentrates
onto **confident** modes, which are not disproportionately **correct** modes.

Track A1 helps precisely because it is a *gentle, deterministic* sharpening with **no
importance weights to degenerate** and no resampling to collapse coverage.

---

## 6. What is robust vs. uncertain

**Robust (multiple runs / large n / clean mechanism):**
- Tempering (β>1) monotonically degrades pass@K & maj@K (n=512 sweep).
- FKC weight is not a correctness ranker (`top_w ≈ p_mean`, n=1000).
- Distinct-sample collapse (5.45 → ~2) fully explains the coverage loss.
- A1 monotone gain (n=2000).
- β=1 voting ≫ every FKC variant (15–19 pts).

**Uncertain / caveats (threats to validity):**
- **20k-step checkpoints only.** Both models are under-trained; the CFG model
  especially (dropout). A longer-trained CFG model could change the CFG-FKC picture
  (noisy `∇log q_u`). CFG-FKC should be re-checked on a better checkpoint before
  being called a definitive negative.
- **Single fixed seed (42)** for most particle runs; the band-0.5 "uptick" is within
  noise.
- Some CFG/voting runs are **interim** (killed at n<2000) — levels approximate, trend
  clear.
- **Sudoku-specific:** the rule-checker is a near-perfect correctness oracle here
  (valid + clue-consistent ⟺ correct). Conclusions about *reward* signals will differ
  on GSM8K, where the analogous reward is only a proxy.

---

## 7. Reproduction

```
conda activate pytorch;  export PYTHONPATH=.      # repo root
python tests/test_fkc.py                          # 16 gates
python tests/test_score_temp.py                   # A1 gates

# A1 sweep (deterministic, raw weights)
SUDOKU_DIFFICULTY=hard python -m evaluation.tasks.sudoku_eval \
  --config configs/tasks/sudoku_bits.py --checkpoint <nocfg>/step=000020000.pt \
  --difficulty hard --sampler deterministic --gamma 0 --ema 0 --steps 180 \
  --limit 2000 --score_temp_tau 0.5

# beta=1 voting (K independent samples)
... --sampler_kind fkc_em --proposal edm_churn --churn_gamma 0.4 --beta 1.0 \
    --num_particles 16 --ema 0

# annealed FKC / CFG-FKC / entropy band
... --beta 1.05                             # annealed
... --beta 1.0 --guidance_scale 1.5         # CFG (Prop 3.1), needs cfg ckpt
... --resample_entropy_frac 0.8             # confine resampling to central band
```
Result JSONs land under `runs/.../sudoku_eval/` (git-ignored). See
`docs/fkc_temperature_findings.md` for the condensed summary.

---

## 8. Recommended next action — Track C (reward-guided sampling)

The evidence says: stop concentrating on *likelihood*, start concentrating on a
*correctness-correlated reward*. For Sudoku the rule-checker is an exact, label-free
reward (uses only the puzzle rules + clamped clues, never the ground truth). Plan:

- **Phase 0 — verification / best-of-K (near-free, do first).** We already decode all
  K particles; add "output a valid + clue-consistent particle if any (else maj@K)".
  Since valid ≈ correct, this **realises pass@K**: non-CFG K=16 → ~94% vs 91% maj. It
  is the honest label-free baseline everything else must beat, and it directly
  attacks the coverage bottleneck (§4.6). Foreshadowed by the `wvote` result (§4.5).
- **Phase 1 — reward-guided SMC (SVDD-style, arXiv:2408.08252).** Reuse the FKC
  particle loop but **replace the likelihood weight with an external reward** on the
  decoded x0-estimate: `log w_k ∝ λ·r(decode(D_k))`, resample toward it, branch via
  churn. Reward = graded constraint satisfaction (dense) on the generated suffix.
  Schedule λ to low–mid σ (where `D_k` is a meaningful grid). Because r *is*
  correctness (unlike the FKC weight), resampling should now concentrate onto
  *correct* modes and raise pass@K — the hypothesis to test, vs Phase-0 verify@K and
  vs β=1 voting at matched K/NFE. `smc.py` is reused unchanged; ~1 day.
- **Phase 2 — refinements:** reward form, λ schedule, per-particle look-ahead count
  (SVDD's M), combine with A1; optionally DTS tree search (arXiv:2506.20701).
- **Phase 3 — Track D (learned value).** Train `v_φ(x_σ,σ,c) ≈ Pr(correct|x_σ,c)` on
  rollouts; Doob-guided score `s⋆ = s_θ + ∇log h_σ`. Deployable (not oracle at test).
  This is the path that generalises to GSM8K, where the reward is only a proxy.

**Open questions for the team:**
1. Is the *guidance* (Phase 1) worth it over trivial *verification* (Phase 0) on
   Sudoku, given the checker is a near-oracle? (Likely yes only where independent-K
   coverage is poor — the hardest puzzles.)
2. Should we first retrain a longer CFG checkpoint to give CFG-FKC / verification a
   fair unconditional branch?
3. Is the real target GSM8K (proxy reward → Track D value learning), with Sudoku only
   as the mechanism-validation testbed?
