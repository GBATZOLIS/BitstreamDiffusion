# Temperature scaling & Feynman–Kac correctors on CoBit: findings

Investigation of principled low-temperature / annealed sampling for CoBit
(continuous VE diffusion over analog bits), on the Sudoku task. Motivated by the
observation that naive bit/token posterior-temperature decoding is not principled
for continuous diffusion (it collapses below T≈0.25). We implemented and evaluated
three increasingly principled approaches, plus a resampling-band refinement.

All Sudoku numbers below are **hard** difficulty, **raw (non-EMA) weights** (EMA
collapses under deterministic sampling at 20k steps), 180 sampling steps unless
noted. Reference baselines to beat: single-sample deterministic ≈ 41% (non-CFG) /
55% (CFG); best stochastic-churn single sample ≈ 67%.

## Methods implemented

- **Track A1 — local score-level temperature** (particle-free, 0 extra NFE).
  Rescales the PF-ODE score by `κ(σ) = (v+σ²)/(τv+σ²)`, `v=0.25` (clean-bit
  variance, *not* the EDM `sigma_data`). κ→1 at high σ (mode allocation
  untouched), κ→1/τ as σ→0 (late sharpening only). τ=1 is a bit-identical no-op.
  Base posterior is retained for self-conditioning / decoding / entropy.
- **FKC — Feynman–Kac Euler–Maruyama sampler** (`smc.py` +
  `FeynmanKacEulerMaruyamaSampler`). Runs K weighted particles per prompt and
  resamples toward the FKC potential.
  - **Anneal-FKC** (`beta>1`, no guidance): single-model tempering `p_c^β`.
    Proposal = β-scaled entropy-gated reverse-SDE drift; weight
    `½β(β−1)Δσ²‖s‖²_free`. This is the **target-score simulation** (a=0) of
    Skreta et al. (arXiv:2503.02819), generalised to CoBit's entropy-gated λ(σ);
    noise is *not* tempered by 1/√β, so it is not the tempered-noise (a=1/2)
    variant.
  - **CFG-FKC** (`guidance_scale=w>0`, `beta=1`): Prop 3.1 two-model geometric
    average `q_u^{1−w} q_c^w`. Drift = standard CFG at weight w; weight
    `½w(w−1)Δσ²‖s_c−s_u‖²_free`. Requires a conditioning-dropout checkpoint.
  - Two proposals: `em` (explicit Langevin, unstable at low NFE) and `edm_churn`
    (EDM churn before the denoiser, stable at 180 steps — used throughout).
  - The two modes are mutually exclusive and independently usable; `beta>1` with
    `guidance_scale>0` is rejected (Prop D.4 territory, out of scope).
- **Entropy-band resampling** (`resample_entropy_frac`). Confine resampling to the
  central entropy-rate band holding fraction f of the log-σ pdf mass (e.g. 0.8 →
  [q₀.₁, q₀.₉]); weights still accumulate everywhere, so the target is unchanged.

**Correctness gates (all green):** β=1,K=1 bit-identical to `EulerMaruyamaSampler`
(and to DDIM+EDM-churn); β=1 zero-weight/no-resample no-op; prompt invariance;
ancestry-gather consistency; duplicate-ancestor branching; fixed-u₀ systematic
resample; analytic 1-D Gaussian `Var→(v+σ²)/β` for λ∈{0,1,1/β,λ(σ)}; CFG w=1
reduces bit-identically to plain conditional; CFG w>1 reweights; β>1+guidance
rejected. See `tests/test_fkc.py`, `tests/test_score_temp.py`.

## Results

### A1 — the one thing that helps (small, monotone, no collapse)
Deterministic PF-ODE, non-CFG checkpoint, n=2000:

| τ | exact-match | valid |
|---|---|---|
| 1.0 (no-op) | 41.2% | 43.4% |
| 0.7 | 42.1% | 45.0% |
| 0.5 | 42.8% | 46.2% |
| 0.3 | 43.4% | 47.4% |

Monotone gain with sharpening (+2.2 / +4.0 pts), **no cliff** — the well-behaved
replacement for the old bit-space posterior-temperature (which collapsed below
T≈0.25). Modest, as expected for a particle-free local approximation.

### The winner is plain β=1 multi-sample voting (no tempering)
`edm_churn`, β=1 (⇒ no reweighting, no resampling), K independent samples:

| model | pass@16 | maj@16 |
|---|---|---|
| non-CFG | ~94% | **~91%** |
| CFG (dropout ckpt) | ~78% | ~74% |

The CFG-trained checkpoint votes **worse** than the non-CFG one — expected, since
`p_uncond=0.1` dropout splits an already-short 20k-step budget, under-training the
conditional path.

### Tempering / guidance / band all fail to beat voting
- **Anneal-FKC** (non-CFG, `edm_churn`): raising β monotonically *lowers* both
  pass@K and maj@K while ESS collapses (β=1→1.05: maj 91%→~72%, min-ESS 16→1).
- **CFG-FKC** (CFG ckpt, w∈{1.25…3}): fully collapses — pass@K == maj@K (all
  particles identical), min-ESS=1, maj ≈ 62–66%. Plain CFG guidance is itself flat
  on this task (w=0→3 leaves accuracy ~61%), and the under-trained unconditional
  branch makes `∇log q_u` noisy, inflating the `‖s_c−s_u‖²` weight → instant
  collapse.
- **Entropy band** (anneal β=1.05, n=1000): does **not** help and slightly hurts
  ESS — the collapsing weight dispersion lives in the low-σ tail *outside* the
  central band, so band-gating removes the resampling that was partially fighting
  the collapse.

### The decisive diagnostic: the FKC weight is not correlated with correctness
Anneal β=1.05, `edm_churn`, n=1000 (`top_w`/`bot_w` = accuracy of the single
highest/lowest-weight particle; `wvote` = weight-weighted vote):

| cell | p_mean | top_w | bot_w | wvote | maj@16 | pass@16 | min-ESS | distinct sols |
|---|---|---|---|---|---|---|---|---|
| no band | 66.9 | 67.2 | 66.9 | 72.5 | 72.2 | 73.2 | 1.73 | 2.70 |
| band 0.8 | 67.0 | 67.1 | 66.6 | 75.6 | 72.6 | 73.4 | 1.00 | 2.09 |

`top_w ≈ p_mean ≈ bot_w`: the highest-weight particle is **no more accurate than a
random or the lowest-weight particle** — the FKC importance weight carries no
per-particle correctness signal. The band gives **no maj@K gain** (72.6 vs 72.2)
and reduces diversity. (A promising n=128 signal — band 0.8 maj 78.1 vs 73.4 —
did **not** survive at n=1000; it was noise.) The only residual is a ~3 pt
`wvote > maj` lift, i.e. a faint aggregate signal, still far below voting.

## Conclusion

Across A1, annealed-FKC (target-score simulation), CFG-FKC (Prop 3.1 geometric
average), and entropy-band resampling, **no tempering / guidance / importance-
weighting scheme beats plain β=1 multi-sample voting** on hard Sudoku, and most
collapse particle diversity (the actual source of gains). The clean proof is that
the FKC weight does not rank samples by correctness (`top_w ≈ p_mean`).

This is the predicted **"confidence ≠ correctness"** regime: the model's likelihood
(and its CFG conditional/unconditional disagreement) is not aligned with task
correctness, so concentrating probability mass cannot close the maj@K → oracle
gap. Doing so requires a **correctness-correlated reward or learned value signal
(Track C/D)**, not temperature. A1 remains the only useful lever here — a small,
safe, deterministic sharpening — because it adds no importance weights to degenerate.
