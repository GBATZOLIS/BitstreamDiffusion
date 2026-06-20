# EM TinyGSM/GSM8K collapse — root-cause analysis (no-checkpoint, code + numerics)

**Branch:** `tasks/sudoku-tinygsm` @ `1efa2b8`  ·  **Sampler:** `EulerMaruyamaSampler`
(`diffusion/continuous/samplers.py:1982`)  ·  **Reproducer:** `reports/em_scale_sim.py`
(drives the *real* `SigmaSchedule` + `EntropyRateLambdaProfile`; no model/checkpoint needed).

> Scope note: the TinyGSM 250k checkpoint and its entropy tables live on CSD3, not on this
> workstation, so the GSM8K eval itself was **not** re-run here. Everything below is a code
> audit plus a faithful numerical reproduction of the per-step noise/drift profile, using a
> synthetic entropy table reconstructed to match the **exact** stated stats (K=128,
> σ∈[0.00209, 76.76], as_saved table peak **0.362 @ σ≈1.85**, mean 0.094, Δlogσ=0.08279,
> Σpdf=1 — all reproduced to 3 sig figs).

---

## TL;DR — the headline lead is half right, and the fix is *not* "scale λ₀ down 100×"

1. **The discretization `sqrt(2·λ·σ·Δσ)` is correct.** It is the exact Euler–Maruyama
   increment of `sqrt(2λσ)·dW_r` with `r=−σ` (Brownian increment variance `|dr|=Δσ`). There is
   **no missing `1/σ` or `Δlogσ` factor** (hypothesis #3 is a dead end). The log-σ-ness of
   `π_α` is fully absorbed by the entropy-grid *step spacing*, not by anything in
   `_integrate_step`.

2. **The asymptotic identity `λ_ent = S_churn·π_α` is correct, and the bulk at λ₀=419 is
   genuinely EDM-equivalent.** On the *actual* entropic (inverse-CDF) grid, the per-step
   relative noise in the entropy-mass band (σ∈[0.5,10]) is **std/σ ≈ 0.91** at λ₀=419 — which
   sits exactly on EDM's operating point (γ0.34→0.892, γ0.41→0.994). The bulk is **not 100×
   too hot.** Scaling λ₀ down ~100× (→≈4) would make the bulk `g=λ₀/N≈0.004` — essentially
   deterministic — throwing away the entire point of EM and reverting toward plain DDIM.

3. **The collapse driver is a low-σ *tail* artifact, not the entropy peak.** The handoff's
   "≈1.6σ per step" *magnitude* is right; its *location* (σ≈1.85) is wrong. The over-noising
   is at σ≈0.06, and at NFE=1024 it is **a single step** — the final grid step
   (σ≈0.058 → σ_min), where the inverse-CDF grid must span a huge log-gap with no entropy mass,
   giving Δσ=0.056 (a **10× jump** over its ~0.006 neighbours) and injecting **√2·σ right
   before decode** — destroying the bit-resolving step. The handoff's Δσ≈0.019 is the
   *uniform-log* spacing; the real entropic Δσ at σ≈1.85 is **0.0048** (steps are densest at the
   peak), which is exactly why the peak is *not* where it blows up.

4. **The failed sweep ran with NO per-step cap (raw `sqrt(2λσΔσ)`).** The `em_step_gamma_cap`
   block in `samplers.py:2016-2066` is **uncommitted working-tree state — it is NOT in commit
   `1efa2b8`** (verified: `git show 1efa2b8:…samplers.py` has bare
   `x_det = x_state + h*(1+lam)*d_cur`, no cap; the user's prompt cites the no-cap noise line
   `samplers.py:2035`). So the collapse numbers are the **RAW** column below. Someone has since
   added a `cap=1.0` locally, but **`cap=1.0` is too loose**: it bounds per-step noise to only
   `√2·σ ≈ 1.41σ` (still catastrophic), and at λ₀∈{340,380} it does not even engage
   (`g_raw≈0.88<1` at the bad step), so it does not save the run. EDM's own per-step stability
   bound is `γ ≤ √2−1 ≈ 0.41`. **The cap should match EDM's, not be 2.4× looser.**

---

## The numbers (from `reports/em_scale_sim.py`, real code paths)

### Per-step relative noise on the actual entropic grid

| NFE | λ₀ | bulk σ∈[0.5,10] mean std/σ | trajectory max std/σ | where max occurs | #steps std/σ>1 |
|----:|---:|--------------------------:|---------------------:|------------------|---------------:|
| 1024 | 419 | **0.906** (≈EDM γ0.41) | 1.469 (1.414 capped) | σ≈0.058 (final step) | **1 / 1023** |
| 1024 | 340 | 0.816 | 1.324 | σ≈0.058 (final step) | **1 / 1023** |
| 1024 | 50  | 0.313 | 0.508 | σ≈0.058 | 0 |
| 1024 | 10  | 0.140 | 0.227 | σ≈0.058 | 0 |
| 256  | 419 | ~1.8 (g=λ₀/N=**1.64**) | 2.80 | σ≈0.095 | **254 / 255** |

Last steps at λ₀=419, NFE=1024 — everything is EDM-like (~0.88) until the final step jumps:

```
step1021  σ=0.0739  Δσ=0.0154  λ=2.12  std/σ=0.94
step1022  σ=0.0584  Δσ=0.0564  λ=1.12  std/σ=1.47   <-- the one bad step (Δσ 10x its neighbours)
```

### Why the tail and not the peak

On the entropic inverse-CDF grid the per-step log-spacing is `Δlogσ ≈ 1/(N·π_α)`, so the
`λ ∝ π_α` and `Δσ ∝ σ/π_α` factors **cancel** → uniform `std/σ = √(2λ₀/N)` across the bulk.
The cancellation only holds where the grid can actually track `π_α`. In the low-σ tail `π_α→0`,
but the grid is **truncated to σ_min** with finite N, so the last 1–2 steps span a large
log-gap the density says shouldn't have any steps. There the cancellation breaks and
`g = λ·Δσ/σ` spikes to ≈1.

### The NFE-scaling trap (important for the planned experiments)

`λ₀ = S_churn = γ·(NFE−1)` is **NFE-specific**, because bulk churn is `g = λ₀/N`:
- λ₀=419 @ NFE1024 → g=0.41 (EDM-equivalent, healthy bulk, 1 bad tail step).
- λ₀=419 @ NFE256 → g=**1.64** (over-noised on **every** step — a different failure).

So a low-λ₀ scan at NFE256 does **not** probe the same per-step regime as the failed NFE1024
runs. To match a healthy EDM γ0.41 at NFE256 you want **λ₀ ≈ 0.41·255 ≈ 105**, not 419. Sweep
in units of `g = λ₀/(NFE−1)`, not raw λ₀.

---

## The one thing this analysis cannot settle without the checkpoint

At NFE1024/λ₀=419 the bulk is EDM-equivalent and **only one step** is over-noised. Two
hypotheses remain, and they predict opposite fixes:

- **(A) The single tail step destroys the output.** It is the *bit-resolving* step, and a √2·σ
  kick at σ≈0.06 right before decoding raw binary bits is plausibly enough to scramble tokens
  → "pure token noise." If so, EM-above-cap is viable and the fix is a 2-line tail clamp.
- **(B) EM's explicit integrator is fundamentally weaker than EDM churn**, independent of the
  tail. EDM *adds-then-denoises* (noise added at σ, then the denoiser is re-evaluated at the
  churned σ̂ and removes it). EM evaluates the score at σ_cur and adds noise **after**, with no
  compensating denoise at the matching level — error can accumulate over 1023 steps even at
  g=0.41. If so, EM-above-cap is not viable as-is and **PC is the right vehicle** (its corrector
  pulls back toward the manifold and its predictor already lands near data — consistent with PC
  succeeding on Sudoku where standalone EM was never validated).

Magnitude analysis alone cannot distinguish these. The experiments below do.

---

## Decisive experiments (in priority order)

All cheap; first three isolate the mechanism.

1. **Determinism gate (must pass first).** `em --lambda_zero 0 --steps 256` must equal
   deterministic DDIM @256 = **0.0917 (121/1319)** *bit-identically*. Confirms no graft/RNG bug.
   (Code path looks clean: λ₀=0 returns `x + h·d` with no `randn` call — `samplers.py:2055`.)

2. **The (A)-vs-(B) disambiguator.** `em --steps 1024 --lambda_zero 419` with the tail clamped —
   either `--em_step_gamma_cap 0.41` (once exposed as a flag; see fix) or a low-σ noise floor.
   - Recovers ≈25% (EDM γ0.41) → **(A)**, ship the tail clamp.
   - Still collapses → **(B)**, pivot to PC.

3. **Per-`g` scan at NFE256**, sweeping `g = λ₀/255 ∈ {0, 0.05, 0.1, 0.2, 0.41}` i.e.
   `λ₀ ∈ {0, 13, 26, 51, 105}` (NOT {340…500} — those are g≈1.3–2.0 at NFE256 and will all
   collapse trivially). Expect monotone-then-peak accuracy if (A); collapse even at g≈0.05 ⇒
   graft bug or (B).

4. **PC at matched g** (`pc --lambda_zero 105 --steps 256`, predictor-only guidance) to confirm
   the corrector path is stable where standalone EM is not.

---

## Recommended fix

**Tighten the EM per-step stability cap to EDM's bound and expose it.** In the sim, at λ₀=419
NFE1024:

| `em_step_gamma_cap` | trajectory max std/σ | bulk mean std/σ |
|--------------------:|---------------------:|----------------:|
| 1.0 (current default) | 1.414 | 0.906 |
| **0.41 (≈√2−1, EDM)** | **0.906** | 0.899 |
| 0.325 | 0.806 | 0.806 |

`cap=0.41` removes the tail spike **while leaving the EDM-equivalent bulk untouched** — the
principled default. Optionally add an explicit low-σ noise floor (zero λ below σ≈0.3, mirroring
EDM's `S_tmin` gating) for belt-and-braces. **Do not** simply lower λ₀: that throws away the
bulk stochasticity that is, by construction, correct.

`em_step_gamma_cap` is currently a constructor arg (`samplers.py:2008`) hard-defaulted to 1.0
and **not** plumbed to the CLI — so the failed sweep had no way to change it. Plumb it through
`load_model_and_sampler` → `--em_step_gamma_cap` and default it to 0.41.

---

## Local validation on the Sudoku-easy checkpoint (GPU, real eval internals)

Ran the actual `load_model_and_sampler`/`sample_bits` path on
`runs/tasks/sudoku/easy/cobit_raw_binary_bits` (128 steps, 64 val puzzles, seed 42):

- **Gate 1 (determinism) PASSES bit-identically:** `em --lambda_zero 0` == deterministic DDIM,
  **n_diff_bits = 0**. The EM graft is correct; no graft/RNG bug.
- **EM and PC are finite and stable** at every λ₀ tested (50 → 1500), with and without the cap —
  no NaN, no blow-up.
- **The cap behaves exactly as the analysis predicts**, measured on the *solution suffix* (the
  prefix is clamped to ground truth, so whole-sequence stats are prefix-dominated; suffix-only is
  the right probe). Fraction of suffix bits flipped vs the deterministic baseline:

  | λ₀ | raw (cap=None) | cap=1.0 | cap=0.41 |
  |---:|---------------:|--------:|---------:|
  | 200 | 0.201 | 0.201 | **0.114** |
  | 600 | 0.201 | 0.201 | **0.114** |
  | 1500 | 0.201 | 0.201 | **0.114** |

  `raw ≡ cap=1.0` (the loose cap is a no-op on decoded output — corroborating point 4 above),
  while **cap=0.41 bounds the suffix tightly to the deterministic manifold** independent of λ₀.

**What this does and does not establish.** It validates *correctness and mechanism* — the
samplers are grafted correctly, run stably, and the 0.41 cap controls over-noising as designed.
It does **not** reproduce the TinyGSM *accuracy* collapse: the Sudoku-easy checkpoint is weak
(deterministic exact-match 0/64 here) and, even raw at λ₀=1500, its suffix only perturbs ~20%
(not → random) because that small model + its entropy table + the robust grid decode don't sit
in the brittle regime that a strong 250k TinyGSM model + structured-text decode do. Confirming
the *accuracy* fix requires the TinyGSM checkpoint on CSD3 (experiment #2).

## Shipped in this pass (code)

- **CUDA guard** on `evaluation/tasks/gsm8k_eval.py` and `sudoku_eval.py`: the eval now raises
  if `torch.cuda.is_available()` is False, unless `--allow_cpu` / `{GSM8K,SUDOKU}_ALLOW_CPU=1`.
  Stops the silent ~100× CPU fallback that invalidated a bad-node run.
- `reports/em_scale_sim.py`: the standalone reproducer behind every number above.
