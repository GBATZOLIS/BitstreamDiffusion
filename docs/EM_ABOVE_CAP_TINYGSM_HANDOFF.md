# Handoff: Euler–Maruyama (EM) sampler to exceed the EDM churn cap on TinyGSM/GSM8K

**Audience:** the Claude Code instance running TinyGSM experiments on the HPC.
**TL;DR:** Your best unguided result (γ0.41 @ NFE1024 ≈ 26.1%) is pinned against the EDM
per-step churn cap γ ≤ √2−1 ≈ 0.4142. The newly-landed **EulerMaruyamaSampler (`--sampler_kind em`)**
implements the *same* entropy-gated reverse SDE but injects noise in x-space — **no √2−1 cap** —
so it can reach **effective churn above the ceiling**. This handoff explains the math, the exact
commands, and a gated experiment plan to push λ₀ past the cap and (likely) unlock higher accuracy.

Pull first: `git fetch && git checkout tasks/sudoku-tinygsm && git pull --ff-only`.

---

## 1. Why this should work (the mechanism)

EDM churn and EM solve the *same* reverse SDE in the small-step limit
(see `docs/EM_PC_SAMPLER_PLAN.md` and the entropy-gated-SDE note):

    dx = (1 + λ(σ))·σ·s_θ(x,σ) dr + sqrt(2·λ(σ)·σ) dW_r

- **EDM churn** reaches noise level σ̂ = σ(1+γ) then denoises back. It is **hard-capped at
  γ ≤ √2−1** (so per-step variance ≤ ~2σ²) to avoid σ̂ overshoot / OOD. That cap is exactly
  what γ0.41@1024 is hitting.
- **EM** adds the Langevin noise directly in x-space at the on-grid σ: `x_new = x_det +
  sqrt(2·λ·σ·Δ)·z`. It never forms σ̂, so there is **no cap** — λ₀ can grow arbitrarily
  (until genuine finite-step blow-up at very high λ, which you detect empirically).

The asymptotic identity is `λ_ent(σ) ≈ S_churn · π_α(log σ)` with `S_churn = γ·(NFE−1)`. So with
`--lambda_normalize as_saved`, **λ₀ is the EDM-equivalent cumulative churn**:

    λ₀  ==  S_churn  ==  γ · (NFE − 1)

At **NFE=1024** (num_intervals = 1023):
- γ0.41 (your current best)  ↔  **λ₀ ≈ 419**
- the cap γ=√2−1=0.4142      ↔  **λ₀ ≈ 424**  ← the wall EDM can't pass
- "effective γ" of any λ₀     =  λ₀ / 1023

So **anything λ₀ > ~424 is above what EDM can do.** That is the experiment.

---

## 2. What landed (and is verified) on the branch

New / changed (all backwards-compatible; `ddim`/`heun` and every existing script unchanged):
- `diffusion/continuous/lambda_profiles.py` — `LambdaProfile`, `EntropyRateLambdaProfile`,
  `make_lambda_profile`, entropy-table loader. Ported verbatim from the validated reference.
- `diffusion/continuous/samplers.py`:
  - `DDIMSampler._integrate_step(...)` — extracted the PF update (behavior-preserving).
  - **`EulerMaruyamaSampler`** — subclass; overrides only `_integrate_step` with the λ-gated
    Langevin term; **refuses EDM churn** (raises if `stochastic.enabled & s_churn>0`); at
    `lambda_zero=0` makes NO `randn` call ⇒ bit-identical to deterministic DDIM.
  - `PredictorCorrectorSampler` — PF predictor + Langevin corrector, with **predictor-only
    guidance** (`guidance_mode=predictor_only`: predictor guided, corrector at the conditional
    score) for the *guidance-under-noise* experiments later. NFE/step = 2.
- `evaluation/tasks/_task_common.py` — `load_model_and_sampler(..., sampler_kind, lambda_zero,
  lambda_profile, lambda_normalize, guidance_mode)`; `em`/`pc` branches.
- `evaluation/tasks/gsm8k_eval.py` and `sudoku_eval.py` — new CLI flags `--sampler_kind {em,pc}`,
  `--lambda_zero`, `--lambda_profile {entropy_rate,flat}`, `--lambda_normalize {as_saved,peak}`,
  `--guidance_mode {predictor_only,all}`. Recorded in result JSON + filename tag
  (`..._kindem_lz<λ0>_as_saved`) so EM runs never collide with DDIM.

**Gates already passed (on Sudoku, the dev task):**
- Gate 0: DDIM after the refactor reproduces its golden bit-identically.
- Gate 1: `em lambda_zero=0` == deterministic DDIM bit-identically.
- PC λ₀=0 deterministic is sane; PC predictor-only-guidance smoke runs clean.

`sigma_data`, the matched-filter head, prefix-clamp, entropy-grid schedule, and self-conditioning
all carry over unchanged (EM/PC reuse DDIM's exact denoise path). The trained `sigma_data` sidecar
(your TinyGSM ~0.3998) is auto-resolved exactly as before.

---

## 3. How to run EM (IMPORTANT usage detail)

EM owns stochasticity through **λ₀**, not through γ. So you run it **deterministic (churn OFF)
+ lambda_zero**:

```
python -m evaluation.tasks.gsm8k_eval \
  --config <your tinygsm eval config> --checkpoint <ckpt.pt> \
  --sampler deterministic            \   # churn OFF — EM refuses churn
  --sampler_kind em                  \   # the entropy-gated SDE sampler
  --schedule entropic --steps 1024   \   # NFE = 1024 (matches your sweep; 1 denoiser call/step)
  --ema 1 --seed 42 --limit 1319     \
  --lambda_zero <λ0> --lambda_profile entropy_rate --lambda_normalize as_saved
```

- Do **NOT** pass `--sampler stochastic --gamma` to EM (it will raise — that's the churn path).
- `--steps 1024` ⇒ NFE 1024, **1 denoiser call/step**, apples-to-apples with your DDIM γ-sweep.
- Requires the entropy tables (`entropy_pdf.pt`, `entropy_sigmas.pt`, `entropy_edges.pt`) in the
  run dir (checkpoint `parent.parent`). Verify they exist for your TinyGSM run; the EM profile
  loader asserts ascending σ / K≥2 / edges=K+1. (Your EDM-entropic runs already used these.)

---

## 4. Validation gates — DO THESE ON YOUR CHECKPOINT FIRST (in order)

Trust no above-cap number until Gates 1–2 pass on the *actual* TinyGSM checkpoint:

- **Gate 1 (EM determinism):** `em lambda_zero=0` (NFE256 or your det config) must equal the
  **deterministic DDIM** accuracy bit-identically (same seed). Confirms the graft is faithful.
- **Gate 2 (EDM-equivalence anchor — the critical one):** `em entropy_rate as_saved
  lambda_zero ≈ 419` at NFE=1024 should **approximately reproduce your EDM γ0.41 result (~26.1%)**.
  Because the λ₀↔γ map is asymptotic, sweep λ₀ ∈ {380, 419, 460} and confirm one lands near 26%.
  This empirically pins the λ₀ scale on *your* model + entropy table. **If Gate 2 doesn't bracket
  ~26%, stop and recheck the entropy table / normalize before going above-cap** — don't trust the
  constant blindly.
- **Gate 3 (the experiment):** only after Gate 2, sweep λ₀ ABOVE the cap (≈424).

---

## 5. The above-cap experiment (NFE=1024, unguided)

Anchor at λ₀≈419 (Gate 2), then climb. Suggested λ₀ grid (effective γ = λ₀/1023 in parens):

| λ₀ | eff. γ | regime |
|----|--------|--------|
| 419 | 0.41 | Gate-2 anchor (≈ EDM best) |
| 500 | 0.49 | first above-cap point |
| 650 | 0.64 | above-cap |
| 850 | 0.83 | above-cap |
| 1100 | 1.08 | high — watch for blow-up |
| 1500 | 1.47 | likely near/over the blow-up ceiling |

**Expected shape (from the OWT entropy-gated-SDE study):** accuracy should keep climbing past the
cap, then eventually **finite-step blow up** at high λ₀ (in that study EM walked up to ~8·S_churn
before collapsing at ~12·S_churn). So sweep upward and stop when you see the blow-up signature:
`invalid_token_rate` spikes and accuracy collapses. The optimum is the last stable point before that.

Run them as independent single-GPU jobs (one λ₀ each), exactly like your γ-sweep, but with the
EM invocation in §3. Start with {500, 650, 850} (most likely to contain the new optimum given
γ0.41 was still rising at the edge), then add {1100, 1500} to find the ceiling.

**Reporting:** for each λ₀ report accuracy, CI, `invalid_token_rate`, and the implied eff-γ. The
headline question: does the best EM λ₀ exceed the EDM-capped 26.1%? If yes by a clear margin,
that's a real unlock (and worth a PC run too — its corrector can add even more stable Langevin).

---

## 6. Caveats / gotchas

- **NFE accounting:** EM = 1 denoiser call/step (same as DDIM) → "NFE 1024" is honest. PC = 2/step
  (predictor+corrector), so PC at "NFE 1024" costs 2× — report *effective* NFE if you compare.
- **EM refuses churn by design.** Use `--sampler deterministic --lambda_zero X`. If you accidentally
  pass `--gamma>0 --sampler stochastic`, EM raises a clear error.
- **λ₀ scale is `as_saved`-specific.** `peak` normalization means something different (λ₀ = peak
  Langevin strength). Use `as_saved` for the γ-anchored sweep so the table above is valid.
- **Blow-up is real at high λ₀, especially if NFE were lower.** At NFE1024 you have the most
  headroom (Δσ ∝ 1/N keeps per-step variance bounded), which is *why* high-NFE is the right place
  to exploit the missing cap — consistent with your finding that high-γ helps most at NFE1024.
- **CFG resume:** unrelated to this, but you flagged it dropped off the queue — verify separately.

---

## 7. Suggested first commands (copy/paste, edit paths)

```
# Gate 2 anchor (must ~reproduce EDM gamma0.41 ~26.1%)
for LZ in 380 419 460; do
  CKPT=... python -m evaluation.tasks.gsm8k_eval --config <cfg> --checkpoint $CKPT \
    --sampler deterministic --sampler_kind em --schedule entropic --steps 1024 \
    --ema 1 --seed 42 --limit 1319 \
    --lambda_zero $LZ --lambda_profile entropy_rate --lambda_normalize as_saved ; done

# Gate 3 above-cap (after Gate 2 brackets ~26%)
for LZ in 500 650 850 1100 1500; do  ... same line, --lambda_zero $LZ ... ; done
```

Results land in `<run_dir>/gsm8k_eval/gsm8k_results_deterministic_g0.0_w0_s1024_sd<...>_ema1_kindem_lz<LZ>_as_saved.json`.
