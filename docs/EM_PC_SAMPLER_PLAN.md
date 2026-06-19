# Plan: port EM + PC entropy-gated SDE samplers into BitstreamDiffusion

**Status:** design only — implement carefully next session.
**Goal:** add Euler–Maruyama (EM) and Predictor–Corrector (PC) reverse-SDE samplers
so we can inject **stochasticity beyond the EDM per-step churn cap (√2−1 ≈ 0.4142)**,
which NFE=512 is currently bumping against (250k: γ=0.4142 → **23.12%**, still the best
and still rising at the cap). Must be **backwards compatible**: `ddim`/`heun` and every
existing eval script keep working unchanged; EM/PC are *new* `sampler_kind` options.

Reference implementation (already written, validated elsewhere):
`CD_Experimental` @ `origin/multimodal-cc3m-joint-bitstream`, package
`diffusion/continuous/samplers/{euler_maruyama,predictor_corrector,lambda_profiles,base,common,stochastic,schedule}.py`.

---

## 1. The algorithm (EM)

Reverse-SDE family, discretized σ_i → σ_{i+1} (`h = σ_next−σ_cur < 0`, `Δ = σ_cur−σ_next > 0`):
```
score = (D − x)/σ²              # D = denoiser x0-hat (from probs)
d     = −σ·score
λ_i   = λ(σ_i)                  # from LambdaProfile; NO √2−1 cap
x_det = x + h·(1+λ_i)·d
x_new = x_det + √(2·λ_i·σ_i·Δ)·z ,  z ~ N(0,I)     # z masked to 0 on the prompt prefix
```
- **λ₀ = 0 ⇒ x_new = x + h·d, and NO `randn` call** ⇒ bit-identical to deterministic
  `DDIMSampler`. This is the determinism gate.
- Stochasticity is owned **entirely** by `λ(σ)`; EM must **refuse** EDM churn
  (`s_churn>0` → hard error), mirroring the reference.
- **Key advantage over EDM churn:** EM adds noise in *x-space* at the on-grid σ; it never
  forms `σ̂ = σ(1+γ)`, so there is **no σ>σ_max OOD problem** and **no cap** — exactly why
  it's the principled way to exceed the churn ceiling.

**PC (predictor–corrector):** EM predictor step + one or more Langevin **corrector**
iterations at the new σ (extra denoiser passes ⇒ higher NFE per step). Read
`predictor_corrector.py` first thing in the impl session to get the exact corrector
step-size rule and NFE accounting before porting.

## 2. λ(σ) profile and the λ₀ ↔ γ mapping

`λ(σ) = λ₀ · table(log σ)`, `table` from the on-disk entropy tables
(`entropy_pdf.pt`, `entropy_sigmas.pt`, `entropy_edges.pt`) — **which our runs already
produce in the exact expected format** (ascending σ, pdf sums to 1; written by the entropy
schedule controller). Two normalizations:
- `peak`: `table = pdf/pdf.max()` → λ₀ is the **peak** Langevin strength.
- `as_saved`: `table = pdf/Δ(logσ)` → log-σ density `π_α(logσ)`; then the asymptotic
  identity `λ_ent(σ) ≈ S_churn·π_α(logσ)` holds **exactly** under **λ₀ = S_churn**.

Since EDM `s_churn = γ·(NFE−1)`, the **as_saved** mapping gives, at NFE=512:
`λ₀ ≈ γ·511`, so the cap γ=0.4142 ↔ **λ₀ ≈ 212**. We will NOT trust this constant blindly —
we anchor it empirically (Gate 2 below).

## 3. Files to add / modify (all additive or behavior-preserving)

**ADD** `diffusion/continuous/lambda_profiles.py`
- Port `lambda_profiles.py` ~verbatim (`LambdaProfile`, `FlatLambdaProfile`,
  `EntropyRateLambdaProfile`, `make_lambda_profile`, `_load_entropy_pdf_table`,
  `_delta_log_sigma`). Standalone; only depends on torch + the entropy `.pt` files.
- First verify our `entropy_pdf/sigmas/edges.pt` satisfy the loader's asserts
  (ascending σ, K≥2, edges = K+1). They appeared to when we ran EDM-entropic eval.

**MODIFY** `diffusion/continuous/samplers.py` (monolithic today; keep DDIM/Heun intact)
- **Refactor (behavior-preserving):** extract the shared per-step block from
  `DDIMSampler` — *denoise → probs (+ optional CFG 2B batching) → `_score_from_probs` →
  prefix clamp* — into a reusable method, e.g. `self._denoise_score(x_state, sigma_eval,
  x0_hat, ctx...) -> (score, probs, sc_carry)`. `DDIMSampler.sample` calls it; its outputs
  and RNG order must be **unchanged** (guard with the DDIM regression test, §5 Gate 0).
- **ADD** `class EulerMaruyamaSampler` reusing: `SigmaSchedule` (grid + entropy tables),
  the new `_denoise_score`, `_score_from_probs`, `_clamp_mask_`, `_zero_mask_`,
  `logits_to_x0_hat`, `_build_mask_conditioning`. Implement §1 step. Constructor kwargs:
  `lambda_profile_name='entropy_rate'`, `lambda_zero=0.0`, `lambda_profile_normalize='peak'`.
  Build the profile per `sample()` call from `entropy_run_dir` (default = ckpt
  `parent.parent`, same as the entropy-grid resolver). Raise on `s_churn>0`.
- **ADD** `class PredictorCorrectorSampler` (EM predictor + Langevin corrector). Port after EM.
- Keep `_compute_edm_gamma`/`_apply_edm_churn`/DDIM/Heun **unchanged**.

**MODIFY** `evaluation/tasks/_task_common.py`
- `load_model_and_sampler(..., sampler_kind)`: add `"em"`/`"euler_maruyama"` →
  `EulerMaruyamaSampler(...)`, `"pc"`/`"predictor_corrector"` → `PredictorCorrectorSampler(...)`,
  threading new params (`lambda_zero`, `lambda_profile`, `lambda_normalize`, PC corrector knobs).
  Leave `ddim`/`heun` branches as-is.
- `configure_stochastic`: for EM/PC force `stochastic.enabled=False`/`s_churn=0` (EM refuses
  churn); stochasticity comes from λ₀.
- `sample_bits`: already passes `entropy_run_dir`; ensure it forwards the sampler's needs.
  No churn for EM/PC.

**MODIFY** `evaluation/tasks/gsm8k_eval.py` (and mirror in `sudoku_eval.py`)
- Extend `--sampler_kind` choices: `ddim, heun, em, pc`.
- New args (default to no-op so existing calls are unchanged):
  `--lambda_zero` (float, default 0.0), `--lambda_profile` (`entropy_rate|flat`, default
  `entropy_rate`), `--lambda_normalize` (`peak|as_saved`, default `as_saved` for the γ-anchor),
  PC: `--pc_corrector_steps`, `--pc_corrector_snr` (read PC for exact names).
- Record `sampler_kind`, `lambda_zero`, `lambda_profile`, `lambda_normalize` in the result
  JSON; extend the output `tag` so EM/PC runs don't collide with DDIM (e.g.
  `..._kind{em}_lz{0.45}_{as_saved}_...`).

**MODIFY** `scripts/tasks/eval_gsm8k.sh`, `eval_gsm8k_sweep_csd3.slurm`, `eval_gsm8k_best_csd3.slurm`
- Pass through `SAMPLER_KIND`, `LAMBDA_ZERO`, `LAMBDA_PROFILE`, `LAMBDA_NORMALIZE`
  (all optional via `${VAR:+--flag "$VAR"}`), default behavior unchanged.

## 4. σ_data / sidecar / prefix-clamp

No change — EM reuses the same denoiser path, so `sigma_data` auto-resolution
(`resolve_sigma_data` → 0.3998 sidecar), matched-filter head, prefix clamping all carry
over unchanged. EM masks the injected noise `z` on the prompt prefix (`_zero_mask_`) and
re-clamps after the step, identical to how DDIM conditions.

## 5. Validation gates (do these IN ORDER before trusting any number)

- **Gate 0 — DDIM regression:** after the `_denoise_score` refactor, a `ddim` eval at a
  fixed seed must reproduce a pre-refactor result **bit-identically** (capture one JSON now,
  e.g. NFE256/γ0.26/250k = 19.26%, as the golden).
- **Gate 1 — EM determinism:** `em` with `lambda_zero=0` must reproduce the **deterministic
  DDIM** number bit-identically (250k NFE256 det = **9.17%**, NFE384 det = 9.25%). Same seed,
  same RNG draws (the λ₀=0 branch must not call `randn`).
- **Gate 2 — EDM-equivalence anchor:** `em`, `entropy_rate`, `normalize='as_saved'`,
  `lambda_zero ≈ γ·(NFE−1)` should *approximately match* the EDM-churn result at γ. Sweep λ₀
  around 212 at NFE=512 to find the value reproducing ~23.1% — confirms port + mapping.
- **Gate 3 — above-cap test (the actual experiment):** with the anchor pinned, sweep
  **2 λ₀ values above the anchor** at NFE=512 on the frozen 250k snapshot; compare to 23.12%.
  If it climbs → stochasticity beyond the cap helps (and PC is worth running); if flat/down →
  we were already near the stochasticity optimum and the cap wasn't the limiter.

## 6. Next-session run matrix (NFE=512, 250k snapshot, σ_data=0.3998)

1. Gate 1: `em lambda_zero=0` (NFE256) → must equal 9.17%.
2. Gate 2: `em entropy_rate as_saved lambda_zero ∈ {180, 212, 245}` (NFE512) → bracket 23.1%.
3. Gate 3: `em entropy_rate as_saved lambda_zero ∈ {anchor×1.3, anchor×1.7}` (NFE512).
4. If promising: port PC, repeat the best λ₀ region with corrector steps.

Use the existing frozen snapshot `runs/tasks/tinygsm/eval_snapshots/step250000/` (create it
the same way as step217858: copy `step=000250000.pt` + entropy tables + `sigma_data.json`).

## 7. Risks & mitigations

- **Refactor breaks DDIM** → Gate 0 golden test; keep the extraction purely mechanical.
- **Entropy-table format mismatch** → verify loader asserts on our `.pt` files first.
- **λ₀ scale confusion** (peak vs as_saved) → anchor empirically (Gate 2), don't trust the
  asymptotic constant; report both the λ₀ and the implied γ-equivalent.
- **PC NFE accounting** — corrector steps add denoiser passes; report *effective* NFE so the
  comparison to DDIM/EM at "NFE=512" stays honest.
- **No σ_max OOD** for EM (noise added in x-space) — a genuine advantage over EDM churn; note
  it in the writeup.
