# Score temperature for the protein (sequence + structure) checkpoints

Branch `feat/score-temperature` (off `feat/mnist-sum`). Adds a global **score
temperature** to `DDIMSampler` and wires it into the protein generation path used by
forward folding, inverse folding and co-generation.

## What it does

At every DDIM step, after the classifier-free-guidance mix, the drift uses a sharpened
mean prediction

    D_tau = x + (D_cfg - x) / tau          (score divided by tau)

- `tau = 1` is **bit-identical** to the existing sampler (tested).
- `tau` is constant at every noise level. This is the "global" score temperature of the
  Sigma paper (Yang et al. 2026, arXiv:2610.02665) and of the rolling-cobit / TinyGSM
  text experiments (`score_temp_mode="global"` in rg625/BitstreamDiffusion). It is
  **not** the older noise-dependent `kappa(sigma)` ("local", Track A1) variant.
- Self-conditioning and the final readout use the **untempered** prediction.
- `score_temp_final="unsharpened"` (default) leaves the last integration step
  untempered, because a tempered final step overshoots by about 1/tau and flips bits
  that are already decided. `"sharpened"` tempers every step (ablation only).
- Same number of model calls as tau = 1: no extra compute.
- Only the `ddim_entropic` solver supports it (the paper protein config uses it).
  `heun` with tau != 1 raises an error.

In the multimodal CoBit STT experiments (same sampler code), tau = 0.3 at the paper
settings (CFG 5, 512 steps, churn) lowered development WER from 10.47% to 9.31%,
and tau = 0.1 collapsed. So check that the outputs still make sense at low tau.

## Code changes

| File | Change |
|---|---|
| `diffusion/continuous/samplers.py` | `DDIMSampler.sample(score_temp=1.0, score_temp_final="unsharpened")` |
| `evaluation/evaluation_drivers/proteins/generate_multimodal.py` | `generate()` reads `cfg.evaluation.score_temp` / `score_temp_final` (or explicit kwargs), validates them, passes them to DDIM, and records them in `out["sampling"]` |
| `configs/proteins/proteins_400m.py` | defaults `score_temp = 1.0`, `score_temp_final = "unsharpened"` (unchanged behaviour) |
| `configs/proteins/proteins_400m_score_temp.py` | sweep config: paper config plus `SCORE_TEMP` from the environment; writes to `<run>/evaluation/score_temp_<tau>/` |
| `tests/test_protein_score_temperature.py` | tau=1 identity for all three directions, tau<1 changes the trajectory at equal cost, wiring reaches DDIM, Heun refused, invalid tau refused, sweep-config checks |

Every protein driver calls `generate_multimodal.generate`, so no driver changes are
needed: the temperature comes from the config file that `scripts/proteins/evaluate.py`
passes to each task.

## How to run

The entropy profile for the checkpoint must already exist (`evaluate.py entropy`,
unchanged). Then, for each tau and each direction:

```bash
for TAU in 1 0.7 0.5 0.3; do
  for TASK in forward inverse cogeneration; do
    SCORE_TEMP=$TAU python scripts/proteins/evaluate.py $TASK \
      --config configs/proteins/proteins_400m_score_temp.py \
      --checkpoint <path/to/best_ema.pt>
  done
done
```

- Results: `runs/<experiment>/evaluation/score_temp_<tau>/<task>.json`. Earlier paper
  results in `runs/<experiment>/evaluation/` are never overwritten.
- Re-run **tau = 1** through this config, rather than reusing the old number, so the
  comparison uses the same code, seed and hardware.
- Check `out["sampling"]["score_temp"]` in each result file to confirm the run used the
  intended tau.
- Ablation only: `SCORE_TEMP_FINAL=sharpened` tempers the final step too, and goes to
  `score_temp_<tau>_sharpened/`.

## Things to watch

1. **Choose tau on validation, not test.** If you pick the best tau on the test targets,
   the reported gain is optimistic. Preferred: sweep on a validation split
   (`cfg.evaluation.validation_data`, CATH 4.3 validation), fix one tau per direction,
   then run the test sets once at tau = 1 and the chosen tau.
2. **Diversity, not only quality.** Lower temperature concentrates samples. For
   co-generation and inverse folding, report diversity and novelty next to
   designability, recovery or scTM. For forward folding, check that the structures
   are not collapsing to a few modes.
3. **Few steps.** The protein config uses 50 DDIM steps. Tempering multiplies each
   step's size by 1/tau, so at 50 steps a low tau gives large steps. The sharpening
   actually achieved depends on the step count. If tau = 0.3 looks broken, also try
   100 steps at the same tau before concluding temperature does not help.
4. **Paired seeds.** `evaluate.py` passes `--seed cfg.evaluation.seed`, and each driver
   derives a per-target seed from it (`unit_seed`), so every tau sees the same
   starting noise per target. Keep the seed fixed across the sweep.
5. **Environment.** Some existing tests fail on Python 3.9 (`zip(strict=...)`,
   `datetime.UTC`). Use the project's normal Python 3.10+ environment.
   `pytest tests/test_protein_score_temperature.py` should pass.
