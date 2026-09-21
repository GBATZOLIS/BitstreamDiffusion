# Handoff — port MNIST-Sum P=28 onto George's codebase and re-run it

*Written 2026-09-21 on csic47. Supersedes the first draft of this file, which recommended
merging the other way; Georgios has decided George's fork is the base and the sampler refactor
comes later.*

**Decision already taken, do not relitigate:** we ship on
`GeorgeKontos14/BitstreamDiffusion` (`main`, tip `4a18fa9`). Its `diffusion/continuous/samplers.py`
is a 2,396-line pre-refactor monolith rather than our 12-file package; that is accepted and will
be refactored later. CC3M/CC12M will be re-run on this codebase afterwards, and Bruno's protein
work merged after that. **MNIST-Sum P=28 goes first because it is the cheapest and it is the one
going in the paper.**

Everything in §1–§2 is the result of an actual diff of the two trees; you do not need to redo it.

---

## 1. What you are porting into, and the two real gaps

`models/sdt.py` differs between the forks by **exactly two features**, 46 lines in 4 hunks —
the `local_pos_denom` option and the whole segment-embedding block. Same classes, same functions,
everything else identical. `utils/schedule_controller.py` is byte-identical. The EDM churn formula
is byte-identical (`min(s_churn / num_intervals, sqrt(2)−1)`), so a gamma of 0.175 means the same
thing on both sides.

George's sampler already has what the MNIST-Sum harness calls:
`create_sampler` with the same `ddim_entropic → DDIMSampler + schedule="entropic"` branch, the
same `_StochasticSamplerCfg` fields (`enabled`, `s_churn`, `s_noise`, `window_mode`) that
`ms_churn` mutates, and `sample()` accepting `conditioning_prefix_full`, `cond_prefix_mask`,
`guidance_scale`, `schedule`, `num_steps`, `entropy_run_dir`, `sigma_min_override`,
`sc_refresh_mode`, `return_probs`. He also already saves per-step entropy snapshots, which we
do not — keep that.

### Gap 1 — `protect_mask` does not exist in his fork (this one is dangerous)

`ms_sample` passes `protect_mask` = the 6 marker bit positions. Its job is in the CFG
*unconditional* branch: markers must be restored to their true codes rather than nulled to 0.5,
matching training, where §3.3 says the conditioning segments "but never the markers" are dropped.
George's `_make_null_full` does `out[pm] = null_val` over the whole conditioned prefix with no
carve-out, and the string `protect` appears nowhere in his sampler or driver.

**Why this is the dangerous one:** it fails silently. Nothing errors; the unconditional branch
just sees markers at 0.5, which the model never saw in training, and guided i2t/t2i quietly
degrade. Our reported numbers are at guidance 2.0, so this is on the critical path. Joint
co-generation is unguided and unaffected.

Fix is small and well-scoped: thread `protect_mask` through `sample()` → the null-branch
construction, restoring `null_full[protect] = prefix_full[protect]`. Ours is ~15 lines in
`diffusion/continuous/samplers/common.py` around line 505 — copy the semantics, not the file.

### Gap 2 — no segment embedding, and we are not adding one

George's `sdt.py` has no segment-embedding implementation. His
`configs/textaudio/mls_632.py:86` nonetheless sets `cfg.model.use_segment_embed = True`, and
**nothing in his repo reads it** — so CoBit-LibriTTS and CoBit-MLS trained without it.

**Settled: MNIST-Sum runs on George's architecture exactly as it is. No segment embedding.**
Do not port ours, do not probe ours, do not reference ours. The unified architecture is whatever
his code does, and CC3M/CC12M will be brought onto it later. This means two things when you port
the configs, both in §3.

While you are there, make an unimplemented `cfg.model.*` flag **raise** rather than be ignored.
A config silently asking for a component that does not exist is what produced this whole
situation.

### Not a gap: the positional-embedding "leak"

For the record, since this is what started the investigation. The intra-patch Fourier feature is
`bit_index % P` — a function of position within a patch, identical for a text bit and an image
bit, in both forks. It carries no modality information; `P` vs `P−1` only changes whether bit 0
and bit P−1 alias onto the same angle. Absolute position *does* determine modality, but only
because the sequence template is fixed, which §3.2 discloses and which holds for any architecture
over a fixed layout. The one genuinely modality-aware component is the segment embedding, which
the paper already declares. **There is no hidden leak.** The exposure is §3.2's "the backbone
architecture [is] identical" — the sentence already carrying a verify-before-submission TODO.

## 2. Port surface

George has **none** of the MNIST-Sum files. In dependency order:

| # | what | source | notes |
|---|---|---|---|
| 1 | `configs/multimodal/cc3m_lfq19_medium_24x1024_joint.py` | ours | **needed**: `_mnist_sum_base.py` does `_load_cc3m()`, which is what makes "everything else is CC3M's recipe by construction" true in the paper. George has no `configs/multimodal/` at all. Needed for the CC3M re-run anyway. |
| 2 | `data/mnist_sum_codec.py`, `data/mnist_sum_bits.py` | ours | plus registration in the dataloader factory |
| 3 | regime / conditioning-mask sampling for the MNIST layout | adapt | see below |
| 4 | `protect_mask` threading | write | §1 Gap 1 |
| 5 | `evaluation/mnist_sum.py`, `evaluation/mnist_sum_classifier.py` | ours | `ms_sample`, `ms_churn`, exact scorers, quadrant CNN |
| 6 | `scripts/multimodal/eval_mnist_sum_offline.py` | ours | the frozen harness |
| 7 | `utils/callbacks/mnist_sum_eval.py` | ours | **keep `run_on_all_ranks = True`** (see gotcha 4) |
| 8 | `configs/multimodal/_mnist_sum_base.py`, `mnist_sum_p28_small.py` | ours | with §3's architecture decision applied |
| 9 | `scripts/multimodal/phase_50_mnist_sum_dataset.py` | ours | corpus builder; **not needed on csic47**, the corpus already exists |

**On item 3.** George's partial denoising is wired through
`from utils.textaudio_utils import _sample_tasks_and_cond_masks, UNCONDITIONAL` — a module-level,
textaudio-specific import in `trainers/trainer.py:41`. So his regime machinery is not generic.
The minimal move is a parallel `utils/mnist_sum_utils.py` with the same interface, dispatched on
the layout. The right move, given this codebase is about to become the single published one, is a
layout-driven regime sampler that textaudio, mnist_sum and later CC3M all share — a trainer that
imports a textaudio-specific task sampler is the same optics problem in a paper claiming
modality-agnosticism as our `if dataset == "mnistsumbits"` was. **Do the minimal version to get
the run going; open an issue for the generic one.** Do not let a refactor block the run.

## 3. Strip our architecture overrides out of the ported configs

The configs you carry across were written against my fork and will silently ask for things
George's code does not have. Two files, three lines:

**`configs/multimodal/cc3m_lfq19_medium_24x1024_joint.py`** (ported only as the recipe base for
`_load_cc3m()`):

```python
cfg.model.use_segment_embed = True      # DELETE — no implementation in this codebase
```

**`configs/multimodal/_mnist_sum_base.py`**:

```python
cfg.model.n_fourier_local = max(1, patch_size // 2)   # DELETE — inherit 4 from cc3m
cfg.model.local_pos_denom = "P"                       # DELETE — no such option here; P-1 is hardcoded
```

That leaves MNIST-Sum on stock George: `n_fourier_local` 4, intra-patch period P−1, no segment
embedding — identical to the audio models. Nothing else in `_mnist_sum_base.py` changes; the
data geometry, trunk scale, run length and callback block all stay.

## 4. There is no number to reproduce — this run *is* the number

Do not treat the published table as a target to hit. It was produced on an architecture we are
abandoning: segment embedding on, `local_pos_denom` "P", `n_fourier_local` 14. **The unified-codebase
result replaces it in the paper**, whatever it comes out as.

Use the old numbers only as a smell test — if the new run lands wildly off (say end-to-end below
90%, or malformed above a percent), something in the port is broken and you should find it rather
than report it. Within a point or two, just report the new number.

## 5. Build, smoke, train

```bash
cd ~ && git clone https://github.com/GeorgeKontos14/BitstreamDiffusion.git cobit-unified
cd cobit-unified && git checkout -b feat/mnist-sum
# a clone is already at ~/forkcmp/george if you want to diff offline
```

Environment: use `pytorch` (`~/miniconda3/envs/pytorch`, torch 2.5.1) — the env every run in this
project has used. Check `requirements.txt` / `textaudio_install.sh` for anything his audio stack
needs that ours did not; skip the Spark-TTS pieces, they are audio-only.

The corpus is already built and does **not** need regenerating:
`datasets/mnist_sum_p28` (1M/10k/10k, 510 MB; baselines val 6.52%, held-out 7.80%). Symlink it
rather than copying. Same for `runs/mnist_sum_quadrant_cnn.pt` — **reuse that classifier, do not
retrain it**, or nothing compares to the published numbers.

Smoke before training. It has already caught two real bugs in this experiment:

```bash
SMOKE_MAX_STEPS=200 MS_EVAL_EVERY=10 MS_DATA_ROOT=datasets/mnist_sum_p28_smoke \
torchrun --nproc_per_node=2 train.py --config configs/multimodal/mnist_sum_p28_small.py
```

Require `results.md` / TensorBoard to contain **all three directions** — not that the process
exited 0. Then train:

* 500,000 steps, DDP on 2×A6000, global batch 512, ~283 ms/step → **~40 h**.
* First eval fires at the first epoch boundary, **step 1,953** (~9 min in) — check it there rather
  than waiting for 50k.
* Accuracy saturates by ~150k in the existing run (e2e 0.973 at 150k vs 0.977 at 500k), so you
  will know by ~hour 12 whether the numbers are landing where they should.

## 6. Evaluate, identically

```bash
python scripts/multimodal/eval_mnist_sum_offline.py \
    --config configs/multimodal/mnist_sum_p28_small.py \
    --ckpt   runs/multimodal/mnist_sum_p28_12x512/checkpoints/step=000500000.pt \
    --out    results/mnist_sum_p28_500k_unified --n 4096 --n_joint 4096
```

Same n, same steps (256/256), same guidance ({2.0, 1.0}), same gammas (i2t 0.175, t2i 0.0), same
cached classifier, EMA, compiled. Confirm `karras_fallback: false`.

Then compare to the published table and regenerate the deliverables:
`figures/mnist_sum/gen_assets.py` → `fig1_overview.py`, `fig2_joint.py`, `fig3_results.py`, and
update `docs/MNIST_SUM_RESULTS.md` (§5, §6, §8, and the draft paper text in §9).

Smell test only, from the superseded run (see §4 — do not treat these as a target):

| | val | held-out |
|---|---|---|
| i2t end-to-end | 97.90 [97.41, 98.30] | 97.92 [97.44, 98.32] |
| t2i all four | 99.98 [99.86, 100.00] | 99.85 [99.68, 99.93] |
| joint consistent | 96.85 [96.27, 97.34] | — |

## 7. Gotchas that will each cost you a day

1. **`protect_mask` fails silently.** §1 Gap 1. If you skip it, guided i2t/t2i degrade with no
   error and you will blame the architecture change. Port it first and verify by diffing a
   guided-vs-unguided run on a few hundred samples.
2. **`best.pt` is the wrong checkpoint.** Selected on validation loss, which the entropy warmup at
   40k makes incomparable across that boundary; its minimum is an artifact near step 39k, ~10
   accuracy points below the final model. Use `step=000500000.pt`.
3. **Entropy tables.** Absent, the sampler silently drops to a Karras grid behind one WARN line.
   Our harness resolves them from the *checkpoint* directory and refuses to run without them —
   preserve that behaviour when porting, and confirm `karras_fallback: false` in every
   `results.json`. George's per-step entropy snapshots make this easier, not harder.
4. **`MNISTSumEvalCallback` must keep `run_on_all_ranks = True`.** Sampling rank-0-only through
   the compiled DDP module deadlocks the *next* training step: rank 0 recompiles for shapes the
   other ranks never see (measured 28 vs 21 recompiles), DDPOptimizer buckets allreduces
   differently, and the ranks wedge with no error until the 60-minute timeout. George's trainer
   registers callbacks differently from ours — re-check this flag survives the port.
5. **`torch.compile` stays on.** Under eager+bf16 this long-sequence model loses image detail
   (eager+fp32 and fp16 are fine — it is bf16 mantissa precision).
6. **One classifier across all arms.** Two arms read by two instruments is not a comparison.
7. **The ceiling has two values.** 99.78% all-four on train-pool composites, 96.09% on test-pool.
   Generated digits imitate the train pool the classifier was fit on, so text→image must be read
   against **99.78**; against 96.09 it reads as a nonsensical "104% of ceiling".
   `docs/MNIST_SUM_RESULTS.md` §7.1.
8. **Check `sc_refresh_mode` at training time.** George's `sample()` defaults to `"refined"`;
   ours and the paper use `"carry"` throughout. `ms_sample` passes it explicitly so evaluation is
   safe, but verify the training path matches.

## 8. Report back

1. The three-direction table from the unified codebase, with intervals. Note beside it how it
   compares to the superseded run, but report the new numbers as the paper numbers.
2. Updated `docs/MNIST_SUM_RESULTS.md` and regenerated figures.
3. A replacement for the §3.2 sentence marked "[Verify against the final audio and protein
   training configurations before submission…]" — after this run it can be verified for image+text
   and audio rather than asserted. Protein stays open until Bruno's merge.
4. Anything you had to change in George's sampler beyond `protect_mask`, so the later refactor
   knows what to preserve.

## 9. Explicitly out of scope for this session

The sampler refactor, the CC3M/CC12M re-run, Bruno's protein merge, and the generic
layout-driven regime sampler. Get P=28 re-run and evaluated first.
