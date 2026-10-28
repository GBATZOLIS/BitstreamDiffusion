# MNIST-Sum — evaluation report (both arms)

> **STATUS — numbers in this document are the SUPERSEDED run, pending replacement.**
>
> Every result below was produced on the other fork's architecture (segment
> embedding on, `local_pos_denom = "P"`, `n_fourier_local = 14`). That
> architecture is being retired. MNIST-Sum P=28 has been re-run on this
> codebase's stock backbone -- no segment embedding, `n_fourier_local` 4,
> intra-patch period P-1, identical to the audio models -- and the unified-codebase
> result REPLACES the P=28 column here, whatever it comes out as. See
> `docs/HANDOFF_UNIFY_ARCH_MNIST_SUM.md`.
>
> The methodology sections (corpus and layout, evaluation protocol, the
> classifier-ceiling finding in §7.1, the checkpoint-selection warning) are
> unaffected and remain correct. The P=14 arm has not been re-run and stays on
> the retired architecture; it is not directly comparable to the new P=28 column.

*Internal report backing the proposed §4.3 subsection. All numbers below are read from
`results/mnist_sum_p{28,14}_500k/results.json`, produced by one harness at n=4,096 per direction per split.*

**Headline.** A tokenizer-free image+text corpus in which every generative direction has a unique correct
answer. Joint co-generation is **96.85%** exactly self-consistent — the paper's only co-generation number
with a ground truth rather than a similarity score.

| | P=28 | P=14 |
|---|---|---|
| image → text, end-to-end (val) | **97.90** | 97.51 |
| text → image, all four (val) | **99.98** | **99.98** |
| joint, consistent | 96.85 | **97.92** |
| generalisation gap (val − held-out) | −0.02 *n.s.* | +0.39 *n.s.* |
| matched cost (ms/step, B=256, A6000) | **187.9** | 326.2 |

---

## 1. Why this belongs in the paper

Every image+text number on CC3M/CC12M is a proxy: FID and CMMD compare distributions, CLIP compares
embeddings, CIDEr compares n-grams, and the joint row is defended with pair coherence 25.67 — a cosine
similarity with **no notion of correct**. None of them can say whether a co-generated pair actually agrees.

MNIST-Sum is built so that they can. An image is four binarised MNIST digits in a 2×2 grid; the text is the
raster-order equation and its sum. Two properties make it a probe of the framework's central claim rather
than a toy:

- **No image tokenizer.** Binary pixels ride as raw bits — no reconstruction floor, so generated and real
  images are exactly comparable, unlike every LFQ number in §4.3 which is bounded by the codec's own rFID.
- **No text tokenizer either.** A 7-bit hand-specified vocabulary (37 number words, `+`, `=`, PAD = 40
  codes). The section can therefore state that the bitstream formulation needs *neither* modality-specific
  component — the strongest available form of "the tokenizers are the only modality-specific part".

## 2. Corpus and layout — Figure 1

![overview](../figures/mnist_sum/fig_mnist_sum_overview.png)

| arm | patch P | text positions | image positions | markers | positions | bits | params |
|---|---|---|---|---|---|---|---|
| P=28 | 28 | 3 (4 codes each) | 112 (1 digit row each) | 6 | 121 | 3,388 | 63.3M |
| P=14 | 14 | 6 (2 codes each) | 224 (½ digit row each) | 6 | 236 | 3,304 | 60.3M |

Both arms use the same 12×512 trunk and the CC3M-derived recipe, differing **only** in how the flat
bitstream is patched. 7 divides both 14 and 28, so neither splits a word code, and a digit row is a whole
number of positions in both; the difference is whether one trunk token *contains* a complete row (P=28) or
must reassemble one across two tokens (P=14).

Corpus: 1,000,000 train / 10,000 val / 10,000 val_holdout, the last using digit tuples never seen in
training (10% of the 10⁴ combinations). Sampling is natural (uniform digits); the modal-sum baseline an
image-blind model scores by always answering the most common sum is **6.52%** (val) and **7.80%** (held-out).

## 3. Training and cost

| | steps | hardware | wall clock | ms/step as trained | ms/step matched | peak mem matched |
|---|---|---|---|---|---|---|
| P=28 | 500,000 | 2×A6000 | 40 h 37 m | 283 | **187.9** | 9.50 GiB |
| P=14 | 500,000 | 2×A100 | ≈55 h | 395 | **326.2** | 15.95 GiB |

"As trained" figures come from the two training runs and are **not** comparable (different hardware). The
matched columns were measured here on one A6000 at batch 256/GPU (global 512), compiled, bf16, identical
conditions: **P=14 costs 1.74× the compute and 1.68× the memory of P=28.**

## 4. Evaluation protocol

One harness, `scripts/multimodal/eval_mnist_sum_offline.py`, identical settings on both arms: final
`step=000500000.pt` (not `best.pt`, whose selection on validation loss is corrupted by the entropy-schedule
warmup at 40k), EMA weights, compiled, entropic schedule, **n = 4,096** per direction per split.

| direction | steps | guidance | churn γ | scored by |
|---|---|---|---|---|
| image → text | 256 | 2.0 and 1.0 | 0.175 | string equality — exact, no instrument |
| text → image | 256 | 2.0 and 1.0 | 0.0 | quadrant CNN, beside its ceiling |
| joint | 256 | unguided (single branch) | 0.0 | quadrant CNN + exact parse |

**Schedule audit passed.** Both arms record `karras_fallback: false` and the identical classifier path. The
harness resolves entropy tables from the *checkpoint* directory and refuses to run without them, so the
silent Karras fallback documented in Appendix A.7 cannot have occurred.

## 5. Results — Figure 3

![results](../figures/mnist_sum/fig_mnist_sum_results.png)

### 5.1 Image → text (exact, no instrument)

**P=28**

| split · guidance | perception | arithmetic | end-to-end | e2e \| percep | malformed | baseline |
|---|---|---|---|---|---|---|
| val · gs 2.0 | 98.14 <sub>[97.68, 98.51]</sub> | 99.66 <sub>[99.43, 99.80]</sub> | **97.90 <sub>[97.41, 98.30]</sub>** | 99.75 | 0.05 | 6.52 |
| held-out · gs 2.0 | 98.05 <sub>[97.58, 98.43]</sub> | 99.83 <sub>[99.65, 99.92]</sub> | **97.92 <sub>[97.44, 98.32]</sub>** | 99.88 | 0.00 | 7.80 |
| val · gs 1.0 | 98.12 <sub>[97.66, 98.49]</sub> | 99.58 <sub>[99.34, 99.74]</sub> | **97.83 <sub>[97.33, 98.23]</sub>** | 99.68 | 0.05 | 6.52 |
| held-out · gs 1.0 | 98.07 <sub>[97.60, 98.45]</sub> | 99.68 <sub>[99.46, 99.81]</sub> | **97.78 <sub>[97.28, 98.19]</sub>** | 99.70 | 0.00 | 7.80 |

**P=14**

| split · guidance | perception | arithmetic | end-to-end | e2e \| percep | malformed | baseline |
|---|---|---|---|---|---|---|
| val · gs 2.0 | 97.97 <sub>[97.50, 98.36]</sub> | 99.49 <sub>[99.22, 99.66]</sub> | **97.51 <sub>[96.99, 97.94]</sub>** | 99.53 | 0.00 | 6.52 |
| held-out · gs 2.0 | 97.90 <sub>[97.41, 98.30]</sub> | 99.15 <sub>[98.81, 99.38]</sub> | **97.12 <sub>[96.56, 97.59]</sub>** | 99.20 | 0.00 | 7.80 |
| val · gs 1.0 | 97.92 <sub>[97.44, 98.32]</sub> | 99.19 <sub>[98.87, 99.43]</sub> | **97.19 <sub>[96.64, 97.66]</sub>** | 99.25 | 0.00 | 6.52 |
| held-out · gs 1.0 | 97.88 <sub>[97.39, 98.27]</sub> | 98.66 <sub>[98.26, 98.97]</sub> | **96.66 <sub>[96.06, 97.16]</sub>** | 98.75 | 0.00 | 7.80 |

End-to-end is **15.0× the modal-sum baseline**. The decomposition holds as a product
(0.9814 × 0.9966 = 0.9781 vs 0.9790 measured), and `e2e | perception` = 99.75% says that once the digits are
read correctly they are essentially always added correctly. Malformed output ≤0.05% everywhere. Per-quadrant
perception is uniform (P=28 val, TL/TR/BL/BR: 99.61 / 99.32 / 99.54 / 99.49) — no positional bias.

### 5.2 Text → image (via the quadrant CNN)

**P=28**

| split · guidance | all four correct | per quadrant | ceiling (test pool) | ceiling (same pool) |
|---|---|---|---|---|
| val · gs 2.0 | **99.98 <sub>[99.86, 100.00]</sub>** | 99.99 | 96.09 | 99.78 |
| held-out · gs 2.0 | **99.85 <sub>[99.68, 99.93]</sub>** | 99.96 | 96.80 | 99.78 |
| val · gs 1.0 | **99.61 <sub>[99.37, 99.76]</sub>** | 99.90 | 96.09 | 99.78 |
| held-out · gs 1.0 | **99.54 <sub>[99.28, 99.70]</sub>** | 99.88 | 96.80 | 99.78 |

**P=14**

| split · guidance | all four correct | per quadrant | ceiling (test pool) | ceiling (same pool) |
|---|---|---|---|---|
| val · gs 2.0 | **99.98 <sub>[99.86, 100.00]</sub>** | 99.99 | 96.09 | 99.78 |
| held-out · gs 2.0 | **99.98 <sub>[99.86, 100.00]</sub>** | 99.99 | 96.80 | 99.78 |
| val · gs 1.0 | **99.49 <sub>[99.22, 99.66]</sub>** | 99.87 | 96.09 | 99.78 |
| held-out · gs 1.0 | **99.34 <sub>[99.04, 99.55]</sub>** | 99.84 | 96.80 | 99.78 |

> **Read text→image against the same-pool ceiling (99.78), not the test-pool one (96.09).** See §7.1. Against
> 99.78 the honest statement is that text→image is **at ceiling with no headroom left to measure**, and
> therefore cannot discriminate the arms.

## 6. Joint co-generation — the headline — Figure 2

![joint](../figures/mnist_sum/fig_mnist_sum_joint.png)

| metric | P=28 | P=14 |
|---|---|---|
| text well-formed | 99.98 <sub>[99.86, 100.00]</sub> | 99.98 <sub>[99.86, 100.00]</sub> |
| addends match the image | 99.78 <sub>[99.58, 99.88]</sub> | 99.80 <sub>[99.62, 99.90]</sub> |
| arithmetic valid | 97.02 <sub>[96.46, 97.50]</sub> | 98.10 <sub>[97.63, 98.47]</sub> |
| **consistent** — equation exactly right for its image | 96.85 <sub>[96.27, 97.34]</sub> | 97.92 <sub>[97.44, 98.32]</sub> |

Generating both halves from noise, **96.85%** of samples (P=28) yield an equation exactly correct for the
image beside it. It decomposes cleanly: the model essentially never writes a malformed equation (99.98%) nor
mislabels its own image (99.78%); **all remaining loss is arithmetic** (97.02%). The 32 sampled figure cases
reproduce the aggregate — 30 consistent, and both failures read their own digits correctly while miscounting
the sum (9+7+8+3 written as twenty-eight; 0+0+1+0 as zero). This row is unguided (C = ∅ is a single branch,
NFE = steps), so it is also the cheapest direction per sample.

## 7. Measurement findings

### 7.1 The ceiling depends on which MNIST pool the composites come from

| composites built from | per quadrant | all four |
|---|---|---|
| MNIST **train** digits (corpus train split) | 99.95 | **99.78** |
| MNIST **test** digits (val) | 99.02 | 96.09 |
| MNIST **test** digits (held-out) | 99.20 | 96.80 |
| generated (t2i, gs 1.0, deterministic) | 99.90 | 99.61 |

The builder draws train composites from MNIST's train digits and val/held-out from its test digits. The
classifier is trained on the train pool — deliberately, so the ceiling is measured on unseen glyphs. The
unintended consequence: **generated digits imitate the pool the classifier knows, while the ceiling is
measured on the pool it does not.** Generated composites read at 99.90 per quadrant, almost exactly the 99.95
scored on real train-pool composites. The model is not producing superhumanly clean digits; it produces
train-pool-like digits, and ~0.9 points per quadrant compounds over four quadrants into the ~3.7-point excess.

### 7.2 Two sampler explanations ruled out

- **Guidance** contributes +0.37 (val) / +0.32 (held-out) points — real, expected direction — but with CFG
  off entirely t2i is still +3.52 above the test-pool ceiling.
- **Determinism** (γ=0, mode-seeking) tested by turning churn on at γ=0.175: all-four moved 99.61 → 99.71
  (val) and 99.54 → 99.61 (held-out), i.e. marginally *up*.

Worth one appendix sentence, since a reviewer would otherwise ask whether the above-ceiling number is a
guidance artefact.

## 8. Cross-arm comparison (internal)

95% Newcombe intervals on the difference, n=4,096 per cell.

| image → text, end-to-end | P=28 | P=14 | difference | 95% CI | |
|---|---|---|---|---|---|
| val · gs 2.0 | 97.90 | 97.51 | **+0.39** | [−0.26, +1.05] | n.s. |
| held-out · gs 2.0 | 97.92 | 97.12 | **+0.81** | [+0.13, +1.49] | **significant** |
| val · gs 1.0 | 97.83 | 97.19 | **+0.63** | [−0.04, +1.32] | n.s. |
| held-out · gs 1.0 | 97.78 | 96.66 | **+1.12** | [+0.41, +1.85] | **significant** |

| joint co-generation | P=28 | P=14 | difference | 95% CI | |
|---|---|---|---|---|---|
| text well-formed | 99.98 | 99.98 | +0.00 | [−0.12, +0.12] | n.s. |
| addends match | 99.78 | 99.80 | −0.02 | [−0.24, +0.19] | n.s. |
| arithmetic valid | 97.02 | 98.10 | **−1.07** | [−1.75, −0.41] | **significant** |
| consistent | 96.85 | 97.92 | **−1.07** | [−1.78, −0.38] | **significant** |

**The arms win in opposite directions and both effects clear the intervals.** P=28 leads image→text by ~1
point, and only on *held-out* tuples; P=14 leads joint co-generation by ~1 point, concentrated entirely in
`arithmetic_valid`. Text→image discriminates nothing, being saturated in both.

A coherent reading this two-point comparison cannot prove: whole-row image patches (P=28) help perception,
while twice as many text positions (P=14: 6 vs 3) help the symbolic side. The decisive test is a **P=56**
arm — 3 text positions like P=28 but 56 image positions — separating "fewer positions is better" from "a
position should be a complete unit" from "text positions drive arithmetic".

### Generalisation gap — not measurable in either arm

| arm | val | held-out | gap | 95% CI | |
|---|---|---|---|---|---|
| P=28 | 97.90 | 97.92 | **−0.02** | [−0.65, +0.60] | n.s. |
| P=14 | 97.51 | 97.12 | **+0.39** | [−0.31, +1.10] | n.s. |

Neither clears its interval at n=4,096. The honest statement is that **digit tuples never seen in training
are read as well as seen ones** — no memorisation signal — rather than a small measured gap.

## 9. Draft text for the paper

> **A tokenizer-free probe with exact answers.** Before the alt-text experiments, we ask what the bitstream
> formulation does on a task where every generative direction has a unique correct answer. An image is four
> binarised MNIST digits in a 2×2 grid (56×56, native 28×28 digits, nothing resized); the text is the
> raster-order equation and its sum. Neither modality is tokenized: binary pixels ride as raw bits, 28 per
> position, and the text uses a 7-bit hand-specified vocabulary of 40 codes, four to a position. There is no
> reconstruction floor on the image side and no codec on either, so generated and real images are exactly
> comparable and the framework is exercised with *no* modality-specific component at all.
>
> We train a 63.3M-parameter 12×512 model for 500k steps on 10⁶ examples, with the CC3M recipe unchanged, and
> evaluate all three directions at n=4,096 with 256 steps. Image→text is scored by string equality:
> end-to-end accuracy is 97.90% against a 6.52% modal-sum baseline, decomposing into 98.14% perception and
> 99.66% arithmetic, with malformed output at 0.05%. Text→image, read by a quadrant classifier whose ceiling
> on real composites is 99.78%, places all four digits correctly in 99.98% of samples. On digit tuples never
> seen in training the model scores 97.92%, indistinguishable from the 97.90% it scores in-distribution
> (difference −0.02, 95% CI [−0.65, +0.60]), so the mapping is learned rather than memorised.
>
> The direction that matters most is joint co-generation, where nothing is given and both halves come from
> one trajectory. Here the framework's consistency can be stated as a rate rather than a similarity: **96.85%**
> of co-generated pairs carry an equation exactly correct for the image produced beside it. It decomposes into
> 99.98% well-formed equations, 99.78% in which the named addends match the generated image, and 97.02% in
> which the arithmetic is valid — so the modalities agree with each other almost perfectly, and what residual
> error exists is symbolic rather than cross-modal. This is the same claim the CC3M joint row defends with
> pair coherence 25.67, measured here against correctness.

### Figure captions

**Figure 1.** MNIST-Sum. (a) One example: four native 28×28 binarised digits on a 56×56 canvas, never
resized. (b) Image and text written into a single 3,388-bit stream of 121 positions × 28 bits, with no
tokenizer on either side: one text position holds four 7-bit word codes, one image position holds 28 raw
pixels — exactly one row of one digit. Text is 2.5% of the stream against the image's 92.6%. (c–e) The three
directions, with real samples from the model.

**Figure 2.** Joint co-generation: image and equation produced together from noise, with nothing given.
First 12 samples in generation order, not selected. A teal border means the equation is exactly correct for
the image beside it; the one failure names all four digits correctly and miscounts the sum — the dominant
error mode.

**Figure 3.** All three directions, both arms, n=4,096, 95% Wilson intervals. (a) image→text decomposes into
perception, arithmetic and end-to-end; the modal-sum baseline at 6.5% is far below the axis. (b) text→image
is saturated at the same-pool ceiling. (c) joint co-generation, where essentially all loss is arithmetic.

## 10. Reproduction

```bash
python scripts/multimodal/phase_50_mnist_sum_dataset.py \
    --out datasets/mnist_sum_p28 --patch_size 28 \
    --n_train 1000000 --n_val 10000 --holdout_combos 0.1 --mnist_raw datasets/MNIST/raw

python train.py --config configs/multimodal/mnist_sum_p28_small.py

python scripts/multimodal/eval_mnist_sum_offline.py \
    --config configs/multimodal/mnist_sum_p28_small.py \
    --ckpt   runs/multimodal/mnist_sum_p28_12x512/checkpoints/step=000500000.pt \
    --out    results/mnist_sum_p28_500k --n 4096 --n_joint 4096
```

Figures and their generating scripts are in `figures/mnist_sum/` (`gen_assets.py` → `fig1_overview.py`,
`fig2_joint.py`, `fig3_results.py`; `bench_one.py` for matched cost). Raw results are
`results/mnist_sum_p{28,14}_500k/results.json`. The shared classifier is cached at
`runs/mnist_sum_quadrant_cnn.pt` and **must be reused across arms** — two arms read by two instruments is not
a comparison.

## 11. What is not measured

- **Text→image is not exactly scored without an instrument**; its ceiling must appear beside every number.
- **The step-count sweep is not run.** The paper's 25× sampler asymmetry could be measured here against exact
  correctness rather than FID and CIDEr. Text→image saturates, so it would be informative mainly on the image
  side at low step counts.
- **Text-only arithmetic is deliberately absent**: there is no "absent image" in a bitstream, so conditioning
  on the addends still generates an image. `e2e | perception` answers the same question without the ambiguity.
- **P=56 has not been trained**, so the patching story in §8 remains an interpretation of two points.

---

*Both arms: 500,000 steps, EMA, compiled, entropic schedule, n=4,096 per direction per split,
`karras_fallback: false`, one shared quadrant classifier. P=28 evaluated on csic47; P=14 trained on CSD3 and
transferred (sha256 verified). Intervals are 95% Wilson; cross-arm differences are 95% Newcombe.*
