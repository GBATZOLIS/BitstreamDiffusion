# MNIST-Sum: replacing the §4.2 numbers with the unified-codebase run

Produced 2026-09-23 on csic47. Source of every number below:
`results/mnist_sum_p28_500k_unified/results.json`, written by
`scripts/multimodal/eval_mnist_sum_offline.py` from
`runs/multimodal/mnist_sum_p28_12x512/checkpoints/step=000500000.pt`.

Protocol identical to the superseded run: n=4,096 per direction per split, 256/256
sampling steps, guidance {2.0, 1.0}, gamma i2t 0.175 / t2i 0.0, EMA weights,
`torch.compile` on, the same cached quadrant classifier
(`runs/mnist_sum_quadrant_cnn.pt`), `karras_fallback: false`.

The model is the SAME experiment on the unified architecture: no segment embedding,
`n_fourier_local` 4, intra-patch period P-1 -- identical to the audio models.

---

## 1. Straight substitutions in §4.2

| claim | currently says | replace with |
|---|---|---|
| image->text, exact | 97.90% | **97.56%** |
| text->image, all four digits | 99.98% | **99.95%** |
| joint co-generation, consistent | 96.85% | **99.46%** |
| held-out tuples, image->text | 97.92% | **97.85%** |
| parameter count | 63.3M | **63.0M** |

`n = 4,096 per direction`, `P=28` and `500K steps` are unchanged.

**The parameter count is easy to miss.** 63.33M -> 63.04M. Dropping
`n_fourier_local` from 14 to 4 takes `pos_dim` from 45 to 25, shrinking the input
projection by 560 x 512 = 286,720 parameters. That is essentially the whole
difference; the segment embedding was zero-initialised and contributed almost none.

## 2. One CLAIM changes, not just its number

The superseded joint result decomposed as 99.98% well-formed, 99.78% addends-match-image,
97.02% arithmetic-valid. That supported a sentence to the effect that *the modalities
agree with each other almost perfectly and the residual error is symbolic rather than
cross-modal.*

**That is no longer true.** The unified decomposition is:

| joint component | superseded | unified |
|---|---|---|
| text well-formed | 99.98 | **100.00** |
| addends match image | 99.78 | **99.71** |
| arithmetic valid | 97.02 | **99.76** |
| **consistent** | **96.85** | **99.46** |

Residual error was 0.22% cross-modal vs 2.98% arithmetic -- a 1:13 split that justified
"symbolic rather than cross-modal". It is now 0.29% cross-modal vs 0.24% arithmetic,
i.e. **roughly even**. Any sentence attributing the residual mainly to arithmetic must go.

Suggested replacement clause:

> ... **99.46%** of co-generated pairs carry an equation exactly correct for the image
> produced beside it, decomposing into 100.00% well-formed equations, 99.71% in which the
> named addends match the generated image, and 99.76% in which the arithmetic is valid.
> The residual error is small enough that it no longer concentrates in either failure
> mode: cross-modal disagreement and arithmetic slips each account for under 0.3%.

I am working from the handoff summary of §4.2, not the file itself -- please check the
exact wording in `04c_image_text_toy.tex` against this before pasting.

## 3. Appendix numbers for `app:image:mnist`

All n=4,096; i2t/t2i at guidance 2.0; joint unguided. 95% Wilson intervals.

**image -> text**

| metric | val | held-out |
|---|---|---|
| end-to-end | 97.56 [97.04, 97.99] | 97.85 [97.36, 98.25] |
| perception (all four) | 97.66 [97.15, 98.08] | 97.88 [97.39, 98.27] |
| arithmetic | 99.90 [99.75, 99.96] | 99.98 [99.86, 100.00] |
| malformed | 0.00 [0.00, 0.09] | 0.00 [0.00, 0.09] |
| modal-sum baseline | 6.52 | 7.80 |

per-slot perception (val): TL 99.41 [99.13, 99.61] · TR 99.12 [98.79, 99.36] ·
BL 99.58 [99.34, 99.74] · BR 99.49 [99.22, 99.66]

**text -> image**

| metric | val | held-out |
|---|---|---|
| all four | 99.95 [99.82, 99.99] | 99.93 [99.78, 99.98] |
| per quadrant | 99.99 [99.96, 100.00] | 99.98 [99.95, 99.99] |

**joint (unguided)**

| metric | value |
|---|---|
| consistent | 99.46 [99.19, 99.65] |
| text well-formed | 100.00 [99.91, 100.00] |
| addends match image | 99.71 [99.49, 99.83] |
| arithmetic valid | 99.76 [99.55, 99.87] |

**Classifier calibration.** The quadrant CNN's ceiling on REAL composites, same
instrument, n=4,096: val 99.02 per quadrant / 96.09 all four; held-out 99.20 / 96.80;
MNIST *train*-pool composites 99.95 / 99.78.

Text->image must be read against the **train-pool** ceiling of 99.78, not 96.09.
The corpus builds val/held-out composites from MNIST's *test* digits while the
classifier is fit on the *train* pool; generated digits imitate the pool the classifier
knows. Read against 96.09 the model scores "104% of ceiling", which is meaningless.
See `docs/MNIST_SUM_RESULTS.md` §7.1.

## 4. Sequence-layout facts: CONFIRMED, no change needed

Measured directly from `data/mnist_sum_codec.resolve_layout(cfg)`:

- 3,388 bits total = **3,136** image (112 tokens x 28) + **84** text (3 positions x 28)
  + **168** marker (6 markers x 28). Sum verified.
- **121** positions = 112 + 3 + 6.
- 7-bit code, **40** symbols used of the 128-entry space (37 number words `zero`..
  `thirty-six`, plus the structural symbols).

## 5. Not done here

`fig_mnist_sum_compact.pdf` could NOT be regenerated: the script is not in either repo
on csic47 (only `fig1_overview.py`, `fig2_joint.py`, `fig3_results.py`, `gen_assets.py`).
The samples HAVE changed -- this is a different trained model -- so the figure does need
regenerating from the new checkpoint. Send the script and it is a few minutes' work.
