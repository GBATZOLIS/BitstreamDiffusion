# MNIST-Sum: an exactly-verifiable image+text experiment

Status: dataset builder, codec, Dataset class, exact scorer, training callback and
two configs are implemented and validated on CPU end-to-end. Not yet trained.
Target hardware flexible (developed to run on 2x A6000 or a single A100).

## What it is

An image is four binarised MNIST digits in a 2x2 grid. The text is the equation
that reads them in raster order (TL, TR, BL, BR) and states their sum:

```
image:  [7][6]        text: "seven + six + three + two = eighteen"
        [3][2]
```

## Why it earns a place in the paper

Every image+text number currently in the paper is a proxy: FID/CMMD for images,
CIDEr/CLIPScore for text, a detector for GenEval. None has a ground truth. That
is precisely why the captioning subsection has to argue about register mismatch,
and why `app:image:captioning` has to state that **no published zero-shot COCO
captioning baseline exists**. MNIST-Sum replaces every proxy with an exact answer
in all three directions:

| direction | exact score |
|---|---|
| image -> text | parse the equation; compare addends and sum to ground truth |
| text -> image | classify the four quadrants; compare to the addends named |
| **joint** | generate both, check they are **mutually consistent** |

The joint row is the strongest argument for running this. The paper's joint
direction is currently defended with *pair-CLIP coherence* (25.67), a soft
similarity with no notion of correct. Here, joint co-generation either produces
an image and an equation that agree, or it does not, and the rate is a number.
No COCO experiment can produce that.

## No image tokenizer

The image is binary, so it is carried as raw bits: a 4x4 pixel patch is 16 bits,
which fits a single 19-bit shared-vocab position (18-bit payload + namespace bit).
Consequences:

* **No rFID floor and no codec confound.** Generated and real images live in
  exactly the same space. Contrast the CC3M/CC12M comparison, where two thirds of
  the raw FID gap turned out to be the tokenizer.
* **Neither modality has a learned codec.** The text vocabulary is hand-specified
  (37 number words, "+", "=", PAD -- 40 codes in 7 bits). Training a tokenizer for
  a closed formal language would add a confound for no benefit.
* **The strongest form of the central claim.** Nothing in the image pipeline is
  learned or modality-specific -- the image *is* bits. `mm_codec` needed no
  change: `assemble_code_row` already accepts any payload in `[0, 2^18)`.

## Geometry

Digits are kept at their native 28x28 and the canvas at 56x56 -- **nothing is
resized**. The patch width P is the architecture's `cfg.model.patch_size`, and
the image is chunked into P-bit pieces digit-major, row-major within a digit, so
no patch ever straddles two digits:

| P | image positions | text positions (Lt=12) | markers | total | bits |
|---|---|---|---|---|---|
| 28 | 112 (one digit row each) | 3 | 6 | **121** | 3,388 |
| 14 | 224 (half a digit row) | 6 | 6 | **236** | 3,304 |
| *CC3M for comparison* | 256 | 48 | 6 | 310 | 5,890 |

Both sequences are shorter than CC3M's, so this trains cheaply. Hilbert ordering
is deliberately NOT used: at P=28 a patch is exactly one digit row, so horizontal
locality lives inside the patch and vertical locality is sequential. This is a
documented deviation from the CC3M/CC12M image segments.

**Patch width does not imply tokenizer width.** The trunk patches the flat
bitstream every P bits, but how many semantic tokens occupy those P bits is a
layout choice, which the method explicitly permits. Text tokens are 7 bits, so
4 pack into a P=28 position and 2 into a P=14 one -- the same vocabulary serves
both widths, which makes P=28 vs P=14 a controlled comparison.

## Two sampling controls that the result depends on

**`--balance_sums`.** The sum of four uniform digits is sharply peaked (CLT):
range 0..36 but concentrated near 18, so a model answering "eighteen" every time
scores ~7% **without reading the image**. Balancing drops that to ~4.3%
(measured). The builder records `most_frequent_sum_baseline` in `meta.json` for
both splits either way, so the headline accuracy is always quotable against it.

**`--holdout_combos`.** There are only 10^4 digit tuples, so a large model can
memorise the mapping. Reserving a fraction of *tuples* for validation (every sum
still appears in training, so no output token is unseen) makes the val number a
test of systematic generalisation rather than recall.

## Scoring protocol (to implement)

### image -> text: decompose the error, do not report one number

The text contains both the addends and the sum, so three separable quantities
can be measured. Reporting only end-to-end accuracy would conflate "misread a
digit" with "cannot add", which are completely different failures:

1. **Perception** -- are the four named addends the true digits? (also per
   position TL/TR/BL/BR, which exposes any raster-order or binding bias)
2. **Arithmetic** -- does the stated sum equal the sum of the *stated* addends?
   This is scored even when perception is wrong, isolating arithmetic from
   reading.
3. **End-to-end** -- does the stated sum equal the true sum?

Also report: malformed-output rate (does not parse as the template), and
end-to-end accuracy against the `most_frequent_sum_baseline`.

### text -> image: classify the quadrants, against a ceiling

Train a small CNN on binarised 28x28 MNIST, apply it to the four quadrants of the
generated 64x64 canvas, and compare to the addends named in the prompt. Report
per-quadrant accuracy and the all-four-correct rate.

The classifier's own accuracy on held-out **real** composites is the ceiling, and
must be reported beside the result -- the same discipline the paper already
applies with the rFID floor. Unlike rFID there is no codec loss here, so the
ceiling should be ~99%, and any shortfall is the generative model.

Note the task is one-to-many (any rendering of those digits is correct), which is
the point: it scores semantic correctness, not pixel fidelity.

### joint: mutual consistency

Sample unconditionally, classify the image, parse the text, and report the rate at
which the equation is exactly correct **for the image that was generated with it**.
Report the marginal validity of each half separately too (is the text a
well-formed equation at all; are the digits legible), so a low consistency score
can be attributed.

### text-only

The equation is solvable from text alone (`"one + two + four + nine ="` ->
`"sixteen"`). If a text-only regime is included in training, this isolates
arithmetic from perception entirely, and connects to the paper's open question of
whether joint training damages text modelling.

## Suggested ablations, in order of value

1. **Digit count 1 / 2 / 4** -- a difficulty ladder on the same machinery;
   directly probes whether failure is perception or composition.
2. **Step count per direction.** The paper's sharpest sampler finding is that the
   image optimum (~10 steps) and text optimum (~256) are 25x apart. Here both
   directions have exact accuracy, so that asymmetry can be measured against
   *correctness* rather than against FID and CIDEr, which are not commensurable.
   This is the cheapest way to corroborate the paper's most surprising claim.
3. **Held-out tuples** (`--holdout_combos`) -- memorisation vs generalisation.

## What is implemented

| file | role |
|---|---|
| `data/mnist_sum_codec.py` | shared code space, layout, pack/unpack, equation<->codes |
| `data/mnist_sum_bits.py` | Dataset; `reconstruct_batch_from_bits` needs no tokenizer |
| `evaluation/mnist_sum.py` | `ms_sample` + the exact three-way scorer |
| `utils/callbacks/mnist_sum_eval.py` | i2t accuracy + t2i grid to TensorBoard |
| `configs/multimodal/mnist_sum_p{28,14}_small.py` | 12x512 (~63M), derived from CC3M |
| `scripts/multimodal/phase_50_mnist_sum_dataset.py` | corpus builder |

The configs are **derived by import-and-override** from
`cc3m_lfq19_medium_24x1024_joint.py` rather than copied, so "identical to CC3M
except for the data pipeline and scale" is auditable rather than asserted.
Everything not listed in `_mnist_sum_base.py` is the CC3M recipe unchanged:
AdamW lr 2e-4 / wd 0.01 / clip 1.0 / warmup 5k, EMA 0.9999, bf16, torch.compile,
`binary_sm` + EDM weighting, self-conditioning p=0.5, the sigma block, the whole
entropy schedule (40k warmup / 10k transition), batch 512, and the 0.35 / 0.35 /
0.30 regime split with `p_uncond=0.1`.

Three places in the shared code resolved the layout through `mm_codec`'s 19-bit
geometry and now dispatch on `cfg.data.dataset`, defaulting to `mm_codec` so
CC3M/CC12M are untouched: the training mask sampler
(`trainers/trainer.py`), the segment embedding (`models/sdt.py`), and -- avoided
rather than patched -- `evaluation/multimodal.py`, which this corpus bypasses via
its own `evaluation/mnist_sum.py`.

## Build

```bash
python scripts/multimodal/phase_50_mnist_sum_dataset.py \
    --out datasets/mnist_sum_p28 --patch_size 28 \
    --n_train 1000000 --n_val 10000 --holdout_combos 0.1
```

Sampling is natural (uniform digits). `--balance_sums` exists but is off by
design: rejection-sampling a uniform sum distorts the digit marginals (sum=0
forces 0,0,0,0), and the shortcut it guards against is already handled by scoring
perception separately.

Emits the standard multimodal cache layout (`meta.json`, `{train,val}/shard_*.uint32.mmap`,
`shard_idx.json`) plus, per split, `labels.npz` (`digits [N,4]`, `sums [N]`) and
`equations.txt` -- the ground truth every metric above is checked against.

Verified end-to-end: text round-trips exactly, and decoded images place the digits
in the labelled raster order.
