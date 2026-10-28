# MNIST-Sum offline evaluation

- checkpoint: `runs/multimodal/mnist_sum_p28_12x512_smoke/checkpoints/step=000000200.pt` (step 200)
- weights: EMA, compiled: True
- n per direction per split: 32; steps i2t/t2i: 8/8


## Classifier ceiling on REAL composites

| split | all four | per quadrant |
|---|---|---|
| val | 97.66 | 99.41 |
| val_holdout | 95.31 | 98.83 |

## image → text (exact, no instrument)

| split · guidance | perception | arithmetic | end-to-end | e2e \| percep | malformed | baseline |
|---|---|---|---|---|---|---|
| val_gs2 | 0.00 <sub>[0.00, 10.72]</sub> | 0.00 <sub>[0.00, 10.72]</sub> | **0.00 <sub>[0.00, 10.72]</sub>** | — | 100.00 <sub>[89.28, 100.00]</sub> | 7.80 |
| val_holdout_gs2 | 0.00 <sub>[0.00, 10.72]</sub> | 0.00 <sub>[0.00, 10.72]</sub> | **0.00 <sub>[0.00, 10.72]</sub>** | — | 100.00 <sub>[89.28, 100.00]</sub> | 7.60 |
| val_gs1 | 0.00 <sub>[0.00, 10.72]</sub> | 0.00 <sub>[0.00, 10.72]</sub> | **0.00 <sub>[0.00, 10.72]</sub>** | — | 100.00 <sub>[89.28, 100.00]</sub> | 7.80 |
| val_holdout_gs1 | 0.00 <sub>[0.00, 10.72]</sub> | 0.00 <sub>[0.00, 10.72]</sub> | **0.00 <sub>[0.00, 10.72]</sub>** | — | 100.00 <sub>[89.28, 100.00]</sub> | 7.60 |

## text → image (via quadrant classifier)

| split · guidance | all four | per quadrant | ceiling (all four) | % of ceiling |
|---|---|---|---|---|
| val_gs2 | **0.00 <sub>[0.00, 10.72]</sub>** | 9.38 <sub>[5.44, 15.67]</sub> | 97.66 | 0.0 |
| val_holdout_gs2 | **0.00 <sub>[0.00, 10.72]</sub>** | 3.12 <sub>[1.22, 7.76]</sub> | 95.31 | 0.0 |
| val_gs1 | **0.00 <sub>[0.00, 10.72]</sub>** | 9.38 <sub>[5.44, 15.67]</sub> | 97.66 | 0.0 |
| val_holdout_gs1 | **0.00 <sub>[0.00, 10.72]</sub>** | 3.12 <sub>[1.22, 7.76]</sub> | 95.31 | 0.0 |

## joint (both halves from noise) — mutual consistency

| metric | value |
|---|---|
| text well-formed | 0.00 <sub>[0.00, 10.72]</sub> |
| addends match the image | 0.00 <sub>[0.00, 10.72]</sub> |
| arithmetic valid | 0.00 <sub>[0.00, 10.72]</sub> |
| **consistent (equation correct for its image)** | **0.00 <sub>[0.00, 10.72]</sub>** |
| classifier ceiling | 97.66 |

Intervals are 95% Wilson. Anything read through the classifier is bounded by its ceiling on real composites, which is reported beside it.