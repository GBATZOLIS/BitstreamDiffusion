# Issue: replace the per-modality regime samplers with one layout-driven sampler

*Filed as part of the MNIST-Sum port (branch `feat/mnist-sum`). Not a blocker for
that run — this is the follow-up the port deliberately deferred.*

## What the code looks like now

`trainers/trainer.py` dispatches partial-denoising regime sampling on the config:

```python
elif cond_mode == "multimodal_mask":
    _, prefix_mask, protect_mask = _sample_regimes_and_cond_masks(...)   # utils/mnist_sum_utils.py
else:
    cond_text_audio = bool(getattr(cond_cfg, 'downstream', False))
    if cond_text_audio:
        from utils.textaudio_utils import _sample_tasks_and_cond_masks   # utils/textaudio_utils.py
        ...
```

Two modules with the same job and the same shape of interface:

| | `utils/textaudio_utils.py` | `utils/mnist_sum_utils.py` |
|---|---|---|
| resolves masks from | `cfg.data.text_seq_len`, `speaker_seq_len`, `cfg.cond.continuation_prefix`, x `bits_per_token` | the codec's `Layout` (`text_bit_mask` / `image_bit_mask` / `marker_bit_mask`) |
| regimes | uncond / TTS / STT / continuation | cond_text / cond_image / joint |
| protected positions | none | the markers |
| entry points | `_sample_tasks_and_cond_masks`, `_fixed_mask` | `_sample_regimes_and_cond_masks`, `_fixed_regime_masks` |

## Why it should change

1. **The optics.** The paper claims modality-agnosticism. A trainer whose regime
   machinery is reached through a module named after one modality pair, chosen by
   an `if`, is the same problem as a `if dataset == "mnistsumbits"` branch — it
   reads as a special case even where the maths is general.
2. **CC3M/CC12M and protein are still to come onto this codebase.** Each will
   otherwise add a third and fourth near-duplicate of the same function.
3. **`protect_mask` only exists on one of the two paths.** It is threaded for
   `multimodal_mask` and is `None` for text+audio. If text+audio ever grows a
   structural position that must survive CFG dropout, the omission will fail the
   same silent way the sampler did before this port (see
   `tests/test_protect_mask.py`).

## Proposed shape

One sampler driven by a layout description rather than by dataset name:

```python
# utils/regimes.py
@dataclass
class Segment:
    name: str                 # "text" | "image" | "speech" | "speaker" | ...
    bit_mask: torch.Tensor    # [S] bool
    structural: bool = False  # never CFG-nulled (markers)

@dataclass
class Regime:
    name: str                 # "i2t" | "t2i" | "joint" | "tts" | "stt" | ...
    condition_on: tuple[str, ...]
    weight: float

def sample_regimes(layout: Sequence[Segment], regimes: Sequence[Regime],
                   B: int, S: int, device) -> tuple[Tensor, Tensor, Tensor]:
    """-> (regime_ids, prefix_mask, protect_mask)"""
```

Each codec already reports its segmentation (`mm_codec` and `mnist_sum_codec`
both expose the `Layout` API); text+audio would need a `Layout` adapter over its
`text_seq_len` / `speaker_seq_len` / `continuation_prefix` arithmetic, which is
the only real work in the change. The regime table then moves into the configs,
where the masking scheme is a stated experimental choice rather than a code path.

## Acceptance

- `trainers/trainer.py` has no modality-specific import and no `cond_mode` branch.
- `protect_mask` is computed the same way for every layout.
- CoBit-LibriTTS / CoBit-MLS training steps are bit-identical to `main` at the
  same seed (the text+audio adapter must reproduce the current masks exactly).
- MNIST-Sum training steps are bit-identical to `feat/mnist-sum`.
