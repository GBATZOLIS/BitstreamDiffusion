"""
utils/mnist_sum_utils.py

Regime / conditioning-mask sampling for the shared-vocabulary image+text
bitstream (``cfg.cond.cond_mode == "multimodal_mask"``).

Deliberately parallel to ``utils/textaudio_utils.py`` rather than a
generalisation of it. That module's ``_sample_tasks_and_cond_masks`` resolves
its masks from ``cfg.data.text_seq_len`` / ``speaker_seq_len`` and a
``bits_per_token`` multiplier, which is specific to the text+audio layout; this
one resolves them through the codec's ``Layout`` API, which reports the
text / image / marker bit masks directly.

Both are layout-specific in the same way, and the trainer currently dispatches
between them on ``cfg.cond``. The right structure is one layout-driven regime
sampler that text+audio, MNIST-Sum and CC3M all share -- see the tracking issue
referenced in docs/HANDOFF_UNIFY_ARCH_MNIST_SUM.md §2. This module is the
minimal version that keeps the published masking semantics bit-exact.

Masking semantics (unchanged from the run that produced the published table):

  cond_text   -> clamp the TEXT region clean,  loss on image  => p(image | text)
  cond_image  -> clamp the IMAGE region clean, loss on text   => p(text | image)
  joint       -> clamp nothing extra,          loss on both   => p(image, text)

Marker positions are ALWAYS clamped clean in every regime (they are structural,
not content), and are returned separately as ``protect_mask`` so that the CFG
dropout path can keep them at their true codes even when the conditioning
modality is dropped. §3.3 of the paper: the conditioning segments are dropped
"but never the markers".
"""
from __future__ import annotations

import torch
from ml_collections import config_dict

# Regime ids. Unlike textaudio's task ids there is no UNCONDITIONAL regime:
# the markers are clamped in every regime, so conditioning is never fully off.
COND_TEXT = 0    # text -> image
COND_IMAGE = 1   # image -> text
JOINT = 2
REGIMES = [COND_TEXT, COND_IMAGE, JOINT]


def _resolve_codec(cfg):
    """Pick the codec whose Layout backs this run.

    Both codecs expose the same Layout API (text / image / marker bit masks), so
    everything below is modality-agnostic. mm_codec's 19-bit geometry backs the
    published CC3M/CC12M checkpoints and stays the default.
    """
    name = str(getattr(getattr(cfg, "data", object()), "dataset", "")).lower()
    if name in {"mnistsumbits", "mnist_sum_bits"}:
        from data import mnist_sum_codec as codec
    else:
        from data import mm_codec as codec
    return codec


def _layout_masks(cfg, S: int, device: torch.device):
    codec = _resolve_codec(cfg)
    layout = codec.resolve_layout(cfg)
    if layout.total_bits != int(S):
        raise ValueError(
            f"multimodal layout total_bits={layout.total_bits} != sequence S={S}. "
            f"Check cfg.data.caption_len_tokens / num_image_tokens vs the cache.")
    return (
        layout.text_bit_mask(device).view(1, S),
        layout.image_bit_mask(device).view(1, S),
        layout.marker_bit_mask(device).view(1, S),
    )


def _sample_regimes_and_cond_masks(
    cfg: config_dict.ConfigDict, B: int, S: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    regime_ids   : [B] long, values in REGIMES
    prefix_mask  : [B,S] bool -- clean / conditioned positions
    protect_mask : [B,S] bool -- marker positions, never CFG-nulled
    """
    text_m, image_m, marker_m = _layout_masks(cfg, S, device)

    cond_cfg = getattr(cfg, "cond", None)
    weights = torch.tensor(
        [
            float(getattr(cond_cfg, "p_cond_text", 0.35)),
            float(getattr(cond_cfg, "p_cond_image", 0.35)),
            float(getattr(cond_cfg, "p_joint", 0.30)),
        ],
        device=device,
        dtype=torch.float32,
    ).clamp_min(0.0)
    weights = weights / weights.sum().clamp_min(1e-8)

    regime_ids = torch.multinomial(weights, num_samples=B, replacement=True)  # [B]
    is_ct = (regime_ids == COND_TEXT).view(B, 1)
    is_ci = (regime_ids == COND_IMAGE).view(B, 1)

    prefix_mask = marker_m.expand(B, S).clone()              # markers always clean
    prefix_mask = prefix_mask | (is_ct & text_m) | (is_ci & image_m)
    protect_mask = marker_m.expand(B, S).contiguous()
    return regime_ids, prefix_mask, protect_mask


def _fixed_regime_masks(
    cfg: config_dict.ConfigDict, B: int, S: int, regime: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic single-regime variant, the analogue of textaudio's
    ``_fixed_mask``. Evaluation resolves its own masks through
    ``evaluation.mnist_sum.task_masks``; this exists for interface parity and
    for diagnostics that want one regime across a whole batch."""
    text_m, image_m, marker_m = _layout_masks(cfg, S, device)
    prefix_mask = marker_m.expand(B, S).clone()
    if regime == COND_TEXT:
        prefix_mask = prefix_mask | text_m
    elif regime == COND_IMAGE:
        prefix_mask = prefix_mask | image_m
    elif regime != JOINT:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
    protect_mask = marker_m.expand(B, S).contiguous()
    return prefix_mask, protect_mask
