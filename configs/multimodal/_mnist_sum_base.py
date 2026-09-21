"""
configs/multimodal/_mnist_sum_base.py

Shared builder for the MNIST-Sum configs. See docs/MNIST_SUM_EXPERIMENT.md.

The config is DERIVED from the CC3M config by import-and-override, rather than
copied. That is deliberate: the paper's claim is that nothing outside the data
pipeline changes between modality pairs, and this makes that claim auditable --
everything not listed in OVERRIDES below is, by construction, byte-identical to
the CC3M recipe (AdamW lr 2e-4 / wd 0.01 / grad-clip 1.0 / warmup 5k, EMA 0.9999,
bf16, torch.compile, binary_sm loss with EDM weighting, self-conditioning p=0.5,
the whole sigma block, and the entire entropy-schedule block including its 40k
warmup and 10k transition).

OVERRIDES, all of which are modality/geometry or scale:
  data.*        -> the MNIST-Sum corpus (no image tokenizer, 7-bit text codec)
  model.patch_size, embed_dim/n_blocks/dim_ff/n_heads  -> P, and 512-wide scale-down
                                                          (NOTHING else in model.* is
                                                          overridden: the backbone is
                                                          stock, as for the audio runs)
  train.mnist_sum_eval                                 -> the exact-accuracy callback
  train.visualization.mm_tasks                         -> off (the MM viz branch
                                                          resolves masks through
                                                          mm_codec's 19-bit layout)
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from ml_collections import config_dict

from data import mnist_sum_codec as C

_CC3M = Path(__file__).with_name("cc3m_lfq19_medium_24x1024_joint.py")


def _load_cc3m():
    spec = importlib.util.spec_from_file_location("_cc3m_cfg", _CC3M)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


def build(*, patch_size: int, caption_len_tokens: int = 12,
          embed_dim: int = 512, n_blocks: int = 12, dim_ff: int = 2048,
          n_heads: int = 8, precomputed_root: str | None = None):
    cfg = _load_cc3m()
    layout = C.build_layout(patch_size, caption_len_tokens)

    cfg.experiment = f"multimodal/mnist_sum_p{patch_size}_{n_blocks}x{embed_dim}"

    # ---------------- data: the only genuinely modality-specific part -------
    root = (precomputed_root or os.environ.get("MS_DATA_ROOT")
            or f"datasets/mnist_sum_p{patch_size}")
    cfg.data.dataset = "MNISTSumBits"
    cfg.data.precomputed_root = root
    cfg.data.caption_len_tokens = caption_len_tokens
    cfg.data.num_image_tokens = layout.num_image_tokens
    cfg.data.num_positions = layout.num_positions
    cfg.data.sequence_len = layout.total_bits
    cfg.data.bits_per_token = patch_size
    cfg.data.text_tokenizer = "mnist_sum_codec/7bit"
    cfg.data.flatten_order = "digit_major_row_major"
    for k in ("image_resolution", "token_grid_h", "token_grid_w"):
        if k in cfg.data:
            del cfg.data[k]
    cfg.data.vocab_size = 2
    cfg.data.channels = 1
    cfg.data.return_shape = "flat"

    # ---------------- model: same architecture, scaled down -----------------
    cfg.model.patch_size = patch_size
    cfg.model.embed_dim = embed_dim
    cfg.model.n_blocks = n_blocks
    cfg.model.dim_ff = dim_ff
    cfg.model.n_heads = n_heads
    # Intra-patch positional encoding is inherited from CC3M unchanged
    # (n_fourier_local = 4, intra-patch period P-1). This is the stock
    # architecture, identical to the audio models: MNIST-Sum deliberately adds
    # no architecture override, so "the backbone is identical across modality
    # pairs" holds by construction rather than by audit.

    # ---------------- run length -------------------------------------------
    # 500k steps, not the 1M CC3M used: measured at 336 ms/step (P=28, bs=512,
    # A100) that is ~47 h, and the exact-accuracy callback means we can stop
    # earlier still if perception/end_to_end saturate. total_steps is a hard cap;
    # epochs is set high enough not to bind first (1M rows / batch 512 = ~1953
    # steps per epoch, so 500k steps is ~256 epochs).
    cfg.optim.total_steps = int(os.environ.get("MS_TOTAL_STEPS", "500000"))
    cfg.train.epochs = 400

    # ---------------- the exact-accuracy callback ---------------------------
    cfg.train.mnist_sum_eval = config_dict.ConfigDict()
    cfg.train.mnist_sum_eval.enabled = True
    cfg.train.mnist_sum_eval.every_k_steps = int(os.environ.get("MS_EVAL_EVERY", "50000"))
    cfg.train.mnist_sum_eval.use_ema = True
    cfg.train.mnist_sum_eval.n_i2t = 256      # exact accuracy sample
    cfg.train.mnist_sum_eval.n_t2i = 64       # image grid
    cfg.train.mnist_sum_eval.steps_i2t = 256
    cfg.train.mnist_sum_eval.steps_t2i = 256

    # Each direction is scored at BOTH guidance settings every firing, so the
    # CFG effect is a measured curve rather than one chosen point. 1.0 is the
    # repo's "no CFG amplification" (matching cc3m's i2t_guidance_scales
    # [1.0, 2.0]); the first entry is the primary and keeps the bare
    # mnist_sum/i2t tag, the rest get a suffix.
    cfg.train.mnist_sum_eval.guidance_scales_i2t = [2.0, 1.0]
    cfg.train.mnist_sum_eval.guidance_scales_t2i = [2.0, 1.0]

    # EDM churn, as a per-step rate. s_churn is derived at the actual step count
    # (s_churn = gamma * (steps - 1)); never set s_churn directly here -- CC3M's
    # static 22.0 is its 128-step viz constant and would mean gamma ~= 0.086 at
    # 256 steps. Values are CC3M's tuned per-direction settings
    # (i2t_gamma 0.175, t2i_gamma 0.0); OWT text landed on 0.130 at NFE=256,
    # best of {0.130, 0.175, 0.180}. t2i is deterministic there by measurement,
    # not by omission -- override to sweep it.
    cfg.train.mnist_sum_eval.gamma_i2t = float(os.environ.get("MS_I2T_GAMMA", 0.175))
    cfg.train.mnist_sum_eval.gamma_t2i = float(os.environ.get("MS_T2I_GAMMA", 0.0))
    cfg.train.mnist_sum_eval.s_noise = 1.003
    cfg.train.mnist_sum_eval.churn_window_mode = "full"
    cfg.train.mnist_sum_eval.sigma_decode = 0.1
    cfg.train.mnist_sum_eval.micro_batch = 64
    cfg.train.mnist_sum_eval.sampler = "ddim_entropic"

    # The generic MM viz branch resolves its masks through mm_codec's 19-bit
    # layout, so it cannot run on this corpus; MNISTSumEvalCallback replaces it.
    cfg.train.visualization.enabled = False

    # ---------------- evaluation block ---------------------------------------
    if "multimodal" in cfg.evaluation:
        cfg.evaluation.multimodal.decode_cache = root
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/last.pt"

    # Smoke override: tiny corpus + tiny batch so a shakedown fits on one GPU and
    # the callback fires on the first epoch.
    _smoke = int(os.environ.get("SMOKE_MAX_STEPS", "0") or 0)
    if _smoke > 0:
        # The CC3M smoke block already applied its own cap inside _load_cc3m(),
        # but the run-length section above unconditionally resets total_steps and
        # epochs, so re-apply the cap here or SMOKE_MAX_STEPS is silently dead.
        # epochs stays high on purpose: the callback only has on_epoch_end, so a
        # smoke needs several short epochs to fire more than once before the cap.
        cfg.optim.total_steps = _smoke
        # Same reason: cfg.experiment was re-derived above, dropping the CC3M
        # smoke block's suffix, which would drop smoke checkpoints into the real
        # run's directory for the real run to then resume from.
        cfg.experiment = f"{cfg.experiment}_smoke"
        cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/last.pt"
        cfg.train.batch_size = int(os.environ.get("SMOKE_BATCH", "16"))
        cfg.train.mnist_sum_eval.every_k_steps = int(os.environ.get("MS_EVAL_EVERY", "10"))
        # Eval shapes default to tiny, but are env-overridable so a smoke can be
        # run at the PRODUCTION eval shapes. The deadlock fix is shape-dependent
        # in principle (it is recompiles that desync), and the production eval
        # drives shapes a tiny smoke never reaches: micro_batch 64 doubled to 128
        # by the CFG batch at the full sequence length. Being able to demonstrate
        # that rather than expect it is worth the four env vars. Defaults are
        # unchanged, so this is inert unless asked for.
        _ev = cfg.train.mnist_sum_eval
        _ev.n_i2t = int(os.environ.get("MS_N_I2T", "8"))
        _ev.n_t2i = int(os.environ.get("MS_N_T2I", "8"))
        _ev.steps_i2t = int(os.environ.get("MS_STEPS_I2T", "8"))
        _ev.steps_t2i = int(os.environ.get("MS_STEPS_T2I", "8"))
        # keep both guidance settings in the smoke: the dual-guidance loop is
        # exactly the plumbing a smoke exists to exercise.
        _ev.micro_batch = int(os.environ.get("MS_MICRO", "8"))

    return cfg
