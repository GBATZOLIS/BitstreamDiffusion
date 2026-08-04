"""M0 v4: one-GPU base-distribution-floor sweep for the 125M multimodal model.

All variants preserve the m0_v3 model, corpus, loss, task mix, entropy warmup,
and effective batch. They differ only in the final fraction of sigma draws that
remain on the base log-normal law. Use the explicit variant configs for the
four-way sweep; this module's default is the recommended 50% mixture.
"""

from configs.proteins.m0_v3 import get_config as _m0_v3


def build_config(base_fraction: float, suffix: str):
    if not 0.0 <= float(base_fraction) <= 1.0:
        raise ValueError(f"base_fraction must be in [0, 1], got {base_fraction}")

    cfg = _m0_v3()
    cfg.experiment = f"proteins/m0_v4_base{suffix}"

    # One 98-GiB GPU. This preserves m0_v3's effective batch of 256 while
    # increasing the per-device micro-batch from 32 to 128.
    cfg.train.batch_size = 128
    cfg.train.global_batch_size = 256
    cfg.train.grad_accum_steps = 2
    cfg.train.expected_world_size = 1

    # Slightly below m0_v3's 1.5e-4. Keep the agreed full training horizon.
    cfg.optim.lr = 1.2e-4
    cfg.optim.warmup = 10_000
    cfg.optim.total_steps = 250_000
    cfg.train.steps_per_epoch = 5_000
    cfg.train.epochs = 110

    # The entropy controller applies per-bin coverage smoothing, then retains
    # this fraction of the original truncated log-normal base distribution.
    cfg.train.entropy_base_fraction = float(base_fraction)

    # A responsive scratch-run EMA that becomes the original long EMA by 10k.
    cfg.train.ema_ramp = type(cfg.train)()
    cfg.train.ema_ramp.enabled = True
    cfg.train.ema_ramp.start_decay = 0.99
    cfg.train.ema_ramp.end_decay = 0.9999
    cfg.train.ema_ramp.steps = 10_000

    # Select best.pt with equal sequence/structure unweighted validation MSE.
    # The legacy EDM-weighted loss remains logged as loss/epoch_val.
    cfg.train.checkpointing.metric = "balanced_mse"
    if not hasattr(cfg.train, "early_stopping"):
        cfg.train.early_stopping = type(cfg.train)()
    cfg.train.early_stopping.enabled = False
    cfg.train.checkpointing.interval.every_steps = 25_000
    cfg.train.checkpointing.resume_interval.every_steps = 2_000

    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/best.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/protein_eval"
    cfg.evaluation.samples_dir = f"runs/{cfg.experiment}/protein_eval/samples"

    cfg.logging.run_name = f"m0_v4_base{suffix}"
    cfg.logging.run_id = f"m0_v4_base{suffix}"
    cfg.logging.group = "m0_v4"
    return cfg


def get_config():
    return build_config(0.50, "050")
