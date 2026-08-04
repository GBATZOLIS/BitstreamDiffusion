"""Fast single-GPU launch preflight for the 35M multimodal M0 run.

Inherits the real 35M config unchanged EXCEPT for a tiny step/batch budget, so it
exercises the exact production path on real artifacts before a multi-day launch:
the full 37.7M trunk, warm-start column surgery from the real EvoDiff-UniRef50
checkpoint, real M0 paired data, the sequence-only replay corpus, independent
modality noise, self-conditioning, trunk freeze then unfreeze at a lower LR, and a
mid-run checkpoint plus exact-cursor resume. It is a preflight, not a training run:
its samples and losses are not publishable.

Run:  python train.py --config configs/proteins/multimodal_lfq18_35m_preflight.py
"""

from configs.proteins.multimodal_lfq18_35m import get_config as _real_config


def get_config():
    cfg = _real_config()
    cfg.experiment = "proteins/_multimodal_lfq18_35m_preflight"

    # Keep the full 37.7M trunk (embed_dim/n_blocks/...) so warm-start column
    # surgery is validated against the real checkpoint shapes. Only shrink the
    # budget and batch so the preflight finishes in seconds on one GPU.
    cfg.data.num_workers = 2  # exercise the worker-safe content-derived task RNG
    cfg.data.prefetch_factor = 2

    cfg.train.batch_size = 8  # global == per-GPU on a single-GPU preflight
    cfg.train.global_batch_size = 8
    cfg.train.expected_world_size = 1
    cfg.train.steps_per_epoch = 10  # several short epochs to exercise epoch rollover
    cfg.train.epochs = 100  # bounded by optim.total_steps below
    cfg.train.validation_max_batches = 2
    cfg.train.vlb.enabled = False  # (also guarded off for the multimodal path)

    # Freeze the trunk for the first few steps, then unfreeze at 0.1x LR, so both
    # curriculum phases run inside the preflight.
    cfg.model.warm_start_freeze_trunk_steps = 6

    cfg.optim.total_steps = 30
    cfg.optim.warmup = 5

    # Roll a resume checkpoint often so a mid-epoch resume cursor is produced.
    cfg.train.checkpointing.interval.every_steps = 1_000_000
    cfg.train.checkpointing.resume_interval.enabled = True
    cfg.train.checkpointing.resume_interval.every_steps = 5

    cfg.evaluation.num_sampling_steps = 12
    # The preflight is a throwaway; keep it out of W&B.
    cfg.logging.use_wandb = False
    cfg.logging.mode = "offline"
    cfg.logging.group = "multimodal_lfq18_35m_preflight"
    return cfg
