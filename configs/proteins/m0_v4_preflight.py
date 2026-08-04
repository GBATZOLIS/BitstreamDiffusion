"""Two-step, no-checkpoint memory preflight for the one-GPU m0_v4 batch."""
from configs.proteins.m0_v4 import build_config


def get_config():
    cfg = build_config(0.50, "preflight")
    cfg.experiment = "proteins/m0_v4_preflight"
    cfg.train.steps_per_epoch = 4
    cfg.train.epochs = 1
    # The first deterministic length bucket is a single structure-only row;
    # include the second bucket so balanced sequence/structure MSE is defined.
    cfg.train.validation_max_batches = 2
    cfg.optim.total_steps = 2
    cfg.train.checkpointing.save_last = False
    cfg.train.checkpointing.save_top_k = 0
    cfg.train.checkpointing.interval.enabled = False
    cfg.train.checkpointing.resume_interval.enabled = False
    cfg.logging.use_wandb = False
    return cfg
