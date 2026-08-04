"""Seed-0 viability gate before committing to the full three-seed study."""
from configs.proteins.dima35m_bitstream import get_config as _production_config


def get_config():
    cfg = _production_config()
    cfg.experiment = "proteins/dima35m_bitstream_viability_seed0"
    cfg.optim.total_steps = 50_000
    cfg.optim.warmup = 1_000
    cfg.train.early_stopping.warmup_steps = 10_000
    cfg.train.early_stopping.patience_epochs = 5
    cfg.train.checkpointing.interval.every_steps = 10_000
    cfg.train.checkpointing.resume_interval.every_steps = 1_000
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/best.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/protein_eval"
    cfg.evaluation.samples_dir = f"runs/{cfg.experiment}/protein_eval/samples"
    cfg.logging.group = "dima35m_viability"
    return cfg
