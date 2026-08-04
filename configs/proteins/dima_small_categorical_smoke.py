from configs.proteins.dima_small_categorical_viability import get_config as _base


def get_config():
    cfg = _base()
    cfg.experiment = "proteins/dima_small_categorical_ddp_smoke"
    cfg.data.limit_train = 1024
    cfg.data.limit_eval = 256
    cfg.data.num_workers = 0
    cfg.train.batch_size = 16
    cfg.train.epochs = 1
    cfg.train.vlb.enabled = False
    cfg.train.early_stopping.enabled = False
    cfg.optim.total_steps = 2
    cfg.optim.warmup = 1
    cfg.train.checkpointing.interval.enabled = False
    cfg.train.checkpointing.resume_interval.enabled = False
    cfg.logging.tensorboard.enabled = False
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/best.pt"
    return cfg
