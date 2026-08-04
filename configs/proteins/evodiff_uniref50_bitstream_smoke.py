"""Small CPU/single-GPU data and training smoke test for the EvoDiff protocol."""
from configs.proteins.evodiff_uniref50_bitstream import get_config as _base


def get_config():
    cfg = _base()
    cfg.experiment = "proteins/evodiff_uniref50_bitstream_smoke"
    cfg.data.limit_train = 128
    cfg.data.limit_eval = 64
    cfg.data.max_len = 128
    cfg.data.sequence_len_tokens = 128
    cfg.data.sequence_len = 128 * 5
    cfg.data.num_workers = 0
    cfg.model.embed_dim = 128
    cfg.model.dim_ff = 256
    cfg.model.n_blocks = 2
    cfg.model.n_heads = 4
    cfg.model.expected_num_parameters = 0
    cfg.train.batch_size = 4
    cfg.train.global_batch_size = 4
    cfg.train.expected_world_size = 1
    cfg.train.steps_per_epoch = 2
    cfg.train.validation_max_batches = 2
    cfg.train.epochs = 1
    cfg.train.vlb.enabled = False
    cfg.train.early_stopping.enabled = False
    cfg.optim.total_steps = 2
    cfg.optim.warmup = 0
    cfg.train.checkpointing.interval.enabled = False
    cfg.train.checkpointing.resume_interval.enabled = False
    cfg.logging.tensorboard.enabled = False
    return cfg
