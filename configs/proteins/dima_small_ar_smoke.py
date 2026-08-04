from configs.proteins.dima_small_ar_viability import get_config as _base


def get_config():
    cfg = _base()
    cfg.experiment = "proteins/dima_small_ar_bos_ddp_smoke"
    cfg.data.limit_train = 512
    cfg.data.limit_eval = 256
    cfg.data.num_workers = 0
    cfg.train.batch_size = 16
    cfg.train.epochs = 1
    cfg.train.use_compile = False
    return cfg
