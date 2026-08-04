"""Small canonical-residue autoregressive viability baseline."""
from ml_collections import config_dict


def get_config():
    cfg = config_dict.ConfigDict()
    cfg.experiment = "proteins/dima_small_ar_viability_seed0"
    cfg.device = "cuda"

    cfg.data = config_dict.ConfigDict()
    cfg.data.dataset = "SwissProtDiMA"
    cfg.data.root = "datasets/swissprot_dima_bf4b2f13"
    cfg.data.hf_revision = "bf4b2f131664fe87ef5e0fc9e53f01f0d030dcab"
    cfg.data.representation = "tokens"
    cfg.data.sequence_len_tokens = 255
    cfg.data.sequence_len = 255
    cfg.data.vocab_size = 21
    cfg.data.bits_per_token = 5
    cfg.data.prepend_bos = True
    cfg.data.num_workers = 8
    cfg.data.prefetch_factor = 4
    cfg.data.pin_memory = True
    cfg.data.drop_last_train = True

    cfg.model = config_dict.ConfigDict()
    cfg.model.vocab_size = 21
    cfg.model.max_seq_len = 254
    cfg.model.n_layer = 6
    cfg.model.n_head = 8
    cfg.model.d_model = 320
    cfg.model.mlp_mult = 4.0
    cfg.model.dropout = 0.1
    cfg.model.rope_base = 10_000.0
    cfg.model.use_flash_attn = True

    cfg.train = config_dict.ConfigDict()
    cfg.train.seed = 42
    cfg.train.deterministic = False
    cfg.train.batch_size = 256
    cfg.train.epochs = 27
    cfg.train.grad_accum_steps = 1
    cfg.train.ema_decay = 0.9999
    cfg.train.eval_with_ema = True
    cfg.train.use_compile = False
    cfg.train.compile_mode = "default"
    cfg.train.use_fp16 = True
    cfg.train.save_last = True
    cfg.train.save_top_k = 2
    cfg.train.checkpoint_mode = "min"

    cfg.optim = config_dict.ConfigDict()
    cfg.optim.optimizer = "AdamW"
    cfg.optim.lr = 2e-4
    cfg.optim.beta1 = 0.9
    cfg.optim.beta2 = 0.99
    cfg.optim.eps = 1e-8
    cfg.optim.weight_decay = 0.01
    cfg.optim.grad_clip = 1.0
    cfg.optim.scheduler = "cosine"
    cfg.optim.warmup = 1000
    cfg.optim.fused = True

    cfg.logging = config_dict.ConfigDict()
    cfg.logging.log_freq = 50
    from configs.proteins._wandb import enable_wandb

    enable_wandb(cfg, group="dima_small_ar_viability")
    return cfg
