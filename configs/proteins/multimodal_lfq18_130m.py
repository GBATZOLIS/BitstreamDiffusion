"""Multimodal 18-bit main scale run at approximately 130M (CoBit-S) (plan phase 5).

Trained on Tier C paired data plus sequence replay, single node then multi-node
via Slurm. Uses the paper's main Bitstream scale so text and protein share a
recognizable capacity point. Inherits the 35M config and scales the trunk.
"""

from configs.proteins.multimodal_lfq18_35m import get_config as _mid_config


def get_config():
    cfg = _mid_config()
    cfg.experiment = "proteins/multimodal_lfq18_130m_seed0"

    cfg.data.shard_dir = "datasets/bitprotein_monomer_m1"
    cfg.data.max_len = 512
    # Tier C source mix; ESMAtlas stays gated (weight 0) until the probe clears it.
    cfg.data.source_weights.pdb = 3.0
    cfg.data.source_weights.afdb_swissprot = 1.5
    cfg.data.source_weights.afdb_rep = 1.0
    cfg.data.source_weights.esmatlas = 0.0

    # ~130M trunk (CoBit-S scale).
    cfg.model.embed_dim = 640
    cfg.model.dim_ff = 2560
    cfg.model.n_blocks = 16
    cfg.model.n_heads = 10
    cfg.model.content_dim_continuous = 96
    cfg.model.head_hidden = 160
    cfg.model.expected_num_parameters = 125_190_000
    # Gradient checkpointing keeps the paired batch on four GPUs (plan 6.5).
    cfg.model.gradient_checkpointing = True

    # cfg.train.batch_size is the GLOBAL batch (the Trainer divides by world_size):
    # 256 over 8 GPUs is 32 per GPU, kept on device by gradient checkpointing above.
    cfg.train.batch_size = 256
    cfg.train.global_batch_size = 256
    cfg.train.expected_world_size = 8
    cfg.train.steps_per_epoch = 5_000
    cfg.train.epochs = 80

    cfg.optim.total_steps = 400_000
    cfg.optim.warmup = 15_000
    cfg.optim.lr = 1.5e-4

    cfg.train.checkpointing.interval.every_steps = 25_000

    cfg.logging.group = "multimodal_lfq18_130m"
    return cfg
