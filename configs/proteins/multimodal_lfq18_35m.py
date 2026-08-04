"""Multimodal 18-bit model at the reproducible 35-40M scale (plan phase 4).

Trained on the full pinned DPLM-220k paired dataset with sequence replay. This
is the reproducible first result and the compute/data-matched point against
DPLM-2 Bit. Inherits the smoke config and scales the shared trunk.
"""

from configs.proteins.multimodal_lfq18_smoke import get_config as _smoke_config


def get_config():
    cfg = _smoke_config()
    cfg.experiment = "proteins/multimodal_lfq18_35m_seed0"

    cfg.data.max_len = 512
    cfg.data.num_workers = 8
    cfg.data.prefetch_factor = 4
    # 25 percent sequence-only UniRef50 replay by token budget (plan 6.6 step 3).
    # The root must be the built sequence-only paired-shard corpus (with a
    # manifest.json + RowRecord .npz shards), NOT the raw EvoDiff dataset dir.
    # Build it with scripts/proteins/setup/prepare_uniref50_replay.py.
    cfg.data.sequence_replay_fraction = 0.25
    cfg.data.sequence_replay_root = "datasets/uniref50_seqonly_replay"

    # ~35-40M trunk, mirroring the DiMA/EvoDiff bitstream scale but at patch 18.
    cfg.model.self_condition = True
    cfg.model.embed_dim = 384
    cfg.model.dim_ff = 1536
    cfg.model.n_blocks = 13
    cfg.model.n_heads = 8
    cfg.model.content_dim_continuous = 64
    cfg.model.head_hidden = 128
    cfg.model.expected_num_parameters = 37_744_000
    cfg.model.warm_start_seq_checkpoint = (
        "runs/proteins/evodiff_uniref50_bitstream_seed0/checkpoints/best.pt"
    )
    cfg.model.warm_start_freeze_trunk_steps = 2_000
    # Curriculum step 2: after the freeze window the warm-started trunk trains at a
    # lower LR than the fresh multimodal adapters (plan 6.6). Without this the trunk
    # would train at the full base LR and the "lower trunk LR" phase is a no-op.
    cfg.model.warm_start_trunk_lr_mult = 0.1

    cfg.train.loss_type = "binary_sm"
    # The Trainer treats cfg.train.batch_size as the GLOBAL batch and divides it by
    # world_size, so set it to the intended global batch (256): on 4 GPUs that is
    # 64 per GPU. The LR and warmup below are tuned for this global batch.
    cfg.train.batch_size = 256
    cfg.train.global_batch_size = 256
    cfg.train.expected_world_size = 4
    cfg.train.steps_per_epoch = 5_000
    cfg.train.epochs = 40
    cfg.train.validation_max_batches = 100
    cfg.train.vlb.enabled = True
    cfg.train.vlb.splits = ["val"]
    cfg.train.vlb.max_batches_val = 50

    cfg.optim.total_steps = 200_000
    cfg.optim.warmup = 10_000
    cfg.optim.lr = 2e-4

    cfg.train.checkpointing.interval.every_steps = 20_000
    cfg.train.checkpointing.resume_interval.every_steps = 2_000

    cfg.evaluation.num_sampling_steps = 250
    cfg.evaluation.length_grid = [100, 200, 300, 400, 500]
    cfg.evaluation.samples_per_length = 200

    # Weights & Biases (on by default, env-overridable via WANDB_MODE/ENTITY/
    # PROJECT/COBIT_WANDB). Mirrors the full TensorBoard stream into the run and
    # writes wandb_run.json for eval to attach. Auth comes from `wandb login`
    # (~/.netrc) or WANDB_API_KEY (kept under .trentinium/keys/wandb.env).
    from configs.proteins._wandb import enable_wandb

    enable_wandb(cfg, group="multimodal_lfq18_35m")
    return cfg
