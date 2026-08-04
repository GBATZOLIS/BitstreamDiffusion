"""Multimodal 18-bit (5 sequence + 13 LFQ structure) smoke config.

This is the base multimodal protein config and the Phase 2 plumbing smoke
(plan section 8.3). It derives from the Swiss-Prot char base and switches the
model into the opt-in multimodal path (``cfg.model.multimodal = True``,
``patch_size = 18``) over the paired DPLM structure tokens. Keep it tiny so an
overfit / single-GPU smoke runs quickly; the 35M and 130M configs inherit this
and scale the trunk.
"""

from configs.proteins.swissprot_char5 import get_config as _base_config
from data.protein_structure_codec import (
    DEFAULT_STRUCT_CODEC,
    PATCH_BITS_PER_RESIDUE,
    SEQ_BITS_PER_RESIDUE,
    STRUCT_BITS_PER_RESIDUE,
)


def get_config():
    cfg = _base_config()
    cfg.experiment = "proteins/multimodal_lfq18_smoke"

    # ------------------------------------------------------------------
    # Data: sharded paired sequence + structure corpus (M0 DPLM-paired)
    # ------------------------------------------------------------------
    cfg.data.dataset = "ProteinMultimodalLFQ"
    cfg.data.shard_dir = "datasets/dplm_paired_m0"
    cfg.data.representation = "binary"
    cfg.data.binarization = "raw_binary"
    cfg.data.patch_bits = PATCH_BITS_PER_RESIDUE  # 18
    cfg.data.seq_bits = SEQ_BITS_PER_RESIDUE  # 5
    cfg.data.struct_bits = STRUCT_BITS_PER_RESIDUE  # 13
    cfg.data.struct_convention_hash = DEFAULT_STRUCT_CODEC.convention_hash()
    cfg.data.min_len = 40
    cfg.data.max_len = 256
    cfg.data.vocab_size = 2
    cfg.data.num_workers = 4
    cfg.data.prefetch_factor = 2
    # Sampler: source balancing + explicit length buckets, resumable.
    cfg.data.source_weights = config_type(cfg)()
    cfg.data.source_weights.pdb = 2.0
    cfg.data.source_weights.afdb_swissprot = 1.0
    cfg.data.source_weights.afdb_rep = 1.0
    cfg.data.source_weights.esmatlas = 0.0
    # Sequence-only UniRef50 replay fraction by token budget (plan 6.6 step 3).
    cfg.data.sequence_replay_fraction = 0.0  # off for the plumbing smoke
    # Paired-format shard dir of sequence-only rows (struct_mask all False) mixed
    # in at the replay fraction. Empty disables replay even if the fraction > 0.
    cfg.data.sequence_replay_root = ""

    # Per-example task mix (plan phase 4). Tuned on validation, not fixed blindly.
    cfg.data.task_weights = config_type(cfg)()
    cfg.data.task_weights.joint = 0.30
    cfg.data.task_weights.forward_folding = 0.20
    cfg.data.task_weights.inverse_folding = 0.20
    cfg.data.task_weights.structure_marginal = 0.10
    cfg.data.task_weights.sequence_marginal = 0.10
    cfg.data.task_weights.motif = 0.10

    # ------------------------------------------------------------------
    # Model: opt-in multimodal path over 18-bit residue patches
    # ------------------------------------------------------------------
    cfg.model.multimodal = True
    cfg.model.patch_size = (
        PATCH_BITS_PER_RESIDUE  # 18: one residue per trunk token
    )
    cfg.model.self_condition = False  # keep the plumbing smoke simple
    cfg.model.embed_dim = 128
    cfg.model.dim_ff = 512
    cfg.model.n_blocks = 4
    cfg.model.n_heads = 4
    cfg.model.content_dim_continuous = 32
    cfg.model.head_hidden = 64
    # The multimodal loss (multimodal_bit_loss) supervises the RAW model logits,
    # so sampling must read them raw too. Override the sequence base's
    # matched_filter_residual scaling to identity, else generation would apply a
    # logit transform the model never trained against and corrupt every bit.
    cfg.model.continuous_logit_scaling = "none"

    # Equal-modality loss weighting (plan 6.3): start equal, not equal-per-bit.
    cfg.model.lambda_seq = 1.0
    cfg.model.lambda_struct = 1.0
    # Independent modality noise; the first integration smoke may share sigma.
    cfg.model.independent_modality_noise = True

    # Warm start from the sequence-only checkpoint via column surgery (plan 6.4).
    cfg.model.warm_start_seq_checkpoint = (
        ""  # set to a UniRef50 best.pt to enable
    )
    cfg.model.warm_start_freeze_trunk_steps = 0  # curriculum step 1 (adapters only)
    cfg.model.warm_start_trunk_lr_mult = 1.0  # curriculum step 2 (lower trunk LR)
    cfg.model.warm_start_use_ema = True  # warm-start from the source EMA weights

    # ------------------------------------------------------------------
    # Training: step-based, tiny budget for the smoke
    # ------------------------------------------------------------------
    cfg.train.loss_type = "binary_ce"
    cfg.train.batch_size = 16
    cfg.train.global_batch_size = 16
    cfg.train.expected_world_size = 1
    cfg.train.steps_per_epoch = 200
    cfg.train.epochs = 5
    cfg.train.validation_max_batches = 20
    cfg.train.vlb.enabled = False

    cfg.optim.total_steps = 1_000
    cfg.optim.warmup = 50
    cfg.optim.lr = 3e-4

    cfg.train.checkpointing.interval.every_steps = 500
    cfg.train.checkpointing.resume_interval.every_steps = 200

    # ------------------------------------------------------------------
    # Evaluation / sampling
    # ------------------------------------------------------------------
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/protein_eval"
    cfg.evaluation.samples_dir = f"runs/{cfg.experiment}/protein_eval/samples"
    cfg.evaluation.num_sampling_steps = 64
    cfg.evaluation.length_grid = [100, 200]
    cfg.evaluation.samples_per_length = 16

    cfg.logging.group = "multimodal_lfq18_smoke"
    return cfg


def config_type(cfg):
    """Return the ConfigDict class so nested sub-configs match the base type."""
    return type(cfg)
