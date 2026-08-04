"""Multimodal 18-bit foundation model, first-run scale ~400M (plan: full run).

This is the recommended FIRST foundation-model size (see
.trentinium/docs/full_run_and_evaluation.md): ~400M is compute-optimal for the
well-fed sequence/trunk budget (~8.3B UniRef50 replay tokens => Chinchilla-optimal
at ~415M), memorizes the scarce ~0.3-0.5B paired-structure tokens the least, and
runs ~1.5x cheaper than 650M. Trained on the M1 corpus plus heavier sequence
replay. Inherits the 130M config and scales the trunk.

Warm start: there is no width-matched (1024-dim) UniRef50 sequence checkpoint, so
the trunk is trained from scratch here and the sequence side is fed by the raised
replay fraction. To warm-start properly, first pretrain a 1024-dim UniRef50
sequence model and point warm_start_seq_checkpoint at it.
"""

from configs.proteins.multimodal_lfq18_130m import get_config as _cobit_s


def get_config():
    cfg = _cobit_s()
    cfg.experiment = "proteins/multimodal_lfq18_400m_seed0"

    # ~393M trunk (measured from models/sdt.py). Depth favored for the cross-modal
    # folding / inverse-folding / co-design conditionals. rpb_max_distance stays 1
    # so SDPA keeps the flash kernel (a larger value forces a float attn bias).
    cfg.model.embed_dim = 1024
    cfg.model.dim_ff = 4096
    cfg.model.n_blocks = 20
    cfg.model.n_heads = 16
    cfg.model.content_dim_continuous = 128
    cfg.model.head_hidden = 224
    cfg.model.expected_num_parameters = 393_000_000

    # No width-matched sequence checkpoint at this trunk width: train from scratch
    # (the raised replay fraction feeds the sequence marginal). See module docstring.
    cfg.model.warm_start_seq_checkpoint = ""
    cfg.model.warm_start_freeze_trunk_steps = 0
    cfg.model.warm_start_trunk_lr_mult = 1.0

    # Raise sequence replay from 0.25 -> 0.40 to dilute paired-data repetition at
    # the larger scale (structure-paired is the scarce, overfit-prone modality).
    cfg.data.sequence_replay_fraction = 0.40

    # Global batch (the Trainer divides by world_size). Use gradient accumulation
    # to keep the effective residue-patch budget constant across node counts.
    cfg.train.batch_size = 256
    cfg.train.global_batch_size = 512
    cfg.train.grad_accum_steps = 2  # effective global batch = 256 * 2 = 512
    cfg.train.expected_world_size = 8

    cfg.optim.lr = 1.1e-4
    cfg.optim.warmup = 4_000
    cfg.optim.total_steps = 300_000  # a cap; early-stop on the multimodal val score
    cfg.optim.beta2 = 0.95
    cfg.optim.min_lr = 1e-6

    cfg.train.checkpointing.interval.every_steps = 20_000
    cfg.logging.group = "multimodal_lfq18_400m"
    return cfg
