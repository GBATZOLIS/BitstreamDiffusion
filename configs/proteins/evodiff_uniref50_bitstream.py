"""35.8M BitStream training protocol on EvoDiff/DPLM UniRef50."""
from configs.proteins.dima35m_bitstream import get_config as _base_config
from data.uniref50 import (
    EVODIFF_COMMIT,
    EVODIFF_UNIREF50_ALPHABET,
    EVODIFF_UNIREF50_ARCHIVE_MD5,
    DPLM_COMMIT,
)


def get_config():
    cfg = _base_config()
    cfg.experiment = "proteins/evodiff_uniref50_bitstream_seed0"

    cfg.data.dataset = "EvoDiffUniRef50"
    cfg.data.root = "datasets/uniref50_evodiff_2020"
    cfg.data.protocol = "evodiff_uniref50_2020_train_valid_rtest"
    cfg.data.archive_md5 = EVODIFF_UNIREF50_ARCHIVE_MD5
    cfg.data.evodiff_commit = EVODIFF_COMMIT
    cfg.data.dplm_commit = DPLM_COMMIT
    cfg.data.tokenizer = "evodiff26"
    if "hf_revision" in cfg.data:
        del cfg.data.hf_revision
    if "dima_commit" in cfg.data:
        del cfg.data.dima_commit
    for legacy_field in ("fasta_name", "max_windows_per_seq", "val_fraction", "test_fraction"):
        if legacy_field in cfg.data:
            del cfg.data[legacy_field]
    cfg.data.alphabet = EVODIFF_UNIREF50_ALPHABET
    cfg.data.representation = "binary"
    cfg.data.binarization = "raw_binary"
    cfg.data.bits_per_token = 5
    cfg.data.max_len = 1022
    cfg.data.min_len = 1
    cfg.data.sequence_len_tokens = 1022
    cfg.data.sequence_len = 1022 * 5
    cfg.data.vocab_size = 2
    cfg.data.num_workers = 8
    cfg.data.prefetch_factor = 4
    cfg.data.limit_train = 0
    cfg.data.limit_eval = 0

    # A logical epoch is 5k optimizer steps.  This avoids waiting for a full
    # pass over 41M rows before validation and mirrors DPLM's step-based budget.
    cfg.train.batch_size = 16
    cfg.train.global_batch_size = 16
    cfg.train.expected_world_size = 4
    cfg.train.steps_per_epoch = 5_000
    cfg.train.validation_max_batches = 100
    cfg.train.epochs = 20
    cfg.train.vlb.splits = ["val"]
    cfg.train.vlb.max_batches_val = 50
    cfg.train.early_stopping.enabled = True
    cfg.train.early_stopping.warmup_steps = 50_000
    cfg.train.early_stopping.patience_epochs = 8
    cfg.train.early_stopping.min_delta = 1e-4

    cfg.optim.total_steps = 100_000
    cfg.optim.warmup = 10_000
    cfg.optim.lr = 2e-4
    cfg.train.checkpointing.interval.every_steps = 10_000
    cfg.train.checkpointing.resume_interval.every_steps = 2_000

    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/best.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/protein_benchmark"
    cfg.evaluation.samples_dir = f"runs/{cfg.experiment}/protein_benchmark/samples"
    if "length_distribution" in cfg.evaluation:
        del cfg.evaluation.length_distribution
    cfg.evaluation.length_grid = [100, 200, 300, 400, 500]
    cfg.evaluation.samples_per_length = 400
    cfg.evaluation.num_samples = 2_000
    cfg.evaluation.num_sampling_steps = 250

    cfg.logging.group = "evodiff_uniref50_bitstream"
    return cfg
