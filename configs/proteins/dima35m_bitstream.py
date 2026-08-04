"""Publishable whole-sequence BitStream protocol matched to DiMA Swiss-Prot."""
from configs.proteins.swissprot_char5 import get_config as _pilot_config


HF_REVISION = "bf4b2f131664fe87ef5e0fc9e53f01f0d030dcab"
DIMA_COMMIT = "18f4a2e67988efe1cc5593fcaec1ece8f09fdac0"


def get_config():
    cfg = _pilot_config()
    cfg.experiment = "proteins/dima35m_bitstream_seed0"

    cfg.data.dataset = "SwissProtDiMA"
    cfg.data.root = "datasets/swissprot_dima_bf4b2f13"
    cfg.data.tokenizer = "canonical20"
    cfg.data.representation = "binary"
    cfg.data.binarization = "raw_binary"
    cfg.data.hf_revision = HF_REVISION
    cfg.data.dima_commit = DIMA_COMMIT
    cfg.data.protocol = "dima_release_exact_rows_truncate254_no_specials"
    cfg.data.sequence_len_tokens = 254
    cfg.data.bits_per_token = 5
    cfg.data.sequence_len = 254 * 5
    cfg.data.min_len = 128
    cfg.data.max_len = 254
    cfg.data.vocab_size = 2
    cfg.data.num_workers = 8
    cfg.data.prefetch_factor = 4

    # 35.778M trainable parameters with the current SDT implementation.
    cfg.model.patch_size = 5
    cfg.model.embed_dim = 384
    cfg.model.dim_ff = 1536
    cfg.model.n_blocks = 13
    cfg.model.n_heads = 8
    cfg.model.expected_num_parameters = 35_777_729

    # Global batch 256; Trainer assigns 64 examples per rank on four GPUs.
    cfg.train.seed = 42
    cfg.train.batch_size = 256
    cfg.train.global_batch_size = 256
    cfg.train.expected_world_size = 4
    cfg.train.epochs = 10_000
    cfg.train.vlb.splits = ["val"]
    cfg.train.vlb.max_batches_val = 50
    cfg.train.vlb.position_scope = "storage"

    cfg.train.early_stopping = type(cfg.train)()
    cfg.train.early_stopping.enabled = True
    cfg.train.early_stopping.patience_epochs = 8
    cfg.train.early_stopping.min_delta = 1e-4
    cfg.train.early_stopping.warmup_steps = 50_000

    # DiMA release trains for one million iterations.
    cfg.optim.total_steps = 1_000_000
    cfg.optim.warmup = 10_000
    cfg.optim.lr = 2e-4

    cfg.train.checkpointing.interval.every_steps = 50_000
    cfg.train.checkpointing.resume_interval.every_steps = 2_000

    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/best.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/protein_eval"
    cfg.evaluation.samples_dir = f"runs/{cfg.experiment}/protein_eval/samples"
    cfg.evaluation.num_sampling_steps = 250
    cfg.evaluation.num_samples = 2048
    cfg.evaluation.length_distribution = (
        "datasets/swissprot_dima_bf4b2f13/protocol/"
        "generation_length_probabilities.npy"
    )

    cfg.logging.group = "dima35m_bitstream"
    return cfg
