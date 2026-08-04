"""Small 20-way categorical continuous-diffusion viability baseline (11.65M)."""
from configs.proteins.dima35m_bitstream_viability import get_config as _bitstream


def get_config():
    cfg = _bitstream()
    cfg.experiment = "proteins/dima_small_categorical_viability_seed0"
    cfg.data.representation = "tokens"
    cfg.data.sequence_len = 254
    cfg.data.vocab_size = 20

    cfg.model.patch_size = 1
    cfg.model.embed_dim = 320
    cfg.model.dim_ff = 1280
    cfg.model.n_blocks = 6
    cfg.model.n_heads = 8
    cfg.model.head_type = "token_full"
    cfg.model.out_dim = 20
    cfg.model.continuous_logit_scaling = "none"
    cfg.model.expected_num_parameters = 11_645_776

    cfg.diffusion.continuous.data_center = 1.0 / 20.0
    cfg.train.loss_type = "token_ce"
    cfg.train.token_sm_chunk_size = 2048
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/best.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/protein_eval"
    cfg.evaluation.samples_dir = f"runs/{cfg.experiment}/protein_eval/samples"
    cfg.logging.group = "dima_small_categorical_viability"
    return cfg
