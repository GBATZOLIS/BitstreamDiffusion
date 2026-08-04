"""Multimodal 18-bit foundation model, flagship scale ~650M (plan: full run).

The GATED flagship (see .trentinium/docs/full_run_and_evaluation.md): ~650M matches
DPLM-2's largest multimodal model on the *same* LFQ structure tokens, giving the
clean same-capacity comparison against DPLM-2 / DPLM-2 Bit. Promote here only once
(a) M1/M2 are built and (b) the seven-task recipe validates at 400M. Fits plain DDP
(measured ~42.4 GB/GPU at B=32, 512 residues) -- no FSDP. Inherits the 400M config.
"""

from configs.proteins.multimodal_lfq18_400m import get_config as _first_run


def get_config():
    cfg = _first_run()
    cfg.experiment = "proteins/multimodal_lfq18_650m_seed0"

    # ~640M trunk (measured from models/sdt.py). 26 layers deep for the conditionals.
    cfg.model.embed_dim = 1152
    cfg.model.dim_ff = 4608
    cfg.model.n_blocks = 26
    cfg.model.n_heads = 16
    cfg.model.content_dim_continuous = 128
    cfg.model.head_hidden = 256
    cfg.model.expected_num_parameters = 640_000_000

    # Slightly lower LR at the larger width; smaller local batch to hold ~42GB/GPU,
    # recovered to the same effective global batch via more accumulation.
    cfg.optim.lr = 1.0e-4
    cfg.train.batch_size = 128
    cfg.train.global_batch_size = 512
    cfg.train.grad_accum_steps = 4  # 128 * 4 = 512 effective global batch

    cfg.logging.group = "multimodal_lfq18_650m"
    return cfg
