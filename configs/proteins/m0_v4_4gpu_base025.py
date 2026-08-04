"""M0 v4 base025 on one four-GPU DDP job.

This keeps the base025 model, data, optimizer, entropy schedule, and effective
global batch unchanged.  The global micro-batch of 128 is divided across four
GPUs (32 per GPU), then accumulated twice for an effective batch of 256.
"""

from configs.proteins.m0_v4 import build_config


def get_config():
    cfg = build_config(0.25, "025")
    cfg.experiment = "proteins/m0_v4_4GPU_base025"
    cfg.train.expected_world_size = 4

    cfg.evaluation.checkpoint_path = (
        f"runs/{cfg.experiment}/checkpoints/best.pt"
    )
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/protein_eval"
    cfg.evaluation.samples_dir = (
        f"runs/{cfg.experiment}/protein_eval/samples"
    )

    cfg.logging.run_name = "m0_v4_4GPU_base025"
    cfg.logging.run_id = "m0_v4_4GPU_base025"
    return cfg
