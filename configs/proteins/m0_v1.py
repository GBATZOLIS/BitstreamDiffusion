"""Optimum M0 run (v1): the proven, DPLM-matched 37.7M multimodal recipe plus the
low-risk config-level improvements from the M0 review.

Inherits configs/proteins/multimodal_lfq18_35m.py unchanged (same model, data,
task mix, replay, warm start, self-conditioning, W&B + TensorBoard sync) so the
compute/token budget stays matched to the DPLM-2 Bit comparison, and only tweaks:

  - optim.beta2 0.99 -> 0.95   (faster second-moment adaptation; standard for
                                score/diffusion training)
  - optim.min_lr made explicit (now honoured by the cosine scheduler)
  - W&B identity: display name "m0_v1", a stable run_id so a resumed/relaunched
    run continues the SAME W&B run instead of forking a new one, and its own run
    directory so it does not collide with the earlier multimodal_lfq18_35m_seed0.

Deliberately NOT changed here (they need code, not config, and are tracked as
follow-ups): task-metric-based checkpoint selection, the ABSENT-modality
placeholder fix for the de-novo marginals, and a multimodal VLB. The larger-batch
throughput variant is left off to avoid gambling the proven recipe.

Launch (4 GPUs):
  .venv/bin/torchrun --nproc_per_node=4 train.py --config configs/proteins/m0_v1.py
"""

from configs.proteins.multimodal_lfq18_35m import get_config as _m0


def get_config():
    cfg = _m0()

    # Own run directory (fresh; does not resume the earlier seed0 run).
    cfg.experiment = "proteins/m0_v1"

    # Optimizer tweak: 0.95 second-moment decay for score/diffusion training.
    cfg.optim.beta2 = 0.95
    cfg.optim.min_lr = 1e-6

    # W&B: full display name m0_v1, stable id so resume continues the same run.
    cfg.logging.run_name = "m0_v1"
    cfg.logging.run_id = "m0_v1"
    cfg.logging.group = "m0"
    return cfg
