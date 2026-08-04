"""M0 run v2: fix the over-training / high-and-unreadable validation loss seen in
m0_v1, without changing model scale, data corpus, or the proven noise/loss recipe.

Diagnosis of m0_v1 (from runs/proteins/m0_v1/training_logs):
  - train loss falls smoothly to ~0.47, but validation loss bottoms at ~1.39 at
    step ~20k (best.pt) and then rises and oscillates (3-13) through 200k. That is
    over-training: 200k steps x 256 global batch = 51.2M examples over a 220,475-row
    paired corpus (~170-230 passes) with a 37.7M model.
  - the headline eval was run on last.pt (step 200k), i.e. the most over-trained
    checkpoint, not best.pt.
  - the val metric is a random-sigma / random-noise denoising loss over only 100
    batches, so it is a high-variance estimator (part of the "high" number is noise).

v2 keeps the m0_v1 recipe (same 37.7M model, DPLM-paired data, task mix, warm start,
self-conditioning, EDM sigma schedule, lambda_seq/lambda_struct, trunk curriculum,
W&B + TensorBoard) and only applies the review's training fixes, so the improvement
is attributable to these knobs alone and stays compute-comparable to the DPLM-2 Bit
point:

  BUDGET / EARLY STOP
  - total_steps 200k -> 50k, warmup 10k -> 2k (val bottomed ~20k; 200k over-trains).
  - finer validation cadence (2.5k steps/epoch) so the minimum is caught.
  - early stopping on validation loss (patience 4 epochs after a 10k-step warmup).

  RELIABLE VALIDATION (so early stop / best.pt act on real generalization)
  - deterministic_validation: fix the sigma/noise stream across epochs.
  - validation_max_batches 100 -> 300 to further cut estimator variance.

  REGULARIZATION (220k rows is small for this many passes)
  - optim.weight_decay 0.01 -> 0.05, model.dropout 0.1 -> 0.2.
  - UniRef50 sequence replay 0.25 -> 0.40 (more diverse data, less memorization,
    protects sequence fluency).

Deliberately NOT changed (would confound the comparison or gamble the recipe):
model width/depth, the M0 corpus, the EDM p_mean/p_std/sigma_data, lambda_seq/
lambda_struct, task_weights, warm-start freeze / trunk LR mult. Follow-ups tracked
elsewhere: task-metric-based checkpoint selection, the ABSENT-modality placeholder
fix, and a multimodal VLB.

Reminder: evaluate this run with best.pt (val-min), not last.pt.

Launch (4 GPUs):
  scripts/launch/train_4gpu.sh configs/proteins/m0_v2.py
"""

from configs.proteins.m0_v1 import get_config as _m0_v1


def get_config():
    cfg = _m0_v1()

    # Own fresh run directory (does not resume m0_v1).
    cfg.experiment = "proteins/m0_v2"

    # ---- budget: stop over-training -------------------------------------------
    cfg.train.steps_per_epoch = 2_500      # finer val cadence than v1's 5_000
    cfg.train.epochs = 20                  # 2_500 x 20 = 50_000 steps
    cfg.optim.total_steps = 50_000         # cosine horizon matches the loop length
    cfg.optim.warmup = 2_000               # v1 used 10_000 over a 200k horizon

    # ---- early stopping on validation loss ------------------------------------
    # (Trainer already supports cfg.train.early_stopping; v1 left it unset.)
    cfg.train.early_stopping = type(cfg.train)()
    cfg.train.early_stopping.enabled = True
    cfg.train.early_stopping.warmup_steps = 10_000   # never stop before 10k steps
    cfg.train.early_stopping.patience_epochs = 4     # 4 non-improving val epochs (~10k steps)
    cfg.train.early_stopping.min_delta = 5e-3        # ignore sub-0.005 val wiggles

    # ---- reliable validation (makes early stop / best.pt trustworthy) ---------
    cfg.train.deterministic_validation = True        # fix sigma/noise stream across epochs
    cfg.train.val_seed = 1234
    cfg.train.validation_max_batches = 300           # v1 used 100 -> lower-variance estimate

    # ---- regularization against overfitting the 220k-row corpus ---------------
    cfg.optim.weight_decay = 0.05                    # v1 used 0.01
    cfg.model.dropout = 0.2                          # v1 used 0.1
    cfg.data.sequence_replay_fraction = 0.40         # v1 used 0.25

    # ---- checkpointing: finer archival cadence for the shorter run ------------
    cfg.train.checkpointing.interval.every_steps = 10_000   # 10/20/30/40/50k + best.pt

    # ---- W&B: distinct run, same project/group as v1 --------------------------
    cfg.logging.run_name = "m0_v2"
    cfg.logging.run_id = "m0_v2"
    cfg.logging.group = "m0"
    return cfg
