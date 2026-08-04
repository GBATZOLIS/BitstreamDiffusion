"""M0 run v3: adopt the proven OWT-medium (462M) *recipe* for the multimodal
protein model -- online entropic sigma scheduling + score-matching with EDM
weighting + an entropic-grid sampler with churn and CFG -- at a larger 130M
(CoBit-S) trunk and a long horizon so the entropic schedule actually engages.

This deliberately departs from the m0_v1/v2 line (a 37.7M, compute/data-matched
point vs DPLM-2 Bit that over-trained the 220k-row corpus). v3 is a recipe/scale
change, mirroring configs/owt/rate_bits_edm_weight.py as closely as the protein
multimodal path allows:

  RECIPE (mirrors the OWT medium config)
  - loss: score matching with EDM weighting -> loss_type="binary_sm",
    loss_weighting="edm". This is the *training loss*; it is separate from the
    distribution the sigmas are DRAWN from (below).
  - sigma sampling: log-normal EDM base (p_mean=-1.2, p_std=1.2), then the online
    ADAPTIVE ENTROPIC schedule takes over -- 40k-step warmup on the EDM base, a
    10k-step transition, then fully entropic (gamma_max=1.0); the schedule is
    refreshed from the FIFO rate buffer every 2k steps. Values copied verbatim
    from the OWT medium config's entropy block. The multimodal step feeds the
    buffer with both modalities' (sigma, denoising-MSE) samples and follows the
    schedule for BOTH sequence and structure noise levels.
  - sampler (eval): the entropic schedule builds the stratified inverse-CDF
    integration grid for both the deterministic (Heun) and the stochastic (EDM
    churn) solver, with CFG available on the conditional tasks. See the eval
    block at the bottom.

  SCALE / BUDGET
  - ~130M (CoBit-S) trunk: 640/2560, 16 blocks, 10 heads (matches
    multimodal_lfq18_130m's trunk exactly).
  - 250k optimizer steps, no early stopping: the 40k+10k entropy schedule needs
    a long tail to matter, so ~200k steps run fully entropic. best.pt (val-min),
    not last.pt, is the checkpoint to evaluate.
  - no warm start: there is no 640-dim-width UniRef50 sequence checkpoint (the
    warm-start column surgery only matches the 384-dim 35M trunk), so the trunk
    trains from scratch and the sequence side is fed by the raised replay
    fraction -- the same choice the 400M config makes at 1024 width.
  - kept on the M0 DPLM-paired corpus (datasets/dplm_paired_m0), NOT the M1
    corpus the 130m config points at, and on 4 GPUs.

  REGULARIZATION (a 125M model on 220k paired rows over 250k steps will overfit)
  - weight_decay 0.05, dropout 0.2, sequence replay 0.40, deterministic + wider
    validation so best.pt tracks real generalization.

Launch (4 GPUs):
  scripts/launch/train_4gpu.sh configs/proteins/m0_v3.py
"""

from configs.proteins.multimodal_lfq18_35m import get_config as _m0


def get_config():
    cfg = _m0()

    # Own fresh run directory (does not resume m0_v1/v2).
    cfg.experiment = "proteins/m0_v3"

    # ------------------------------------------------------------------
    # Model: ~130M (CoBit-S) trunk. Copies multimodal_lfq18_130m's trunk dims
    # verbatim, but keeps the M0 data path and 4-GPU batch of the 35M config.
    # ------------------------------------------------------------------
    cfg.model.embed_dim = 640
    cfg.model.dim_ff = 2560
    cfg.model.n_blocks = 16
    cfg.model.n_heads = 10
    cfg.model.content_dim_continuous = 96
    cfg.model.head_hidden = 160
    cfg.model.expected_num_parameters = 125_190_000
    cfg.model.gradient_checkpointing = True  # documents intent (no-op in trunk today)
    cfg.model.self_condition = True
    cfg.model.dropout = 0.2  # v1 used 0.1; regularize the larger trunk on M0 data

    # No width-matched (640-dim) UniRef50 sequence checkpoint -> train from
    # scratch; the raised replay fraction feeds the sequence marginal.
    cfg.model.warm_start_seq_checkpoint = ""
    cfg.model.warm_start_freeze_trunk_steps = 0
    cfg.model.warm_start_trunk_lr_mult = 1.0

    # ------------------------------------------------------------------
    # Data: keep the M0 DPLM-paired corpus; heavier sequence replay to dilute
    # the scarce paired-structure repetition at 125M.
    # ------------------------------------------------------------------
    cfg.data.sequence_replay_fraction = 0.40  # v1 used 0.25

    # ------------------------------------------------------------------
    # Loss: score matching + EDM weighting (mirrors the OWT medium config).
    # This is separate from the entropic distribution used to DRAW sigmas.
    # ------------------------------------------------------------------
    cfg.train.loss_type = "binary_sm"
    cfg.train.loss_weighting = "edm"
    cfg.train.sigma_sampling_strategy = "log-normal"  # EDM base under the entropy schedule

    # ------------------------------------------------------------------
    # Adaptive entropy-based sigma sampling (verbatim from the OWT medium config)
    # ------------------------------------------------------------------
    cfg.train.entropy_offline.enabled = False
    cfg.train.entropy_compute = True
    cfg.train.entropy_use_for_sampling = True
    cfg.train.entropy_buffer_size = 800_000
    cfg.train.entropy_num_bins = 128
    cfg.train.entropy_min_per_bin = 100
    cfg.train.entropy_update_every_steps = 2_000    # refresh the schedule every 2k steps
    cfg.train.entropy_warmup_steps = 40_000         # EDM base for the first 40k steps
    cfg.train.entropy_transition_steps = 10_000     # then blend EDM -> entropic over 10k
    cfg.train.entropy_gamma_max = 1.0               # then fully entropic
    cfg.train.entropy_mode = "regularized"
    cfg.train.entropy_regularizer_c = 0.1
    cfg.train.entropy_regularizer_n = 3.0
    cfg.train.entropy_target = "sqrt-rate"
    cfg.train.entropy_plot_every_k_epochs = 5

    # ------------------------------------------------------------------
    # Budget: long horizon, no early stopping (250k optimizer steps).
    # 5_000 micro-batches/epoch and grad_accum=2 -> 2_500 optim steps/epoch;
    # epochs is a safe cap above the step budget so total_steps binds first.
    # ------------------------------------------------------------------
    cfg.train.batch_size = 128          # GLOBAL per optim micro-step -> 32/GPU on 4 GPUs
    cfg.train.global_batch_size = 256   # effective, via grad accumulation
    cfg.train.grad_accum_steps = 2      # 128 * 2 = 256 effective global batch
    cfg.train.expected_world_size = 4
    cfg.train.steps_per_epoch = 5_000
    cfg.train.epochs = 110

    cfg.optim.total_steps = 250_000
    cfg.optim.warmup = 10_000
    cfg.optim.lr = 1.5e-4
    cfg.optim.beta2 = 0.95
    cfg.optim.min_lr = 1e-6
    cfg.optim.weight_decay = 0.05       # v1 used 0.01

    # Reliable validation so best.pt / the val curve track generalization.
    cfg.train.deterministic_validation = True
    cfg.train.val_seed = 1234
    cfg.train.validation_max_batches = 300

    # Archival cadence over the long run (+ best.pt + last.pt).
    cfg.train.checkpointing.save_top_k = 2
    cfg.train.checkpointing.interval.every_steps = 25_000   # 25/50/.../250k
    cfg.train.checkpointing.resume_interval.every_steps = 2_000

    # ------------------------------------------------------------------
    # Eval sampler: entropic stratified grid for BOTH the deterministic (Heun)
    # and the stochastic (EDM churn) solver, plus CFG on the conditional tasks.
    # These are the knobs to sweep at eval; churn/CFG are off by default so the
    # baseline deterministic entropic-grid run is unchanged.
    #   - schedule="entropic": SigmaSchedule builds the grid by stratified
    #     inverse-CDF sampling of the learned entropy CDF (entropy_*.pt in the
    #     run dir, written during training). entropic_blend_alpha in [0,1] mixes
    #     with the Karras grid (0 = pure entropic).
    #   - stochastic.*: EDM Algorithm-2 churn. Sweep s_churn (gamma budget), e.g.
    #     {10, 40, 80}; window_mode="entropy_cdf" ties the churn [s_tmin,s_tmax]
    #     window to the entropy-CDF quantiles.
    #   - guidance_scale: classifier-free guidance weight for the conditional
    #     tasks (forward/inverse folding, motif). Sweep >0 to sharpen; 0 = off.
    # ------------------------------------------------------------------
    # This is a NEW run that owns both its training and its evaluation: the base
    # config froze out_dir/samples_dir to the smoke experiment's f-string, so
    # re-point every eval output at the m0_v3 run dir (else eval would write into
    # runs/proteins/multimodal_lfq18_smoke/...). Evaluate best.pt, not last.pt.
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/best.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/protein_eval"
    cfg.evaluation.samples_dir = f"runs/{cfg.experiment}/protein_eval/samples"
    cfg.evaluation.schedule = "entropic"
    cfg.evaluation.entropic_blend_alpha = 0.0
    cfg.evaluation.guidance_scale = 0.0

    cfg.evaluation.stochastic = type(cfg.train)()
    cfg.evaluation.stochastic.enabled = False
    cfg.evaluation.stochastic.s_churn = 0.0
    cfg.evaluation.stochastic.s_noise = 1.0
    cfg.evaluation.stochastic.window_mode = "entropy_cdf"
    cfg.evaluation.stochastic.s_tmin = 0.0
    cfg.evaluation.stochastic.s_tmax = 1.0e9  # inf-like; overridden when window_mode="entropy_cdf"

    # ------------------------------------------------------------------
    # W&B: distinct run, same M0 project/group.
    # ------------------------------------------------------------------
    cfg.logging.run_name = "m0_v3"
    cfg.logging.run_id = "m0_v3"
    cfg.logging.group = "m0"
    return cfg
