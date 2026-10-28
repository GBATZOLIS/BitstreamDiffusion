"""Training and evaluation settings for the 400M protein paper run."""

from ml_collections import config_dict


def get_config():
    cfg = config_dict.ConfigDict()

    cfg.framework = "continuous_score"
    cfg.experiment = "proteins/paper"
    cfg.device = "cuda:0"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    cfg.data = config_dict.ConfigDict()
    cfg.data.dataset = "ProteinMultimodalLFQ"
    cfg.data.root = "datasets/dplm_paired_m0"
    cfg.data.shard_dir = "datasets/dplm_paired_m0"
    cfg.data.sequence_replay_root = "datasets/uniref50_replay_v2"
    cfg.data.sequence_replay_fraction = 0.2

    cfg.data.representation = "binary"
    cfg.data.binarization = "raw_binary"
    cfg.data.tokenizer = "char"
    cfg.data.vocab_size = 2
    cfg.data.channels = 1
    cfg.data.flatten_order = "flatten"

    cfg.data.bits_per_token = 5
    cfg.data.seq_bits = 5
    cfg.data.struct_bits = 13
    cfg.data.patch_bits = 18
    cfg.data.sequence_len_tokens = 512
    cfg.data.sequence_len = 2560
    cfg.data.min_len = 40
    cfg.data.max_len = 512
    cfg.data.long_chain_policy = "random_crop"
    cfg.data.struct_convention_hash = "71220cdbbb22b4ff"

    cfg.data.num_workers = 8
    cfg.data.prefetch_factor = 4
    cfg.data.pin_memory = True

    cfg.data.source_weights = config_dict.ConfigDict()
    cfg.data.source_weights.afdb_rep = 0.0
    cfg.data.source_weights.afdb_swissprot = 1.0
    cfg.data.source_weights.esmatlas = 0.0
    cfg.data.source_weights.pdb = 2.0

    cfg.data.use_mixture_schedule = True
    cfg.data.task_weights = config_dict.ConfigDict()
    cfg.data.task_weights.forward_folding = 0.2
    cfg.data.task_weights.inverse_folding = 0.2
    cfg.data.task_weights.joint = 0.3
    cfg.data.task_weights.motif = 0.1
    cfg.data.task_weights.partial_pair = 0.04
    cfg.data.task_weights.sequence_inpaint = 0.05
    cfg.data.task_weights.sequence_marginal = 0.1
    cfg.data.task_weights.sequence_redesign = 0.03
    cfg.data.task_weights.structure_inpaint = 0.07
    cfg.data.task_weights.structure_marginal = 0.1
    cfg.data.task_weights.structure_redesign = 0.03

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg.model = config_dict.ConfigDict()
    cfg.model.name = "sdt"
    cfg.model.protein = True
    cfg.model.use_flash_attn = True
    cfg.model.self_condition = True
    cfg.model.center_inputs = True
    cfg.model.patch_size = 18

    cfg.model.embed_dim = 1024
    cfg.model.dim_ff = 4096
    cfg.model.n_blocks = 20
    cfg.model.n_heads = 16

    cfg.model.head_type = "optimal_skip_mlp"
    cfg.model.out_dim = 1
    cfg.model.head_hidden = 224
    cfg.model.content_dim_continuous = 128
    cfg.model.dropout = 0.1

    cfg.model.use_rope_trunk = True
    cfg.model.rope_base = 10_000.0
    cfg.model.abs_pos_mode = "local_only"
    cfg.model.n_fourier_global = 32
    cfg.model.n_fourier_local = 4
    cfg.model.use_adaln = True
    cfg.model.rpb_max_distance = 1
    cfg.model.use_swiglu = True

    cfg.model.continuous_logit_scaling = "none"
    cfg.model.final_norm = True
    cfg.model.qk_norm = True
    cfg.model.asymmetric_modality_time = True
    cfg.model.independent_modality_noise = True
    cfg.model.length_conditioning = True
    cfg.model.residue_pos_features = True
    cfg.model.residue_pos_fraction = True
    cfg.model.lambda_seq = 1.0
    cfg.model.lambda_struct = 1.0

    # ------------------------------------------------------------------
    # Continuous diffusion
    # ------------------------------------------------------------------
    cfg.diffusion = config_dict.ConfigDict()
    cfg.diffusion.continuous = config_dict.ConfigDict()
    cfg.diffusion.continuous.sigma_min = 0.002
    cfg.diffusion.continuous.sigma_max = 80.0
    cfg.diffusion.continuous.rho = 7.0
    cfg.diffusion.continuous.sigma_data = 0.5
    cfg.diffusion.continuous.data_center = 0.5
    cfg.diffusion.continuous.p_mean = -1.2
    cfg.diffusion.continuous.p_std = 1.2

    cfg.cond = config_dict.ConfigDict()
    cfg.cond.p_uncond = 0.1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    cfg.train = config_dict.ConfigDict()
    cfg.train.deterministic = False
    cfg.train.seed = 42
    cfg.train.use_compile = False
    cfg.train.use_fp16 = False
    cfg.train.loss_type = "binary_sm"
    cfg.train.loss_weighting = "edm"
    cfg.train.global_loss_normalizer = True

    cfg.train.batch_size = 256
    cfg.train.epochs = 60
    cfg.train.steps_per_epoch = 5000
    cfg.train.grad_accum_steps = 2
    cfg.train.ema_decay = 0.9999
    cfg.train.sigma_sampling_strategy = "log-normal"
    cfg.train.self_condition_prob = 0.5
    cfg.train.coupled_sigma_prob = 0.3

    cfg.train.checkpointing = config_dict.ConfigDict()
    cfg.train.checkpointing.save_last = True
    cfg.train.checkpointing.save_top_k = 2
    cfg.train.checkpointing.mode = "min"

    cfg.train.checkpointing.interval = config_dict.ConfigDict()
    cfg.train.checkpointing.interval.enabled = True
    cfg.train.checkpointing.interval.every_steps = 20_000
    cfg.train.checkpointing.interval.keep_last = 3

    cfg.train.checkpointing.resume_interval = config_dict.ConfigDict()
    cfg.train.checkpointing.resume_interval.enabled = True
    cfg.train.checkpointing.resume_interval.every_steps = 2_000

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------
    cfg.optim = config_dict.ConfigDict()
    cfg.optim.optimizer = "AdamW"
    cfg.optim.lr = 1.1e-4
    cfg.optim.weight_decay = 0.01
    cfg.optim.beta1 = 0.9
    cfg.optim.beta2 = 0.95
    cfg.optim.eps = 1e-8
    cfg.optim.grad_clip = 1.0
    cfg.optim.scheduler = "cosine_decay"
    cfg.optim.total_steps = 300_000
    cfg.optim.warmup = 10_000
    cfg.optim.min_lr = 1e-6

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    run_dir = f"runs/{cfg.experiment}"
    cfg.evaluation = config_dict.ConfigDict()
    cfg.evaluation.checkpoint_path = f"{run_dir}/checkpoints/best.pt"
    cfg.evaluation.best_checkpoint_path = f"{run_dir}/checkpoints/best.pt"
    cfg.evaluation.out_dir = f"{run_dir}/evaluation"
    cfg.evaluation.samples_dir = f"{run_dir}/evaluation/samples"

    cfg.evaluation.solver = "ddim_entropic"
    cfg.evaluation.schedule = "entropic"
    cfg.evaluation.num_sampling_steps = 50
    cfg.evaluation.guidance_scale = 5.0
    cfg.evaluation.entropic_blend_alpha = 0.0
    cfg.evaluation.sigma_min = 0.002
    cfg.evaluation.sigma_max = 80.0
    cfg.evaluation.struct_sigma_scale = 1.0
    cfg.evaluation.use_amp = False
    cfg.evaluation.use_compile = False

    cfg.evaluation.stochastic = config_dict.ConfigDict()
    cfg.evaluation.stochastic.enabled = True
    cfg.evaluation.stochastic.s_churn = 10.0
    cfg.evaluation.stochastic.s_noise = 1.0
    cfg.evaluation.stochastic.window_mode = "entropy_cdf"
    cfg.evaluation.stochastic.entropy_quantile_lo = 0.1
    cfg.evaluation.stochastic.entropy_quantile_hi = 0.9
    cfg.evaluation.stochastic.entropy_fallback = "error"

    cfg.evaluation.entropy = config_dict.ConfigDict()
    cfg.evaluation.entropy.num_bins = 128
    cfg.evaluation.entropy.samples_per_unit = 8
    cfg.evaluation.entropy.num_backbones = 128
    cfg.evaluation.entropy.seed = 20260827
    cfg.evaluation.entropy.rate_power = 0.5
    cfg.evaluation.entropy.regularizer_c = 0.1
    cfg.evaluation.entropy.regularizer_n = 3.0

    cfg.evaluation.validation_data = "datasets/cath43/validation.npz"
    cfg.evaluation.inverse_data = "datasets/cath43/test.npz"
    cfg.evaluation.forward_targets = (
        "datasets/benchmarks/cameo2022_dplm21/targets.json"
    )
    cfg.evaluation.struct_tokenizer = "datasets/dplm_struct_tokenizer"
    cfg.evaluation.length_grid = [100, 200, 300, 400, 500]
    cfg.evaluation.samples_per_length = 16
    cfg.evaluation.num_samples = 4
    cfg.evaluation.seed = 20260827

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    cfg.logging = config_dict.ConfigDict()
    cfg.logging.use_wandb = False

    cfg.logging.tensorboard = config_dict.ConfigDict()
    cfg.logging.tensorboard.enabled = True
    cfg.logging.tensorboard.log_dir = "auto"
    cfg.logging.tensorboard.scalar_every_steps = 20
    cfg.logging.tensorboard.flush_secs = 30
    cfg.logging.tensorboard.max_queue = 2000
    cfg.logging.tensorboard.sync_to_run_dir = True
    cfg.logging.tensorboard.sync_every_epochs = 1
    cfg.logging.tensorboard.sync_every_steps = 500
    cfg.logging.tensorboard.copy_existing_to_scratch = True
    cfg.logging.tensorboard.fail_silently = True

    return cfg
