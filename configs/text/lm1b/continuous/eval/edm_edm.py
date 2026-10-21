from ml_collections import config_dict


# LM1B unconditional continuous EDM baseline on raw-binary bitstreams.
#
# Evaluation-only config:
#   - checkpoint: 900K only
#   - metric: external PPL
#   - sampler: ddim_entropic
#   - sc_refresh_mode: carry
#   - target NFE: 512
#   - expected sampler steps: 128
#   - ATI eta: 0
#   - stochastic churn: disabled
#   - terminal_sigma: 0.08
#   - score_mode: full


def _ckpts_900k_only():
    return [
        "step=000900000.pt",
    ]


def get_config():
    cfg = config_dict.ConfigDict()

    # ------------------------------------------------------------------
    # Framework / experiment
    # ------------------------------------------------------------------
    cfg.framework = "continuous_score"
    cfg.experiment = "paper/unconditional_text/lm1b/continuous_edm_raw_binary_bits_1M"
    cfg.device = "cuda"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    cfg.data = config_dict.ConfigDict()
    cfg.data.dataset = "LM1B"
    cfg.data.root = "datasets/lm1b"
    cfg.data.tokenizer_name = "bert-base-uncased"

    cfg.data.representation = "binary"
    cfg.data.binarization = "raw_binary"
    cfg.data.token_space = "tokenizer_id"

    cfg.data.sequence_len_tokens = 128
    cfg.data.bits_per_token = 15
    cfg.data.sequence_len = 128 * 15

    cfg.data.val_fraction = 0.005
    cfg.data.vocab_size = 2
    cfg.data.channels = 1
    cfg.data.flatten_order = "flatten"

    cfg.data.num_workers = 12
    cfg.data.prefetch_factor = 4
    cfg.data.pin_memory = True

    # ------------------------------------------------------------------
    # Unconditional benchmark setting
    # ------------------------------------------------------------------
    cfg.cond = config_dict.ConfigDict()
    cfg.cond.enabled = False
    cfg.cond.sample_prompt_len = False
    cfg.cond.cond_len_tokens = 0
    cfg.cond.cond_len_chars = 0
    cfg.cond.p_uncond = 1.0
    cfg.cond.noise_prefix = True
    cfg.cond.loss_on_suffix_only = False
    cfg.cond.null_strategy = "half"

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg.model = config_dict.ConfigDict()
    cfg.model.name = "sdt"
    cfg.model.use_flash_attn = True
    cfg.model.self_condition = True
    cfg.model.center_inputs = True
    cfg.model.patch_size = 15

    cfg.model.embed_dim = 768
    cfg.model.dim_ff = 3072
    cfg.model.n_blocks = 12
    cfg.model.n_heads = 12

    cfg.model.head_type = "optimal_skip_mlp"
    cfg.model.out_dim = 1
    cfg.model.head_hidden = 128
    cfg.model.head_embed_dim = 64

    cfg.model.n_pos_features = 1
    cfg.model.dropout = 0.1
    cfg.model.content_dim_discrete = 64
    cfg.model.content_dim_continuous = 64

    cfg.model.head_use_cross_attn = True
    cfg.model.head_use_local_mixer = True
    cfg.model.head_use_self_attn = False
    cfg.model.head_variant = "single"
    cfg.model.head_kernel = 3
    cfg.model.head_dilation = 1

    cfg.model.use_rope_trunk = True
    cfg.model.rope_base = 10_000.0
    cfg.model.abs_pos_mode = "local_only"
    cfg.model.n_fourier_global = 32
    cfg.model.n_fourier_local = 4
    cfg.model.use_adaln = True
    cfg.model.rpb_max_distance = 1
    cfg.model.use_swiglu = True
    cfg.model.scale_by_sigma = False

    cfg.model.continuous_logit_scaling = "matched_filter_residual"
    cfg.model.matched_filter_center = 0.5
    cfg.model.matched_filter_scale = 1.0
    cfg.model.matched_filter_clip = 30.0

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

    # ------------------------------------------------------------------
    # Training metadata / compatibility
    # ------------------------------------------------------------------
    cfg.train = config_dict.ConfigDict()
    cfg.train.deterministic = False
    cfg.train.seed = 42

    cfg.train.use_compile = True
    cfg.train.compile_mode = "default"

    cfg.train.use_fp16 = True
    cfg.train.amp_dtype = "bf16"
    cfg.train.allow_tf32 = True
    cfg.train.loss_type = "binary_sm"
    cfg.train.loss_weighting = "edm"
    cfg.train.batch_size = 512
    cfg.train.epochs = 74
    cfg.train.ema_decay = 0.9999
    cfg.train.sigma_sampling_strategy = "log-normal"
    cfg.train.self_condition_prob = 0.5

    cfg.train.entropy_offline = config_dict.ConfigDict()
    cfg.train.entropy_offline.enabled = False
    cfg.train.entropy_compute = True
    cfg.train.entropy_use_for_sampling = False
    cfg.train.entropy_buffer_size = 500_000
    cfg.train.entropy_num_bins = 128
    cfg.train.entropy_min_per_bin = 200
    cfg.train.entropy_update_every_steps = 2000
    cfg.train.entropy_warmup_steps = 10_000
    cfg.train.entropy_transition_steps = 5_000
    cfg.train.entropy_gamma_max = 1.0
    cfg.train.entropy_mode = "regularized"
    cfg.train.entropy_regularizer_c = 0.1
    cfg.train.entropy_regularizer_n = 3.0
    cfg.train.entropy_target = "rate"
    cfg.train.entropy_plot_every_k_epochs = 5

    cfg.train.checkpointing = config_dict.ConfigDict()
    cfg.train.checkpointing.save_last = True
    cfg.train.checkpointing.save_top_k = 2
    cfg.train.checkpointing.mode = "min"

    cfg.train.checkpointing.interval = config_dict.ConfigDict()
    cfg.train.checkpointing.interval.enabled = True
    cfg.train.checkpointing.interval.every_steps = 50_000
    cfg.train.checkpointing.interval.keep_last = 0

    cfg.train.checkpointing.resume_interval = config_dict.ConfigDict()
    cfg.train.checkpointing.resume_interval.enabled = True
    cfg.train.checkpointing.resume_interval.every_steps = 5_000

    cfg.train.sanity = config_dict.ConfigDict()
    cfg.train.sanity.enabled = False
    cfg.train.sanity.run_epoch = -1

    cfg.train.generation = config_dict.ConfigDict()
    cfg.train.generation.enabled = False
    cfg.train.generation.splits = ["val"]
    cfg.train.generation.every_epochs = 4
    cfg.train.generation.num_samples = 64
    cfg.train.generation.num_sampling_steps = 128
    cfg.train.generation.samplers = ["ddim_entropic"]
    cfg.train.generation.terminal_sigmas = [0.08]
    cfg.train.generation.entropic_blend_alpha = 0.0
    cfg.train.generation.entropy_ckpt_path = None
    cfg.train.generation.guidance_scales = [0.0]
    cfg.train.generation.micro_batch_size = 64
    cfg.train.generation.sc_refresh_mode = "carry"
    cfg.train.generation.sigma_max = None

    cfg.train.external_ppl = config_dict.ConfigDict()
    cfg.train.external_ppl.enabled = False

    cfg.train.mauve = config_dict.ConfigDict()
    cfg.train.mauve.enabled = False
    cfg.train.mauve.every_k_epochs = 4
    cfg.train.mauve.splits = ["val"]
    cfg.train.mauve.num_samples = 4096
    cfg.train.mauve.featurizer_name = "gpt2-large"
    cfg.train.mauve.max_tokens = cfg.data.sequence_len_tokens
    cfg.train.mauve.device_id = 0
    cfg.train.mauve.micro_batch_size = 512
    cfg.train.mauve.samplers = ["ddim_entropic"]
    cfg.train.mauve.terminal_sigmas = [0.08]
    cfg.train.mauve.guidance_scales = [0.0]
    cfg.train.mauve.num_sampling_steps = 128
    cfg.train.mauve.sigma_max = None
    cfg.train.mauve.sc_refresh_mode = "carry"

    cfg.train.visualization = config_dict.ConfigDict()
    cfg.train.visualization.enabled = False
    cfg.train.visualization.every_k_epochs = 4
    cfg.train.visualization.splits = ["val"]
    cfg.train.visualization.num_samples = 16
    cfg.train.visualization.save_txt = True
    cfg.train.visualization.save_jsonl = True
    cfg.train.visualization.show_prefix_suffix = True
    cfg.train.visualization.micro_batch_size = 16
    cfg.train.visualization.samplers = ["ddim_entropic"]
    cfg.train.visualization.terminal_sigmas = [0.08]
    cfg.train.visualization.guidance_scales = [0.0]
    cfg.train.visualization.num_sampling_steps = 128
    cfg.train.visualization.sigma_max = None
    cfg.train.visualization.sc_refresh_mode = "carry"

    cfg.train.vlb = config_dict.ConfigDict()
    cfg.train.vlb.enabled = False
    cfg.train.vlb.every_k_epochs = 1
    cfg.train.vlb.batch_size = 64
    cfg.train.vlb.sigma_sampling = "log-uniform"
    cfg.train.vlb.sigma_min_eval = 0.08
    cfg.train.vlb.sigma_max_eval = None
    cfg.train.vlb.num_mc_samples_per_batch = 1
    cfg.train.vlb.include_prior = False
    cfg.train.vlb.use_amp = True
    cfg.train.vlb.splits = ["val"]
    cfg.train.vlb.max_batches_train = None
    cfg.train.vlb.max_batches_val = None
    cfg.train.vlb.progress = False
    cfg.train.vlb.allow_conditional_clean_prefix = True
    cfg.train.vlb.force_unconditional_path = False
    cfg.train.vlb.debug_integrand = False
    cfg.train.vlb.debug_first_n_batches = 1
    cfg.train.vlb.debug_num_sigma_bins = 6
    cfg.train.vlb.debug_compare_null_prefix = True
    cfg.train.vlb.debug_compare_noise_prefix = True
    cfg.train.vlb.null_prefix_value = 0.0
    cfg.train.vlb.null_prefix_mode = "constant"

    # ------------------------------------------------------------------
    # Optimizer / scheduler metadata
    # ------------------------------------------------------------------
    cfg.optim = config_dict.ConfigDict()
    cfg.optim.optimizer = "AdamW"
    cfg.optim.lr = 3e-4
    cfg.optim.weight_decay = 0.01
    cfg.optim.beta1 = 0.9
    cfg.optim.beta2 = 0.99
    cfg.optim.eps = 1e-8
    cfg.optim.grad_clip = 1.0
    cfg.optim.scheduler = "cosine_decay"
    cfg.optim.total_steps = 1_000_000
    cfg.optim.warmup = 2_500

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    cfg.evaluation = config_dict.ConfigDict()

    cfg.evaluation.checkpoint_path = (
        f"runs/{cfg.experiment}/checkpoints/step=000900000.pt"
    )

    cfg.evaluation.out_dir = (
        f"runs/{cfg.experiment}/"
        f"external_ppl_ckpt900k_ddim_entropic_carry_128steps_nfe512_full_det"
    )
    cfg.evaluation.samples_dir = f"{cfg.evaluation.out_dir}/samples"
    cfg.evaluation.results_csv = f"{cfg.evaluation.out_dir}/results.csv"
    cfg.evaluation.shared_text_cache_dir = f"{cfg.evaluation.out_dir}/shared_text_cache"

    cfg.evaluation.use_amp = True
    cfg.evaluation.amp_dtype = "bf16"

    # Safer for evaluation-only runs and checkpoint loading.
    cfg.evaluation.use_compile = False
    cfg.evaluation.compile_mode = "default"

    cfg.evaluation.compile = config_dict.ConfigDict()
    cfg.evaluation.compile.warmup = False
    cfg.evaluation.compile.warmup_steps = 0

    # Compatibility/fallback. The sampling sweep below controls the actual config.
    cfg.evaluation.num_sampling_steps = 128

    # ------------------------------------------------------------------
    # ATI: disabled, fully deterministic
    # ------------------------------------------------------------------
    cfg.evaluation.ati = config_dict.ConfigDict()
    cfg.evaluation.ati.enabled = False
    cfg.evaluation.ati.eta = 0.0

    # ------------------------------------------------------------------
    # Stochastic churn: disabled, fully deterministic
    # ------------------------------------------------------------------
    cfg.evaluation.stochastic = config_dict.ConfigDict()
    cfg.evaluation.stochastic.enabled = False
    cfg.evaluation.stochastic.s_churn = 0.0
    cfg.evaluation.stochastic.s_noise = 1.0
    cfg.evaluation.stochastic.window_mode = "entropy_cdf"
    cfg.evaluation.stochastic.entropy_quantile_lo = 0.10
    cfg.evaluation.stochastic.entropy_quantile_hi = 0.90
    cfg.evaluation.stochastic.s_tmin = None
    cfg.evaluation.stochastic.s_tmax = None
    cfg.evaluation.stochastic.entropy_fallback = "deterministic"

    # ------------------------------------------------------------------
    # Sampling sweep used by external PPL and entropy cache matching
    # ------------------------------------------------------------------
    cfg.evaluation.sampling_sweep = config_dict.ConfigDict()
    cfg.evaluation.sampling_sweep.enabled = True
    cfg.evaluation.sampling_sweep.target_nfes = [512]
    cfg.evaluation.sampling_sweep.specs = [
        config_dict.ConfigDict(
            {
                "sampler_name": "ddim_entropic",
                "sc_refresh_modes": ["carry"],
                "target_nfes": [512],
                "ati_etas": [0.0],
                "stochastic_enabled": False,
            }
        )
    ]

    # ------------------------------------------------------------------
    # MAUVE disabled here, but kept compatible with same cache layout
    # ------------------------------------------------------------------
    cfg.evaluation.mauve = config_dict.ConfigDict()
    cfg.evaluation.mauve.enabled = False
    cfg.evaluation.mauve.num_samples = 0
    cfg.evaluation.mauve.micro_batch_size = 512
    cfg.evaluation.mauve.featurizer_name = "gpt2-large"
    cfg.evaluation.mauve.max_tokens = cfg.data.sequence_len_tokens
    cfg.evaluation.mauve.samplers = ["ddim_entropic"]
    cfg.evaluation.mauve.terminal_sigmas = [0.08]
    cfg.evaluation.mauve.guidance_scales = [0.0]
    cfg.evaluation.mauve.num_sampling_steps = 128
    cfg.evaluation.mauve.sigma_max = None
    cfg.evaluation.mauve.shared_cache_dir = cfg.evaluation.shared_text_cache_dir
    cfg.evaluation.mauve.compute_full_text = True
    cfg.evaluation.mauve.compute_suffix_text = True
    cfg.evaluation.mauve.compute_repetition = True
    cfg.evaluation.mauve.seed = 42
    cfg.evaluation.mauve.checkpoints = _ckpts_900k_only()
    cfg.evaluation.mauve.sc_refresh_mode = "carry"

    # ------------------------------------------------------------------
    # External PPL
    # ------------------------------------------------------------------
    cfg.evaluation.external_ppl = config_dict.ConfigDict()
    cfg.evaluation.external_ppl.enabled = True
    cfg.evaluation.external_ppl.backend = "hf_causal_lm"
    cfg.evaluation.external_ppl.hf_model_name = "openai-community/gpt2-large"
    cfg.evaluation.external_ppl.hf_revision = None
    cfg.evaluation.external_ppl.hf_dtype = "bfloat16"
    cfg.evaluation.external_ppl.attn_implementation = "sdpa"

    cfg.evaluation.external_ppl.num_samples = 1024
    cfg.evaluation.external_ppl.micro_batch_size = 512

    cfg.evaluation.external_ppl.samplers = ["ddim_entropic"]
    cfg.evaluation.external_ppl.terminal_sigmas = [0.08]
    cfg.evaluation.external_ppl.guidance_scales = [0.0]
    cfg.evaluation.external_ppl.num_sampling_steps = 128
    cfg.evaluation.external_ppl.sigma_max = None
    cfg.evaluation.external_ppl.sc_refresh_mode = "carry"

    cfg.evaluation.external_ppl.score_mode = "full"
    cfg.evaluation.external_ppl.compute_real_reference = True

    cfg.evaluation.external_ppl.shared_cache_dir = cfg.evaluation.shared_text_cache_dir
    cfg.evaluation.external_ppl.seed = 42
    cfg.evaluation.external_ppl.checkpoints = _ckpts_900k_only()

    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_decode = False
    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_once = True
    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_max_rows = 8

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    cfg.logging = config_dict.ConfigDict()
    cfg.logging.use_wandb = False
    cfg.logging.entity = "continuousDLMs"
    cfg.logging.project = "lm1b"
    cfg.logging.group = "lm1b_continuous_raw_binary_bits_trunk768"
    cfg.logging.mode = "offline"
    cfg.logging.watch_model = False
    cfg.logging.log_freq = 10
    cfg.logging.run_id = None

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