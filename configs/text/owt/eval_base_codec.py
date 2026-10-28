from ml_collections import config_dict


def get_config():
    cfg = config_dict.ConfigDict()

    # ------------------------------------------------------------------
    # Framework / experiment
    # ------------------------------------------------------------------
    cfg.framework = "continuous_score"
    cfg.experiment = "paper/unconditional_text/owt/base_codec_continuous_rate_raw_binary_bits_1M"
    cfg.device = "cuda"

    # ------------------------------------------------------------------
    # Data - Direct 16-Bit Raw Binary
    # ------------------------------------------------------------------
    cfg.data = config_dict.ConfigDict()
    cfg.data.dataset = "OpenWebText"
    cfg.data.root = "datasets/openwebtext_gpt2_trainm100k"
    cfg.data.tokenizer_name = "gpt2"

    # Single-stage codec: direct GPT-2 tokenizer-id bits, no BPE16 code tokenizer.
    cfg.data.sequence_codec = "base"

    # Kept for compatibility with shared dataset/code paths; not used by sequence_codec="base".
    cfg.data.code_tokenizer_path = "tokenizer_gpt2id_bpe16_65536_base1024.json"
    cfg.data.code_tokenizer_meta_path = "tokenizer_gpt2id_bpe16_65536_base1024.meta.json"
    cfg.data.base_sequence_len_tokens = 1024
    cfg.data.code_cache_batch_size = 2048
    cfg.data.code_cache_overwrite = False

    cfg.data.representation = "binary"
    cfg.data.binarization = "raw_binary"
    cfg.data.token_space = "tokenizer_id"

    cfg.data.sequence_len_tokens = 1024
    cfg.data.bits_per_token = 16
    cfg.data.sequence_len = 1024 * 16

    cfg.data.wrap = True
    cfg.data.train_split = "train[:-100000]"
    cfg.data.valid_split = "train[-100000:]"
    cfg.data.insert_train_eos = True
    cfg.data.insert_valid_eos = True

    cfg.data.cache_encode_batch_size = 1000
    cfg.data.cache_write_batch_size = 8192
    cfg.data.cache_num_proc = 8
    cfg.data.cache_overwrite = False

    cfg.data.vocab_size = 2
    cfg.data.channels = 1
    cfg.data.flatten_order = "flatten"

    cfg.data.num_workers = 12
    cfg.data.prefetch_factor = 4
    cfg.data.pin_memory = True

    # ------------------------------------------------------------------
    # Conditioning
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

    cfg.model.patch_size = 16
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
    # Train fields needed by eval/model construction
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

    cfg.train.epochs = 100
    cfg.train.ema_decay = 0.9999
    cfg.train.sigma_sampling_strategy = "log-normal"
    cfg.train.self_condition_prob = 0.5

    # Entropy-training metadata, kept consistent with training config.
    cfg.train.entropy_offline = config_dict.ConfigDict()
    cfg.train.entropy_offline.enabled = False
    cfg.train.entropy_compute = True
    cfg.train.entropy_use_for_sampling = True
    cfg.train.entropy_buffer_size = 800_000
    cfg.train.entropy_num_bins = 128
    cfg.train.entropy_min_per_bin = 100
    cfg.train.entropy_update_every_steps = 2000
    cfg.train.entropy_warmup_steps = 40_000
    cfg.train.entropy_transition_steps = 10_000
    cfg.train.entropy_gamma_max = 1.0
    cfg.train.entropy_mode = "regularized"
    cfg.train.entropy_regularizer_c = 0.1
    cfg.train.entropy_regularizer_n = 3.0
    cfg.train.entropy_target = "sqrt-rate"
    cfg.train.entropy_plot_every_k_epochs = 5

    # ------------------------------------------------------------------
    # Optimizer metadata
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
    # Evaluation: base-codec 600K, Pareto-comparison 256-NFE sweep
    # ------------------------------------------------------------------
    cfg.evaluation = config_dict.ConfigDict()
    cfg.evaluation.checkpoint_path = (
        f"runs/{cfg.experiment}/checkpoints/step=000600000.pt"
    )

    cfg.evaluation.out_dir = (
        f"runs/{cfg.experiment}/evaluation_external_ppl_600K_pareto_compare_256nfe"
    )
    cfg.evaluation.samples_dir = (
        f"runs/{cfg.experiment}/evaluation_external_ppl_600K_pareto_compare_256nfe/samples"
    )
    cfg.evaluation.results_csv = (
        f"runs/{cfg.experiment}/evaluation_external_ppl_600K_pareto_compare_256nfe/results.csv"
    )
    cfg.evaluation.shared_text_cache_dir = (
        f"runs/{cfg.experiment}/evaluation_external_ppl_600K_pareto_compare_256nfe/shared_text_cache"
    )

    cfg.evaluation.use_amp = True
    cfg.evaluation.amp_dtype = "bf16"
    cfg.evaluation.num_sampling_steps = 255
    cfg.evaluation.use_compile = True
    cfg.evaluation.compile_mode = "default"

    cfg.evaluation.compile = config_dict.ConfigDict()
    cfg.evaluation.compile.warmup = True
    cfg.evaluation.compile.warmup_steps = 8

    cfg.evaluation.ati = config_dict.ConfigDict()
    cfg.evaluation.ati.enabled = False
    cfg.evaluation.ati.eta = 0.0

    # Base stochastic defaults; individual sampling specs override these.
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
    cfg.evaluation.external_ppl.micro_batch_size = 64

    cfg.evaluation.external_ppl.samplers = ["ddim_entropic"]
    cfg.evaluation.external_ppl.terminal_sigmas = [0.08]
    cfg.evaluation.external_ppl.guidance_scales = [0.0]
    cfg.evaluation.external_ppl.num_sampling_steps = 255
    cfg.evaluation.external_ppl.sc_refresh_mode = "carry"

    cfg.evaluation.external_ppl.sigma_max = None
    cfg.evaluation.external_ppl.score_mode = "full"
    cfg.evaluation.external_ppl.compute_real_reference = True
    cfg.evaluation.external_ppl.shared_cache_dir = cfg.evaluation.shared_text_cache_dir
    cfg.evaluation.external_ppl.seed = 42
    cfg.evaluation.external_ppl.checkpoints = ["step=000600000.pt"]

    # Disabled for standard base codec.
    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_decode = False
    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_max_rows = 8
    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_once = True

    # ------------------------------------------------------------------
    # Sampling sweep: 256-NFE Pareto-comparison knots
    #
    # Goal:
    #   Compare the base-codec 600K model against the full Pareto front of
    #   the other codec/model without running a huge sweep.
    #
    # We include:
    #   - deterministic eta=0.0 anchor
    #   - deterministic eta=0.6 ATI anchor
    #   - low-churn / best-PPL region:
    #       gamma = 0.095, 0.130
    #   - LangFlow-competitive / transition region:
    #       gamma = 0.170, 0.200, 0.215
    #   - high-entropy transition region:
    #       gamma = 0.230, 0.250
    #
    # Fixed:
    #   s_noise = 1.003
    #   entropy window q=[0.10, 0.90]
    #
    # target_nfe = 256
    # num_intervals = 255
    # s_churn = gamma_target * 255
    #
    # Gamma -> s_churn:
    #   0.095 -> 24.225
    #   0.130 -> 33.150
    #   0.170 -> 43.350
    #   0.200 -> 51.000
    #   0.215 -> 54.825
    #   0.230 -> 58.650
    #   0.250 -> 63.750
    #
    # Total:
    #   2 deterministic + 7 gammas x 2 etas = 16 configs
    # ------------------------------------------------------------------
    cfg.evaluation.sampling_sweep = config_dict.ConfigDict()
    cfg.evaluation.sampling_sweep.enabled = True
    cfg.evaluation.sampling_sweep.target_nfes = []
    cfg.evaluation.sampling_sweep.specs = []

    def add_spec(
        *,
        target_nfe: int,
        ati_eta: float,
        stochastic_enabled: bool,
        gamma_target: float | None = None,
        s_noise: float = 1.003,
        qlo: float = 0.10,
        qhi: float = 0.90,
    ):
        spec = config_dict.ConfigDict()
        spec.sampler_name = "ddim_entropic"
        spec.sc_refresh_modes = ["carry"]
        spec.target_nfes = [int(target_nfe)]
        spec.ati_etas = [float(ati_eta)]

        spec.stochastic_enabled = bool(stochastic_enabled)

        if stochastic_enabled:
            if gamma_target is None:
                raise ValueError("gamma_target must be provided when stochastic_enabled=True")

            num_intervals = max(1, int(target_nfe) - 1)
            s_churn = float(gamma_target * num_intervals)

            spec.s_churn = float(s_churn)
            spec.s_noise = float(s_noise)
            spec.window_mode = "entropy_cdf"
            spec.entropy_quantile_lo = float(qlo)
            spec.entropy_quantile_hi = float(qhi)
            spec.entropy_fallback = "deterministic"
            spec.s_tmin = None
            spec.s_tmax = None

            # Metadata/debugging only; sampler uses s_churn directly.
            spec.gamma_target = float(gamma_target)

        cfg.evaluation.sampling_sweep.specs.append(spec)

    target_nfe = 256

    # Deterministic anchors.
    add_spec(
        target_nfe=target_nfe,
        ati_eta=0.0,
        stochastic_enabled=False,
    )
    add_spec(
        target_nfe=target_nfe,
        ati_eta=0.6,
        stochastic_enabled=False,
    )

    gamma_grid_compare_pareto = [
        0.095,
        0.130,
        0.170,
        0.200,
        0.215,
        0.230,
        0.250,
    ]

    eta_grid_compare_pareto = [
        0.6,
        0.9,
    ]

    for gamma_target in gamma_grid_compare_pareto:
        for eta in eta_grid_compare_pareto:
            add_spec(
                target_nfe=target_nfe,
                ati_eta=eta,
                stochastic_enabled=True,
                gamma_target=gamma_target,
                s_noise=1.003,
                qlo=0.10,
                qhi=0.90,
            )

    print(
        "[sampling_sweep] base-codec 600K Pareto-comparison 256-NFE specs = "
        f"{len(cfg.evaluation.sampling_sweep.specs)} configs "
        f"(2 deterministic + {len(gamma_grid_compare_pareto)} gammas x "
        f"{len(eta_grid_compare_pareto)} etas)"
    )

    # ------------------------------------------------------------------
    # MAUVE
    # ------------------------------------------------------------------
    cfg.evaluation.mauve = config_dict.ConfigDict()
    cfg.evaluation.mauve.enabled = False

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    cfg.logging = config_dict.ConfigDict()
    cfg.logging.use_wandb = False
    cfg.logging.entity = "continuousDLMs"
    cfg.logging.project = "owt"
    cfg.logging.group = "owt_base_codec_continuous_raw_binary_bits_600K_eval"
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