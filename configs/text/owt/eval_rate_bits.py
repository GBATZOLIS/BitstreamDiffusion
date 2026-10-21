from ml_collections import config_dict


def get_config():
    cfg = config_dict.ConfigDict()

    # ------------------------------------------------------------------
    # Framework / experiment
    # ------------------------------------------------------------------
    cfg.framework = "continuous_score"
    cfg.experiment = "paper/unconditional_text/owt/continuous_rate_raw_binary_bits_1M"
    cfg.device = "cuda"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    cfg.data = config_dict.ConfigDict()
    cfg.data.dataset = "OpenWebText"
    cfg.data.root = "datasets/openwebtext_gpt2_trainm100k"
    cfg.data.tokenizer_name = "gpt2"

    # second-stage GPT2 -> fixed-length BPE16 codec
    cfg.data.sequence_codec = "gpt2id_bpe16"
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
    # Diffusion
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
    # Train
    # ------------------------------------------------------------------
    cfg.train = config_dict.ConfigDict()
    cfg.train.seed = 42
    cfg.train.use_compile = True
    cfg.train.compile_mode = "default"
    cfg.train.use_fp16 = True
    cfg.train.amp_dtype = "bf16"
    cfg.train.allow_tf32 = True
    cfg.train.loss_type = "binary_sm"
    cfg.train.loss_weighting = "edm"
    cfg.train.batch_size = 512

    # ------------------------------------------------------------------
    # Optim
    # ------------------------------------------------------------------
    cfg.optim = config_dict.ConfigDict()
    cfg.optim.total_steps = 1_000_000

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    cfg.evaluation = config_dict.ConfigDict()
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/step=000550000.pt"
    
    # Clean output directories to isolate this sweep
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/evaluation_external_ppl_550K_stochastic_sweep"
    cfg.evaluation.samples_dir = f"runs/{cfg.experiment}/evaluation_external_ppl_550K_stochastic_sweep/samples"
    cfg.evaluation.results_csv = f"runs/{cfg.experiment}/evaluation_external_ppl_550K_stochastic_sweep/results.csv"
    cfg.evaluation.shared_text_cache_dir = (
        f"runs/{cfg.experiment}/evaluation_external_ppl_550K_stochastic_sweep/shared_text_cache"
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

    # ------------------------------------------------------------------
    # Base Stochastic Fallbacks
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

    # These get superseded by the Sweep config below
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
    cfg.evaluation.external_ppl.checkpoints = ["step=000550000.pt"]

    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_decode = True
    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_max_rows = 8
    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_once = True

    # ------------------------------------------------------------------
    # Sampling sweep: Deterministic Baseline + 3 Stochastic Variants
    # ------------------------------------------------------------------
    cfg.evaluation.sampling_sweep = config_dict.ConfigDict()
    cfg.evaluation.sampling_sweep.enabled = True
    cfg.evaluation.sampling_sweep.target_nfes = []
    cfg.evaluation.sampling_sweep.specs = []

    # 1. Deterministic baseline
    spec1 = config_dict.ConfigDict()
    spec1.sampler_name = "ddim_entropic"
    spec1.sc_refresh_modes = ["carry"]
    spec1.target_nfes = [256]
    spec1.ati_etas = [0.0]
    spec1.stochastic_enabled = False
    cfg.evaluation.sampling_sweep.specs.append(spec1)

    # 2. Conservative Stochastic Churn
    spec2 = config_dict.ConfigDict()
    spec2.sampler_name = "ddim_entropic"
    spec2.sc_refresh_modes = ["carry"]
    spec2.target_nfes = [256]
    spec2.ati_etas = [0.0]
    spec2.stochastic_enabled = True
    spec2.s_churn = 4.0
    spec2.s_noise = 1.003
    spec2.window_mode = "entropy_cdf"
    spec2.entropy_quantile_lo = 0.10
    spec2.entropy_quantile_hi = 0.90
    cfg.evaluation.sampling_sweep.specs.append(spec2)

    # 3. Moderate Stochastic Churn
    spec3 = config_dict.ConfigDict()
    spec3.sampler_name = "ddim_entropic"
    spec3.sc_refresh_modes = ["carry"]
    spec3.target_nfes = [256]
    spec3.ati_etas = [0.0]
    spec3.stochastic_enabled = True
    spec3.s_churn = 8.0
    spec3.s_noise = 1.003
    spec3.window_mode = "entropy_cdf"
    spec3.entropy_quantile_lo = 0.10
    spec3.entropy_quantile_hi = 0.90
    cfg.evaluation.sampling_sweep.specs.append(spec3)

    # 4. Tight-Window Stochastic Churn
    spec4 = config_dict.ConfigDict()
    spec4.sampler_name = "ddim_entropic"
    spec4.sc_refresh_modes = ["carry"]
    spec4.target_nfes = [256]
    spec4.ati_etas = [0.0]
    spec4.stochastic_enabled = True
    spec4.s_churn = 8.0
    spec4.s_noise = 1.005
    spec4.window_mode = "entropy_cdf"
    spec4.entropy_quantile_lo = 0.15
    spec4.entropy_quantile_hi = 0.85
    cfg.evaluation.sampling_sweep.specs.append(spec4)

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

    return cfg