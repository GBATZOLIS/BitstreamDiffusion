import os
from ml_collections import config_dict

def get_config():
    cfg = config_dict.ConfigDict()
    
    # Grab the seed from the bash environment (default to 42)
    eval_seed = int(os.environ.get("EVAL_SEED", 42))

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
    # Train & Optim
    # ------------------------------------------------------------------
    cfg.train = config_dict.ConfigDict()
    cfg.train.seed = eval_seed
    cfg.train.use_compile = True
    cfg.train.compile_mode = "default"
    cfg.train.use_fp16 = True
    cfg.train.amp_dtype = "bf16"
    cfg.train.allow_tf32 = True
    cfg.train.loss_type = "binary_sm"
    cfg.train.loss_weighting = "edm"
    cfg.train.batch_size = 512
    
    cfg.optim = config_dict.ConfigDict()
    cfg.optim.total_steps = 1_000_000
    
    ckpt_name = "step=000750000.pt"

    # ------------------------------------------------------------------
    # Evaluation Base
    # ------------------------------------------------------------------
    cfg.evaluation = config_dict.ConfigDict()
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/{ckpt_name}"
    
    out_base = f"runs/{cfg.experiment}/evaluation_external_ppl_750K_full_band_sweep"
    cfg.evaluation.out_dir = out_base
    cfg.evaluation.samples_dir = f"{out_base}/samples"
    cfg.evaluation.results_csv = f"{out_base}/results.csv"
    cfg.evaluation.shared_text_cache_dir = f"{out_base}/shared_text_cache"

    cfg.evaluation.use_amp = True
    cfg.evaluation.amp_dtype = "bf16"
    cfg.evaluation.use_compile = True
    cfg.evaluation.compile_mode = "default"
    
    cfg.evaluation.compile = config_dict.ConfigDict()
    cfg.evaluation.compile.warmup = True
    cfg.evaluation.compile.warmup_steps = 8

    # Base Stochastic Fallbacks (Overridden in Sweep)
    cfg.evaluation.stochastic = config_dict.ConfigDict()
    cfg.evaluation.stochastic.enabled = False
    cfg.evaluation.stochastic.s_churn = 0.0
    cfg.evaluation.stochastic.s_noise = 1.003
    cfg.evaluation.stochastic.window_mode = "entropy_cdf"
    cfg.evaluation.stochastic.entropy_quantile_lo = 0.0
    cfg.evaluation.stochastic.entropy_quantile_hi = 1.0
    cfg.evaluation.stochastic.s_tmin = None
    cfg.evaluation.stochastic.s_tmax = None
    cfg.evaluation.stochastic.entropy_fallback = "deterministic"

    # ------------------------------------------------------------------
    # External PPL Config
    # ------------------------------------------------------------------
    cfg.evaluation.external_ppl = config_dict.ConfigDict()
    cfg.evaluation.external_ppl.enabled = True
    cfg.evaluation.external_ppl.backend = "hf_causal_lm"
    cfg.evaluation.external_ppl.hf_model_name = "openai-community/gpt2-large"
    cfg.evaluation.external_ppl.hf_revision = None
    cfg.evaluation.external_ppl.hf_dtype = "bfloat16"
    cfg.evaluation.external_ppl.attn_implementation = "sdpa"
    cfg.evaluation.external_ppl.num_samples = 1024
    cfg.evaluation.external_ppl.micro_batch_size = 256
    cfg.evaluation.external_ppl.samplers = ["ddim_entropic"]
    cfg.evaluation.external_ppl.terminal_sigmas = [0.08]
    cfg.evaluation.external_ppl.guidance_scales = [0.0]
    cfg.evaluation.external_ppl.num_sampling_steps = 255 # Handled dynamically mostly, but safe default
    cfg.evaluation.external_ppl.sc_refresh_mode = "carry"
    cfg.evaluation.external_ppl.sigma_max = None
    cfg.evaluation.external_ppl.score_mode = "full"
    cfg.evaluation.external_ppl.compute_real_reference = True
    cfg.evaluation.external_ppl.shared_cache_dir = cfg.evaluation.shared_text_cache_dir
    cfg.evaluation.external_ppl.seed = eval_seed
    cfg.evaluation.external_ppl.checkpoints = [ckpt_name]
    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_decode = True
    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_max_rows = 8
    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_once = True
    
    cfg.evaluation.mauve = config_dict.ConfigDict()
    cfg.evaluation.mauve.enabled = False

    # ------------------------------------------------------------------
    # Full Stochastic Band Pareto Sweep
    # ------------------------------------------------------------------
    cfg.evaluation.sampling_sweep = config_dict.ConfigDict()
    cfg.evaluation.sampling_sweep.enabled = True
    cfg.evaluation.sampling_sweep.target_nfes = []
    cfg.evaluation.sampling_sweep.specs = []

    def add_spec(*, target_nfe, ati_eta, stochastic_enabled, gamma_target=None, s_noise=1.003):
        spec = config_dict.ConfigDict()
        spec.sampler_name = "ddim_entropic"
        spec.sc_refresh_modes = ["carry"]
        spec.target_nfes = [int(target_nfe)]
        spec.ati_etas = [float(ati_eta)]
        spec.stochastic_enabled = bool(stochastic_enabled)
        
        if stochastic_enabled:
            num_intervals = max(1, int(target_nfe) - 1)
            spec.s_churn = float(gamma_target * num_intervals)
            spec.s_noise = float(s_noise)
            spec.window_mode = "entropy_cdf"
            spec.entropy_quantile_lo = 0.0 # ENFORCE FULL BAND
            spec.entropy_quantile_hi = 1.0 # ENFORCE FULL BAND
            spec.entropy_fallback = "deterministic"
            spec.s_tmin = None
            spec.s_tmax = None
            spec.gamma_target = float(gamma_target)
        cfg.evaluation.sampling_sweep.specs.append(spec)

    # 1. Small NFEs (<= 80): Sweep ATI eta & Gamma
    small_nfe_grids = {
        64: [0.01, 0.02, 0.03, 0.04, 0.05],
        80: [0.02, 0.03, 0.04, 0.05, 0.06]
    }
    eta_grid = [0.0, 0.3, 0.6]

    for nfe, gammas in small_nfe_grids.items():
        for eta in eta_grid:
            add_spec(target_nfe=nfe, ati_eta=eta, stochastic_enabled=False)
            for gamma in gammas:
                add_spec(target_nfe=nfe, ati_eta=eta, stochastic_enabled=True, gamma_target=gamma)

    # 2. Large NFEs (> 80): Force eta=0.0 & Sweep Gamma only
    large_nfe_grids = {
        100: [0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095],
        115: [0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090, 0.100],
        128: [0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.105],
        160: [0.045, 0.060, 0.075, 0.090, 0.105, 0.120, 0.135, 0.150],
        192: [0.050, 0.070, 0.090, 0.110, 0.130, 0.150, 0.170, 0.190],
        256: [0.050, 0.080, 0.110, 0.140, 0.170, 0.200, 0.230, 0.260],
        512: [0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400],
    }

    for nfe, gammas in large_nfe_grids.items():
        add_spec(target_nfe=nfe, ati_eta=0.0, stochastic_enabled=False)
        for gamma in gammas:
            add_spec(target_nfe=nfe, ati_eta=0.0, stochastic_enabled=True, gamma_target=gamma)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    cfg.logging = config_dict.ConfigDict()
    cfg.logging.use_wandb = False

    return cfg