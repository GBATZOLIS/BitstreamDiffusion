import os
from ml_collections import config_dict

def get_config():
    cfg = config_dict.ConfigDict()

    eval_seed = int(os.environ.get("EVAL_SEED", 42))

    # ------------------------------------------------------------------
    # Framework / experiment
    # ------------------------------------------------------------------
    cfg.framework = "continuous_score"
    cfg.experiment = "paper/unconditional_text/lm1b/continuous_rate_raw_binary_bits_1M_edm_weighting"
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
    # Training / Optim (Placeholders for eval script)
    # ------------------------------------------------------------------
    cfg.train = config_dict.ConfigDict()
    cfg.train.deterministic = False
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

    # ------------------------------------------------------------------
    # Evaluation Base
    # ------------------------------------------------------------------
    ckpt_name = "step=001000000.pt"
    
    cfg.evaluation = config_dict.ConfigDict()
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/{ckpt_name}"
    
    out_base = f"runs/{cfg.experiment}/evaluation_external_ppl_1M_full_band_sweep"
    cfg.evaluation.out_dir = out_base
    cfg.evaluation.samples_dir = f"{out_base}/samples"
    cfg.evaluation.results_csv = f"{out_base}/results.csv"
    cfg.evaluation.shared_text_cache_dir = f"{out_base}/shared_text_cache"

    cfg.evaluation.use_amp = True
    cfg.evaluation.amp_dtype = "bf16"
    cfg.evaluation.use_compile = True
    cfg.evaluation.compile_mode = "default"

    cfg.evaluation.compile = config_dict.ConfigDict()
    cfg.evaluation.compile.warmup = False
    cfg.evaluation.compile.warmup_steps = 0

    cfg.evaluation.ati = config_dict.ConfigDict()
    cfg.evaluation.ati.enabled = False
    cfg.evaluation.ati.eta = 0.0

    # Base Stochastic Fallbacks (Overridden in sweep)
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

    cfg.evaluation.mauve = config_dict.ConfigDict()
    cfg.evaluation.mauve.enabled = False

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
    cfg.evaluation.external_ppl.num_sampling_steps = 255 # Handled dynamically but safe default
    cfg.evaluation.external_ppl.sigma_max = None
    cfg.evaluation.external_ppl.score_mode = "full"
    cfg.evaluation.external_ppl.compute_real_reference = True
    cfg.evaluation.external_ppl.shared_cache_dir = cfg.evaluation.shared_text_cache_dir
    cfg.evaluation.external_ppl.seed = eval_seed
    cfg.evaluation.external_ppl.checkpoints = [ckpt_name]
    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_decode = False
    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_once = True
    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_max_rows = 8

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
    eta_grid = [0.0, 0.3, 0.6]
    nfe_64_gammas = [0.01, 0.02, 0.03, 0.04, 0.05]

    for eta in eta_grid:
        add_spec(target_nfe=64, ati_eta=eta, stochastic_enabled=False)
        for gamma in nfe_64_gammas:
            add_spec(target_nfe=64, ati_eta=eta, stochastic_enabled=True, gamma_target=gamma)

    # 2. Large NFEs (> 80): Force eta=0.0 & Sweep Gamma only
    lm1b_large_nfe_grids = {
        96:  [0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090],
        128: [0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.105],
        192: [0.050, 0.070, 0.090, 0.110, 0.130, 0.150, 0.170, 0.190],
        256: [0.050, 0.080, 0.110, 0.140, 0.170, 0.200, 0.230, 0.260],
        384: [0.050, 0.090, 0.130, 0.170, 0.210, 0.250, 0.290, 0.330],
        512: [0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400],
    }

    for nfe, gammas in lm1b_large_nfe_grids.items():
        add_spec(target_nfe=nfe, ati_eta=0.0, stochastic_enabled=False)
        for gamma in gammas:
            add_spec(target_nfe=nfe, ati_eta=0.0, stochastic_enabled=True, gamma_target=gamma)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    cfg.logging = config_dict.ConfigDict()
    cfg.logging.use_wandb = False
    cfg.logging.entity = "continuousDLMs"
    cfg.logging.project = "lm1b"
    cfg.logging.group = "lm1b_continuous_raw_binary_bits_trunk768"
    cfg.logging.mode = "offline"

    return cfg