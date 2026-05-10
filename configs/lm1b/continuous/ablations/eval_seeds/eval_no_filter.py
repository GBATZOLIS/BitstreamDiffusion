import os
from ml_collections import config_dict

def get_config():
    cfg = config_dict.ConfigDict()
    eval_seed = int(os.environ.get("EVAL_SEED", 42))

    cfg.framework = "continuous_score"
    cfg.experiment = "paper/unconditional_text/lm1b/continuous_rate_raw_binary_bits_1M_no_matched_filter"
    cfg.device = "cuda"

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

    cfg.cond = config_dict.ConfigDict()
    cfg.cond.enabled = False

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

    # ABLATION CHANGE: No analytic filter
    cfg.model.continuous_logit_scaling = "none"
    cfg.model.matched_filter_center = 0.5
    cfg.model.matched_filter_scale = 1.0
    cfg.model.matched_filter_clip = 30.0

    cfg.diffusion = config_dict.ConfigDict()
    cfg.diffusion.continuous = config_dict.ConfigDict()
    cfg.diffusion.continuous.sigma_min = 0.002
    cfg.diffusion.continuous.sigma_max = 80.0
    cfg.diffusion.continuous.rho = 7.0
    cfg.diffusion.continuous.sigma_data = 0.5
    cfg.diffusion.continuous.data_center = 0.5
    cfg.diffusion.continuous.p_mean = -1.2
    cfg.diffusion.continuous.p_std = 1.2

    cfg.train = config_dict.ConfigDict()
    cfg.train.seed = 42
    cfg.train.use_fp16 = True
    cfg.train.amp_dtype = "bf16"
    cfg.train.allow_tf32 = True

    cfg.train.loss_type = "binary_sm"
    cfg.train.loss_weighting = "edm"
    cfg.train.batch_size = 512

    cfg.train.entropy_offline = config_dict.ConfigDict()
    cfg.train.entropy_use_for_sampling = True

    cfg.evaluation = config_dict.ConfigDict()
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/step=000900000.pt"
    
    out_dir_name = f"evaluation_table1_step900K_seed{eval_seed}"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/{out_dir_name}"
    cfg.evaluation.samples_dir = f"{cfg.evaluation.out_dir}/samples"
    cfg.evaluation.results_csv = f"{cfg.evaluation.out_dir}/results.csv"
    cfg.evaluation.shared_text_cache_dir = f"{cfg.evaluation.out_dir}/shared_text_cache"

    cfg.evaluation.use_amp = True
    cfg.evaluation.amp_dtype = "bf16"
    cfg.evaluation.num_sampling_steps = 128
    cfg.evaluation.use_compile = True
    cfg.evaluation.compile_mode = "default"
    
    cfg.evaluation.compile = config_dict.ConfigDict()
    cfg.evaluation.compile.warmup = True
    cfg.evaluation.compile.warmup_steps = 8

    cfg.evaluation.ati = config_dict.ConfigDict()
    cfg.evaluation.ati.enabled = False
    cfg.evaluation.ati.eta = 0.0

    cfg.evaluation.sampling_sweep = config_dict.ConfigDict()
    cfg.evaluation.sampling_sweep.enabled = True
    cfg.evaluation.sampling_sweep.target_nfes = [128]
    cfg.evaluation.sampling_sweep.specs = [
        config_dict.ConfigDict({
            "sampler_name": "ddim_entropic",
            "sc_refresh_modes": ["carry"],
            "target_nfes": [128],
            "ati_etas": [0.0],
            "stochastic_enabled": False,
        })
    ]

    cfg.evaluation.external_ppl = config_dict.ConfigDict()
    cfg.evaluation.external_ppl.enabled = True
    cfg.evaluation.external_ppl.backend = "hf_causal_lm"
    cfg.evaluation.external_ppl.hf_model_name = "openai-community/gpt2-large"
    cfg.evaluation.external_ppl.hf_dtype = "bfloat16"
    cfg.evaluation.external_ppl.attn_implementation = "sdpa"
    cfg.evaluation.external_ppl.num_samples = 1024
    cfg.evaluation.external_ppl.micro_batch_size = 512
    cfg.evaluation.external_ppl.samplers = ["ddim_entropic"]
    cfg.evaluation.external_ppl.terminal_sigmas = [0.08]
    cfg.evaluation.external_ppl.guidance_scales = [0.0]
    cfg.evaluation.external_ppl.num_sampling_steps = 128
    cfg.evaluation.external_ppl.score_mode = "full"
    cfg.evaluation.external_ppl.compute_real_reference = True
    cfg.evaluation.external_ppl.shared_cache_dir = cfg.evaluation.shared_text_cache_dir
    cfg.evaluation.external_ppl.seed = eval_seed
    cfg.evaluation.external_ppl.checkpoints = ["step=000900000.pt"]

    return cfg