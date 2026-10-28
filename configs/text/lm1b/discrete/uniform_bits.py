from ml_collections import config_dict


# LM1B unconditional discrete uniform-bits baseline
# Aligned as closely as possible to the absorb-bits matched baseline:
# same trunk scale, same callback cadence, same sampler family/step count.
# The key differences are only those required by bit-space modeling:
# binary representation, vocab_size=2, seq_len=128*15, patch_size=15,
# and the uniform-state process instead of absorbing masked process.


def get_config():
    cfg = config_dict.ConfigDict()

    # ------------------------------------------------------------------
    # Framework / experiment
    # ------------------------------------------------------------------
    cfg.framework = "discrete_sedd"
    cfg.experiment = "paper/unconditional_text/lm1b/uniform_bits_matched_to_tokens_33B"
    cfg.device = "cuda"

    # ------------------------------------------------------------------
    # Data: exact same semantic bitstreams as the continuous model
    # ------------------------------------------------------------------
    cfg.data = config_dict.ConfigDict()
    cfg.data.dataset = "LM1B"
    cfg.data.root = "datasets/lm1b"
    cfg.data.tokenizer_name = "bert-base-uncased"

    cfg.data.representation = "binary"
    cfg.data.binarization = "semantic"
    cfg.data.token_space = "semantic_rank"

    cfg.data.sequence_len_tokens = 128
    cfg.data.bits_per_token = 15
    cfg.data.sequence_len = 128 * 15
    cfg.data.val_fraction = 0.005

    # uniform binary alphabet: {0,1}
    cfg.data.vocab_size = 2
    cfg.data.mask_token_id = -1

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

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg.model = config_dict.ConfigDict()
    cfg.model.name = "sdt"
    cfg.model.use_flash_attn = True
    cfg.model.self_condition = False
    cfg.model.center_inputs = False

    # 15 bits per text token -> 128 trunk tokens, matching the token baseline
    # in number of trunk positions.
    cfg.model.patch_size = 15

    # Match the absorb-bits baseline trunk
    cfg.model.embed_dim = 768
    cfg.model.dim_ff = 3072
    cfg.model.n_blocks = 11
    cfg.model.n_heads = 12

    cfg.model.out_dim = cfg.data.vocab_size
    cfg.model.n_pos_features = 1
    cfg.model.dropout = 0.1

    # Match the absorb-bits auxiliary widths
    cfg.model.content_dim_discrete = 64
    cfg.model.content_dim_continuous = 64

    cfg.model.head_type = "optimal_skip_mlp"
    cfg.model.head_hidden = 192
    cfg.model.head_embed_dim = 64
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
    cfg.model.continuous_logit_scaling = "none"

    # ------------------------------------------------------------------
    # Discrete diffusion: uniform-state process over bits
    # ------------------------------------------------------------------
    cfg.diffusion = config_dict.ConfigDict()
    cfg.diffusion.discrete = config_dict.ConfigDict()
    cfg.diffusion.discrete.q_matrix_type = "uniform"
    cfg.diffusion.discrete.schedule = "geometric"
    cfg.diffusion.discrete.t_max = 1.0
    cfg.diffusion.discrete.eps = 1e-3
    cfg.diffusion.discrete.sigma_min = 1e-3
    cfg.diffusion.discrete.sigma_max = 1.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    cfg.train = config_dict.ConfigDict()
    cfg.train.deterministic = False
    cfg.train.seed = 42

    cfg.train.use_compile = True
    cfg.train.compile_mode = "default"
    cfg.train.use_fp16 = True
    cfg.train.amp_dtype = "bf16"
    cfg.train.allow_tf32 = True

    cfg.train.loss_normalize_by_seq = True
    cfg.train.loss_units = "bpd"

    cfg.train.batch_size = 512
    cfg.train.epochs = 9
    cfg.train.ema_decay = 0.9999

    cfg.train.checkpointing = config_dict.ConfigDict()
    cfg.train.checkpointing.save_last = True
    cfg.train.checkpointing.save_top_k = 2
    cfg.train.checkpointing.mode = "min"

    cfg.train.checkpointing.interval = config_dict.ConfigDict()
    cfg.train.checkpointing.interval.enabled = True
    cfg.train.checkpointing.interval.every_steps = 50_000
    cfg.train.checkpointing.interval.keep_last = 0

    cfg.train.sanity = config_dict.ConfigDict()
    cfg.train.sanity.enabled = False
    cfg.train.sanity.run_epoch = -1

    # ------------------------------------------------------------------
    # Training-time generation / visualization / MAUVE
    # ------------------------------------------------------------------
    cfg.train.generation = config_dict.ConfigDict()
    cfg.train.generation.enabled = True
    cfg.train.generation.splits = ["val"]
    cfg.train.generation.every_epochs = 1
    cfg.train.generation.num_samples = 64
    cfg.train.generation.num_sampling_steps = 128
    cfg.train.generation.samplers = ["tweedie"]
    cfg.train.generation.guidance_scales = [0.0]
    cfg.train.generation.micro_batch_size = 64
    cfg.train.generation.t_eps = 1e-3

    cfg.train.external_ppl = config_dict.ConfigDict()
    cfg.train.external_ppl.enabled = False

    cfg.train.mauve = config_dict.ConfigDict()
    cfg.train.mauve.enabled = True
    cfg.train.mauve.every_k_epochs = 1
    cfg.train.mauve.splits = ["val"]
    cfg.train.mauve.num_samples = 4096
    cfg.train.mauve.featurizer_name = "gpt2-large"
    cfg.train.mauve.max_tokens = cfg.data.sequence_len_tokens
    cfg.train.mauve.device_id = 0
    cfg.train.mauve.micro_batch_size = 512
    cfg.train.mauve.samplers = ["tweedie"]
    cfg.train.mauve.guidance_scales = [0.0]

    cfg.train.visualization = config_dict.ConfigDict()
    cfg.train.visualization.enabled = True
    cfg.train.visualization.every_k_epochs = 1
    cfg.train.visualization.splits = ["val"]
    cfg.train.visualization.num_samples = 16
    cfg.train.visualization.save_txt = True
    cfg.train.visualization.save_jsonl = True
    cfg.train.visualization.show_prefix_suffix = True
    cfg.train.visualization.micro_batch_size = 16

    # ------------------------------------------------------------------
    # Optimizer / scheduler
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
    cfg.optim.total_steps = 500_000
    cfg.optim.warmup = 2_500

    # ------------------------------------------------------------------
    # Evaluation stub
    # ------------------------------------------------------------------
    cfg.evaluation = config_dict.ConfigDict()
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/best.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/evaluation"
    cfg.evaluation.samples_dir = f"runs/{cfg.experiment}/evaluation/samples"
    cfg.evaluation.results_csv = f"runs/{cfg.experiment}/evaluation/results.csv"
    cfg.evaluation.shared_text_cache_dir = f"runs/{cfg.experiment}/evaluation/shared_text_cache"
    cfg.evaluation.use_amp = True
    cfg.evaluation.amp_dtype = "bf16"
    cfg.evaluation.num_sampling_steps = 128

    cfg.evaluation.mauve = config_dict.ConfigDict()
    cfg.evaluation.mauve.enabled = False

    cfg.evaluation.external_ppl = config_dict.ConfigDict()
    cfg.evaluation.external_ppl.enabled = False

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    cfg.logging = config_dict.ConfigDict()
    cfg.logging.use_wandb = False
    cfg.logging.entity = "continuousDLMs"
    cfg.logging.project = "lm1b"
    cfg.logging.group = "lm1b_uniform_bits_matched_to_tokens_33b"
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