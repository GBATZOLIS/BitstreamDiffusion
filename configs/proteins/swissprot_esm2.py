from ml_collections import config_dict

# Swiss-Prot ESM-2-tokenizer 6-bit pilot (20k steps). Run after the smoke passes.
# Model is the bitstream-diffusion denoiser (SDT); ESM-2 (nvidia/bionemo HF)
# provides the amino-acid tokenizer and the realism scorer used at eval time.

def get_config():
    cfg = config_dict.ConfigDict()

    # ------------------------------------------------------------------
    # Framework / experiment
    # ------------------------------------------------------------------
    cfg.framework = "continuous_score"
    cfg.experiment = "proteins/swissprot_esm2"
    cfg.device = "cuda"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    cfg.data = config_dict.ConfigDict()
    cfg.data.dataset = "SwissProt"
    cfg.data.root = "datasets/swissprot"
    cfg.data.fasta_name = "uniprot_sprot.fasta.gz"

    cfg.data.tokenizer = "esm2"  # ESM-2 amino-acid tokenizer (33 tokens, 6 bits)
    cfg.data.esm2_model = "nvidia/esm2_t6_8M_UR50D"

    cfg.data.representation = "binary"
    cfg.data.binarization = "raw_binary"

    cfg.data.sequence_len_tokens = 512
    cfg.data.bits_per_token = 6
    cfg.data.sequence_len = 512 * 6

    cfg.data.min_len = 20
    cfg.data.max_windows_per_seq = None
    cfg.data.val_fraction = 0.01
    cfg.data.test_fraction = 0.01

    cfg.data.vocab_size = 2
    cfg.data.channels = 1
    cfg.data.flatten_order = "flatten"

    cfg.data.num_workers = 12
    cfg.data.prefetch_factor = 4
    cfg.data.pin_memory = True

    # ------------------------------------------------------------------
    # Unconditional setting
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
    cfg.model.patch_size = 6

    cfg.model.embed_dim = 384
    cfg.model.dim_ff = 1536
    cfg.model.n_blocks = 8
    cfg.model.n_heads = 8

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
    # Training
    # ------------------------------------------------------------------
    cfg.train = config_dict.ConfigDict()
    cfg.train.deterministic = False
    cfg.train.seed = 42
    cfg.train.use_compile = False
    cfg.train.compile_mode = "default"
    cfg.train.use_fp16 = True
    cfg.train.amp_dtype = "bf16"
    cfg.train.allow_tf32 = True
    cfg.train.loss_type = "binary_sm"
    cfg.train.loss_weighting = "edm"
    cfg.train.batch_size = 256
    cfg.train.epochs = 1000
    cfg.train.ema_decay = 0.9999
    cfg.train.sigma_sampling_strategy = "log-normal"
    cfg.train.self_condition_prob = 0.5

    cfg.train.entropy_offline = config_dict.ConfigDict()
    cfg.train.entropy_offline.enabled = False
    cfg.train.entropy_compute = False
    cfg.train.entropy_use_for_sampling = False
    cfg.train.entropy_buffer_size = 200_000
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
    cfg.train.checkpointing.interval.every_steps = 5_000
    cfg.train.checkpointing.interval.keep_last = 0

    cfg.train.checkpointing.resume_interval = config_dict.ConfigDict()
    cfg.train.checkpointing.resume_interval.enabled = True
    cfg.train.checkpointing.resume_interval.every_steps = 1_000

    cfg.train.sanity = config_dict.ConfigDict()
    cfg.train.sanity.enabled = False
    cfg.train.sanity.run_epoch = -1

    cfg.train.generation = config_dict.ConfigDict()
    cfg.train.generation.enabled = False

    cfg.train.external_ppl = config_dict.ConfigDict()
    cfg.train.external_ppl.enabled = False

    cfg.train.mauve = config_dict.ConfigDict()
    cfg.train.mauve.enabled = False

    cfg.train.visualization = config_dict.ConfigDict()
    cfg.train.visualization.enabled = False

    # VLB bound on the validation set is bit-agnostic and useful to track.
    cfg.train.vlb = config_dict.ConfigDict()
    cfg.train.vlb.enabled = True
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
    cfg.train.vlb.max_batches_val = 50
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
    # Optimizer / scheduler
    # ------------------------------------------------------------------
    cfg.optim = config_dict.ConfigDict()
    cfg.optim.optimizer = "AdamW"
    cfg.optim.lr = 2e-4
    cfg.optim.weight_decay = 0.01
    cfg.optim.beta1 = 0.9
    cfg.optim.beta2 = 0.99
    cfg.optim.eps = 1e-8
    cfg.optim.grad_clip = 1.0
    cfg.optim.scheduler = "cosine_decay"
    cfg.optim.total_steps = 20_000
    cfg.optim.warmup = 200

    # ------------------------------------------------------------------
    # Evaluation (used by evaluation/protein_metrics.py)
    # ------------------------------------------------------------------
    cfg.evaluation = config_dict.ConfigDict()
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/last.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/protein_eval"
    cfg.evaluation.samples_dir = f"runs/{cfg.experiment}/protein_eval/samples"
    cfg.evaluation.use_amp = True
    cfg.evaluation.amp_dtype = "bf16"
    cfg.evaluation.num_sampling_steps = 128
    cfg.evaluation.use_compile = False
    cfg.evaluation.compile_mode = "default"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    cfg.logging = config_dict.ConfigDict()
    from configs.proteins._wandb import enable_wandb

    enable_wandb(cfg, group="swissprot_esm2")
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
