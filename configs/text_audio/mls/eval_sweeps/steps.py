import os
from ml_collections import config_dict


def get_config():
    cfg = config_dict.ConfigDict()

    cfg.framework = "continuous_score"
    cfg.experiment = "cobit_mls"
    cfg.device = "cuda"

    # ------------------------------------------------------------------
    # Data -- shared 18-bit multimodal bits
    # ------------------------------------------------------------------

    cfg.data = config_dict.ConfigDict()
    cfg.data.dataset = 'textaudio'
    cfg.data.root = 'datasets/'
    cfg.data.text_tokenizer = 'o200k_base'
    cfg.data.speech_tokenizer = 'stabilityai/stable-codec-speech-16k'
    cfg.data.speech_tokenizer_bottleneck = '1x46656_400bps'
    cfg.data.speech_tokenizer_bottleneck_dims = None

    cfg.data.representation = 'binary'
    cfg.data.binarization = 'raw_binary'
    cfg.data.token_space = 'tokenizer_id'

    cfg.data.text_seq_len = 100
    cfg.data.speaker_seq_len = 32
    cfg.data.speech_seq_len = 500
    cfg.data.seq_len_tokens = 632
    cfg.data.bits_per_token = 18
    cfg.data.sequence_len = 632*18
    cfg.data.sample_rate = 16000

    cfg.data.text_vocab_size = 200019
    cfg.data.speaker_vocab_size = 4096
    cfg.data.speech_vocab_size = 46656
    cfg.data.speech_offset = 204115
    cfg.data.vocab_size = 2
    cfg.data.vocab_size_base = 250773
    cfg.data.channels = 1
    cfg.data.flatten_order = 'flatten'

    cfg.data.num_workers = 8
    cfg.data.prefetch_factor = 4
    cfg.data.pin_memory = True

    # ------------------------------------------------------------------
    # Conditioning -- random per-modality masking
    # ------------------------------------------------------------------
    cfg.cond = config_dict.ConfigDict()
    cfg.cond.enabled = True
    cfg.cond.continuation_prefix = 75
    cfg.cond.unconditional_rate = 0.25
    cfg.cond.texttospeech_rate = 0.25
    cfg.cond.speechtotext_rate = 0.25
    cfg.cond.continuation_rate = 0.25
    cfg.cond.downstream = True

    # ------------------------------------------------------------------
    # Model -- LARGE 26 x 1152 (~633M), patch_size 18, + segment embedding
    # ------------------------------------------------------------------
    cfg.model = config_dict.ConfigDict()
    cfg.model.name = "sdt"
    cfg.model.use_flash_attn = True
    cfg.model.self_condition = True
    cfg.model.center_inputs = True

    cfg.model.patch_size = 18
    cfg.model.use_segment_embed = True

    cfg.model.embed_dim = 1152
    cfg.model.dim_ff = 4608
    cfg.model.n_blocks = 26
    cfg.model.n_heads = 18                            # head_dim = 64

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
    # Continuous diffusion (same proven EDM schedule)
    # ------------------------------------------------------------------
    cfg.diffusion = config_dict.ConfigDict()
    cfg.diffusion.continuous = config_dict.ConfigDict()
    cfg.diffusion.continuous.sigma_min = 0.01
    cfg.diffusion.continuous.sigma_max = 80.0
    cfg.diffusion.continuous.rho = 7.0
    cfg.diffusion.continuous.sigma_data = 0.5
    cfg.diffusion.continuous.data_center = 0.5
    cfg.diffusion.continuous.p_mean = -1.2
    cfg.diffusion.continuous.p_std = 1.2

    # ------------------------------------------------------------------
    # Training -- MAIN: 2M steps, global batch 512 (32/GPU x 16 GH200)
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

    cfg.train.batch_size = 512           # global; trainer shards batch//world_size (512/16 = 32/GPU)
    cfg.train.epochs = 110               # ~2M steps at 512 over ~11M samples (~21.8K steps/epoch); total_steps governs
    cfg.train.ema_decay = 0.9999
    cfg.train.sigma_sampling_strategy = "log-normal"
    cfg.train.self_condition_prob = 0.5

    # Entropy schedule (same recipe as pilot).
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
    cfg.train.checkpointing.resume_interval.every_steps = 2_000

    cfg.train.sanity = config_dict.ConfigDict()
    cfg.train.sanity.enabled = False
    cfg.train.sanity.run_epoch = -1

    cfg.train.textaudio = config_dict.ConfigDict()
    cfg.train.textaudio.enabled = False
    cfg.train.textaudio.run_on_sanity = True
    cfg.train.textaudio.every_epochs = 10
    cfg.train.textaudio.split = 'val'
    cfg.train.textaudio.num_samples = 64
    cfg.train.textaudio.whisper_model = 'openai/whisper-medium'

    cfg.train.textaudio.sampling_sweep = config_dict.ConfigDict()

    # ------------------------------------------------------------------
    # Optimizer / scheduler -- MAIN horizon 2M (cosine, stop-anytime)
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
    cfg.optim.total_steps = 2_000_000
    cfg.optim.warmup = 10_000

    # ------------------------------------------------------------------
    # Smoke mode (env-driven) -- single-GPU / R1 probe on a CC12M smoke cache
    # ------------------------------------------------------------------
    _smoke = int(os.environ.get("SMOKE_MAX_STEPS", "0") or 0)
    if _smoke > 0:
        cfg.experiment = f"{cfg.experiment}_smoke"
        cfg.data.precomputed_root = f"{cfg.data.precomputed_root}_SMOKE"
        cfg.optim.total_steps = _smoke
        cfg.train.epochs = 1
        cfg.train.checkpointing.interval.every_steps = max(_smoke // 2, 1)
        cfg.train.checkpointing.resume_interval.every_steps = max(_smoke // 4, 1)
        cfg.train.visualization.every_k_epochs = 1
        cfg.train.vlb.every_k_epochs = 1
        cfg.train.batch_size = int(os.environ.get("SMOKE_BATCH", "32"))
        cfg.train.use_compile = bool(int(os.environ.get("SMOKE_COMPILE", "1")))
        cfg.model.use_flash_attn = bool(int(os.environ.get("SMOKE_FLASH", "1")))

    # ------------------------------------------------------------------
    # Evaluation -- multimodal both-way + joint (inherits pilot protocol)
    # ------------------------------------------------------------------
    cfg.evaluation = config_dict.ConfigDict()
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/last.pt"

    cfg.evaluation.multimodal_text_audio = config_dict.ConfigDict()
    cfg.evaluation.multimodal_text_audio.batch_size = 64
    cfg.evaluation.multimodal_text_audio.whisper_model = "openai/whisper-large"
    
    cfg.evaluation.multimodal_text_audio.wlm_statistics_path = "datasets/test_common/fsd_ref_stats_cont_common_wlm.npz"
    cfg.evaluation.multimodal_text_audio.e2v_statistics_path = "datasets/test_common/fsd_ref_stats_cont_common_e2v.npz"
    cfg.evaluation.multimodal_text_audio.sampling_sweep = config_dict.ConfigDict()
    cfg.evaluation.multimodal_text_audio.sampling_sweep.specs = []

    from evaluation.nfe import steps_for_target_nfe

    def add_eval_textaudio_spec(
        sampler_name, target_nfes, stochastic_enabled=False, gamma_targets=None,
        guidance_scales=None, sc_refresh_mode="carry", tasks=None,
    ):
        spec = config_dict.ConfigDict()
        spec.sampler_name = sampler_name
        spec.sc_refresh_mode = sc_refresh_mode
        spec.target_nfes = list(target_nfes)
        spec.stochastic_enabled = bool(stochastic_enabled)
        if stochastic_enabled:
            gamma_targets = list(gamma_targets) if gamma_targets is not None else [0.25]
            s_churns = []
            for target_nfe in target_nfes:
                num_intervals, _ = steps_for_target_nfe(
                    framework=cfg.framework,
                    sampler_name=sampler_name,
                    target_nfe=target_nfe,
                    self_condition=cfg.model.self_condition,
                    sc_refresh_mode=sc_refresh_mode,
                    return_probs=True,
                )
                s_churns.extend(round(g * num_intervals, 2) for g in gamma_targets)
            spec.s_churns = s_churns
        if guidance_scales is not None:
            spec.guidance_scales = list(guidance_scales)
        if tasks is not None:
            spec.tasks = list(tasks)
        cfg.evaluation.multimodal_text_audio.sampling_sweep.specs.append(spec)

    nfes = [4,8,16,24,32,48,64,128,256,512]

    for nfe in nfes:
        add_eval_textaudio_spec(
            'ddim_entropic', target_nfes=[nfe], stochastic_enabled=True,
            gamma_targets=[0.175], guidance_scales=[0.0], tasks=['joint'],
        )
        add_eval_textaudio_spec(
            'ddim_entropic', target_nfes=[nfe], stochastic_enabled=True,
            gamma_targets=[0.175], guidance_scales=[5.0], tasks=['tts', 'stt', 'cont'],
        )


    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    cfg.logging = config_dict.ConfigDict()
    cfg.logging.use_wandb = False
    cfg.logging.entity = None
    cfg.logging.project = "cobit_mls"
    cfg.logging.group = "full"
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

