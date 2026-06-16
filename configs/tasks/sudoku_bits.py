# configs/tasks/sudoku_bits.py
#
# CoBit on the S-FLM Sudoku task (parity replication).
#   - vocab 12, 4 bits/token, 180 tokens -> 720-bit bitstream
#   - prompt = [BOS] puzzle [BOS] (91 tokens), conditioned (kept clean),
#     excluded from the loss; loss on the 89-token solution suffix.
#   - difficulty (easy/medium/hard = 40/35/30 clues) via env SUDOKU_DIFFICULTY.
#   - model trunk matches S-FLM: 8 blocks / 512-d / 8 heads.
#   - 20k steps, global batch 256, EMA 0.9999, 180 sampling steps.
import os
from ml_collections import config_dict


def get_config():
    cfg = config_dict.ConfigDict()

    difficulty = os.environ.get("SUDOKU_DIFFICULTY", "easy")
    assert difficulty in {"easy", "medium", "hard"}, difficulty

    cfg.framework = "continuous_score"
    cfg.experiment = f"tasks/sudoku/{difficulty}/cobit_raw_binary_bits"
    cfg.device = "cuda"

    # ------------------------------------------------------------------ data
    cfg.data = config_dict.ConfigDict()
    cfg.data.dataset = "sudoku"
    cfg.data.root = "datasets/sudoku"
    cfg.data.difficulty = difficulty
    cfg.data.num_train = int(os.environ.get("SUDOKU_NUM_TRAIN", "48000"))
    cfg.data.num_valid = int(os.environ.get("SUDOKU_NUM_VALID", "2000"))
    cfg.data.data_seed = 42
    cfg.data.sudoku_num_workers = int(os.environ.get("SUDOKU_GEN_WORKERS", "8"))

    cfg.data.representation = "binary"
    cfg.data.binarization = "raw_binary"
    cfg.data.token_space = "tokenizer_id"
    cfg.data.bits_per_token = 4
    cfg.data.sequence_len_tokens = 180
    cfg.data.sequence_len = 180 * 4  # 720
    cfg.data.token_vocab_size = 12   # for decode/eval (not the model vocab)

    cfg.data.vocab_size = 2          # binary model state
    cfg.data.channels = 1
    cfg.data.flatten_order = "flatten"

    cfg.data.num_workers = 4
    cfg.data.prefetch_factor = 4
    cfg.data.pin_memory = True

    # ------------------------------------------------------- conditioning
    # Prefix mask comes per-example from the batch; these defaults make the
    # prompt clean (clamped) and excluded from the loss.
    cfg.cond = config_dict.ConfigDict()
    cfg.cond.enabled = True
    cfg.cond.sample_prompt_len = False
    cfg.cond.cond_len_tokens = 91   # informational; batch mask is authoritative
    cfg.cond.cond_len_chars = 0
    cfg.cond.p_uncond = 0.0
    cfg.cond.noise_prefix = False
    cfg.cond.loss_on_suffix_only = True
    cfg.cond.null_strategy = "half"

    # ----------------------------------------------------------------- model
    cfg.model = config_dict.ConfigDict()
    cfg.model.name = "sdt"
    cfg.model.use_flash_attn = True
    cfg.model.self_condition = True
    cfg.model.center_inputs = True
    cfg.model.patch_size = 4         # 1 sudoku token = 4 bits

    cfg.model.embed_dim = 512
    cfg.model.dim_ff = 2048
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

    # ------------------------------------------------------- continuous diff
    cfg.diffusion = config_dict.ConfigDict()
    cfg.diffusion.continuous = config_dict.ConfigDict()
    cfg.diffusion.continuous.sigma_min = 0.002
    cfg.diffusion.continuous.sigma_max = 80.0
    cfg.diffusion.continuous.rho = 7.0
    cfg.diffusion.continuous.sigma_data = 0.5
    cfg.diffusion.continuous.data_center = 0.5
    cfg.diffusion.continuous.p_mean = -1.2
    cfg.diffusion.continuous.p_std = 1.2

    # -------------------------------------------------------------- training
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

    cfg.train.batch_size = 256
    cfg.train.epochs = 100000          # safety cap; optim.total_steps stops it
    cfg.train.ema_decay = 0.9999
    cfg.train.sigma_sampling_strategy = "log-normal"
    cfg.train.self_condition_prob = 0.5

    # entropy-rate schedule (online) — shorter warmup for the 20k-step run.
    cfg.train.entropy_offline = config_dict.ConfigDict()
    cfg.train.entropy_offline.enabled = False
    cfg.train.entropy_compute = True
    cfg.train.entropy_use_for_sampling = True
    cfg.train.entropy_buffer_size = 400_000
    cfg.train.entropy_num_bins = 128
    cfg.train.entropy_min_per_bin = 50
    cfg.train.entropy_update_every_steps = 1000
    cfg.train.entropy_warmup_steps = 4_000
    cfg.train.entropy_transition_steps = 2_000
    cfg.train.entropy_gamma_max = 1.0
    cfg.train.entropy_mode = "regularized"
    cfg.train.entropy_regularizer_c = 0.1
    cfg.train.entropy_regularizer_n = 3.0
    cfg.train.entropy_target = "sqrt-rate"
    cfg.train.entropy_plot_every_k_epochs = 50

    cfg.train.checkpointing = config_dict.ConfigDict()
    cfg.train.checkpointing.save_last = True
    cfg.train.checkpointing.save_top_k = 1
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

    # Disable text/owt-specific eval callbacks during training; we evaluate
    # the task offline with evaluation/tasks/sudoku_eval.py.
    for name in ("generation", "external_ppl", "mauve", "visualization", "vlb"):
        sub = config_dict.ConfigDict()
        sub.enabled = False
        setattr(cfg.train, name, sub)

    # ------------------------------------------------------- optim / sched
    cfg.optim = config_dict.ConfigDict()
    cfg.optim.optimizer = "AdamW"
    cfg.optim.lr = 3e-4
    cfg.optim.weight_decay = 0.0
    cfg.optim.beta1 = 0.9
    cfg.optim.beta2 = 0.999
    cfg.optim.eps = 1e-8
    cfg.optim.grad_clip = 1.0
    cfg.optim.scheduler = "constant"
    cfg.optim.total_steps = 20_000
    cfg.optim.warmup = 2_500

    # --------------------------------------------------------- evaluation
    cfg.evaluation = config_dict.ConfigDict()
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/step=000020000.pt"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/sudoku_eval"
    cfg.evaluation.use_amp = True
    cfg.evaluation.amp_dtype = "bf16"
    cfg.evaluation.num_sampling_steps = 180
    cfg.evaluation.use_compile = False
    cfg.evaluation.ati = config_dict.ConfigDict()
    cfg.evaluation.ati.enabled = False
    cfg.evaluation.ati.eta = 0.0
    cfg.evaluation.sampling_sweep = config_dict.ConfigDict()
    cfg.evaluation.sampling_sweep.enabled = False
    cfg.evaluation.sampling_sweep.target_nfes = []
    cfg.evaluation.sampling_sweep.specs = []
    cfg.evaluation.mauve = config_dict.ConfigDict()
    cfg.evaluation.mauve.enabled = False

    # ----------------------------------------------------------- logging
    cfg.logging = config_dict.ConfigDict()
    cfg.logging.use_wandb = False
    cfg.logging.project = "cobit_sudoku"
    cfg.logging.mode = "offline"
    cfg.logging.run_id = None
    cfg.logging.tensorboard = config_dict.ConfigDict()
    cfg.logging.tensorboard.enabled = True
    cfg.logging.tensorboard.scalar_every_steps = 20
    cfg.logging.tensorboard.sync_every_epochs = 1
    cfg.logging.tensorboard.sync_every_steps = 500
    cfg.logging.tensorboard.fail_silently = True

    return cfg
