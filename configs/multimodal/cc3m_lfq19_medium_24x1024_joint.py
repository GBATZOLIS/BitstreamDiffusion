# configs/multimodal/cc3m_lfq19_medium_24x1024_joint.py
#
# Joint image+text bitstream diffusion on Conceptual Captions 3M (CC3M),
# evaluated zero-shot on MS-COCO-30K (T2I FID+CLIP, I2T CLIP). First step toward
# true multimodality. Same 462M SDT-medium 24x1024 trunk / EDM schedule / optimizer
# as the ImageNet config (configs/ImageNet/lfq_f16_18b_medium_24x1024.py).
#
# Shared 19-BIT vocabulary (data/mm_codec.py):
#   bit 18 = namespace: 0=image (LFQ low-18-bit), 1=text/control (o200k id + markers)
#   patch_size = 19  ->  one transformer position = one token of any modality.
# Layout per example:
#   [SOS][SOT] text(Lt) [EOT][SOI] image(256) [EOI][EOS]
#   markers are STRUCTURAL (always clamped clean).
#
# Conditioning: random per-modality masking (cfg.cond.cond_mode="multimodal_mask"),
# training p(image|text), p(text|image), and the joint p(image,text) in one model.
#
# Prereq cache:  scripts/multimodal/phase_10_prepare_mm_tokens.py
# Run dir:       runs/multimodal/cc3m_lfq19_medium_24x1024_joint/
# Launcher:      scripts/multimodal/train_cc3m_joint_hpc.sh

import os
from ml_collections import config_dict

from data import mm_codec


def get_config():
    cfg = config_dict.ConfigDict()

    cfg.framework = "continuous_score"
    cfg.experiment = "multimodal/cc3m_lfq19_medium_24x1024_joint"
    cfg.device = "cuda"

    # ------------------------------------------------------------------
    # Data -- shared 19-bit multimodal bits (CC3M)
    # ------------------------------------------------------------------
    # o200k caption-length analysis on CC3M (17.7K-caption sample):
    #   mean=10.8, median=10, p99=27, p99.9=37, max=46 tokens.
    # Lt=48 covers the CC3M max with headroom for (longer) zero-shot COCO eval
    # captions => ~0% truncation, at the cost of only +16 positions vs Lt=32.
    CAPTION_LEN_TOKENS = 48
    NUM_IMAGE_TOKENS = 256           # Open-MAGVIT2 f16 (16x16)
    _layout = mm_codec.build_layout(CAPTION_LEN_TOKENS, NUM_IMAGE_TOKENS)

    cfg.data = config_dict.ConfigDict()
    cfg.data.dataset = "MultiModalLFQBits"
    cfg.data.precomputed_root = "datasets/cc3m_mm_lfq19_256"

    cfg.data.text_tokenizer = "o200k_base"
    cfg.data.caption_len_tokens = CAPTION_LEN_TOKENS
    cfg.data.num_image_tokens = NUM_IMAGE_TOKENS

    # Image-token geometry (for the image-slice decode path).
    cfg.data.image_resolution = 256
    cfg.data.token_grid_h = 16
    cfg.data.token_grid_w = 16
    cfg.data.flatten_order = "hilbert"            # image tokens stored Hilbert (matches ImageNet)

    # Shared-vocab sequence geometry (the SDT trunk only sees a flat bitstream).
    cfg.data.bits_per_token = mm_codec.BITS_PER_TOKEN     # 19 (shared patch token)
    cfg.data.num_positions = _layout.num_positions        # Lt + 262
    cfg.data.sequence_len = _layout.total_bits            # num_positions * 19
    cfg.data.return_shape = "flat"

    cfg.data.vocab_size = 2          # binary continuous codec
    cfg.data.channels = 1

    cfg.data.num_workers = 6
    cfg.data.prefetch_factor = 4
    cfg.data.pin_memory = True

    # ------------------------------------------------------------------
    # Conditioning -- random per-modality masking (joint + both conditionals)
    # ------------------------------------------------------------------
    cfg.cond = config_dict.ConfigDict()
    cfg.cond.enabled = True
    cfg.cond.cond_mode = "multimodal_mask"
    cfg.cond.noise_prefix = False        # clamp the conditioned modality + markers clean
    cfg.cond.loss_on_suffix_only = True  # loss only on the noised (target) modality
    cfg.cond.null_strategy = "half"      # dropped conditioning -> 0.5 (markers protected)
    cfg.cond.sample_prompt_len = False
    # Per-example regime probabilities (normalized internally).
    cfg.cond.p_cond_text = 0.35          # text->image
    cfg.cond.p_cond_image = 0.35         # image->text
    cfg.cond.p_joint = 0.30              # joint (both noised)
    cfg.cond.p_uncond = 0.1              # CFG dropout of the conditioning modality

    # ------------------------------------------------------------------
    # Model -- MEDIUM 24 x 1024 (~462M), patch_size 19, + segment embedding
    # ------------------------------------------------------------------
    cfg.model = config_dict.ConfigDict()
    cfg.model.name = "sdt"
    cfg.model.use_flash_attn = True
    cfg.model.self_condition = True
    cfg.model.center_inputs = True

    cfg.model.patch_size = mm_codec.BITS_PER_TOKEN   # 19: one position == one token

    cfg.model.embed_dim = 1024
    cfg.model.dim_ff = 4096
    cfg.model.n_blocks = 24
    cfg.model.n_heads = 16

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
    cfg.train.use_compile = True
    cfg.train.compile_mode = "default"
    cfg.train.use_fp16 = True
    cfg.train.amp_dtype = "bf16"
    cfg.train.allow_tf32 = True
    cfg.train.loss_type = "binary_sm"
    cfg.train.loss_weighting = "edm"

    cfg.train.batch_size = 512
    cfg.train.epochs = 400
    cfg.train.ema_decay = 0.9999
    cfg.train.sigma_sampling_strategy = "log-normal"
    cfg.train.self_condition_prob = 0.5

    # Entropy schedule (same recipe as ImageNet/OWT medium).
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

    # Text-only callbacks off.
    cfg.train.generation = config_dict.ConfigDict()
    cfg.train.generation.enabled = False
    cfg.train.external_ppl = config_dict.ConfigDict()
    cfg.train.external_ppl.enabled = False
    cfg.train.mauve = config_dict.ConfigDict()
    cfg.train.mauve.enabled = False

    # ------------------------------------------------------------------
    # In-training visualization: three multimodal panels each epoch.
    #   joint -> sample both (image grid captioned by co-sampled text)
    #   t2i   -> clamp held-out text+markers, sample image
    #   i2t   -> clamp held-out image+markers, sample text (image+caption table)
    # The visualization callback dispatches on dataset.is_multimodal.
    # ------------------------------------------------------------------
    cfg.train.visualization = config_dict.ConfigDict()
    cfg.train.visualization.enabled = True
    cfg.train.visualization.every_k_epochs = 4
    cfg.train.visualization.splits = ["val"]
    cfg.train.visualization.num_samples = 16
    cfg.train.visualization.num_sampling_steps = 128
    cfg.train.visualization.samplers = ["ddim_entropic"]
    cfg.train.visualization.terminal_sigmas = [0.08]
    # No guidance for any branch: t2i/i2t use gs=1.0 (pure conditional, no CFG
    # amplification); joint stays unconditional (the callback forces gs=0 there).
    cfg.train.visualization.guidance_scales = [1.0]
    cfg.train.visualization.entropic_blend_alpha = 0.0
    cfg.train.visualization.entropy_ckpt_path = None
    cfg.train.visualization.micro_batch_size = 16
    cfg.train.visualization.sc_refresh_mode = "carry"
    cfg.train.visualization.sigma_max = None
    cfg.train.visualization.save_to_disk = True
    cfg.train.visualization.mm_tasks = ["joint", "t2i", "i2t"]   # read by the MM viz branch
    # Render each task with BOTH solvers: "deterministic" = DDIM (no churn),
    # "stochastic" = same DDIM path with EDM churn on (cfg.evaluation.stochastic,
    # flipped on per-pass by the callback). Outputs are tagged multimodal/<solver>/<task>.
    cfg.train.visualization.mm_solvers = ["deterministic", "stochastic"]

    cfg.train.vlb = config_dict.ConfigDict()
    cfg.train.vlb.enabled = True
    cfg.train.vlb.every_k_epochs = 2
    cfg.train.vlb.batch_size = 32
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
    # The VLB code assumes a leading-run prefix; the multimodal mask is not a
    # leading run, so evaluate the unconditional (joint) bound only for now.
    cfg.train.vlb.allow_conditional_clean_prefix = False
    cfg.train.vlb.force_unconditional_path = True

    # ------------------------------------------------------------------
    # Optimizer / scheduler (same as ImageNet/OWT medium)
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
    cfg.optim.total_steps = 1_000_000
    cfg.optim.warmup = 5_000

    # ------------------------------------------------------------------
    # Smoke mode (env-driven)
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
        # Smoke-only knobs so a shakedown fits on a single GPU (e.g. one A6000).
        # Production (SMOKE_MAX_STEPS unset) is untouched: batch 512, compile+flash on.
        cfg.train.batch_size = int(os.environ.get("SMOKE_BATCH", "16"))
        cfg.train.use_compile = bool(int(os.environ.get("SMOKE_COMPILE", "1")))
        cfg.model.use_flash_attn = bool(int(os.environ.get("SMOKE_FLASH", "1")))

    # ------------------------------------------------------------------
    # Evaluation -- multimodal both-way + joint
    # ------------------------------------------------------------------
    _eval_ckpt_step = str(os.environ.get("EVAL_CKPT_STEP", "last"))
    if _eval_ckpt_step in ("last", "best"):
        _eval_ckpt_name = f"{_eval_ckpt_step}.pt"
        _eval_tag = f"evaluation_{_eval_ckpt_step}"
    else:
        _eval_ckpt_name = f"step={_eval_ckpt_step}.pt"
        _eval_tag = f"evaluation_step{_eval_ckpt_step}"
    # MM_TAG isolates concurrent single-guidance jobs (each writes its own
    # results.csv / samples dir), so a 4-GPU guidance sweep cannot interleave rows.
    _mm_tag = str(os.environ.get("MM_TAG", "")).strip()
    if _mm_tag:
        _eval_tag = f"{_eval_tag}/{_mm_tag}"

    cfg.evaluation = config_dict.ConfigDict()
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/{_eval_ckpt_name}"
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/{_eval_tag}"
    cfg.evaluation.samples_dir = f"runs/{cfg.experiment}/{_eval_tag}/samples"
    cfg.evaluation.results_csv = f"runs/{cfg.experiment}/{_eval_tag}/results.csv"

    cfg.evaluation.use_amp = True
    cfg.evaluation.amp_dtype = "bf16"
    cfg.evaluation.num_sampling_steps = 256
    cfg.evaluation.use_compile = True
    cfg.evaluation.compile_mode = "default"
    cfg.evaluation.compile = config_dict.ConfigDict()
    cfg.evaluation.compile.warmup = True
    cfg.evaluation.compile.warmup_steps = 8

    # Multimodal eval block (consumed by evaluation/evaluation_drivers/multimodal.py).
    #
    # Env overrides (same style as EVAL_CKPT_STEP above) so one launcher can run
    # the guidance sweep and the final 30K pass without editing this file:
    #   MM_TASKS="t2i"            comma-separated subset of {t2i,i2t,joint,clip_diag,probe}
    #   MM_STEPS=256              reverse-integration steps (NFE = 2x this with CFG on)
    #   MM_BS=64                  sampling batch size
    #   MM_T2I_N=30000            #captions -> #generated images for FID
    #   MM_T2I_GS="1.0,1.5"       CFG scales to run for t2i (one FID row each)
    #   MM_I2T_N / MM_I2T_GS      same for the captioning direction
    #   MM_JOINT_N                unconditional pair count
    #   MM_TAG=gs1.5              appended to the eval output dir, so parallel
    #                             single-scale jobs never collide on results.csv
    def _env_list(name, default):
        raw = os.environ.get(name)
        if raw is None:
            return default
        return [float(x) for x in raw.replace(" ", "").split(",") if x]

    cfg.evaluation.multimodal = config_dict.ConfigDict()
    cfg.evaluation.multimodal.enabled = True
    cfg.evaluation.multimodal.tasks = [
        t for t in str(os.environ.get("MM_TASKS", "t2i,i2t,joint")).split(",") if t
    ]
    # "ddim_entropic" = DDIM on the entropy-derived sigma schedule (needs
    # entropy_{pdf,cdf,sigmas}.pt in the run dir; this codebase RAISES if they
    # are absent rather than falling back to a Karras grid behind a warning,
    # which is what made a missing table a silent accuracy regression before). "ddim_karras" = the SAME DDIM solver on the Karras schedule, which is
    # the clean control for isolating the schedule's effect. Overridable so the
    # schedule can be ablated per direction: it is immaterial for t2i (paired, same
    # seeds: FID 28.036 vs 28.061 at gs=4) but the i2t comparison had only ever been
    # run at the degenerate deterministic/gs=1 cell, where 14% garbage tokens floor
    # every n-gram metric and would mask a real difference.
    cfg.evaluation.multimodal.sampler = str(os.environ.get("MM_SAMPLER", "ddim_entropic"))
    cfg.evaluation.multimodal.num_sampling_steps = int(os.environ.get("MM_STEPS", 256))
    # Stop the reverse integration at log10(sigma) = -1 (sigma=0.1): the model is poorly
    # trained below this (the sqrt-rate/entropy schedule shifts training-sigma mass to higher
    # sigma), so integrating into the untrained small-sigma tail only adds error.
    cfg.evaluation.multimodal.sigma_decode = 0.1
    cfg.evaluation.multimodal.batch_size = int(os.environ.get("MM_BS", 64))
    # ---- solver: per-task EDM churn rate (gamma per step; 0 = deterministic ODE) ----
    # From the CC3M probe at step 750k (probe_step750000_512nfe, N=1024), the two
    # directions want DIFFERENT samplers:
    #   t2i  CMMD:  18.74 deterministic | 22.42 gamma=0.1 | 23.67 gamma=0.175
    #               -> churn HURTS images, monotonically. Deterministic.
    #   i2t  align: 23.49 deterministic | 24.28 gamma=0.1 | 24.38 gamma=0.175
    #               -> churn HELPS captions (GT ceiling 27.09). Use 0.175.
    # The driver converts these to s_churn = gamma*(num_steps-1) at the ACTUAL step
    # count; the static cfg.evaluation.stochastic.s_churn below is the 128-step viz
    # constant and must NOT be reused at 256 steps (it would give gamma ~= 0.086).
    cfg.evaluation.multimodal.t2i_gamma = float(os.environ.get("MM_T2I_GAMMA", 0.0))
    cfg.evaluation.multimodal.i2t_gamma = float(os.environ.get("MM_I2T_GAMMA", 0.175))
    cfg.evaluation.multimodal.caption_gamma = float(os.environ.get("MM_CAPTION_GAMMA", 0.175))
    cfg.evaluation.multimodal.joint_gamma = float(os.environ.get("MM_JOINT_GAMMA", 0.0))
    # Self-conditioning refresh: 'carry' (1 net eval/step, what the probe reports on
    # Drive used) or 'refined' (2/step, the default everywhere else in this repo --
    # generation_driver.py and nfe.py). Sweepable so the choice is measured, not assumed.
    cfg.evaluation.multimodal.sc_refresh_mode = str(os.environ.get("MM_SC_REFRESH", "carry"))
    # Secondary CLIP backbones, scored on the same generated images in the same
    # pass. Our primary is ViT-B/32; published CLIP scores are typically ViT-L/14
    # and the scales differ, so a B/32 number cannot be placed beside theirs.
    # Format: "MODEL/PRETRAINED", comma-separated via MM_CLIP_EXTRA.
    cfg.evaluation.multimodal.clip_extra = [
        x for x in str(os.environ.get("MM_CLIP_EXTRA", "ViT-L-14/openai")).split(",") if x
    ]
    # T2I (zero-shot COCO-30K): captions -> images, FID vs COCO real + CLIP.
    cfg.evaluation.multimodal.t2i_num_samples = int(os.environ.get("MM_T2I_N", 30_000))
    cfg.evaluation.multimodal.t2i_guidance_scales = _env_list(
        "MM_T2I_GS", [1.0, 1.5, 2.0, 3.0])
    cfg.evaluation.multimodal.coco_captions_path = "datasets/coco30k/captions_val30k.jsonl"
    cfg.evaluation.multimodal.coco_real_stats_path = "datasets/coco30k/inception_stats_val.npz"
    cfg.evaluation.multimodal.coco_images_root = "datasets/coco30k/images_val"
    # Real-COCO CLIP bank for CMMD / SWD / diversity (scripts/multimodal/phase_21).
    # CMMD is ~unbiased in N, so it is the distributional number that stays
    # quotable from a 5K sweep where FID can only rank. Optional: if the file is
    # absent those columns are simply omitted.
    cfg.evaluation.multimodal.coco_clip_bank_path = (
        "datasets/coco30k/coco_clip_bank_ViT-B-32_openai.npz")
    # I2T: COCO images -> captions, CLIP score. Uses the frozen COCO image refs
    # built by phase_20 --encode_i2t (real COCO images -> 19-bit ref rows) so the
    # I2T CLIP score is computed on COCO (UniDiffuser-comparable), not in-domain CC3M.
    cfg.evaluation.multimodal.i2t_num_samples = int(os.environ.get("MM_I2T_N", 5_000))
    cfg.evaluation.multimodal.i2t_guidance_scales = _env_list("MM_I2T_GS", [1.0, 2.0])
    cfg.evaluation.multimodal.coco_i2t_refs_path = "datasets/coco30k/coco_i2t_refs.uint32.npy"
    # Captioning metrics (BLEU-4/METEOR/CIDEr/SPICE) on the COCO Karpathy TEST split
    # -- the only split with published captioning baselines (UniDiffuser, L-Verse,
    # OFA...). Assets from scripts/multimodal/phase_22_karpathy_caption_assets.py.
    # Enable with MM_TASKS=caption (needs java for PTBTokenizer/METEOR).
    cfg.evaluation.multimodal.karpathy_refs_path = "datasets/coco_karpathy/karpathy_test_refs.uint32.npy"
    cfg.evaluation.multimodal.karpathy_ids_path = "datasets/coco_karpathy/karpathy_test_ids.json"
    cfg.evaluation.multimodal.karpathy_gt_path = "datasets/coco_karpathy/karpathy_test_gt.json"
    cfg.evaluation.multimodal.caption_num_samples = int(os.environ.get("MM_CAPTION_N", 5_000))
    cfg.evaluation.multimodal.caption_guidance_scales = _env_list("MM_CAPTION_GS", [1.0])
    cfg.evaluation.multimodal.caption_spice = bool(
        int(os.environ.get("MM_CAPTION_SPICE", "1")))
    # Joint: sample N pairs, FID(image half) + CLIP(pair).
    cfg.evaluation.multimodal.joint_num_samples = int(os.environ.get("MM_JOINT_N", 30_000))
    # CLIP backbone: OpenAI ViT-B/32 (QuickGELU) so absolute CLIP scores are
    # directly comparable to UniDiffuser / SD-lineage reported numbers. The loader
    # forces QuickGELU for the 'openai' tag (evaluation/multimodal.py::_load_clip).
    cfg.evaluation.multimodal.clip_model = "ViT-B-32"
    cfg.evaluation.multimodal.clip_pretrained = "openai"

    # clip_diag: the cheap first-pass eval. On held-out pairs we already have,
    # sample the target modality and compare CLIP embeddings of the generation to
    # the ground truth (+ GT ceiling). No FID / COCO assets needed.
    #   run via:  python -m evaluation.run_eval --config <cfg> --metrics multimodal \
    #             --mm_tasks clip_diag [--mm_num_samples N]
    cfg.evaluation.multimodal.clip_diag = config_dict.ConfigDict()
    cfg.evaluation.multimodal.clip_diag.num_samples = 512
    cfg.evaluation.multimodal.clip_diag.subtasks = ["t2i", "i2t"]
    cfg.evaluation.multimodal.clip_diag.guidance_scales = [1.0]
    cfg.evaluation.multimodal.clip_diag.use_coco = False   # lights up once phase_20 COCO assets exist

    # probe: quick single-checkpoint performance probe across the full matrix
    # (paths x guidance x solvers) with qualitative dumps + CLIP gen-vs-GT metrics.
    #   run via:  --metrics multimodal --mm_tasks probe   (2-GPU launcher splits solvers)
    cfg.evaluation.multimodal.probe = config_dict.ConfigDict()
    cfg.evaluation.multimodal.probe.tasks = ["t2i", "i2t", "joint"]
    cfg.evaluation.multimodal.probe.guidance_scales = [1.0, 2.0, 3.0, 4.0]   # 1.0 = no CFG amplification
    cfg.evaluation.multimodal.probe.solvers = ["deterministic", "stochastic"]
    # Stochastic gamma sweep: run a stochastic unit per gamma (0.175 = OWT-validated,
    # 0.1 = smaller churn to probe the image-quality vs text-fluency tradeoff).
    cfg.evaluation.multimodal.probe.gamma_targets = [0.175, 0.1]
    cfg.evaluation.multimodal.probe.num_samples = 1024       # for CLIP/CMMD/SWD/fluency metrics (still stable at ~1k)
    cfg.evaluation.multimodal.probe.num_sampling_steps = 256  # nominal steps -> 512 NFE (guided). Run at BS=128 (256-step live-mem growth under compile)
    cfg.evaluation.multimodal.probe.lm_model = "distilgpt2"  # external LM for caption perplexity
    cfg.evaluation.multimodal.probe.real_bank_path = None    # default: datasets/<root>/clip_real_bank_<clipid>.npz
    # Per-step EDM churn target for the stochastic solver. The probe pins
    # s_churn = gamma_target*(num_steps-1) at runtime, so this gamma (OWT-validated
    # ~0.175 @ 256 steps) transfers at whatever num_sampling_steps the probe uses.
    cfg.evaluation.multimodal.probe.gamma_target = 0.175
    cfg.evaluation.multimodal.probe.batch_size = 256        # sampling micro-batch
    cfg.evaluation.multimodal.probe.decode_batch_size = 64  # Open-MAGVIT2 decode chunk (decoder OOMs at 256)
    # Qualitative viewer: a FIXED random subset (same indices across every case, so
    # the same GT pair drives t2i and i2t at each guidance/solver -> comparable).
    cfg.evaluation.multimodal.probe.qual_examples = 16
    cfg.evaluation.multimodal.probe.qual_seed = 0
    # Frozen eval protocol id (see scripts/multimodal/EVAL_PROTOCOL.md + the
    # protocol.json that phase_20 writes alongside the COCO assets).
    cfg.evaluation.multimodal.protocol = "zeroshot_coco_fid30k_v1.0"

    # Stochastic (EDM-churn) sampler config. Kept DISABLED here so the standalone
    # eval driver stays deterministic; the in-training viz callback flips
    # `enabled` True for its "stochastic" solver pass only (legacy derivation:
    # enabled + s_churn>0 -> stochastic_mode "edm_churn"). s_churn=22 over 128
    # steps -> gamma ~= 0.175 (the proven SDT churn rate); window "full" needs no
    # entropy tables.
    cfg.evaluation.stochastic = config_dict.ConfigDict()
    cfg.evaluation.stochastic.enabled = False
    cfg.evaluation.stochastic.s_churn = 22.0
    cfg.evaluation.stochastic.s_noise = 1.003
    cfg.evaluation.stochastic.window_mode = "full"
    cfg.evaluation.stochastic.s_tmin = None
    cfg.evaluation.stochastic.s_tmax = None

    # Classifier-free-guidance schedule (per-sigma weight). "constant" = legacy
    # (byte-identical default). The entropy modes use the run-dir entropy tables
    # (entropy_cdf.pt / entropy_pdf.pt / entropy_sigmas.pt):
    #   "entropy_band"      -> guide with base_w only inside a CONTIGUOUS sigma-band
    #                          [sigma_lo,sigma_hi] mapped from a band of the entropy
    #                          CDF (entropy_quantile_lo/hi), weight 1.0 outside.
    #                          Reuses the churn entropy_cdf quantile->sigma resolver.
    #   "entropy_modulated" -> smooth w(sigma)=1+(base_w-1)*sqrt(pi_hat(sigma)).
    #   "interval"          -> legacy quantile-THRESHOLD on the density (back-compat).
    cfg.evaluation.guidance = config_dict.ConfigDict()
    cfg.evaluation.guidance.mode = "constant"          # constant | entropy_band | entropy_modulated | interval
    cfg.evaluation.guidance.sqrt_entropy_rate = True
    # entropy_band: central-mass CDF quantiles -> [sigma_lo, sigma_hi] (0.10/0.90 = central 80%).
    cfg.evaluation.guidance.entropy_quantile_lo = 0.10
    cfg.evaluation.guidance.entropy_quantile_hi = 0.90
    cfg.evaluation.guidance.interval_quantile = 0.5    # legacy interval mode: guide where pi_hat >= this quantile

    # Text-only eval drivers off.
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
    cfg.logging.project = "multimodal_cc3m"
    cfg.logging.group = "cc3m_lfq19_medium_24x1024_joint"
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
