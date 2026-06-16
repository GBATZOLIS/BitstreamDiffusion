# configs/owt/eval_cobit_m_750K.py
#
# Evaluation config for the released CoBit-M (462M) OpenWebText checkpoint
# (step=000750000, EMA). Reproduces the CoBit-M operating points reported in
# Table 2 of the paper, plus the lower-perplexity 256-NFE point named in the
# caption.
#
#   Setting    gamma   s_churn   GenPPL   Entropy   Table 2 row
#   --------------------------------------------------------------------
#   256 NFE    0.21    53.55     19.48    5.40      CoBit-M, 256 NFE (plotted)
#   256 NFE    0.13    33.15     18.47    5.378     caption (low-PPL 256 point)
#   384 NFE    0.24    91.92     13.06    5.33      CoBit-M, 384 NFE
#   512 NFE    0.26   132.86      9.87    5.25      CoBit-M, 512 NFE
#   --------------------------------------------------------------------
#   Real OWT reference (computed alongside):         15.07    5.44
#
# GenPPL is GPT-2-Large perplexity; entropy is GPT-2-token unigram entropy.
# Both follow the same protocol as Table 1 (Section 4.1). All cells use
# N=1024 samples and seed=42. The reported numbers were produced on GH200;
# expect agreement to within seed/hardware noise (Table-1-style ±0.5 GenPPL).
# The per-step churn ceiling note: s_churn = gamma*(N-1), and the edm_churn
# sampler clamps per-step gamma at min(s_churn/N, sqrt(2)-1≈0.4142), so the
# effective per-step gamma equals gamma_target for every cell here.
#
# Architecture / data / diffusion / codec MUST be byte-identical to the
# training config configs/owt/rate_bits_edm_weight_medium_24x1024.py — the
# checkpoint will not load otherwise.
#
# Submission-time overrides (no edit needed):
#   EVAL_CKPT_STEP=000700000 <launcher>     # evaluate a different step
#   EVAL_SEED=43             <launcher>     # different sampling seed
#   EVAL_CELLS=table2|low_ppl|all           # which sampling cells to run
#       table2  (default) — the three plotted Table-2 rows (256/384/512)
#       low_ppl           — the caption 256-NFE gamma=0.13 point only
#       all               — table2 + low_ppl

import os
from ml_collections import config_dict


def get_config():
    cfg = config_dict.ConfigDict()

    _eval_ckpt_step = str(os.environ.get("EVAL_CKPT_STEP", "000750000"))
    _eval_seed = int(os.environ.get("EVAL_SEED", 42))
    _eval_cells = str(os.environ.get("EVAL_CELLS", "table2")).lower()
    # Sample count — the released numbers use 1024. Lower it only for a quick
    # smoke test (EVAL_NUM_SAMPLES=16); the reported GenPPL/entropy require 1024.
    _eval_num_samples = int(os.environ.get("EVAL_NUM_SAMPLES", 1024))
    _eval_ckpt_name = f"step={_eval_ckpt_step}.pt"
    # Optional suffix to isolate output dirs when running several processes
    # concurrently (e.g. one single-GPU process per GPU over disjoint cells).
    _eval_out_suffix = str(os.environ.get("EVAL_OUT_SUFFIX", ""))
    _eval_tag = f"evaluation_cobit_m_table2_step{_eval_ckpt_step}{_eval_out_suffix}"

    # ------------------------------------------------------------------
    # Framework / experiment  (must match training)
    # ------------------------------------------------------------------
    cfg.framework = "continuous_score"
    cfg.experiment = "paper/unconditional_text/owt/continuous_rate_raw_binary_bits_medium_24x1024"
    cfg.device = "cuda"

    # ------------------------------------------------------------------
    # Data  (must match training)
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
    cfg.data.num_workers = 6
    cfg.data.prefetch_factor = 4
    cfg.data.pin_memory = True

    # ------------------------------------------------------------------
    # Conditioning  (unconditional benchmark)
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
    # Model  ── MEDIUM 24 × 1024  (≈462M)  — must match training
    # ------------------------------------------------------------------
    cfg.model = config_dict.ConfigDict()
    cfg.model.name = "sdt"
    cfg.model.use_flash_attn = True
    cfg.model.self_condition = True
    cfg.model.center_inputs = True
    cfg.model.patch_size = 16

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
    # Continuous diffusion  (must match training)
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
    # Train block  (only fields run_eval reads at eval time)
    # ------------------------------------------------------------------
    cfg.train = config_dict.ConfigDict()
    cfg.train.seed = _eval_seed
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
    # Evaluation
    # ------------------------------------------------------------------
    cfg.evaluation = config_dict.ConfigDict()
    cfg.evaluation.checkpoint_path = f"runs/{cfg.experiment}/checkpoints/{_eval_ckpt_name}"
    # Medium-model entropy-rate schedule artefacts — distinct from the CoBit-S
    # tables under assets/entropy_tables/owt/.
    cfg.evaluation.entropy_run_dir = "assets/entropy_tables/owt_medium"

    _out_base = f"runs/{cfg.experiment}/{_eval_tag}"
    cfg.evaluation.out_dir = _out_base
    cfg.evaluation.samples_dir = f"{_out_base}/samples"
    cfg.evaluation.results_csv = f"{_out_base}/results.csv"
    cfg.evaluation.shared_text_cache_dir = f"{_out_base}/shared_text_cache"

    cfg.evaluation.use_amp = True
    cfg.evaluation.amp_dtype = "bf16"
    cfg.evaluation.use_compile = True
    cfg.evaluation.compile_mode = "default"

    cfg.evaluation.compile = config_dict.ConfigDict()
    cfg.evaluation.compile.warmup = True
    cfg.evaluation.compile.warmup_steps = 8

    cfg.evaluation.ati = config_dict.ConfigDict()
    cfg.evaluation.ati.enabled = False
    cfg.evaluation.ati.eta = 0.0

    # Inert default — GenerationDriver overwrites per-spec.
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
    # External PPL config
    # ------------------------------------------------------------------
    cfg.evaluation.external_ppl = config_dict.ConfigDict()
    cfg.evaluation.external_ppl.enabled = True
    cfg.evaluation.external_ppl.backend = "hf_causal_lm"
    cfg.evaluation.external_ppl.hf_model_name = "openai-community/gpt2-large"
    cfg.evaluation.external_ppl.hf_revision = None
    cfg.evaluation.external_ppl.hf_dtype = "bfloat16"
    cfg.evaluation.external_ppl.attn_implementation = "sdpa"
    cfg.evaluation.external_ppl.num_samples = _eval_num_samples
    cfg.evaluation.external_ppl.micro_batch_size = 64
    cfg.evaluation.external_ppl.samplers = ["ddim_entropic"]
    cfg.evaluation.external_ppl.terminal_sigmas = [0.08]
    cfg.evaluation.external_ppl.guidance_scales = [0.0]
    cfg.evaluation.external_ppl.num_sampling_steps = 255  # fallback; sampling_sweep wins per spec
    cfg.evaluation.external_ppl.sc_refresh_mode = "carry"
    cfg.evaluation.external_ppl.sigma_max = None
    cfg.evaluation.external_ppl.use_amp = True
    cfg.evaluation.external_ppl.decode_mode = "full"
    cfg.evaluation.external_ppl.score_mode = "full"
    cfg.evaluation.external_ppl.compute_real_reference = True
    cfg.evaluation.external_ppl.shared_cache_dir = cfg.evaluation.shared_text_cache_dir
    cfg.evaluation.external_ppl.seed = _eval_seed
    cfg.evaluation.external_ppl.checkpoints = [_eval_ckpt_name]
    cfg.evaluation.external_ppl.entropy_enabled = True
    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_decode = True
    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_max_rows = 8
    cfg.evaluation.external_ppl.debug_owt_gpt2id_bpe16_once = True

    cfg.evaluation.mauve = config_dict.ConfigDict()
    cfg.evaluation.mauve.enabled = False

    # ------------------------------------------------------------------
    # NFE x gamma sweep — exactly the Table-2 CoBit-M operating points.
    # s_churn = gamma * (target_nfe - 1).
    # ------------------------------------------------------------------
    def _make_spec(gamma: float, target_nfe: int) -> "config_dict.ConfigDict":
        s = config_dict.ConfigDict()
        s.sampler_name = "ddim_entropic"
        s.sc_refresh_modes = ["carry"]
        s.target_nfes = [int(target_nfe)]
        s.ati_etas = [0.0]
        s.stochastic_enabled = True
        s.stochastic_mode = "edm_churn"
        s.s_churn = float(gamma * (int(target_nfe) - 1))
        s.s_noise = 1.003
        s.window_mode = "entropy_cdf"
        s.entropy_quantile_lo = 0.0
        s.entropy_quantile_hi = 1.0
        s.entropy_fallback = "deterministic"
        s.s_tmin = None
        s.s_tmax = None
        s.gamma_target = float(gamma)
        return s

    # (target_nfe, gamma) -> Table-2 cell
    _table2_cells = [
        (256, 0.21),   # GenPPL 19.48, H 5.40  (plotted near-real-entropy point)
        (384, 0.24),   # GenPPL 13.06, H 5.33
        (512, 0.26),   # GenPPL  9.87, H 5.25
    ]
    _low_ppl_cells = [
        (256, 0.13),   # GenPPL 18.47, H 5.378 (caption low-PPL 256-NFE point)
    ]
    if _eval_cells == "table2":
        _cells = _table2_cells
    elif _eval_cells == "low_ppl":
        _cells = _low_ppl_cells
    elif _eval_cells == "all":
        _cells = _table2_cells + _low_ppl_cells
    elif ":" in _eval_cells:
        # Explicit per-cell list, e.g. EVAL_CELLS="256:0.21,512:0.26".
        # Useful to split cells across GPUs (one single-GPU process per GPU).
        _cells = []
        for tok in _eval_cells.split(","):
            tok = tok.strip()
            if not tok:
                continue
            _nfe_s, _gamma_s = tok.split(":")
            _cells.append((int(_nfe_s), float(_gamma_s)))
    else:
        raise ValueError(
            "EVAL_CELLS must be 'table2', 'low_ppl', 'all', or an explicit "
            f"'nfe:gamma,nfe:gamma' list; got {_eval_cells!r}"
        )

    cfg.evaluation.sampling_sweep = config_dict.ConfigDict()
    cfg.evaluation.sampling_sweep.enabled = True
    cfg.evaluation.sampling_sweep.target_nfes = sorted({n for n, _ in _cells})
    cfg.evaluation.sampling_sweep.specs = []
    for _nfe, _gamma in _cells:
        cfg.evaluation.sampling_sweep.specs.append(_make_spec(_gamma, _nfe))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    cfg.logging = config_dict.ConfigDict()
    cfg.logging.use_wandb = False

    return cfg
