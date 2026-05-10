from ml_collections import config_dict

from configs.lm1b.continuous.rate_bits_1M_edm_weight import get_config as get_base_config


def get_config():
    cfg = get_base_config()

    # ------------------------------------------------------------------
    # Evaluation identity
    # ------------------------------------------------------------------
    cfg.evaluation.out_dir = (
        f"runs/{cfg.experiment}/external_ppl_ckpt1M_lm1b_churn_window_ablation_nfe256"
    )
    cfg.evaluation.samples_dir = f"{cfg.evaluation.out_dir}/samples"
    cfg.evaluation.results_csv = (
        f"{cfg.evaluation.out_dir}/results_churn_window_ablation_nfe256.csv"
    )
    cfg.evaluation.shared_text_cache_dir = f"{cfg.evaluation.out_dir}/shared_text_cache"
    cfg.evaluation.external_ppl.shared_cache_dir = cfg.evaluation.shared_text_cache_dir

    cfg.evaluation.external_ppl.enabled = True
    cfg.evaluation.external_ppl.num_samples = 1024
    cfg.evaluation.external_ppl.micro_batch_size = 512
    cfg.evaluation.external_ppl.seed = 42
    cfg.evaluation.external_ppl.compute_real_reference = True

    cfg.evaluation.num_sampling_steps = 255
    cfg.evaluation.external_ppl.num_sampling_steps = 255

    # ------------------------------------------------------------------
    # Defaults overridden by sweep specs
    # ------------------------------------------------------------------
    cfg.evaluation.ati = config_dict.ConfigDict()
    cfg.evaluation.ati.enabled = False
    cfg.evaluation.ati.eta = 0.0

    cfg.evaluation.stochastic = config_dict.ConfigDict()
    cfg.evaluation.stochastic.enabled = False
    cfg.evaluation.stochastic.s_churn = 0.0
    cfg.evaluation.stochastic.s_noise = 1.003
    cfg.evaluation.stochastic.window_mode = "entropy_cdf"
    cfg.evaluation.stochastic.entropy_quantile_lo = 0.10
    cfg.evaluation.stochastic.entropy_quantile_hi = 0.90
    cfg.evaluation.stochastic.s_tmin = None
    cfg.evaluation.stochastic.s_tmax = None
    cfg.evaluation.stochastic.entropy_fallback = "deterministic"

    # ------------------------------------------------------------------
    # Sweep: entropy-CDF churn-window ablation at NFE 256.
    # Fixed sampler: ddim_entropic.
    # ------------------------------------------------------------------
    cfg.evaluation.sampling_sweep = config_dict.ConfigDict()
    cfg.evaluation.sampling_sweep.enabled = True
    cfg.evaluation.sampling_sweep.target_nfes = []
    cfg.evaluation.sampling_sweep.specs = []

    def add_spec(
        *,
        target_nfe: int,
        sampler_name: str,
        ati_eta: float,
        stochastic_enabled: bool,
        gamma_target: float | None = None,
        s_noise: float = 1.003,
        qlo: float = 0.10,
        qhi: float = 0.90,
        tag_note: str = "",
    ):
        spec = config_dict.ConfigDict()
        spec.sampler_name = str(sampler_name)
        spec.sc_refresh_modes = ["carry"]
        spec.target_nfes = [int(target_nfe)]
        spec.ati_etas = [float(ati_eta)]
        spec.stochastic_enabled = bool(stochastic_enabled)
        spec.tag_note = str(tag_note)

        if stochastic_enabled:
            if gamma_target is None:
                raise ValueError("gamma_target must be provided when stochastic_enabled=True")

            num_intervals = max(1, int(target_nfe) - 1)
            s_churn = float(gamma_target * num_intervals)

            spec.s_churn = float(s_churn)
            spec.s_noise = float(s_noise)

            # Keep entropy_cdf even for q=[0,1], so q=[0,1] is the full
            # entropy-supported window and is directly comparable.
            spec.window_mode = "entropy_cdf"
            spec.entropy_quantile_lo = float(qlo)
            spec.entropy_quantile_hi = float(qhi)
            spec.entropy_fallback = "deterministic"
            spec.s_tmin = None
            spec.s_tmax = None
            spec.gamma_target = float(gamma_target)

        cfg.evaluation.sampling_sweep.specs.append(spec)

    target_nfe = 256
    eta_val = 0.0
    sampler_name = "ddim_entropic"

    add_spec(
        target_nfe=target_nfe,
        sampler_name=sampler_name,
        ati_eta=eta_val,
        stochastic_enabled=False,
        tag_note="window_ddim_entropic_nfe256_det",
    )

    churn_windows = [
        # Location sweep: fixed 25% entropy-CDF mass.
        (0.00, 0.25, "loc_low_tail_25"),
        (0.10, 0.35, "loc_lower_mid_25"),
        (0.375, 0.625, "loc_center_25"),
        (0.65, 0.90, "loc_upper_mid_25"),
        (0.75, 1.00, "loc_high_tail_25"),

        # Width sweep: centered windows.
        (0.45, 0.55, "width_center_10"),
        (0.40, 0.60, "width_center_20"),
        (0.30, 0.70, "width_center_40"),
        (0.20, 0.80, "width_center_60"),
        (0.10, 0.90, "width_default_80"),

        # Full entropy-supported window.
        (0.00, 1.00, "width_full_100"),
    ]

    gamma_grid = [0.13, 0.15, 0.17, 0.185, 0.20, 0.215]

    for qlo, qhi, window_name in churn_windows:
        for gamma in gamma_grid:
            add_spec(
                target_nfe=target_nfe,
                sampler_name=sampler_name,
                ati_eta=eta_val,
                stochastic_enabled=True,
                gamma_target=gamma,
                s_noise=1.003,
                qlo=qlo,
                qhi=qhi,
                tag_note=(
                    f"window_{window_name}"
                    f"_q{qlo:.3f}_{qhi:.3f}"
                    f"_gamma{gamma}"
                ),
            )

    return cfg