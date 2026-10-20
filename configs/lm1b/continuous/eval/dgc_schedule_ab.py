"""CoBit-S LM1B: entropy-rate schedule vs DGC schedule at fixed NFE and gamma.

Inherits everything from rate_eval_seeds.py (released checkpoint, Table 1 settings)
and changes ONLY the sigma grid, by pointing entropy_run_dir at a different table.

Env vars:
  SCHEDULE   cobit (default, shipped table) | dgc (the DGC grid, scripts/dgc/build_dgc_tables.py)
  EVAL_SEED  seed (default 42). Same seed across arms => same initial noise and churn draws.
  NFES       comma list of NFE budgets (default "256")
  GAMMA      churn gamma (default 0.200, the paper's LM1B headline)
  DET        "1" to also run the deterministic (no-churn) arm (default "1")
  CHURN_GRID per-NFE S_churn values, e.g. "64:2,4,7/256:25.5,38.25,51" (S_churn = gamma*(NFE-1),
             the units of CoBit's own sweep configs). Overrides NFES and GAMMA when set.

Churn window: full band, q in [0, 1] of each table's entropy CDF, as in the paper's Table 1 runs;
with equal S_churn the per-step gamma is the same in both arms and only the grid positions differ.
"""
import os

from configs.lm1b.continuous.eval.rate_eval_seeds import get_config as get_base_config
from ml_collections import config_dict

TABLES = {
    "cobit": "assets/entropy_tables/lm1b",
    "dgc": "assets/entropy_tables/lm1b_dgc",
}


def get_config():
    cfg = get_base_config()

    schedule = os.environ.get("SCHEDULE", "cobit").strip().lower()
    if schedule not in TABLES:
        raise ValueError(f"SCHEDULE must be one of {sorted(TABLES)}, got {schedule!r}")
    seed = int(os.environ.get("EVAL_SEED", 42))
    nfes = [int(x) for x in os.environ.get("NFES", "256").split(",") if x.strip()]
    gamma = float(os.environ.get("GAMMA", "0.200"))
    run_det = os.environ.get("DET", "1") == "1"

    cfg.evaluation.entropy_run_dir = TABLES[schedule]

    # Separate output + sample cache per arm and seed so nothing is shared or overwritten.
    cfg.evaluation.out_dir = f"runs/{cfg.experiment}/dgc_ab/{schedule}/seed{seed}"
    cfg.evaluation.samples_dir = f"{cfg.evaluation.out_dir}/samples"
    cfg.evaluation.results_csv = f"{cfg.evaluation.out_dir}/results.csv"
    cfg.evaluation.shared_text_cache_dir = f"{cfg.evaluation.out_dir}/shared_text_cache"
    cfg.evaluation.external_ppl.shared_cache_dir = cfg.evaluation.shared_text_cache_dir
    cfg.evaluation.external_ppl.seed = seed
    cfg.train.seed = seed
    if os.environ.get("DGC_NUM_SAMPLES"):  # smoke runs only; the A/B itself uses the inherited 1024
        n = int(os.environ["DGC_NUM_SAMPLES"])
        cfg.evaluation.external_ppl.num_samples = n
        cfg.evaluation.external_ppl.micro_batch_size = min(n, cfg.evaluation.external_ppl.micro_batch_size)

    cfg.evaluation.sampling_sweep.target_nfes = list(nfes)
    cfg.evaluation.sampling_sweep.specs = []

    def add_spec(*, target_nfe, stochastic_enabled, gamma_target=None):
        spec = config_dict.ConfigDict()
        spec.sampler_name = "ddim_entropic"  # -> schedule "entropic" = inverse CDF of the table
        spec.sc_refresh_modes = ["carry"]
        spec.target_nfes = [int(target_nfe)]
        spec.ati_etas = [0.0]
        spec.stochastic_enabled = bool(stochastic_enabled)
        if stochastic_enabled:
            spec.s_churn = float(gamma_target * max(1, int(target_nfe) - 1))
            spec.s_noise = 1.003
            spec.window_mode = "entropy_cdf"
            spec.entropy_quantile_lo = 0.0
            spec.entropy_quantile_hi = 1.0
            spec.entropy_fallback = "deterministic"
            spec.s_tmin = None
            spec.s_tmax = None
            spec.gamma_target = float(gamma_target)
        cfg.evaluation.sampling_sweep.specs.append(spec)

    grid = os.environ.get("CHURN_GRID", "").strip()
    if grid:
        per_nfe = {}
        for item in grid.split("/"):
            nfe, churns = item.split(":")
            per_nfe[int(nfe)] = [float(c) for c in churns.split(",") if c.strip()]
        cfg.evaluation.sampling_sweep.target_nfes = sorted(per_nfe)
        for nfe in sorted(per_nfe):
            for s_churn in per_nfe[nfe]:
                add_spec(target_nfe=nfe, stochastic_enabled=True, gamma_target=s_churn / max(1, nfe - 1))
            if run_det:
                add_spec(target_nfe=nfe, stochastic_enabled=False)
        return cfg

    for nfe in nfes:
        add_spec(target_nfe=nfe, stochastic_enabled=True, gamma_target=gamma)
        if run_det:
            add_spec(target_nfe=nfe, stochastic_enabled=False)

    return cfg
