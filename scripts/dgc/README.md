# DGC sampling grid for CoBit (experimental)

This folder tests the step allocation from Wainwright's DGC analysis
([arXiv:2607.26285](https://arxiv.org/abs/2607.26285)) as CoBit's sampling σ grid.
Only the grid changes. The released CoBit-S LM1B checkpoint, the stochastic sampler
(S_noise 1.003, full-band churn, ati_eta 0, terminal σ 10^-1.10), 1024 samples and the
GenPPL / unigram-entropy protocol are those of the paper's Table 1.

Status: exploratory, one sampling seed (42), not part of the paper's results.

## The grid

CoBit's entropy-rate table has density in u = log σ

    p(u) ∝ g(σ) · (h(σ)/σ²)^α,      g(σ) = σⁿ / (σⁿ + cⁿ),   α = ½, c = 0.1, n = 3,

with h the per-bit denoising MSE. DGC places steps with density ∝ √h'(t) in log t for the
heat time t = σ², which in u is

    p_DGC(u) ∝ √( (dh/du) / σ² ).

`build_dgc_tables.py` recovers h from the shipped table (h ∝ σ² (p/g)^(1/α)), smooths it,
differentiates it and writes the DGC table on the same bins:

    python scripts/dgc/build_dgc_tables.py assets/entropy_tables/lm1b assets/entropy_tables/lm1b
    # writes assets/entropy_tables/lm1b_dgc/ (bit for bit the committed table)

With the same S_churn, the per-step churn γ is the same on both grids, but the effective
Langevin strength λ(σ) ≈ S_churn · p(log σ) (Appendix G, Eq. 48) follows the grid, so the DGC
grid moves stochastic correction toward low σ as well as moving the solver steps.

## Running

`configs/lm1b/continuous/eval/dgc_schedule_ab.py` reads these environment variables:

| variable | meaning |
|---|---|
| `SCHEDULE` | `cobit` (shipped table, default) or `dgc` |
| `EVAL_SEED` | sampling seed (default 42); the same seed gives the same initial noise in both arms |
| `NFES`, `GAMMA` | NFE budgets and per-step churn (default `256`, `0.200`) |
| `DET` | `1` also runs the deterministic sampler (default) |
| `CHURN_GRID` | per-NFE S_churn values, e.g. `64:1.5,3,5/256:20,30,40`; γ = S_churn/(NFE−1), capped at √2−1 by the sampler; overrides `NFES` and `GAMMA` |

Paired A/B at the Table 1 operating point (needs the released checkpoint and the LM1B caches,
as for the Table 1 evaluation):

    SEEDS="42" ARMS="cobit dgc" bash scripts/dgc/run_ab.sh

A churn sweep on the DGC grid:

    export SCHEDULE=dgc EVAL_SEED=42 DET=1 CHURN_GRID="64:1.5,3,5/96:3,6,10/128:4,7,11/192:12,20,30/256:20,30,40,64,77,90,102/384:8,16,28,40,70,100,140,157/512:2,5,10,30,60,100,150,179,197,211"
    torchrun --standalone --nproc_per_node=1 -m evaluation.run_eval \
      --config configs/lm1b/continuous/eval/dgc_schedule_ab.py --metrics external_ppl
    python -m evaluation.compute_entropy_from_caches \
      --config configs/lm1b/continuous/eval/dgc_schedule_ab.py --include_real

## Summaries and figures

    python scripts/dgc/summarize_ab.py runs                          # mean ± sd over seeds, paired differences vs cobit
    python scripts/dgc/plot_lm1b_dgc_sweep.py OUT runs                # every DGC point next to the paper's Figure 2 -> OUT.png/.pdf/.csv
    python scripts/dgc/plot_lm1b_dgc_fronts.py OUT.csv FIG_DIR       # two front figures (up to NFE 384, and up to 512)

`lm1b_fig2_points.json` holds the points of the paper's Figure 2 (LM1B GenPPL–entropy),
read from the figure's vector paths; the reference markers reproduce Table 1 to two decimals.
It also holds the same-pipeline A/B at NFE 256, γ 0.200 and the real-data reference used in the
figures.
