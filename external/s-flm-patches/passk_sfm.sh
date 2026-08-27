#!/bin/bash
# pass@k runner for S-FLM (needs the sphere-arch config family, not the ancestral one).
# Defaults reproduce their best GSM8K variant: top-1 velocity (their Table 2, 18.0%).
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY=${PY:-$HOME/miniconda3/envs/sfm2/bin/python}
CKPT=${CKPT:-${REPO_ROOT}/checkpoints/tinygsm/sfm/sphere_arch_truncated_adaptive_no_renorm.ckpt}
K=${K:-32}; STEPS=${STEPS:-1024}; BS=${BS:-128}
VELOCITY=${VELOCITY:-exact}; TOPK_VELOCITY=${TOPK_VELOCITY:-1}
DATA=${DATA:-${REPO_ROOT}/data_gsm8k_test_full.json}
CACHE=${CACHE:-${REPO_ROOT}/data_cache}
OUT=${OUT:-${REPO_ROOT}/eval_runs/passk/sfm_K${K}_s${STEPS}}
GLOBAL_BS=512; BUF_SIZE=$((50 * GLOBAL_BS))
cd "${REPO_ROOT}"; mkdir -p "${OUT}"
"${PY}" -u -m main \
    mode=gsm8k_eval eval.checkpoint_path="${CKPT}" eval.strict_loading=false \
    data=gsm8k-test data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir="${CACHE}" data.data_path="${DATA}" \
    model=small-sphere-arch model.length=512 model.normalize_input_embed=False \
    algo=sfm algo.renormalize_weights=True algo.invert_time_convention=false \
    noise=log-linear-adaptive noise.alpha_max=0.121 noise.adaptive_refit_every=50 \
    noise.adaptive_buffer_size=${BUF_SIZE} noise.adaptive_ema=0.9 noise.adaptive_uniform_mix=1e-3 \
    sampler=sfm sampler.noise_removal=greedy sampler.velocity="${VELOCITY}" \
    sampler.top_k_velocity="${TOPK_VELOCITY}" sampler.steps="${STEPS}" \
    loader.eval_batch_size="${BS}" loader.num_workers=4 \
    trainer.num_nodes=1 trainer.devices=1 \
    gsm8k.num_samples="${K}" gsm8k.output_dir="${OUT}" \
    +wandb.offline=True hydra.run.dir="${OUT}"
