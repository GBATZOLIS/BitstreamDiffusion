#!/bin/bash
# pass@k / maj@k runner for the TinyGSM->GSM8K baselines.
#   ALGO   : mdlm | duo-base | sfm | ar | flm ...   (hydra algo= key)
#   CKPT   : path to .ckpt
#   TEMP   : sampler temperature (1.0 = standard, 0.1 = their optimized low-T)
#   K      : samples per problem (MUST be matched across methods)
#   STEPS  : sampling steps (1024 = paper protocol)
#   DATA   : gsm8k test json (default: the SAME 1319-problem file CoBit is scored on)
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY=${PY:-$HOME/miniconda3/envs/sfm/bin/python}
ALGO=${ALGO:-duo-base}
NAME=${NAME:-${ALGO}}
CKPT=${CKPT:-${REPO_ROOT}/checkpoints/tinygsm/duo.ckpt}
TEMP=${TEMP:-1.0}
K=${K:-32}
STEPS=${STEPS:-1024}
BS=${BS:-32}
DATA=${DATA:-${REPO_ROOT}/data_gsm8k_test_full.json}
CACHE=${CACHE:-${REPO_ROOT}/data_cache}
OUT=${OUT:-${REPO_ROOT}/eval_runs/passk/${NAME}_T${TEMP}_K${K}_s${STEPS}}
DEVICES=${DEVICES:-1}
cd "${REPO_ROOT}"
mkdir -p "${OUT}"
"${PY}" -u -m main \
    mode=gsm8k_eval \
    eval.checkpoint_path="${CKPT}" \
    data=gsm8k-test \
    data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir="${CACHE}" \
    data.data_path="${DATA}" \
    model=small \
    model.length=512 \
    algo="${ALGO}" \
    sampler=ancestral \
    sampler.steps="${STEPS}" \
    sampler.temperature="${TEMP}" \
    loader.eval_batch_size="${BS}" \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices="${DEVICES}" \
    gsm8k.num_samples="${K}" \
    gsm8k.output_dir="${OUT}" \
    +wandb.offline=True \
    hydra.run.dir="${OUT}"
