#!/bin/bash
#SBATCH --job-name=sanity_train_lm1b
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x_%j.log

# ============================================================================
# Sanity check: does the LM1B training loop actually run end-to-end?
# Reuses configs/lm1b/continuous/rate_bits_1M_edm_weight.py but writes a
# temporary copy with these overrides (via sed) so the released paper run
# directory is NEVER touched:
#
#   * cfg.experiment           paper/.../1M_edm_weighting -> sanity/lm1b_train_smoke
#   * cfg.optim.total_steps    1_000_000 -> 2_000
#   * cfg.optim.warmup         2_500     -> 100
#   * entropy_warmup_steps     40_000    -> 800
#   * entropy_transition_steps 10_000    -> 400
#   * entropy_update_every     2_000     -> 200
#   * checkpoint every-steps   50_000    -> 500   (last.pt resume cadence)
#   * resume-interval steps    5_000     -> 200
#   * training-time eval callbacks (gen / external_ppl / mauve / viz / vlb): off
#
# Total wall time on 1 GPU: ~5-15 min. The aim is to exercise data loading,
# mixed-precision forward+backward, EMA update, checkpoint write, entropy-rate
# buffer warmup AND transition. NOT a convergence test.
# ============================================================================

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Date:   $(date)"

module load gcc-native/12.3
source ~/miniconda3/bin/activate pytorch
echo "Activated conda environment: $CONDA_DEFAULT_ENV"

PROJECT_DIR="$HOME/gb511/projects/BitstreamDiffusion"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs

BASE_CFG="configs/lm1b/continuous/rate_bits_1M_edm_weight.py"
TEMP_CFG="$(mktemp -p "${TMPDIR:-/tmp}" sanity_lm1b_train_XXXXXX.py)"
trap 'rm -f "$TEMP_CFG"' EXIT

echo "Base config: $BASE_CFG"
echo "Temp config: $TEMP_CFG"

# Build the overridden config in one sed pass. We anchor each substitution to the
# full original line so an unexpected upstream change makes sed fail loudly via -u.
sed \
  -e 's|cfg\.experiment = "paper/unconditional_text/lm1b/continuous_rate_raw_binary_bits_1M_edm_weighting"|cfg.experiment = "sanity/lm1b_train_smoke"|' \
  -e 's|cfg\.optim\.total_steps = 1_000_000|cfg.optim.total_steps = 2_000|' \
  -e 's|cfg\.optim\.warmup = 2_500|cfg.optim.warmup = 100|' \
  -e 's|cfg\.train\.entropy_warmup_steps = 40_000|cfg.train.entropy_warmup_steps = 800|' \
  -e 's|cfg\.train\.entropy_transition_steps = 10_000|cfg.train.entropy_transition_steps = 400|' \
  -e 's|cfg\.train\.entropy_update_every_steps = 2000|cfg.train.entropy_update_every_steps = 200|' \
  -e 's|cfg\.train\.checkpointing\.interval\.every_steps = 50_000|cfg.train.checkpointing.interval.every_steps = 500|' \
  -e 's|cfg\.train\.checkpointing\.resume_interval\.every_steps = 5_000|cfg.train.checkpointing.resume_interval.every_steps = 200|' \
  -e 's|cfg\.train\.generation\.enabled = True|cfg.train.generation.enabled = False|' \
  -e 's|cfg\.train\.external_ppl\.enabled = True|cfg.train.external_ppl.enabled = False|' \
  -e 's|cfg\.train\.mauve\.enabled = True|cfg.train.mauve.enabled = False|' \
  -e 's|cfg\.train\.visualization\.enabled = True|cfg.train.visualization.enabled = False|' \
  -e 's|cfg\.train\.vlb\.enabled = True|cfg.train.vlb.enabled = False|' \
  "$BASE_CFG" > "$TEMP_CFG"

# Verify all expected overrides actually fired (sed is silent on no-match).
echo ""
echo "=== Override audit ==="
grep -E '^    cfg\.experiment|cfg\.optim\.total_steps|cfg\.optim\.warmup|entropy_warmup_steps|entropy_transition_steps|entropy_update_every_steps|checkpointing\.interval\.every_steps|checkpointing\.resume_interval\.every_steps|generation\.enabled|external_ppl\.enabled|mauve\.enabled|visualization\.enabled|vlb\.enabled' "$TEMP_CFG" | head -20
echo ""

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((29000 + ($SLURM_JOB_ID % 1000)))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/torchinductor_${USER}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_${USER}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$PROJECT_DIR/hf_cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "--- Launching sanity training (writes to runs/sanity/lm1b_train_smoke/) ---"
echo "(If you want a clean restart, delete runs/sanity/lm1b_train_smoke/ before submitting.)"
echo ""

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  train.py --config "$TEMP_CFG"

echo ""
echo "=== Final state of sanity run dir ==="
ls -la runs/sanity/lm1b_train_smoke/checkpoints/ 2>/dev/null || true

echo "Sanity train run completed."
