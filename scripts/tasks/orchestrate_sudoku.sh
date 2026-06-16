#!/usr/bin/env bash
# Autonomous Sudoku orchestration (runs inside its own tmux session, so it
# survives SSH/VSCode/Claude-Code termination). Pipelined so each difficulty is
# evaluated as soon as a GPU frees, rather than waiting for all three:
#   1. assumes easy (GPU0) + medium (GPU1) are already training under tmux;
#   2. when easy finishes, trains hard on the freed GPU0 (background);
#   3. when medium finishes, GPU1 is free -> eval easy + medium on GPU1
#      WHILE hard keeps training on GPU0;
#   4. when hard finishes, eval hard on GPU1.
# Headline eval = DDIM 'ddim_entropic' + EDM churn; we report BOTH EMA and raw
# weights (EMA decay 0.9999 has only ~2 time constants at the 20k-step budget).
set -eo pipefail
cd "$(dirname "$0")/../.."

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pytorch}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export TOKENIZERS_PARALLELISM=false

run_dir() { echo "runs/tasks/sudoku/$1/cobit_raw_binary_bits"; }
ckpt_of() {
  local d; d="$(run_dir "$1")/checkpoints"
  if [ -f "$d/step=000020000.pt" ]; then echo "$d/step=000020000.pt"; else echo "$d/last.pt"; fi
}
finished() { grep -qa "FINISHED difficulty=$1" "logs/tasks/train_sudoku_$1.log" 2>/dev/null; }
wait_finished() { echo "[orch] waiting for $1 to finish ..."; until finished "$1"; do sleep 60; done; echo "[orch] $1 finished $(date)"; }

# eval_difficulty <difficulty> <eval_gpu>
eval_difficulty() {
  local d="$1" gpu="$2" CK; CK="$(ckpt_of "$d")"
  echo "[orch] eval $d on GPU$gpu  ckpt=$CK  $(date)"
  for ema in 1 0; do
    for g in 0.0 0.05 0.10; do
      if [ "$g" = "0.0" ]; then S=deterministic; else S=stochastic; fi
      CUDA_VISIBLE_DEVICES="$gpu" python -m evaluation.tasks.sudoku_eval \
        --config configs/tasks/sudoku_bits.py \
        --checkpoint "$CK" --difficulty "$d" \
        --sampler "$S" --sampler_kind ddim --schedule entropic \
        --gamma "$g" --steps 180 --limit 2000 --ema "$ema" \
        >> "logs/tasks/eval_sudoku_${d}.log" 2>&1 || echo "[orch] eval $d ema=$ema g=$g FAILED"
    done
  done
  echo "[orch] eval $d DONE $(date)"
}

# 1) easy finishes -> start hard on the freed GPU0.
wait_finished easy
echo "[orch] launching hard on GPU0 $(date)"
DIFF=hard GPU=0 bash scripts/tasks/train_sudoku_one_gpu.sh > logs/tasks/train_sudoku_hard.log 2>&1 &

# 2) medium finishes -> GPU1 free -> eval easy + medium there (hard still on GPU0).
wait_finished medium
eval_difficulty easy 1
eval_difficulty medium 1

# 3) hard finishes -> eval hard on GPU1.
wait_finished hard
eval_difficulty hard 1

echo "[orch] ============ SUMMARY ============"
python - <<'PY'
import json, glob
for d in ("easy","medium","hard"):
    base=f"runs/tasks/sudoku/{d}/cobit_raw_binary_bits/sudoku_eval"
    best=None
    for f in sorted(glob.glob(base+"/sudoku_results_*.json")):
        r=json.load(open(f))
        acc=r.get("exact_match_accuracy",0.0)
        tag=f"{'ema' if r.get('ema',True) else 'raw'}/{r['sampler']}/g{r['gamma']}"
        print(f"{d:6s} {tag:24s} exact={acc*100:6.2f}%  valid={r.get('valid_sudoku_rate',0)*100:5.1f}%  invalid_tok={r.get('invalid_token_rate',0)*100:.3f}%")
        if best is None or acc>best[0]: best=(acc,tag)
    if best: print(f"  -> {d} BEST: {best[1]} = {best[0]*100:.2f}%")
PY
echo "[orch] DONE $(date)"
