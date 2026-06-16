#!/usr/bin/env bash
# Autonomous Sudoku orchestration (runs inside its own tmux session, so it
# survives SSH/VSCode/Claude-Code termination):
#   1. assumes easy (GPU0) + medium (GPU1) are already training under tmux;
#   2. when easy finishes, trains hard on the freed GPU0;
#   3. when all three reach 20k steps, runs the headline DDIM + EDM-churn eval
#      (gamma sweep incl. deterministic) for each difficulty -> results JSON.
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

# 1) wait for easy, then start hard on GPU0 (freed by easy).
wait_finished easy
echo "[orch] launching hard on GPU0 $(date)"
DIFF=hard GPU=0 bash scripts/tasks/train_sudoku_one_gpu.sh > logs/tasks/train_sudoku_hard.log 2>&1 &

# 2) wait for medium and hard.
wait_finished medium
wait_finished hard

# 3) evaluate each difficulty: deterministic (g=0) + stochastic churn sweep.
echo "[orch] all training done — starting evals $(date)"
for d in easy medium hard; do
  CK="$(ckpt_of "$d")"
  echo "[orch] eval $d  ckpt=$CK"
  for g in 0.0 0.05 0.10; do
    if [ "$g" = "0.0" ]; then S=deterministic; else S=stochastic; fi
    python -m evaluation.tasks.sudoku_eval \
      --config configs/tasks/sudoku_bits.py \
      --checkpoint "$CK" --difficulty "$d" \
      --sampler "$S" --sampler_kind ddim --schedule entropic \
      --gamma "$g" --steps 180 --limit 2000 \
      >> "logs/tasks/eval_sudoku_${d}.log" 2>&1 || echo "[orch] eval $d g=$g FAILED"
  done
done

echo "[orch] ============ SUMMARY ============"
python - <<'PY'
import json, glob, os
for d in ("easy","medium","hard"):
    base=f"runs/tasks/sudoku/{d}/cobit_raw_binary_bits/sudoku_eval"
    best=None
    for f in sorted(glob.glob(base+"/sudoku_results_*.json")):
        r=json.load(open(f))
        acc=r.get("exact_match_accuracy",0.0)
        tag=f"{r['sampler']}/g{r['gamma']}"
        print(f"{d:6s} {tag:20s} exact={acc*100:6.2f}%  valid={r.get('valid_sudoku_rate',0)*100:5.1f}%  invalid_tok={r.get('invalid_token_rate',0)*100:.3f}%")
        if best is None or acc>best[0]: best=(acc,tag)
    if best: print(f"  -> {d} BEST: {best[1]} = {best[0]*100:.2f}%")
PY
echo "[orch] DONE $(date)"
