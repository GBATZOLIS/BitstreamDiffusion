#!/usr/bin/env bash
# Wait for the in-flight K=32 cells (beta 1.06 / 1.08) to finish, then launch:
#   GPU0: K=64, beta=1.05, 512 steps, n=256   -> K-scaling test vs the K=32/512 cell
#   GPU1: K=1 control @1024 (n=256), then K=64, beta=1.05, 1024 steps, n=128
#
# The 1024-step arm needs its own K=1 control: the step count moves the baseline,
# so reusing the 512-step control would confound step count with beta.
#
# NOTE: the wait loop greps for the python eval process. The pattern lives inside
# this file rather than on our own command line, so pgrep cannot match this script
# itself (an earlier inline version self-matched and looped forever).
set -u
cd "$(dirname "$0")/../.."
OUT=runs/tasks/tinygsm/cobit_raw_binary_bits/gsm8k_eval_fkc/sweep_K32_s512_n256
OUT1024=runs/tasks/tinygsm/cobit_raw_binary_bits/gsm8k_eval_fkc/sweep_s1024_n256
PAT="tasks.gsm8k_eval"
mkdir -p "$OUT" "$OUT1024" logs

echo "[next] waiting for in-flight cells to drain..."
while pgrep -f "$PAT" > /dev/null; do sleep 30; done
echo "[next] clear at $(date +%H:%M) -- launching"

setsid nohup bash -c "cd $PWD && CUDA_VISIBLE_DEVICES=0 MODE=beta LIMIT=256 K=64 BATCH=1 STEPS=512 \
  BETAS='1.05' POLICY=ess FINALR=1 OUT=$OUT bash scripts/tasks/fkc_gsm8k_sweep.sh" \
  > logs/K64_b105_s512.log 2>&1 < /dev/null &
echo "[next] GPU0 K=64 beta=1.05 s512 n=256 pid=$!"

setsid nohup bash -c "cd $PWD && \
  CUDA_VISIBLE_DEVICES=1 MODE=beta LIMIT=256 K=1 BATCH=64 STEPS=1024 BETAS='1.0' \
    POLICY=ess FINALR=1 OUT=$OUT1024 bash scripts/tasks/fkc_gsm8k_sweep.sh; \
  CUDA_VISIBLE_DEVICES=1 MODE=beta LIMIT=128 K=64 BATCH=1 STEPS=1024 BETAS='1.05' \
    POLICY=ess FINALR=1 OUT=$OUT1024 bash scripts/tasks/fkc_gsm8k_sweep.sh" \
  > logs/K64_b105_s1024.log 2>&1 < /dev/null &
echo "[next] GPU1 control@1024 then K=64 beta=1.05 s1024 n=128 pid=$!"
