#!/usr/bin/env bash
# Launch the n=256 FKC beta sweep DETACHED, so it survives SSH disconnect / Claude Code exit.
# setsid + nohup + </dev/null puts each chain in its own session with no controlling
# terminal, so SIGHUP on disconnect cannot reach it.
#
#   bash scripts/tasks/run_n256_detached.sh          # start
#   bash scripts/tasks/status_n256.sh                # check later (any new shell)
set -u
cd "$(dirname "$0")/../.."
OUT=runs/tasks/tinygsm/cobit_raw_binary_bits/gsm8k_eval_fkc/sweep_K32_s512_n256
mkdir -p "$OUT" logs

# GPU0: K=1 control, then beta 1.03, 1.01 at K=32
setsid nohup bash -c "
  CUDA_VISIBLE_DEVICES=0 MODE=beta LIMIT=256 K=1  BATCH=64 STEPS=512 BETAS='1.0' \
    POLICY=ess FINALR=1 OUT=$OUT bash scripts/tasks/fkc_gsm8k_sweep.sh
  CUDA_VISIBLE_DEVICES=0 MODE=beta LIMIT=256 K=32 BATCH=2  STEPS=512 BETAS='1.03 1.01' \
    POLICY=ess FINALR=1 OUT=$OUT bash scripts/tasks/fkc_gsm8k_sweep.sh
" > logs/n256_gpu0.log 2>&1 < /dev/null &
echo "GPU0 chain pid=$!"

# GPU1: beta 1.05, 1.02, 1.04 at K=32
setsid nohup bash -c "
  CUDA_VISIBLE_DEVICES=1 MODE=beta LIMIT=256 K=32 BATCH=2 STEPS=512 BETAS='1.05 1.02 1.04' \
    POLICY=ess FINALR=1 OUT=$OUT bash scripts/tasks/fkc_gsm8k_sweep.sh
" > logs/n256_gpu1.log 2>&1 < /dev/null &
echo "GPU1 chain pid=$!"

sleep 2
echo "--- detached (own session, immune to SIGHUP) ---"
ps -eo pid,sid,cmd | grep -E "fkc_gsm8k_sweep|gsm8k_eval" | grep -v grep | head
