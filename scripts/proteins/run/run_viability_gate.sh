#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

run_stage() {
  local name="$1"
  shift
  echo "[$(date -Is)] START ${name}"
  "$@"
  echo "[$(date -Is)] DONE  ${name}"
}

run_stage bitstream_train \
  .venv/bin/torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/proteins/dima35m_bitstream_viability.py
run_stage bitstream_generate_gate \
  .venv/bin/python -m evaluation.generate_dima_bitstream \
  --config configs/proteins/dima35m_bitstream_viability.py \
  --checkpoint runs/proteins/dima35m_bitstream_viability_seed0/checkpoints/best.pt \
  --num-samples 256 --micro-batch-size 64 --num-steps 128 \
  --out-dir runs/proteins/dima35m_bitstream_viability_seed0/frozen_gate_n256
run_stage bitstream_basic_metrics \
  .venv/bin/python -m evaluation.dima_frozen_metrics \
  --generated-fasta runs/proteins/dima35m_bitstream_viability_seed0/frozen_gate_n256/generated.fasta \
  --num-samples 256 --metrics basic \
  --out runs/proteins/dima35m_bitstream_viability_seed0/frozen_gate_n256/frozen_metrics.json

run_stage categorical_train \
  .venv/bin/torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/proteins/dima_small_categorical_viability.py
run_stage categorical_generate_gate \
  .venv/bin/python -m evaluation.generate_dima_categorical \
  --config configs/proteins/dima_small_categorical_viability.py \
  --checkpoint runs/proteins/dima_small_categorical_viability_seed0/checkpoints/best.pt \
  --num-samples 256 --micro-batch-size 64 --num-steps 128 \
  --out-dir runs/proteins/dima_small_categorical_viability_seed0/frozen_gate_n256
run_stage categorical_basic_metrics \
  .venv/bin/python -m evaluation.dima_frozen_metrics \
  --generated-fasta runs/proteins/dima_small_categorical_viability_seed0/frozen_gate_n256/generated.fasta \
  --num-samples 256 --metrics basic \
  --out runs/proteins/dima_small_categorical_viability_seed0/frozen_gate_n256/frozen_metrics.json

run_stage ar_train \
  .venv/bin/torchrun --standalone --nproc_per_node=4 -m autoregression.train \
  --config configs/proteins/dima_small_ar_viability.py
run_stage ar_generate_gate \
  .venv/bin/python -m evaluation.generate_dima_ar \
  --config configs/proteins/dima_small_ar_viability.py \
  --checkpoint runs/proteins/dima_small_ar_viability_seed0/checkpoints_ar/best.pt \
  --num-samples 256 --micro-batch-size 64 \
  --out-dir runs/proteins/dima_small_ar_viability_seed0/frozen_gate_n256
run_stage ar_basic_metrics \
  .venv/bin/python -m evaluation.dima_frozen_metrics \
  --generated-fasta runs/proteins/dima_small_ar_viability_seed0/frozen_gate_n256/generated.fasta \
  --num-samples 256 --metrics basic \
  --out runs/proteins/dima_small_ar_viability_seed0/frozen_gate_n256/frozen_metrics.json

run_stage build_gate_table \
  .venv/bin/python -m evaluation.build_protein_table \
  --result BitStream-35.8M runs/proteins/dima35m_bitstream_viability_seed0/frozen_gate_n256/frozen_metrics.json \
  --result Categorical-11.6M runs/proteins/dima_small_categorical_viability_seed0/frozen_gate_n256/frozen_metrics.json \
  --result AR-9.8M runs/proteins/dima_small_ar_viability_seed0/frozen_gate_n256/frozen_metrics.json \
  --out-dir runs/proteins/viability_gate_tables

echo "[$(date -Is)] VIABILITY QUEUE COMPLETE"
