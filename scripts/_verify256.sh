#!/bin/bash
# 256-NFE-only verification of all three slim EMA checkpoints.
#   GPU0: CoBit-M (256 @ gamma 0.21 & 0.13)  ->  then CoBit-S OWT (256)
#   GPU1: CoBit-S LM1B (256)
# Then post-hoc entropy + a results table vs paper targets.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate pytorch 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/torchinductor_${USER}" TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_${USER}"
export EVAL_SEED=42
mkdir -p logs

ev () { # gpu config [extra-env-prefix]
  local gpu="$1" cfg="$2"
  CUDA_VISIBLE_DEVICES="$gpu" torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    --master_port=$((29300 + gpu + RANDOM % 50)) \
    -m evaluation.run_eval --config "$cfg" --metrics external_ppl
  CUDA_VISIBLE_DEVICES="$gpu" python -m evaluation.compute_entropy_from_caches --config "$cfg" --include_real || true
}

(
  EVAL_CELLS="256:0.21,256:0.13" EVAL_OUT_SUFFIX=_256 ev 0 configs/owt/eval_cobit_m_750K.py
  ev 0 configs/owt/eval_750K_seed.py
) > logs/verify256_gpu0.log 2>&1 &
P0=$!
( ev 1 configs/lm1b/continuous/eval/rate_eval_seeds.py ) > logs/verify256_gpu1.log 2>&1 &
P1=$!
wait $P0; echo "[v256] GPU0 chain done"
wait $P1; echo "[v256] GPU1 chain done"

echo "[v256] ===== RESULTS (NFE=256) ====="
python - <<'PY'
import json, glob, os
from collections import defaultdict
def parse(p):
    g=defaultdict(dict)
    if not p or not os.path.exists(p): return g
    for line in open(p):
        try: r=json.loads(line)
        except: continue
        m=r.get('metric','')
        if m in ('gen_full_external_ppl','gen_full_token_unigram_entropy','real_full_external_ppl','real_full_token_unigram_entropy'):
            try: nfe=int(float(r['nfe'])); sc=round(float(r.get('s_churn',0) or 0),2)
            except: continue
            g[(nfe,sc)][m]=float(r['value'])
    return g
def show(title,p,tg):
    g=parse(p); print(f"\n### {title}\n   {p}")
    if not g: print("   (NO RESULTS)"); return
    print(f"   {'NFE':>4} {'gamma':>6} {'GenPPL':>9} {'target':>7} {'H':>7} {'realPPL':>8}")
    for k in sorted(g):
        nfe,sc=k;m=g[k];gam=round(sc/(nfe-1),3) if nfe>1 else 0
        print(f"   {nfe:>4} {gam:>6.2f} {m.get('gen_full_external_ppl',float('nan')):>9.3f} "
              f"{str(tg.get((nfe,round(gam,2)),'')):>7} {m.get('gen_full_token_unigram_entropy',float('nan')):>7.3f} "
              f"{m.get('real_full_external_ppl',float('nan')):>8.3f}")
b="runs/paper/unconditional_text"
show("CoBit-M 462M (Table 2, 256)", f"{b}/owt/continuous_rate_raw_binary_bits_medium_24x1024/evaluation_cobit_m_table2_step000750000_256/results.jsonl", {(256,0.21):19.48,(256,0.13):18.47})
show("CoBit-S OWT 130M (Table 1)", f"{b}/owt/continuous_rate_raw_binary_bits_1M/evaluation_cleanup_smoketest/results.jsonl", {(256,0.13):27.06,(256,0.18):34.35})
lm=glob.glob(f"{b}/lm1b/continuous_rate_raw_binary_bits_1M_edm_weighting/evaluation*/results.jsonl")
show("CoBit-S LM1B 130M (Table 1)", lm[0] if lm else "", {(256,0.20):59.76})
print("\n[v256] done.")
PY
