#!/bin/bash
# Internal: wait for the two single-GPU CoBit-M runs, then run the two CoBit-S
# verifications (one per GPU), then print all GenPPL/entropy vs paper targets.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate pytorch 2>/dev/null || true

MA="$1"; MB="$2"   # the two CoBit-M logs

wait_done () { # $1=logfile
  for _ in $(seq 1 360); do
    grep -qa 'done; results under' "$1" 2>/dev/null && return 0
    sleep 15
  done
  return 1
}

echo "[orch] waiting for CoBit-M runs to finish..."
wait_done "$MA"; echo "[orch] M_a done"
wait_done "$MB"; echo "[orch] M_b done"

echo "[orch] launching CoBit-S verifications (OWT on GPU0, LM1B on GPU1)..."
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=29201 \
  -m evaluation.run_eval --config configs/owt/eval_750K_seed.py --metrics external_ppl > logs/verify_s_owt.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=29202 \
  -m evaluation.run_eval --config configs/lm1b/continuous/eval/rate_eval_seeds.py --metrics external_ppl > logs/verify_s_lm1b.log 2>&1 &
P2=$!
wait $P1; echo "[orch] S-OWT gen done"
wait $P2; echo "[orch] S-LM1B gen done"
echo "[orch] computing entropy..."
python -m evaluation.compute_entropy_from_caches --config configs/owt/eval_750K_seed.py --include_real >> logs/verify_s_owt.log 2>&1 || true
python -m evaluation.compute_entropy_from_caches --config configs/lm1b/continuous/eval/rate_eval_seeds.py --include_real >> logs/verify_s_lm1b.log 2>&1 || true

echo "[orch] ====== ALL VERIFICATIONS COMPLETE — parsing ======"
python - <<'PY'
import json, glob, os
from collections import defaultdict
def parse(jsonl):
    g=defaultdict(dict)
    if not os.path.exists(jsonl): return g
    for line in open(jsonl):
        try: r=json.loads(line)
        except: continue
        m=r.get('metric','')
        if m in ('gen_full_external_ppl','gen_full_token_unigram_entropy','real_full_external_ppl','real_full_token_unigram_entropy'):
            try: nfe=int(float(r['nfe'])); sc=round(float(r.get('s_churn',0) or 0),2)
            except: continue
            g[(nfe,sc)][m]=float(r['value'])
    return g
def show(title, jsonl, targets):
    g=parse(jsonl)
    print(f"\n### {title}   ({jsonl})")
    if not g: print("   (no results)"); return
    print(f"   {'NFE':>4} {'gamma':>6} {'GenPPL':>9} {'target':>7} {'H':>7} {'realPPL':>8}")
    for k in sorted(g):
        nfe,sc=k; m=g[k]; gam=round(sc/(nfe-1),3) if nfe>1 else 0
        t=targets.get((nfe,round(gam,2)),'')
        print(f"   {nfe:>4} {gam:>6.2f} {m.get('gen_full_external_ppl',float('nan')):>9.3f} {str(t):>7} "
              f"{m.get('gen_full_token_unigram_entropy',float('nan')):>7.3f} {m.get('real_full_external_ppl',float('nan')):>8.3f}")
base="runs/paper/unconditional_text"
show("CoBit-M (a: 256/0.21, 512/0.26)", f"{base}/owt/continuous_rate_raw_binary_bits_medium_24x1024/evaluation_cobit_m_table2_step000750000_a/results.jsonl",
     {(256,0.21):19.48,(512,0.26):9.87})
show("CoBit-M (b: 384/0.24, 256/0.13)", f"{base}/owt/continuous_rate_raw_binary_bits_medium_24x1024/evaluation_cobit_m_table2_step000750000_b/results.jsonl",
     {(384,0.24):13.06,(256,0.13):18.47})
show("CoBit-S OWT (Table 1)", f"{base}/owt/continuous_rate_raw_binary_bits_1M/evaluation_cleanup_smoketest/results.jsonl",
     {(256,0.13):27.06,(256,0.18):34.35})
# LM1B out dir:
lm=glob.glob(f"{base}/lm1b/continuous_rate_raw_binary_bits_1M_edm_weighting/evaluation*/results.jsonl")
show("CoBit-S LM1B (Table 1)", lm[0] if lm else "MISSING", {(256,0.20):59.76})
print("\n[orch] done.")
PY
