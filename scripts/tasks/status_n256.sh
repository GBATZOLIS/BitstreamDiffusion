#!/usr/bin/env bash
# Status of the detached n=256 FKC sweep. Safe to run from any new shell.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/pytorch/bin/python}
OUT=runs/tasks/tinygsm/cobit_raw_binary_bits/gsm8k_eval_fkc/sweep_K32_s512_n256

echo "=== alive? ==="
ps -eo etime,cmd | grep gsm8k_eval | grep -v grep \
  | grep -oE "^ *[0-9:-]+|--beta [0-9.]+|--num_particles [0-9]+" | paste - - - || echo "(nothing running)"

echo
echo "=== completed cells ==="
"$PY" - <<'EOF'
import json, glob, os
out = 'runs/tasks/tinygsm/cobit_raw_binary_bits/gsm8k_eval_fkc/sweep_K32_s512_n256'
rows = []
for f in glob.glob(os.path.join(out, '*.json')):
    d = json.load(open(f))
    rows.append((d['beta'], d['num_particles'], d['num_examples'], d['min_ess'],
                 d['total_resample_events'], d['particle_mean_accuracy'],
                 d['maj_at_k'], d['mean_distinct_answers']))
rows.sort()
if not rows:
    print('(none finished yet)')
else:
    print(f"{'beta':<8}{'K':>4}{'n':>6}{'min_ess':>9}{'resamp':>8}{'part_acc':>10}{'maj@K':>8}{'distinct':>10}")
    for b,k,n,e,r,p,m,dd in rows:
        print(f'{b:<8g}{k:>4d}{n:>6d}{e:>9.2f}{r:>8d}{100*p:>9.1f}%{100*m:>7.1f}%{dd:>10.2f}')
EOF

echo
echo "=== in-flight progress (last line per GPU) ==="
for f in logs/n256_gpu0.log logs/n256_gpu1.log; do
  [ -f "$f" ] || continue
  echo "-- $f"
  grep -E "^############ MODE" "$f" | tail -n 1
  grep -E "^\[gsm8k-fkc\] [0-9]" "$f" | tail -n 1
done
