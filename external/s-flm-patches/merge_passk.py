#!/usr/bin/env python
"""Merge pass@k results across DISJOINT shards of the GSM8K test set.

Shards run on different machines can be concatenated because pass@k is computed
per problem and then averaged. Every shard must share checkpoint, K, steps,
temperature and precision -- only the problem set may differ.

Accepts both formats:
  * s-flm  results.json      -> records grouped by prompt  -> (c, K) per problem
  * CoBit  gsm8k_results_*.json -> per_prompt_particle_acc  -> (c, K) per problem

  python merge_passk.py out.json shardA/results.json shardB/results.json
"""
import json, sys
from math import comb
from collections import defaultdict, Counter


def load_shard(path):
    """-> (list of (n_samples, n_correct), label, meta)"""
    d = json.load(open(path))
    if 'records' in d and d.get('records') and 'problem_idx' in d['records'][0]:
        by = defaultdict(list)
        for r in d['records']:
            by[r['prompt']].append(r)
        # correctness recomputed per record is unavailable here; use stored multi_sample
        # if present, else fall back to re-grading (slow) -- we store per-record grade below.
        out = []
        for prompt, recs in by.items():
            n = len(recs)
            c = sum(int(r.get('correct', 0)) for r in recs)
            out.append((n, c))
        return out, 's-flm', {'K': d.get('multi_sample', {}).get('K')}
    if 'per_prompt_particle_acc' in d:
        K = int(d['num_particles'])
        return ([(K, int(round(f * K))) for f in d['per_prompt_particle_acc']],
                'cobit', {'K': K})
    raise ValueError(f'unrecognised result format: {path}')


def pass_at_k(n, c, k):
    if k > n:
        raise ValueError(f'k={k} > n={n}')
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def main():
    out_path, shards = sys.argv[1], sys.argv[2:]
    allp, labels = [], []
    for s in shards:
        p, lab, meta = load_shard(s)
        print(f'  {s}: {len(p):4d} problems, K={meta["K"]} ({lab})')
        allp += p; labels.append(lab)
    n_prob = len(allp)
    Ks = {n for n, _ in allp}
    if len(Ks) != 1:
        print(f'  WARNING: shards have different K: {Ks} -- curve capped at min')
    K = min(Ks)
    curve = {k: sum(pass_at_k(n, c, k) for n, c in allp) / n_prob
             for k in range(1, K + 1)}
    res = {'num_problems': n_prob, 'K': K, 'shards': shards,
           'pass_at_k': curve, 'pass_at_1': curve[1], f'pass_at_{K}': curve[K]}
    json.dump(res, open(out_path, 'w'), indent=1)
    print(f'\n  merged {n_prob} problems  pass@1={curve[1]*100:.2f}%  '
          f'pass@{K}={curve[K]*100:.2f}%  -> {out_path}')
    if n_prob != 1319:
        print(f'  NOTE: {n_prob} != 1319 -- shards do not tile the full test set')


if __name__ == '__main__':
    main()
