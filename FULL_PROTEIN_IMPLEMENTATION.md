# FULL_PROTEIN_IMPLEMENTATION

Implementation plan for the CoBit protein programme: one continuous bitstream-diffusion
model over an 18-bit residue patch that performs sequence generation, forward folding,
inverse folding, structure completion, joint co-generation and motif scaffolding, trained
on a multi-node no-GRES Slurm cluster.

Status: 2026-07-31. This continues `.trentinium/docs/all_in_protein_bitstream_plan.md`
(the design document, which remains authoritative for scientific positioning) and replaces
its execution sections. Where the two disagree, this file wins on execution and the design
doc wins on claims and framing.

Evidence policy. Every quantitative claim below is tagged `[measured]`, `[computed]`,
`[estimated]` or `[unverified]`. Numbers tagged `[measured]` were produced by running code
or reading artefacts in this working tree. Numbers tagged `[computed]` are derived
arithmetically from measured inputs and stated assumptions. Anything `[estimated]` is a
judgement call and should be re-measured before it drives a compute request. Several
throughput figures were taken on contended GPUs and carry roughly +/-40% uncertainty; they
are marked.

---

## 1. Target environment and what it forces

Confirmed constraints for the training cluster:

| Constraint | Value | Consequence for this plan |
|---|---|---|
| GPUs | 8 x H100 80GB SXM per node | Memory is not the binding constraint at 400M; activation checkpointing is still required to keep per-GPU micro-batch high |
| GRES | **Not available** | Every launcher must use `--exclusive` whole-node allocation and discover the GPU count at runtime |
| Max walltime | **4 hours** | A multi-hundred-GPU-hour run is 10-30 chained jobs; exact resume and fast checkpointing move from "nice" to load-bearing |
| Preemption | Jobs can be preempted | `SIGUSR1` checkpoint-and-exit handler is mandatory; without it each preemption discards up to `resume_interval.every_steps` of work |
| Filesystem | **Shared only, no node-local NVMe** | Node-local staging must target `/tmp` or `/dev/shm` (RAM-backed) and be verified; the dataset must become memory-mappable so 8 ranks share one page-cache copy rather than 8 heap copies |
| Compute budget | **~2,000 GPU-hours total** | Rules out a 650M flagship and rules out M2/ESMAtlas; forces an explicit budget ledger (section 12) |

Two of these reshape the technical plan more than anything else.

**The 4-hour limit** means the campaign is a chain of short jobs, not a long run. Exact
resume already exists and works (sampler cursor, collator RNG, optimizer, EMA, scaler and
global RNG are all checkpointed at `trainers/trainer.py:2504-2542` and restored at
`trainers/trainer.py:1706-1740`) `[measured]`. What does not exist is any signal handling
or any wall-clock awareness, so today the job is killed at an arbitrary point and resumes
from the last periodic checkpoint. At the flagship scale that is up to 30 minutes of lost
work per link, repeated 20-30 times.

**Shared-filesystem-only** means the current data path does not scale. `ProteinMultimodalDataset.__init__`
(`data/protein_multimodal.py:316-340`) calls `np.load` on each `.npz` and materialises every
array before the split filter at line 331, so each rank holds a private full copy of every
split. Measured today on M0: 390 MB for the paired train instance, a second 390 MB for the
val instance (the same arrays again, because only `self._index` is split-filtered), and
1,111 MB for the replay instance, i.e. **~1.9 GB per rank and ~15.1 GB per 8-GPU node
before any pinned dataloader buffers** `[measured]`. `mmap_mode='r'` is silently ignored for
`.npz` (verified), so the fix is a format change, not a flag.

---

## 2. Executive decisions

These are the calls this plan makes. Each is argued in the referenced section.

1. **One joint any-to-any model remains the primary product.** Task specialists are trained
   only as scientific controls. (Section 9.1)
2. **Flagship size is 393M** (`configs/proteins/multimodal_lfq18_400m.py` trunk).
   **The 650M run is cut.** The structure-data budget does not support it and the compute
   budget does not either. (Section 9.5)
3. **M1 is scoped down** from ~1.25M AFDB representatives to a target of **400k-600k**
   confident representatives plus experimental PDB, because the download and the cluster
   list are both unsolved and the tokenisation cost is real. **M2 / ESMAtlas is cut.**
   (Section 7)
4. **Node count for the flagship is 1-2 nodes, 4 maximum.** Beyond that the pinned global
   batch drives the per-GPU micro-batch into the inefficient regime and the effective batch
   stops growing. More nodes is not more throughput here. (Section 10.2)
5. **A correctness release (P0) lands before any flagship GPU-hour is spent.** Seven defects
   found in this audit each independently invalidated an M0 conclusion or would silently
   corrupt a multi-node run. (Section 5)
6. **Evaluation is budgeted as a first-class consumer of compute**, at roughly 20% of the
   total, and is made shardable. It is currently single-process and single-GPU. (Section 11)

---

## 3. Where the code actually is

The `proteins` branch is far more complete than a greenfield. The following is verified
status, not aspiration.

### 3.1 Working and proven

| Component | Evidence |
|---|---|
| 18-bit residue patch, LFQ codec pinned MSB-first | `data/protein_structure_codec.py`; `convention_hash 71220cdbbb22b4ff`; manifest mismatch is rejected at `data/protein_multimodal.py:303-309` |
| Task masks: ABSENT/OBSERVED/NOISY per residue per modality | `data/protein_tasks.py:43-51`, five whole-example modes at 63-69 |
| Per-modality independent sigma and per-bit sigma map | `data/protein_tasks.py:275-291`, `trainers/multimodal_step.py:98-110` |
| Equal-modality loss reduction | `diffusion/continuous/losses.py:575-647` |
| Production trainer routes dict batches to the multimodal step, preserving the custom sampler and collator | `trainers/trainer.py:1004-1043` |
| Exact-resume checkpoint: sampler cursor, collator RNG, optimizer, scheduler, scaler, EMA, global RNG | `trainers/trainer.py:2504-2542`, `1706-1740` |
| Gradient accumulation with `no_sync` and 1/accum loss scaling | `trainers/trainer.py:2004-2014`, `2056` |
| Rank-consistent batch sampler: all ranks get the same length, disjoint index slices, no collective | `data/protein_multimodal_sampler.py:139-151` |
| Entropy schedule broadcast from rank 0 | `utils/schedule_controller.py:388-394`, `434-445` |
| Trunk attends over **L residues, not L*18 bits** | `models/sdt.py:954-959` `_patchify` folds 18 bits into the channel dim |
| M0 corpus built and trained | `datasets/dplm_paired_m0`, 220,475 rows, 27 shards, 411 MB `[measured]` |
| Frozen DPLM-2 structure tokenizer downloaded and hash-pinned | `datasets/dplm_struct_tokenizer/`, 451 MB, `provenance.json` `[measured]` |
| Warm-start column surgery from a 5-bit sequence checkpoint | `utils/protein_warmstart.py:29-129`; verified on a 650M twin: exact=380, slot_copied=5, skipped=0 `[measured]` |
| Structure evaluation drivers written | ~247 KB across `folding.py`, `inverse_folding.py`, `cogeneration.py`, `scaffolding.py`, `self_consistency.py`, `structure_metrics.py`, `tokenizer_audit.py` |

Measured parameter counts by instantiating each config `[measured]`:
37,743,425 (35M) / 125,192,769 (130M) / 392,959,217 (400M) / 639,682,177 (650M).

### 3.2 Written but never executed

`evaluation/proteins/tokenizer_audit.py` (883 lines) has never produced an output artefact.
`evaluation/proteins/scaffolding.py` (1,033 lines) is a complete motif evaluator including
multi-segment `MotifProblem`, RMSD, sequence preservation, ProteinMPNN+ESMFold
self-consistency and solved-target counting — but **no motif problem JSON exists anywhere in
the repo**. Same shape for `folding.py` (needs a CAMEO target manifest that nothing builds)
and `inverse_folding.py` (needs CATH 4.3 backbones that nothing builds). The pattern is
consistent: the consumers are built, the corpora are not.

`evaluation/proteins/foldseek_cluster` and `mmseqs_cluster` (`structure_metrics.py:401-470`)
have **zero call sites** `[measured]`.

### 3.3 Stubs that would fail silently

| Item | Failure mode |
|---|---|
| `prepare_afdb_representatives.py` | Pins `AFDB_MODEL_VERSION=4`; live probe shows **v1-v5 all return HTTP 404, only v6 returns 200** `[measured]`. A 404 is swallowed at line 399-401 into a counter, so a full run writes a success manifest and exits 0 having saved zero structures |
| `prepare_afdb_representatives.py --cluster-list` | Mandatory argument; **nothing in the repo produces the cluster list**, and the historical Steinegger mirror returns 404 `[measured]` |
| `prepare_esmatlas.py` | Hard-requires `shard_index.json`; no producer exists; the script knows no real ESMAtlas URL beyond a host constant |
| `tokenize_structures.py` | Cannot run: importing the tokenizer fails with `ModuleNotFoundError: openfold`, then `ValueError: mutable default LoRAConfig` under Python 3.11 `[measured]`. A working shim exists only as private functions inside `scripts/proteins/run/decode_m0_dashboard.py:25-52` |
| `cfg.model.gradient_checkpointing` | Set in two configs, **read by nothing** `[measured]` |
| `cfg.train.expected_world_size`, `cfg.train.global_batch_size` | Written in 12 configs, read only by a reporting script `[measured]` |
| `setup.sh` `run_prep()` | Swallows non-zero exits into a WARN, so the AFDB and ESMAtlas failures above are invisible |

---

## 4. What the M0 result actually says

This matters because it determines where to spend the budget.

| Metric | Conditioned | Shuffled control | Source |
|---|---|---|---|
| Inverse folding AAR (`last.pt`) | 37.4% best-of-4, 34.7% mean | 7.4% / 7.9% | `runs/proteins/m0_v1/protein_eval/dashboard/token_metrics.json` `[measured]` |
| Inverse folding paired lift | 0.300, CI95 [0.266, 0.334], n=32 | — | same |
| Forward folding structure bit accuracy | 65.7% / 64.5% | 52.9% / 51.8% | same |
| Forward folding lift | CI95 [0.114, 0.143] | — | same |
| Forward folding decoded TM | 0.366 | 0.286 shuffled, 0.261 unconditional | `geometry_metrics.json` `[measured]` |
| Exact LFQ token match | 0.97% | 0.33% predicted under bit independence | `[computed]` from bit_mean 0.6451 |

Three readings follow, and they point the same way.

**The cross-modal signal is real and strong at the token level.** Both lifts have 95%
confidence intervals that exclude zero by a wide margin. The joint mixture is not preventing
the model from learning sequence-structure dependence.

**The failure appears between "right bits" and "right geometry".** 65.7% bit accuracy but
0.97% exact-token match means bit errors are nearly independent: at 64.5% mean bit accuracy,
independent bits predict `0.645^13 = 0.33%` exact matches, and the observed 0.97% is only
2.9x above independence `[computed]`. The model is not concentrating probability on valid
neighbouring codewords. **This is the single most important scientific finding in the
audit** and it has a direct architectural cause (section 8.4) and a decisive cheap
experiment (section 9.6).

**One correction to the design doc's framing.** The reported forward-folding TM has a codec
ceiling of exactly 1.0 by construction: `geometry_metrics.json` declares
`reference_semantics = "reference LFQ tokens decoded by the same frozen tokenizer; not
experimental/native coordinates"` and the key is `tm_score_to_decoded_reference`
`[measured]`. Perfect token prediction gives byte-identical coordinates. So 0.366 is
unambiguously a **model** failure, not a representation ceiling, and structure metrics
should **not** be reported as `TM/TM_ceiling` for this metric. The tokenizer audit is still
required — for absolute comparison against native coordinates, against DPLM-2's published
numbers, and as the denominator for ESMFold self-consistency — but it does not gate the
interpretation of the numbers already in hand.

**Designability has never been measured.** No scTM, no scRMSD, no pLDDT of an independent
refold. That is the missing gate, and it is cheap relative to training.

---

## 5. P0: the correctness release

Seven defects. Each is independently capable of invalidating a multi-week run. All of them
land as **one release** before any flagship GPU-hour, because they jointly define what "the
recipe" means and landing them separately makes the M1 runs non-comparable to each other.

### P0-1. ABSENT modality bits leak the ground truth

`per_bit_sigma_map_torch` (`data/protein_tasks.py:287-289`) assigns sigma 0 to **both**
ABSENT and OBSERVED, and `trainers/multimodal_step.py:108-110` computes
`x_t = x0 + sigma_map * noise`. For an ABSENT modality, `x_t` is therefore **exactly `x0`** —
the true structure bits are visible in the content channel, distinguished only by a
zero-initialised state embedding.

Consequence: on paired rows, the `sequence_marginal` task is secretly inverse folding and
`structure_marginal` is secretly forward folding. At generation the same states produce
pure noise in those slots (`generate_multimodal.py:430-433` passes
`conditioning_prefix_full=None`), so the model meets an input distribution it never saw.
A third convention exists: replay rows store `struct_index=0`
(`prepare_uniref50_replay.py:169`), i.e. the all-zero 13-bit pattern.

Corrected scale: **12% of optimizer micro-steps** under the m0_v3 recipe are affected, not
20% — replay batches are 100% `sequence_marginal` on rows whose ABSENT structure is a
constant, hence not leaked `[computed]`. This does not weaken the defect. It sharpens it:
the ABSENT state embedding is being trained on **two contradictory conventions
simultaneously**.

Fix: add `absent_fill = cfg.diffusion.continuous.data_center` (0.5). In
`MultimodalTaskCollator.__call__` after `data/protein_multimodal.py:478`, overwrite the
patch at ABSENT slots with that constant (`patch` becomes float32, not uint8). In
`generate_multimodal.py:414-433`, add ABSENT positions to the clamp mask at the same
constant. Add a test asserting that for every task, `x_t` at ABSENT positions is
statistically independent of `x0`.

### P0-2. Time conditioning is symmetric under modality swap

`two_modality_time_sigma` returns `time_fn(sigma_seq) + time_fn(sigma_struct)`, which is
permutation-invariant. Verified numerically: `(sigma_seq, sigma_struct) = (0.002, 8.0)` and
`(8.0, 0.002)` produce a **bit-identical** embedding, max abs diff 0.0 `[measured]`.

Every AdaLN scale/shift/gate in every block is therefore blind to *which* modality is being
denoised. Forward folding and inverse folding — which differ by exactly this swap — receive
the same conditioning signal. The model can only recover the distinction from per-bit `c_in`
scaling and the zero-initialised state embeddings.

Fix (`models/protein_multimodal.py`, `models/sdt.py:1158-1164`): replace the sum with
`Linear(2E, E)` over the concatenation, or, warm-start friendlier, keep the sum with two
learned per-modality gain vectors initialised to 1. Cost at 400M: ~2.1M params (+0.5%).
Also add a small task-id embedding into `t_emb` so the trunk is told the task explicitly.

### P0-3. The boolean key-padding mask is inverted

`models/sdt.py:146-148` passes `key_padding_mask` (documented at line 59 as `True = pad/ignore`)
straight to SDPA, where a boolean `attn_mask` means `True = attend`. Verified: with
`kpm = [True, False, False, False]`, perturbing the "padded" key changes the output by
0.868 while perturbing a real key changes it by exactly 0.000 `[measured]` — attention sees
**only** the padding.

This is latent today because `pad_len` is always 0 for protein data (`S = L*18` is already a
multiple of `P=18`) and rpb is disabled. It becomes a live corruption the moment length-band
bucketing is introduced, which section 10.3 requires.

Fix: `attn_mask = ~kpm`, plus a unit test asserting a masked key has zero influence.

### P0-4. All ranks draw identical sigmas and identical noise

`trainers/trainer.py:838` calls `_maybe_set_seed(cfg)` with **no rank offset**
(`trainer.py:227-240`; its docstring claims `DistributedSampler handles the shuffling
offset`, which is false for the custom multimodal sampler). Every rank then executes an
identical RNG sequence on identical tensor shapes — because `SourceLengthBatchSampler` picks
the same `(source, length)` bucket on every rank — so `proc.sample_sigma`, the entropic
draw, `torch.randn_like(x0)` and the self-conditioning mask all return **bit-identical**
values on all ranks.

Consequence: at world size 64 the batch is 64x larger but carries only
`micro_batch_per_gpu` distinct noise levels. Variance reduction in sigma-space is
world_size times worse than the batch size implies, and the entropy-rate histogram sees the
same sigmas repeated on every rank.

Fix: after `create_model()` and the DDP wrap (so weight init still used the shared seed),
re-seed the stochastic stream with `seed * 1_000_003 + rank`. Keep `_mm_should_replay` on
its rank-independent generator so the replay decision stays consistent. Persist and restore
the stochastic seed in the checkpoint.

### P0-5. Checkpoint selection is anti-correlated with task quality

On the identical eval protocol, `last.pt` scores **37.4% inverse-folding AAR** while the
validation-selected `best.pt` scores **19.2%** `[measured]`
(`runs/proteins/m0_v1/protein_eval/dashboard/` vs `dashboard_best250/`). Selecting on
denoising validation loss cost 18 AAR points on the headline task.

`balanced_mse`, introduced in m0_v4 to fix this, is **worse**, not better: in all five
m0_v4 runs its minimum occurs at epoch 2-3 — 5,000 to 7,500 of a 250,000-step budget (2-3%)
— and then degrades monotonically for the rest of the run `[measured]`
(`runs/proteins/m0_v4_base000/train.out`; `best.pt` shares an mtime with the epoch-1 file).
Any promotion gate keyed on either denoising metric would kill an M1 stage at 2% of its
budget.

There is a compounding cause the plan must control for: **train and validation are drawn
from different source distributions.** Training uses `SourceLengthBatchSampler` with
`source_weights` giving PDB 2/3 of batches; validation uses an unweighted pass over a split
that is 3,797 AFDB against 313 PDB (92.4% AFDB) `[measured]`. Validation loss is measured on
a distribution the model spends one third of its updates on, *and* (after the entropic
transition) under a different sigma distribution.

Fix: add `TaskScoreEvaluator` invoked every `cfg.train.task_eval.every_steps` (default
25,000) on a frozen 256-protein subset, running the existing sampler at 32 steps for
inverse-folding AAR, forward-folding structure bit accuracy, and joint sequence
validity/uniqueness. Composite `= 0.5*AAR + 0.5*struct_bit_acc`, each minus its cached
shuffled-control value so the gate measures conditioning rather than memorisation. Add
`checkpointing.metric = 'task_score'`. Keep both denoising metrics logged. Until the metrics
are shown to agree, **evaluate both `best.pt` and `last.pt` at every gate** and report the
gap as a diagnostic.

### P0-6. The configured task mixture is not the realized mixture

Replay is drawn independently of the task mix (`trainers/trainer.py:1975-1983`) and forces
`sequence_marginal = 1.0` (`trainers/trainer.py:136`). At `sequence_replay_fraction=0.40`
the realized per-example proportions are seq_marg 0.46 / joint 0.18 / fwd 0.12 / inv 0.12 /
struct_marg 0.06 / motif 0.06 `[computed]`. Nobody reading `configs/proteins/m0_v3.py` can
see this, and the trainer never logs realized proportions.

Counted in **residues** rather than examples it is worse: replay rows average 510 residues
against 239 for paired rows under the configured source weights, so replay is **58.7% of
all residue-tokens**, and only **27.5% of residue-tokens carry structure supervision**
against 86.2% carrying sequence supervision `[computed]`. The scarce, weak modality gets the
minority of the gradient.

Two further defects sit in the same mechanism:
- The replay decision is keyed on `global_step`, which is constant across an accumulation
  window (`trainers/trainer.py:1982`). At `grad_accum_steps=2` or `4`, **every optimizer
  step is 100% replay or 100% paired, never a blend.**
- The training loop pulls a paired batch, then **discards it** and pulls a replay batch
  (`trainers/trainer.py:2765-2786`), while still incrementing `_epoch_batches_done`. At 40%
  replay that is 40% of all collation CPU burned for nothing.

Fix: one unified draw over `(corpus, task)` pairs whose weights **are** the realized step
proportions. Key it on a micro-step counter, or better, mix replay rows into each
micro-batch at the collator level. Add a per-1000-step realized-proportion counter logged as
`train/mix/<corpus>/<task>`, asserted at run end to be within 2% of the configured mixture.

### P0-7. Epoch budget counts micro-batches, so the schedule never completes

The epoch budget is enforced on `_epoch_batches_done` (`trainers/trainer.py:2782`,
`2866-2870`), which increments per **micro**-batch, while `global_step` and
`lr_sched.step()` advance only on accumulation boundaries. Resolving the 650M config:
`epochs=80 * steps_per_epoch=5000 / grad_accum_steps=4` = **100,000 reachable optimizer
steps of a configured 300,000** `[computed]`. The cosine then stops at progress 0.324,
leaving the final LR at ~7.6e-5 of a 1.0e-4 peak — **no anneal at all**, and the run exits
normally so nothing flags it. The 400M config reaches 200,000 of 300,000 (67%).

This is a distributed concern because `grad_accum_steps` is exactly the knob that changes
with node count.

Fix: make the epoch budget count optimizer steps, and add a config-invariant test asserting
`epochs * steps_per_epoch // grad_accum_steps >= optim.total_steps` for every
`configs/proteins/*.py`.

### P0 acceptance

One 125M smoke run reproducing the M0 `last.pt` numbers under the new code, giving a single
clean baseline for the whole M1 ladder. Estimated engineering: **1.5-2 weeks**. Given the
flagship itself is only ~40 node-hours, the engineering, not the compute, is the schedule.

---

## 6. Task coverage: what is missing for the six requested capabilities

The user's six capabilities map onto the code as follows.

| Requested | Probabilistic form | Status |
|---|---|---|
| seq -> 3D (forward folding) | `p(z\|s)` | Exists (`forward_folding`) |
| 3D -> seq (inverse folding) | `p(s\|z)` | Exists (`inverse_folding`) |
| de novo co-generation | `p(s,z)` | Exists (`joint`), but see 6.3 |
| motif scaffolding | `p(s,z\|motif)` | Exists but **degenerately trained** (6.1) |
| **seq -> seq** | `p(s_miss\|s_obs)` | **Absent** (6.2) |
| **3D -> 3D** | `p(z_miss\|z_obs)` | **Absent** (6.2) |

`build_example_states` (`data/protein_tasks.py:113-144`) writes a single scalar state into
every residue for all five whole-example modes. The only per-residue mode is `motif`, and
`build_motif_states` (`data/protein_tasks.py:104-109`) is the **only** per-residue OBSERVED
writer in the codebase — and it always writes **both** modality columns. There is no state
grid anywhere in which structure is observed on a span, structure is noisy elsewhere, and
sequence is absent. `grep -rn 'inpaint|infill|redesign|partial_seq'` over `data/`,
`evaluation/proteins/`, `trainers/`, `models/` returns zero task-related hits `[measured]`.

### 6.1 Motif training is a single fixed geometry

`data/protein_multimodal.py:474` calls `T.build_example_states(task, L)` with no spans, so
`_default_motif_span` returns exactly one span covering the middle `L//3`, centred, with
both modalities observed. **Every motif training example in every run so far had identical
geometry** — at the M0 median length 270, always `[90, 180)`. Ten percent of training
compute goes into it, and the evaluation benchmark uses multi-segment, variable-position
motifs.

Important correction to the naive fix: in `evaluation/proteins/scaffolding.py:280-282` the
segment **count and sizes come from the benchmark problem definition**; only the total
design length and the `k+1` gap split are random. So a training sampler that draws `k` from
a categorical and sizes from a Dirichlet would create a *second*, different mismatch. The
right target is: sample `k` and sizes from the empirical distribution **of the benchmark's
own problems**, and reuse `scaffolding.sample_placement`'s multinomial gap split verbatim so
train and eval placement agree by construction.

Also add an independent per-modality observation draw (p=0.15 sequence-only, p=0.15
structure-only, p=0.70 both), since the benchmark includes sequence-only and structure-only
conditioning.

### 6.2 New tasks to implement

Add to `data/protein_tasks.py`:

```python
def sample_observed_mask(rng, length, *, coverage_range=(0.10, 0.90), ...) -> np.ndarray:
    """[L] bool, True where the modality is OBSERVED. Shares span geometry with motifs."""

def build_span_states(length, *, seq_observed=None, struct_observed=None,
                      seq_absent=False, struct_absent=False) -> np.ndarray:
    """[L,2] int8. None mask means NOISY everywhere for that modality."""
```

and register six new task names:

| Task | State grid | Capability |
|---|---|---|
| `sequence_inpaint` | seq observed on spans, struct ABSENT | seq -> seq infilling |
| `sequence_redesign` | seq observed on spans, struct all OBSERVED | targeted redesign at fixed backbone |
| `structure_inpaint` | struct observed on spans, seq ABSENT | **true 3D -> 3D completion** |
| `structure_redesign` | struct observed on spans, seq all OBSERVED | constrained backbone edit at fixed sequence |
| `partial_pair` | two **independent** span draws | "sequence everywhere, structure on 30%" |
| `structure_denoise` | struct all NOISY, started from corrupted native z | SDEdit-style 3D -> 3D |

`structure_denoise` needs a sampler change: `HeunSampler.sample`
(`diffusion/continuous/samplers.py:980-999`) always initialises from pure noise at
`sigmas[0]` (`1043-1047`) with no way to inject a partially noised state. Add `x_init` and
`sigma_start` arguments.

Generation side: extend `build_task_conditioning`
(`evaluation/proteins/generate_multimodal.py:103-175`) with `seq_observed_mask` /
`struct_observed_mask` kwargs and a `states` escape hatch, route them through
`_extract_observed` (178-214), and add the names to `KNOWN_TASKS` (line 77).

### 6.3 Co-generation is sampled off the training distribution

Training draws `sigma_seq` and `sigma_struct` **independently**
(`trainers/multimodal_step.py:41-48`, `independent_modality_noise` defaults True), but the
protein denoiser hardcodes a **single shared sigma** for both modalities at inference
(`generate_multimodal.py:303-304`: `per_bit_sigma_map_torch(states_b, sig, sig)`). So every
co-generation trajectory runs exactly on the diagonal `sigma_seq == sigma_struct`.

With `p_mean=-1.2, p_std=1.2`, only **4.7%** of training examples land within 0.1 log-units
of that diagonal and 11.7% within 0.25 `[measured]` over 2e6 sampled pairs. The model is
asked at inference to do something it saw in about 5% of its training steps.

This is arguably a larger cap on co-generation than P0-2. Fix either by putting explicit
mass on the coupled diagonal during training (add a `coupled_sigma_prob`, e.g. 0.3, that
sets `sigma_struct = sigma_seq`), or by carrying two independent sigma schedules through a
new denoiser signature. Recommendation: **do both** — the training-side fix is one line and
removes the distribution mismatch; the sampler-side change unlocks staggered per-modality
schedules, which is a genuinely novel capability for a bitstream model and a good figure.

### 6.4 Length model for de novo generation

For `p(z)` and `p(s,z)` there is no input to supply `L`. `sample_length_from_prior` is dead
code and **no length prior exists in any manifest** `[measured]`. Today the only source of
`L` is a hardcoded 5-element grid.

Implement: freeze one empirical length prior per source at corpus build time, store it in
`manifest.json`, sample source first then `L` from that source's bucketed prior. Report
headline tables on the fixed grid (100, 200, 300, 400, 500) for comparability, and use the
prior only for aggregate distributional metrics. Log the realized generated-length histogram
against the target.

### 6.5 Classifier-free guidance is both unreachable and untrained

Two independent defects:

1. **Unreachable.** `cfg.evaluation.guidance_scale = 0.0` and **none of the four drivers
   passes the kwarg** — `folding.py:409-417`, `inverse_folding.py:643-651`,
   `scaffolding.py:643-645`, `cogeneration.py:434-436` all omit it `[measured]`.
   `samplers.py:1041` then computes `use_cfg = cond_enabled and guidance_scale > 0.0`, always
   False.
2. **Untrained.** `cfg.cond.p_uncond` is read only in `_step_continuous`; the multimodal step
   **never reads `cfg.cond` at all** `[measured]`. The model has never seen a dropped
   condition. Worse, the sampler's unconditional branch sets observed bits to 0.5 while
   `_MultimodalDenoiser` passes the **same** states grid to both halves of the doubled batch
   (`generate_multimodal.py:295-304`), so the "unconditional" half is still told those
   residues are OBSERVED and still gets sigma 0 for them. The guidance direction is not a
   true conditional-minus-unconditional difference.

Turning guidance up today would make results **worse**, not better.

Fix: add a fourth state code `DROPPED = 3`, extend `MultimodalBitEmbeddings.state_embed`
from 3 to 4 states, draw a per-example `Bernoulli(p_cfg_drop=0.1)` in the multimodal step
that rewrites the OBSERVED column to DROPPED and sets those bits to the null value, and have
`_MultimodalDenoiser` build **two** state grids and select per half of the doubled batch.
Then thread `guidance_scale` through all four drivers. A guidance-sweep helper already
exists for the sequence path (`evaluation/utils.py:746-772`) and only needs calling.

This is the cheapest untried route to better conditional quality and should be swept before
any fine-tuning is considered.

---

## 7. Data plan

### 7.1 Corpus ladder, rescoped for a 2,000 GPU-hour budget

| Tier | Content | Rows (target) | Status | Decision |
|---|---|---|---|---|
| M0 | `airkingbd/pdb_swissprot` | 220,475 | **Built** `[measured]` | Keep as the controlled comparison corpus and the A/B substrate |
| S | UniRef50 sequence-only replay | rebuild to ~8-12B residues | Built but **defective** (7.5) | **Rebuild** — highest value-per-hour item in the data plan |
| M1a | Experimental RCSB monomers | 200k-260k chains | **No script** | Build (7.3) |
| M1b | AFDB representatives | **400k-600k** (scoped down from 1.25M) | Script is a stub (7.2) | Build, scoped |
| Eval | CAMEO 2022, CATH 4.3, motif benchmark, private time split | — | **No scripts** | Build (7.4) — blocking for three of six tasks |
| M2 | ESMAtlas | 2.1M | Unbuildable | **Cut** |

Rationale for scoping M1b down: at 393M parameters, 400M paired structure residues is
already only ~1 structure token per parameter, so the marginal value of going from 600k to
1.25M chains is small relative to its cost (download wall-time, tokenisation GPU-hours, and
a much larger decontamination job). The compute freed is better spent on evaluation and on
three seeds of the 125M development model.

### 7.2 AFDB representatives: two blocking problems

**Dead URL version.** Bump `AFDB_MODEL_VERSION` to 6 and make it discoverable: probe
`AF-{acc}-F1-model_v{v}.cif` for `v in (6,5,4)` once at startup and pin the first 200 into
the manifest. Add a hard gate: if `missing_from_afdb / candidates_seen > 0.05` after the
first 1,000 candidates, **abort**, do not continue. Switch the default fetch to BinaryCIF
(`.bcif`), measured at 156,315 B against 309,542 B for the same entry `[measured]`, and add a
biotite BinaryCIF branch to `_parse_with_biotite`. Replace the serial `urllib` loop
(`:627-715`) with a bounded 32-64 connection thread pool.

Measured serial rate is 6.9 files/s single-connection, i.e. ~50 wall-hours for 1.25M
structures `[measured]`; at 600k chains with 32 connections this is a few hours.

**No cluster list producer.** Write `scripts/proteins/setup/build_afdb_cluster_list.py`
emitting `repid<TAB>avg_plddt<TAB>n_members<TAB>avg_len`. Preferred source is the
Barrio-Hernandez 2023 AFDB/Foldseek cluster tables, but the historical mirror
`afdb-cluster.steineggerlab.workers.dev` now returns **HTTP 404** for the index and both
known filenames `[measured]`, so the script must resolve the current mirror and pin a
sha256. Fallback that needs no third party: pull the AFDB-SwissProt subset
(`swissprot_cif_v6.tar`, 40.06 GB `[measured]`) or `sequences.fasta` (118.0 GB `[measured]`)
and run `mmseqs easy-cluster --min-seq-id 0.3 -c 0.8` yourself.

Transfer estimate for a 600k-chain M1b: **~94 GB as `.bcif`** `[computed]` (the script's own
1.19 TiB estimate is ~3x high because `ESTIMATED_CIF_BYTES` is set to 1,000,000).

### 7.3 Missing preparation scripts

All URLs below were probed live `[measured]`.

| Script | Source | Size | Expected output |
|---|---|---|---|
| `prepare_rcsb_monomers.py` | RCSB Search+Data API for the entry list; `files.rcsb.org/download/{id}.cif.gz` or `rsync.rcsb.org::ftp_data`; `pdb_seqres.txt.gz` (66.0 MB) | 60-80 GB gz, 3-4 GB extracted backbones | 200k-260k chains after 30% identity clustering and a 50-512 filter; carries a hard date-cutoff field so the private time split carves from the same download |
| `prepare_cath43.py` | `dl.fbaipublicfiles.com/fair-esm/data/cath4.3_topologysplit_202206/chain_set.jsonl` (536.9 MB); note `split.jsonl` HEAD returns 403, retry with GET | ~250 MB extracted | ~21,600 chains, ~1,120 in the topology test split. `chain_set.jsonl` already carries N/CA/C/O plus sequence, so this is a reshape |
| `prepare_cameo_targets.py` | CAMEO 2022 target list, or (more reproducible) a deposition-date split carved from the RCSB download | <100 MB | 180-400 targets in the `folding.py:275-299` manifest schema |
| `prepare_motif_benchmark.py` | `zenodo.org/records/15424801/files/motif_scaffolding_pdbs.tar.gz` (9.29 MB, HTTP 200) | <50 MB | The RFdiffusion/FrameFlow 24-problem set in the exact `MotifProblem` schema, with motif residues tokenized through the frozen LFQ encoder |
| `prepare_casp.py` | `predictioncenter.org/download_area/CASP14/targets/`, CASP15 | few hundred MB | ~230 domains, tokenizer reconstruction only, gated on the declared training cutoff |
| `build_decontaminated_splits.py` | — | — | See 7.6 |
| `merge_corpora.py` | — | — | See 7.7 |

### 7.4 Structure tokenisation

The frozen tokenizer cannot currently be loaded by any reusable code path. Move the working
shim from `scripts/proteins/run/decode_m0_dashboard.py:25-52` into
`evaluation/proteins/dplm_struct_tokenizer.py` as a module-level `_install_byprot_compat()`.
Note the repo already carries **two divergent byprot bootstrap implementations**
(`evaluation/proteins/dplm_loader.py:11-50` and the `decode_m0_dashboard` one) — unify them,
do not add a third. Add `openfold` to the `dplm-inference` extra and a smoke test that loads
the tokenizer and encodes one 64-residue chain.

Measured encoder throughput after applying the shim, on an RTX PRO 6000 `[measured]`:
0.0947 s at L=128, 0.1597 s at L=300, 0.2333 s at L=512, i.e. **6.3 chains/s at batch 1**.
For 600k chains that is **~26 GPU-hours** `[computed]`. Batched throughput is **unmeasured**
because no batched code path exists; an earlier claim of 1.7x speedup at B=64 could not be
reproduced and should be treated as unverified. The real lever is data parallelism: add
`--shard-range i/N` so 8 GPUs each own a disjoint slice, reducing 600k chains to **~3.3
wall-hours on one node** `[computed]`.

Two scaling defects in `tokenize_structures.py` must be fixed first:
- `:617-623` rewrites the **entire** `progress.json` and `cache_index.json` on every shard
  flush. At 1.25M chains that is 2,441 flushes, ~2.2 hours of pure JSON serialisation and
  533 GB of metadata rewrites `[computed]`. Replace with append-only `.jsonl` plus a small
  head file.
- `:779-794` writes one `.npy` per chain into a flat directory. Replace with one packed
  uint16 cache per input shard plus an offsets array.

### 7.5 The UniRef50 replay corpus is defective and must be rebuilt

`prepare_uniref50_replay.py:140-142` takes the **first** `max_rows` entries of the EvoDiff
train index, which is ordered by descending length. Measured consequences `[measured]`:

- The first 420k train indices have mean length **2,255** residues against a corpus mean of
  **283.1** and median **195**.
- After truncation at 512, **99.3%** of the 400,000 built rows sit exactly at the cap.
- The corpus is therefore 400k N-terminal fragments of the ~1% longest proteins.
- It uses **1.74%** of available sequence residues (204M built of 11.76B available).

This corpus drives 40% of m0_v3/m0_v4 training and is the **entire** sequence-marginal arm.
Any conclusion about `p(s)` quality drawn from those runs is unsafe.

Fix: `rng.permutation(train_indices)` before the loop; add `--target-residues` in place of
the fixed row cap; stream into shards rather than materialising a Python list; record the
achieved length histogram in the manifest so the regression is visible. Target 8-12B
residues (~30-45M rows, ~12 GB as uint8 seq_ids only). **Wire it into `setup.sh`** — it is
currently not invoked anywhere.

Related: the paired builder has the same class of defect. `prepare_dplm_paired.py:279` does
`length = min(length_full, max_len)`, i.e. it **truncates rather than filters**;
`truncated_rows = 21,574` of 220,475 (**9.79%**) `[measured]`. A truncated structure token
sequence is a chopped domain with a broken C-terminus, and all 21,712 such rows land in the
same exact-length bucket, making it 50x the median bucket size. The plan's stated policy is
a 50-512 **filter**. Decide explicitly; recommendation is to filter, and separately to add a
random-crop augmentation (section 8.7) so long chains are still seen.

### 7.6 Decontamination

There is no decontamination pipeline. `grep` for `decontam` or `dedup` in the protein path
returns nothing `[measured]`. The current M0 split is a per-source cluster hash
(`prepare_dplm_paired.py:230-236`) with **disjoint cluster namespaces** across
`afdb_swissprot` (78,226 clusters) and `pdb` (6,344 clusters) `[measured]`, so cross-source
homology between train and val is entirely unmitigated.

`build_decontaminated_splits.py` must:
1. Export one global FASTA of every candidate training row **across all sources** plus every
   evaluation row (CAMEO, CATH test, motif problems, private time split).
2. `mmseqs easy-search` train against eval; emit a blocklist of train ids with >=30% identity
   to any eval sequence.
3. `mmseqs easy-cluster --min-seq-id 0.3 -c 0.8` over the survivors to produce a **single
   cross-source cluster namespace**.
4. Write `split_assignment.json` (stable_id -> split) plus `decontamination_report.json`.
5. Optional `foldseek easy-search` structural-neighbour audit (report only).

Then change `prepare_dplm_paired.py` and `tokenize_structures.py` to **consume**
`split_assignment.json` instead of hashing per-source cluster ids.

### 7.7 Shard format: the change that makes multi-node possible

Four defects, all in `data/protein_multimodal.py`:

1. **Everything is loaded eagerly.** Lines 316-329 materialise every array of every shard
   *before* the split filter at 331. The val instance is a second full copy of the same
   bytes.
2. **`.npz` cannot be memory-mapped.** `mmap_mode='r'` is silently ignored (verified).
3. **Per-row JSON sidecars.** M0's sidecars are 96.9 MB (439.4 B/row) against ~25 B/row for a
   columnar form — a **17x** reduction — and they create 1.4M+ Python `str` objects that
   break copy-on-write in DataLoader workers `[measured]`.
4. **No corpus union.** `ProteinMultimodalDataset` takes exactly one `shard_dir`. M0 shards +
   tokenized AFDB + RCSB **cannot be composed at load time**, and the 130M/400M/650M configs
   all point at `datasets/bitprotein_monomer_m1`, a path that **nothing in the repo
   produces** — repo-wide grep for `bitprotein` returns that one config line and nothing else
   `[measured]`.

Target format:

```
<corpus>/
  shard_train_00000/{seq_ids,struct_index,seq_mask,struct_mask,offsets}.npy   # uncompressed, mmap-able
  index.npz          # lengths int32, source_id uint8, split uint8, cluster_hash uint64, plddt float32
  sources.json       # source id vocabulary
  manifest.json      # codec convention hashes, schema_version, length priors per source
```

Plus a `merge_corpora.py` that unions several built corpora into one manifest with a
consistent split assignment and a single cluster namespace. **This merge script is the join
point between every corpus builder and the trainer, and it is entirely absent.**

Row schema additions the plan requires and the code does not carry: `residue_mask`,
per-residue pLDDT (the current `RowRecord` stores a **scalar**, so the plan's "mean pLDDT
above 70 and at least 80% of residues above 70" filter is literally unimplementable and
`struct_mask` cannot be set from confidence at residue granularity), `coverage`,
`chain_break_flag`, and decontamination metadata. Also: the field currently named
`coordinate_hash` is computed as `sha256(struct_index)` — it is a **structure-token** hash,
not a coordinate hash, so it cannot detect the same physical structure arriving from two
sources, which is exactly the cross-source dedup the plan calls for. Rename it to
`struct_token_hash` and add a real coordinate hash (`tokenize_structures.py:150-153` already
computes one; just plumb it through). Bump `schema_version` and have the dataset reject older
shards rather than silently misreading.

Projected sizes with the new format `[computed]`: M1 at ~800k rows is roughly 1.2 GB of
arrays plus 20 MB of index, against 2.72 GB in the current format. Per-rank resident memory
drops from ~2.6 GB to near zero (page cache, shared across the node's 8 ranks).

---

## 8. Model and architecture

### 8.1 The good news: attention is over residues

`models/sdt.py:954-959` reshapes `[B, L*18, d]` to `[B, L, 18*d]` before `patch_proj`, so the
transformer sees **L positions, not L*18** `[measured]`. At L=512 that is 512 tokens.
Measured forward FLOPs at 650M/L=512 are 474.0 GFLOP, of which only **6.6% is attention**
`[measured]`; at L=1024 it is 1010.4 GFLOP with 12.4% attention. The model is dense-GEMM
bound, not attention bound. This is why the whole programme is affordable.

### 8.2 Activation checkpointing (blocking)

Absent. `cfg.model.gradient_checkpointing` is set in two configs and read by nothing; the
130M config's comment claims the batch is "kept on device by gradient checkpointing above",
which is false.

Measured with a 6-line `torch.utils.checkpoint(use_reentrant=False)` prototype `[measured]`:

| Config | Without | With |
|---|---|---|
| 650M, L=512, B=16 | 22.76 GiB | 12.26 GiB (-46%) |
| 650M, L=512, B=32 | ~37 GiB (OOM on a shared card) | 14.67 GiB |
| 650M, L=1024, B=16 | — | 12.47 GiB |

Activation memory becomes nearly flat in both B and L, because the 9.54 GiB of static
params+grads+AdamW dominates. Practical conclusion: **with checkpointing, 650M at L=1024 and
B=16 per GPU needs no gradient accumulation at all on an 80 GB card.** Step-time cost is
between +9.5% and +21% across two contended measurements; re-measure on an idle H100.

`use_reentrant=False` is mandatory — reentrant checkpointing plus DDP raises "Expected to
mark a variable ready only once", and with `find_unused_parameters=True` it can silently skip
gradient synchronisation, causing ranks to diverge.

### 8.3 AdaLN-Zero consumes a third of the parameter budget

Measured `[measured]`: at 650M, `adaln1 + adaln2` is 207,207,936 of 639,682,177 parameters
(**32.4%**); at 400M it is 125,952,000 of 392,959,217 (**32.0%**). These are per-example, not
per-token, matmuls, so they contribute almost nothing to per-token FLOPs while consuming a
third of the parameters and a third of the optimizer state.

Equivalently, the "393M flagship" has only ~267M of real trunk capacity. This directly
affects any capacity-matched comparison claim against DPLM-2.

Fix (PixArt-alpha / DiT-Air scheme): one shared `Linear(E, 6E)` plus a per-block learned
offset table `nn.Parameter(torch.zeros(n_blocks, 6, E))`. At 400M this frees ~120M
parameters, buying roughly 6-8 extra layers at the same total budget. **This breaks
warm-start key compatibility, so it must be decided before any long run**, with a converter
added to `utils/protein_warmstart.py`.

### 8.4 The bit-independence problem, and the cheapest experiment in the programme

Recall from section 4: 65.7% bit accuracy but only 0.97% exact-token match, 2.9x above the
bit-independence prediction. The architectural cause is visible in the head. `models/sdt.py:560-574`
maps each residue token to 18 independent hidden vectors and applies the **same** 256->256->1
MLP per slot **with no cross-slot interaction**. The 13 LFQ structure bits — 8,192 codewords
— are each decoded independently from a shared residue vector. That is exactly the regime
where a sampled codeword lands off-manifold and decodes to arbitrary geometry.

Note the asymmetry: intra-patch mixing already exists on the **input** side (`patch_proj` is
a full Linear over `18 * content_dim`). Only the output side is factorised.

Two candidate fixes, both cheap:

- **Intra-patch head block.** After `h = global_feat + local_feat`, reshape to
  `[B*n, 18, hidden]` and apply 1-2 tiny transformer blocks over the slot axis. Cost at
  650M: ~0.8M params (+0.12%) and +3.1% of forward FLOPs `[computed]`. Gate behind
  `cfg.model.head_intra_patch_blocks` (default 0) so it is warm-start compatible.
- **Codebook-aware auxiliary loss.** A soft-argmax over the 8,192 decoded codes, or a small
  auxiliary index cross-entropy head, added as a 2-arm ablation.

**Run this as a 2-arm ablation at 125M before the flagship launches.** It is the single
highest-information experiment available and costs ~60 GPU-hours. If continuous per-bit
diffusion cannot be made to concentrate on the LFQ code manifold, that is a fundamental
limitation of bitstream diffusion on product-quantized latents and the paper's structure-side
claim must be re-scoped — better to know at 125M than at 393M.

### 8.5 Trunk stability (blocking before a multi-week run)

The model contains **zero `nn.LayerNorm` modules** when `use_adaln=True` — `AdaLNZero` uses
functional `F.layer_norm` with no learned affine — and there is **no final normalisation**
between the residual stack and the output projections. There is also **no QK-norm** anywhere
`[measured]`.

Missing final-norm and missing QK-norm are the two most common causes of late-training bf16
loss spikes at 20+ layers. Both are ~10-line, ~zero-parameter additions, and both **change
the state dict**, so they must land before the flagship, not after.

### 8.6 Position and length signal

`cfg.model.abs_pos_mode` resolves to `local_only`, which **zeroes** the global Fourier
features; `rpb_max_distance <= 1` so `self.rpb is None`. The only residue-level position
signal left is trunk RoPE, which is purely relative. Even with `abs_pos_mode='full'` the
global features are **length-normalised** (`denom_g = s-1`), so they encode fraction-of-chain,
never residue index `[measured]`.

Furthermore, all 73 positional channels are functionally dead at P=18: the 64 global channels
are literally zero, and the surviving 9 are functions of `position % P` only, hence constant
within each slot across all residues, making them exactly redundant with
`mm_embed.slot_embed` and `patch_proj`'s bias. That is 18*73*E dead parameters — 1,513,728 at
E=1152 `[computed]`.

**The model cannot count residues and does not know the chain length.** Motif scaffolding
specifies contigs at absolute offsets; de novo generation specifies a target length. Both are
invisible to the content stream. This is a plausible contributor to the weak M0 TM and it is
a checkpoint-shape decision, so it must be made before the flagship. Recommended: add an
explicit length embedding into `t_emb`, plus terminus markers (first/last residue flags), plus
fractional-position features. Cheap, and all three are things the tasks demonstrably need.

Related, cheapest geometry lever: **re-enable relative position bias at residue granularity,
per-head, with log-spaced buckets.** It is 90% built and switched off. Measured costs
`[measured, contended]`: shared-head (current scalar) rpb is free on memory but +20-35% step
time because SDPA drops out of the flash kernel; per-head 16-way bias costs +7.73 GiB and
+65-97% step time. Recommendation: apply the per-head bias to **only the first 4-6 blocks**,
where local structure forms, and combine with activation checkpointing.

### 8.7 Context length and cropping

`cfg.data.max_len = 512` is only a load-time filter (`data/protein_multimodal.py:334`), and
**there is no cropping path anywhere** — longer proteins are silently **dropped**, not
cropped `[measured]`. So raising `max_len` is the only way to see them at all.

Add a random-crop augmentation: a 512-context model can then train on long-protein geometry at
512 cost. Decide explicitly and record the decision in the manifest.

### 8.8 Free compute: the self-conditioning forward is wasted

`trainers/multimodal_step.py:124-137` computes `sc_mask = torch.rand(B) < p_sc`, then runs
`model(...)` on the **whole batch** and only afterwards does `x0_hat[sc_mask] = est[sc_mask]`.
With `p_sc = 0.5` and B >= 16, `P(sc_mask.any()) = 1 - 2^-16 ~ 1`, so the extra full-batch
forward is unconditional in practice. Training is **4x forward, not 3.5x** — every compute
estimate in the old plan is ~14% low, and restricting the no-grad pass to the `sc_mask`
subset is a **free ~12% saving** `[computed]`.

Separately: self-conditioning is **disabled during validation** (`is_train` gate at
`multimodal_step.py:122`), so `best.pt` is chosen in a regime the sampler never uses, while
sampling self-conditions on every one of 250 steps.

### 8.9 torch.compile is a first-class throughput item

With only 6.6% of FLOPs in attention and 97% of parameters in dense GEMMs, the MFU headwind is
memory-bandwidth-bound elementwise work: each of the 52 AdaLN modulations per forward does an
`F.layer_norm` plus two broadcast multiplies plus a gate multiply over `[B, L, E]`, none of
which fuses in eager. Measured achieved throughput at 650M/L=512/B=16 was **49.9 TFLOPS on a
contended RTX PRO 6000** `[measured]`.

The usual objection to compiling this model is the ~470 distinct exact lengths. **Length-band
bucketing removes that objection**, so plan the two together: bucketed padding first, then
compile with a handful of static shapes. Every GPU-hour estimate below hinges on whether the
achieved rate is 10% or 25% of peak.

---

## 9. Training strategy

### 9.1 One joint model, with specialists as controls

**Recommendation: keep the single joint any-to-any model as the primary product. Do not hedge
with a portfolio of task-specialist products.** Train specialists only as scientific controls,
and at most release two LoRA task deltas as clearly separate artefacts.

The argument is from the M0 numbers, not from preference.

*The joint mixture is not what failed.* Where conditioning is measured directly on tokens,
the joint model's cross-modal signal is unambiguous and tight — inverse folding lift 0.300
with CI95 [0.266, 0.334], forward folding lift CI95 [0.114, 0.143]. Both exclude zero widely.
The weakness appears only **after** the decoder. A failure that appears between "the model
predicts the right bits" and "the decoded geometry is right" is a representation/head failure,
and **a task specialist cannot fix a decoder.**

*The joint model has never actually been trained on its nominal recipe.* The realized mixture
was 46% pure sequence-marginal, the sequence corpus was length-degenerate, motif spans were a
single fixed geometry, 12% of steps had leaked ABSENT bits, all ranks drew identical noise,
and checkpoint selection actively picked a checkpoint worth 18 fewer AAR points. Concluding
"the joint mixture is diluting the model" from that evidence would be unsound.

*The joint model is the paper.* A collection of separately fine-tuned models cannot be
described as an all-in checkpoint, per the project's own rule.

**Trade-off, stated honestly:** if specialists later beat the joint model by more than 3 AAR
points or 0.03 TM at matched size and matched **residue** budget, the paper must report that
and the headline narrows from "one model does everything as well as specialists" to "one model
does everything, at a quantified cost". That is still publishable, and it is what DPLM-2
reports for itself.

The good news is that a specialist costs zero engineering: it is a config with
`task_weights = {one_task: 1.0}`.

### 9.2 The stage ladder

| Stage | Model | Corpus | Mixture | Budget | Gate to next |
|---|---|---|---|---|---|
| **S-1** Preflight | 125M | M0 | current | ~10 GPU-h | Reproduce M0 `last.pt` numbers under P0 code; 2-node NCCL and resume smoke pass |
| **S0** Sequence pretrain | 393M (18-bit path, structure ABSENT) | rebuilt UniRef50, ~8B residues | 100% `sequence_marginal` | ~120 GPU-h | Held-out bit MSE plateaued; unconditional 200-mer ESM-2 pppl <= 1.3x natural; 100% valid and unique |
| **S1** Joint flagship | 393M, warm-started from S0 | M0 + M1a + M1b + replay | 3-phase curriculum (9.3) | ~450 GPU-h | Conditional lift CI excludes zero **both directions**; forward-folding TM above shuffled with CI exclusion; joint samples non-collapsed; sequence-only validation within 5% of S0 |
| **C** Controls | 4 x 125M | same | single-task each | ~130 GPU-h | Run **in parallel with S0**, not after S1 |
| **A** Ablations | 125M | M0 | varied | ~150 GPU-h | Head intra-patch, permuted LFQ, entropy base fraction, mixture |

Running S0 makes the warm-start curriculum real rather than dead code. Today **no
width-matched sequence checkpoint exists** — the only one is 384-wide/35.8M
(`runs/proteins/evodiff_uniref50_bitstream_seed0/config.json`) — which is why `m0_v3` and the
400M config both disable warm start with an explicit comment. Because S0 runs the **same
18-bit architecture** with structure ABSENT everywhere, the warm start becomes a plain
`load_state_dict` and `utils/protein_warmstart.py` leaves the critical path entirely (keep the
column-surgery module only for the legacy 384-wide transfer).

S1 warm-start settings: `warm_start_freeze_trunk_steps = 2000`, `warm_start_trunk_lr_mult`
ramping 0.3 -> 1.0 over 20,000 steps. Note the current `trunk_lr_mult` is a **constant
multiplier applied forever** (`trainers/trainer.py:2081-2083`), which at the 35M config's
value of 0.1 is a permanent capacity handicap rather than a gentle restart. Replace with a
ramp; it is a pure function of `global_step`, so resume is automatic.

One caution: enabling the freeze window permanently turns on DDP `find_unused_parameters`
for the **whole** run (`trainers/trainer.py:1199-1208` sets it at construction time and there
is no path that turns it off after unfreezing). That costs 5-10% throughput for the sake of a
2,000-step window. Either re-wrap after unfreezing, or use a **zero-LR trunk group** instead
of `requires_grad=False` so parameters are always used.

### 9.3 Task mixture, as a single distribution over (corpus, task)

Replace the two-stage (replay-draw, then task-draw) design with one normalized distribution
whose values **are** the realized step proportions.

| Phase | Span | joint | fwd_fold | inv_fold | struct_inpaint | seq_inpaint | motif | struct_marg | seq (replay) |
|---|---|---|---|---|---|---|---|---|---|
| A: conditional warm-in | 0-15% | .12 | .24 | .24 | .06 | .04 | .05 | .05 | .20 |
| B: steady | 15-70% | .22 | .17 | .17 | .08 | .06 | .09 | .04 | .17 |
| C: deployment-matched | 70-100% | .28 | .12 | .12 | .09 | .07 | .14 | .03 | .15 |

Rationale. **Conditional-first** because conditional targets are lower-entropy: the model can
lock cross-modal correspondence before being asked to sample both modalities from noise, and
the M0 evidence shows the conditional signal is exactly the part that works. **Joint-and-motif-heavy
at the end** because the terminal task distribution is what the sampler inherits — this is the
direct countermeasure to the DPLM-2.1 folding-SFT finding. **Replay pinned at 15-20%** of all
steps rather than 40%, because epoch capping is a better instrument against overfitting than
dilution.

Implement as `TaskMixtureSchedule`: a list of `(progress_fraction, weight_dict)` breakpoints
with **piecewise-constant** interpolation, so the loader stays deterministic and resumable.
The collator already exposes `set_epoch`; add `set_mixture` and have the trainer call it at
each epoch boundary, persisting the active mixture in the checkpoint. Log per-task loss so
the phase-boundary discontinuities in the loss curve are attributable.

**Do not add per-task difficulty reweighting in the first pass.** The loss already reduces
per modality over its own mask, so a task supervising no structure contributes zero to
`L_struct` and the modality means stay unbiased. A difficulty weight on top would double-count
with the mixture weight and make the two levers non-identifiable.

But be aware of a subtlety that makes the mixture knob **not** the quantity reaching the
optimizer: `multimodal_bit_loss._reduce` divides by a **batch-level** mask sum, so a task's
gradient share is proportional to its target-**bit** count, not its draw probability. Motif,
observing ~1/3 of residues, contributes ~2/3 of a full example's bits. Resolved effective mix
on the sequence head under the current weights: joint 45%, inverse_folding 30%,
sequence_marginal 15%, motif 10% `[computed]`. Every new span task with a variable observed
fraction silently reshuffles the effective mix of the existing tasks. **Log the realized
per-task bit shares, not just the draw weights.**

### 9.4 The loss normalizer is per-rank and per-micro-step

`diffusion/continuous/losses.py:620-628` computes `den = m.sum().clamp_min(1.0)` over the
**local** batch, while the trainer averages per-micro-step losses and DDP averages per-rank
gradients. The optimized objective is therefore a **mean of per-rank ratios**, not the global
masked mean — and it **changes with world size and with `grad_accum_steps`**.

At small per-GPU batches there is a material chance a rank has zero structure-target examples,
in which case `clamp_min(1.0)` silently converts an undefined mean into a **zero** and that
rank still contributes a full `1/world_size` share of the averaged gradient.

Single-node development runs and multi-node flagship runs are therefore not optimizing the
same objective. Fix: all-reduce the per-modality numerator and denominator before dividing.

### 9.5 Model size: why 393M and not 650M

The bottleneck is not capacity.

Paired structure supervision is 61.4M residues at M0 and roughly **200M residues** at the
rescoped M1 (800k rows x ~250). At 393M parameters that is **0.5 structure tokens per
parameter**; at 640M it is 0.31. A Chinchilla-style 20:1 ratio on 200M tokens nominally
supports a ~10M-parameter structure model `[computed]`. Meanwhile m0_v3 at 125M had already
made ~96 row-average passes over the paired corpus by step 135k while validation regressed
monotonically.

That row average badly understates the repetition on the modality that matters. The sampler
assigns **2/3 of all paired batches to the `pdb` source**, which holds only 21,727 of 216,237
train rows. Realized repetition by step 135k is **636 passes over those 21,727 PDB proteins**
(and 35.5 over the 194,510 AFDB rows); for the completed 250k-step run it is **1,178 PDB
passes** `[computed]`. **Any epoch cap must be expressed per source, not per corpus, or it
will be off by an order of magnitude.**

Scaling to 640M multiplies memorisation pressure on exactly the scarce modality. The real
bottlenecks, in order:

1. **Head architecture** — the bit-independence problem of section 8.4.
2. **Conditioning plumbing** — symmetric time embedding, ABSENT leak, no length signal, dead
   position features, untrained CFG.
3. **Data** — 200M paired structure residues is the hard ceiling on what any size can learn.

393M is the largest size defensible against the structure-data budget while still being large
enough that a reviewer cannot dismiss the result as under-trained. **Cut the 650M run.**
Spend the saved ~400 GPU-hours on the unbiased sequence pretrain, the head ablation, and three
seeds of the 125M development model. If a capacity-matched DPLM-2 comparison is demanded
later, run it then, as a scaling point rather than a development vehicle.

### 9.6 Fine-tuning strategy

Order of operations, cheapest first:

1. **Sweep classifier-free guidance and sampler knobs.** Free, and completely untried
   (section 6.5). Sweep `guidance_scale` in {0, 1.0, 1.5, 2.0, 3.0} x NFE in {100, 250, 500}
   x churn in {0, 10, 40}. Report quality and diversity together — added stochasticity may
   improve diversity while hurting recovery.
2. **LoRA task deltas** (rank 16-32, alpha 32, on attention qkv/out and FFN), with the
   modality adapters (`patch_proj`, `unpatch_proj_content`, `head`, `mm_embed`) trained
   **fully**. `peft==0.11.1` is already pinned in `pyproject.toml` and **nothing imports it**
   `[measured]`. The trainer already splits trunk from adapters
   (`trainers/trainer.py:1883-1892`), so this is a small extension. A rank-16 delta on a 393M
   trunk is ~1-2% of parameters, so the base joint behaviour is recoverable by construction
   and every task variant ships as a small delta on **one** base checkpoint.
3. **Full fine-tuning: once, as an ablation only**, purely to quantify the forgetting cost on
   co-generation so the paper can state it.

**Never fine-tune the general checkpoint in place.** Add a regression gate: any released delta
is evaluated on joint co-generation and rejected if it degrades beyond a preregistered
tolerance. Make that a check in `evaluation/build_protein_table.py`, not a documented
intention.

### 9.7 The entropy schedule is a live risk, not a settled recipe

`m0_v3_analsis.log:32-41` measures the pathology directly: the learned entropy mass below
sigma=0.1 is **3.7e-6** against ~18% under the `LogNormal(-1.2, 1.2)` base `[measured]`. EDM
weighting is approximately `1/sigma^2 + 4`, so the loss is dominated by exactly the region the
schedule eliminated. Validation loss rose monotonically from step 50k (4.011) to 135k (6.667)
while training loss fell.

The m0_v4 `base_fraction` sweep {0.00, 0.25, 0.50, 1.00} was the right experiment, but it is
**abandoned, not pending**: the four single-GPU arms were killed at ~step 55,000 (22% of
budget) on 2026-07-22 and never restarted, while one arm — `m0_v4_4GPU_base025` — completed
the full 250,000-step budget on 2026-07-23 and is the most-trained model in the repository
`[measured]`. The sweep also **cannot answer the question**, because it logged only
`edm_loss` and `balanced_mse`, both already known not to track task quality, and the two
disagree about the winner at the last common epoch.

Additionally, the entropy FIFO is **per-rank** with no cross-rank reduction
(`trainers/trainer.py:2044-2051`); rank 0 alone fits the histogram before broadcasting. At
world size 64 the schedule governing all 64 GPUs is fit to 1/64 of the samples. Fix: have
every rank bin its own samples into `num_bins` sum/count tensors and `all_reduce` those two
`[num_bins]` vectors before rank 0 normalises. Cost is 2*128 floats every 2,000 steps.

Note that `multimodal_lfq18_130m/400m/650m` all carry `entropy_compute = False`
`[measured]` — the entropic recipe is **not wired into any foundation-scale config**. Fixing
the all-reduce without first porting the recipe fixes nothing.

**Gate:** do not launch S1 on an entropy schedule that has not been shown to beat the pure
base distribution **on a task metric**. Re-run the base-fraction arms at 125M with the P0
code and `task_score` selection, then adopt the winner in every M1 config.

---

## 10. Distributed execution

### 10.1 What already works

Process-group init from torchrun env vars; DDP with per-rank device binding; gradient
accumulation with `no_sync` and correct 1/accum scaling; rank-0-only atomic checkpoint writes;
collective-safe validation with all-reduce; epoch-end barrier; DDP- and compile-safe EMA
(`utils/ema.py` peels both `.module` and `._orig_mod`).

One genuinely good property worth preserving: `cfg.train.batch_size` is the **global** batch
and is divided by world size at the point of use (`trainers/trainer.py:1034`), while the
sampler multiplies back up (`data/protein_multimodal_sampler.py:146`). The product is
world-size invariant, so **the set of rows in each global batch is identical whether you run 8
or 128 GPUs**. Do not casually replace this with a `micro_batch_per_gpu * world_size` scheme
without also fixing the sampler; the two are a package and the current design already has the
invariance the plan wants.

### 10.2 How many nodes to actually ask for

This is the concrete answer to "1, 2, 3, 4, N nodes".

Three independent walls appear as node count rises:

**Wall 1: per-GPU batch collapses.** `assert cfg.train.batch_size % world_size == 0` then
integer-divides. The 400M config's `batch_size=256` gives 32/GPU at 1 node, 16 at 2 nodes,
8 at 4, 4 at 8, and **2 at 16 nodes**. Measured throughput falls steeply in that regime —
between 3.2x and 4.4x worse per GPU at B=2 versus B=16-32, across two independent measurements
`[measured, contended]`. Also, 3, 5, 6, 7 and 12 nodes **fail the divisibility assert
outright** after sitting in the queue.

**Wall 2: gradient all-reduce dominates.** At 393M, fp32 gradients are 1.572 GB; a ring
all-reduce moves ~3.1 GB per rank per optimizer step `[computed]`. At 128 GPUs with an
effective batch of 512 the per-step compute is ~0.05 s against ~0.10-0.16 s of communication
at a realistic 20-30 GB/s algbw. **More than half the run would be communication**, even with
perfect overlap. There is currently **no DDP communication tuning of any kind** — no
`gradient_as_bucket_view`, no `static_graph`, no bf16 compression hook, no `bucket_cap_mb`
`[measured]`.

**Wall 3: effective batch stops growing.** `SourceLengthBatchSampler` draws
`replace = bucket.size < global_batch` from **exact-length** buckets. Measured bucket depth
`[measured]`: `afdb_swissprot` 463 lengths, median 375; `pdb` **445 lengths, median 37,
minimum 1** — and the configured weights send **2/3 of batches to `pdb`**. Expected distinct
rows per drawn batch: 90.0 at gb=128 (70%), 143.2 at 256 (56%), 190.6 at 512 (37%), 288.1 at
2048 (**14%**) `[computed]`.

Correction worth carrying, because it changes the operational conclusion: this is **not** a
multi-node-only problem — `batch_size` is world-size invariant, so the degeneracy is already
present at one node. And the as-configured effective batch of 512 is reached by **grad
accumulation** (4 independent `batch_id` draws of 128, each a different source and length), so
it contains roughly 4 x 90 = **360 distinct proteins (~70%)**, not 190. The wall bites only if
you scale nodes by **raising `cfg.train.batch_size`** rather than by raising
`grad_accum_steps`.

**Recommendation:**

| Nodes | Verdict |
|---|---|
| 1 | Default for everything: controls, ablations, S0 |
| 2 | Recommended for S1. ~16 examples/GPU, still efficient, halves wall-clock |
| 4 | Acceptable ceiling after the batch-semantics fix, with per-GPU micro-batch held >= 16 via grad-accum |
| 8+ | **Not recommended** for this model at this corpus size. Communication-bound and the effective batch does not grow |

Scale by **raising `grad_accum_steps`, not `batch_size`**, until the sampler is fixed. Add a
scaling table to the config docs: nodes -> micro-batch -> accum -> global batch -> **unique
rows per batch**.

### 10.3 Sampler fix (required before 4+ nodes)

Change from exact-length to **length-band** bucketing with padding: bucket key
`ceil(L / stride)` with stride 32, pad each batch to the band maximum, and emit a real
`residue_mask`. A 32-residue stride turns the 37-row PDB median into roughly 1,200 rows.

Be clear about the cost, because an earlier framing understated it. **The trunk does not
support ragged per-row lengths today.** `SDT.forward` has no length or pad-mask argument at
all; `_pad_to_multiple` only rounds `S` up to a multiple of `P=18` and produces the *same*
trailing pad for every row; and `MultimodalTaskCollator.__call__:445-449` **raises** on a
batch containing more than one length. Length-band bucketing therefore requires: a new forward
argument, a per-row pad mask, the P0-3 mask-inversion fix, pad-aware task masks and per-bit
sigma maps, a pad-aware loss reduction, and updates to the same-length assumption in the
existing tests. This is an **L**, not an **M**.

Also replace `rng.choice(..., replace=...)` with an epoch-level shuffled cursor per bucket, add
per-source epoch cursors, and log the realized repetition factor per source. Today
`set_epoch` is cosmetic — every batch is an independent i.i.d. draw keyed by
`(seed, epoch, batch_id)` — so **no coverage guarantee exists** and any claim about "epochs
over the paired corpus" is a number the sampler does not deliver.

### 10.4 Validation collapses with world size (blocking for multi-node)

`DistributedLengthBucketBatchSampler` truncates each exact-length group to
`(len(group)//world_size)*world_size` and **drops any group smaller than world size**
(`data/proteins.py:810-813`). The M0 validation split is 4,110 rows over 445 distinct lengths,
median bucket **7**, with 33 lengths holding a single row.

Measured coverage by instantiating the real class for every rank at the 400M config's batch
size `[measured]`:

| World size | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| Val rows scored | 100% | 83.2% | **61.3%** | 32.7% | 14.8% | **9.3%** |

At one node (8 GPUs) the metric driving `best.pt` and early stopping is computed on **61% of
the validation set**; at 8 nodes, **9%**. The dropped rows are the rare lengths, so the metric
is length-biased *and* not comparable between a debug run and a production run. This is
blocking, not medium.

Fix: a fixed, allocation-independent validation protocol — pad partial groups with
zero-weight rows and carry the weights through the all-reduced accumulators, or pin a
canonical validation index list whose length is divisible by the maximum supported world size.
Log the retained fraction at startup.

Note the same truncation applies to the **train** split of the DiMA baseline
(`data/proteins.py:830-841` selects it unconditionally for both splits, unlike the UniRef50 and
multimodal loaders). Since DiMA is a comparison arm for the headline claim, that is an
unfair-comparison risk, not just an efficiency issue.

### 10.5 Checkpointing at 4-hour walltime

Checkpoints are **16.0 bytes/parameter** (fp32 model + fp32 EMA + AdamW `exp_avg` and
`exp_avg_sq`), verified against a measured 2,003,472,446-byte file for a 125,192,769-parameter
model — 0.02% error `[measured]`. So: **6.3 GB at 393M**, 10.2 GB at 640M.

Problems at scale:

- **Synchronous rank-0 save stalls the world.** Measured 6.9 s for a 650M state on local NVMe
  with page cache (1.38 GiB/s) `[measured]`; on a contended shared filesystem with 64 ranks
  also reading, 20-90 s is realistic `[estimated]`. The process-group timeout is hardcoded at
  **20 minutes** (`train.py:20`).
- **Three copies on a new-best epoch.** `last.pt` + `epoch=NNNN-val=X.pt` + `best.pt` = up to
  19 GB of serial writes at 393M. `interval.keep_last = 0` in the resolved configs, so the
  prune branch **never runs** `[measured]`. Projected ~113 GB per 393M run.
- **Every rank `torch.load`s the entire checkpoint on resume.** `trainers/trainer.py:1668` has
  no `mmap=True` and no rank-0-load-plus-broadcast. At 393M on 2 nodes that is **100 GB of
  shared-filesystem reads at the start of every one of ~15 chain links** `[computed]`.

Fixes: pinned-CPU state copy plus a background writer thread joined before the next save;
explicit `dist.barrier()` after the save so ranks stall at a barrier rather than mid-collective;
`cfg.system.dist_timeout_minutes` defaulting to 60; keep the EMA shadow on CPU permanently
(saves 1.6 GB of GPU memory at 393M as well); `torch.load(..., mmap=True)` or local-rank-0
stage-then-share on resume; finite `interval.keep_last`; and drop optimizer state from archival
`step=*.pt` checkpoints, halving them.

**Time-based checkpointing is required.** The current cadence is `every_steps`; at 4-hour
walltime the cadence must be wall-clock (`every_seconds`, default 900) plus a deadline check.
Add `cfg.train.max_wall_seconds` read from `SLURM_JOB_END_TIME`, checked at the accumulation
boundary, doing barrier-save-barrier-exit at T-10 minutes. **A trainer that knows its deadline
beats one that races a signal.**

### 10.6 Preemption handling

There is **no signal handling anywhere** — `grep` for `signal`, `SIGTERM`, `SIGUSR1` across
`train.py`, `trainers/`, `utils/` returns only `scipy.signal` `[measured]`. The `finally`
block flushes TensorBoard and W&B but **never checkpoints**, and `except KeyboardInterrupt`
re-raises without saving. Slurm's default preemption signal is SIGTERM, which kills Python
without running `finally` at all.

Add `utils/slurm_signals.py` with a flag-setting handler (never `torch.save` from inside a
signal handler — it can interleave with a CUDA launch or an in-flight NCCL collective), and
check the flag at the optimizer-step boundary with an `all_reduce(MAX)` so every rank agrees
before anyone exits. Check every `COBIT_PREEMPT_CHECK_EVERY` steps (default 20) to keep the
extra collective off the hot path.

Budget: 6.3 GB at 400 MB/s to shared storage is **16 s**, comfortably inside a 180-second
grace window `[computed]`.

### 10.7 NCCL and process-group hygiene

**No NCCL environment variables are set in any training launcher** `[measured]`. The only
`NCCL_*` in the repo is `NCCL_P2P_DISABLE=1` in `scripts/owt/eval_cobit_m.sh:50`, a
workstation workaround that would be actively harmful on an NVLink cluster. On a multi-homed
node NCCL will auto-select an interface and can pick a management NIC or a `docker0` bridge,
producing a hang that looks like a data-loader stall. At 10 GbE the 393M gradient all-reduce
would be **87% of step time** `[computed]`.

Also: `train.py:24` calls `torch.cuda.set_device` **after** `init_process_group` (line 20) and
never passes `device_id`. PyTorch is already warning about this in production logs — verbatim
from `runs/proteins/m0_v3/train.out` `[measured]`:

```
[rank0]:[W720 21:33:21] ProcessGroupNCCL.cpp:5188] Guessing device ID based on global rank.
This can cause a hang if rank to GPU mapping is heterogeneous.
```

The repo already gets the order right elsewhere (`evaluation/distributed.py:41-51`); copy that
pattern.

### 10.8 FSDP: document, do not build

Measured static state under DDP + AdamW + EMA is **24 bytes/parameter** (4 param + 4 grad + 8
AdamW + 4 EMA + ~4 DDP bucket) `[computed]`, and measured peak at 650M/B=32/L=512 is ~42 GB
including activations. Reserving 25 GB for activations on an 80 GB card, **FSDP becomes
necessary at roughly 2.3B parameters** — 3.5x the largest size this plan contemplates —
rising to ~2.75B with the EMA on CPU and ~4B with `ZeroRedundancyOptimizer` `[computed]`.

**Do not implement FSDP.** It would add sharded-checkpoint complexity, break rank-0-only saves
and the warm-start surgery, and buy nothing. If optimizer memory ever becomes the constraint,
`torch.distributed.optim.ZeroRedundancyOptimizer` is ~20 lines and keeps DDP semantics.

### 10.9 Config resume must not freeze machine shape

`train.py:77-82` merges the saved `runs/<exp>/config.json` over the freshly imported Python
config whenever `last.pt` exists, skipping only `logging`, `system` and `evaluation`. On a
20-link chain this means **extending `total_steps`, correcting a mixture, or fixing the
checkpoint metric requires hand-editing a JSON file inside the run directory**, and a submitted
config change is silently ignored with only "Resuming experiment using saved config" printed.

What is genuinely dangerous is narrower than it first appears, because `batch_size` is
world-size invariant. The problematic persisted keys are `data.shard_dir`,
`data.sequence_replay_root`, `data.root` and `data.num_workers` — if link 1 staged to a
per-job scratch path, link 2 points at a directory that no longer exists and **every rank
crashes simultaneously with FileNotFoundError**.

Fix: an allowlist of resume-mutable keys re-derived from the environment **after** the merge
(`data.*` paths, `num_workers`, `grad_accum_steps`, `checkpointing.*`), with hard assertions
that `optim.total_steps`, `global_batch_size` and every `model.*` key match the saved values.
Use a **stable** node-local staging path (`/tmp/$USER/cobit`, not per-job) so an old path stays
valid across a requeue.

---

## 11. Slurm launchers for a no-GRES cluster

### 11.1 What breaks today

`scripts/launch/train_slurm.sbatch:5` declares `#SBATCH --gres=gpu:4`, which a no-GRES Slurm
rejects outright. Worse, if the directive is simply deleted, line 41
(`GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}"`) falls through to the
literal **4**, because both variables are populated only by the GRES plugin. On 8-GPU nodes
the job then **runs successfully at half the allocation**, prints `gpus_per_node=4`, and the
world size — and therefore the derived local batch and every token-budget number — is silently
wrong by 2x. That is a worse failure than a hang.

Other issues: no `--exclusive` (without GRES arbitration two co-scheduled jobs both see all 8
GPUs and bind the same devices); no `--signal`; `OMP_NUM_THREADS` set from
`SLURM_CPUS_PER_TASK` with `--ntasks-per-node=1`, giving every one of 8 ranks 32 threads;
all nodes writing into one interleaved log; `logs/` is **gitignored**, so on a fresh clone
Slurm fails to open the output file *before* the script's own `mkdir -p logs` ever runs; and no
W&B credential sourcing (present in `train_4gpu.sh` but absent from the sbatch).

Feature inventory in the repo today `[measured]`: `--gres` 3 files, `--dependency` 1,
`--requeue` 1; `--exclusive` **0**, `sbcast` **0**, `--signal` **0**, `USR1` **0**,
`scontrol requeue` **0**, launcher tests **0**.

### 11.2 Structure

```
scripts/launch/nogres/
  common.sh                    # sourced helpers: GPU discovery, rendezvous, CPU/NCCL env, staging
  submit.sh                    # the ONLY supported submission path; creates logs/slurm first
  train_multinode.sbatch       # torchrun variant (default)
  train_multinode_srun.sbatch  # pure-srun variant, one task per GPU
  srun_rank_shim.sh            # SLURM_PROCID -> RANK translation
  chain.sh                     # --dependency=singleton chaining with a progress guard
  preflight.sbatch             # 2-node, 100-step gate
  make_data_tarball.sh         # build the staging payload once
  site.env.example             # copy to ~/.cobit_site.env
```

### 11.3 Runtime GPU discovery

The core of the no-GRES change. Count devices **on every allocated node** and refuse to launch
if heterogeneous, because `torchrun --nproc_per_node` is a single number and a mixed allocation
deadlocks at the first collective rather than failing fast:

```bash
cobit_detect_gpus_alloc() {
  local counts n
  counts=$(srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 \
            bash -c 'nvidia-smi -L 2>/dev/null | grep -c "^GPU "' 2>/dev/null | sort -u)
  n=$(wc -l <<<"$counts")
  if [[ "$n" -ne 1 || -z "$counts" ]]; then
    echo "cobit: heterogeneous or undetectable GPU counts: [$(tr '\n' ' ' <<<"$counts")]" >&2
    return 1
  fi
  [[ "$counts" -lt 1 ]] && { echo "cobit: allocation reports 0 GPUs per node" >&2; return 1; }
  printf '%s\n' "$counts"
}
```

`CUDA_VISIBLE_DEVICES` takes precedence over `nvidia-smi` when set, since a site prolog may pin
a subset. Every caller treats 0 as fatal; there is no silent fallback.

### 11.4 Primary template (torchrun)

```bash
#!/usr/bin/env bash
#SBATCH --job-name=cobit_m1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/slurm/%x_%j.driver.log
#SBATCH --error=logs/slurm/%x_%j.driver.log
#
# N nodes x (all GPUs on the node), NO --gres.
# Submit only via scripts/launch/nogres/submit.sh, which creates logs/slurm/ before
# sbatch opens the output file and injects --account/--partition from the site env.
#
#  * --exclusive is MANDATORY: without GRES arbitration a shared node lets a second
#    job's ranks bind the same physical GPUs.
#  * GPUs per node are discovered at runtime; SLURM_GPUS_ON_NODE does not exist here.
#  * --signal=B:USR1@180 lands in the batch shell; the trap forwards USR1 to the job
#    step so every rank writes last.pt, then requeues.

set -uo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/launch/nogres/common.sh"
[[ -f "$HOME/.cobit_site.env" ]] && source "$HOME/.cobit_site.env"

CONFIG="${1:-${COBIT_CONFIG:-configs/proteins/m1_joint_400m.py}}"
[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 2; }

LOGDIR="$REPO_ROOT/logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"; mkdir -p "$LOGDIR"
PYTHON="$(cobit_python)"
NNODES="${SLURM_NNODES:?}"
GPUS_PER_NODE="$(cobit_detect_gpus_alloc)" || exit 3
WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))

cobit_rdzv_env                       # MASTER_ADDR from scontrol; MASTER_PORT from job id
cobit_cpu_env "$GPUS_PER_NODE"       # OMP_NUM_THREADS = min(8, cpus/gpus); dataloader workers
cobit_nccl_env                       # NCCL_SOCKET_IFNAME, timeouts, PYTORCH_CUDA_ALLOC_CONF
cobit_cache_env                      # per-rank Triton/Inductor cache dirs
cobit_wandb_env                      # WANDB_MODE=offline by default

STAGE_TAR="${COBIT_STAGE_TAR:-$REPO_ROOT/datasets/_stage/cobit_data.tar}"
cobit_stage_data "$STAGE_TAR" "$(cat "$STAGE_TAR.sha256" 2>/dev/null || echo nostamp)" || exit 4

export COBIT_NNODES="$NNODES" COBIT_GPUS_PER_NODE="$GPUS_PER_NODE" \
       COBIT_WORLD_SIZE="$WORLD_SIZE" COBIT_LOGDIR="$LOGDIR" PYTHON

cat <<EOF
=== cobit no-GRES launch =========================================
job            : ${SLURM_JOB_NAME} / ${SLURM_JOB_ID} (restart ${SLURM_RESTART_COUNT:-0})
config         : ${CONFIG}
nodes          : ${NNODES}  (${SLURM_JOB_NODELIST})
gpus per node  : ${GPUS_PER_NODE}   [runtime-discovered, no GRES]
world size     : ${WORLD_SIZE}
rendezvous     : ${MASTER_ADDR}:${MASTER_PORT}
threads        : OMP=${OMP_NUM_THREADS} dataloader=${COBIT_DATALOADER_WORKERS}
data root      : ${COBIT_DATA_ROOT}
==================================================================
EOF

_forward_usr1() {
  echo "[$(date -Is)] USR1 in batch shell: draining job step, then requeueing."
  scancel --signal=USR1 "$SLURM_JOB_ID" || true   # targets job STEPS, not this shell
  sleep "${COBIT_CKPT_GRACE:-150}"
  [[ "${COBIT_SELF_REQUEUE:-1}" == "1" ]] && { scontrol requeue "$SLURM_JOB_ID" || true; }
}
trap _forward_usr1 USR1
trap 'scancel --signal=TERM "$SLURM_JOB_ID" || true' TERM

srun --ntasks="$NNODES" --ntasks-per-node=1 \
     --cpus-per-task="${SLURM_CPUS_ON_NODE:-1}" \
     --kill-on-bad-exit=1 --unbuffered \
     --output="$LOGDIR/node-%n.out" --error="$LOGDIR/node-%n.out" \
     bash -c '
       set -euo pipefail
       echo "[$(date -Is)] node=$SLURM_NODEID host=$(hostname) gpus=$COBIT_GPUS_PER_NODE"
       exec "$PYTHON" -m torch.distributed.run \
         --nnodes="$COBIT_NNODES" --node_rank="$SLURM_NODEID" \
         --nproc_per_node="$COBIT_GPUS_PER_NODE" \
         --rdzv_backend=c10d --rdzv_id="$SLURM_JOB_ID" \
         --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
         --rdzv-conf=timeout=1800 --max_restarts=0 \
         --log-dir="$COBIT_LOGDIR/torchrun-node$SLURM_NODEID" --redirects=3 --tee=1 \
         train.py --config "'"$CONFIG"'"
     ' &
SRUN_PID=$!
wait "$SRUN_PID"; RC=$?
while kill -0 "$SRUN_PID" 2>/dev/null; do wait "$SRUN_PID"; RC=$?; done
echo "[$(date -Is)] srun exited rc=$RC"
exit "$RC"
```

`--max_restarts=0` is deliberate: elastic restart is unsafe here because the sampler cursor
lives in the checkpoint, not in the rendezvous.

### 11.5 Pure-srun variant

Some no-GRES sites forbid nested launchers (torchrun forking 8 children inside one srun task
confuses cgroup accounting and cpu-binding). **`train.py` needs no change** — line 15 tests
only for `RANK`/`WORLD_SIZE` in the environment and `init_process_group` uses the default
`env://` method, so `MASTER_ADDR`/`MASTER_PORT` exported by the batch shell and inherited via
`srun --export=ALL` are sufficient. A 5-line shim suffices:

```bash
#!/usr/bin/env bash
# scripts/launch/nogres/srun_rank_shim.sh
# Translate Slurm's per-task identity into torch's env:// contract, then exec.
set -euo pipefail
export RANK="${SLURM_PROCID:?must be exec'd by srun}"
export WORLD_SIZE="${SLURM_NTASKS:?}"
export LOCAL_RANK="${SLURM_LOCALID:?}"
export GROUP_RANK="${SLURM_NODEID:-0}"
export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE:-${COBIT_GPUS_PER_NODE:-1}}"
: "${MASTER_ADDR:?not inherited}"; : "${MASTER_PORT:?not inherited}"

# If a site task-prolog pinned exactly one device per task, the only valid local
# index is 0; without this, ranks 1..7 raise "invalid device ordinal".
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  _n=$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | grep -c '[0-9A-Za-z]')
  [[ "$_n" -eq 1 ]] && export LOCAL_RANK=0
fi
# Per-rank scratch so 8 ranks do not race on one Triton/Inductor cache dir.
for v in TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR; do
  [[ -n "${!v:-}" ]] && { export "$v=${!v}/r${RANK}"; mkdir -p "${!v}"; }
done
exec "$@"
```

The sbatch deliberately omits `#SBATCH --ntasks-per-node` (the GPU count is unknown until the
allocation runs) and passes it to `srun` at runtime, with
`--distribution=block:block --cpu-bind=cores`.

Note this variant is **better** for preemption: Slurm delivers USR1 directly to every rank,
which is exactly what the handler wants.

### 11.6 Chaining under a 4-hour limit

Use `--dependency=singleton` with a fixed job name rather than tracking job ids across 20
links. Slurm then runs exactly one job with that name at a time, each resuming from `last.pt`.

Add a **progress guard**: at job start, read `global_step` from `last.pt` and compare against
the value recorded by the previous link. If it has not advanced, `scancel` the remaining chain.
Without this, a config error burns the entire chain — 20 links x 2 nodes x 4 hours is 1,280
node-hours reproducing the same crash, because `--dependency=afterany` advances on failure by
design.

Also run a **dry-run config import on the login node** before submitting
(`WORLD_SIZE=nodes*gpus python -c 'import cfg; apply_env_overrides(cfg)'`), so a node count that
does not divide the global batch fails in one second instead of after hours in the queue.

### 11.7 Node-local staging without node-local NVMe

With shared-filesystem-only, "node-local" means `/tmp` (usually a local disk or overlay) or
`/dev/shm` (RAM-backed, counts against node RAM). Both are per-node, which is what matters:
the goal is to turn `N*8` shared-FS readers into `N`.

Use `sbcast` to broadcast one tarball over Slurm's hierarchical tree, then extract once per
node under a `.stamp` guard so a requeue on the same node is a no-op. For a 2-node job that is
one shared-FS read instead of 16.

Current payload: 1.49 GB (M0 393 MB + replay 1.1 GB) `[measured]`. After the corpus rebuild,
budget ~3-4 GB.

Use a **stable per-user path** (`/tmp/$USER/cobit`), not per-job, because a chained job must
find what the previous link staged and `cfg.data` paths are frozen into `config.json` on the
first launch.

Verification: a `verify_staged.sh` that checks a sha256 per node, since a silently truncated
extract on one node produces a corrupt-data failure that looks like a model bug.

### 11.8 Preflight gate

Every failure in section 11.1 manifests within two minutes of a real 2-node run and is
completely invisible to the existing in-process tests. `preflight.sbatch` runs 100 real steps
of the target config on 2 nodes and reports: discovered GPUs per node, resolved
`(local_bs, world_size, grad_accum, effective_global_batch, reachable optimizer steps vs
total_steps)`, NCCL all-reduce algbw for 256 MB, step time, peak memory, checkpoint save
latency, and a USR1 round-trip proving `last.pt` mtime advances after the signal.

**Require it to pass before submitting any chain.** Ten minutes of queue protects hundreds of
GPU-hours.

---

## 12. Compute budget

### 12.1 Throughput basis

Measured anchor `[measured]`: 125M model, one 96 GB Blackwell card, micro-batch 128,
`grad_accum 2`, 1.75-2.13 micro-iterations/s from `runs/proteins/m0_v4_base050/train.out`.
That is ~63,000-67,000 residue-tokens/s, implying **~67 TFLOP/s achieved** at 4x-forward
accounting `[computed]` — roughly **7-10% MFU** on that card.

Planning figure for H100: **120 TFLOP/s achieved** (~12% of 989 peak) `[estimated]`, rising to
perhaps 200-250 with activation checkpointing, bucketed padding and `torch.compile`. All
figures below use 120 and are therefore conservative. **These carry +/-40% uncertainty** —
every throughput measurement available was taken on contended GPUs. Re-measure in the
single-tenant preflight before this drives a compute request.

Independent cross-check `[measured]`: 4-GPU DDP at micro-batch 32/GPU consumed ~105-110
GPU-hours per 250k steps while 1 GPU at micro-batch 128 consumed ~64-65 for the same work —
**1.6-1.7x worse per GPU-hour**. Multi-GPU scaling that shrinks the per-GPU batch is not free.

### 12.2 Ledger

| Item | GPU-hours | Basis |
|---|---:|---|
| P0 preflight and smoke runs (S-1) | 40 | 125M, several short runs |
| Tokenizer audit | 5 | `[computed]`, encoder-bound |
| M1 structure tokenisation (600k chains) | 30 | `[computed]` from 6.3 chains/s measured |
| Eval-set tokenisation (CAMEO, CATH, motif) | 5 | `[computed]` |
| **S0** sequence pretrain, 393M, ~8B residues | 130 | `[computed]` at 120 TFLOP/s, 4x forward |
| **S1** joint flagship, 393M, 300k steps | 450 | `[computed]`; ~28 wall-hours on 2 nodes |
| Controls: 4 x 125M single-task, residue-matched | 130 | `[computed]` |
| Ablations: head intra-patch (2 arms), permuted LFQ, entropy base fraction (2 arms), mixture | 150 | `[computed]` |
| Three 125M seeds of the development model | 100 | `[computed]` |
| **Evaluation**: 2 full matrix passes | 300 | `[computed]`, see 12.3 |
| Preemption and restart overhead at 12% | 165 | `[estimated]` from a 4-hour walltime with chaining |
| **Subtotal** | **1,505** | |
| Reserve (re-runs, failed launches, a third eval pass) | 495 | |
| **Total** | **2,000** | |

Two observations the plan should act on.

**Evaluation is 15-20% of the budget and is currently single-process.** It must be sharded
(section 11 of the eval workstream) or it becomes the wall-clock bottleneck regardless of how
fast training is.

**Wall-clock, not GPU-hours, is the binding constraint.** S1 at 450 GPU-hours is 28 wall-hours
on 2 nodes, which is **8 chained 4-hour jobs** plus queue time. The whole programme is roughly
190 node-hours, i.e. ~50 chain links on 2 nodes. Checkpoint/resume reliability is worth more
than throughput optimisation here.

### 12.3 Evaluation cost

One full matrix pass on one checkpoint `[computed]`:

| Component | GPU-hours |
|---|---:|
| Diffusion sampling, all six tasks | 68 |
| ESMFold self-consistency (~54,600 folds) | 53 |
| Inverse folding on CATH (sampling) | 36 |
| Forward folding on CAMEO (sampling) | 15 |
| Motif scaffolding (24 problems x 100) | 19 |
| Co-generation, 5 lengths x 100 | 9 |
| **Total per checkpoint** | **120-210** |

The upper bound assumes ESM-2 650M exact pseudo-perplexity as currently coded, which alone is
**88 GPU-hours** for a single sequence-only table — one 650M forward *per residue*, over 1,000
sequences `[computed]`. **Cap or subsample it**; it is the single largest avoidable line item
in the evaluation budget.

With sharding across 8 GPUs a pass is 15-26 wall-hours; without it, **5-9 days on one GPU**.

---

## 13. Evaluation stack

### 13.1 Offline staging (blocking)

Compute nodes have no internet. Today `self_consistency.py:127-129` and `metrics.py:189-197`
call `from_pretrained('facebook/esmfold_v1')`, `metrics.py:124` calls ESM-2 650M,
`metrics.py:159-164` calls two Rostlab ProtT5 repos, `generate_dplm.py:40` calls
`airkingbd/dplm_150m`. **None are cached** — `hf_cache/hub` holds only `esm2_t6_8M` and the
pdb_swissprot dataset `[measured]`. Every one raises on a compute node.

Also: the fair-esm fallback at `self_consistency.py:149` is **dead**, because `openfold` is
missing from `.venvs/dplm`.

Required:
- Extend `cache_evaluation_models.py` with ProteinMPNN (a git clone plus weights copy, not an
  HF snapshot).
- Replace `snapshot_download('Rostlab/prot_t5_xl_uniref50')` with
  `allow_patterns=['*.model','*.json','spiece*']` — today it pulls **~11 GB** to obtain a ~1 MB
  sentencepiece model `[computed]`.
- Add `HF_HOME`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `TORCH_HOME` to a sourced env
  file used by every launcher. Nothing sets these today.
- Add `--verify` mode that loads every model once on the login node and writes a
  sha256+size manifest, so a missing shard fails at stage time rather than six hours into a job.

Note `facebook/esmfold_v1` via transformers is **~14.2 GB** (3.53B params fp32), not the ~2.8 GB
the old plan states — that figure is the fair-esm trunk-only file and understates the
transformers route ~5x `[computed]`. Total offline pre-stage: **~35 GB of models and binaries
plus reference DBs**.

Already cached locally and worth reusing: ESM-2 3B (5.3 GB) and ESM-IF1 (1.6 GB), the latter
referenced by **no code** `[measured]`.

### 13.2 External binaries

`which foldseek mmseqs USalign TMalign mkdssp dssp` returns **nothing**, and
`setup_evaluation.sh` installs none of them (its profiles are only core/evodiff/dplm/all)
`[measured]`. Add a `tools` profile: pinned static foldseek and mmseqs2 tarballs with sha256
verification (a 25 MB mmseqs `.deb` already sits at `/home/brunod/` and can be `dpkg -x`'d
without root), US-align compiled with one `g++ -O3`, and mkdssp or vendored pydssp. Export
`TOOLS_BIN` on `PATH` from one sourced env file.

Add `foldseek_search()` and `mmseqs_search()` easy-search wrappers next to the existing
easy-cluster ones — both novelty **and** decontamination need them.

### 13.3 Sharding and resumability (blocking)

`grep -rn 'torch.distributed|init_process_group|LOCAL_RANK|SLURM' evaluation/proteins/*.py`
returns **zero hits** `[measured]`, and no driver has `--shard-id`/`--num-shards`/`--resume`.
Meanwhile a complete distributed-eval toolkit already exists and is used only by the text path:
`evaluation/distributed.py` (205 lines) provides `init_distributed_if_needed`, `shard_count`,
`all_gather_int`, `broadcast_*` and `gather_varlen_firstdim_to_rank0`.

Add a common `--shard-id/--num-shards` pair to `folding.py`, `inverse_folding.py`,
`cogeneration.py`, `scaffolding.py` and `tokenizer_audit.py` that deterministically partitions
the work-unit list, writes per-shard JSONL into `<out>/shards/`, and skips items whose record
already exists. Add `evaluation/proteins/merge_shards.py`. Add an `--esmfold-batch` path —
`self_consistency.py:136,158` folds **one sequence per call**.

Ship `scripts/launch/nogres/eval.sbatch` using the same whole-node discovery, with
`SLURM_PROCID -> shard-id`. Critically, **keep ESMFold off the master-only callback path**: the
epoch-end sequence runs rank-0 callbacks immediately before a `dist.barrier()`, so any long
callback there trips the process-group timeout and kills the allocation.

### 13.4 3D -> 3D evaluation has no driver at all

The M0 dashboard's `structure_to_structure` key
(`scripts/proteins/run/evaluate_m0_dashboard.py:241`) is a **misnomer**: it reports the
unconditional structure marginal (token JSD, unique-token fraction, position-matched bit
accuracy), not any conditional completion `[measured]`.

Build `evaluation/proteins/inpainting.py`: for each held-out chain, mask contiguous structure
spans at 10/25/50/75% deletion plus a loop-only and a domain-terminus regime. Report
**context-superposed** per-region RMSD and lDDT of the completed span — superposing on the
observed context, not globally, is what distinguishes completion from re-folding — plus
junction CA-CA continuity, seam clash count, whole-chain TM to native, and LFQ bit accuracy on
the masked region.

### 13.5 A prerequisite nobody has noticed

`foldseek_cluster` takes a `pdb_dir`, and the **only PDB writer in the repository** is
`scripts/proteins/run/decode_m0_dashboard.py:69`, which emits `ALA A{residue:4d}` for **every
residue** `[measured]`. Structural diversity and novelty by Foldseek clustering is not merely
unmeasured — it is unreachable until a real PDB writer emits the co-generated sequence's
three-letter codes. Schedule that before any structural novelty claim.

---

## 14. Code restructuring recommendations

Ordered by value, with an honest verdict on whether each is worth doing before M1.

### 14.1 Do before M1

**Config system.** Measured duplication `[measured]`: `multimodal_lfq18_400m` has 211 leaf
keys, of which **152 (72%) are byte-identical to a text-only Swiss-Prot config** — including
`data.root='datasets/swissprot'`, `data.fasta_name='uniprot_sprot.fasta.gz'`,
`data.tokenizer='char'`, an 8-key `cfg.cond.*` block and a 21-key `cfg.train.vlb.*` block with
`enabled=True` that the multimodal path silently ignores. The four `swissprot_*.py` files total
740 non-comment lines with only 224 unique. A reader of the M1 config cannot tell which keys are
live, and `config.json` and W&B record 152 misleading values as run parameters.

Replace the inheritance chain with `configs/proteins/base/build.py` exposing
`build_protein_config(model_size, corpus, mixture, budget)` plus a `finalize(cfg)` that derives
every evaluation path from `cfg.experiment`. Target: `m0/m1 x size x node-count` expressible
without 28 near-copies.

There is a concrete bug this fixes: `multimodal_lfq18_130m/400m/650m` all inherit
`cfg.evaluation.checkpoint_path = 'runs/proteins/swissprot_char5/checkpoints/last.pt'` and
`out_dir = 'runs/proteins/multimodal_lfq18_smoke/protein_eval'` and never re-point them
`[measured]`. Running evaluation with the M1 config would **score a text-only Swiss-Prot
checkpoint and overwrite the smoke run's outputs**. `m0_v3.py:158-169` documents and patches
this locally, but the base was never fixed, and `m0_v4_preflight.py` has already regressed.

**Contract tests.** A single `tests/test_config_contracts.py` that, for every
`configs/proteins/*.py`, asserts: eval paths start with `runs/{cfg.experiment}`; reachable
optimizer steps >= `optim.total_steps`; declared `expected_num_parameters` matches the
instantiated model (currently asserted **nowhere** — any of the recommended layout changes can
silently change model size and invalidate a capacity claim); and the mixture sums to 1.

**Data format.** Section 7.7. Blocking for multi-node.

**Launchers.** Section 11. Blocking.

### 14.2 Defer until after M1

**Decomposing `trainers/trainer.py`.** It is 3,091 lines and genuinely a god-object:
loader construction, DDP setup, warm-start curriculum, replay mixing, entropy scheduling,
checkpointing, validation aggregation and callbacks all live in it. A clean decomposition
would be `Trainer` + `DistributedContext` + `CheckpointManager` + `LoaderBuilder` +
`MultimodalRoute`.

**Verdict: do not do this before M1.** The multimodal path is intertwined with the text path
through ~15 `is_multimodal` branches, and every one of those branches is currently the only
thing keeping the text runs byte-identical. A refactor of that size, landing at the same time
as the seven P0 correctness fixes, would make it impossible to attribute any behaviour change.
Land P0 first, get one clean 125M baseline, **then** decompose against that baseline as a
pure-refactor PR with a bit-identical-loss test.

The exception: extract `CheckpointManager` now, because P0 and section 10.5 both touch it
heavily anyway (async writer, signal handling, time-based cadence, `mmap` resume), and it has a
clean boundary.

### 14.3 Test coverage

Currently 8 test files, none of which start a real process group `[measured]`. Every failure in
this document is invisible to them. Required additions:

| Test | Asserts |
|---|---|
| `test_distributed_smoke.py` | With gloo + `mp.spawn` at world 4: losses match a world-1 run at matched grad-accum; `no_sync` accumulation equals one large batch; rank-0 `last.pt` resumes at world 2 with the right `global_step` and sampler cursor; all ranks get the same batch length |
| `test_rank_rng_independence.py` | Sigma vectors differ across simulated ranks while model state dicts are identical |
| `test_grad_accum_equivalence.py` | `grad_accum=k` with micro-batch `b` equals `grad_accum=1` with batch `k*b` |
| `test_attention_masking.py` | A masked key has **zero** influence on the output (P0-3) |
| `test_absent_independence.py` | For every task, `x_t` at ABSENT positions is independent of `x0` (P0-1) |
| `test_time_conditioning_asymmetry.py` | `t_emb(a, b) != t_emb(b, a)` (P0-2) |
| `test_config_contracts.py` | Section 14.1 |
| `test_nogres_launchers.py` | Every sbatch contains `--exclusive`, contains no `--gres`, and passes `bash -n` |
| `test_struct_tokenizer_load.py` | Tokenizer loads and encodes a 64-residue chain |
| `test_mixture_realized.py` | Realized `(corpus, task)` proportions over 10k draws are within 2% of configured |

### 14.4 Dependency and repo hygiene

- `peft==0.11.1` is pinned and imported by nothing; `openfold` is required and **not declared**.
- The three environments the plan requires (train / dplm-tokenizer / eval-tools) are partially
  separated (`.venvs/dplm` exists) but `setup_evaluation.sh` has no `tools` profile.
- `evaluation/proteins/` is **entirely untracked** in git `[measured]` — ~247 KB of the
  structure-evaluation stack is not under version control. Commit it.
- `requirements.txt` was deleted; check nothing still references it.
- Do not commit: `m0_m1_work.html.orig`, `*.orig` files, `m0_v3_analsis.log` (move to
  `.trentinium/`), `__pycache__`.
- `logs/` is gitignored but Slurm needs `logs/slurm/` to exist **before** the job script runs;
  add `logs/slurm/.gitkeep` with a `.gitignore` negation, and have `submit.sh` `mkdir -p` as
  belt and braces.

### 14.5 Telemetry

`grep -n 'perf_counter|time.time()|tokens_per|throughput' trainers/trainer.py` returns
**nothing** `[measured]`. On a multi-node run there is no way to distinguish a slow interconnect
from a slow data loader from a slow checkpoint save, which makes every diagnostic in this
document unverifiable in production.

Add rank-0 scalars for ms/optimizer-step, residue-tokens/s/GPU, dataloader wait fraction,
all-reduce time and checkpoint save latency; plus a one-time startup log of the resolved
`(local_bs, world_size, grad_accum, effective_global_batch, tokens/optimizer-step, reachable
steps vs total_steps)` and a `_validate_distributed_setup()` that asserts
`world_size == nnodes * gpus_detected`, that every rank sees the same config hash, and that
`runs/<experiment>` is visible from every node (rank 0 writes a token, all ranks stat it,
barrier). Each is two lines and turns a multi-day silent loss into a ten-second startup error.

---

## 15. Workstream schedule

Five workstreams. W1 and W2 are parallel and both gate W4.

| # | Workstream | Contents | Duration | Gate |
|---|---|---|---|---|
| **W1** | Correctness | The seven P0 items, section 5 | 1.5-2 weeks | 125M smoke reproduces M0 `last.pt` under new code |
| **W2** | Data | Shard format, corpus merge, replay rebuild, AFDB fix + cluster list, RCSB, decontamination, eval sets, tokenizer load fix | 2-3 weeks | Two independent cache builds produce identical row counts and hashes; no training accession postdates the cutoff |
| **W3** | Infra | No-GRES launchers, signal handling, time-based checkpointing, async save, NCCL env, staging, telemetry, distributed tests | 1 week | 2-node preflight passes including a USR1 round-trip |
| **W4** | Architecture | Activation checkpointing, asymmetric time, length/position signal, final norm + QK-norm, head intra-patch (behind a flag), shared AdaLN, CFG dropout | 1.5 weeks | Instantiates at the declared parameter count; warm-start converter passes; 125M ablation arms launch |
| **W5** | Evaluation | Offline staging, external binaries, sharding, inpainting driver, real PDB writer, merge_shards | 2 weeks | Full matrix runs end-to-end on one 125M checkpoint on the cluster |

Then the run ladder: tokenizer audit -> S-1 preflight -> (S0 || controls || ablations) ->
S1 -> full evaluation.

Critical-path note: **W2 is the true critical path, not W1.** S1 depends on a corpus that does
not exist, whose two source scripts are both stubs, whose cluster list has no producer, and
whose tokenizer cannot currently be loaded. Budget it accordingly and start it first.

---

## 16. Risk register

Severity x likelihood. Only risks with a concrete mitigation are listed.

| # | Risk | Consequence | Mitigation |
|---|---|---|---|
| R1 | **Continuous per-bit diffusion does not concentrate on the LFQ code manifold** | Bit accuracy improves with scale while decoded TM does not — exactly the observed pattern (65.7% bits, 0.97% exact tokens, only 2.9x above independence). This would be a fundamental limit of bitstream diffusion on product-quantized latents and undercuts the structure-side claim | Run the head intra-patch + codebook-aware-auxiliary 2-arm ablation at 125M **before** the flagship (section 8.4). Highest-information experiment in the programme |
| R2 | **M1 corpus is never built** | S1 has no training data; the whole ladder stalls | Start W2 first; scope M1b to 600k chains; keep M0 as a fallback corpus that already exists and already trains |
| R3 | Promotion gates keyed on denoising loss | Already realized once: `best.pt` cost 18 AAR points; `balanced_mse` fires at 2-3% of budget | P0-5 task-metric selection; evaluate both `best.pt` and `last.pt` at every gate until they agree |
| R4 | Entropy schedule eliminates low-noise training | The model never learns the final bit decisions where the code manifold resolves — plausibly R1's proximate cause. Measured: entropy mass below sigma=0.1 is 3.7e-6 vs ~18% base | Re-run the base-fraction arms at 125M under P0 code with task-metric selection; do not launch S1 on an unvalidated schedule |
| R5 | Overfitting the scarce paired corpus | Realized 636-1,178 passes over the 21,727 PDB rows at 125M. At 393M this is worse | Cap epochs **per source**, computed from the manifest residue count as a config assertion; keep dropout 0.2 / weight decay 0.05; hold replay at 15-20% |
| R6 | Silent half-allocation from the GRES fallback | Runs complete normally at 4 of 8 GPUs; world size, effective batch and every token budget wrong by 2x | `cobit_detect_gpus_alloc` with 0 treated as fatal; assert `world_size == nnodes * gpus_detected` at startup; preflight prints the count per node |
| R7 | Pure-srun launch fans out to N*8 rank-0 processes | All write `last.pt` concurrently; 8 processes contend for GPU 0 per node; no error message and no rank check can catch it because **every process believes it is master** | Slurm env fallback in `train.py` plus a hard guard raising when `SLURM_NTASKS>1` and resolved world size is 1 |
| R8 | Validation metric collapses with world size | 61% of val rows at 1 node, 9% at 8 nodes; length-biased and non-comparable | Allocation-independent validation protocol (10.4); log retained fraction at startup |
| R9 | Preemption discards up to 30 min per link | Over ~50 chain links, hundreds of GPU-hours | `utils/slurm_signals.py` + `--signal=B:USR1@180` + time-based checkpointing + wall-clock deadline awareness |
| R10 | Rank-0 checkpoint write exceeds the 20-min PG timeout | Whole allocation aborts with an opaque NCCL error, deterministically every 2,000 steps | Background writer, barrier after save, 60-min timeout, `mmap` resume |
| R11 | NCCL binds a management NIC | All-reduce drops below 1 GB/s; the run looks model-bound, not network-bound | Pin `NCCL_SOCKET_IFNAME` in the site env; preflight prints measured algbw |
| R12 | Chain burns queue allocation on a repeated failure | 20 links reproducing one crash | `--dependency=singleton` + progress guard that cancels when `global_step` did not advance; login-node dry-run before submit |
| R13 | Specialists beat the joint model | The "one checkpoint, all tasks" framing collapses mid-programme | The five-control suite (~130 GPU-h) run **in parallel with S0**; if a gap appears, re-tune mixture and capacity before conceding; report a quantified cost of unification |
| R14 | Task-specialized fine-tuning destroys co-generation (the DPLM-2.1 folding-SFT failure) | The released "all-in" checkpoint quietly becomes a folding model | LoRA deltas on a frozen base only; a co-generation regression gate in `build_protein_table.py` |
| R15 | Evaluation becomes the wall-clock bottleneck | 5-9 days per checkpoint on one GPU, exceeding any walltime | Shard every driver; cache folds by sequence hash; cap ESM-2 pppl; frozen subsets during development, full sweep only on final checkpoints |
| R16 | Compute estimates are wrong by 40% | Budget overrun or under-request | Every throughput number here was taken on contended GPUs and is labelled; single-tenant preflight before any compute request |
| R17 | Multi-node objective differs from single-node | Per-rank loss normalizer + per-rank entropy fit + identical per-rank noise mean the flagship does not optimize what the development runs did | P0-4 plus the all-reduced loss normalizer (9.4) plus the all-reduced entropy histogram (9.7) |
| R18 | Judge-model circularity | Self-consistency overstates designability | The structure decoder, the folding model and any judge must be distinct; report continuous distributions alongside pass rates; fixed thresholds preregistered per task |
| R19 | Evaluation leakage | Inflated folding and generation metrics | Cross-source decontamination in one namespace (7.6); a private recent-PDB time split untouched until final evaluation; report both sequence and structural nearest-neighbour similarity for generated samples |

### Responsible scope

Unchanged from the design doc and restated because it gates release. This is computational
protein generation, not validated protein engineering. Before any public sample or model
release: screen generated sequences against known toxins, virulence factors and controlled
pathogen proteins; document training-data taxonomy and the limits of automated screening; make
no function or binding claim from a language-model or folding score alone; and keep any later
pathogen-, toxin-, ligand- or antibody-targeted work separate, under its own review.

---

## 17. Decisions still open

These need a human call and are not resolvable from the code.

1. **Filter or truncate long chains?** 9.79% of M0 is truncated fragments today. Recommendation:
   filter, plus random-crop augmentation. Affects corpus size at M1.
2. **Shared AdaLN?** Frees ~120M parameters at 400M but breaks warm-start key compatibility, so
   it must be decided before S0, not after. Recommendation: yes, with a converter.
3. **Absolute position and length signal.** Recommendation: add length embedding + terminus
   markers + fractional position. Checkpoint-shape change, so decide before S0.
4. **How many seeds for the flagship?** Three seeds at 393M is ~1,350 GPU-hours and does not fit.
   Recommendation: three seeds at 125M, single seed at 393M with multiple-checkpoint bootstrap
   evaluation, and disclose this.
5. **Is the matched DPLM-2 Bit A/B in scope at 2,000 GPU-hours?** It requires training a second
   engine to the same token budget. Recommendation: **out of scope for this budget**; state it
   as the next milestone and lean on the native-vs-permuted LFQ control as the within-programme
   evidence that bit structure matters.

---

## 18. Summary

The codebase is in better shape than the M0 results suggest. The joint any-to-any machinery
works, the trunk is efficient (residue-level attention, not bit-level), exact resume is real,
and roughly 250 KB of structure-evaluation code is already written.

What is missing splits cleanly in three:

**Correctness.** Seven defects, each of which independently invalidated an M0 conclusion or
would silently corrupt a multi-node run: leaked ABSENT bits, modality-symmetric time
conditioning, an inverted attention mask, identical noise on every rank, a checkpoint metric
anti-correlated with task quality, a fictitious task mixture, and an epoch budget that stops
the LR schedule at a third of its span. These are one to two weeks of engineering and they gate
everything.

**Data.** The M1 corpus does not exist, its two source scripts are stubs (AFDB v4 returns 404
everywhere; ESMAtlas needs an index nobody produces), the cluster list has no producer, the
frozen tokenizer cannot be loaded by any reusable code path, the replay corpus is 400k
N-terminal fragments of the longest 1% of proteins, and there is no decontamination pipeline,
no corpus merge step, and no evaluation set for three of the six tasks. This is the critical
path.

**Capability.** Two of the six requested tasks — seq-to-seq and 3D-to-3D — have no state grid,
no training mass and no evaluation driver. Motif scaffolding is trained on one fixed geometry.
Co-generation is sampled from a ~5% slice of its training distribution. CFG is both unreachable
and untrained.

The scientific finding that should drive priorities is that M0's conditional signal is
genuinely strong at the token level (both lifts have CIs excluding zero widely) and fails only
after decoding, with bit errors that are nearly independent. That is a head-architecture and
conditioning-plumbing problem, not a capacity problem — which is why this plan spends its
budget on correctness, data and evaluation, caps the flagship at 393M, and cuts the 650M run.
