"""Sampler-based task score for checkpoint selection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from data.protein_structure_codec import (
    PATCH_BITS_PER_RESIDUE,
    SEQ_BITS_PER_RESIDUE,
)

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class TaskScoreConfig:
    """Knobs for the selection-gate evaluation."""

    # Cost model: inverse and forward each make one sampler call; a
    # joint sample makes three (co-generation plus the two conditional
    # cycle checks). DDIM is one model forward per step; Heun is
    # approximately two.
    #
    #   DDIM forwards = num_steps * (2*num_proteins + 3*num_joint)
    #
    # At the defaults that is 2*32*(256+256+64) = 36,864 forwards,
    # divided by world size because the evaluator shards. Anything that
    # pushes a single call past a few minutes should reduce
    # num_proteins, not num_steps: below ~24 steps the sampler stops
    # resembling the one evaluation uses, which would reintroduce the
    # very gap P0-5 exists to close.
    num_proteins: int = 256
    num_joint: int = 64
    num_steps: int = 32
    max_length: int = 1024
    length_bin_edges: Sequence[int] = field(
        default_factory=lambda: (128, 256, 512, 768, 1024)
    )
    seed: int = 12345
    solver: str = "ddim_entropic"
    schedule: str = "karras"
    entropic_blend_alpha: float = 1.0
    guidance_scale: float = 0.0
    tasks: Sequence[str] = field(
        default_factory=lambda: ("inverse_folding", "forward_folding", "joint")
    )

    @classmethod
    def from_cfg(cls, cfg) -> TaskScoreConfig:
        node = getattr(getattr(cfg, "train", None), "task_eval", None)
        if node is None:
            return cls()
        return cls(
            num_proteins=int(getattr(node, "num_proteins", 256)),
            num_joint=int(getattr(node, "num_joint", 64)),
            num_steps=int(getattr(node, "num_steps", 32)),
            max_length=int(getattr(node, "max_length", 1024)),
            length_bin_edges=tuple(
                int(x)
                for x in getattr(
                    node, "length_bin_edges", (128, 256, 512, 768, 1024)
                )
            ),
            seed=int(getattr(node, "seed", 12345)),
            solver=str(getattr(node, "solver", "ddim_entropic")),
            schedule=str(getattr(node, "schedule", "karras")),
            entropic_blend_alpha=float(
                getattr(node, "entropic_blend_alpha", 1.0)
            ),
            guidance_scale=float(getattr(node, "guidance_scale", 0.0)),
            tasks=tuple(
                getattr(
                    node,
                    "tasks",
                    ("inverse_folding", "forward_folding", "joint"),
                )
            ),
        )


def _seq_ids_to_string(ids: np.ndarray) -> str:
    return "".join(CANONICAL_AA[int(i)] for i in ids)


class TaskScoreEvaluator:
    """Score a checkpoint on the tasks, by sampling, on a frozen subset."""

    def __init__(
        self,
        cfg,
        dataset,
        device,
        *,
        score_cfg: TaskScoreConfig | None = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.cfg = cfg
        self.device = device
        self.score_cfg = score_cfg or TaskScoreConfig.from_cfg(cfg)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        # The FULL frozen subset is identical on every rank (same seed,
        # same dataset), and each rank then evaluates a strided slice of
        # it.
        #
        # Sharding rather than running this on rank 0 matters for a
        # reason that is not about speed. A master-only evaluation
        # leaves every other rank blocked in a collective while rank 0
        # runs tens of thousands of forwards, and this gate fires at an
        # epoch boundary immediately before a dist.barrier() -- the
        # exact shape that trips the process-group timeout and kills the
        # whole allocation. With the work spread, no rank is idle and
        # the wall time drops by world_size.
        self.all_rows = self._freeze_subset(dataset)
        self.rows = self.all_rows[self.rank :: self.world_size]

    # -- subset ---------------------------------------------------------
    def _freeze_subset(self, dataset) -> list[dict[str, object]]:
        rng = np.random.default_rng(self.score_cfg.seed)
        order = rng.permutation(len(dataset))
        edges = tuple(self.score_cfg.length_bin_edges)
        quota = max(1, self.score_cfg.num_proteins // len(edges))
        buckets: list[list[dict[str, object]]] = [[] for _ in edges]
        overflow: list[dict[str, object]] = []
        for idx in order:
            row = dataset[int(idx)]
            length = int(row["length"])
            if length > self.score_cfg.max_length or length < 16:
                continue
            if not bool(np.asarray(row["seq_mask"]).all()):
                continue
            if not bool(np.asarray(row["struct_mask"]).all()):
                continue
            bin_id = next(
                (i for i, edge in enumerate(edges) if length <= edge), None
            )
            if bin_id is None:
                continue
            candidate = {
                "seq_ids": np.asarray(row["seq_ids"], dtype=np.int64),
                "struct_index": np.asarray(
                    row["struct_index"], dtype=np.int64
                ),
                "length": length,
                "subset_index": -1,
            }
            if len(buckets[bin_id]) < quota:
                buckets[bin_id].append(candidate)
            elif len(overflow) < self.score_cfg.num_proteins:
                overflow.append(candidate)
            if all(len(bucket) >= quota for bucket in buckets):
                break
        if not any(buckets):
            raise RuntimeError(
                "TaskScoreEvaluator found no fully-paired rows in the given "
                "dataset; the selection gate cannot be built"
            )
        # Freeze a length-stratified panel rather than letting the
        # abundant short chains crowd every >512 row out of checkpoint
        # selection.
        rows = [row for bucket in buckets for row in bucket[:quota]]
        if len(rows) < self.score_cfg.num_proteins:
            rows.extend(overflow)
            rows = rows[: self.score_cfg.num_proteins]
        for i, row in enumerate(rows):
            row["subset_index"] = i
        return rows

    # -- generation -----------------------------------------------------
    def _generate(self, model, task: str, row: dict[str, object], observed):
        from evaluation.evaluation_drivers.proteins import generate_multimodal

        generate = generate_multimodal.generate

        material = (
            f"{self.score_cfg.seed}|{int(row['subset_index'])}|{task}"
        ).encode()
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (
            2**31
        )

        stochastic = getattr(
            getattr(self.cfg, "evaluation", None), "stochastic", None
        )
        previous = None
        if stochastic is not None:
            previous = (
                bool(getattr(stochastic, "enabled", False)),
                float(getattr(stochastic, "s_churn", 0.0)),
            )
            stochastic.enabled = False
            stochastic.s_churn = 0.0
        try:
            return generate(
                model,
                self.cfg,
                task,
                int(row["length"]),
                1,
                self.device,
                observed=observed,
                solver=self.score_cfg.solver,
                schedule=self.score_cfg.schedule,
                entropic_blend_alpha=self.score_cfg.entropic_blend_alpha,
                guidance_scale=self.score_cfg.guidance_scale,
                num_steps=self.score_cfg.num_steps,
                seed=seed,
            )
        finally:
            if stochastic is not None and previous is not None:
                stochastic.enabled, stochastic.s_churn = previous

    # -- metrics --------------------------------------------------------
    #
    # Each of these returns SUMS and COUNTS rather than means, so the
    # trainer can all-reduce across ranks and divide once. Reducing
    # means would weight a rank holding 3 proteins the same as a rank
    # holding 33.
    def _control_partner(self, row: dict[str, object]) -> dict[str, object]:
        """The mismatched row this row's prediction is also scored against."""
        n = len(self.all_rows)
        return self.all_rows[(int(row["subset_index"]) + n // 2) % n]

    def _inverse_folding(self, model) -> dict[str, float]:
        """AAR from the native backbone, plus the shuffled-target control."""
        aar_sum = 0.0
        shuf_sum = 0.0
        count = 0.0
        for row in self.rows:
            out = self._generate(
                model,
                "inverse_folding",
                row,
                {"struct_index": row["struct_index"]},
            )
            pred = out["seq_strings"][0]
            native = _seq_ids_to_string(row["seq_ids"])
            aar_sum += float(
                np.mean([a == b for a, b in zip(pred, native, strict=True)])
            )
            # Shuffled control: score the SAME prediction against a
            # DIFFERENT protein. This is the floor a model reaches by
            # emitting the marginal amino-acid distribution -- around 8%
            # here -- and subtracting it is what makes the gate measure
            # conditioning rather than memorisation of that marginal.
            other_seq = _seq_ids_to_string(
                self._control_partner(row)["seq_ids"]
            )
            k = min(len(pred), len(other_seq))
            if k:
                shuf_sum += float(
                    np.mean(
                        [
                            a == b
                            for a, b in zip(
                                pred[:k], other_seq[:k], strict=True
                            )
                        ]
                    )
                )
            count += 1.0
        return {
            "if_aar_sum": aar_sum,
            "if_shuf_sum": shuf_sum,
            "if_count": count,
        }

    def _forward_folding(self, model) -> dict[str, float]:
        """
        Structure-bit accuracy from the native sequence, plus its
        control.
        """
        from data.protein_structure_codec import DEFAULT_STRUCT_CODEC

        acc_sum = 0.0
        shuf_sum = 0.0
        count = 0.0
        for row in self.rows:
            out = self._generate(
                model,
                "forward_folding",
                row,
                {"seq_ids": row["seq_ids"]},
            )
            L = int(row["length"])
            bits = (
                out["bits"]
                .reshape(1, L, PATCH_BITS_PER_RESIDUE)[
                    0, :, SEQ_BITS_PER_RESIDUE:
                ]
                .numpy()
                .astype(np.uint8)
            )
            native_bits = DEFAULT_STRUCT_CODEC.index_to_bits_np(
                row["struct_index"]
            ).astype(np.uint8)
            acc_sum += float((bits == native_bits).mean())
            other_bits = DEFAULT_STRUCT_CODEC.index_to_bits_np(
                self._control_partner(row)["struct_index"]
            ).astype(np.uint8)
            k = min(bits.shape[0], other_bits.shape[0])
            if k:
                shuf_sum += float((bits[:k] == other_bits[:k]).mean())
            count += 1.0
        return {
            "ff_acc_sum": acc_sum,
            "ff_shuf_sum": shuf_sum,
            "ff_count": count,
        }

    def _joint(self, model) -> dict[str, float]:
        """Native-free joint cycle consistency plus collapse diagnostics."""
        from data.protein_structure_codec import DEFAULT_STRUCT_CODEC

        # Only the globally-first num_joint rows, so the co-generation
        # sample is the same set at every world size.
        wanted = {
            r["subset_index"]
            for r in self.all_rows[: self.score_cfg.num_joint]
        }
        seqs: list[str] = []
        seq_cycle_sum = 0.0
        struct_cycle_sum = 0.0
        struct_balance_sum = 0.0
        for row in self.rows:
            if row["subset_index"] not in wanted:
                continue
            out = self._generate(model, "joint", row, None)
            sequence = out["seq_strings"][0]
            seqs.append(sequence)
            length = int(row["length"])
            bits = (
                out["bits"]
                .reshape(1, length, PATCH_BITS_PER_RESIDUE)[
                    0, :, SEQ_BITS_PER_RESIDUE:
                ]
                .numpy()
                .astype(np.uint8)
            )
            struct_index = DEFAULT_STRUCT_CODEC.bits_to_index_np(bits)
            seq_ids = np.fromiter(
                (CANONICAL_AA.index(aa) for aa in sequence),
                dtype=np.int64,
                count=len(sequence),
            )
            inv = self._generate(
                model, "inverse_folding", row, {"struct_index": struct_index}
            )["seq_strings"][0]
            fwd = self._generate(
                model, "forward_folding", row, {"seq_ids": seq_ids}
            )
            fwd_bits = (
                fwd["bits"]
                .reshape(1, length, PATCH_BITS_PER_RESIDUE)[
                    0, :, SEQ_BITS_PER_RESIDUE:
                ]
                .numpy()
                .astype(np.uint8)
            )
            seq_cycle_sum += float(
                np.mean([a == b for a, b in zip(inv, sequence, strict=True)])
            )
            struct_cycle_sum += float((fwd_bits == bits).mean())
            # One is balanced across each LFQ dimension; zero means
            # every bit dimension is constant within the chain.
            struct_balance_sum += float(
                np.clip(
                    1.0 - 2.0 * np.abs(bits.mean(axis=0) - 0.5).mean(), 0, 1
                )
            )
        self.last_joint_sequences = list(seqs)
        if not seqs:
            return {
                "joint_count": 0.0,
                "joint_maxfrac_sum": 0.0,
                "joint_seq_cycle_sum": 0.0,
                "joint_struct_cycle_sum": 0.0,
                "joint_struct_balance_sum": 0.0,
            }
        # The codeword MAP decoder cannot emit a non-amino-acid, so
        # "validity" here is about degeneracy: a chain that is one
        # repeated residue is the collapse mode this is watching for.
        # Summed, so it reduces across ranks like everything else;
        # uniqueness is counted on rank-local hashes and is therefore a
        # lower bound at world size > 1, which is stated in the metric
        # name rather than hidden.
        maxfrac_sum = sum(
            max(s.count(c) / max(1, len(s)) for c in set(s)) for s in seqs
        )
        return {
            "joint_count": float(len(seqs)),
            "joint_unique_local": float(len(set(seqs))),
            "joint_maxfrac_sum": float(maxfrac_sum),
            "joint_seq_cycle_sum": float(seq_cycle_sum),
            "joint_struct_cycle_sum": float(struct_cycle_sum),
            "joint_struct_balance_sum": float(struct_balance_sum),
        }

    # -- entry point ----------------------------------------------------
    # The accumulator keys, in a fixed order, so the trainer can pack
    # them into one tensor for a single all-reduce.
    ACCUMULATOR_KEYS = (
        "if_aar_sum",
        "if_shuf_sum",
        "if_count",
        "ff_acc_sum",
        "ff_shuf_sum",
        "ff_count",
        "joint_count",
        "joint_unique_local",
        "joint_maxfrac_sum",
        "joint_seq_cycle_sum",
        "joint_struct_cycle_sum",
        "joint_struct_balance_sum",
    )

    @torch.no_grad()
    def evaluate(self, model) -> dict[str, float]:
        """Return this rank's SUMS and COUNTS, not means."""
        was_training = model.training
        model.eval()
        try:
            metrics: dict[str, float] = {k: 0.0 for k in self.ACCUMULATOR_KEYS}
            if "inverse_folding" in self.score_cfg.tasks:
                metrics.update(self._inverse_folding(model))
            if "forward_folding" in self.score_cfg.tasks:
                metrics.update(self._forward_folding(model))
            if "joint" in self.score_cfg.tasks:
                metrics.update(self._joint(model))
        finally:
            if was_training:
                model.train()
        return metrics


def finalize(accum: dict[str, float], num_steps: int = 0) -> dict[str, float]:
    """Turn all-reduced accumulators into the reported metrics."""

    def _mean(num: str, den: str, default: float = 0.0) -> float:
        count = float(accum.get(den, 0.0))
        return float(accum.get(num, 0.0)) / count if count > 0 else default

    aar = _mean("if_aar_sum", "if_count")
    aar_shuf = _mean("if_shuf_sum", "if_count")
    bit = _mean("ff_acc_sum", "ff_count", 0.5)
    bit_shuf = _mean("ff_shuf_sum", "ff_count", 0.5)
    joint_n = float(accum.get("joint_count", 0.0))
    joint_seq = _mean("joint_seq_cycle_sum", "joint_count")
    joint_struct = _mean("joint_struct_cycle_sum", "joint_count", 0.5)
    joint_seq_lift = float(np.clip((joint_seq - 0.05) / 0.95, 0.0, 1.0))
    joint_struct_lift = float(np.clip((joint_struct - 0.5) / 0.5, 0.0, 1.0))
    joint_consistency = 0.5 * (joint_seq_lift + joint_struct_lift)
    joint_maxfrac = _mean("joint_maxfrac_sum", "joint_count", 1.0)
    joint_struct_balance = _mean(
        "joint_struct_balance_sum", "joint_count", 0.0
    )
    # Repetition at or below 35% receives no penalty; a single-residue
    # chain scores zero. Exact global uniqueness is supplied by the
    # trainer after an all-gather, so this diagnostic is invariant to
    # 16-vs-32-way sharding.
    joint_seq_noncollapse = float(
        np.clip((1.0 - joint_maxfrac) / (1.0 - 0.35), 0.0, 1.0)
    )
    joint_unique = (
        float(accum.get("joint_unique_global", 0.0)) / joint_n
        if joint_n > 0
        else 0.0
    )
    joint_health = min(
        joint_seq_noncollapse, joint_struct_balance, joint_unique
    )
    joint_component = joint_consistency * joint_health

    out = {
        "inverse_folding_aar": aar,
        "inverse_folding_aar_shuffled": aar_shuf,
        "inverse_folding_aar_lift": aar - aar_shuf,
        "forward_folding_bit_acc": bit,
        "forward_folding_bit_acc_shuffled": bit_shuf,
        "forward_folding_bit_acc_lift": bit - bit_shuf,
        "joint_sequence_cycle_aar": joint_seq,
        "joint_structure_cycle_bit_acc": joint_struct,
        "joint_consistency_chance_adjusted": joint_consistency,
        "joint_health": joint_health,
        "joint_score_component": joint_component,
        "task_score": ((aar - aar_shuf) + (bit - bit_shuf) + joint_component)
        / 3.0,
        "task_score_num_proteins": float(accum.get("if_count", 0.0)),
        "task_score_num_steps": float(num_steps),
    }
    if joint_n > 0:
        out["joint_unique_fraction"] = joint_unique
        out["joint_max_residue_fraction"] = joint_maxfrac
        out["joint_sequence_noncollapse"] = joint_seq_noncollapse
        out["joint_struct_bit_balance"] = joint_struct_balance
    return out


def append_selection_log(
    log_dir,
    payload: dict[str, object],
) -> Path:
    """Append one JSON line recording a selection decision."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "checkpoint_selection.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path
