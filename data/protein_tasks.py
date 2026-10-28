"""Build observed, predicted, and missing masks for protein tasks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from utils.regimes import ConditioningState, apply_condition_dropout_states

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

from .protein_structure_codec import (
    PATCH_BITS_PER_RESIDUE,
    SEQ_BITS_PER_RESIDUE,
    STRUCT_BITS_PER_RESIDUE,
)

# Modality-state codes.
ABSENT = int(ConditioningState.ABSENT)
OBSERVED = int(ConditioningState.OBSERVED)
NOISY = int(ConditioningState.NOISY)
# DROPPED is the classifier-free-guidance null state: the modality *is*
# present at this residue and would have been OBSERVED, but the
# condition was dropped, so the model is shown the null value instead.
#
# It must be a fourth code rather than a reuse of ABSENT. Two reasons.
# First, without a distinct state the "unconditional" half of a CFG
# batch is still told those residues are OBSERVED and still gets sigma 0
# for them, so the guidance direction is not a
# conditional-minus-unconditional difference at all. Second, ABSENT
# already means something different and load-bearing -- "this row has no
# structure" -- and collapsing the two would teach the model that a
# dropped condition and a sequence-only corpus row are the same thing.
#
# For noise, loss and clamping, DROPPED behaves exactly like ABSENT:
# sigma 0, never a loss target, content pinned to the null constant.
# Only the embedding differs.
DROPPED = int(ConditioningState.DROPPED)
NUM_STATES = 4

# Modality indices.
SEQ = 0
STRUCT = 1
NUM_MODALITIES = 2

# Slot -> modality index for the 18-bit patch.
SLOT_MODALITY = np.array(
    [SEQ] * SEQ_BITS_PER_RESIDUE + [STRUCT] * STRUCT_BITS_PER_RESIDUE,
    dtype=np.int64,
)
assert SLOT_MODALITY.shape[0] == PATCH_BITS_PER_RESIDUE


# Whole-example task modes: (seq_state, struct_state). Motif is
# per-residue and is built separately by build_motif_states.
TASK_MODES: dict[str, tuple[int, int]] = {
    "joint": (NOISY, NOISY),  # p(s, z) co-generation
    "inverse_folding": (NOISY, OBSERVED),  # p(s | z)
    "forward_folding": (OBSERVED, NOISY),  # p(z | s)
    "sequence_marginal": (NOISY, ABSENT),  # p(s)
    "structure_marginal": (ABSENT, NOISY),  # p(z)
}


# -----------------------------------------------------------------------------
# Per-residue span tasks: sequence-to-sequence and 3D-to-3D
# -----------------------------------------------------------------------------
#
# These CANNOT live in TASK_MODES, which maps a task to one scalar state
# pair for the whole example. A span task observes part of a chain and
# generates the rest, so its state varies per residue -- the same shape
# as motif, which is why they are registered separately and dispatched
# the same way.
#
# The four combinations are all the ways to condition one modality on a
# partial observation of itself:
#
#   sequence_inpaint    seq observed on spans, NOISY elsewhere, struct ABSENT
#                       -> sequence-to-sequence. The 3D modality is genuinely
# missing, so the model must complete the sequence from
#                          sequence context alone.
#   structure_inpaint   struct observed on spans, NOISY elsewhere, seq ABSENT
#                       -> 3D-to-3D. Backbone completion from backbone context.
#   sequence_redesign   seq observed on spans, struct fully OBSERVED
#                       -> targeted redesign at a fixed backbone.
#   structure_redesign  struct observed on spans, seq fully OBSERVED
#                       -> constrained backbone edit at a fixed sequence.
#
# The inpaint pair is what makes seq-to-seq and 3D-to-3D real
# capabilities; the redesign pair is the conditional version of each and
# costs nothing extra.
SPAN_TASKS: dict[str, tuple[int, int]] = {
    # task -> (modality observed on spans, state of the OTHER modality)
    "sequence_inpaint": (SEQ, ABSENT),
    "structure_inpaint": (STRUCT, ABSENT),
    "sequence_redesign": (SEQ, OBSERVED),
    "structure_redesign": (STRUCT, OBSERVED),
}

# partial_pair draws the two observed masks INDEPENDENTLY, which is the
# realistic deployment case the four tasks above cannot express: "I have
# the sequence everywhere and the backbone for 30% of it". It cannot be
# a member of SPAN_TASKS because that maps a task to one observed
# modality plus a scalar state for the other.
PARTIAL_PAIR = "partial_pair"

# structure_denoise is deliberately NOT a training task.
#
# SDEdit-style 3D->3D refinement -- start from a corrupted native
# structure at an intermediate noise level and denoise the rest of the
# way -- requires no special training: ordinary diffusion training
# already supervises every sigma, and the only thing missing was the
# ability to *start* the sampler from a partially noised state instead
# of pure noise. That is a sampler capability
# (HeunSampler.sample(x_init=..., sigma_start=...)), not a state grid.
#
# Registering it as a training task would have put mass on a state grid
# identical to structure_marginal and taught the model nothing new,
# while making the realized mixture harder to read. It is listed here so
# a reader looking for the sixth task of plan section 6.2 finds the
# reason rather than an omission.
INFERENCE_ONLY_TASKS: tuple[str, ...] = ("structure_denoise",)


def all_task_names() -> tuple[str, ...]:
    """Every task the collator can draw, in a stable order."""
    return tuple(TASK_MODES) + ("motif",) + tuple(SPAN_TASKS) + (PARTIAL_PAIR,)


# Number of observed segments, as the empirical distribution of the
# RFdiffusion / FrameFlow motif-scaffolding benchmark. Training span
# geometry is drawn from the SAME distribution the benchmark uses,
# because the previous training default was a single fixed centred span
# covering the middle third of every chain -- so every motif example in
# every run had identical geometry while the benchmark is multi-segment
# and variable-position.
#
# These numbers are MEASURED from the built benchmark artefact rather
# than transcribed, because a transcribed count silently drifts from the
# corpus it is supposed to describe. The previous hardcoded value was
# (16, 5, 2,
# 1)/24, while
# datasets/motif_benchmark/motif_problems.json as built holds 25
# problems with segment counts (17, 5, 2, 1) -- so the distribution
# driving training geometry did not match the benchmark actually on
# disk. load_benchmark_segment_probs re-derives it from the artefact;
# this constant is only the fallback for a tree where the benchmark has
# not been built yet.
BENCHMARK_SEGMENT_COUNT_PROBS: tuple[float, ...] = (
    17 / 25.0,
    5 / 25.0,
    2 / 25.0,
    1 / 25.0,
)

DEFAULT_MOTIF_BENCHMARK_PATH = "datasets/motif_benchmark/motif_problems.json"

_SEGMENT_PROBS_CACHE: dict[str, tuple[float, ...]] = {}


def load_benchmark_segment_probs(
    path: str | None = None,
) -> tuple[float, ...]:
    """Segment-count distribution measured from the built motif benchmark."""
    key = str(path or DEFAULT_MOTIF_BENCHMARK_PATH)
    if key in _SEGMENT_PROBS_CACHE:
        return _SEGMENT_PROBS_CACHE[key]
    probs = BENCHMARK_SEGMENT_COUNT_PROBS
    try:
        import json

        with open(key, encoding="utf-8") as handle:
            payload = json.load(handle)
        entries = payload["problems"] if isinstance(payload, dict) else payload
        counts: dict[int, int] = {}
        for entry in entries:
            n = len(entry["segments"])
            counts[n] = counts.get(n, 0) + 1
        if counts:
            top = max(counts)
            total = float(sum(counts.values()))
            probs = tuple(counts.get(n, 0) / total for n in range(1, top + 1))
    except (OSError, KeyError, ValueError, TypeError):
        pass
    _SEGMENT_PROBS_CACHE[key] = probs
    return probs


def split_into_gaps(
    total: int, num_gaps: int, rng: np.random.Generator
) -> list[int]:
    """Split total residues into num_gaps non-negative bins, UNIFORMLY."""
    if num_gaps <= 0:
        return []
    total = int(total)
    if total <= 0:
        return [0] * num_gaps
    if num_gaps == 1:
        return [total]
    cuts = np.sort(
        rng.choice(total + num_gaps - 1, size=num_gaps - 1, replace=False)
    )
    bounds = np.concatenate([[-1], cuts, [total + num_gaps - 1]])
    return [
        int(b - a - 1) for a, b in zip(bounds[:-1], bounds[1:], strict=True)
    ]


def sample_span_mask(
    rng: np.random.Generator,
    length: int,
    *,
    coverage_range: tuple[float, float] = (0.10, 0.90),
    segment_count_probs: Sequence[float] | None = None,
) -> np.ndarray:
    """[length] bool, True where the modality is OBSERVED (the context)."""
    length = int(length)
    if length <= 0:
        return np.zeros(0, dtype=bool)

    if segment_count_probs is None:
        segment_count_probs = load_benchmark_segment_probs()

    lo, hi = coverage_range
    coverage = float(rng.uniform(lo, hi))
    observed_total = int(round(coverage * length))
    # Keep at least one observed and one generated residue, or the task
    # degenerates into an unconditional or a fully-observed example.
    observed_total = max(1, min(length - 1, observed_total))

    max_segments = min(len(segment_count_probs), observed_total)
    probs = np.asarray(segment_count_probs[:max_segments], dtype=np.float64)
    probs = probs / probs.sum()
    num_segments = int(rng.choice(np.arange(1, max_segments + 1), p=probs))

    # Segment sizes: split the observed budget, then guarantee each is
    # >= 1.
    sizes = split_into_gaps(observed_total - num_segments, num_segments, rng)
    sizes = [s + 1 for s in sizes]

    # Gaps before, between and after the segments (same split as the
    # benchmark).
    gaps = split_into_gaps(length - sum(sizes), num_segments + 1, rng)

    mask = np.zeros(length, dtype=bool)
    cursor = 0
    for i, size in enumerate(sizes):
        cursor += gaps[i]
        mask[cursor : cursor + size] = True
        cursor += size
    return mask


def build_span_states(
    length: int,
    *,
    seq_observed: np.ndarray | None = None,
    struct_observed: np.ndarray | None = None,
    seq_absent: bool = False,
    struct_absent: bool = False,
) -> np.ndarray:
    """[length, 2] int8 state grid for a per-residue span task."""
    states = np.full((int(length), NUM_MODALITIES), NOISY, dtype=np.int8)
    if seq_observed is not None:
        states[np.asarray(seq_observed, dtype=bool), SEQ] = OBSERVED
    if struct_observed is not None:
        states[np.asarray(struct_observed, dtype=bool), STRUCT] = OBSERVED
    if seq_absent:
        states[:, SEQ] = ABSENT
    if struct_absent:
        states[:, STRUCT] = ABSENT
    return states


# A reasonable default paired-training mix (plan phase 4: start roughly
# balanced across the three core joint/folding/inverse tasks, small
# marginal/inpainting mass). Weights are tuned on validation, not fixed
# blindly.
#
# The span tasks carry real mass here. They previously carried none: the
# state grids for sequence_inpaint and structure_inpaint existed but no
# weight dict named them, so seq-to-seq and 3D-to-3D were reachable at
# generation time and completely untrained -- the worst of both, because
# the drivers would run and produce numbers.
DEFAULT_TASK_WEIGHTS: dict[str, float] = {
    "joint": 0.22,
    "forward_folding": 0.15,
    "inverse_folding": 0.15,
    "structure_inpaint": 0.07,
    "sequence_inpaint": 0.05,
    "structure_redesign": 0.03,
    "sequence_redesign": 0.03,
    "partial_pair": 0.04,
    "motif": 0.10,
    "structure_marginal": 0.08,
    "sequence_marginal": 0.08,
}

# The replay corpus is sequence-only, so exactly one task is meaningful
# on it.
REPLAY_CORPUS = "replay"
PAIRED_CORPUS = "paired"
REPLAY_TASK = "sequence_marginal"


# -----------------------------------------------------------------------------
# The (corpus, task) mixture as ONE distribution
# -----------------------------------------------------------------------------
#
# P0-6. Replay used to be drawn independently of the task mix and then
# forced sequence_marginal = 1.0, so at sequence_replay_fraction=0.40
# the realized per-example proportions were seq_marg 0.46 / joint 0.18 /
# fwd 0.12 / inv 0.12 / struct_marg 0.06 / motif 0.06 -- and nobody
# reading the config could see it. Counted in residues it was worse:
# replay rows average 510 residues against 239 for paired rows, so
# replay was 58.7% of all residue-tokens and only 27.5% of
# residue-tokens carried structure supervision.
#
# The fix is to make the configured numbers BE the realized step
# proportions, by drawing (corpus, task) from one normalized
# distribution.
#
# The phase schedule follows plan section 9.3. Conditional-first,
# because conditional targets are lower-entropy and the M0 evidence
# shows the conditional signal is exactly the part that works;
# joint-and-motif-heavy at the end, because the terminal task
# distribution is what the sampler inherits, which is the direct
# countermeasure to the DPLM-2.1 folding-SFT failure.
DEFAULT_MIXTURE_PHASES: list[tuple[float, dict[tuple[str, str], float]]] = [
    (
        0.00,  # A: conditional warm-in
        {
            (REPLAY_CORPUS, "sequence_marginal"): 0.20,
            (PAIRED_CORPUS, "forward_folding"): 0.22,
            (PAIRED_CORPUS, "inverse_folding"): 0.22,
            (PAIRED_CORPUS, "joint"): 0.10,
            (PAIRED_CORPUS, "structure_inpaint"): 0.05,
            (PAIRED_CORPUS, "sequence_inpaint"): 0.03,
            (PAIRED_CORPUS, "sequence_redesign"): 0.02,
            (PAIRED_CORPUS, "structure_redesign"): 0.02,
            (PAIRED_CORPUS, "partial_pair"): 0.02,
            (PAIRED_CORPUS, "motif"): 0.05,
            (PAIRED_CORPUS, "structure_marginal"): 0.05,
            (PAIRED_CORPUS, "sequence_marginal"): 0.02,
        },
    ),
    (
        0.15,  # B: steady
        {
            (REPLAY_CORPUS, "sequence_marginal"): 0.17,
            (PAIRED_CORPUS, "joint"): 0.20,
            (PAIRED_CORPUS, "forward_folding"): 0.15,
            (PAIRED_CORPUS, "inverse_folding"): 0.15,
            (PAIRED_CORPUS, "structure_inpaint"): 0.07,
            (PAIRED_CORPUS, "sequence_inpaint"): 0.05,
            (PAIRED_CORPUS, "sequence_redesign"): 0.03,
            (PAIRED_CORPUS, "structure_redesign"): 0.03,
            (PAIRED_CORPUS, "partial_pair"): 0.04,
            (PAIRED_CORPUS, "motif"): 0.08,
            (PAIRED_CORPUS, "structure_marginal"): 0.03,
        },
    ),
    (
        0.70,  # C: deployment-matched
        {
            (REPLAY_CORPUS, "sequence_marginal"): 0.15,
            (PAIRED_CORPUS, "joint"): 0.25,
            (PAIRED_CORPUS, "forward_folding"): 0.10,
            (PAIRED_CORPUS, "inverse_folding"): 0.10,
            (PAIRED_CORPUS, "structure_inpaint"): 0.08,
            (PAIRED_CORPUS, "sequence_inpaint"): 0.06,
            (PAIRED_CORPUS, "sequence_redesign"): 0.03,
            (PAIRED_CORPUS, "structure_redesign"): 0.03,
            (PAIRED_CORPUS, "partial_pair"): 0.04,
            (PAIRED_CORPUS, "motif"): 0.13,
            (PAIRED_CORPUS, "structure_marginal"): 0.03,
        },
    ),
]


class TaskMixtureSchedule:
    """Piecewise-constant schedule over (corpus, task) draw probabilities."""

    def __init__(
        self,
        phases: Sequence[tuple[float, dict[tuple[str, str], float]]]
        | None = None,
    ):
        raw = list(phases if phases is not None else DEFAULT_MIXTURE_PHASES)
        if not raw:
            raise ValueError("TaskMixtureSchedule needs at least one phase")
        raw.sort(key=lambda item: float(item[0]))
        self.phases: list[tuple[float, dict[tuple[str, str], float]]] = []
        for start, weights in raw:
            total = float(sum(max(0.0, float(v)) for v in weights.values()))
            if total <= 0:
                raise ValueError(
                    f"mixture phase at progress {start} has zero total weight"
                )
            self.phases.append(
                (
                    float(start),
                    {
                        k: max(0.0, float(v)) / total
                        for k, v in weights.items()
                    },
                )
            )

    def weights_at(self, progress: float) -> dict[tuple[str, str], float]:
        p = min(1.0, max(0.0, float(progress)))
        active = self.phases[0][1]
        for start, weights in self.phases:
            if p >= start:
                active = weights
            else:
                break
        return active

    def corpus_fraction(self, progress: float, corpus: str) -> float:
        return float(
            sum(
                w
                for (c, _t), w in self.weights_at(progress).items()
                if c == corpus
            )
        )

    def task_weights_for_corpus(
        self, progress: float, corpus: str
    ) -> dict[str, float]:
        """Renormalized task distribution *within* one corpus."""
        active = {
            t: w
            for (c, t), w in self.weights_at(progress).items()
            if c == corpus and w > 0
        }
        total = float(sum(active.values()))
        if total <= 0:
            return {}
        return {t: w / total for t, w in active.items()}

    def phase_index(self, progress: float) -> int:
        p = min(1.0, max(0.0, float(progress)))
        idx = 0
        for i, (start, _w) in enumerate(self.phases):
            if p >= start:
                idx = i
        return idx


def task_modality_states(task: str) -> tuple[int, int]:
    """Return (seq_state, struct_state) for a whole-example task mode."""
    if task not in TASK_MODES:
        raise KeyError(
            f"unknown whole-example task {task!r}; use one of "
            f"{sorted(TASK_MODES)}"
        )
    return TASK_MODES[task]


def build_motif_states(
    length: int,
    motif_spans: Sequence[tuple[int, int]],
) -> np.ndarray:
    """
    Per-residue states for motif scaffolding: motif residues observed,
    rest noisy.
    """
    states = np.full((length, NUM_MODALITIES), NOISY, dtype=np.int8)
    for start, end in motif_spans:
        s = max(0, int(start))
        e = min(int(length), int(end))
        if e > s:
            states[s:e, SEQ] = OBSERVED
            states[s:e, STRUCT] = OBSERVED
    return states


def build_example_states(
    task: str,
    length: int,
    *,
    seq_available: bool = True,
    struct_available: bool = True,
    motif_spans: Sequence[tuple[int, int]] | None = None,
    rng: np.random.Generator | None = None,
    seq_observed_mask: np.ndarray | None = None,
    struct_observed_mask: np.ndarray | None = None,
    coverage_range: tuple[float, float] = (0.10, 0.90),
) -> np.ndarray:
    """Build a [length, 2] state grid for one example under a task."""
    if task == "motif":
        spans = (
            motif_spans
            if motif_spans is not None
            else _default_motif_span(length)
        )
        states = build_motif_states(length, spans).astype(np.int8)
    elif task == PARTIAL_PAIR:
        # Two INDEPENDENT draws. Using one mask for both modalities
        # would make this a re-spelling of motif; the point is the
        # asymmetric case where the two modalities are known over
        # different, unaligned parts of the chain.
        masks = []
        for supplied in (seq_observed_mask, struct_observed_mask):
            if supplied is not None:
                mask = np.asarray(supplied, dtype=bool)
                if mask.shape[0] != length:
                    raise ValueError(
                        f"observed mask has length {mask.shape[0]}, "
                        f"expected {length}"
                    )
            else:
                if rng is None:
                    raise ValueError(
                        f"task {task!r} needs explicit observed masks or "
                        f"an rng"
                    )
                mask = sample_span_mask(
                    rng, length, coverage_range=coverage_range
                )
            masks.append(mask)
        states = build_span_states(
            length, seq_observed=masks[0], struct_observed=masks[1]
        )
    elif task in SPAN_TASKS:
        observed_modality, other_state = SPAN_TASKS[task]
        supplied = (
            seq_observed_mask
            if observed_modality == SEQ
            else struct_observed_mask
        )
        if supplied is not None:
            mask = np.asarray(supplied, dtype=bool)
            if mask.shape[0] != length:
                raise ValueError(
                    f"observed mask has length {mask.shape[0]}, expected "
                    f"{length}"
                )
        else:
            if rng is None:
                raise ValueError(
                    f"task {task!r} needs either an explicit observed mask"
                    f" or an rng to draw one"
                )
            mask = sample_span_mask(rng, length, coverage_range=coverage_range)
        kwargs = {
            "seq_observed": mask if observed_modality == SEQ else None,
            "struct_observed": mask if observed_modality == STRUCT else None,
        }
        # The other modality is either fully ABSENT (the inpaint pair,
        # which is what makes this genuinely single-modality completion)
        # or fully OBSERVED (the redesign pair).
        if other_state == ABSENT:
            kwargs["seq_absent"] = observed_modality != SEQ
            kwargs["struct_absent"] = observed_modality != STRUCT
        states = build_span_states(length, **kwargs)
        if other_state == OBSERVED:
            other = STRUCT if observed_modality == SEQ else SEQ
            states[:, other] = OBSERVED
    else:
        seq_state, struct_state = task_modality_states(task)
        states = np.empty((length, NUM_MODALITIES), dtype=np.int8)
        states[:, SEQ] = seq_state
        states[:, STRUCT] = struct_state

    if not seq_available:
        states[:, SEQ] = ABSENT
    if not struct_available:
        states[:, STRUCT] = ABSENT
    return states


def _default_motif_span(length: int) -> list[tuple[int, int]]:
    """
    A single centered motif covering the middle third, as a training
    default.
    """
    if length < 3:
        return [(0, length)]
    third = max(1, length // 3)
    start = (length - third) // 2
    return [(start, start + third)]


def sample_task(rng: np.random.Generator, weights: dict[str, float]) -> str:
    """Draw a task name from a weight dict (weights need not be normalized)."""
    names = list(weights.keys())
    w = np.asarray(
        [max(0.0, float(weights[n])) for n in names], dtype=np.float64
    )
    total = w.sum()
    if total <= 0:
        raise ValueError("task weights sum to zero")
    return str(rng.choice(names, p=w / total))


# -----------------------------------------------------------------------------
# Derived maps (numpy). Torch equivalents below share the same
# semantics.
# -----------------------------------------------------------------------------


def per_bit_sigma_map_np(
    states: np.ndarray,  # [B, L, 2] int
    sigma_seq: np.ndarray,  # [B]
    sigma_struct: np.ndarray,  # [B]
) -> np.ndarray:
    """Per-bit sigma [B, L*18]: modality sigma where NOISY, else 0."""
    states = np.asarray(states)
    b, l, m = states.shape
    if m != NUM_MODALITIES:
        raise ValueError("states last axis must be 2 (seq, struct)")
    sigma_mod = np.stack(
        [
            np.broadcast_to(np.asarray(sigma_seq)[:, None], (b, l)),
            np.broadcast_to(np.asarray(sigma_struct)[:, None], (b, l)),
        ],
        axis=-1,
    ).astype(np.float32)  # [B, L, 2]
    noisy = states == NOISY
    sigma_res = np.where(noisy, sigma_mod, 0.0)  # [B, L, 2]
    # Expand modality sigma across that modality's bit slots.
    per_slot = sigma_res[:, :, SLOT_MODALITY]  # [B, L, 18]
    return per_slot.reshape(b, l * PATCH_BITS_PER_RESIDUE)


def target_loss_masks_np(
    states: np.ndarray,  # [B, L, 2]
    residue_mask: np.ndarray,  # [B, L] bool, real residues
) -> tuple[np.ndarray, np.ndarray]:
    """Return (seq_target, struct_target) bit masks [B, L*5], [B, L*13]."""
    states = np.asarray(states)
    residue_mask = np.asarray(residue_mask).astype(bool)
    b, l, _ = states.shape
    seq_res = (states[:, :, SEQ] == NOISY) & residue_mask  # [B, L]
    struct_res = (states[:, :, STRUCT] == NOISY) & residue_mask
    seq_bits = np.repeat(seq_res, SEQ_BITS_PER_RESIDUE, axis=1)  # [B, L*5]
    struct_bits = np.repeat(
        struct_res, STRUCT_BITS_PER_RESIDUE, axis=1
    )  # [B, L*13]
    return seq_bits, struct_bits


def full_modality_target_masks_np(
    states: np.ndarray,  # [B, L, 2]
    residue_mask: np.ndarray,  # [B, L] bool
) -> tuple[np.ndarray, np.ndarray]:
    """Full-patch target masks [B, L*18] for each modality separately."""
    states = np.asarray(states)
    residue_mask = np.asarray(residue_mask).astype(bool)
    b, l, _ = states.shape
    seq_res = (states[:, :, SEQ] == NOISY) & residue_mask  # [B, L]
    struct_res = (states[:, :, STRUCT] == NOISY) & residue_mask
    seq_full = np.zeros((b, l, PATCH_BITS_PER_RESIDUE), dtype=bool)
    struct_full = np.zeros((b, l, PATCH_BITS_PER_RESIDUE), dtype=bool)
    seq_full[:, :, :SEQ_BITS_PER_RESIDUE] = seq_res[:, :, None]
    struct_full[:, :, SEQ_BITS_PER_RESIDUE:] = struct_res[:, :, None]
    return (
        seq_full.reshape(b, l * PATCH_BITS_PER_RESIDUE),
        struct_full.reshape(b, l * PATCH_BITS_PER_RESIDUE),
    )


def per_bit_state_ids_np(states: np.ndarray) -> np.ndarray:
    """
    Per-bit modality-state id [B, L*18] (each bit inherits its modality
    state).
    """
    states = np.asarray(states)
    b, l, _ = states.shape
    per_slot = states[:, :, SLOT_MODALITY]  # [B, L, 18]
    return per_slot.reshape(b, l * PATCH_BITS_PER_RESIDUE)


def observed_clamp_mask_np(
    states: np.ndarray,  # [B, L, 2]
    residue_mask: np.ndarray,  # [B, L] bool
) -> np.ndarray:
    """
    Per-bit clamp mask [B, L*18]: True where a real residue's modality
    is OBSERVED.
    """
    states = np.asarray(states)
    residue_mask = np.asarray(residue_mask).astype(bool)
    b, l, _ = states.shape
    obs = (states == OBSERVED) & residue_mask[:, :, None]  # [B, L, 2]
    per_slot = obs[:, :, SLOT_MODALITY]  # [B, L, 18]
    return per_slot.reshape(b, l * PATCH_BITS_PER_RESIDUE)


def null_content_bit_mask_np(states: np.ndarray) -> np.ndarray:
    """
    Per-bit mask [B, L*18]: True where the content must be the null
    value.
    """
    states = np.asarray(states)
    b, l, _ = states.shape
    null = (states == ABSENT) | (states == DROPPED)  # [B, L, 2]
    per_slot = null[:, :, SLOT_MODALITY]  # [B, L, 18]
    return per_slot.reshape(b, l * PATCH_BITS_PER_RESIDUE)


def apply_condition_dropout(
    states: np.ndarray,
    drop: np.ndarray,
) -> np.ndarray:
    """Rewrite OBSERVED -> DROPPED for the examples flagged in drop [B]."""
    states = np.array(states, dtype=np.int8, copy=True)
    drop = np.asarray(drop, dtype=bool)
    if drop.shape[0] != states.shape[0]:
        raise ValueError(
            f"drop mask has length {drop.shape[0]}, expected {states.shape[0]}"
        )
    if not drop.any():
        return states
    rows = np.nonzero(drop)[0]
    block = states[rows]
    block[block == OBSERVED] = DROPPED
    states[rows] = block
    return states


def absent_bit_mask_np(states: np.ndarray) -> np.ndarray:
    """Per-bit mask [B, L*18]: True where the modality state is ABSENT."""
    states = np.asarray(states)
    b, l, _ = states.shape
    absent = states == ABSENT  # [B, L, 2]
    per_slot = absent[:, :, SLOT_MODALITY]  # [B, L, 18]
    return per_slot.reshape(b, l * PATCH_BITS_PER_RESIDUE)


# -----------------------------------------------------------------------------
# Torch equivalents (data-loading / GPU path)
# -----------------------------------------------------------------------------


def per_bit_sigma_map_torch(states, sigma_seq, sigma_struct):
    if not _HAS_TORCH:
        raise RuntimeError("torch is not available")
    b, l, m = states.shape
    slot_mod = torch.as_tensor(SLOT_MODALITY, device=states.device)
    sigma_mod = torch.stack(
        [
            sigma_seq.view(b, 1).expand(b, l),
            sigma_struct.view(b, 1).expand(b, l),
        ],
        dim=-1,
    ).to(torch.float32)  # [B, L, 2]
    sigma_res = torch.where(
        states == NOISY, sigma_mod, torch.zeros_like(sigma_mod)
    )
    per_slot = sigma_res[:, :, slot_mod]  # [B, L, 18]
    return per_slot.reshape(b, l * PATCH_BITS_PER_RESIDUE)


def target_loss_masks_torch(states, residue_mask):
    if not _HAS_TORCH:
        raise RuntimeError("torch is not available")
    residue_mask = residue_mask.bool()
    seq_res = (states[:, :, SEQ] == NOISY) & residue_mask
    struct_res = (states[:, :, STRUCT] == NOISY) & residue_mask
    seq_bits = seq_res.repeat_interleave(SEQ_BITS_PER_RESIDUE, dim=1)
    struct_bits = struct_res.repeat_interleave(STRUCT_BITS_PER_RESIDUE, dim=1)
    return seq_bits, struct_bits


def observed_clamp_mask_torch(states, residue_mask):
    if not _HAS_TORCH:
        raise RuntimeError("torch is not available")
    slot_mod = torch.as_tensor(SLOT_MODALITY, device=states.device)
    obs = (states == OBSERVED) & residue_mask.bool().unsqueeze(-1)
    per_slot = obs[:, :, slot_mod]
    return per_slot.reshape(
        states.shape[0], states.shape[1] * PATCH_BITS_PER_RESIDUE
    )


def null_content_bit_mask_torch(states):
    """Torch twin of null_content_bit_mask_np."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is not available")
    slot_mod = torch.as_tensor(SLOT_MODALITY, device=states.device)
    null = (states == ABSENT) | (states == DROPPED)
    per_slot = null[:, :, slot_mod]
    return per_slot.reshape(
        states.shape[0], states.shape[1] * PATCH_BITS_PER_RESIDUE
    )


def apply_condition_dropout_torch(states, drop):
    """Torch twin of apply_condition_dropout; drop is [B] bool."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is not available")
    return apply_condition_dropout_states(states, drop)
