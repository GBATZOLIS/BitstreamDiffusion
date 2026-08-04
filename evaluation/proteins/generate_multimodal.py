"""Conditional multimodal protein generation by clamped bitstream diffusion.

One 18-bit-per-residue model denoises the joint amino-acid sequence and DPLM-2
LFQ structure tokens. Every user-facing task (de novo co-generation, folding,
inverse folding, single-modality generation, motif scaffolding) is obtained from
that single model by clamping the observed bits and diffusing the rest. This
module builds the clamp specification for each task and drives the Heun sampler
with it.

Residue-patch layout, from ``data.protein_structure_codec``, is
``[seq_0..seq_4 | struct_0..struct_12]``: slots 0..4 are the five sequence bits
and slots 5..17 are the thirteen LFQ structure bits. The full diffused payload of
``L`` residues therefore has ``S = L * 18`` bits, laid out residue-major.

Task-to-clamp mapping (plan section 9):
  inverse_folding    structure observed (slots 5..17 clamped), sequence diffused;
  forward_folding     sequence observed (slots 0..4 clamped), structure diffused;
  joint               nothing clamped, both modalities diffused;
  sequence_marginal   nothing clamped, structure absent from the loss;
  structure_marginal  nothing clamped, sequence absent from the loss;
  motif               both modalities observed on the motif residues, both
                      diffused elsewhere.

Length is chosen before sampling and lives outside the diffused payload: the
model never generates or edits the residue count. The clamp API operates on a
fixed ``S = L * 18`` grid, so ``L`` must be fixed first. For conditional tasks
(folding, inverse folding, motif) the length is set by the observed structure or
sequence. For de novo tasks (joint, sequence_marginal, structure_marginal) there
is no observation to fix it, so the length is drawn from the empirical
distribution of training-set lengths before sampling; see
:func:`sample_length_from_prior`.

The module imports with only numpy and torch present. The diffusion sampler and
forward process are imported lazily inside :func:`generate`, since they are only
needed once a real model and config are supplied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from data.protein_multimodal import CANONICAL_AA
from data.protein_structure_codec import (
    DEFAULT_STRUCT_CODEC,
    PATCH_BITS_PER_RESIDUE,
    SEQ_BITS_PER_RESIDUE,
    STRUCT_BITS_PER_RESIDUE,
)
from data.protein_tasks import (
    NUM_MODALITIES,
    OBSERVED,
    SEQ,
    STRUCT,
    TASK_MODES,
    build_example_states,
    observed_clamp_mask_np,
    per_bit_sigma_map_torch,
)
from data.proteins import build_token_to_bits_table

# The 20 canonical amino acids and their frozen 5-bit codewords. The table is the
# same big-endian raw-binary code used by the training-time sequence encoder and
# by ``data.proteins.bitstreams_to_token_ids``, so decoding is its exact inverse.
SEQ_VOCAB_SIZE = len(CANONICAL_AA)  # 20
_SEQ_CODEWORDS = (
    build_token_to_bits_table(SEQ_VOCAB_SIZE, SEQ_BITS_PER_RESIDUE)
    .numpy()
    .astype(np.float32)
)  # [20, 5]
_SEQ_CODEWORDS_U8 = _SEQ_CODEWORDS.astype(np.uint8)  # [20, 5]

# Tasks recognized by this module: the whole-example modes plus per-residue motif.
KNOWN_TASKS = frozenset(set(TASK_MODES) | {"motif"})


def sample_length_from_prior(
    empirical_lengths: Sequence[int],
    num_samples: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw generation lengths from an empirical length distribution.

    De novo tasks have no observed modality to fix the residue count, so the
    length is chosen before sampling by drawing (with replacement) from the
    empirical distribution of training-set lengths. The chosen length only sets
    the diffused payload size ``L * 18``; it is not itself diffused. Returns an
    int64 array of ``num_samples`` lengths.
    """
    rng = rng if rng is not None else np.random.default_rng()
    pool = np.asarray(empirical_lengths, dtype=np.int64)
    if pool.size == 0:
        raise ValueError("empirical_lengths must be non-empty")
    return rng.choice(pool, size=int(num_samples), replace=True).astype(
        np.int64
    )


def build_task_conditioning(
    task: str,
    seq_bits: Optional[np.ndarray],
    struct_bits: Optional[np.ndarray],
    length: int,
    *,
    motif_spans: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the clamp specification for one task on ``length`` residues.

    Returns ``(cond_full, cond_mask)``, both 1-D tensors of length ``S = L * 18``
    laid out residue-major over the 18-bit patch. ``cond_mask`` is a boolean mask
    that is True exactly at the observed (clamped) bit positions; ``cond_full``
    holds the clean bit values there and zeros elsewhere. They feed directly into
    ``HeunSampler.sample`` as ``conditioning_prefix_full`` / ``cond_prefix_mask``.

    The observed modality per task follows the plan section 9 mapping. For
    inverse folding the 13 structure bits per residue (slots 5..17) are observed,
    so ``struct_bits`` of shape ``[L, 13]`` must be supplied. For forward folding
    the 5 sequence bits per residue (slots 0..4) are observed, so ``seq_bits`` of
    shape ``[L, 5]`` must be supplied. For joint and the two marginals nothing is
    clamped. For motif both modalities are observed on the motif residues (given
    by ``motif_spans`` as half-open ``[start, end)`` ranges, defaulting to a
    single centered span), so both ``seq_bits`` and ``struct_bits`` are required.
    """
    if task not in KNOWN_TASKS:
        raise KeyError(
            f"unknown task {task!r}; use one of {sorted(KNOWN_TASKS)}"
        )
    L = int(length)
    if L <= 0:
        raise ValueError("length must be a positive integer")
    S = L * PATCH_BITS_PER_RESIDUE

    # Per-residue modality states, then the observed-bit clamp mask they imply.
    states = build_example_states(task, L, motif_spans=motif_spans)  # [L, 2]
    residue_mask = np.ones((1, L), dtype=bool)
    clamp = observed_clamp_mask_np(states[None, :, :], residue_mask)[
        0
    ]  # [L*18] bool

    seq_observed = bool((states[:, SEQ] == OBSERVED).any())
    struct_observed = bool((states[:, STRUCT] == OBSERVED).any())
    if seq_observed and seq_bits is None:
        raise ValueError(
            f"task {task!r} observes the sequence modality; pass seq_bits of shape "
            f"[{L}, {SEQ_BITS_PER_RESIDUE}]"
        )
    if struct_observed and struct_bits is None:
        raise ValueError(
            f"task {task!r} observes the structure modality; pass struct_bits of shape "
            f"[{L}, {STRUCT_BITS_PER_RESIDUE}]"
        )

    patch = np.zeros((L, PATCH_BITS_PER_RESIDUE), dtype=np.float32)
    if seq_bits is not None:
        sb = np.asarray(seq_bits, dtype=np.float32)
        if sb.shape != (L, SEQ_BITS_PER_RESIDUE):
            raise ValueError(
                f"seq_bits must have shape [{L}, {SEQ_BITS_PER_RESIDUE}], got {sb.shape}"
            )
        patch[:, :SEQ_BITS_PER_RESIDUE] = sb
    if struct_bits is not None:
        zb = np.asarray(struct_bits, dtype=np.float32)
        if zb.shape != (L, STRUCT_BITS_PER_RESIDUE):
            raise ValueError(
                f"struct_bits must have shape [{L}, {STRUCT_BITS_PER_RESIDUE}], got {zb.shape}"
            )
        patch[:, SEQ_BITS_PER_RESIDUE:] = zb

    cond_full = torch.from_numpy(patch.reshape(S).copy())
    cond_mask = torch.from_numpy(np.ascontiguousarray(clamp.reshape(S)))
    return cond_full, cond_mask


def _extract_observed(
    observed: Optional[Mapping[str, object]],
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[Sequence[Tuple[int, int]]],
]:
    """Pull observed sequence/structure bits and motif spans out of ``observed``.

    Accepts the clean bits directly (``seq_bits`` / ``struct_bits``) or the more
    natural per-residue identities (``seq_ids`` canonical-20 ids, ``struct_index``
    LFQ uint16 ids), which are converted to bits with the frozen codecs. Returns
    ``(seq_bits, struct_bits, motif_spans)`` with missing entries as None.
    """
    if observed is None:
        return None, None, None

    seq_bits = observed.get("seq_bits")
    struct_bits = observed.get("struct_bits")
    motif_spans = observed.get("motif_spans")

    if seq_bits is None and observed.get("seq_ids") is not None:
        ids = np.asarray(observed["seq_ids"], dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError(
                "seq_ids must be a 1-D array of canonical-20 amino-acid ids"
            )
        if ids.size and (ids.min() < 0 or ids.max() >= SEQ_VOCAB_SIZE):
            raise ValueError(f"seq_ids out of range [0, {SEQ_VOCAB_SIZE - 1}]")
        seq_bits = _SEQ_CODEWORDS_U8[ids]  # [L, 5]

    if struct_bits is None and observed.get("struct_index") is not None:
        struct_bits = DEFAULT_STRUCT_CODEC.index_to_bits_np(
            np.asarray(observed["struct_index"])
        )  # [L, 13]

    return seq_bits, struct_bits, motif_spans


def _decode_sequences_map(probs: torch.Tensor, length: int) -> List[str]:
    """Decode sequence bits to amino-acid strings by per-residue codeword MAP.

    ``probs`` are the per-bit Bernoulli probabilities ``[B, S]`` returned by the
    sampler. For each residue the five sequence-slot probabilities are scored
    against the 20 canonical 5-bit codewords, and the maximum-likelihood codeword
    is chosen. Restricting to the 20 valid codewords avoids the 12 non-amino-acid
    5-bit patterns that independent per-bit thresholding could otherwise produce.
    Returns a list of ``B`` strings, each of length ``length``.
    """
    B = int(probs.shape[0])
    p = (
        probs.reshape(B, length, PATCH_BITS_PER_RESIDUE)[
            :, :, :SEQ_BITS_PER_RESIDUE
        ]
        .to(torch.float32)
        .cpu()
        .numpy()
    )  # [B, L, 5]
    eps = 1e-6
    p = np.clip(p, eps, 1.0 - eps)
    # log-likelihood of a codeword differs from sum_j bit_j * logit_j only by a
    # per-residue constant, so the argmax over codewords uses the logit directly.
    logit = np.log(p) - np.log1p(-p)  # [B, L, 5]
    scores = logit @ _SEQ_CODEWORDS.T  # [B, L, 20]
    ids = scores.argmax(axis=-1)  # [B, L]
    return ["".join(CANONICAL_AA[i] for i in row) for row in ids]


def _decode_struct_index(bits: torch.Tensor, length: int) -> np.ndarray:
    """Decode the 13 structure bits per residue to LFQ token ids ``[B, L]`` uint16."""
    B = int(bits.shape[0])
    z = (
        bits.reshape(B, length, PATCH_BITS_PER_RESIDUE)[
            :, :, SEQ_BITS_PER_RESIDUE:
        ]
        .to(torch.uint8)
        .cpu()
        .numpy()
    )  # [B, L, 13]
    index = DEFAULT_STRUCT_CODEC.bits_to_index_np(
        z
    )  # [B, L] int64 in [0, 8191]
    return index.astype(np.uint16)


class _MultimodalDenoiser(torch.nn.Module):
    """Adapt the 18-bit multimodal model to the sampler's scalar-sigma call.

    ``HeunSampler`` (and the shared ``_model_logits_continuous`` helper) invokes
    ``model(x_t, sigma, x0_hat)`` with a single per-example scalar ``sigma``.
    Training instead drives the model with the full multimodal contract (plan
    sections 6.1/6.2): a *per-bit* sigma map, the two modality sigmas for time
    conditioning, and the per-residue modality-state grid for the slot/state
    embeddings. If generation dropped that contract the trunk would run the
    sequence-only path it was never trained under. This wrapper rebuilds the
    exact training contract at every solver step from the fixed task state grid.

    ``states`` is the ``[L, 2]`` per-residue ``(seq, struct)`` state grid for the
    task (the same grid the collator builds during training). Observed and absent
    bits carry sigma 0 so the matched-filter input scaling treats them as clean;
    the sampler's clamp holds the observed bits at their conditioning values.
    """

    def __init__(self, model, states: np.ndarray):
        super().__init__()
        self.model = model
        st = torch.as_tensor(np.asarray(states), dtype=torch.long)
        if st.dim() != 2 or st.size(-1) != NUM_MODALITIES:
            raise ValueError(
                f"states must be [L, {NUM_MODALITIES}], got {tuple(st.shape)}"
            )
        # Non-persistent buffer so .to(device)/.eval() move it with the model.
        self.register_buffer("_states_row", st, persistent=False)

    def forward(self, x_t, sigma, x0_hat=None):
        B = x_t.size(0)
        L = self._states_row.size(0)
        states_b = (
            self._states_row.unsqueeze(0).expand(B, L, NUM_MODALITIES).to(x_t.device)
        )
        sig = sigma.reshape(-1).to(device=x_t.device, dtype=torch.float32)
        if sig.numel() != B:
            # Under CFG the sampler doubles the batch; both halves share sigma.
            sig = sig[:1].expand(B)
        # Both modalities share the step's scalar sigma; non-noisy bits get 0.
        sigma_map = per_bit_sigma_map_torch(states_b, sig, sig)  # [B, S]
        modality_sigmas = torch.stack([sig, sig], dim=-1)  # [B, 2]
        return self.model(
            x_t, sigma_map, x0_hat, slot_state=states_b, modality_sigmas=modality_sigmas
        )


def generate(
    model,
    cfg,
    task: str,
    length: int,
    num_samples: int,
    device,
    observed: Optional[Mapping[str, object]] = None,
    *,
    guidance_scale: Optional[float] = None,
    schedule: Optional[str] = None,
    entropic_blend_alpha: Optional[float] = None,
    entropy_run_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Sample ``num_samples`` proteins of ``length`` residues under ``task``.

    Drives :class:`HeunSampler` over the ``S = L * 18`` bit payload, clamping the
    observed bits for the task (reapplied after every solver step by the sampler).
    ``observed`` supplies the clean conditioning for the folding, inverse-folding,
    and motif tasks, either as ``seq_bits`` / ``struct_bits`` or as ``seq_ids`` /
    ``struct_index`` / ``motif_spans``; it is ignored for de novo tasks. The length
    is fixed before sampling and is not part of the diffused payload (see the
    module docstring).

    Sampler knobs (each falls back to the matching ``cfg.evaluation`` field when
    left ``None``, so a sweep can vary them per call without mutating ``cfg``):

    - ``guidance_scale``: classifier-free guidance weight. The observed bits are
      the conditioning, so CFG only bites on the conditional tasks (forward /
      inverse folding, motif); it is a harmless no-op for the de novo joint /
      marginal tasks (no clamp -> the sampler leaves guidance off). ``0`` disables.
    - ``schedule``: ``"entropic"`` builds the stratified inverse-CDF integration
      grid from the learned entropy CDF (``entropy_*.pt`` in the run dir) for both
      the deterministic Heun and the stochastic EDM-churn solver; ``"karras"`` is
      the rho=7 grid. If ``"entropic"`` is requested but the tables are missing
      this falls back to Karras with a warning (so eval never crashes).
    - ``entropic_blend_alpha`` in ``[0, 1]`` mixes entropic (0) with Karras (1).
    - ``entropy_run_dir``: where the ``entropy_*.pt`` tables live; defaults to the
      run dir two levels above ``cfg.evaluation.checkpoint_path``.

    EDM churn (the stochastic solver / gamma sweep) is read by the sampler from
    ``cfg.evaluation.stochastic`` and composes with any of the above.

    Returns a dict with keys:
      ``bits``          the thresholded ``[B, S]`` uint8 bit tensor;
      ``seq_strings``   list of ``B`` amino-acid strings (per-residue codeword MAP);
      ``struct_index``  ``[B, L]`` uint16 LFQ structure token ids.
    """
    L = int(length)
    B = int(num_samples)
    S = L * PATCH_BITS_PER_RESIDUE

    # Resolve sampler knobs: explicit arg overrides the cfg.evaluation default.
    ev = getattr(cfg, "evaluation", None)
    if guidance_scale is None:
        guidance_scale = float(getattr(ev, "guidance_scale", 0.0) or 0.0)
    guidance_scale = float(guidance_scale)
    if schedule is None:
        schedule = getattr(ev, "schedule", None)
    if entropic_blend_alpha is None:
        entropic_blend_alpha = float(getattr(ev, "entropic_blend_alpha", 0.0) or 0.0)
    if entropy_run_dir is None:
        ckpt = getattr(ev, "checkpoint_path", None)
        if ckpt:
            entropy_run_dir = str(Path(ckpt).parent.parent)

    # Entropic grid needs the learned tables; fall back to Karras (never crash)
    # if a run has not written them yet.
    if schedule is not None and str(schedule).lower() == "entropic":
        rd = Path(entropy_run_dir) if entropy_run_dir else None
        have_tables = rd is not None and all(
            (rd / name).exists()
            for name in ("entropy_pdf.pt", "entropy_cdf.pt", "entropy_sigmas.pt")
        )
        if not have_tables:
            print(
                "[generate_multimodal] entropic schedule requested but entropy "
                f"tables not found under {entropy_run_dir!r}; falling back to "
                "the karras grid."
            )
            schedule = "karras"

    repr_mode = str(getattr(cfg.data, "representation", "binary")).lower()
    if repr_mode != "binary":
        raise RuntimeError(
            "generate_multimodal requires cfg.data.representation == 'binary' "
            "(the 18-bit-per-residue patch model); got "
            f"{cfg.data.representation!r}"
        )

    seq_bits, struct_bits, motif_spans = _extract_observed(observed)
    cond_full, cond_mask = build_task_conditioning(
        task, seq_bits, struct_bits, L, motif_spans=motif_spans
    )

    # Per-residue modality-state grid for this task (same grid the collator builds
    # during training). It drives the slot/state embeddings and the per-bit sigma
    # map so generation denoises under the exact contract the model was trained on.
    states = build_example_states(task, L, motif_spans=motif_spans)  # [L, 2]

    # Heavy imports are deferred so the module stays importable with numpy/torch only.
    from diffusion.continuous.processes import ContinuousForwardProcess
    from diffusion.continuous.samplers import HeunSampler

    denoiser = _MultimodalDenoiser(model, states).to(device)

    # The multimodal model is trained with the loss on its RAW logits
    # (multimodal_bit_loss), so the reverse process must read them raw. The shared
    # config inherits matched_filter_residual scaling from the sequence path; if it
    # is left on, HeunSampler's logit postprocessing applies a transform the model
    # never trained against and corrupts every denoised bit. Force identity scaling
    # for the multimodal sampler and restore the caller's value afterwards.
    prev_logit_scaling = getattr(cfg.model, "continuous_logit_scaling", "none")
    cfg.model.continuous_logit_scaling = "none"

    if bool(cond_mask.any()):
        cond_kwargs = dict(
            conditioning_prefix_full=cond_full.to(device),
            cond_prefix_mask=cond_mask.to(device),
        )
    else:
        cond_kwargs = dict(
            conditioning_prefix_full=None, cond_prefix_mask=None
        )

    num_steps = int(cfg.evaluation.num_sampling_steps)

    try:
        sampler = HeunSampler(denoiser, ContinuousForwardProcess(cfg), cfg)
        _x, probs = sampler.sample(
            B,
            S,
            guidance_scale=guidance_scale,
            schedule=schedule,
            entropic_blend_alpha=entropic_blend_alpha,
            entropy_run_dir=entropy_run_dir,
            num_steps=num_steps,
            return_probs=True,
            **cond_kwargs,
        )
    finally:
        cfg.model.continuous_logit_scaling = prev_logit_scaling

    probs = probs.detach().to(torch.float32).cpu()
    bits = (probs > 0.5).to(torch.uint8)  # [B, S]

    return {
        "bits": bits,
        "seq_strings": _decode_sequences_map(probs, L),
        "struct_index": _decode_struct_index(bits, L),
    }
