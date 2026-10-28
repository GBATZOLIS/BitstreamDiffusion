"""
Generate proteins while keeping observed sequence or structure bits
fixed.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from pathlib import Path

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
    DROPPED,
    NUM_MODALITIES,
    OBSERVED,
    PARTIAL_PAIR,
    SPAN_TASKS,
    TASK_MODES,
    apply_condition_dropout,
    build_example_states,
    null_content_bit_mask_np,
    observed_clamp_mask_np,
    per_bit_sigma_map_torch,
)
from data.proteins import build_token_to_bits_table

# The 20 canonical amino acids and their frozen 5-bit codewords. The
# table is the same big-endian raw-binary code used by the training-time
# sequence encoder and by data.proteins.bitstreams_to_token_ids, so
# decoding is its exact inverse.
SEQ_VOCAB_SIZE = len(CANONICAL_AA)  # 20
_SEQ_CODEWORDS = (
    build_token_to_bits_table(SEQ_VOCAB_SIZE, SEQ_BITS_PER_RESIDUE)
    .numpy()
    .astype(np.float32)
)  # [20, 5]
_SEQ_CODEWORDS_U8 = _SEQ_CODEWORDS.astype(np.uint8)  # [20, 5]

# Tasks recognized by this module: the whole-example modes, plus the
# per-residue tasks (motif and the four span tasks). The span tasks are
# what make sequence-to-sequence (sequence_inpaint) and 3D-to-3D
# (structure_inpaint) reachable at generation time.
KNOWN_TASKS = frozenset(
    set(TASK_MODES) | {"motif", PARTIAL_PAIR} | set(SPAN_TASKS)
)


def sample_length_from_prior(
    empirical_lengths: Sequence[int],
    num_samples: int,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw generation lengths from an empirical length distribution."""
    rng = rng if rng is not None else np.random.default_rng()
    pool = np.asarray(empirical_lengths, dtype=np.int64)
    if pool.size == 0:
        raise ValueError("empirical_lengths must be non-empty")
    return rng.choice(pool, size=int(num_samples), replace=True).astype(
        np.int64
    )


def build_task_conditioning(
    task: str,
    seq_bits: np.ndarray | None,
    struct_bits: np.ndarray | None,
    length: int,
    *,
    motif_spans: Sequence[tuple[int, int]] | None = None,
    absent_fill: float = 0.5,
    seq_observed_mask: np.ndarray | None = None,
    struct_observed_mask: np.ndarray | None = None,
    bit_observed_mask: np.ndarray | None = None,
    states: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the clamp specification for one task on length residues."""
    if task not in KNOWN_TASKS:
        raise KeyError(
            f"unknown task {task!r}; use one of {sorted(KNOWN_TASKS)}"
        )
    L = int(length)
    if L <= 0:
        raise ValueError("length must be a positive integer")
    S = L * PATCH_BITS_PER_RESIDUE

    # Per-residue modality states, then the observed-bit clamp mask they
    # imply.
    #
    # Span tasks (sequence_inpaint / structure_inpaint and the redesign
    # pair) need to know WHICH residues are the given context. At
    # generation that is not drawn at random: the caller pins it,
    # because the whole point is to complete a specific partial sequence
    # or a specific partial backbone. The states escape hatch lets a
    # driver supply an arbitrary grid directly.
    if states is None:
        states = build_example_states(
            task,
            L,
            motif_spans=motif_spans,
            seq_observed_mask=seq_observed_mask,
            struct_observed_mask=struct_observed_mask,
        )  # [L, 2]
    else:
        states = np.asarray(states, dtype=np.int8)
        if states.shape != (L, 2):
            raise ValueError(
                f"states must have shape [{L}, 2], got {states.shape}"
            )
    residue_mask = np.ones((1, L), dtype=bool)
    clamp = observed_clamp_mask_np(states[None, :, :], residue_mask)[
        0
    ]  # [L*18] bool

    if bit_observed_mask is not None:
        bit_mask = np.asarray(bit_observed_mask, dtype=bool)
        if bit_mask.shape != (L, PATCH_BITS_PER_RESIDUE):
            raise ValueError(
                f"bit_observed_mask must have shape [{L}, "
                f"{PATCH_BITS_PER_RESIDUE}], got {bit_mask.shape}"
            )
        clamp = clamp.reshape(L, PATCH_BITS_PER_RESIDUE) | bit_mask
        clamp = clamp.reshape(S)

    clamp_patch = clamp.reshape(L, PATCH_BITS_PER_RESIDUE)
    seq_observed = bool(clamp_patch[:, :SEQ_BITS_PER_RESIDUE].any())
    struct_observed = bool(clamp_patch[:, SEQ_BITS_PER_RESIDUE:].any())
    if seq_observed and seq_bits is None:
        raise ValueError(
            f"task {task!r} observes the sequence modality; pass seq_bits "
            f"of shape [{L}, {SEQ_BITS_PER_RESIDUE}]"
        )
    if struct_observed and struct_bits is None:
        raise ValueError(
            f"task {task!r} observes the structure modality; pass "
            f"struct_bits of shape [{L}, {STRUCT_BITS_PER_RESIDUE}]"
        )

    patch = np.zeros((L, PATCH_BITS_PER_RESIDUE), dtype=np.float32)
    if seq_bits is not None:
        sb = np.asarray(seq_bits, dtype=np.float32)
        if sb.shape != (L, SEQ_BITS_PER_RESIDUE):
            raise ValueError(
                f"seq_bits must have shape [{L}, {SEQ_BITS_PER_RESIDUE}], "
                f"got {sb.shape}"
            )
        patch[:, :SEQ_BITS_PER_RESIDUE] = sb
    if struct_bits is not None:
        zb = np.asarray(struct_bits, dtype=np.float32)
        if zb.shape != (L, STRUCT_BITS_PER_RESIDUE):
            raise ValueError(
                f"struct_bits must have shape [{L}, "
                f"{STRUCT_BITS_PER_RESIDUE}], got {zb.shape}"
            )
        patch[:, SEQ_BITS_PER_RESIDUE:] = zb

    # P0-1, generation side. The collator overwrites the bits of an
    # ABSENT (or CFG-DROPPED) modality with absent_fill during training,
    # so the model has only ever seen that constant there. Clamping the
    # same positions to the same constant here is what makes generation
    # match training; without it the sampler would start those slots
    # from pure noise and the model would meet an input distribution it
    # never saw. The two fill values must agree, so both default to the
    # diffusion data centre (0.5).
    null_bits = null_content_bit_mask_np(states[None, :, :])[0]  # [L*18] bool
    patch_flat = patch.reshape(S)
    patch_flat[null_bits] = float(absent_fill)
    clamp_flat = clamp.reshape(S) | null_bits

    cond_full = torch.from_numpy(patch_flat.copy())
    cond_mask = torch.from_numpy(np.ascontiguousarray(clamp_flat))
    return cond_full, cond_mask


def build_clean_payload(
    seq_bits: np.ndarray | None,
    struct_bits: np.ndarray | None,
    length: int,
    *,
    fill: float = 0.5,
) -> torch.Tensor:
    """[S] clean bit payload for SDEdit-style refinement (x_init)."""
    L = int(length)
    patch = np.full((L, PATCH_BITS_PER_RESIDUE), float(fill), dtype=np.float32)
    if seq_bits is not None:
        patch[:, :SEQ_BITS_PER_RESIDUE] = np.asarray(
            seq_bits, dtype=np.float32
        )
    if struct_bits is not None:
        patch[:, SEQ_BITS_PER_RESIDUE:] = np.asarray(
            struct_bits, dtype=np.float32
        )
    return torch.from_numpy(patch.reshape(L * PATCH_BITS_PER_RESIDUE).copy())


def _extract_observed(
    observed: Mapping[str, object] | None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    Sequence[tuple[int, int]] | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Pull the observed context out of observed."""
    if observed is None:
        return None, None, None, None, None, None, None

    seq_bits = observed.get("seq_bits")
    struct_bits = observed.get("struct_bits")
    motif_spans = observed.get("motif_spans")
    seq_observed_mask = observed.get("seq_observed_mask")
    struct_observed_mask = observed.get("struct_observed_mask")
    states = observed.get("states")
    bit_observed_mask = observed.get("bit_observed_mask")
    if seq_observed_mask is not None:
        seq_observed_mask = np.asarray(seq_observed_mask, dtype=bool)
    if struct_observed_mask is not None:
        struct_observed_mask = np.asarray(struct_observed_mask, dtype=bool)
    if bit_observed_mask is not None:
        bit_observed_mask = np.asarray(bit_observed_mask, dtype=bool)

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

    return (
        seq_bits,
        struct_bits,
        motif_spans,
        seq_observed_mask,
        struct_observed_mask,
        states,
        bit_observed_mask,
    )


def _decode_sequences_map(probs: torch.Tensor, length: int) -> list[str]:
    """
    Decode sequence bits to amino-acid strings by per-residue codeword
    MAP.
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
    # log-likelihood of a codeword differs from sum_j bit_j * logit_j
    # only by a per-residue constant, so the argmax over codewords uses
    # the logit directly.
    logit = np.log(p) - np.log1p(-p)  # [B, L, 5]
    scores = logit @ _SEQ_CODEWORDS.T  # [B, L, 20]
    ids = scores.argmax(axis=-1)  # [B, L]
    return ["".join(CANONICAL_AA[i] for i in row) for row in ids]


def _decode_struct_index(bits: torch.Tensor, length: int) -> np.ndarray:
    """
    Decode the 13 structure bits per residue to LFQ token ids [B, L]
    uint16.
    """
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
    """Adapt the 18-bit multimodal model to the sampler's scalar-sigma call."""

    def __init__(
        self,
        model,
        states: np.ndarray,
        *,
        cfg_half_batch: int | None = None,
        struct_sigma_scale: float = 1.0,
        bit_observed_mask: np.ndarray | None = None,
    ):
        super().__init__()
        self.model = model
        st = torch.as_tensor(np.asarray(states), dtype=torch.long)
        if st.dim() != 2 or st.size(-1) != NUM_MODALITIES:
            raise ValueError(
                f"states must be [L, {NUM_MODALITIES}], got {tuple(st.shape)}"
            )
        # Non-persistent buffer so .to(device)/.eval() move it with the
        # model.
        self.register_buffer("_states_row", st, persistent=False)
        if bit_observed_mask is None:
            self._bit_observed_row = None
        else:
            bit_mask = torch.as_tensor(
                np.asarray(bit_observed_mask, dtype=bool).reshape(-1),
                dtype=torch.bool,
            )
            expected = int(st.size(0)) * PATCH_BITS_PER_RESIDUE
            if bit_mask.numel() != expected:
                raise ValueError(
                    f"bit_observed_mask has {bit_mask.numel()} bits, "
                    f"expected {expected}"
                )
            self.register_buffer(
                "_bit_observed_row", bit_mask, persistent=False
            )

        # Classifier-free guidance. The sampler doubles the batch into
        # [conditional | unconditional] and already writes the null
        # VALUE into the unconditional half's observed bits -- but it
        # has no way to change the STATE grid, so without this the
        # "unconditional" half was still told those residues are
        # OBSERVED and still received sigma 0 for them. The guidance
        # direction was therefore a difference between two conditional
        # predictions, which is why turning guidance up made results
        # worse rather than better.
        #
        # Here the second half gets its own grid with OBSERVED rewritten
        # to DROPPED, matching exactly what the training step showed the
        # model when it dropped a condition.
        self._cfg_half_batch = int(cfg_half_batch) if cfg_half_batch else None
        if self._cfg_half_batch:
            dropped = apply_condition_dropout(
                np.asarray(states)[None, :, :], np.array([True])
            )[0]
            self.register_buffer(
                "_states_row_uncond",
                torch.as_tensor(dropped, dtype=torch.long),
                persistent=False,
            )
        else:
            self._states_row_uncond = None

        # Staggered per-modality schedules. The sampler carries ONE
        # scalar sigma per step, so the two modalities are locked to the
        # diagonal by construction; scaling structure's sigma relative
        # to sequence's is the cheapest way to run them out of step,
        # which is a capability a bitstream model has and a
        # single-tensor model does not. 1.0 reproduces the diagonal
        # exactly.
        self.struct_sigma_scale = float(struct_sigma_scale)

    def _states_for_batch(self, batch: int, device) -> torch.Tensor:
        L = self._states_row.size(0)
        half = self._cfg_half_batch
        if half and batch == 2 * half:
            cond = self._states_row.unsqueeze(0).expand(
                half, L, NUM_MODALITIES
            )
            uncond = self._states_row_uncond.unsqueeze(0).expand(
                half, L, NUM_MODALITIES
            )
            return torch.cat([cond, uncond], dim=0).to(device)
        return (
            self._states_row.unsqueeze(0)
            .expand(batch, L, NUM_MODALITIES)
            .to(device)
        )

    def _explicit_condition_for_batch(
        self, batch: int, device
    ) -> torch.Tensor | None:
        if self._bit_observed_row is None:
            return None
        row = self._bit_observed_row.to(device)
        return row.unsqueeze(0).expand(batch, -1)

    def forward(self, x_t, sigma, x0_hat=None):
        B = x_t.size(0)
        states_b = self._states_for_batch(B, x_t.device)
        sig = sigma.reshape(-1).to(device=x_t.device, dtype=torch.float32)
        if sig.numel() != B:
            # Under CFG the sampler doubles the batch; both halves share
            # sigma.
            sig = sig[:1].expand(B)
        sig_seq = sig
        sig_struct = (
            sig
            if self.struct_sigma_scale == 1.0
            else (sig * self.struct_sigma_scale)
        )
        sigma_map = per_bit_sigma_map_torch(
            states_b, sig_seq, sig_struct
        )  # [B, S]
        explicit_condition = self._explicit_condition_for_batch(B, x_t.device)
        bit_state_ids = None
        if explicit_condition is not None:
            from models.protein_multimodal import build_state_ids_from_states

            bit_state_ids = build_state_ids_from_states(states_b)
            bit_state_ids = bit_state_ids.masked_fill(
                explicit_condition, int(OBSERVED)
            )
            half = self._cfg_half_batch
            if half and B == 2 * half:
                bit_state_ids[half:] = bit_state_ids[half:].masked_fill(
                    explicit_condition[half:], int(DROPPED)
                )
            # Both CFG halves receive clean, clamped content in these
            # slots: native values for the conditional half and the null
            # value for the unconditional half. OBSERVED and DROPPED
            # therefore both carry sigma zero, exactly as in the
            # training-time state contract.
            sigma_map = sigma_map.masked_fill(explicit_condition, 0.0)
        modality_sigmas = torch.stack([sig_seq, sig_struct], dim=-1)  # [B, 2]
        residue_mask = torch.ones(
            B, states_b.size(1), dtype=torch.bool, device=x_t.device
        )
        return self.model(
            x_t,
            sigma_map,
            x0_hat,
            slot_state=states_b,
            bit_state_ids=bit_state_ids,
            modality_sigmas=modality_sigmas,
            residue_mask=residue_mask,
        )


def generate(
    model,
    cfg,
    task: str,
    length: int,
    num_samples: int,
    device,
    observed: Mapping[str, object] | None = None,
    *,
    solver: str | None = None,
    guidance_scale: float | None = None,
    schedule: str | None = None,
    entropic_blend_alpha: float | None = None,
    entropy_run_dir: str | None = None,
    sigma_start: float | None = None,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    struct_sigma_scale: float | None = None,
    num_steps: int | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    """Sample num_samples proteins of length residues under task."""
    L = int(length)
    B = int(num_samples)
    S = L * PATCH_BITS_PER_RESIDUE

    ev = getattr(cfg, "evaluation", None)

    def _resolve_setting(name, explicit, selected):
        return selected if explicit is None else explicit

    selected_solver = str(getattr(ev, "solver", "heun") or "heun")
    solver = str(_resolve_setting("solver", solver, selected_solver)).lower()
    solver_aliases = {
        "ddim": "ddim_entropic",
        "ddim_entropic": "ddim_entropic",
        "heun": "heun",
        "heun_entropic": "heun",
    }
    if solver not in solver_aliases:
        raise ValueError(
            f"unknown continuous protein solver {solver!r}; expected "
            f"'ddim_entropic' or 'heun'"
        )
    solver = solver_aliases[solver]

    selected_guidance = float(getattr(ev, "guidance_scale", 0.0) or 0.0)
    guidance_scale = _resolve_setting(
        "guidance_scale", guidance_scale, selected_guidance
    )
    if guidance_scale is None:
        guidance_scale = selected_guidance
    guidance_scale = float(guidance_scale)
    selected_schedule = getattr(ev, "schedule", None)
    schedule = _resolve_setting("schedule", schedule, selected_schedule)
    selected_blend = float(getattr(ev, "entropic_blend_alpha", 0.0) or 0.0)
    entropic_blend_alpha = float(
        _resolve_setting(
            "entropic_blend_alpha", entropic_blend_alpha, selected_blend
        )
    )
    selected_sigma_min = getattr(ev, "sigma_min", None)
    selected_sigma_max = getattr(ev, "sigma_max", None)
    sigma_min = _resolve_setting("sigma_min", sigma_min, selected_sigma_min)
    sigma_max = _resolve_setting("sigma_max", sigma_max, selected_sigma_max)
    selected_struct_scale = float(
        getattr(ev, "struct_sigma_scale", 1.0) or 1.0
    )
    struct_sigma_scale = float(
        _resolve_setting(
            "struct_sigma_scale", struct_sigma_scale, selected_struct_scale
        )
    )
    selected_steps = int(getattr(ev, "num_sampling_steps", 400))
    num_steps = int(_resolve_setting("num_steps", num_steps, selected_steps))
    ckpt = getattr(ev, "checkpoint_path", None)
    locked_entropy_dir = (
        Path(str(ckpt)).expanduser().resolve().parent.parent if ckpt else None
    )
    if entropy_run_dir is None and locked_entropy_dir is not None:
        entropy_run_dir = str(locked_entropy_dir)
    if str(schedule).lower() == "entropic":
        directory = Path(entropy_run_dir) if entropy_run_dir else None
        names = ("entropy_pdf.pt", "entropy_cdf.pt", "entropy_sigmas.pt")
        if directory is None or not all(
            (directory / name).is_file() for name in names
        ):
            raise FileNotFoundError(
                f"Entropy tables are missing in {entropy_run_dir!r}. Run "
                f"scripts/proteins/evaluate.py entropy first."
            )

    repr_mode = str(getattr(cfg.data, "representation", "binary")).lower()
    if repr_mode != "binary":
        raise RuntimeError(
            f"generate_multimodal requires cfg.data.representation == "
            f"'binary' (the 18-bit-per-residue patch model); got "
            f"{cfg.data.representation!r}"
        )

    (
        seq_bits,
        struct_bits,
        motif_spans,
        seq_observed_mask,
        struct_observed_mask,
        supplied_states,
        bit_observed_mask,
    ) = _extract_observed(observed)

    # Per-residue modality-state grid for this task (the same grid the
    # collator builds during training). It drives the slot/state
    # embeddings and the per-bit sigma map so generation denoises under
    # the exact contract the model was trained on.
    #
    # For the span tasks the observed mask is NOT drawn at random here:
    # the caller pins which residues are the given context, because
    # completing a specific partial sequence or a specific partial
    # backbone is the whole task.
    if supplied_states is not None:
        states = np.asarray(supplied_states, dtype=np.int8)
        if states.shape != (L, NUM_MODALITIES):
            raise ValueError(
                f"observed['states'] must be [{L}, {NUM_MODALITIES}], got "
                f"{states.shape}"
            )
    else:
        states = build_example_states(
            task,
            L,
            motif_spans=motif_spans,
            seq_observed_mask=seq_observed_mask,
            struct_observed_mask=struct_observed_mask,
        )  # [L, 2]

    absent_fill = float(getattr(cfg.diffusion.continuous, "data_center", 0.5))
    cond_full, cond_mask = build_task_conditioning(
        task,
        seq_bits,
        struct_bits,
        L,
        motif_spans=motif_spans,
        absent_fill=absent_fill,
        bit_observed_mask=bit_observed_mask,
        states=states,
    )

    # Heavy imports are deferred so the module stays importable with
    # numpy/torch only.
    from diffusion.continuous.processes import ContinuousForwardProcess
    from diffusion.continuous.samplers import DDIMSampler, HeunSampler

    # Null-filled ABSENT/DROPPED slots are fixed for train/inference
    # consistency, but are not semantic conditioning.  Only OBSERVED
    # bits (including an explicit bit reveal) justify a CFG
    # conditional/unconditional pair.
    semantic_cond_mask = observed_clamp_mask_np(
        states[None, :, :], np.ones((1, L), dtype=bool)
    )[0]
    if bit_observed_mask is not None:
        semantic_cond_mask = semantic_cond_mask | bit_observed_mask.reshape(S)
    cfg_active = bool(guidance_scale > 0.0) and bool(semantic_cond_mask.any())
    effective_guidance_scale = guidance_scale if cfg_active else 0.0
    denoiser = _MultimodalDenoiser(
        model,
        states,
        cfg_half_batch=B if cfg_active else None,
        struct_sigma_scale=struct_sigma_scale,
        bit_observed_mask=bit_observed_mask,
    ).to(device)

    # The multimodal model is trained with the loss on its RAW logits
    # (multimodal_bit_loss), so the reverse process must read them raw.
    # The shared config inherits matched_filter_residual scaling from
    # the sequence path; if it is left on, the sampler's logit
    # postprocessing applies a transform the model never trained against
    # and corrupts every denoised bit. Force identity scaling for the
    # multimodal sampler and restore the caller's value afterwards.
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

    # SDEdit-style refinement: start from the supplied clean payload
    # corrupted to sigma_start rather than from pure noise. This is what
    # makes 3D -> 3D refinement a real capability of the bitstream
    # sampler; every bit of the output still comes out of the diffusion
    # trajectory, the trajectory just starts partway down it.
    x_init = None
    if sigma_start is not None:
        x_init = build_clean_payload(
            seq_bits, struct_bits, L, fill=absent_fill
        ).to(device)[None, :]

    devices = (
        [torch.device(device)] if torch.device(device).type == "cuda" else []
    )
    rng_context = (
        torch.random.fork_rng(devices=devices, enabled=True)
        if seed is not None
        else contextlib.nullcontext()
    )
    try:
        with rng_context:
            if seed is not None:
                torch.manual_seed(int(seed))
                if devices:
                    torch.cuda.manual_seed_all(int(seed))
            sampler_cls = (
                DDIMSampler if solver == "ddim_entropic" else HeunSampler
            )
            sampler = sampler_cls(denoiser, ContinuousForwardProcess(cfg), cfg)
            _x, probs = sampler.sample(
                B,
                S,
                guidance_scale=effective_guidance_scale,
                schedule=schedule,
                entropic_blend_alpha=entropic_blend_alpha,
                entropy_run_dir=entropy_run_dir,
                num_steps=num_steps,
                sigma_min_override=(
                    None if sigma_min is None else float(sigma_min)
                ),
                sigma_max_override=(
                    None if sigma_max is None else float(sigma_max)
                ),
                return_probs=True,
                x_init=x_init,
                sigma_start=sigma_start,
                **cond_kwargs,
            )
    finally:
        cfg.model.continuous_logit_scaling = prev_logit_scaling

    probs = probs.detach().to(torch.float32).cpu()
    bits = (probs > 0.5).to(torch.uint8)  # [B, S]

    stochastic_cfg = getattr(ev, "stochastic", None)
    sampling = {
        "solver": solver,
        "schedule": schedule,
        "entropic_blend_alpha": entropic_blend_alpha,
        "guidance_scale": guidance_scale,
        "effective_guidance_scale": effective_guidance_scale,
        "cfg_active": cfg_active,
        "num_sampling_steps": num_steps,
        "seed": None if seed is None else int(seed),
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "struct_sigma_scale": struct_sigma_scale,
        "stochastic": {
            "enabled": bool(getattr(stochastic_cfg, "enabled", False)),
            "s_churn": float(getattr(stochastic_cfg, "s_churn", 0.0)),
            "s_noise": float(getattr(stochastic_cfg, "s_noise", 1.0)),
            "window_mode": str(
                getattr(stochastic_cfg, "window_mode", "deterministic")
            ),
            "entropy_quantile_lo": float(
                getattr(stochastic_cfg, "entropy_quantile_lo", 0.0)
            ),
            "entropy_quantile_hi": float(
                getattr(stochastic_cfg, "entropy_quantile_hi", 1.0)
            ),
            "entropy_fallback": str(
                getattr(stochastic_cfg, "entropy_fallback", "deterministic")
            ),
        },
    }
    return {
        "bits": bits,
        "seq_strings": _decode_sequences_map(probs, L),
        "struct_index": _decode_struct_index(bits, L),
        "sampling": sampling,
    }
