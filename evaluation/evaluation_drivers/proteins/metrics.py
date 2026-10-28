from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

from data.uniref50 import EVODIFF_UNIREF50_ALPHABET

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"


def _composition(sequences: Sequence[str], alphabet: str) -> np.ndarray:
    index = {symbol: i for i, symbol in enumerate(alphabet)}
    counts = np.zeros(len(alphabet), dtype=np.float64)
    for sequence in sequences:
        for symbol in sequence:
            if symbol in index:
                counts[index[symbol]] += 1
    return counts / counts.sum() if counts.sum() else counts


def _jsd(left: np.ndarray, right: np.ndarray) -> float:
    middle = 0.5 * (left + right)
    value = 0.0
    for distribution in (left, right):
        mask = distribution > 0
        value += 0.5 * float(
            np.sum(
                distribution[mask] * np.log2(distribution[mask] / middle[mask])
            )
        )
    return value


def basic_metrics(generated: Sequence[str], references: Sequence[str]) -> dict:
    allowed, canonical = set(EVODIFF_UNIREF50_ALPHABET), set(CANONICAL_AA)
    valid = [
        sequence
        for sequence in generated
        if sequence and set(sequence) <= allowed
    ]
    total_residues = sum(map(len, generated))
    canonical_residues = sum(
        sum(symbol in canonical for symbol in seq) for seq in generated
    )
    lengths = np.asarray(
        [len(sequence) for sequence in generated], dtype=np.int64
    )
    reference_lengths = np.asarray(
        [len(sequence) for sequence in references], dtype=np.int64
    )
    return {
        "num_samples": len(generated),
        "valid_fraction": len(valid) / max(1, len(generated)),
        "canonical_sequence_fraction": sum(
            set(seq) <= canonical for seq in valid
        )
        / max(1, len(valid)),
        "canonical_residue_fraction": canonical_residues
        / max(1, total_residues),
        "unique_fraction": len(set(valid)) / max(1, len(valid)),
        "length_mean": float(lengths.mean()),
        "reference_length_mean": float(reference_lengths.mean()),
        "lengths_exactly_matched": bool(
            np.array_equal(lengths, reference_lengths)
        ),
        "amino_acid_composition_jsd": _jsd(
            _composition(valid, EVODIFF_UNIREF50_ALPHABET),
            _composition(references, EVODIFF_UNIREF50_ALPHABET),
        ),
    }


def frechet_distance(
    left: np.ndarray, right: np.ndarray, eps: float = 1e-6
) -> float:
    mu_left, mu_right = left.mean(0), right.mean(0)
    cov_left, cov_right = (
        np.cov(left, rowvar=False),
        np.cov(right, rowvar=False),
    )
    covmean, _ = linalg.sqrtm(cov_left.dot(cov_right), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_left.shape[0]) * eps
        covmean = linalg.sqrtm((cov_left + offset).dot(cov_right + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    difference = mu_left - mu_right
    return float(
        difference.dot(difference)
        + np.trace(cov_left)
        + np.trace(cov_right)
        - 2 * np.trace(covmean)
    )


def mmd_rbf(left: np.ndarray, right: np.ndarray) -> float:
    x, y = torch.from_numpy(left), torch.from_numpy(right)
    xx, yy, xy = x @ x.T, y @ y.T, x @ y.T
    dxx = xx.diag()[:, None] + xx.diag()[None, :] - 2 * xx
    dyy = yy.diag()[:, None] + yy.diag()[None, :] - 2 * yy
    dxy = xx.diag()[:, None] + yy.diag()[None, :] - 2 * xy
    value = torch.zeros((), dtype=x.dtype)
    for bandwidth in (10, 15, 20, 50):
        value += (
            torch.exp(-0.5 * dxx / bandwidth)
            + torch.exp(-0.5 * dyy / bandwidth)
            - 2 * torch.exp(-0.5 * dxy / bandwidth)
        ).mean()
    return float(value.item())


# ESM-2 650M is the size at which the pseudo-perplexity of generated
# protein sequences is conventionally reported, so it is the size that
# makes our number comparable to the published ones. EvoDiff scores with
# ESM2_650M in analysis/sequence_perp.py, and DPLM's NetConfig defaults
# to esm2_t33_650M_UR50D (src/byprot/models/utils.py).
ESM2_PPPL_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
ESM2_PPPL_MODEL_REVISION = "08e4846e537177426273712802403f7ba8261b6c"

# The plan's S0 sequence-pretrain gate: unconditional 200-mers must
# reach a pseudo-perplexity no worse than this multiple of natural
# sequences'.
S0_PPPL_RATIO_THRESHOLD = 1.3


@torch.no_grad()
def esm2_pppl_scores(
    sequences: Sequence[str], device: torch.device, *, batch_size: int = 16
) -> np.ndarray:
    """Masked-marginal pseudo-perplexity of each sequence under ESM-2 650M."""
    from transformers import EsmForMaskedLM, EsmTokenizer

    model_id = ESM2_PPPL_MODEL_ID
    tokenizer = EsmTokenizer.from_pretrained(
        model_id,
        revision=ESM2_PPPL_MODEL_REVISION,
        local_files_only=True,
    )
    model = (
        EsmForMaskedLM.from_pretrained(
            model_id,
            revision=ESM2_PPPL_MODEL_REVISION,
            local_files_only=True,
            use_safetensors=True,
        )
        .to(device)
        .eval()
    )
    scores = []
    for sequence in sequences:
        encoded = tokenizer(
            sequence, return_tensors="pt", add_special_tokens=True
        ).to(device)
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        losses = []
        positions = range(1, ids.size(1) - 1)
        for start in range(0, ids.size(1) - 2, batch_size):
            selected = list(positions)[start : start + batch_size]
            masked = ids.repeat(len(selected), 1)
            for row, position in enumerate(selected):
                masked[row, position] = tokenizer.mask_token_id
            logits = model(
                masked, attention_mask=mask.repeat(len(selected), 1)
            ).logits
            row = torch.arange(len(selected), device=device)
            pos = torch.tensor(selected, device=device)
            target = ids[0, pos]
            losses.append(
                -torch.log_softmax(logits[row, pos].float(), -1)[row, target]
            )
        scores.append(float(torch.cat(losses).mean().exp().cpu()))
    del model
    return np.asarray(scores, dtype=np.float64)


def _subsample(sequences: Sequence[str], limit: int, seed: int) -> list:
    """
    Take at most limit sequences, chosen reproducibly without
    replacement.
    """
    pool = list(sequences)
    if limit <= 0 or len(pool) <= limit:
        return pool
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(pool), size=limit, replace=False)
    return [pool[int(i)] for i in sorted(picks)]


def esm2_pppl_gate(
    generated: Sequence[str],
    natural: Sequence[str],
    device: torch.device,
    *,
    max_sequences: int = 64,
    ratio_threshold: float = S0_PPPL_RATIO_THRESHOLD,
    batch_size: int = 16,
    length_tolerance: float = 0.1,
    seed: int = 0,
) -> dict:
    """Score the S0 sequence-pretrain gate: generated pppl <= 1.3x natural."""
    generated_sample = _subsample(generated, max_sequences, seed)
    natural_sample = _subsample(natural, max_sequences, seed)
    if not generated_sample:
        raise ValueError(
            "esm2_pppl_gate needs at least one generated sequence."
        )
    if not natural_sample:
        raise ValueError("esm2_pppl_gate needs at least one natural sequence.")

    generated_scores = esm2_pppl_scores(
        generated_sample, device, batch_size=batch_size
    )
    natural_scores = esm2_pppl_scores(
        natural_sample, device, batch_size=batch_size
    )

    generated_mean = float(np.mean(generated_scores))
    natural_mean = float(np.mean(natural_scores))
    ratio = generated_mean / natural_mean if natural_mean > 0 else float("inf")

    generated_length = float(np.mean([len(s) for s in generated_sample]))
    natural_length = float(np.mean([len(s) for s in natural_sample]))
    length_matched = bool(
        natural_length > 0
        and abs(generated_length - natural_length) / natural_length
        <= length_tolerance
    )

    return {
        "model_id": ESM2_PPPL_MODEL_ID,
        "generated_pppl_mean": generated_mean,
        "generated_pppl_median": float(np.median(generated_scores)),
        "natural_pppl_mean": natural_mean,
        "natural_pppl_median": float(np.median(natural_scores)),
        "ratio": ratio,
        "ratio_threshold": float(ratio_threshold),
        "passed": bool(ratio <= float(ratio_threshold)),
        "num_generated_scored": len(generated_sample),
        "num_natural_scored": len(natural_sample),
        "num_generated_available": len(generated),
        "num_natural_available": len(natural),
        "generated_length_mean": generated_length,
        "natural_length_mean": natural_length,
        "length_matched": length_matched,
        "length_tolerance": float(length_tolerance),
        "generated_pppl": generated_scores.tolist(),
        "natural_pppl": natural_scores.tolist(),
    }


@torch.no_grad()
def prot_t5_embeddings(
    sequences: Sequence[str], device: torch.device, *, batch_size: int = 8
) -> np.ndarray:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_uniref50", legacy=True
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = (
        T5EncoderModel.from_pretrained(
            "Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=dtype
        )
        .to(device)
        .eval()
    )
    outputs = []
    for start in range(0, len(sequences), batch_size):
        formatted = [
            " ".join(list(re.sub(r"[UZOBJ]", "X", sequence)))
            for sequence in sequences[start : start + batch_size]
        ]
        batch = tokenizer(formatted, return_tensors="pt", padding=True).to(
            device
        )
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        outputs.append(
            ((hidden * mask).sum(1) / mask.sum(1)).float().cpu().numpy()
        )
    del model
    return np.concatenate(outputs)


@torch.no_grad()
def esmfold_plddt_scores(
    sequences: Sequence[str], device: torch.device
) -> np.ndarray:
    from transformers import AutoTokenizer, EsmForProteinFolding

    model_id = "facebook/esmfold_v1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = (
        EsmForProteinFolding.from_pretrained(model_id, low_cpu_mem_usage=True)
        .to(device)
        .eval()
    )
    if device.type == "cuda":
        model.esm = model.esm.half()
    values = []
    for sequence in sequences:
        inputs = tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        ).to(device)
        output = model(**inputs)
        values.append(
            float(output.plddt[0, : len(sequence)].float().mean().cpu())
        )
    del model
    return np.asarray(values, dtype=np.float64)


def cached_array(path: Path, compute) -> np.ndarray:
    if path.exists():
        return np.load(path, allow_pickle=False)
    value = np.asarray(compute())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)
    return value
