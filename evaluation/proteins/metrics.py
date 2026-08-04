from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

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


@torch.no_grad()
def esm2_pppl_scores(
    sequences: Sequence[str], device: torch.device, *, batch_size: int = 16
) -> np.ndarray:
    from transformers import EsmForMaskedLM, EsmTokenizer

    model_id = "facebook/esm2_t33_650M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_id)
    model = EsmForMaskedLM.from_pretrained(model_id).to(device).eval()
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
