"""Frozen sequence/structure metrics matching the released DiMA protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from scipy import linalg
from scipy.stats import wasserstein_distance

from data.proteins import DIMA_CANONICAL_AA


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fasta(path: Path) -> List[str]:
    sequences, pieces = [], []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if pieces:
                    sequences.append("".join(pieces))
                    pieces = []
            else:
                pieces.append(line)
    if pieces:
        sequences.append("".join(pieces))
    return sequences


def load_indexed(root: Path, split: str, limit: int | None = None) -> List[str]:
    protocol = root / "protocol"
    data = np.memmap(protocol / f"{split}.sequences.bin", dtype=np.uint8, mode="r")
    offsets = np.load(protocol / f"{split}.offsets.npy", mmap_mode="r")
    count = len(offsets) - 1 if limit is None else min(limit, len(offsets) - 1)
    return [
        bytes(data[int(offsets[i]) : int(offsets[i + 1])]).decode("ascii")
        for i in range(count)
    ]


def composition(sequences: Iterable[str]) -> np.ndarray:
    index = {aa: i for i, aa in enumerate(DIMA_CANONICAL_AA)}
    counts = np.zeros(20, dtype=np.float64)
    for sequence in sequences:
        for aa in sequence:
            counts[index[aa]] += 1
    return counts / counts.sum()


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    midpoint = 0.5 * (p + q)
    def kl(left, right):
        mask = left > 0
        return np.sum(left[mask] * np.log2(left[mask] / right[mask]))
    return float(0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint))


def basic_metrics(generated: List[str], references: List[str], train: List[str]) -> dict:
    alphabet = set(DIMA_CANONICAL_AA)
    valid = [sequence for sequence in generated if 128 <= len(sequence) <= 254 and set(sequence) <= alphabet]
    train_set = set(train)
    lengths = np.asarray([len(sequence) for sequence in valid], dtype=np.float64)
    reference_lengths = np.asarray([len(sequence) for sequence in references], dtype=np.float64)
    return {
        "num_samples": len(generated),
        "valid_fraction": len(valid) / max(1, len(generated)),
        "unique_fraction": len(set(valid)) / max(1, len(valid)),
        "exact_novelty": sum(sequence not in train_set for sequence in valid) / max(1, len(valid)),
        "length_mean": float(lengths.mean()),
        "length_median": float(np.median(lengths)),
        "reference_length_mean": float(reference_lengths.mean()),
        "length_wasserstein": float(wasserstein_distance(lengths, reference_lengths)),
        "amino_acid_composition_jsd": jsd(composition(valid), composition(references)),
    }


@torch.no_grad()
def dima_esm_pppl(
    sequences: List[str], device: torch.device, *, batch_size: int = 64, max_len: int = 254
) -> float:
    """Exact released DiMA ESM2-650M per-sequence PPPL, then arithmetic mean."""
    from transformers import EsmForMaskedLM, EsmTokenizer

    model_id = "facebook/esm2_t33_650M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_id)
    model = EsmForMaskedLM.from_pretrained(
        model_id, add_cross_attention=False, is_decoder=False
    ).to(device).eval()
    all_pppl = []
    special_ids = torch.tensor(tokenizer.all_special_ids, device=device)
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        encoded = tokenizer.batch_encode_plus(
            batch,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        nll = torch.zeros_like(input_ids, dtype=torch.float32)
        for position in range(input_ids.size(1)):
            masked = input_ids.clone()
            masked[:, position] = tokenizer.mask_token_id
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=masked, attention_mask=attention_mask).logits
            token_nll = -torch.log_softmax(logits[:, position, :].float(), dim=-1)
            nll[:, position] = token_nll.gather(1, input_ids[:, position, None]).squeeze(1)
        non_special = ~(input_ids.unsqueeze(-1) == special_ids).any(dim=-1)
        per_sequence = (nll * non_special).sum(1) / non_special.sum(1).clamp_min(1)
        all_pppl.extend(per_sequence.exp().cpu().tolist())
    return float(np.mean(all_pppl))


@torch.no_grad()
def prot_t5_embeddings(
    sequences: List[str], device: torch.device, *, max_len: int = 254, batch_size: int = 64
) -> np.ndarray:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_uniref50", legacy=True
    )
    model = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float16
    ).to(device).eval()
    outputs = []
    for start in range(0, len(sequences), batch_size):
        formatted = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequences[start:start + batch_size]]
        batch = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        outputs.append(pooled.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def frechet_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    mu_x, mu_y = x.mean(0), y.mean(0)
    cov_x, cov_y = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    covmean, _ = linalg.sqrtm(cov_x.dot(cov_y), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_x.shape[0]) * eps
        covmean = linalg.sqrtm((cov_x + offset).dot(cov_y + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    difference = mu_x - mu_y
    return float(difference.dot(difference) + np.trace(cov_x) + np.trace(cov_y) - 2 * np.trace(covmean))


def dima_mmd(x: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    left = torch.tensor(x, device=device)
    right = torch.tensor(y, device=device)
    xx, yy, xy = left @ left.T, right @ right.T, left @ right.T
    dxx = xx.diag()[:, None] + xx.diag()[None, :] - 2 * xx
    dyy = yy.diag()[:, None] + yy.diag()[None, :] - 2 * yy
    dxy = xx.diag()[:, None] + yy.diag()[None, :] - 2 * xy
    value = torch.zeros((), device=device)
    for bandwidth in (10, 15, 20, 50):
        value += (
            torch.exp(-0.5 * dxx / bandwidth)
            + torch.exp(-0.5 * dyy / bandwidth)
            - 2 * torch.exp(-0.5 * dxy / bandwidth)
        ).mean()
    return float(value.item())


@torch.no_grad()
def esmfold_plddt(sequences: List[str], device: torch.device) -> float:
    from transformers import AutoTokenizer, EsmForProteinFolding

    model_id = "facebook/esmfold_v1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = EsmForProteinFolding.from_pretrained(
        model_id, low_cpu_mem_usage=True
    ).to(device).eval()
    values = []
    for sequence in sequences:
        inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False).to(device)
        output = model(**inputs)
        values.append(float(output.plddt[0, : len(sequence)].float().mean().item()))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-fasta", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/swissprot_dima_bf4b2f13"))
    parser.add_argument("--metrics", nargs="+", choices=["basic", "esm_pppl", "prot_t5", "plddt"], default=["basic"])
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--esm-samples", type=int, default=512)
    parser.add_argument("--plddt-samples", type=int, default=512)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    generated = read_fasta(args.generated_fasta)[: args.num_samples]
    if len(generated) != args.num_samples:
        raise ValueError(f"Expected {args.num_samples} generations, found {len(generated)}")
    references = load_indexed(args.dataset_root, "test", args.num_samples)
    train = load_indexed(args.dataset_root, "train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {"protocol": "DiMA release commit 18f4a2e, HF revision bf4b2f13"}

    if "basic" in args.metrics:
        results.update(basic_metrics(generated, references, train))
    if "esm_pppl" in args.metrics:
        results["esm2_650m_pppl"] = dima_esm_pppl(generated[: args.esm_samples], device)
        results["esm2_650m_pppl_reference"] = dima_esm_pppl(references[: args.esm_samples], device)
    if "prot_t5" in args.metrics:
        gen_embeddings = prot_t5_embeddings(generated, device)
        ref_embeddings = prot_t5_embeddings(references, device)
        results["prot_t5_frechet_distance"] = frechet_distance(gen_embeddings, ref_embeddings)
        results["prot_t5_mmd_rbf"] = dima_mmd(gen_embeddings, ref_embeddings, device)
    if "plddt" in args.metrics:
        results["esmfold_plddt"] = esmfold_plddt(generated[: args.plddt_samples], device)

    out = args.out or args.generated_fasta.parent / "frozen_metrics.json"
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_fasta": {"path": str(args.generated_fasta), "sha256": sha256(args.generated_fasta)},
        "dataset_manifest_sha256": sha256(args.dataset_root / "frozen_manifest.json"),
        "metrics_requested": args.metrics,
        "results": results,
    }
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
