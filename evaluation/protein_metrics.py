from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn.functional as F
from ml_collections import config_dict

from data.proteins import (
    DEFAULT_ESM2_MODEL,
    ProteinTokenizer,
    bitstreams_to_token_ids,
    load_token_id_cache,
    make_protein_tokenizer,
    tokenizer_tag_from_config,
)


# -----------------------------------------------------------------------------
# Per-row decoding (tokenizer-aware)
# -----------------------------------------------------------------------------

def _decode_row(
    row: Sequence[int], tokenizer: ProteinTokenizer, *, min_len: int = 1
) -> Dict:
    """Decode without repairing malformed output and record strict failures."""
    row = [int(token) for token in row]
    has_bos = bool(row) and row[0] == tokenizer.bos_id
    eos_positions = [i for i, token in enumerate(row) if token == tokenizer.eos_id]
    has_eos = bool(eos_positions)
    eos_index = eos_positions[0] if has_eos else len(row)
    body_start = 1 if has_bos else 0
    body = row[body_start:eos_index]
    tail = row[eos_index + 1 :] if has_eos else []

    invalid_body = [
        token for token in body if token < 0 or token >= tokenizer.vocab_size
    ]
    special_body = [token for token in body if token in tokenizer.special_ids]
    nonresidue_body = [token for token in body if token not in tokenizer.residue_ids]
    tail_violations = [token for token in tail if token != tokenizer.pad_id]

    failures: List[str] = []
    if not has_bos:
        failures.append("missing_bos")
    if not has_eos:
        failures.append("missing_eos")
    if len(eos_positions) != 1:
        failures.append("eos_count_not_one")
    if invalid_body:
        failures.append("invalid_code_before_eos")
    if special_body:
        failures.append("special_token_before_eos")
    if nonresidue_body:
        failures.append("nonresidue_before_eos")
    if tail_violations:
        failures.append("nonpad_after_eos")
    if len(body) < int(min_len):
        failures.append("too_short")

    residues = [
        tokenizer.id_to_char[token]
        for token in body
        if token in tokenizer.residue_ids
    ]
    return {
        "body_len": len(body),
        "n_aa": len(residues),
        "n_special_body": len(special_body),
        "n_invalid_body": len(invalid_body),
        "n_invalid_full": sum(
            token < 0 or token >= tokenizer.vocab_size for token in row
        ),
        "n_tail_violations": len(tail_violations),
        "residues": "".join(residues),
        "has_bos": has_bos,
        "has_eos": has_eos,
        "eos_index": int(eos_index) if has_eos else None,
        "strict_valid": not failures,
        "failure_reasons": failures,
    }


def decode_token_ids(
    token_ids: torch.Tensor, tokenizer: ProteinTokenizer, *, min_len: int = 1
) -> List[Dict]:
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    rows = token_ids.long().cpu().tolist()
    return [_decode_row(row, tokenizer, min_len=min_len) for row in rows]


# -----------------------------------------------------------------------------
# Composition + JSD
# -----------------------------------------------------------------------------

def _residue_alphabet(tokenizer: ProteinTokenizer) -> List[str]:
    return sorted(set(tokenizer.id_to_char.values()))


def _residue_frequencies(sequences: Sequence[str], alphabet: List[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(alphabet)}
    counts = np.zeros(len(alphabet), dtype=np.float64)
    for s in sequences:
        for c in s:
            j = idx.get(c)
            if j is not None:
                counts[j] += 1.0
    total = counts.sum()
    if total <= 0:
        return counts
    return counts / total


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # Base-2 JSD in [0, 1].
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_protein_metrics(
    gen_token_ids: torch.Tensor,
    tokenizer: ProteinTokenizer,
    *,
    reference_token_ids: Optional[torch.Tensor] = None,
    train_sequences: Optional[Set[str]] = None,
    min_len: int = 1,
) -> Dict:
    gen = decode_token_ids(gen_token_ids, tokenizer, min_len=min_len)
    n = len(gen)
    valid = [d for d in gen if d["strict_valid"]]

    tot_body = sum(d["body_len"] for d in gen)
    tot_aa = sum(d["n_aa"] for d in gen)
    tot_special = sum(d["n_special_body"] for d in gen)
    tot_invalid = sum(d["n_invalid_full"] for d in gen)
    tot_invalid_body = sum(d["n_invalid_body"] for d in gen)
    tot_positions = int(gen_token_ids.numel())

    valid_lengths = [d["n_aa"] for d in valid]
    valid_sequences = [d["residues"] for d in valid]
    failure_counts: Dict[str, int] = {}
    for record in gen:
        for reason in record["failure_reasons"]:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

    metrics: Dict = {
        "num_samples": int(n),
        "num_valid_samples": int(len(valid)),
        "valid_sample_fraction": (len(valid) / n) if n else float("nan"),
        "valid_token_rate_pre_eos": (tot_aa / tot_body) if tot_body else float("nan"),
        "invalid_code_rate_full_storage": (
            tot_invalid / tot_positions if tot_positions else float("nan")
        ),
        "invalid_code_rate_pre_eos": (
            tot_invalid_body / tot_body if tot_body else float("nan")
        ),
        "special_token_rate_pre_eos": (
            tot_special / tot_body if tot_body else float("nan")
        ),
        "frac_no_eos": (
            float(np.mean([not d["has_eos"] for d in gen])) if n else float("nan")
        ),
        "failure_counts": failure_counts,
        "valid_gen_len_mean": (
            float(np.mean(valid_lengths)) if valid_lengths else float("nan")
        ),
        "valid_gen_len_median": (
            float(np.median(valid_lengths)) if valid_lengths else float("nan")
        ),
        "valid_gen_len_std": (
            float(np.std(valid_lengths)) if valid_lengths else float("nan")
        ),
        "unique_fraction_valid": (
            len(set(valid_sequences)) / len(valid_sequences)
            if valid_sequences else float("nan")
        ),
    }

    alphabet = _residue_alphabet(tokenizer)
    generated_frequency = _residue_frequencies(valid_sequences, alphabet)

    if reference_token_ids is not None:
        reference = decode_token_ids(reference_token_ids, tokenizer, min_len=min_len)
        reference_valid = [d for d in reference if d["strict_valid"]]
        reference_lengths = [d["n_aa"] for d in reference_valid]
        reference_frequency = _residue_frequencies(
            [d["residues"] for d in reference_valid], alphabet
        )
        metrics["reference_num_samples"] = int(len(reference_valid))
        metrics["reference_len_mean"] = (
            float(np.mean(reference_lengths)) if reference_lengths else float("nan")
        )
        metrics["reference_len_median"] = (
            float(np.median(reference_lengths)) if reference_lengths else float("nan")
        )
        metrics["amino_acid_composition_jsd"] = _jensen_shannon_divergence(
            generated_frequency, reference_frequency
        )

    if train_sequences is not None:
        novel = sum(sequence not in train_sequences for sequence in valid_sequences)
        metrics["exact_novelty_valid"] = (
            novel / len(valid_sequences) if valid_sequences else float("nan")
        )

    return metrics


# -----------------------------------------------------------------------------
# ESM-2 realism scorer (BioNeMo / nvidia HF checkpoints)
# -----------------------------------------------------------------------------

def _facebook_esm2_equivalent(model_id: str) -> Optional[str]:
    # NVIDIA's nvidia/esm2_* checkpoints are conversions of the identical
    # facebook/esm2_* weights; the facebook ones load with plain transformers.
    name = model_id.split("/")[-1]
    return f"facebook/{name}" if name.startswith("esm2_") else None


def _load_esm2_masked_lm(model_id: str, device: torch.device):
    # NVIDIA nv_esm checkpoints need trust_remote_code (+ TransformerEngine); fall
    # back to the numerically identical facebook weights if that path is missing.
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
        return tok, model.to(device).eval()
    except Exception as e:  # noqa: BLE001
        fb = _facebook_esm2_equivalent(model_id)
        if fb is None or fb == model_id:
            raise
        print(f"[protein_metrics] {model_id} scorer load failed ({type(e).__name__}); using {fb}")
        tok = AutoTokenizer.from_pretrained(fb)
        model = AutoModelForMaskedLM.from_pretrained(fb)
        return tok, model.to(device).eval()


@torch.no_grad()
def esm2_pseudo_perplexity(
    sequences: Sequence[str],
    *,
    model_id: str,
    device: torch.device,
    micro_batch_size: int = 32,
    max_seqs: int = 256,
) -> Optional[float]:
    """
    Exact leave-one-residue-out pseudo-perplexity from an ESM-2 masked LM.

    For every non-special residue, replace that residue with the tokenizer's
    mask token and score the original residue at the masked position. Masked
    variants are evaluated in micro-batches. This is substantially more
    expensive than a teacher-forced pass, but unlike scoring an unmasked input
    it is a valid masked-LM pseudo-likelihood estimate.
    """
    seqs = [s for s in sequences if len(s) > 0][: int(max_seqs)]
    if not seqs:
        return None
    try:
        tok, model = _load_esm2_masked_lm(model_id, device)
    except Exception as e:  # noqa: BLE001
        print(f"[protein_metrics] ESM-2 scorer unavailable ({e}); skipping.")
        return None

    mask_token_id = getattr(tok, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError("ESM-2 tokenizer has no mask_token_id; cannot compute pseudo-perplexity.")

    variant_batch_size = max(1, int(micro_batch_size))
    special_ids = {int(i) for i in tok.all_special_ids}
    total_nll = 0.0
    total_tok = 0

    for seq in seqs:
        enc = tok(seq, return_tensors="pt", padding=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise ValueError(
                f"Expected one tokenized sequence with shape [1,L], got {tuple(input_ids.shape)}"
            )

        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).bool()
        residue_mask = attention_mask.clone()
        for special_id in special_ids:
            residue_mask &= input_ids.ne(special_id)

        positions = residue_mask[0].nonzero(as_tuple=False).squeeze(-1)
        if positions.numel() == 0:
            continue

        true_ids = input_ids[0, positions]
        seq_len = int(input_ids.size(1))

        for start in range(0, int(positions.numel()), variant_batch_size):
            pos = positions[start : start + variant_batch_size]
            targets = true_ids[start : start + variant_batch_size]
            batch_size = int(pos.numel())
            row = torch.arange(batch_size, device=device)

            masked_ids = input_ids.expand(batch_size, seq_len).clone()
            masked_ids[row, pos] = int(mask_token_id)

            model_inputs = {"input_ids": masked_ids}
            for key, value in enc.items():
                if key == "input_ids":
                    continue
                if value.dim() >= 1 and value.size(0) == 1:
                    model_inputs[key] = value.expand(batch_size, *value.shape[1:])
                else:
                    model_inputs[key] = value

            logits = model(**model_inputs).logits.float()
            masked_logits = logits[row, pos, :]
            total_nll += float(F.cross_entropy(masked_logits, targets, reduction="sum").item())
            total_tok += batch_size

    if total_tok == 0:
        return None
    return math.exp(total_nll / total_tok)

# -----------------------------------------------------------------------------
# Reference loaders
# -----------------------------------------------------------------------------

def load_reference_token_ids(
    root: Path, split: str, tok_tag: str, seq_len_tokens: int, min_len: int,
    *, limit: Optional[int] = None,
) -> torch.Tensor:
    mm, _ = load_token_id_cache(root, split, tok_tag, seq_len_tokens, min_len)
    n = int(mm.shape[0]) if limit is None else min(int(mm.shape[0]), int(limit))
    arr = np.array(mm[:n], dtype=np.int64, copy=True)
    return torch.from_numpy(arr)


def load_train_sequence_set(
    root: Path, tok_tag: str, seq_len_tokens: int, min_len: int, tokenizer: ProteinTokenizer,
    *, limit: Optional[int] = None,
) -> Set[str]:
    _, meta = load_token_id_cache(root, "train", tok_tag, seq_len_tokens, min_len)
    total = int(meta["n_sequences"])
    if limit is not None and int(limit) < total:
        print(
            f"[protein_metrics] WARNING: exact_novelty uses only the first "
            f"{int(limit):,}/{total:,} train windows (cap). Novelty may be overestimated. "
            f"Pass --train_novelty_cap 0 to use all."
        )
    train_ids = load_reference_token_ids(root, "train", tok_tag, seq_len_tokens, min_len, limit=limit)
    return {d["residues"] for d in decode_token_ids(train_ids, tokenizer)}


# -----------------------------------------------------------------------------
# Generation + evaluation runner
# -----------------------------------------------------------------------------

@torch.no_grad()
def generate_token_ids(
    cfg,
    model,
    device: torch.device,
    *,
    tokenizer: ProteinTokenizer,
    num_samples: int,
    micro_batch_size: int,
    sampler_name: str,
    decode_strategy: str = "protein_grammar_map",
) -> torch.Tensor:
    from evaluation.utils import sample_text_sequences_for_external

    bits_per_token = int(cfg.data.bits_per_token)
    chunks: List[torch.Tensor] = []
    remaining = int(num_samples)
    while remaining > 0:
        batch_size = min(int(micro_batch_size), remaining)
        bits, _prefix, _cL = sample_text_sequences_for_external(
            cfg=cfg,
            model=model,
            device=device,
            num_samples=batch_size,
            sampler_name=sampler_name,
            return_dict=False,
            decode_strategy=decode_strategy,
            codebook_size=int(tokenizer.vocab_size),
            bits_per_code=bits_per_token,
            valid_body_codes=sorted(tokenizer.residue_ids),
            bos_code=int(tokenizer.bos_id),
            eos_code=int(tokenizer.eos_id),
            pad_code=int(tokenizer.pad_id),
            min_body_codes=int(getattr(cfg.data, "min_len", 1)),
            warmup=False,
            ddp=False,
        )
        chunks.append(bitstreams_to_token_ids(bits.cpu(), bits_per_token))
        remaining -= batch_size
    return torch.cat(chunks, dim=0)


def _sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def build_run_manifest(
    *,
    config_path: Path,
    checkpoint_path: Path,
    dataset_root: Path,
    args: Dict,
    artifacts: Dict[str, Path],
) -> Dict:
    dataset_manifest_path = dataset_root / "raw" / "dataset_manifest.json"
    dataset_manifest = None
    if dataset_manifest_path.exists():
        with open(dataset_manifest_path, "r", encoding="utf-8") as f:
            dataset_manifest = json.load(f)

    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_args": args,
        "git": {
            "commit": _git_output("rev-parse", "HEAD"),
            "status_porcelain": _git_output("status", "--porcelain"),
        },
        "config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        },
        "dataset": {
            "root": str(dataset_root),
            "manifest": dataset_manifest,
            "manifest_sha256": _sha256(dataset_manifest_path),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpus": gpu_names,
        },
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in artifacts.items()
        },
    }


def main():
    ap = argparse.ArgumentParser("Generate protein samples and compute protein metrics")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None, help="Override cfg.evaluation.checkpoint_path")
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--micro_batch_size", type=int, default=128)
    ap.add_argument("--num_steps", type=int, default=128)
    ap.add_argument("--terminal_sigma", type=float, default=0.08)
    ap.add_argument("--sampler", type=str, default="heun_karras")
    ap.add_argument(
        "--decode_strategy",
        choices=["protein_grammar_map", "codebook_map", "threshold", "bernoulli"],
        default="protein_grammar_map",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference_split", choices=["val", "test"], default="test")
    ap.add_argument("--train_novelty_cap", type=int, default=0, help="0 = use all train windows")
    ap.add_argument("--no_esm2_score", action="store_true", help="Disable the evaluation-only ESM-2 scorer")
    ap.add_argument("--esm2_model", type=str, default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--esm2_max_seqs", type=int, default=256)
    ap.add_argument("--esm2_micro_batch", type=int, default=32)
    ap.add_argument("--out", type=str, default=None, help="Output JSON path")
    ap.add_argument("--save_fasta", type=str, default=None, help="Valid-sample FASTA path")
    args = ap.parse_args()

    from evaluation.utils import load_config, unwrap_all, load_checkpoint
    from models import create_model
    from utils.ema import EMA

    cfg = load_config(args.config)
    if args.checkpoint is not None:
        cfg.evaluation.checkpoint_path = args.checkpoint

    if not hasattr(cfg.train, "generation"):
        cfg.train.generation = config_dict.ConfigDict()
    cfg.train.generation.num_sampling_steps = int(args.num_steps)
    cfg.train.generation.terminal_sigmas = [float(args.terminal_sigma)]
    cfg.train.generation.entropic_blend_alpha = 0.0
    cfg.train.generation.entropy_ckpt_path = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    root = Path(cfg.data.root)
    tok_tag = tokenizer_tag_from_config(cfg)
    seq_len_tokens = int(cfg.data.sequence_len_tokens)
    min_len = int(getattr(cfg.data, "min_len", 20))
    tokenizer = make_protein_tokenizer(cfg)

    model = create_model(cfg).to(device)
    ema = EMA(unwrap_all(model), decay=0.0)
    checkpoint_path = Path(cfg.evaluation.checkpoint_path)
    load_checkpoint(model, ema, checkpoint_path, device, apply_ema=True)
    model.eval()

    print(
        f"Generating {args.num_samples} samples with sampler={args.sampler} "
        f"steps={args.num_steps} decode={args.decode_strategy} ..."
    )
    gen_ids = generate_token_ids(
        cfg,
        model,
        device,
        tokenizer=tokenizer,
        num_samples=int(args.num_samples),
        micro_batch_size=int(args.micro_batch_size),
        sampler_name=str(args.sampler),
        decode_strategy=str(args.decode_strategy),
    )

    cap = None if int(args.train_novelty_cap) <= 0 else int(args.train_novelty_cap)
    reference_ids = load_reference_token_ids(
        root, args.reference_split, tok_tag, seq_len_tokens, min_len
    )
    train_set = load_train_sequence_set(
        root, tok_tag, seq_len_tokens, min_len, tokenizer, limit=cap
    )

    decoded = decode_token_ids(gen_ids, tokenizer, min_len=min_len)
    valid_sequences = [d["residues"] for d in decoded if d["strict_valid"]]
    metrics = compute_protein_metrics(
        gen_ids,
        tokenizer,
        reference_token_ids=reference_ids,
        train_sequences=train_set,
        min_len=min_len,
    )
    metrics.update(
        checkpoint=str(checkpoint_path),
        num_sampling_steps=int(args.num_steps),
        terminal_sigma=float(args.terminal_sigma),
        sampler=str(args.sampler),
        decode_strategy=str(args.decode_strategy),
        tokenizer=tok_tag,
        reference_split=str(args.reference_split),
    )

    if not args.no_esm2_score:
        reference_decoded = decode_token_ids(reference_ids, tokenizer, min_len=min_len)
        reference_sequences = [
            d["residues"] for d in reference_decoded if d["strict_valid"]
        ]
        print(
            f"Scoring {min(len(valid_sequences), int(args.esm2_max_seqs))} valid "
            f"generations with evaluation-only ESM-2 ({args.esm2_model}) ..."
        )
        metrics["esm2_model"] = str(args.esm2_model)
        metrics["esm2_pseudo_perplexity"] = esm2_pseudo_perplexity(
            valid_sequences,
            model_id=str(args.esm2_model),
            device=device,
            micro_batch_size=int(args.esm2_micro_batch),
            max_seqs=int(args.esm2_max_seqs),
        )
        metrics["esm2_pseudo_perplexity_reference"] = esm2_pseudo_perplexity(
            reference_sequences,
            model_id=str(args.esm2_model),
            device=device,
            micro_batch_size=int(args.esm2_micro_batch),
            max_seqs=int(args.esm2_max_seqs),
        )

    out_path = (
        Path(args.out)
        if args.out
        else Path(cfg.evaluation.out_dir)
        / f"protein_metrics_{args.reference_split}_seed{args.seed}_n{args.num_samples}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = out_path.parent / f"{out_path.stem}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    token_path = artifact_dir / "generated_token_ids.pt"
    torch.save(gen_ids.cpu(), token_path)

    records_path = artifact_dir / "generated_records.jsonl"
    with open(records_path, "w", encoding="utf-8") as f:
        for i, (row, record) in enumerate(zip(gen_ids.tolist(), decoded)):
            payload = dict(record)
            payload["sample_id"] = int(i)
            payload["token_ids"] = [int(t) for t in row]
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    fasta_path = Path(args.save_fasta) if args.save_fasta else artifact_dir / "generated_valid.fasta"
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fasta_path, "w", encoding="utf-8") as f:
        valid_idx = 0
        for i, record in enumerate(decoded):
            if not record["strict_valid"]:
                continue
            f.write(f">gen_{i} valid_index={valid_idx}\n{record['residues']}\n")
            valid_idx += 1

    manifest = build_run_manifest(
        config_path=Path(args.config),
        checkpoint_path=checkpoint_path,
        dataset_root=root,
        args=vars(args),
        artifacts={
            "token_ids": token_path,
            "records": records_path,
            "valid_fasta": fasta_path,
        },
    )
    manifest_path = artifact_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    metrics["artifacts"] = {
        "directory": str(artifact_dir),
        "token_ids": str(token_path),
        "records": str(records_path),
        "valid_fasta": str(fasta_path),
        "manifest": str(manifest_path),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    # Attach these metrics to the training run's W&B run when one exists, so eval
    # numbers live alongside the loss curves. No-op if the run was not W&B-logged.
    try:
        from utils.wandb_eval import log_eval_metrics

        run_dir = Path(cfg.evaluation.out_dir).parent
        scalar_metrics = {k: v for k, v in metrics.items() if k != "artifacts"}
        if log_eval_metrics(
            str(run_dir), scalar_metrics, tag=str(args.reference_split)
        ):
            print(f"Logged eval metrics to W&B (run dir {run_dir}).")
    except Exception as exc:  # pragma: no cover - eval must not fail on logging
        print(f"[wandb] eval metric logging skipped: {exc!r}")

    print("\n=== Protein metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif key != "artifacts":
            print(f"  {key}: {value}")
    print(f"\nSaved metrics to {out_path}")
    print(f"Saved reproducibility artifacts to {artifact_dir}")


if __name__ == "__main__":
    main()
