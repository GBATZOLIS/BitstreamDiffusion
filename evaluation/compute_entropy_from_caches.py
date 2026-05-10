from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from data import get_loader
from evaluation.evaluation_drivers.utils import _resolve_eval_dirs
from evaluation.sampling_specs import build_sampling_specs
from evaluation.text_generations import resolve_checkpoints
from evaluation.text_metrics import token_unigram_entropy_from_token_ids
from evaluation.utils import load_config
from utils.text_decode import (
    bitstreams_to_token_ids_raw_binary,
    code_ids_to_gpt2_token_id_lists_for_eval,
    decode_bitstreams_to_token_ids_for_eval,
    decode_token_sequences_to_token_ids_for_eval,
    extract_dataset_attr,
    load_openwebtext_gpt2id_bpe16_assets,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _load_jsonl_cache(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    meta: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            if i == 0 and "_meta" in obj:
                meta = obj["_meta"] or {}
            else:
                rows.append(obj)

    return meta, rows


def _cache_file_info(path: Path) -> Dict[str, Any]:
    cache_key = path.parent.name
    split = path.parent.parent.name
    checkpoint = path.parent.parent.parent.name
    shared_cache_root = path.parent.parent.parent.parent
    eval_dir = shared_cache_root.parent

    return {
        "cache_file": str(path),
        "tag": path.stem,
        "cache_key": cache_key,
        "split": split,
        "checkpoint": checkpoint,
        "shared_cache_root": str(shared_cache_root),
        "eval_dir": str(eval_dir),
    }


def _generation_meta_fields(meta: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "generation_wall_time_sec",
        "world_size",
        "num_samples",
        "sequence_len_tokens",
        "sequence_len_model",
        "full_lm_tokens",
        "generated_lm_tokens",
        "full_model_positions",
        "generated_model_positions",
        "samples_per_sec",
        "full_lm_tokens_per_sec",
        "generated_lm_tokens_per_sec",
        "lm_tokens_per_sec",
        "full_model_positions_per_sec",
        "generated_model_positions_per_sec",
        "model_positions_per_sec",
        "sampler_name",
        "terminal_sigma",
        "guidance_scale",
        "num_steps",
        "nfe",
        "target_nfe",
        "sc_refresh_mode",
        "sigma_max",
        "use_ema",
        "ati_eta",
    ]
    return {k: meta[k] for k in keys if k in meta}


def _iter_cache_files(shared_root: Path) -> Iterable[Path]:
    if not shared_root.exists():
        return []
    return sorted(shared_root.rglob("*.jsonl"))


def _append_rows_csv(existing_csv: Path, new_rows: List[Dict[str, Any]]) -> None:
    existing: List[Dict[str, Any]] = []
    if existing_csv.exists():
        with existing_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.append(dict(row))

    def _key(r: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            r.get("checkpoint"),
            r.get("split"),
            r.get("tag"),
            r.get("metric"),
            r.get("view"),
            r.get("cache_key"),
        )

    seen = {_key(r) for r in existing}
    merged = existing[:]

    for row in new_rows:
        if _key(row) not in seen:
            merged.append(row)
            seen.add(_key(row))

    fieldnames = sorted({k for row in merged for k in row.keys() if k is not None})
    existing_csv.parent.mkdir(parents=True, exist_ok=True)
    with existing_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in merged:
            writer.writerow(row)


def _append_rows_jsonl(existing_jsonl: Path, new_rows: List[Dict[str, Any]]) -> None:
    existing: List[Dict[str, Any]] = []
    if existing_jsonl.exists():
        with existing_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))

    def _key(r: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            r.get("checkpoint"),
            r.get("split"),
            r.get("tag"),
            r.get("metric"),
            r.get("view"),
            r.get("cache_key"),
        )

    seen = {_key(r) for r in existing}
    merged = existing[:]

    for row in new_rows:
        if _key(row) not in seen:
            merged.append(row)
            seen.add(_key(row))

    existing_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with existing_jsonl.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_expected_external_ppl_tags(cfg) -> set[str]:
    ext_cfg = getattr(getattr(cfg, "evaluation", object()), "external_ppl", None)
    if ext_cfg is None:
        raise ValueError("Missing cfg.evaluation.external_ppl")

    specs = build_sampling_specs(cfg=cfg, metric_cfg=ext_cfg)
    if specs is None:
        samplers = list(getattr(ext_cfg, "samplers", ["ddim_entropic"]))
        terminal_sigmas = list(getattr(ext_cfg, "terminal_sigmas", [0.08]))
        guidance_scales = list(getattr(ext_cfg, "guidance_scales", [0.0]))
        out = set()
        for sampler_name in samplers:
            for sigma in terminal_sigmas:
                for gs in guidance_scales:
                    sigma_tag = f"{torch.log10(torch.tensor(float(sigma))).item():.2f}" if float(sigma) > 0.0 else "none"
                    out.add(f"{sampler_name}_term{sigma_tag}_gs{float(gs):.1f}")
        return out

    return {str(spec["tag"]) for spec in specs}


def _build_expected_checkpoints(cfg) -> set[str]:
    ext_cfg = getattr(getattr(cfg, "evaluation", object()), "external_ppl", None)
    ckpt_spec = getattr(ext_cfg, "checkpoints", None)
    base_ckpt_dir = Path(cfg.evaluation.checkpoint_path).parent
    checkpoints_to_run = resolve_checkpoints(ckpt_spec, base_ckpt_dir)
    if ckpt_spec is None and len(checkpoints_to_run) == 0:
        return {Path(cfg.evaluation.checkpoint_path).name}
    return {p.name for p in checkpoints_to_run} if checkpoints_to_run else {Path(cfg.evaluation.checkpoint_path).name}


def _rows_to_tensor(rows: List[Dict[str, Any]], key: str) -> Optional[torch.Tensor]:
    vals = []
    for row in rows:
        v = row.get(key, None)
        if v is None:
            return None
        vals.append(v)
    if len(vals) == 0:
        return None
    return torch.tensor(vals, dtype=torch.long)


@torch.no_grad()
def _avg_entropy_from_token_rows(token_rows: List[List[int]]) -> float:
    if len(token_rows) == 0:
        return float("nan")
    vals = []
    for row in token_rows:
        vals.append(token_unigram_entropy_from_token_ids(torch.tensor(row, dtype=torch.long)))
    return float(sum(vals) / max(len(vals), 1))


def _extract_token_rows(
    *,
    cfg,
    rows: List[Dict[str, Any]],
    ref: bool,
    dataset_obj: Any,
) -> List[List[int]]:
    """
    Return exact evaluation token rows.

    For the OWT gpt2id_bpe16 setup, this uses the centralized cleanup logic
    in text_decode.py and returns variable-length cleaned GPT-2 token rows.

    For other setups, it falls back to dense token-id decoding and converts rows
    to Python lists.
    """
    ds_name = str(getattr(cfg.data, "dataset", "")).strip().lower()
    seq_codec = str(getattr(cfg.data, "sequence_codec", "base")).strip().lower()

    if ref:
        bits = _rows_to_tensor(rows, "ref_bits")
        toks = _rows_to_tensor(rows, "ref_tokens")
    else:
        bits = _rows_to_tensor(rows, "gen_bits")
        toks = _rows_to_tensor(rows, "gen_tokens")

    # Special case: OWT second-stage GPT2->BPE16 codec.
    # We must preserve the true cleaned GPT-2 row lengths, otherwise EOS padding
    # deflates token entropy.
    if ds_name == "openwebtext" and seq_codec == "gpt2id_bpe16":
        root = Path(getattr(cfg.data, "root", "./datasets/openwebtext"))
        tokenizer_name = str(getattr(cfg.data, "tokenizer_name", "gpt2"))
        code_tokenizer_path = str(getattr(cfg.data, "code_tokenizer_path"))
        code_tokenizer_meta_path = getattr(cfg.data, "code_tokenizer_meta_path", None)

        gpt2_tok = extract_dataset_attr(dataset_obj, "tokenizer") if dataset_obj is not None else None
        code_tok = extract_dataset_attr(dataset_obj, "code_tokenizer") if dataset_obj is not None else None
        code_meta = extract_dataset_attr(dataset_obj, "code_meta") if dataset_obj is not None else None

        if gpt2_tok is None or code_tok is None or code_meta is None:
            gpt2_tok, code_tok, code_meta = load_openwebtext_gpt2id_bpe16_assets(
                root=root,
                code_tokenizer_path=code_tokenizer_path,
                code_tokenizer_meta_path=code_tokenizer_meta_path,
                tokenizer_name=tokenizer_name,
            )

        if bits is not None:
            code_ids = bitstreams_to_token_ids_raw_binary(
                bits,
                bits_per_token=int(getattr(cfg.data, "bits_per_token", 16)),
                cfg=cfg,
            )
        elif toks is not None:
            code_ids = toks.to(torch.long)
        else:
            raise RuntimeError("Cache does not contain raw gen/ref bits or token sequences.")

        return code_ids_to_gpt2_token_id_lists_for_eval(
            code_ids,
            gpt2_tokenizer=gpt2_tok,
            code_tokenizer=code_tok,
            code_meta=code_meta,
            return_debug_stats=False,
        )

    # Generic fallback: decode to dense token ids, then convert each row to list.
    if bits is not None:
        dense = decode_bitstreams_to_token_ids_for_eval(
            cfg,
            bits,
            dataset_obj=dataset_obj,
        )
        return [row.tolist() for row in dense]

    if toks is not None:
        dense = decode_token_sequences_to_token_ids_for_eval(
            cfg,
            toks,
            dataset_obj=dataset_obj,
        )
        return [row.tolist() for row in dense]

    raise RuntimeError("Cache does not contain raw gen/ref bits or token sequences.")


def _make_rows_for_cache(
    *,
    cfg,
    cache_path: Path,
    rows: List[Dict[str, Any]],
    meta: Dict[str, Any],
    include_real: bool,
    dataset_obj: Any,
) -> List[Dict[str, Any]]:
    info = _cache_file_info(cache_path)
    common = {
        "timestamp_utc": _utc_now_iso(),
        "experiment": str(cfg.experiment),
        "framework": str(cfg.framework),
        "dataset": str(getattr(cfg.data, "dataset", "")),
        **info,
        **_generation_meta_fields(meta),
        "metric_source": "posthoc_cached_token_entropy",
        "num_rows": len(rows),
        "view": "full",
    }

    gen_token_rows = _extract_token_rows(
        cfg=cfg,
        rows=rows,
        ref=False,
        dataset_obj=dataset_obj,
    )
    gen_entropy = _avg_entropy_from_token_rows(gen_token_rows)

    out = [
        {
            **common,
            "metric": "gen_full_token_unigram_entropy",
            "value": gen_entropy,
            "details": "source=cache,entropy=token_unigram,clean_eval_rows=1",
        }
    ]

    if include_real:
        ref_token_rows = _extract_token_rows(
            cfg=cfg,
            rows=rows,
            ref=True,
            dataset_obj=dataset_obj,
        )
        ref_entropy = _avg_entropy_from_token_rows(ref_token_rows)

        out.append(
            {
                **common,
                "metric": "real_full_token_unigram_entropy",
                "value": ref_entropy,
                "details": "source=cache,entropy=token_unigram,clean_eval_rows=1",
            }
        )

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--include_real", action="store_true")
    ap.add_argument("--split", type=str, default="test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir, _, results_csv = _resolve_eval_dirs(cfg)
    results_jsonl = results_csv.with_suffix(".jsonl")

    shared_root = Path(
        getattr(getattr(cfg, "evaluation", object()), "shared_text_cache_dir", None)
        or (out_dir / "shared_text_cache")
    )

    ext_cfg = getattr(getattr(cfg, "evaluation", object()), "external_ppl", None)
    if ext_cfg is None:
        raise ValueError("Missing cfg.evaluation.external_ppl")

    include_real = bool(args.include_real or getattr(ext_cfg, "compute_real_reference", True))
    expected_tags = _build_expected_external_ppl_tags(cfg)
    expected_checkpoints = _build_expected_checkpoints(cfg)
    expected_split = str(args.split)

    cache_files = list(_iter_cache_files(shared_root))
    if len(cache_files) == 0:
        print(f"No cache files found under: {shared_root}")
        return

    loader = get_loader(cfg, split=expected_split)
    dataset_obj = getattr(loader, "dataset", None)
    if dataset_obj is None:
        raise RuntimeError(f"Could not resolve dataset object for split='{expected_split}'")

    all_rows: List[Dict[str, Any]] = []

    for cache_path in cache_files:
        if cache_path.stem not in expected_tags:
            continue

        meta, rows = _load_jsonl_cache(cache_path)
        info = _cache_file_info(cache_path)

        if str(info["checkpoint"]) not in expected_checkpoints:
            continue
        if str(info["split"]) != expected_split:
            continue
        if len(rows) == 0:
            continue

        out_rows = _make_rows_for_cache(
            cfg=cfg,
            cache_path=cache_path,
            rows=rows,
            meta=meta,
            include_real=include_real,
            dataset_obj=dataset_obj,
        )
        all_rows.extend(out_rows)

    if len(all_rows) == 0:
        print("No matching external_ppl cache files were found.")
        print(f"shared_root={shared_root}")
        print(f"expected_checkpoints={sorted(expected_checkpoints)}")
        print(f"num_expected_tags={len(expected_tags)}")
        return

    all_rows.sort(
        key=lambda r: (
            str(r.get("checkpoint", "")),
            str(r.get("sampler_name", "")),
            _safe_int(r.get("nfe")) if r.get("nfe") is not None else 10**9,
            str(r.get("tag", "")),
            str(r.get("metric", "")),
        )
    )

    _append_rows_csv(results_csv, all_rows)
    _append_rows_jsonl(results_jsonl, all_rows)

    print(f"Appended {len(all_rows)} entropy rows.")
    print(f"CSV:   {results_csv}")
    print(f"JSONL: {results_jsonl}")


if __name__ == "__main__":
    main()