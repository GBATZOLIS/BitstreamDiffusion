#!/usr/bin/env python3
"""Evaluate canonical ground truth for text-audio tasks.

Results are written beneath ``datasets/test_common`` by default. Select one or
more tasks with ``--tasks``; without it, all tasks are evaluated.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from utils.textaudio_utils import CONT_TRIM_SECONDS, TextAudioEvaluator, _trim_prefix_seconds


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _merge_result(path: Path, task: str, metrics: dict[str, Any]) -> None:
    results = _read_json(path) if path.is_file() else {}
    results[task] = metrics
    _write_json(path, results)


def _meta_ids(path: Path) -> tuple[dict[str, Any], list[str]]:
    meta = _read_json(path)
    ids = [str(value) for value in meta.get("ids", [])]
    expected = int(meta.get("n_sequences", -1))
    if not ids or len(ids) != expected or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: expected {expected} unique IDs, found {len(ids)}")
    return meta, ids


def _decode_text_cache(cache_path: Path, meta: dict[str, Any]) -> list[str]:
    import tiktoken

    rows = int(meta["n_sequences"])
    width = int(meta["seq_len_tokens"])
    expected_bytes = rows * width * np.dtype(np.uint32).itemsize
    if cache_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{cache_path}: found {cache_path.stat().st_size} bytes, expected {expected_bytes}"
        )
    cache = np.memmap(cache_path, dtype=np.uint32, mode="r", shape=(rows, width))
    text_length = int(meta["text_seq_len"])
    pad_token = int(meta["pad_token_text"])
    tokenizer = tiktoken.get_encoding("o200k_base")
    return [
        tokenizer.decode([int(token) for token in row[:text_length] if int(token) != pad_token])
        for row in cache
    ]


def _load_wavs(path: Path, expected: int) -> tuple[list[np.ndarray], int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        wavs = [np.asarray(wav, dtype=np.float32).flatten() for wav in archive["wavs"]]
        sample_rate = int(archive["sample_rate"])
    if len(wavs) != expected or any(wav.size == 0 for wav in wavs):
        raise ValueError(f"{path}: expected {expected} non-empty waveforms, found {len(wavs)}")
    return wavs, sample_rate


def _load_dataset_and_matches(meta: dict[str, Any], heldout_map: dict[str, str]):
    from datasets import Audio, load_dataset

    config = meta.get("hf_config")
    if config is None:
        dataset = load_dataset(meta["hf_path"], split=meta["hf_split"])
    else:
        dataset = load_dataset(meta["hf_path"], config, split=meta["hf_split"])
    dataset = dataset.cast_column("audio", Audio(decode=False))
    source_ids = [str(value) for value in dataset["id"]]
    id_to_index = {sample_id: index for index, sample_id in enumerate(source_ids)}
    actual_ids = [str(value) for value in meta["ids"]]
    missing = [sample_id for sample_id in actual_ids if sample_id not in id_to_index]
    if missing:
        raise ValueError(f"TTS IDs absent from source dataset: {missing[:5]}")
    speaker_ids = [
        str(dataset[id_to_index[sample_id]]["speaker_id"])
        for sample_id in actual_ids
    ]
    missing_speakers = sorted(set(speaker_ids) - set(heldout_map))
    if missing_speakers:
        raise ValueError(f"Speakers absent from held-out map: {missing_speakers[:5]}")
    matched_ids = [heldout_map[speaker] for speaker in speaker_ids]
    missing_heldout = sorted(set(matched_ids) - set(id_to_index))
    if missing_heldout:
        raise ValueError(f"Held-out IDs absent from source dataset: {missing_heldout[:5]}")
    return dataset, id_to_index, actual_ids, speaker_ids, matched_ids


def _load_actual_embeddings(
    path: Path, ids_key: str, embeddings_key: str, expected_ids: list[str],
) -> np.ndarray:
    with np.load(path) as archive:
        ids = [str(value) for value in archive[ids_key]]
        embeddings = np.asarray(archive[embeddings_key], dtype=np.float32)
    if ids != expected_ids:
        raise ValueError(f"{path}: embedding IDs are not in TTS cache order")
    if embeddings.ndim != 2 or len(embeddings) != len(ids):
        raise ValueError(f"{path}: invalid embedding shape {embeddings.shape}")
    return embeddings


def _extract_heldout_embeddings(
    dataset, id_to_index: dict[str, int], heldout_map: dict[str, str],
    checkpoint: Path, device: torch.device,
):
    from scripts.text_audio.cache_utils import decode_wav
    from utils.speaker_verification import ECAPA_TDNN_SMALL

    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"], strict=False)
    model.to(device).eval()
    speakers = list(heldout_map)
    heldout_ids = [heldout_map[speaker] for speaker in speakers]
    embeddings = []
    with torch.inference_mode():
        for index, sample_id in enumerate(heldout_ids):
            wav = decode_wav(dataset[id_to_index[sample_id]]["audio"]["bytes"])
            value = model(torch.from_numpy(wav).unsqueeze(0).to(device)).squeeze(0)
            embeddings.append(value.cpu())
            if (index + 1) % 50 == 0:
                print(f"[heldout speakers] embedded {index + 1}/{len(heldout_ids)}", flush=True)
    values = F.normalize(torch.stack(embeddings), dim=-1).numpy().astype(np.float32)
    return speakers, heldout_ids, values


def _score_and_write(
    *, output_dir: Path, actual_ids: list[str], actual_speaker_ids: list[str],
    matched_heldout_ids: list[str], actual_embeddings: np.ndarray,
    heldout_speaker_ids: list[str], unique_heldout_ids: list[str],
    heldout_embeddings: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_speaker = {
        speaker: heldout_embeddings[index]
        for index, speaker in enumerate(heldout_speaker_ids)
    }
    matched_embeddings = np.stack([by_speaker[speaker] for speaker in actual_speaker_ids])
    actual_unit = actual_embeddings / np.maximum(
        np.linalg.norm(actual_embeddings, axis=1, keepdims=True), 1e-12
    )
    matched_unit = matched_embeddings / np.maximum(
        np.linalg.norm(matched_embeddings, axis=1, keepdims=True), 1e-12
    )
    similarities = np.sum(actual_unit * matched_unit, axis=1).astype(np.float32)
    speaker_array = np.asarray(actual_speaker_ids)
    per_speaker = {}
    for speaker, heldout_id in zip(heldout_speaker_ids, unique_heldout_ids):
        values = similarities[speaker_array == speaker]
        per_speaker[speaker] = {
            "heldout_id": heldout_id,
            "n_actual_samples": int(len(values)),
            "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()),
        }
    summary = {
        "n_actual_samples": len(actual_ids), "n_speakers": len(heldout_speaker_ids),
        "mean": float(similarities.mean()), "std": float(similarities.std()),
        "min": float(similarities.min()), "max": float(similarities.max()),
        "per_speaker": per_speaker,
    }
    np.savez(
        output_dir / "tts_heldout_spksim.npz",
        actual_ids=np.asarray(actual_ids), speaker_ids=speaker_array,
        matched_heldout_ids=np.asarray(matched_heldout_ids),
        actual_embeddings=actual_embeddings,
        matched_heldout_embeddings=matched_embeddings,
        cosine_similarity=similarities,
        unique_speaker_ids=np.asarray(heldout_speaker_ids),
        unique_heldout_ids=np.asarray(unique_heldout_ids),
        unique_heldout_embeddings=heldout_embeddings,
    )
    with (output_dir / "tts_heldout_spksim.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("actual_id", "speaker_id", "matched_heldout_id", "cosine_similarity"))
        writer.writerows(zip(actual_ids, actual_speaker_ids, matched_heldout_ids, similarities))
    _write_json(output_dir / "tts_heldout_spksim_summary.json", summary)

def _run_heldout_tts_analysis(
    *,
    meta: dict[str, Any],
    embeddings_path: Path,
    heldout_path: Path,
    speaker_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    from scripts.text_audio.cache_utils import load_heldout_map

    heldout_map = {
        str(speaker): str(sample_id)
        for speaker, sample_id in load_heldout_map(str(heldout_path)).items()
    }
    dataset, id_to_index, actual_ids, speaker_ids, matched_ids = (
        _load_dataset_and_matches(meta, heldout_map)
    )
    actual_embeddings = _load_actual_embeddings(
        embeddings_path, "clean_ids", "clean_embeddings", actual_ids
    )
    used_speakers = sorted(set(speaker_ids))
    used_map = {speaker: heldout_map[speaker] for speaker in used_speakers}
    heldout_speakers, unique_heldout_ids, heldout_embeddings = _extract_heldout_embeddings(
        dataset, id_to_index, used_map, speaker_checkpoint, device
    )
    _score_and_write(
        output_dir=output_dir,
        actual_ids=actual_ids,
        actual_speaker_ids=speaker_ids,
        matched_heldout_ids=matched_ids,
        actual_embeddings=actual_embeddings,
        heldout_speaker_ids=heldout_speakers,
        unique_heldout_ids=unique_heldout_ids,
        heldout_embeddings=heldout_embeddings,
    )
    summary = _read_json(output_dir / "tts_heldout_spksim_summary.json")
    del dataset, actual_embeddings, heldout_embeddings
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _load_training_cache(cache_path: Path, meta_path: Path, limit_rows: int | None):
    meta = _read_json(meta_path)
    total_rows = int(meta["n_sequences"])
    width = int(meta["seq_len_tokens"])
    expected_bytes = total_rows * width * np.dtype(np.uint32).itemsize
    if cache_path.stat().st_size != expected_bytes:
        raise ValueError(f"{cache_path}: cache size does not match metadata")
    num_rows = total_rows if limit_rows is None else min(total_rows, int(limit_rows))
    cache = np.memmap(cache_path, np.uint32, "r", shape=(total_rows, width))
    start = int(meta["text_seq_len"])
    speaker_slice = slice(start, start + int(meta["speaker_seq_len"]))
    return cache, meta, num_rows, speaker_slice


def _load_unit_bicodec_codebook(model_dir: Path, vocab_size: int) -> torch.Tensor:
    spark_root = Path("Spark-TTS").resolve()
    if str(spark_root) not in sys.path:
        sys.path.insert(0, str(spark_root))
    from sparktts.models.bicodec import BiCodec

    model = BiCodec.load_from_checkpoint(model_dir=model_dir).cpu().eval()
    quantizer = model.speaker_encoder.quantizer
    indices = torch.arange(vocab_size).reshape(-1, 1, 1)
    with torch.inference_mode():
        codebook = quantizer.get_output_from_indices(indices).squeeze(1).float()
    if codebook.ndim != 2 or len(codebook) != vocab_size:
        raise ValueError(f"Unexpected BiCodec codebook shape: {tuple(codebook.shape)}")
    codebook = F.normalize(codebook, dim=-1, eps=1e-12).cpu()
    del model
    return codebook


def _sample_diversity(
    tokens: np.ndarray, vocab_size: int, unit_codebook: torch.Tensor,
) -> tuple[np.ndarray, float, float]:
    n, positions = tokens.shape
    if np.any(tokens < 0) or np.any(tokens >= vocab_size):
        raise ValueError("Training cache contains an invalid speaker token")
    pair_count = n * (n - 1) / 2
    entropies, hamming, latent_distance = [], [], []
    for position in range(positions):
        values = tokens[:, position]
        counts = np.bincount(values, minlength=vocab_size).astype(np.float64)
        probabilities = counts[counts > 0] / n
        entropies.append(float(-(probabilities * np.log2(probabilities)).sum()))
        identical = float((counts * (counts - 1) / 2).sum())
        hamming.append(1.0 - identical / pair_count)
        vectors = unit_codebook[torch.from_numpy(values)].double()
        cosine_sum = float((vectors.sum(dim=0).square().sum() - n) / 2)
        latent_distance.append(1.0 - cosine_sum / pair_count)
    return np.asarray(entropies), float(np.mean(hamming)), float(np.mean(latent_distance))


def _matched_size_reference(
    *, cache, num_rows: int, speaker_slice: slice, speaker_offset: int,
    speaker_vocab: int, sample_size: int, repeats: int, seed: int,
    unit_codebook: torch.Tensor,
) -> dict[str, Any]:
    if sample_size > num_rows:
        raise ValueError(f"sample size {sample_size} exceeds {num_rows} training rows")
    rng = np.random.default_rng(seed)
    entropy_trials, hamming_trials, latent_trials = [], [], []
    for trial in range(repeats):
        indices = rng.choice(num_rows, size=sample_size, replace=False)
        tokens = np.asarray(cache[indices, speaker_slice], dtype=np.int64) - speaker_offset
        entropy, hamming, latent = _sample_diversity(tokens, speaker_vocab, unit_codebook)
        entropy_trials.append(entropy)
        hamming_trials.append(hamming)
        latent_trials.append(latent)
        print(f"[training diversity] trial {trial + 1}/{repeats}", flush=True)
    entropy_values = np.stack(entropy_trials)
    scalar_entropy = entropy_values.mean(axis=1)
    ddof = 1 if repeats > 1 else 0

    def summary(values) -> dict[str, float]:
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "sample_standard_deviation": float(values.std(ddof=ddof)),
        }

    return {
        "sample_size": sample_size,
        "repeats": repeats,
        "seed": seed,
        "sampling": "independent draws without replacement within each trial",
        "metrics": {
            "mean_per_position_entropy_bits": summary(scalar_entropy),
            "mean_pairwise_normalized_hamming_distance": summary(hamming_trials),
            "mean_pairwise_latent_cosine_distance": summary(latent_trials),
        },
        "per_position_entropy_bits": {
            "mean": entropy_values.mean(axis=0).tolist(),
            "sample_standard_deviation": entropy_values.std(axis=0, ddof=ddof).tolist(),
        },
        "trial_values": {
            "mean_per_position_entropy_bits": scalar_entropy.tolist(),
            "mean_pairwise_normalized_hamming_distance": hamming_trials,
            "mean_pairwise_latent_cosine_distance": latent_trials,
        },
    }


def _full_entropy_reference(
    *, cache, num_rows: int, speaker_slice: slice, speaker_offset: int,
    speaker_vocab: int, chunk_rows: int,
) -> dict[str, Any]:
    positions = speaker_slice.stop - speaker_slice.start
    counts = np.zeros((positions, speaker_vocab), dtype=np.int64)
    for start in range(0, num_rows, chunk_rows):
        stop = min(start + chunk_rows, num_rows)
        tokens = np.asarray(cache[start:stop, speaker_slice], dtype=np.int64) - speaker_offset
        if np.any(tokens < 0) or np.any(tokens >= speaker_vocab):
            raise ValueError(f"Invalid speaker token in training rows {start}:{stop}")
        for position in range(positions):
            counts[position] += np.bincount(tokens[:, position], minlength=speaker_vocab)
        print(f"[training entropy] {stop}/{num_rows}", flush=True)
    entropies = []
    for position_counts in counts:
        probabilities = position_counts[position_counts > 0] / num_rows
        entropies.append(float(-(probabilities * np.log2(probabilities)).sum()))
    return {
        "num_samples": num_rows,
        "mean_per_position_entropy_bits": float(np.mean(entropies)),
        "per_position_entropy_bits": entropies,
        "pairwise_metrics_computed": False,
    }

def _run_joint_diversity(args: argparse.Namespace) -> dict[str, Any]:

    cache, meta, num_rows, speaker_slice = _load_training_cache(
        args.training_cache, args.training_meta, args.limit_rows
    )
    speaker_offset = int(meta["speaker_offset"])
    speaker_vocab = int(meta["speaker_vocab"])
    unit_codebook = _load_unit_bicodec_codebook(args.bicodec_model_dir, speaker_vocab)
    matched = _matched_size_reference(
        cache=cache,
        num_rows=num_rows,
        speaker_slice=speaker_slice,
        speaker_offset=speaker_offset,
        speaker_vocab=speaker_vocab,
        sample_size=args.sample_size,
        repeats=args.repeats,
        seed=args.seed,
        unit_codebook=unit_codebook,
    )
    full = _full_entropy_reference(
        cache=cache,
        num_rows=num_rows,
        speaker_slice=speaker_slice,
        speaker_offset=speaker_offset,
        speaker_vocab=speaker_vocab,
        chunk_rows=args.chunk_rows,
    )
    return {
        "cache": str(args.training_cache.resolve()),
        "metadata": str(args.training_meta.resolve()),
        "bicodec_model_dir": str(args.bicodec_model_dir.resolve()),
        "population_rows_in_metadata": int(meta["n_sequences"]),
        "analyzed_rows": int(num_rows),
        "speaker_sequence_length": int(meta["speaker_seq_len"]),
        "speaker_vocab_size": speaker_vocab,
        "latent_metric_definition": (
            "mean over positions and unordered sample pairs of 1 - "
            "cosine(unit BiCodec quantized latent[token_i], unit BiCodec quantized latent[token_j])"
        ),
        "matched_size_training_reference": matched,
        "full_size_training_reference": full,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+", choices=("tts", "cont", "cont_taste", "joint"),
        default=("tts", "cont_taste", "joint"),
    )
    parser.add_argument("--reference-dir", type=Path, default=Path("datasets/test_common"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("datasets/test_common/ground_truth")
    )
    parser.add_argument("--tts-meta", default="cache_test_clean_tts_632.meta.json")
    parser.add_argument("--tts-cache", default="cache_test_clean_tts_632_common.uint32")
    parser.add_argument("--cont-meta", default="cache_test_cont.meta.json")
    parser.add_argument("--ref-audio", default="cache_test_clean_tts.ref_audio.npz")
    parser.add_argument("--speaker-embeddings", default="ref_speaker_embeddings_common.npz")
    parser.add_argument("--cont-speaker-embeddings", default="ref_speaker_embeddings_cont.npz")
    parser.add_argument("--heldout-map", default="heldout.csv")
    parser.add_argument(
        "--speaker-checkpoint", type=Path,
        default=Path("assets/text_audio/wavlm_large_finetune.pth"),
    )
    parser.add_argument("--text-model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--training-cache", type=Path, default=Path("datasets/textaudio/cache_textaudio_train.uint32"))
    parser.add_argument("--training-meta", type=Path, default=Path("datasets/textaudio/cache_textaudio_train.meta.json"))
    parser.add_argument(
        "--bicodec-model-dir", type=Path,
        default=Path("Spark-TTS/pretrained_models/SparkTTS-0.5B/BiCodec"),
    )
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.sample_size < 2 or args.repeats < 1 or args.chunk_rows < 1:
        parser.error("--sample-size >= 2, --repeats >= 1, and --chunk-rows >= 1 are required")
    args.tasks = list(dict.fromkeys(
        "cont" if task == "cont_taste" else task for task in args.tasks
    ))
    return args


def main() -> None:
    args = _parse_args()
    tasks = set(args.tasks)
    reference_dir = args.reference_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tts_meta: dict[str, Any] | None = None
    tts_ids: list[str] = []
    tts_wavs: list[np.ndarray] = []
    tts_texts: list[str] = []
    sample_rate = 16_000
    if tasks & {"tts", "cont"}:
        tts_meta, tts_ids = _meta_ids(reference_dir / args.tts_meta)
        tts_wavs, sample_rate = _load_wavs(reference_dir / args.ref_audio, len(tts_ids))
        tts_texts = _decode_text_cache(reference_dir / args.tts_cache, tts_meta)
        if len(tts_texts) != len(tts_ids):
            raise ValueError("TTS text/cache row count mismatch")

    cont_meta: dict[str, Any] | None = None
    cont_ids: list[str] = []
    cont_rows: list[int] = []
    if "cont" in tasks:
        cont_meta, cont_ids = _meta_ids(reference_dir / args.cont_meta)
        tts_row = {sample_id: index for index, sample_id in enumerate(tts_ids)}
        missing = [sample_id for sample_id in cont_ids if sample_id not in tts_row]
        if missing:
            raise ValueError(f"Continuation IDs absent from TTS cache: {missing[:5]}")
        cont_rows = [tts_row[sample_id] for sample_id in cont_ids]
        with np.load(reference_dir / args.cont_speaker_embeddings) as archive:
            embedding_ids = [str(value) for value in archive["ids"]]
        if embedding_ids != cont_ids:
            raise ValueError("Continuation speaker embeddings are not in cache-ID order")

    if "tts" in tasks:
        for filename in (args.speaker_embeddings, args.heldout_map):
            if not (reference_dir / filename).is_file():
                raise FileNotFoundError(reference_dir / filename)
    if "joint" in tasks:
        for path in (args.training_cache, args.training_meta, args.bicodec_model_dir):
            if not path.exists():
                raise FileNotFoundError(path)

    print(
        f"Validated tasks={sorted(tasks)}; TTS={len(tts_ids)}, continuation={len(cont_ids)}",
        flush=True,
    )
    if args.validate_only:
        return

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    evaluator: TextAudioEvaluator | None = None
    if tasks & {"tts", "cont"}:
        evaluator = TextAudioEvaluator(
            sr=sample_rate,
            speaker_checkpoint=str(args.speaker_checkpoint),
            text_model=args.text_model,
            continuation_speaker_ref_path=str(reference_dir / args.cont_speaker_embeddings),
            _dbg_func=lambda message: print(f"[GroundTruth] {message}", flush=True),
        )

    if "tts" in tasks:
        assert tts_meta is not None and evaluator is not None
        heldout_summary = _run_heldout_tts_analysis(
            meta=tts_meta,
            embeddings_path=reference_dir / args.speaker_embeddings,
            heldout_path=reference_dir / args.heldout_map,
            speaker_checkpoint=args.speaker_checkpoint,
            output_dir=args.output_dir / "tts",
            device=device,
        )
        metrics = {
            "UTMOS": float(evaluator.utmos_score(tts_wavs)),
            "Heldout-SpkSim": float(heldout_summary["mean"]),
        }
        _merge_result(args.output_dir / "results.json", "tts", metrics)
        _write_json(
            args.output_dir / "tts" / "samples.json",
            [{"sample_id": sample_id} for sample_id in tts_ids],
        )
        print(f"TTS ground truth: {metrics}", flush=True)

    if "cont" in tasks:
        assert cont_meta is not None and evaluator is not None
        trim_seconds = float(cont_meta.get("trim_seconds", CONT_TRIM_SECONDS))
        full_wavs = [tts_wavs[index] for index in cont_rows]
        full_texts = [tts_texts[index] for index in cont_rows]
        valid_rows: list[int] = []
        continuations: list[np.ndarray] = []
        for row, wav in enumerate(full_wavs):
            continuation = _trim_prefix_seconds(wav, trim_seconds, sample_rate)
            if continuation is not None and continuation.size:
                valid_rows.append(row)
                continuations.append(continuation)
        if not continuations:
            raise ValueError("No ground-truth utterances extend beyond their prompts")
        valid_ids = [cont_ids[row] for row in valid_rows]
        valid_full_wavs = [full_wavs[row] for row in valid_rows]
        valid_texts = [full_texts[row] for row in valid_rows]
        metrics = {
            "UTMOS": float(evaluator.utmos_score(valid_full_wavs)),
            "SpkSim": float(evaluator.spksim(
                continuations, device, continuation=True, reference_indices=valid_rows
            )),
            "GenPPL-text": float(evaluator.gen_ppl(valid_texts, device)),
        }
        _merge_result(args.output_dir / "results.json", "cont_taste", metrics)
        _write_json(
            args.output_dir / "cont_taste" / "samples.json",
            [
                {"sample_id": sample_id, "ref_text": text}
                for sample_id, text in zip(valid_ids, valid_texts)
            ],
        )
        print(f"Continuation ground truth: {metrics}", flush=True)

    if "joint" in tasks:
        diversity = _run_joint_diversity(args)
        joint_path = args.output_dir / "joint" / "bicodec_diversity.json"
        _write_json(joint_path, diversity)
        matched_metrics = diversity["matched_size_training_reference"]["metrics"]
        joint_metrics = {
            name: float(summary["mean"]) for name, summary in matched_metrics.items()
        }
        joint_metrics["full_mean_per_position_entropy_bits"] = float(
            diversity["full_size_training_reference"]["mean_per_position_entropy_bits"]
        )
        _merge_result(args.output_dir / "results.json", "joint", joint_metrics)
        print(f"Joint ground truth written to {joint_path}", flush=True)


if __name__ == "__main__":
    main()
