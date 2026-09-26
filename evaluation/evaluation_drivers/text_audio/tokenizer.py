#!/usr/bin/env python3
"""Evaluate StableCodec reconstruction for text-audio evaluation caches.

Outputs are written beneath ``datasets/tokenizer`` by default. The TTS and
continuation caches share one codec pass when selected together; SALMON is
reported per consistency task.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.textaudio_utils import (
    CONT_TRIM_SECONDS,
    SALMON_JUDGE_PER_TASK,
    TextAudioEvaluator,
    _trim_prefix_seconds,
)

MIN_SPKSIM_SECONDS = 0.05
DEFAULT_SALMON_DATASET = "SpeechPPL/SALMon_with_meta"


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


def _load_wavs(path: Path, expected: int) -> tuple[list[np.ndarray], int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        wavs = [np.asarray(wav, dtype=np.float32).flatten() for wav in archive["wavs"]]
        sample_rate = int(archive["sample_rate"])
    if len(wavs) != expected or any(wav.size == 0 for wav in wavs):
        raise ValueError(f"{path}: expected {expected} non-empty waveforms, found {len(wavs)}")
    return wavs, sample_rate


def _validate_embeddings(path: Path, expected_ids: list[str], id_key: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    embedding_key = "embeddings" if id_key == "ids" else "clean_embeddings"
    with np.load(path) as archive:
        ids = [str(value) for value in archive[id_key]]
        embeddings = archive[embedding_key]
    if ids != expected_ids:
        raise ValueError(f"{path}: {id_key} are not in canonical cache order")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(expected_ids):
        raise ValueError(f"{path}: invalid {embedding_key} shape {embeddings.shape}")


@torch.inference_mode()
def _roundtrip_wavs(
    source_wavs: list[np.ndarray], codec: Any, device: torch.device, label: str
) -> list[np.ndarray]:
    downsampling_ratio = int(codec.model.downsampling_ratio)
    reconstructed: list[np.ndarray] = []
    for index, source in enumerate(source_wavs):
        source = np.asarray(source, dtype=np.float32).flatten()
        source_length = source.size
        audio = torch.from_numpy(source).view(1, 1, -1).to(device)
        padding = (-source_length) % downsampling_ratio
        if padding:
            audio = torch.nn.functional.pad(audio, (0, padding))
        audio = codec.volume_norm(audio)
        _, tokens = codec.encode(audio, posthoc_bottleneck=True, normalize=False)
        decoded = codec.decode(tokens, posthoc_bottleneck=True)
        reconstructed.append(decoded[0, 0, :source_length].cpu().float().numpy())
        if (index + 1) % 100 == 0 or index + 1 == len(source_wavs):
            print(f"[{label}] encoded and decoded {index + 1}/{len(source_wavs)}", flush=True)
    return reconstructed


def _load_salmon_task(
    dataset_name: str, split: str, task: str, reference_dir: Path
) -> tuple[Any, np.ndarray, np.ndarray]:
    from datasets import Audio, load_dataset

    print(f"[SALMON/{task}] loading {dataset_name!r} split={split!r}", flush=True)
    dataset = load_dataset(dataset_name, task, split=split)
    required = {"ind", "continuation_audio_positive", "continuation_audio_negative"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"SALMON/{task} is missing columns: {sorted(missing)}")
    dataset = dataset.cast_column("continuation_audio_positive", Audio(decode=False))
    embeddings_path = reference_dir / f"judge_embeddings_{task}.npz"
    if not embeddings_path.is_file():
        raise FileNotFoundError(embeddings_path)
    with np.load(embeddings_path) as archive:
        positive = np.asarray(archive["positive"], dtype=np.float32)
        negative = np.asarray(archive["negative"], dtype=np.float32)
    if positive.ndim != 2 or negative.shape != positive.shape:
        raise ValueError(
            f"{embeddings_path}: incompatible positive {positive.shape} and negative {negative.shape}"
        )
    if len(dataset) != positive.shape[0]:
        raise ValueError(
            f"SALMON/{task}: dataset has {len(dataset)} rows but embeddings have {positive.shape[0]}"
        )
    return dataset, positive, negative


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+",
        choices=("tts", "cont", "cont_taste", "salmon", "cont_salmon"),
        default=("tts", "cont_taste", "cont_salmon"),
        help="Evaluation tasks; cont and salmon are aliases for cont_taste and cont_salmon.",
    )
    parser.add_argument("--reference-dir", type=Path, default=Path("datasets/test_common"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/tokenizer"))
    parser.add_argument("--tts-meta", default="cache_test_clean_tts_632.meta.json")
    parser.add_argument("--cont-meta", default="cache_test_cont.meta.json")
    parser.add_argument("--ref-audio", default="cache_test_clean_tts.ref_audio.npz")
    parser.add_argument("--tts-reference-embeddings", default="ref_speaker_embeddings_common.npz")
    parser.add_argument("--cont-reference-embeddings", default="ref_speaker_embeddings_cont.npz")
    parser.add_argument("--wlm-statistics", default="fsd_ref_stats_cont_common_wlm.npz")
    parser.add_argument("--e2v-statistics", default="fsd_ref_stats_cont_common_e2v.npz")
    parser.add_argument("--speech-tokenizer", default="stabilityai/stable-codec-speech-16k")
    parser.add_argument("--speech-bottleneck", default="1x46656_400bps")
    parser.add_argument(
        "--speaker-checkpoint", default="assets/text_audio/wavlm_large_finetune.pth"
    )
    parser.add_argument("--e2v-model", default="iic/emotion2vec_base")
    parser.add_argument("--fsd-batch-size", type=int, default=32)
    parser.add_argument("--salmon-dataset", default=DEFAULT_SALMON_DATASET)
    parser.add_argument("--salmon-split", default="train")
    parser.add_argument(
        "--salmon-tasks", nargs="+", choices=tuple(SALMON_JUDGE_PER_TASK),
        default=tuple(SALMON_JUDGE_PER_TASK),
    )
    parser.add_argument("--salmon-reference-dir", type=Path, default=None)
    parser.add_argument("--judge-max-length", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.fsd_batch_size < 1:
        parser.error("--fsd-batch-size must be positive")
    aliases = {"cont": "cont_taste", "salmon": "cont_salmon"}
    args.tasks = list(dict.fromkeys(aliases.get(task, task) for task in args.tasks))
    if args.salmon_reference_dir is None:
        args.salmon_reference_dir = args.reference_dir / "salmon"
    return args


def main() -> None:
    args = _parse_args()
    tasks = set(args.tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = args.reference_dir

    tts_ids: list[str] = []
    source_wavs: list[np.ndarray] = []
    cont_ids: list[str] = []
    cont_rows: list[int] = []
    valid_cont_archive_rows: list[int] = []
    valid_cont_tts_rows: list[int] = []
    valid_cont_ids: list[str] = []
    if tasks & {"tts", "cont_taste"}:
        _, tts_ids = _meta_ids(reference_dir / args.tts_meta)
        source_wavs, sample_rate = _load_wavs(reference_dir / args.ref_audio, len(tts_ids))
        if sample_rate != args.sample_rate:
            raise ValueError(f"Reference audio sample rate is {sample_rate}, expected {args.sample_rate}")
        _validate_embeddings(
            reference_dir / args.tts_reference_embeddings, tts_ids, "clean_ids"
        )

    if "cont_taste" in tasks:
        cont_meta, cont_ids = _meta_ids(reference_dir / args.cont_meta)
        trim_seconds = float(cont_meta.get("trim_seconds", CONT_TRIM_SECONDS))
        if trim_seconds != CONT_TRIM_SECONDS:
            raise ValueError(
                f"Continuation cache uses {trim_seconds}s prompts; expected {CONT_TRIM_SECONDS}s"
            )
        _validate_embeddings(
            reference_dir / args.cont_reference_embeddings, cont_ids, "ids"
        )
        for filename in (args.wlm_statistics, args.e2v_statistics):
            if not (reference_dir / filename).is_file():
                raise FileNotFoundError(reference_dir / filename)
        tts_row = {sample_id: index for index, sample_id in enumerate(tts_ids)}
        missing = [sample_id for sample_id in cont_ids if sample_id not in tts_row]
        if missing:
            raise ValueError(f"Continuation IDs absent from TTS cache: {missing[:5]}")
        cont_rows = [tts_row[sample_id] for sample_id in cont_ids]
        minimum = int(np.ceil(MIN_SPKSIM_SECONDS * args.sample_rate))
        for archive_row, source_row in enumerate(cont_rows):
            continuation = _trim_prefix_seconds(
                source_wavs[source_row], CONT_TRIM_SECONDS, args.sample_rate
            )
            if continuation is not None and continuation.size >= minimum:
                valid_cont_archive_rows.append(archive_row)
                valid_cont_tts_rows.append(source_row)
                valid_cont_ids.append(cont_ids[archive_row])
        if not valid_cont_ids:
            raise ValueError("No continuation references are long enough for SpkSim")

    prepared_salmon: dict[str, tuple[Any, np.ndarray, np.ndarray]] = {}
    if "cont_salmon" in tasks:
        for task in args.salmon_tasks:
            prepared_salmon[task] = _load_salmon_task(
                args.salmon_dataset, args.salmon_split, task, args.salmon_reference_dir
            )

    print(
        f"Validated tasks={args.tasks}; TTS={len(tts_ids)}, "
        f"continuation={len(valid_cont_ids)}, SALMON={sum(len(x[0]) for x in prepared_salmon.values())}",
        flush=True,
    )
    if args.validate_only:
        return

    from stable_codec import StableCodec

    device = torch.device(args.device)
    codec = StableCodec(pretrained_model=args.speech_tokenizer, device=device).eval()
    codec.set_posthoc_bottleneck(args.speech_bottleneck)

    evaluator: TextAudioEvaluator | None = None
    reconstructed_by_row: dict[int, np.ndarray] = {}
    if tasks & {"tts", "cont_taste"}:
        evaluator = TextAudioEvaluator(
            sr=args.sample_rate,
            speaker_checkpoint=args.speaker_checkpoint,
            speaker_ref_path=str(reference_dir / args.tts_reference_embeddings),
            continuation_speaker_ref_path=str(reference_dir / args.cont_reference_embeddings),
            wlm_statistics_path=str(reference_dir / args.wlm_statistics),
            e2v_model=args.e2v_model,
            e2v_statistics_path=str(reference_dir / args.e2v_statistics),
            _dbg_func=lambda message: print(f"[Tokenizer] {message}", flush=True),
        )
        rows_to_reconstruct = (
            list(range(len(tts_ids))) if "tts" in tasks else cont_rows
        )
        reconstructed = _roundtrip_wavs(
            [source_wavs[row] for row in rows_to_reconstruct], codec, device,
            "TTS/Continuation",
        )
        reconstructed_by_row = dict(zip(rows_to_reconstruct, reconstructed))

    if "tts" in tasks:
        assert evaluator is not None
        tts_wavs = [reconstructed_by_row[row] for row in range(len(tts_ids))]
        metrics = {
            "UTMOS": float(evaluator.utmos_score(tts_wavs)),
            "SpkSim": float(evaluator.spksim(tts_wavs, device)),
        }
        _merge_result(args.output_dir / "results.json", "tts", metrics)
        _write_json(
            args.output_dir / "tts" / "samples.json",
            [{"sample_id": sample_id} for sample_id in tts_ids],
        )
        print(f"TTS tokenizer reconstruction: {metrics}", flush=True)

    if "cont_taste" in tasks:
        assert evaluator is not None
        cont_wavs = [reconstructed_by_row[row] for row in cont_rows]
        trimmed = [
            _trim_prefix_seconds(
                reconstructed_by_row[cont_rows[archive_row]],
                CONT_TRIM_SECONDS, args.sample_rate,
            )
            for archive_row in valid_cont_archive_rows
        ]
        if any(wav is None for wav in trimmed):
            raise ValueError("A reconstructed continuation ends within its prompt")
        fsd_dir = args.output_dir / "cont_taste" / "fsd_features"
        metrics = {
            "UTMOS": float(evaluator.utmos_score(cont_wavs)),
            "SpkSim": float(evaluator.spksim(
                trimmed, device, continuation=True,
                reference_indices=valid_cont_archive_rows,
            )),
            "FSD-wlm": float(evaluator.fsd(
                cont_wavs, device, args.fsd_batch_size,
                save_dir=fsd_dir, sample_keys=cont_ids, sample_key_name="sample_ids",
            )),
            "FSD-e2v": float(evaluator.fsd_e2v(
                cont_wavs, device, args.fsd_batch_size,
                save_dir=fsd_dir, sample_keys=cont_ids, sample_key_name="sample_ids",
            )),
        }
        _merge_result(args.output_dir / "results.json", "cont_taste", metrics)
        _write_json(
            args.output_dir / "cont_taste" / "samples.json",
            [
                {"sample_id": sample_id, "spksim_eligible": row in valid_cont_archive_rows}
                for row, sample_id in enumerate(cont_ids)
            ],
        )
        print(f"Continuation tokenizer reconstruction: {metrics}", flush=True)

    if "cont_salmon" in tasks:
        from scripts.text_audio.cache_utils import decode_wav
        from utils.judge_models import JudgeModel

        judges: dict[str, Any] = {}
        salmon_metrics: dict[str, float] = {}
        for task in args.salmon_tasks:
            dataset, positive, negative = prepared_salmon.pop(task)
            source = [
                decode_wav(audio["bytes"], target_sr=args.sample_rate)
                for audio in dataset["continuation_audio_positive"]
            ]
            if any(wav.size == 0 for wav in source):
                raise ValueError(f"SALMON/{task} contains an empty positive continuation")
            reconstructed = _roundtrip_wavs(source, codec, device, f"SALMON/{task}")
            judge_name = SALMON_JUDGE_PER_TASK[task]
            if judge_name not in judges:
                judges[judge_name] = JudgeModel(
                    judge_name, args.sample_rate, device, args.judge_max_length
                )
            judge = judges[judge_name]
            generated = judge.embed_batch([torch.from_numpy(wav) for wav in reconstructed])
            accuracy = float(judge.score(
                generated,
                torch.from_numpy(positive).to(device),
                torch.from_numpy(negative).to(device),
            ))
            salmon_metrics[task] = accuracy
            _write_json(
                args.output_dir / "cont_salmon" / task / "samples.json",
                [{"sample_id": int(index)} for index in dataset["ind"]],
            )
            print(f"SALMON tokenizer reconstruction/{task}: {accuracy:.6f}", flush=True)
            del dataset, source, reconstructed, generated
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _merge_result(args.output_dir / "results.json", "cont_salmon", salmon_metrics)


if __name__ == "__main__":
    main()
