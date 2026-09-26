#!/usr/bin/env python3
"""Staged unconditional generation with a text LM, speaker LM, and Spark-TTS."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .transformer import SpeakerTokenTransformer
except ImportError:  # direct script execution
    from transformer import SpeakerTokenTransformer


CASCADE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
ALL_STAGES = ("text", "speaker", "tts")


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def probability(value: str) -> float:
    result = float(value)
    if not 0.0 < result <= 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=positive_int, required=True)
    parser.add_argument(
        "--stages", nargs="+", choices=ALL_STAGES, default=list(ALL_STAGES),
        help="Stages to run, in canonical text -> speaker -> tts order",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--local-files-only", action="store_true")

    text = parser.add_argument_group("text generation")
    text.add_argument("--text-model", default="HuggingFaceTB/SmolLM2-135M")
    text.add_argument(
        "--text-prefix", default="",
        help="Optional prefix; leave empty for BOS-only unconditional generation",
    )
    text.add_argument("--text-batch-size", type=positive_int, default=64)
    text.add_argument(
        "--max-text-tokens", type=positive_int, default=64,
        help="Maximum number of newly generated text-model tokens per sample",
    )
    text.add_argument("--min-text-tokens", type=nonnegative_int, default=8)
    text.add_argument("--text-temperature", type=positive_float, default=1.0)
    text.add_argument("--text-top-p", type=probability, default=0.95)
    text.add_argument(
        "--text-top-k", type=nonnegative_int, default=0,
        help="Zero disables top-k truncation",
    )
    text.add_argument("--text-repetition-penalty", type=positive_float, default=1.0)
    text.add_argument("--text-retries", type=nonnegative_int, default=3)

    speaker = parser.add_argument_group("speaker-token generation")
    speaker.add_argument(
        "--speaker-checkpoint", type=Path,
        default=CASCADE_ROOT / "runs/speaker_transformer/last.pt",
    )
    speaker.add_argument("--speaker-batch-size", type=positive_int, default=4096)
    speaker.add_argument("--speaker-temperature", type=positive_float, default=1.0)
    speaker.add_argument(
        "--speaker-top-k", type=nonnegative_int, default=100,
        help="Zero disables top-k truncation",
    )

    speech = parser.add_argument_group("Spark-TTS generation")
    speech.add_argument("--spark-repo", type=Path, default=REPO_ROOT / "Spark-TTS")
    speech.add_argument(
        "--spark-model-dir", type=Path,
        default=REPO_ROOT / "Spark-TTS/pretrained_models/SparkTTS-0.5B",
    )
    speech.add_argument(
        "--max-speech-tokens", type=positive_int, default=500,
        help="Maximum BiCodec semantic IDs decoded for each waveform",
    )
    speech.add_argument("--speech-temperature", type=positive_float, default=0.8)
    speech.add_argument("--speech-top-k", type=positive_int, default=50)
    speech.add_argument("--speech-top-p", type=probability, default=0.95)
    speech.add_argument(
        "--speech-retries", type=nonnegative_int, default=2,
        help="Retries when Spark emits no semantic IDs",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def unload(*objects) -> None:
    for value in objects:
        del value
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=str)
        stream.write("\n")
    os.replace(temporary, path)


def load_text_records(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records[int(record["index"])] = record
            except (ValueError, KeyError) as error:
                raise ValueError(f"Invalid {path}:{line_number}: {error}") from error
    return records


def append_records(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def model_inputs(tokenizer, count: int, prefix: str, device: torch.device):
    if prefix:
        return tokenizer(
            [prefix] * count, return_tensors="pt", padding=True,
            add_special_tokens=True,
        ).to(device)
    bos = tokenizer.bos_token_id
    if bos is None:
        raise ValueError("Text tokenizer has no BOS token for unconditional generation")
    return {
        "input_ids": torch.full((count, 1), bos, dtype=torch.long, device=device),
        "attention_mask": torch.ones((count, 1), dtype=torch.long, device=device),
    }


def text_batch(model, tokenizer, indices, args, device, attempt=0):
    batch_seed = args.seed + indices[0] + attempt * 1_000_003
    seed_everything(batch_seed)
    inputs = model_inputs(tokenizer, len(indices), args.text_prefix, device)
    input_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=args.max_text_tokens,
            min_new_tokens=min(args.min_text_tokens, args.max_text_tokens),
            temperature=args.text_temperature,
            top_p=args.text_top_p,
            top_k=args.text_top_k,
            repetition_penalty=args.text_repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    records = []
    for row, index in zip(outputs, indices):
        token_ids = row[input_length:].detach().cpu().tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        records.append({
            "index": index,
            "sample_id": f"sample_{index:05d}",
            "seed": args.seed + index,
            "batch_seed": batch_seed,
            "text": text,
            "text_token_ids": token_ids,
            "text_token_count": len(token_ids),
        })
    return records


def generate_texts(args, device: torch.device, records: dict[int, dict], path: Path):
    pending = [
        index for index in range(args.num_samples)
        if args.overwrite or index not in records
    ]
    if not pending:
        print("Text stage: all samples already present")
        return records

    if args.overwrite:
        records = {}
        path.unlink(missing_ok=True)
    print(f"Loading text model: {args.text_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.text_model, local_files_only=args.local_files_only
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.text_model,
        torch_dtype=torch_dtype(args.dtype),
        local_files_only=args.local_files_only,
    ).eval().to(device)

    progress = tqdm(range(0, len(pending), args.text_batch_size),
                    desc="Text LM", unit="batch", dynamic_ncols=True)
    for offset in progress:
        indices = pending[offset:offset + args.text_batch_size]
        batch_records = text_batch(model, tokenizer, indices, args, device)
        for attempt in range(1, args.text_retries + 1):
            empty = [record["index"] for record in batch_records if not record["text"]]
            if not empty:
                break
            replacements = text_batch(model, tokenizer, empty, args, device, attempt)
            replacement_map = {record["index"]: record for record in replacements}
            batch_records = [
                replacement_map.get(record["index"], record) for record in batch_records
            ]
        empty = [record["index"] for record in batch_records if not record["text"]]
        if empty:
            raise RuntimeError(f"Text LM produced empty text after retries: {empty[:10]}")
        append_records(path, batch_records)
        records.update({record["index"]: record for record in batch_records})

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def load_speaker_tokens(path: Path, count: int) -> torch.Tensor | None:
    if not path.is_file():
        return None
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        value = value["speaker_token_ids"]
    value = torch.as_tensor(value).long()
    if value.shape != (count, 32):
        raise ValueError(f"{path} has shape {tuple(value.shape)}, expected ({count}, 32)")
    if value.min() < 0 or value.max() >= 4096:
        raise ValueError(f"{path} contains speaker IDs outside [0, 4095]")
    return value


def generate_speakers(args, device: torch.device, path: Path) -> torch.Tensor:
    existing = None if args.overwrite else load_speaker_tokens(path, args.num_samples)
    if existing is not None:
        print("Speaker stage: all samples already present")
        return existing

    checkpoint = torch.load(
        args.speaker_checkpoint, map_location="cpu", weights_only=False
    )
    model = SpeakerTokenTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    batches = []
    progress = tqdm(range(0, args.num_samples, args.speaker_batch_size),
                    desc="Speaker tokens", unit="batch", dynamic_ncols=True)
    for first in progress:
        size = min(args.speaker_batch_size, args.num_samples - first)
        seed_everything(args.seed + 10_000_019 + first)
        batches.append(model.sample(
            batch_size=size,
            temperature=args.speaker_temperature,
            top_k=args.speaker_top_k or None,
            device=device,
        ).cpu())
    result = torch.cat(batches)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "speaker_token_ids": result,
        "checkpoint": str(args.speaker_checkpoint),
        "temperature": args.speaker_temperature,
        "top_k": args.speaker_top_k,
        "seed": args.seed,
    }, temporary)
    os.replace(temporary, path)
    del model
    del checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def spark_prompt(text: str, global_ids: torch.Tensor) -> str:
    global_tokens = "".join(
        f"<|bicodec_global_{int(token)}|>" for token in global_ids
    )
    return "".join((
        "<|task_tts|>",
        "<|start_content|>", text, "<|end_content|>",
        "<|start_global_token|>", global_tokens, "<|end_global_token|>",
    ))


def parse_semantic_ids(tokenizer, generated_ids, maximum: int) -> torch.Tensor:
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    values = [
        int(value)
        for value in re.findall(r"bicodec_semantic_(\d+)", decoded)
    ]
    return torch.tensor(values[:maximum], dtype=torch.long).unsqueeze(0)


def write_manifest(output_dir: Path, count: int) -> None:
    rows = []
    for index in range(count):
        metadata_path = output_dir / f"sample_{index:05d}.json"
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as stream:
                rows.append(json.load(stream))
    fields = (
        "sample_id", "output_path", "seed", "text_token_count",
        "speech_token_count", "inference_seconds", "output_seconds",
        "generated_text",
    )
    temporary = output_dir / "samples.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_dir / "samples.csv")


def synthesize(args, device: torch.device, texts, speakers) -> None:
    missing_texts = [index for index in range(args.num_samples) if index not in texts]
    if missing_texts:
        raise RuntimeError(f"Missing generated texts: {missing_texts[:10]}")
    if speakers is None:
        raise RuntimeError("Speaker tokens are required for the TTS stage")

    sys.path.insert(0, str(args.spark_repo.resolve()))
    from sparktts.models.bicodec import BiCodec
    from sparktts.utils.file import load_config

    config = load_config(str(args.spark_model_dir / "config.yaml"))
    sample_rate = int(config["sample_rate"])
    print(f"Loading Spark LLM: {args.spark_model_dir / 'LLM'}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.spark_model_dir / "LLM",
        local_files_only=args.local_files_only,
    )
    spark = AutoModelForCausalLM.from_pretrained(
        args.spark_model_dir / "LLM",
        torch_dtype=torch_dtype(args.dtype),
        local_files_only=args.local_files_only,
    ).eval().to(device)
    print(f"Loading BiCodec: {args.spark_model_dir / 'BiCodec'}")
    bicodec = BiCodec.load_from_checkpoint(
        args.spark_model_dir / "BiCodec"
    ).eval().to(device)

    pending = [
        index for index in range(args.num_samples)
        if args.overwrite
        or not (args.output_dir / f"sample_{index:05d}.wav").is_file()
        or not (args.output_dir / f"sample_{index:05d}.json").is_file()
    ]
    progress = tqdm(pending, desc="Spark-TTS", unit="sample", dynamic_ncols=True)
    for index in progress:
        sample_id = f"sample_{index:05d}"
        text = texts[index]["text"]
        global_ids = speakers[index]
        prompt = spark_prompt(text, global_ids)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        semantic_ids = None
        started = time.perf_counter()
        for attempt in range(args.speech_retries + 1):
            sample_seed = args.seed + 20_000_033 + index + attempt * 1_000_003
            seed_everything(sample_seed)
            with torch.inference_mode():
                output = spark.generate(
                    **inputs,
                    do_sample=True,
                    max_new_tokens=args.max_speech_tokens + 16,
                    temperature=args.speech_temperature,
                    top_k=args.speech_top_k,
                    top_p=args.speech_top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = output[0, inputs.input_ids.shape[1]:]
            semantic_ids = parse_semantic_ids(
                tokenizer, generated, args.max_speech_tokens
            )
            if semantic_ids.numel():
                break
        if semantic_ids is None or not semantic_ids.numel():
            raise RuntimeError(f"Spark produced no semantic IDs for {sample_id}")

        with torch.inference_mode():
            waveform = bicodec.detokenize(
                semantic_ids.to(device),
                global_ids.view(1, 1, -1).to(device),
            )
        synchronize(device)
        elapsed = time.perf_counter() - started
        waveform = waveform.detach().float().squeeze().cpu().numpy()
        if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
            raise RuntimeError(f"BiCodec returned invalid audio for {sample_id}")
        wav_path = args.output_dir / f"{sample_id}.wav"
        temporary_wav = wav_path.with_suffix(".wav.tmp")
        sf.write(temporary_wav, waveform, sample_rate, format="WAV")
        os.replace(temporary_wav, wav_path)

        record = {
            "sample_id": sample_id,
            "index": index,
            "output_path": wav_path.name,
            "seed": args.seed + index,
            "generated_text": text,
            "text_token_count": texts[index]["text_token_count"],
            "global_token_ids": global_ids.tolist(),
            "speech_token_count": semantic_ids.numel(),
            "inference_seconds": round(elapsed, 6),
            "output_seconds": round(waveform.size / sample_rate, 6),
            "sample_rate": sample_rate,
        }
        atomic_json(args.output_dir / f"{sample_id}.json", record)
        progress.set_postfix(
            speech_tokens=semantic_ids.numel(),
            audio_s=f"{waveform.size / sample_rate:.1f}",
            last_s=f"{elapsed:.1f}",
            refresh=False,
        )
    write_manifest(args.output_dir, args.num_samples)
    del spark
    del tokenizer
    del bicodec
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_args(args) -> None:
    if args.min_text_tokens > args.max_text_tokens:
        raise ValueError("--min-text-tokens cannot exceed --max-text-tokens")
    requested = set(args.stages)
    if requested != set(ALL_STAGES):
        # Canonical ordering is applied regardless of CLI ordering.
        args.stages = [stage for stage in ALL_STAGES if stage in requested]


def main() -> None:
    args = arguments()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)
    atomic_json(
        args.output_dir / "config.json",
        {key: str(value) if isinstance(value, Path) else value
         for key, value in vars(args).items()},
    )

    text_path = args.output_dir / "texts.jsonl"
    texts = load_text_records(text_path)
    if "text" in args.stages:
        texts = generate_texts(args, device, texts, text_path)

    speaker_path = args.output_dir / "speaker_tokens.pt"
    speakers = load_speaker_tokens(speaker_path, args.num_samples)
    if "speaker" in args.stages:
        speakers = generate_speakers(args, device, speaker_path)

    if "tts" in args.stages:
        if not texts:
            texts = load_text_records(text_path)
        if speakers is None:
            speakers = load_speaker_tokens(speaker_path, args.num_samples)
        synthesize(args, device, texts, speakers)

    print(f"Completed stages: {', '.join(args.stages)}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
