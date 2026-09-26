from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = REPO_ROOT / "baselines" / "text_audio"
DEFAULT_WORK_DIR = BASELINE_ROOT / "outputs"
DEFAULT_PROMPT_DIR = DEFAULT_WORK_DIR / "prompts" / "cont_taste"
DEFAULT_RUN_DIR = DEFAULT_WORK_DIR / "evaluation"
MODEL_SR = 16_000
CONT_PROMPT_SECONDS = 3.0

BASELINE_TASKS: dict[str, tuple[str, ...]] = {
    "cascade": ("joint",),
    "taste": ("joint", "cont_taste"),
    "spiritlm": ("cont_taste",),
    "llama_mimi": ("cont_taste",),
    "gpa": ("tts", "stt"),
}

BASELINE_SUBDIRS = {
    "taste": "TASTE-SpokenLM",
    "spiritlm": "spiritlm",
    "llama_mimi": "llama-mimi",
    "cascade": "cascade",
    "gpa": "GPA",
}


@dataclass(frozen=True)
class BaselineLayout:
    baseline: str
    run_dir: Path

    @property
    def tag_dir(self) -> Path:
        return self.run_dir / self.baseline

    def task_dir(self, task: str) -> Path:
        return self.tag_dir / task


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def probability(value: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return result


def canonical_tasks(baseline: str, tasks: Sequence[str] | None) -> list[str]:
    allowed = BASELINE_TASKS[baseline]
    if not tasks:
        return list(allowed)
    aliases = {"cont": "cont_taste"}
    result = []
    for task in tasks:
        task = aliases.get(task, task)
        if task not in allowed:
            raise ValueError(
                f"{baseline!r} does not support task {task!r}; "
                f"available tasks: {', '.join(allowed)}"
            )
        if task not in result:
            result.append(task)
    return result


@contextmanager
def sys_path_prepend(path: Path):
    path = str(path.resolve())
    old_path = list(sys.path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = old_path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device | str | None = None) -> None:
    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.synchronize()
        return
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_python(script: Path, args: Sequence[Any]) -> None:
    command = [sys.executable, str(script), *[str(x) for x in args]]
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def write_manifest(run_dir: Path, baseline: str, tasks: Sequence[str], sample_rate: int = MODEL_SR) -> None:
    header = {
        "baseline": baseline,
        "tasks": list(tasks),
        "sample_rate": int(sample_rate),
        "stt_partitions": ["clean", "other"],
    }
    atomic_json(run_dir / baseline / "manifest.json", {"header": header})


def _read_meta(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _decode_source_audio(example: dict[str, Any]) -> np.ndarray:
    from scripts.text_audio.cache_utils import decode_wav

    audio = example.get("audio")
    if isinstance(audio, dict) and "array" in audio and audio["array"] is not None:
        wav = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio.get("sampling_rate", MODEL_SR))
        if sr != MODEL_SR:
            import torchaudio
            wav_t = torch.from_numpy(wav).float().unsqueeze(0)
            wav = torchaudio.functional.resample(wav_t, sr, MODEL_SR).squeeze(0).numpy()
        return wav.astype(np.float32, copy=False)
    if isinstance(audio, dict) and "bytes" in audio:
        return decode_wav(audio["bytes"], target_sr=MODEL_SR).astype(np.float32, copy=False)
    raise ValueError("Dataset example has no decodable audio field")




def _dataset_rows_by_id(meta: dict[str, Any], ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    from datasets import Audio, load_dataset

    wanted = set(map(str, ids))
    dataset = load_dataset(meta["hf_path"], meta.get("hf_config"), split=meta.get("hf_split", "test"))
    dataset = dataset.cast_column("audio", Audio(decode=False))
    rows: dict[str, dict[str, Any]] = {}
    for row in dataset:
        sample_id = str(row["id"])
        if sample_id in wanted:
            rows[sample_id] = row
            if len(rows) == len(wanted):
                break
    missing = [sample_id for sample_id in ids if sample_id not in rows]
    if missing:
        raise RuntimeError(f"Source dataset is missing ids: {missing[:5]}")
    return rows


def _write_npz_wavs(wavs: Sequence[np.ndarray], ids: Sequence[str], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rels = []
    for sample_id, wav in zip(ids, wavs):
        rel = f"{sample_id}.wav"
        sf.write(out_dir / rel, np.asarray(wav, dtype=np.float32), MODEL_SR)
        rels.append(rel)
    return rels


def prepare_gpa_manifests(
    *,
    output_dir: Path,
    tasks: Sequence[str],
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Create GPA ASR/TTS CSV manifests from the prepared test_common caches."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, Path] = {}

    if "tts" in tasks:
        manifest = output_dir / "tts.csv"
        if overwrite or not manifest.exists():
            meta = _read_meta(REPO_ROOT / "datasets/test_common/cache_test_clean_tts_632.meta.json")
            ids = list(map(str, meta["ids"]))
            if limit is not None:
                ids = ids[:limit]
            rows_by_id = _dataset_rows_by_id(meta, ids)
            refs = np.load(REPO_ROOT / "datasets/test_common/cache_test_clean_tts.ref_audio.npz", allow_pickle=True)
            ref_wavs = list(refs["wavs"][:len(ids)])
            ref_rels = _write_npz_wavs(ref_wavs, ids, output_dir / "tts_refs")
            heldout = {}
            heldout_path = REPO_ROOT / "datasets/test_common/heldout.csv"
            if heldout_path.exists():
                with heldout_path.open(newline="", encoding="utf-8") as stream:
                    reader = csv.reader(stream)
                    for row in reader:
                        if len(row) >= 2 and row[0] != "speaker_id":
                            heldout[str(row[0])] = str(row[1])
            out_rows = []
            for sample_id, ref_rel in zip(ids, ref_rels):
                src = rows_by_id[sample_id]
                speaker_id = str(src.get("speaker_id", "unknown"))
                out_rows.append({
                    "sample_id": sample_id,
                    "audio_path": f"tts_refs/{ref_rel}",
                    "text": str(src.get("text") or src.get("text_normalized") or "").strip(),
                    "duration": f"{len(ref_wavs[len(out_rows)]) / MODEL_SR:.6f}",
                    "speaker_id": speaker_id,
                    "reference_id": heldout.get(speaker_id, sample_id),
                    "reference_path": f"tts_refs/{ref_rel}",
                })
            _write_rows(manifest, ["sample_id", "audio_path", "text", "duration", "speaker_id", "reference_id", "reference_path"], out_rows)
        manifests["tts"] = manifest

    if "stt" in tasks:
        for partition in ("clean", "other"):
            manifest = output_dir / f"stt_{partition}.csv"
            if overwrite or not manifest.exists():
                meta = _read_meta(REPO_ROOT / f"datasets/test_common/cache_test_{partition}_asr_632.meta.json")
                ids = list(map(str, meta["ids"]))
                if limit is not None:
                    ids = ids[:limit]
                rows_by_id = _dataset_rows_by_id(meta, ids)
                refs = np.load(REPO_ROOT / f"datasets/test_common/cache_test_{partition}_asr.ref_audio.npz", allow_pickle=True)
                wavs = list(refs["wavs"][:len(ids)])
                rels = _write_npz_wavs(wavs, ids, output_dir / f"stt_{partition}_audio")
                out_rows = []
                for sample_id, rel, wav in zip(ids, rels, wavs):
                    src = rows_by_id[sample_id]
                    out_rows.append({
                        "sample_id": sample_id,
                        "audio_path": f"stt_{partition}_audio/{rel}",
                        "text": str(src.get("text") or src.get("text_normalized") or "").strip(),
                        "duration": f"{len(wav) / MODEL_SR:.6f}",
                    })
                _write_rows(manifest, ["sample_id", "audio_path", "text", "duration"], out_rows)
            manifests[f"stt_{partition}"] = manifest
    return manifests

def prepare_continuation_prompts(
    *,
    output_dir: Path = DEFAULT_PROMPT_DIR,
    cache_meta: Path = REPO_ROOT / "datasets/test_common/cache_test_cont.meta.json",
    limit: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Write prompt WAVs plus a FlowSLM-style ``prompts.csv`` manifest."""
    from datasets import Audio, load_dataset

    output_dir = Path(output_dir)
    wav_dir = output_dir / "wavs"
    csv_path = output_dir / "prompts.csv"
    if csv_path.exists() and not overwrite:
        return csv_path

    meta = _read_meta(Path(cache_meta))
    ids = list(meta["ids"])
    if limit is not None:
        ids = ids[:limit]
    wanted = set(ids)
    by_id: dict[str, dict[str, Any]] = {}

    dataset = load_dataset(meta["hf_path"], meta.get("hf_config"), split=meta.get("hf_split", "test"))
    dataset = dataset.cast_column("audio", Audio(decode=False))
    for row in dataset:
        sample_id = str(row["id"])
        if sample_id in wanted:
            by_id[sample_id] = row
            if len(by_id) == len(wanted):
                break

    missing = [sample_id for sample_id in ids if sample_id not in by_id]
    if missing:
        raise RuntimeError(f"Source dataset is missing continuation ids: {missing[:5]}")

    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    trim_samples = round(float(meta.get("trim_seconds", CONT_PROMPT_SECONDS)) * MODEL_SR)
    for sample_id in ids:
        wav = _decode_source_audio(by_id[sample_id])[:trim_samples]
        if wav.size < trim_samples:
            raise ValueError(f"{sample_id} is shorter than the continuation prompt")
        rel = f"{sample_id}.wav"
        sf.write(wav_dir / rel, wav, MODEL_SR)
        rows.append({"path": rel, "prompt_length": f"{wav.size / MODEL_SR:.6f}"})

    _write_rows(csv_path, ["path", "prompt_length"], rows)
    return csv_path


def prepare_cascade_speaker_cache(
    *,
    output: Path = BASELINE_ROOT / "cascade/data/speaker_tokens_train.uint16",
    source: Path = REPO_ROOT / "datasets/textaudio/cache_textaudio_train.uint32",
    source_meta: Path = REPO_ROOT / "datasets/textaudio/cache_textaudio_train.meta.json",
    overwrite: bool = False,
) -> None:
    script = BASELINE_ROOT / "cascade/extract_speaker_cache.py"
    args: list[Any] = ["--source", source, "--source-meta", source_meta, "--output", output]
    if overwrite:
        args.append("--overwrite")
    run_python(script, args)


def train_cascade_speaker_transformer(extra_args: Sequence[str] = ()) -> None:
    run_python(BASELINE_ROOT / "cascade/transformer.py", list(extra_args))


def generate_cascade(
    *,
    output_dir: Path,
    num_samples: int,
    stages: Sequence[str] | None = None,
    overwrite: bool = False,
    extra_args: Sequence[str] = (),
) -> None:
    args: list[Any] = ["--output-dir", output_dir, "--num-samples", num_samples]
    if stages:
        args.extend(["--stages", *stages])
    if overwrite:
        args.append("--overwrite")
    args.extend(extra_args)
    run_python(BASELINE_ROOT / "cascade/generate_cascade.py", args)


def normalize_joint_samples(task_dir: Path) -> list[dict[str, Any]]:
    task_dir = Path(task_dir)
    samples = []
    csv_path = task_dir / "samples.csv"
    if csv_path.exists():
        for row in _load_rows(csv_path):
            gen_wav = row.get("output_path") or row.get("gen_wav")
            text = row.get("generated_text") or row.get("gen_text") or ""
            if gen_wav:
                samples.append({
                    "sample_id": row.get("sample_id") or Path(gen_wav).stem,
                    "gen_wav": gen_wav,
                    "gen_text": text,
                })
    else:
        for path in sorted(task_dir.glob("sample_*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            gen_wav = row.get("output_path") or path.with_suffix(".wav").name
            samples.append({
                "sample_id": row.get("sample_id", path.stem),
                "gen_wav": gen_wav,
                "gen_text": row.get("generated_text") or row.get("gen_text") or "",
            })
    atomic_json(task_dir / "samples.json", samples)
    return samples


def normalize_continuation_samples(task_dir: Path) -> list[dict[str, Any]]:
    task_dir = Path(task_dir)
    rows = _load_rows(task_dir / "samples.csv")
    samples = []
    for row in rows:
        gen_wav = row.get("output_path") or row.get("gen_wav")
        if not gen_wav:
            continue
        samples.append({
            "sample_id": row.get("sample_id") or Path(gen_wav).stem,
            "gen_wav": gen_wav,
        })
    atomic_json(task_dir / "samples.json", samples)
    return samples


def normalize_task_outputs(task_dir: Path, task: str) -> list[dict[str, Any]]:
    if task == "joint":
        return normalize_joint_samples(task_dir)
    if task == "cont_taste":
        return normalize_continuation_samples(task_dir)
    if task in ("tts", "tts_gt"):
        return normalize_tts_samples(task_dir)
    if task in ("stt", "stt_clean", "stt_other"):
        return normalize_stt_samples(task_dir)
    raise ValueError(f"No normalizer implemented for task {task!r}")


def _sample_random_taste_speaker(batch_size: int, dim: int, device: torch.device, generator=None) -> torch.Tensor:
    import torch.nn.functional as F

    embedding = torch.randn(batch_size, dim, generator=generator, device=device, dtype=torch.float32)
    return F.normalize(embedding, p=2, dim=-1)


def _load_prompt_manifest(prompt_csv: Path, limit: int | None = None) -> list[dict[str, str]]:
    rows = _load_rows(prompt_csv)
    if not rows:
        raise ValueError(f"No prompts found in {prompt_csv}")
    missing = {"path", "prompt_length"}.difference(rows[0])
    if missing:
        raise ValueError(f"Missing prompt manifest columns: {', '.join(sorted(missing))}")
    return rows if limit is None else rows[:limit]


def generate_taste_joint(
    *,
    output_dir: Path,
    num_samples: int,
    seed: int = 0,
    model_id: str = "MediaTek-Research/Llama-1B-TASTE-V0",
    extra_words: int = 32,
    max_generation_steps: int = 512,
    max_decoder_context_tokens: int = 4096,
    text_top_p: float = 0.3,
    taste_top_p: float = 0.0,
    text_temperature: float = 0.5,
    repetition_penalty: float = 1.1,
    attn_implementation: str = "eager",
    resume: bool = False,
) -> None:
    import torchaudio
    from tqdm import tqdm

    taste_root = BASELINE_ROOT / "TASTE-SpokenLM"
    with sys_path_prepend(taste_root):
        from taste_speech import TasteForCausalLM, TasteProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("TASTE inference requires a CUDA GPU")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    model = TasteForCausalLM.from_pretrained(model_id, attn_implementation=attn_implementation).to(device).eval()
    processor = TasteProcessor.from_pretrained(model_id)
    generator = processor.get_generator(device=device).eval()

    kwargs = {
        "llm_tokenizer": processor.llm_tokenizer,
        "asr_tokenizer": processor.audio_tokenizer,
        "extra_words": extra_words,
        "text_top_p": text_top_p,
        "taste_top_p": taste_top_p,
        "text_temperature": text_temperature,
        "repetition_penalty": repetition_penalty,
    }
    # Clean submodule compatibility: pass newer safety knobs only when present.
    import inspect
    sig = inspect.signature(model.inference_completion)
    if "max_generation_steps" in sig.parameters:
        kwargs["max_generation_steps"] = max_generation_steps
    if "max_decoder_context_tokens" in sig.parameters:
        kwargs["max_decoder_context_tokens"] = max_decoder_context_tokens

    csv_path = output_dir / "samples.csv"
    fieldnames = ["sample_id", "output_path", "seed", "inference_seconds", "output_seconds", "generated_text"]
    completed = set()
    if resume:
        completed = {i for i in range(num_samples) if (output_dir / f"sample_{i:05d}.wav").exists()}
    append = resume and csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a" if append else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for index in tqdm([i for i in range(num_samples) if i not in completed], desc="TASTE joint", unit="sample"):
            sample_id = f"sample_{index:05d}"
            sample_seed = seed + index
            seed_everything(sample_seed)
            speaker_generator = torch.Generator(device=device).manual_seed(sample_seed)
            speaker_embedding = _sample_random_taste_speaker(1, 192, device, speaker_generator)
            synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                out = model.inference_completion(speaker_embeds=speaker_embedding, conditional_mode="zero", **kwargs)
                speech, sr = generator.inference(
                    speech_token_ids=out["speech_token_ids"],
                    speech_token_lengths=out["speech_token_lengths"],
                    flow_embedding=speaker_embedding,
                )
            synchronize(device)
            elapsed = time.perf_counter() - started
            wav_path = output_dir / f"{sample_id}.wav"
            torchaudio.save(str(wav_path), speech.float().cpu(), sr)
            row = {
                "sample_id": sample_id,
                "output_path": wav_path.name,
                "seed": sample_seed,
                "inference_seconds": f"{elapsed:.6f}",
                "output_seconds": f"{speech.shape[-1] / sr:.6f}",
                "generated_text": out.get("generated_text", ""),
            }
            writer.writerow(row)
            stream.flush()
            atomic_json(output_dir / f"{sample_id}.json", row)
    normalize_joint_samples(output_dir)


def generate_taste_continuations(
    *,
    prompt_dir: Path,
    prompt_csv: Path,
    output_dir: Path,
    num_samples: int | None = None,
    seed: int = 0,
    model_id: str = "MediaTek-Research/Llama-1B-TASTE-V0",
    extra_words: int = 8,
    asr_max_new_tokens: int = 128,
    generated_part_only: bool = False,
    max_continuation_seconds: float | None = None,
    text_top_p: float = 0.3,
    taste_top_p: float = 0.0,
    text_temperature: float = 0.5,
    repetition_penalty: float = 1.1,
    attn_implementation: str = "eager",
    resume: bool = False,
) -> None:
    import inspect
    import torchaudio
    from tqdm import tqdm

    taste_root = BASELINE_ROOT / "TASTE-SpokenLM"
    with sys_path_prepend(taste_root):
        from taste_speech import TasteForCausalLM, TasteProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("TASTE continuation inference requires a CUDA GPU")
    rows = _load_prompt_manifest(prompt_csv, num_samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    model = TasteForCausalLM.from_pretrained(model_id, attn_implementation=attn_implementation).to(device).eval()
    processor = TasteProcessor.from_pretrained(model_id)
    generator = processor.get_generator(device=device).eval()
    whisper_context = int(model.config.asr_config.max_target_positions)
    whisper_special_tokens = 5 if model.audio_tower.add_eos else 4
    max_prompt_asr_tokens = whisper_context - whisper_special_tokens

    input_keys = (
        "speaker_embeds", "audio_features", "audio_feature_lengths",
        "asr_token_ids", "asr_token_lengths", "asr_word_ids",
        "llm_token_ids", "llm_token_lengths", "llm_word_ids",
    )
    kwargs = {
        "llm_tokenizer": processor.llm_tokenizer,
        "asr_tokenizer": processor.audio_tokenizer,
        "extra_words": extra_words,
        "text_top_p": text_top_p,
        "taste_top_p": taste_top_p,
        "text_temperature": text_temperature,
        "repetition_penalty": repetition_penalty,
        "out_generated_part_only": generated_part_only,
    }
    proc_sig = inspect.signature(processor.__call__)
    supports_asr_limit = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in proc_sig.parameters.values())

    csv_path = output_dir / "samples.csv"
    fieldnames = ["sample_id", "output_path", "inference_seconds", "output_seconds", "generated_text"]
    completed = set()
    if resume:
        completed = {Path(r["path"]).stem for r in rows if (output_dir / Path(r["path"]).with_suffix(".wav").name).exists()}
    append = resume and csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a" if append else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for index, row in tqdm([(i, r) for i, r in enumerate(rows) if Path(r["path"]).stem not in completed], desc="TASTE cont", unit="sample"):
            prompt_path = prompt_dir / row["path"]
            sample_id = prompt_path.stem
            seed_everything(seed + index)
            synchronize(device)
            started = time.perf_counter()
            call_kwargs = {"ref_audio_list": [str(prompt_path)]}
            if supports_asr_limit:
                call_kwargs["asr_max_new_tokens"] = asr_max_new_tokens
            processed = processor(str(prompt_path), MODEL_SR, **call_kwargs)
            original_len = int(processed["asr_token_lengths"][0])
            if original_len > max_prompt_asr_tokens:
                keep = processed["asr_word_ids"][0] <= int(processed["asr_word_ids"][0, max_prompt_asr_tokens]) - 1
                for prefix in ("asr", "llm"):
                    processed[f"{prefix}_word_ids"] = processed[f"{prefix}_word_ids"][:, keep]
                    processed[f"{prefix}_token_ids"] = processed[f"{prefix}_token_ids"][:, keep]
                    processed[f"{prefix}_token_lengths"] = torch.tensor([int(keep.sum())], dtype=processed[f"{prefix}_token_lengths"].dtype)
            inputs = {key: processed[key].to(device) for key in input_keys}
            with torch.inference_mode():
                out = model.inference_completion(**inputs, conditional_mode="audio", **kwargs)
                speech, sr = generator.inference(
                    speech_token_ids=out["speech_token_ids"],
                    speech_token_lengths=out["speech_token_lengths"],
                    flow_embedding=inputs["speaker_embeds"],
                )
            synchronize(device)
            elapsed = time.perf_counter() - started
            wav_path = output_dir / f"{sample_id}.wav"
            if max_continuation_seconds is not None:
                speech = speech[..., : round(max_continuation_seconds * sr)]
            torchaudio.save(str(wav_path), speech.float().cpu(), sr)
            writer.writerow({
                "sample_id": sample_id,
                "output_path": wav_path.name,
                "inference_seconds": f"{elapsed:.6f}",
                "output_seconds": f"{speech.shape[-1] / sr:.6f}",
                "generated_text": out.get("generated_text", ""),
            })
            stream.flush()
    normalize_continuation_samples(output_dir)


def _load_prompt_wav(path: Path, duration: float, sample_rate: int) -> torch.Tensor:
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    needed = round(duration * sample_rate)
    if wav.shape[-1] < needed:
        raise ValueError(f"{path} is too short for requested prompt length {duration:.3f}s")
    return wav[:, :needed]


def generate_spiritlm_continuations(
    *,
    prompt_dir: Path,
    prompt_csv: Path,
    output_dir: Path,
    model_id: str,
    num_samples: int | None = None,
    max_new_tokens: int = 200,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_continuation_seconds: float | None = None,
    speaker_id: int = 2,
    seed: int = 0,
    continuation_only: bool = False,
) -> None:
    import torchaudio
    from tqdm import tqdm
    from transformers import GenerationConfig, set_seed

    spirit_root = BASELINE_ROOT / "spiritlm"
    with sys_path_prepend(spirit_root):
        import spiritlm.model.spiritlm_model as sm
        from spiritlm.model.spiritlm_model import ContentType, GenerationInput, OutputModality, Spiritlm

    old_ensure = sm._ensure_model_name
    def _ensure_model_name_or_local(name: str):
        try:
            return old_ensure(name)
        except AssertionError:
            p = Path(name)
            return p.name if p.exists() else old_ensure(name)
    sm._ensure_model_name = _ensure_model_name_or_local

    rows = _load_prompt_manifest(prompt_csv, num_samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model = Spiritlm(model_id)
    cfg = GenerationConfig(temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, do_sample=True)
    csv_rows = []
    for index, row in enumerate(tqdm(rows, desc="SpiritLM cont", unit="sample")):
        prompt_path = prompt_dir / row["path"]
        prompt = _load_prompt_wav(prompt_path, float(row["prompt_length"]), MODEL_SR)[0]
        sample_id = prompt_path.stem
        synchronize(model.device)
        started = time.perf_counter()
        outputs = model.generate(
            output_modality=OutputModality.SPEECH,
            interleaved_inputs=[GenerationInput(content=prompt, content_type=ContentType.SPEECH)],
            generation_config=cfg,
            speaker_id=speaker_id,
        )
        synchronize(model.device)
        chunks = [torch.as_tensor(o.content).flatten() for o in outputs if o.content_type == ContentType.SPEECH]
        if not chunks:
            raise RuntimeError(f"SpiritLM generated no speech for {sample_id}")
        cont = torch.cat(chunks).float().cpu()
        if max_continuation_seconds is not None:
            cont = cont[: round(max_continuation_seconds * MODEL_SR)]
        out = cont if continuation_only else torch.cat([prompt.cpu(), cont])
        wav_path = output_dir / f"{sample_id}.wav"
        torchaudio.save(str(wav_path), out.unsqueeze(0), MODEL_SR)
        csv_rows.append({
            "sample_id": sample_id,
            "output_path": wav_path.name,
            "inference_seconds": f"{time.perf_counter() - started:.6f}",
            "continuation_seconds": f"{cont.numel() / MODEL_SR:.6f}",
        })
    _write_rows(output_dir / "samples.csv", csv_rows[0].keys() if csv_rows else ["sample_id", "output_path"], csv_rows)
    normalize_continuation_samples(output_dir)



def generate_llama_mimi_continuations(
    *,
    prompt_dir: Path,
    prompt_csv: Path,
    output_dir: Path,
    num_samples: int | None = None,
    model_id: str = "llm-jp/Llama-Mimi-1.3B",
    codec_model: str = "kyutai/mimi",
    max_length: int = 1024,
    temperature: float = 0.8,
    top_k: int = 30,
    max_continuation_seconds: float | None = None,
    seed: int = 0,
    continuation_only: bool = False,
    resume: bool = False,
) -> None:
    import torchaudio
    from tqdm import tqdm
    from transformers import AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer, MimiModel, StoppingCriteria

    class StopOnAudioEnd(StoppingCriteria):
        def __init__(self, tokenizer) -> None:
            self.target_ids = tokenizer("</audio>", add_special_tokens=False).input_ids
        def __call__(self, input_ids, scores, **kwargs) -> bool:
            n = len(self.target_ids)
            return input_ids.shape[-1] >= n and all(row == self.target_ids for row in input_ids[:, -n:].tolist())

    def codes_to_text(codes: torch.Tensor, nq: int) -> str:
        flat = codes.transpose(1, 2).reshape(-1).tolist()
        toks = []
        for off in range(0, len(flat), nq):
            toks.extend(f"<{int(flat[off+i])}_{i}>" for i in range(nq))
        return "<audio>" + "".join(toks)

    def text_to_codes(text: str, nq: int) -> torch.Tensor:
        matches = re.findall(r"<(\d+)_(\d+)>", text)
        frames = []
        for off in range(0, len(matches), nq):
            frame = matches[off:off+nq]
            if len(frame) < nq or [int(i) for _, i in frame] != list(range(nq)):
                break
            frames.append([int(v) for v, _ in frame])
        if not frames:
            return torch.empty((1, nq, 0), dtype=torch.long)
        return torch.tensor(frames, dtype=torch.long).T.unsqueeze(0)

    if not torch.cuda.is_available():
        raise RuntimeError("Llama-Mimi continuation inference requires a CUDA GPU")
    device = torch.device("cuda:0")
    rows = _load_prompt_manifest(prompt_csv, num_samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    nq = int(model.config.num_quantizers)
    codec = MimiModel.from_pretrained(codec_model).eval().to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(codec_model)
    sr = int(feature_extractor.sampling_rate)
    stop = StopOnAudioEnd(tokenizer)

    csv_rows = []
    completed = set()
    if resume:
        completed = {Path(r["path"]).stem for r in rows if (output_dir / Path(r["path"]).with_suffix(".wav").name).exists()}
    for index, row in enumerate(tqdm([r for r in rows if Path(r["path"]).stem not in completed], desc="Llama-Mimi cont", unit="sample")):
        prompt_path = prompt_dir / row["path"]
        sample_id = prompt_path.stem
        seed_everything(seed + index)
        prompt = _load_prompt_wav(prompt_path, float(row["prompt_length"]), sr)
        inputs_audio = feature_extractor(
            raw_audio=prompt.squeeze(0).cpu().numpy(), sampling_rate=sr, return_tensors="pt"
        ).to(device)
        synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            prompt_codes = codec.encode(inputs_audio["input_values"], inputs_audio.get("padding_mask"), num_quantizers=nq).audio_codes
        prompt_text = codes_to_text(prompt_codes, nq)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[-1]
        if input_len >= max_length:
            raise ValueError(f"{sample_id}: encoded prompt length {input_len} exceeds --max-length {max_length}")
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_length=max_length, do_sample=True,
                temperature=temperature, top_k=top_k, stopping_criteria=[stop],
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=False)
        valid = text_to_codes(text, nq)
        prompt_frames = prompt_codes.shape[-1]
        if valid.shape[-1] < prompt_frames:
            raise RuntimeError(f"{sample_id}: generated fewer valid frames than prompt")
        continuation = valid[..., prompt_frames:]
        if max_continuation_seconds is not None:
            continuation = continuation[..., : math.floor(max_continuation_seconds * float(codec.config.frame_rate))]
        output_codes = continuation if continuation_only else torch.cat([valid[..., :prompt_frames], continuation], dim=-1)
        if output_codes.shape[-1] == 0:
            wav = torch.empty((1, 0), dtype=torch.float32)
        else:
            with torch.inference_mode():
                wav = codec.decode(output_codes.to(device))[0][0].detach().float().cpu()
        synchronize(device)
        wav_path = output_dir / f"{sample_id}.wav"
        torchaudio.save(str(wav_path), wav, sr)
        csv_rows.append({
            "sample_id": sample_id,
            "output_path": wav_path.name,
            "inference_seconds": f"{time.perf_counter() - started:.6f}",
            "continuation_seconds": f"{wav.shape[-1] / sr:.6f}",
        })
    _write_rows(output_dir / "samples.csv", csv_rows[0].keys() if csv_rows else ["sample_id", "output_path"], csv_rows)
    normalize_continuation_samples(output_dir)


def _resolve_manifest_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve()


def _gpa_run_tts_with_optional_global(
    *,
    model,
    tts_processor,
    tokenizer,
    spark_tokenizer,
    spark_detokenizer,
    ref_audio_path: str,
    text: str,
    global_tokens=None,
    tts_stop_id: int = 151665,
    max_new_tokens: int = 4096,
    max_semantic_tokens: int = 1000,
    device: str = "cpu",
):
    from inference.decode_policies import strip_last_if_stop
    from inference.prompts import build_tts_conversation, extract_semantic_ids_from_text
    from inference.tts import build_tts_inputs

    if global_tokens is None:
        reference_output = spark_tokenizer.tokenize([ref_audio_path])
        global_tokens = reference_output["global_tokens"]
    else:
        global_tokens = global_tokens.to(device)
    global_token_ids = global_tokens[0, 0].detach().cpu().tolist()
    conversation = build_tts_conversation(text=text, global_token_ids=global_token_ids)
    inputs = build_tts_inputs(tts_processor, conversation)
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            inputs[key] = value.to(device)

    prompt_length = inputs["input_ids"].shape[1]
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        repetition_penalty=1.1,
        temperature=0.3,
    )
    generated_ids = strip_last_if_stop(generated[0, prompt_length:], tts_stop_id)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    semantic_ids = extract_semantic_ids_from_text(generated_text)
    if not semantic_ids:
        raise RuntimeError("GPA TTS generated no semantic speech tokens")
    semantic_ids = semantic_ids[:max_semantic_tokens]
    semantic_tokens = torch.tensor(semantic_ids, dtype=torch.long).unsqueeze(0)
    waveform = spark_detokenizer.detokenize(
        semantic_tokens=semantic_tokens,
        global_tokens=global_tokens,
    )
    if torch.is_tensor(waveform):
        waveform = waveform.detach().cpu().squeeze().float().numpy()
    return waveform, semantic_ids, generated_text


def generate_gpa(
    *,
    task: str,
    manifest: Path,
    output_dir: Path,
    model_path: str | None = None,
    audio_tokenizer_path: str | None = None,
    num_samples: int | None = None,
    seed: int = 0,
    resume: bool = False,
    device: str | None = None,
    attn_impl: str = "sdpa",
    sampling_rate: int = MODEL_SR,
    asr_max_audio_seconds: int = 30,
    asr_max_new_tokens: int = 256,
    tts_stop_id: int = 151665,
    tts_max_new_tokens: int = 4096,
    tts_max_semantic_tokens: int = 1000,
) -> list[dict[str, Any]]:
    """Run GPA-v1.5 ASR or TTS over a prepared manifest."""
    from tqdm import tqdm

    gpa_root = BASELINE_ROOT / "GPA" / "GPA_1.5"
    with sys_path_prepend(gpa_root):
        from inference.asr import run_asr
        from inference.assets import resolve_audio_tokenizer_dir, resolve_model_dir
        from inference.model_loader import default_device, load_spark_stack, load_text_stack

    device = device or default_device()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest).resolve()
    rows = _load_rows(manifest)
    if num_samples is not None:
        rows = rows[:num_samples]
    if not rows:
        raise ValueError(f"No rows found in {manifest}")

    model_dir = resolve_model_dir(model_path)
    tokenizer, processor, tts_processor, model, normalized_attn = load_text_stack(
        model_path=model_dir, device=device, attn_impl=attn_impl,
    )
    print(f"Loaded GPA {model_dir} on {device} with {normalized_attn}", flush=True)

    spark_tokenizer = spark_detokenizer = None
    speaker_cache: dict[str, torch.Tensor] = {}
    if task == "tts":
        audio_dir = resolve_audio_tokenizer_dir(audio_tokenizer_path)
        spark_tokenizer, spark_detokenizer = load_spark_stack(audio_tokenizer_path=audio_dir, device=device)
        output_wavs = output_dir / "wavs"
        output_wavs.mkdir(parents=True, exist_ok=True)
        ref_by_speaker: dict[str, Path] = {}
        for row in rows:
            speaker = str(row["speaker_id"])
            ref_path = _resolve_manifest_path(manifest, row["reference_path"])
            previous = ref_by_speaker.setdefault(speaker, ref_path)
            if previous != ref_path:
                raise ValueError(f"Speaker {speaker} has multiple GPA reference paths")
        synchronize(device)
        for speaker, ref_path in tqdm(ref_by_speaker.items(), desc="GPA TTS speakers", unit="speaker"):
            speaker_cache[speaker] = spark_tokenizer.tokenize([str(ref_path)])["global_tokens"].detach()
        synchronize(device)

    csv_path = output_dir / "samples.csv"
    completed = set()
    if resume and csv_path.exists():
        completed = {row["sample_id"] for row in _load_rows(csv_path)}
    pending = [(i, row) for i, row in enumerate(rows) if row["sample_id"] not in completed]
    if task == "asr":
        fieldnames = ["sample_id", "prediction", "reference", "inference_seconds", "audio_seconds"]
    elif task == "tts":
        fieldnames = ["sample_id", "reference_id", "output_path", "seed", "inference_seconds", "output_seconds", "semantic_tokens"]
    else:
        raise ValueError(f"Unsupported GPA task: {task}")

    append = resume and csv_path.exists() and csv_path.stat().st_size > 0
    samples: list[dict[str, Any]] = []
    with csv_path.open("a" if append else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for index, row in tqdm(pending, desc=f"GPA {task}", unit="sample"):
            seed_everything(seed + index)
            audio_path = _resolve_manifest_path(manifest, row["audio_path"])
            synchronize(device)
            started = time.perf_counter()
            if task == "asr":
                prediction = run_asr(
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    audio_path=str(audio_path),
                    sampling_rate=sampling_rate,
                    max_audio_seconds=asr_max_audio_seconds,
                    max_new_tokens=asr_max_new_tokens,
                    device=device,
                )
                synchronize(device)
                out_row = {
                    "sample_id": row["sample_id"],
                    "prediction": prediction,
                    "reference": row["text"],
                    "inference_seconds": f"{time.perf_counter() - started:.6f}",
                    "audio_seconds": row.get("duration", ""),
                }
                samples.append({"sample_id": row["sample_id"], "gen_text": prediction, "ref_text": row["text"]})
            else:
                ref_path = _resolve_manifest_path(manifest, row["reference_path"])
                waveform, semantic_ids, _ = _gpa_run_tts_with_optional_global(
                    model=model,
                    tts_processor=tts_processor,
                    tokenizer=tokenizer,
                    spark_tokenizer=spark_tokenizer,
                    spark_detokenizer=spark_detokenizer,
                    ref_audio_path=str(ref_path),
                    text=row["text"],
                    global_tokens=speaker_cache[str(row["speaker_id"])],
                    tts_stop_id=tts_stop_id,
                    max_new_tokens=tts_max_new_tokens,
                    max_semantic_tokens=tts_max_semantic_tokens,
                    device=device,
                )
                synchronize(device)
                wav_path = output_dir / "wavs" / f"{row['sample_id']}.wav"
                sf.write(str(wav_path), waveform, sampling_rate)
                rel = str(Path("wavs") / wav_path.name)
                out_row = {
                    "sample_id": row["sample_id"],
                    "reference_id": row.get("reference_id", ""),
                    "output_path": rel,
                    "seed": seed + index,
                    "inference_seconds": f"{time.perf_counter() - started:.6f}",
                    "output_seconds": f"{len(waveform) / sampling_rate:.6f}",
                    "semantic_tokens": len(semantic_ids),
                }
                samples.append({"sample_id": row["sample_id"], "ref_text": row["text"], "gen_wav": rel})
            writer.writerow(out_row)
            stream.flush()
    if resume and (output_dir / "samples.json").exists():
        old = json.loads((output_dir / "samples.json").read_text(encoding="utf-8"))
        by_id = {row["sample_id"]: row for row in old}
        by_id.update({row["sample_id"]: row for row in samples})
        samples = [by_id[row["sample_id"]] for row in rows if row["sample_id"] in by_id]
    atomic_json(output_dir / "samples.json", samples)
    return samples


def normalize_tts_samples(task_dir: Path) -> list[dict[str, Any]]:
    task_dir = Path(task_dir)
    if (task_dir / "samples.json").exists():
        return json.loads((task_dir / "samples.json").read_text(encoding="utf-8"))
    rows = _load_rows(task_dir / "samples.csv")
    samples = [
        {"sample_id": row["sample_id"], "ref_text": row.get("reference", row.get("text", "")), "gen_wav": row.get("output_path", "")}
        for row in rows
    ]
    atomic_json(task_dir / "samples.json", samples)
    return samples


def normalize_stt_samples(task_dir: Path) -> list[dict[str, Any]]:
    task_dir = Path(task_dir)
    if (task_dir / "samples.json").exists():
        return json.loads((task_dir / "samples.json").read_text(encoding="utf-8"))
    rows = _load_rows(task_dir / "samples.csv")
    samples = [
        {"sample_id": row["sample_id"], "gen_text": row.get("prediction", row.get("gen_text", "")), "ref_text": row.get("reference", row.get("ref_text", ""))}
        for row in rows
    ]
    atomic_json(task_dir / "samples.json", samples)
    return samples
