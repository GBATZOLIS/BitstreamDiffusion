#!/usr/bin/env python3
"""Build the portable, metrics-free webpage for the core CoBit results."""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT / "cobit_core_results_demo"
TITLE = "Joint Text-audio generation with CoBit"
TTS_VC_TITLE = "Text-to-speech: Speaker realization for CoBit MLS"
TTS_VC_META = ROOT / "datasets/test_common/cache_test_clean_tts_vc_632.meta.json"
TTS_VC_PROMPT_IDS = (
    "1320-122612-0014",
    "2300-131720-0001",
    "4446-2275-0026",
)
TTS_VC_COLUMNS = (
    ("speaker_id", "Speaker", "text"),
    ("ref_wav", "Held-out Reference", "audio"),
    ("gen_wav", "Generated Audio", "audio"),
)

TASKS = {
    "joint": ("Joint Generation", "joint", (
        ("gen_text", "Generated Text", "text"),
        ("gen_wav", "Generated Audio", "audio"),
    )),
    "tts": ("Text-to-Speech", "tts", (
        ("ref_text", "Reference Text", "text"),
        ("ref_wav", "Reference Audio", "audio"),
        ("gen_wav", "Generated Audio", "audio"),
    )),
    "stt": ("Speech-to-Text", "stt", (
        ("ref_wav", "Reference Audio", "audio"),
        ("ref_text", "Reference Text", "text"),
        ("gen_text", "Generated Text", "text"),
    )),
    "cont": ("Continuation", "cont_taste", (
        ("gen_text", "Generated Text", "text"),
        ("gen_wav", "Generated Audio", "audio"),
    )),
}

@dataclass(frozen=True)
class Model:
    label: str
    slug: str
    run_dir: Path
    tags: dict[str, str]

MODELS = (
    Model("CoBit-MLS", "cobit-mls",
          ROOT / "runs/cobit_mls/evaluation/textaudio_eval/last", {
        "joint": "ddim_entropic_nfe512_stoch-g0.175",
        "tts": "ddim_entropic_nfe512_stoch-g0.27_gs5",
        "tts_vc": "ddim_entropic_nfe512_stoch-g0.27_gs5",
        "stt": "ddim_entropic_nfe512_stoch-g0.27_gs5",
        "cont": "ddim_entropic_nfe512_stoch-g0.4_gs5",
    }),
    Model("CoBit-LibriTTS", "cobit-libritts",
          ROOT / "runs/cobit_libritts/evaluation/textaudio_eval/last", {
        "joint": "ddim_entropic_nfe256_stoch-g0.175",
        "tts": "ddim_entropic_nfe512_stoch-g0.23_gs6",
        "stt": "ddim_entropic_nfe512_stoch-g0.37_gs5",
        "cont": "ddim_entropic_nfe512_stoch-g0.15_gs5",
    }),
)

CSS = r"""
:root{--bg:#fff;--alt:#f7f8fa;--text:#1e293b;--muted:#64748b;--border:#e2e8f0;--accent:#2563eb;--accent-bg:#eff6ff;--radius:8px;--font:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:64px}body{margin:0;color:var(--text);background:var(--bg);font-family:var(--font);line-height:1.55}
.topnav{position:sticky;top:0;z-index:100;display:flex;justify-content:flex-end;min-height:52px;padding:.45rem 2rem;background:rgba(255,255,255,.96);border-bottom:1px solid var(--border);box-shadow:0 1px 3px rgba(0,0,0,.08)}
.nav-links{display:flex;flex-wrap:wrap;gap:.25rem}.nav-links a{padding:.35rem .75rem;color:var(--accent);border-radius:4px;font-size:.875rem;font-weight:600;text-decoration:none}.nav-links a:hover{background:var(--accent-bg)}
.container{width:min(1440px,100%);margin:auto;padding:2rem}h1{margin:.25rem 0 2.5rem;font-size:clamp(1.8rem,4vw,2.6rem)}section{margin-bottom:3.5rem;padding-top:.75rem;border-top:2px solid var(--accent)}h2{margin:0 0 1rem;font-size:1.45rem}
.tab-strip{display:flex;flex-wrap:wrap;border-bottom:2px solid var(--border)}.tab-btn{margin:0 3px -2px 0;padding:.55rem 1.2rem;border:1px solid var(--border);border-bottom:0;border-radius:6px 6px 0 0;background:var(--alt);color:var(--muted);cursor:pointer;font:inherit;font-size:.9rem;font-weight:600}.tab-btn:hover{color:var(--accent);background:var(--accent-bg)}.tab-btn.active{color:var(--accent);background:var(--bg);border-bottom:2px solid var(--bg)}
.model-panel{padding-top:1rem}.prompt-text{margin:0 0 1rem;font-size:1.1rem;font-weight:600}.table-wrap{max-height:42rem;overflow:auto;border:1px solid var(--border);border-radius:var(--radius)}table{width:100%;border-collapse:collapse;font-size:.9rem}th{position:sticky;top:0;z-index:10;padding:.7rem .8rem;background:var(--alt);border-bottom:1px solid var(--border);text-align:left;white-space:nowrap}td{padding:.75rem .8rem;border-top:1px solid var(--border);vertical-align:top}tbody tr:first-child td{border-top:0}tbody tr:nth-child(even){background:var(--alt)}tbody tr:hover{background:var(--accent-bg)}
.text-cell{min-width:260px;max-width:480px;white-space:normal}.audio-cell{min-width:250px;white-space:nowrap}audio{width:240px;height:36px}.empty{color:var(--muted);font-style:italic}
@media(max-width:720px){.topnav{justify-content:flex-start;padding:.4rem 1rem}.container{padding:1.25rem 1rem}.table-wrap{max-height:36rem}audio{width:200px}}
"""
JS = r"""
document.querySelectorAll('.tab-btn').forEach(function(button){button.addEventListener('click',function(){const task=button.dataset.task,model=button.dataset.model;document.querySelectorAll('.tab-btn[data-task="'+task+'"]').forEach(function(x){x.classList.toggle('active',x===button)});document.querySelectorAll('.model-panel[data-task="'+task+'"]').forEach(function(x){x.hidden=x.dataset.model!==model})})});
"""

def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))

def load_rows(model: Model, task: str, cap: int) -> list[dict[str, Any]]:
    tag = model.tags[task]
    source_task = TASKS[task][1]
    if task != "stt":
        source_dir = model.run_dir / tag / source_task
        return [dict(x, _source_dir=source_dir)
                for x in load_json(source_dir / "samples.json")[:cap]]
    clean_cap = cap // 2
    rows = []
    for partition, amount in (("clean", clean_cap), ("other", cap-clean_cap)):
        source_dir = model.run_dir / tag / f"stt_{partition}"
        rows += [dict(x, _source_dir=source_dir)
                 for x in load_json(source_dir / "samples.json")[:amount]]
    return rows

def load_tts_vc_rows(model: Model) -> list[dict[str, Any]]:
    """Return prompt-major rows for the selected prompts and every held-out speaker."""
    meta = load_json(TTS_VC_META)
    source_dir = model.run_dir / model.tags["tts_vc"] / "tts_vc"
    samples = load_json(source_dir / "samples.json")
    n = int(meta["n_sequences"])
    fields = ("row_speaker_ids", "row_heldout_ids", "row_prompt_source_ids")
    if len(samples) != n or any(len(meta.get(field, [])) != n for field in fields):
        raise ValueError("tts_vc samples and cache row metadata are not aligned")

    prompts = list(TTS_VC_PROMPT_IDS)
    speakers = list(dict.fromkeys(str(value) for value in meta["row_speaker_ids"]))
    by_key = {}
    for index, (speaker, heldout_id, prompt_id) in enumerate(zip(
        meta["row_speaker_ids"], meta["row_heldout_ids"], meta["row_prompt_source_ids"]
    )):
        key = (str(prompt_id), str(speaker))
        if key in by_key:
            raise ValueError(f"duplicate tts_vc row for prompt/speaker {key}")
        by_key[key] = (index, str(heldout_id))

    rows = []
    for prompt_id in prompts:
        for speaker in speakers:
            key = (prompt_id, speaker)
            if key not in by_key:
                raise ValueError(f"missing tts_vc row for prompt/speaker {key}")
            index, heldout_id = by_key[key]
            sample = samples[index]
            rows.append({
                "prompt_id": prompt_id,
                "ref_text": sample.get("ref_text"),
                "speaker_id": speaker,
                "heldout_id": heldout_id,
                "gen_wav": sample.get("gen_wav"),
                "_source_dir": source_dir,
            })
    return rows

def load_heldout_wavs(meta: dict[str, Any], sample_ids: set[str]) -> tuple[dict[str, Any], int]:
    """Load only the selected held-out utterances from the evaluation dataset."""
    from datasets import Audio, Dataset, load_dataset
    from scripts.text_audio.cache_utils import decode_wav

    hf_home = Path(os.environ.get("HF_HOME", ROOT.parent / "huggingface_cache"))
    cache_root = Path(os.environ.get("HF_DATASETS_CACHE", hf_home / "datasets"))
    dataset_name = str(meta["hf_path"]).replace("/", "___")
    arrow_name = f'{str(meta["hf_path"]).split("/")[-1]}-{meta["hf_split"]}.arrow'
    cached = sorted(
        (cache_root / dataset_name / str(meta["hf_config"])).glob(
            f"*/*/{arrow_name}"
        )
    )
    if cached:
        dataset = Dataset.from_file(str(cached[-1])).cast_column(
            "audio", Audio(decode=False)
        )
    else:
        dataset = load_dataset(
            str(meta["hf_path"]), str(meta["hf_config"]),
            split=str(meta["hf_split"]), cache_dir=str(cache_root),
        ).cast_column("audio", Audio(decode=False))
    index = {str(sample_id): i for i, sample_id in enumerate(dataset["id"])}
    missing = sorted(sample_ids - set(index))
    if missing:
        raise ValueError(f"held-out sample IDs absent from source dataset: {missing}")
    return {
        sample_id: decode_wav(dataset[index[sample_id]]["audio"]["bytes"])
        for sample_id in sample_ids
    }, 16_000

def resolve_audio(model: Model, row: dict[str, Any], field: str) -> Path:
    value = row.get(field)
    if not value:
        raise ValueError(f"Missing {field}: {row}")
    path = ((Path(row["_source_dir"]) / value) if field == "gen_wav"
            else (model.run_dir / value))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path

def copy_audio(source: Path, out: Path, model: Model, task: str,
               index: int, field: str) -> str:
    rel = Path("audio") / model.slug / task / f"sample_{index:02d}" / f"{field}{source.suffix.lower()}"
    dest = out / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return rel.as_posix()

def write_wav(source: Any, sample_rate: int, dest: Path) -> None:
    import wave
    import numpy as np

    arr = np.asarray(source, dtype=np.float32).flatten()
    pcm = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())

def render_cell(value: Any, kind: str) -> str:
    if value is None or value == "":
        return '<td class="empty">—</td>'
    if kind == "audio":
        src = html.escape(str(value), quote=True)
        return f'<td class="audio-cell"><audio controls preload="none"><source src="{src}" type="audio/wav">Your browser does not support audio playback.</audio></td>'
    return f'<td class="text-cell">{html.escape(str(value)).replace(chr(10), "<br>")}</td>'

def render_table(columns: tuple, rows: list[dict[str, Any]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in columns)
    body = "".join("<tr>" + "".join(render_cell(row.get(key), kind)
                   for key, _, kind in columns) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

def page(data: dict[str, dict[str, list[dict[str, Any]]]],
         tts_vc_rows: list[dict[str, Any]]) -> str:
    nav = "".join(f'<a href="#{key}">{html.escape(meta[0])}</a>'
                  for key, meta in TASKS.items())
    nav += f'<a href="#tts-vc">{html.escape(TTS_VC_TITLE)}</a>'
    sections = []
    for task, (name, _, columns) in TASKS.items():
        tabs, panels = [], []
        for i, model in enumerate(MODELS):
            tabs.append(f'<button class="tab-btn{" active" if i == 0 else ""}" data-task="{task}" data-model="{model.slug}">{html.escape(model.label)}</button>')
            panels.append(f'<div class="model-panel" data-task="{task}" data-model="{model.slug}"{"" if i == 0 else " hidden"}>{render_table(columns, data[task][model.slug])}</div>')
        sections.append(f'<section id="{task}"><h2>{html.escape(name)}</h2><div class="tab-strip">{"".join(tabs)}</div>{"".join(panels)}</section>')
    prompt_ids = list(dict.fromkeys(row["prompt_id"] for row in tts_vc_rows))
    tabs, panels = [], []
    for i, prompt_id in enumerate(prompt_ids):
        prompt_rows = [row for row in tts_vc_rows if row["prompt_id"] == prompt_id]
        texts = {row["ref_text"] for row in prompt_rows}
        if len(texts) != 1:
            raise ValueError(f"tts_vc prompt {prompt_id} has inconsistent text")
        prompt_attr = html.escape(str(prompt_id), quote=True)
        tabs.append(
            f'<button class="tab-btn{" active" if i == 0 else ""}" '
            f'data-task="tts-vc" data-model="{prompt_attr}">{html.escape(str(prompt_id))}</button>'
        )
        panels.append(
            f'<div class="model-panel" data-task="tts-vc" data-model="{prompt_attr}"'
            f'{"" if i == 0 else " hidden"}><h3 class="prompt-text">'
            f'{html.escape(str(next(iter(texts))))}</h3>'
            f'{render_table(TTS_VC_COLUMNS, prompt_rows)}</div>'
        )
    sections.append(
        f'<section id="tts-vc"><h2>{html.escape(TTS_VC_TITLE)}</h2>'
        f'<div class="tab-strip">{"".join(tabs)}</div>{"".join(panels)}</section>'
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>{html.escape(TITLE)}</title><style>{CSS}</style></head>
<body><nav class="topnav"><div class="nav-links">{nav}</div></nav><main class="container"><h1>{html.escape(TITLE)}</h1>{''.join(sections)}</main><script>{JS}</script></body></html>
"""

def build(out: Path, zip_path: Path, cap: int, force: bool) -> None:
    if cap < 1:
        raise ValueError("Sample count must be positive")
    for path in (out, zip_path):
        if path.exists() and not force:
            raise FileExistsError(f"{path} exists; pass --force to replace it")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    data: dict[str, dict[str, list[dict[str, Any]]]] = {}
    count = 0
    try:
        for task, (_, _, columns) in TASKS.items():
            data[task] = {}
            for model in MODELS:
                source_rows = load_rows(model, task, cap)
                if len(source_rows) != cap:
                    raise ValueError(f"{model.label}/{task}: expected {cap}, got {len(source_rows)}")
                shown = []
                for i, source_row in enumerate(source_rows):
                    row = {}
                    for field, _, kind in columns:
                        if kind == "audio":
                            row[field] = copy_audio(resolve_audio(model, source_row, field), out, model, task, i, field)
                            count += 1
                        else:
                            row[field] = source_row.get(field)
                    shown.append(row)
                data[task][model.slug] = shown
                print(f"Prepared {model.label}/{task}: {len(shown)} samples", flush=True)
        cobit_mls = next(model for model in MODELS if model.slug == "cobit-mls")
        source_rows = load_tts_vc_rows(cobit_mls)
        vc_meta = load_json(TTS_VC_META)
        heldout_ids = {row["heldout_id"] for row in source_rows}
        heldout_wavs, heldout_sample_rate = load_heldout_wavs(vc_meta, heldout_ids)
        ref_paths = {}
        for row in source_rows:
            speaker = row["speaker_id"]
            if speaker not in ref_paths:
                rel = (Path("audio") / cobit_mls.slug / "tts-vc" / "references"
                       / f"speaker_{speaker}.wav")
                write_wav(
                    heldout_wavs[row["heldout_id"]], heldout_sample_rate, out / rel
                )
                ref_paths[speaker] = rel.as_posix()
                count += 1
        tts_vc_rows = []
        for i, source_row in enumerate(source_rows):
            tts_vc_rows.append({
                "prompt_id": source_row["prompt_id"],
                "ref_text": source_row["ref_text"],
                "speaker_id": source_row["speaker_id"],
                "ref_wav": ref_paths[source_row["speaker_id"]],
                "gen_wav": copy_audio(
                    resolve_audio(cobit_mls, source_row, "gen_wav"),
                    out, cobit_mls, "tts-vc", i, "gen_wav",
                ),
            })
            count += 1
        print(
            f"Prepared {TTS_VC_TITLE}: {len(tts_vc_rows)} rows "
            f"(3 prompts x {len(ref_paths)} speakers)", flush=True,
        )
        (out / "index.html").write_text(page(data, tts_vc_rows), encoding="utf-8")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{zip_path.name}.", suffix=".tmp", dir=zip_path.parent)
        os.close(fd)
        temp = Path(temp_name)
        try:
            with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
                for source in sorted(out.rglob("*")):
                    if source.is_file():
                        archive.write(source, Path(out.name) / source.relative_to(out))
            os.replace(temp, zip_path)
            zip_path.chmod(0o644)
        finally:
            temp.unlink(missing_ok=True)
    except Exception:
        shutil.rmtree(out, ignore_errors=True)
        raise
    print(f"Wrote {out / 'index.html'}", flush=True)
    print(f"Copied {count} audio files", flush=True)
    print(f"Wrote {zip_path}", flush=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_OUT.with_suffix(".zip"))
    parser.add_argument("--samples-per-model-task", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.output_dir.resolve(), args.zip_path.resolve(),
          args.samples_per_model_task, args.force)

if __name__ == "__main__":
    main()
