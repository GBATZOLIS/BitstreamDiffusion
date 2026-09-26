#!/usr/bin/env python3
"""Build the portable, metrics-free webpage for the core CoBit results."""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "cobit_core_results_demo"
TITLE = "Joint Text-audio generation with CoBit"

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
.model-panel{padding-top:1rem}.table-wrap{max-height:42rem;overflow:auto;border:1px solid var(--border);border-radius:var(--radius)}table{width:100%;border-collapse:collapse;font-size:.9rem}th{position:sticky;top:0;z-index:10;padding:.7rem .8rem;background:var(--alt);border-bottom:1px solid var(--border);text-align:left;white-space:nowrap}td{padding:.75rem .8rem;border-top:1px solid var(--border);vertical-align:top}tbody tr:first-child td{border-top:0}tbody tr:nth-child(even){background:var(--alt)}tbody tr:hover{background:var(--accent-bg)}
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

def page(data: dict[str, dict[str, list[dict[str, Any]]]]) -> str:
    nav = "".join(f'<a href="#{key}">{html.escape(meta[0])}</a>'
                  for key, meta in TASKS.items())
    sections = []
    for task, (name, _, columns) in TASKS.items():
        tabs, panels = [], []
        for i, model in enumerate(MODELS):
            tabs.append(f'<button class="tab-btn{" active" if i == 0 else ""}" data-task="{task}" data-model="{model.slug}">{html.escape(model.label)}</button>')
            panels.append(f'<div class="model-panel" data-task="{task}" data-model="{model.slug}"{"" if i == 0 else " hidden"}>{render_table(columns, data[task][model.slug])}</div>')
        sections.append(f'<section id="{task}"><h2>{html.escape(name)}</h2><div class="tab-strip">{"".join(tabs)}</div>{"".join(panels)}</section>')
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
        (out / "index.html").write_text(page(data), encoding="utf-8")
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
