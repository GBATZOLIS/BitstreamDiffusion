# autoregression/callbacks.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Any, Dict

import torch
from torch.cuda.amp import autocast

# Prefer using the canonical helpers from your dataset module if available.
try:
    from data.text8 import text_to_token_ids, token_ids_to_text  # type: ignore
except Exception:
    text_to_token_ids = None
    token_ids_to_text = None

# Fallback mapping (only used if imports above fail).
_TEXT8_ALPHABET = list("abcdefghijklmnopqrstuvwxyz") + ["_"]  # '_' means space
_TEXT8_CHAR2ID = {ch: i for i, ch in enumerate(_TEXT8_ALPHABET)}
_TEXT8_ID2CHAR = {i: ch for ch, i in _TEXT8_CHAR2ID.items()}

# --- NEW: WikiText103 helpers ---
def _load_tokenizer(tokenizer_path: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise ImportError("WikiText callback requires `pip install tokenizers`.") from e
    return Tokenizer.from_file(str(tokenizer_path))

def _token_id_for(tok, token: str) -> Optional[int]:
    try:
        tid = tok.token_to_id(token)
        if tid is not None:
            return int(tid)
    except Exception:
        pass
    return None

def _clean_wikitext_artifacts(text: str) -> str:
    # Keep consistent with dataset version; avoid importing to prevent circular deps.
    import re as _re
    text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    text = _re.sub(r"\s+([,.:;?!%)])", r"\1", text)
    text = _re.sub(r"([(])\s+", r"\1", text)
    text = text.replace(" ' ", "'")
    return text.strip()


def _require(cfg, path: str):
    cur = cfg
    for part in path.split("."):
        if not hasattr(cur, part):
            raise KeyError(f"Missing required config field cfg.{path}")
        cur = getattr(cur, part)
    return cur

def _norm_ds(name: object) -> str:
    return str(name or "").strip().lower()

def _encode_text8(prompt: str) -> List[int]:
    if text_to_token_ids is not None:
        return text_to_token_ids(prompt).tolist()

    s = prompt.lower()
    out: List[int] = []
    for ch in s:
        if ch == " ":
            ch = "_"
        out.append(_TEXT8_CHAR2ID.get(ch, _TEXT8_CHAR2ID["_"]))
    return out


def _decode_text8(ids: List[int]) -> str:
    if token_ids_to_text is not None:
        return token_ids_to_text(torch.tensor(ids, dtype=torch.long))

    chars = []
    for i in ids:
        ch = _TEXT8_ID2CHAR.get(int(i), "_")
        chars.append(" " if ch == "_" else ch)
    return "".join(chars)


@torch.no_grad()
def maybe_generate_text8(
    *,
    raw_model,
    cfg,
    epoch: int,
    global_step: int,
    device: torch.device,
    run_dir: Path,
    writer=None,
    is_master: bool = True,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    ema=None,
    use_ema_available: bool = False,
):
    """
    Text8 autoregressive generation callback (char-level tokens only).

    All generation settings must be defined under:
      cfg.train.generation.{enabled,every_epochs,num_samples,max_new_tokens,temperature,top_k,prompt,start_token_id,use_ema}
    """
    if not is_master:
        return

    # Sanity: we only support Text8 + tokens for now.
    if _norm_ds(_require(cfg, "data.dataset")) != "text8":
        raise ValueError("maybe_generate_text8() only supports Text8 (case-insensitive).")
    if str(_require(cfg, "data.representation")).lower() != "tokens":
        raise ValueError("maybe_generate_text8() requires cfg.data.representation == 'tokens'.")

    g = _require(cfg, "train.generation")

    enabled = bool(_require(cfg, "train.generation.enabled"))
    if not enabled:
        return

    every_epochs = int(_require(cfg, "train.generation.every_epochs"))
    if every_epochs <= 0:
        return
    if (epoch + 1) % every_epochs != 0:
        return

    num_samples = int(_require(cfg, "train.generation.num_samples"))
    max_new_tokens = int(_require(cfg, "train.generation.max_new_tokens"))
    temperature = float(_require(cfg, "train.generation.temperature"))
    top_k_raw = int(_require(cfg, "train.generation.top_k"))
    top_k: Optional[int] = None if top_k_raw <= 0 else top_k_raw

    prompt = str(_require(cfg, "train.generation.prompt"))
    start_token_id = int(_require(cfg, "train.generation.start_token_id"))

    gen_use_ema = bool(_require(cfg, "train.generation.use_ema"))
    applied_ema = False
    if gen_use_ema and use_ema_available and ema is not None:
        ema.apply(raw_model)
        applied_ema = True

    # Build prompt batch
    if prompt.strip():
        prompt_ids = _encode_text8(prompt)
        if len(prompt_ids) == 0:
            prompt_ids = [start_token_id]
    else:
        prompt_ids = [start_token_id]

    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :].repeat(num_samples, 1)

    raw_model.eval()
    with autocast(enabled=use_amp, dtype=amp_dtype):
        out = raw_model.generate(
            idx=idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    out_cpu = out.detach().to("cpu")
    samples = [_decode_text8(out_cpu[i].tolist()) for i in range(num_samples)]

    if applied_ema:
        ema.restore(raw_model)

    # Console
    print(f"\n[AR] === Generation @ epoch {epoch+1} (step={global_step}) ===")
    for i, s in enumerate(samples):
        print(f"[sample {i:02d}] {s}")
    print("[AR] === End generation ===\n")

    # File
    gen_dir = run_dir / "generations_ar"
    gen_dir.mkdir(parents=True, exist_ok=True)
    out_path = gen_dir / f"epoch_{epoch+1:04d}_step_{global_step:08d}.txt"
    out_path.write_text(
        "\n\n".join([f"[sample {i:02d}]\n{samples[i]}" for i in range(len(samples))]),
        encoding="utf-8",
    )

    # TensorBoard
    if writer is not None:
        tb_text = "\n\n".join([f"sample {i:02d}:\n{samples[i]}" for i in range(len(samples))])
        writer.add_text("ar/generated_samples", tb_text, global_step)

    raw_model.train()

@torch.no_grad()
def maybe_generate_wikitext103(
    *,
    raw_model,
    cfg,
    epoch: int,
    global_step: int,
    device: torch.device,
    run_dir: Path,
    writer=None,
    is_master: bool = True,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    ema=None,
    use_ema_available: bool = False,
):
    """
    WikiText-103 autoregressive generation callback (token IDs).

    Expects:
      cfg.data.dataset in {"WikiText103","WikiText-103","WikiText"}
      cfg.data.representation == "tokens"
      cfg.data.tokenizer_path points to tokenizer_wiki_65k.json
    """
    if not is_master:
        return

    ds_name = _norm_ds(_require(cfg, "data.dataset"))
    if ds_name not in {"wikitext-103", "wikitext103", "wikitext"}:
        raise ValueError("maybe_generate_wikitext103() only supports WikiText-103 datasets.")
    if str(_require(cfg, "data.representation")).lower() != "tokens":
        raise ValueError("maybe_generate_wikitext103() requires cfg.data.representation == 'tokens'.")

    enabled = bool(_require(cfg, "train.generation.enabled"))
    if not enabled:
        return

    every_epochs = int(_require(cfg, "train.generation.every_epochs"))
    if every_epochs <= 0 or (epoch + 1) % every_epochs != 0:
        return

    num_samples = int(_require(cfg, "train.generation.num_samples"))
    max_new_tokens = int(_require(cfg, "train.generation.max_new_tokens"))
    temperature = float(_require(cfg, "train.generation.temperature"))
    top_k_raw = int(_require(cfg, "train.generation.top_k"))
    top_k: Optional[int] = None if top_k_raw <= 0 else top_k_raw

    prompt = str(_require(cfg, "train.generation.prompt"))

    # Load tokenizer (required)
    tok_path = Path(getattr(cfg.data, "tokenizer_path", "./datasets/wikitext-103/tokenizer_wiki_65k.json")).expanduser()
    tok = _load_tokenizer(tok_path)

    # Start token: prefer [EOS] if exists, else configured id, else 0.
    eos_id = _token_id_for(tok, "[EOS]")
    start_token_id = int(_require(cfg, "train.generation.start_token_id"))
    if eos_id is not None:
        start_token_id = eos_id

    gen_use_ema = bool(_require(cfg, "train.generation.use_ema"))
    applied_ema = False
    if gen_use_ema and use_ema_available and ema is not None:
        ema.apply(raw_model)
        applied_ema = True

    # prompt -> ids
    if prompt.strip():
        prompt_ids = tok.encode(prompt).ids
        if len(prompt_ids) == 0:
            prompt_ids = [start_token_id]
    else:
        prompt_ids = [start_token_id]

    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :].repeat(num_samples, 1)

    raw_model.eval()
    with autocast(enabled=use_amp, dtype=amp_dtype):
        out = raw_model.generate(
            idx=idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    out_cpu = out.detach().to("cpu")
    samples_raw = [tok.decode(out_cpu[i].tolist()) for i in range(num_samples)]
    samples_clean = [_clean_wikitext_artifacts(s) for s in samples_raw]

    if applied_ema:
        ema.restore(raw_model)

    # Console
    print(f"\n[AR] === WikiText-103 Generation @ epoch {epoch+1} (step={global_step}) ===")
    for i in range(num_samples):
        print(f"[raw {i:02d}]\n{samples_raw[i]}\n")
        print(f"[clean {i:02d}]\n{samples_clean[i]}\n")
    print("[AR] === End generation ===\n")

    # File
    gen_dir = run_dir / "generations_ar"
    gen_dir.mkdir(parents=True, exist_ok=True)
    out_path = gen_dir / f"wiki_epoch_{epoch+1:04d}_step_{global_step:08d}.txt"
    out_path.write_text(
        "\n\n".join(
            [f"[raw {i:02d}]\n{samples_raw[i]}\n\n[clean {i:02d}]\n{samples_clean[i]}" for i in range(num_samples)]
        ),
        encoding="utf-8",
    )

    # TensorBoard
    if writer is not None:
        writer.add_text("ar/wiki_generated_raw", "\n\n".join(samples_raw), global_step)
        writer.add_text("ar/wiki_generated_clean", "\n\n".join(samples_clean), global_step)

    raw_model.train()
