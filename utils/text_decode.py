from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from utils.ecc_secded import (
    ecc_chunk_len,
    ecc_decode_batch_bitstream,
    ecc_decode_token_bits,
    ecc_from_cfg,
)


def _norm_ds(name: object) -> str:
    return (
        str(name or "")
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def _safe_name(s: str) -> str:
    return str(s).replace("/", "_").replace("-", "_").replace(":", "_")


# -----------------------------------------------------------------------------
# Gray helpers
# -----------------------------------------------------------------------------
def gray_to_int(n: int) -> int:
    """Scalar Gray-to-int."""
    mask = n
    while mask != 0:
        mask >>= 1
        n ^= mask
    return n


def gray_to_int_vectorized(val: torch.Tensor, num_bits: int = 16) -> torch.Tensor:
    """
    Vectorized Gray-to-int for tensors.
    """
    del num_bits
    val = val ^ (val >> 1)
    val = val ^ (val >> 2)
    val = val ^ (val >> 4)
    val = val ^ (val >> 8)
    val = val ^ (val >> 16)
    return val


# -----------------------------------------------------------------------------
# Tokenizer compatibility helpers
# -----------------------------------------------------------------------------
def _tokenizer_batch_decode(
    tokenizer: Any,
    token_ids_list: List[List[int]],
    *,
    skip_special_tokens: bool = False,
) -> List[str]:
    if hasattr(tokenizer, "decode_batch"):
        return tokenizer.decode_batch(
            token_ids_list,
            skip_special_tokens=skip_special_tokens,
        )

    if hasattr(tokenizer, "batch_decode"):
        return tokenizer.batch_decode(
            token_ids_list,
            skip_special_tokens=skip_special_tokens,
        )

    return [
        tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        for ids in token_ids_list
    ]


def extract_dataset_attr(ds, name: str):
    if hasattr(ds, name):
        return getattr(ds, name)
    if hasattr(ds, "dataset"):
        return extract_dataset_attr(ds.dataset, name)
    return None


def _get_unk_id(tokenizer, fallback: int = 1) -> int:
    uid = getattr(tokenizer, "unk_token_id", None)
    if uid is not None:
        try:
            return int(uid)
        except Exception:
            pass

    for tok in ("<unk>", "<UNK>", "[UNK]"):
        try:
            if hasattr(tokenizer, "token_to_id"):
                uid = tokenizer.token_to_id(tok)
                if uid is not None and int(uid) >= 0:
                    return int(uid)
        except Exception:
            pass

        try:
            if hasattr(tokenizer, "convert_tokens_to_ids"):
                uid = tokenizer.convert_tokens_to_ids(tok)
                if uid is not None and int(uid) >= 0:
                    return int(uid)
        except Exception:
            pass

    return int(fallback)


def _get_tokenizer_vocab_size(tokenizer: Any) -> Optional[int]:
    try:
        v = getattr(tokenizer, "vocab_size", None)
        if v is not None:
            return int(v)
    except Exception:
        pass

    try:
        if hasattr(tokenizer, "get_vocab_size"):
            return int(tokenizer.get_vocab_size())
    except Exception:
        pass

    try:
        if hasattr(tokenizer, "get_vocab"):
            return int(len(tokenizer.get_vocab()))
    except Exception:
        pass

    return None


def pua_string_to_gpt2_ids(
    s: str,
    *,
    base_char_offset: int,
    vocab_size_gpt2: int,
    invalid_policy: str = "raise",
    eos_id: Optional[int] = None,
) -> List[int]:
    """
    Convert a PUA string back to GPT-2 token ids.

    invalid_policy:
      - "raise": raise on any non-PUA / out-of-range char
      - "skip": skip invalid chars
      - "eos": replace invalid chars with eos_id
    """
    out: List[int] = []
    invalid_policy = str(invalid_policy).lower().strip()

    for ch in s:
        tid = ord(ch) - int(base_char_offset)
        if 0 <= tid < int(vocab_size_gpt2):
            out.append(int(tid))
            continue

        if invalid_policy == "skip":
            continue

        if invalid_policy == "eos":
            if eos_id is None:
                raise RuntimeError("invalid_policy='eos' requires eos_id")
            out.append(int(eos_id))
            continue

        raise RuntimeError(
            f"Decoded character outside GPT-2 id range: ord={ord(ch)} "
            f"offset={base_char_offset} -> tid={tid}"
        )

    return out


@torch.no_grad()
def _safe_code_ids_to_pua_string(
    code_ids: torch.Tensor,
    *,
    code_tokenizer: Any,
    actual_code_vocab_size: int,
    pad_id: Optional[int],
    eoseq_id: Optional[int],
    unk_id: Optional[int],
) -> Tuple[str, Dict[str, int]]:
    """
    Single-sequence helper.

    Policy:
      - stop on exact code-level PAD / EOSEQ
      - keep valid in-range code ids
      - keep explicit UNK if configured
      - skip invalid ids
    """
    ids = code_ids.detach().view(-1).to(torch.long).cpu().tolist()

    kept: List[int] = []
    stats: Dict[str, int] = {
        "n_input": len(ids),
        "n_kept": 0,
        "n_invalid_skipped": 0,
        "hit_pad": 0,
        "hit_eoseq": 0,
        "n_unk_kept": 0,
        "stop_idx_code": -1,
    }

    for idx, tid in enumerate(ids):
        tid = int(tid)

        if pad_id is not None and tid == int(pad_id):
            stats["hit_pad"] = 1
            stats["stop_idx_code"] = int(idx)
            break

        if eoseq_id is not None and tid == int(eoseq_id):
            stats["hit_eoseq"] = 1
            stats["stop_idx_code"] = int(idx)
            break

        if 0 <= tid < int(actual_code_vocab_size):
            kept.append(tid)
            continue

        if unk_id is not None and tid == int(unk_id):
            kept.append(tid)
            stats["n_unk_kept"] += 1
            continue

        stats["n_invalid_skipped"] += 1

    stats["n_kept"] = len(kept)

    if not kept:
        return "", stats

    try:
        pua = code_tokenizer.decode(kept, skip_special_tokens=False)
    except TypeError:
        pua = code_tokenizer.decode(kept)

    return ("" if pua is None else str(pua)), stats

def load_openwebtext_gpt2id_bpe16_assets(
    root: Path,
    code_tokenizer_path: str,
    code_tokenizer_meta_path: Optional[str],
    tokenizer_name: str = "gpt2",
):
    try:
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "gpt2id_bpe16 decoding requires `pip install tokenizers transformers`."
        ) from e

    tok_path = Path(code_tokenizer_path)
    if not tok_path.is_absolute():
        tok_path = root / tok_path

    if code_tokenizer_meta_path is None:
        meta_path = tok_path.with_suffix(".meta.json")
    else:
        meta_path = Path(code_tokenizer_meta_path)
        if not meta_path.is_absolute():
            meta_path = root / meta_path

    if not tok_path.exists():
        raise RuntimeError(f"Missing code tokenizer JSON: {tok_path}")
    if not meta_path.exists():
        raise RuntimeError(f"Missing code tokenizer meta: {meta_path}")

    code_tok = Tokenizer.from_file(str(tok_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    gpt2_tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if gpt2_tok.pad_token_id is None:
        gpt2_tok.pad_token = gpt2_tok.eos_token

    return gpt2_tok, code_tok, meta


@torch.no_grad()
def _coerce_token_id_tensor(samples_tokens: torch.Tensor) -> torch.Tensor:
    """
    Accept:
      - [S]
      - [B, S]
      - [B, S, V]
    Returns:
      - [B, S] long token ids
    """
    if samples_tokens.dim() == 1:
        return samples_tokens.unsqueeze(0).to(torch.long)

    if samples_tokens.dim() == 2:
        if samples_tokens.dtype == torch.long:
            return samples_tokens
        return samples_tokens.to(torch.long)

    if samples_tokens.dim() == 3:
        return samples_tokens.argmax(dim=-1).to(torch.long)

    raise ValueError(
        f"Expected token samples with shape [S], [B,S], or [B,S,V], got {tuple(samples_tokens.shape)}"
    )


# -----------------------------------------------------------------------------
# Vectorized decoders
# -----------------------------------------------------------------------------
@torch.no_grad()
def bits_to_text_semantic_vectorized(
    bits: torch.Tensor,
    tokenizer: Any,
    new_to_old_map: Dict[int, int],
    *,
    bits_per_token: int,
    cfg: Optional[Any] = None,
    normalize_text8: bool = False,
) -> List[str]:
    B, S_total = bits.shape
    device = bits.device

    ecc = ecc_from_cfg(cfg) if cfg is not None else None
    use_ecc = bool(ecc is not None and ecc.enabled)

    if use_ecc:
        data_bits_flat, _, uncorrectable_mask = ecc_decode_batch_bitstream(bits, ecc)
        m = int(ecc.data_bits)
        T = data_bits_flat.size(1) // m
        data_bits = data_bits_flat.view(B, T, m)
    else:
        m = int(bits_per_token)
        T = S_total // m
        data_bits = bits[:, : T * m].view(B, T, m)
        uncorrectable_mask = None

    powers = 2 ** torch.arange(m - 1, -1, -1, device=device)
    gray_vals = (data_bits * powers).sum(dim=-1).to(torch.long)

    ranks = gray_to_int_vectorized(gray_vals)

    unk_id = _get_unk_id(tokenizer, fallback=1)

    if new_to_old_map:
        max_rank = max(int(k) for k in new_to_old_map.keys())
        lookup_table = torch.full((max_rank + 2,), int(unk_id), dtype=torch.long, device=device)
        k_t = torch.tensor(list(new_to_old_map.keys()), dtype=torch.long, device=device)
        v_t = torch.tensor(list(new_to_old_map.values()), dtype=torch.long, device=device)
        lookup_table[k_t] = v_t

        ranks = ranks.clamp(0, max_rank + 1)
        token_ids = lookup_table[ranks]
    else:
        token_ids = ranks

    if use_ecc and uncorrectable_mask is not None:
        token_ids[uncorrectable_mask] = int(unk_id)

    token_ids_list = token_ids.cpu().tolist()
    texts = _tokenizer_batch_decode(
        tokenizer,
        token_ids_list,
        skip_special_tokens=False,
    )

    if normalize_text8:
        texts = [t.replace("_", " ") for t in texts]

    return texts


@torch.no_grad()
def bits_to_text_raw_binary_vectorized(
    bits: torch.Tensor,
    tokenizer: Any,
    *,
    bits_per_token: int,
    cfg: Optional[Any] = None,
) -> List[str]:
    B, S_total = bits.shape
    device = bits.device

    ecc = ecc_from_cfg(cfg) if cfg is not None else None
    use_ecc = bool(ecc is not None and ecc.enabled)

    if use_ecc:
        data_bits_flat, _, uncorrectable_mask = ecc_decode_batch_bitstream(bits, ecc)
        m = int(ecc.data_bits)
        T = data_bits_flat.size(1) // m
        data_bits = data_bits_flat.view(B, T, m)
    else:
        m = int(bits_per_token)
        T = S_total // m
        data_bits = bits[:, : T * m].view(B, T, m)
        uncorrectable_mask = None

    powers = 2 ** torch.arange(m - 1, -1, -1, device=device)
    token_ids = (data_bits * powers).sum(dim=-1).to(torch.long)

    unk_id = _get_unk_id(tokenizer, fallback=1)
    vocab_size = _get_tokenizer_vocab_size(tokenizer)

    if vocab_size is not None and vocab_size > 0:
        # --- PATCHED: Modulo wrap-around for redundancy ---
        token_ids = token_ids % int(vocab_size)

    if use_ecc and uncorrectable_mask is not None:
        token_ids[uncorrectable_mask] = int(unk_id)

    token_ids_list = token_ids.cpu().tolist()
    return _tokenizer_batch_decode(
        tokenizer,
        token_ids_list,
        skip_special_tokens=False,
    )


@torch.no_grad()
def bitstreams_to_token_ids_raw_binary(
    bits: torch.Tensor,
    *,
    bits_per_token: int,
    cfg: Optional[Any] = None,
) -> torch.Tensor:
    if bits.dim() == 1:
        bits = bits.unsqueeze(0)
    if bits.dim() != 2:
        raise ValueError(f"Expected [B,S] or [S], got {tuple(bits.shape)}")

    ecc = ecc_from_cfg(cfg) if cfg is not None else None
    use_ecc = bool(ecc is not None and ecc.enabled)

    bits = bits.long()

    if use_ecc:
        data_bits_flat, _, uncorrectable_mask = ecc_decode_batch_bitstream(bits, ecc)
        m = int(ecc.data_bits)
        B = int(bits.size(0))
        T = data_bits_flat.size(1) // m
        data_bits = data_bits_flat.view(B, T, m)
    else:
        m = int(bits_per_token)
        B, S_total = bits.shape
        T = S_total // m
        data_bits = bits[:, : T * m].view(B, T, m)
        uncorrectable_mask = None

    device = data_bits.device
    powers = 2 ** torch.arange(m - 1, -1, -1, device=device)
    token_ids = (data_bits * powers).sum(dim=-1).to(torch.long)

    if use_ecc and uncorrectable_mask is not None:
        token_ids[uncorrectable_mask] = 1

    return token_ids.cpu()


@torch.no_grad()
def code_ids_to_gpt2_token_ids_for_eval(
    code_ids: torch.Tensor,
    *,
    gpt2_tokenizer: Any,
    code_tokenizer: Any,
    code_meta: Dict[str, Any],
    return_debug_stats: bool = False,
):
    """
    Batched robust inverse mapping for sequence_codec='gpt2id_bpe16'.

    Input:
      code_ids: [B, T] or [T]

    Output:
      - if return_debug_stats=False:
            gpt2_ids_padded: [B, T_gpt2_max] long, padded with EOS
      - if return_debug_stats=True:
            (gpt2_ids_padded, debug_stats)

    Correct stopping / trimming policy for OpenWebText:
      1. Stop ONLY at the code-token level (first PAD or EOSEQ).
      2. Decode all surviving code ids to the PUA string.
      3. Map all PUA chars back to GPT-2 token ids.
      4. NEVER truncate at internal GPT-2 EOS.
      5. Optionally strip:
         - one leading synthetic BOS-like EOS token
         - one trailing synthetic EOS token
    """
    if code_ids.dim() == 1:
        code_ids = code_ids.unsqueeze(0)
    if code_ids.dim() != 2:
        raise ValueError(f"Expected code_ids with shape [T] or [B,T], got {tuple(code_ids.shape)}")

    if int(code_ids.size(0)) == 0:
        empty = torch.empty((0, 0), dtype=torch.long)
        return (empty, []) if return_debug_stats else empty

    actual_code_vocab_size = int(
        code_meta.get(
            "actual_code_vocab_size",
            _get_tokenizer_vocab_size(code_tokenizer),
        )
    )
    if actual_code_vocab_size <= 0:
        raise RuntimeError("Could not determine actual_code_vocab_size for code tokenizer")

    pad_id = int(code_meta["pad_id"]) if code_meta.get("pad_id", None) is not None else None
    unk_id = int(code_meta["unk_id"]) if code_meta.get("unk_id", None) is not None else None
    eoseq_id = int(code_meta["eoseq_id"]) if code_meta.get("eoseq_id", None) is not None else None
    base_char_offset = int(code_meta["base_char_offset"])

    eos_token_id = getattr(gpt2_tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise RuntimeError("GPT-2 tokenizer must expose eos_token_id")
    eos_token_id = int(eos_token_id)

    vocab_size_gpt2 = _get_tokenizer_vocab_size(gpt2_tokenizer)
    if vocab_size_gpt2 is None:
        raise RuntimeError("Could not determine GPT-2 tokenizer vocab size")
    vocab_size_gpt2 = int(vocab_size_gpt2)

    rows: List[List[int]] = []
    debug_rows: List[Dict[str, Any]] = []
    max_len = 0

    for i in range(int(code_ids.size(0))):
        row_ids = code_ids[i]

        # Stage 1: authoritative stop at code level.
        pua_string, code_stats = _safe_code_ids_to_pua_string(
            row_ids,
            code_tokenizer=code_tokenizer,
            actual_code_vocab_size=actual_code_vocab_size,
            pad_id=pad_id,
            eoseq_id=eoseq_id,
            unk_id=unk_id,
        )

        # Stage 2: inverse codec back to GPT-2 ids.
        # IMPORTANT: do not use GPT-2 EOS as a stopping criterion.
        gpt2_ids_full = pua_string_to_gpt2_ids(
            pua_string,
            base_char_offset=base_char_offset,
            vocab_size_gpt2=vocab_size_gpt2,
            invalid_policy="skip",
            eos_id=eos_token_id,
        )

        # Stage 3: strip only synthetic edge markers, never internal EOS.
        dropped_leading_bos_like_eos = 0
        dropped_trailing_synthetic_eos = 0

        work_ids = list(gpt2_ids_full)

        if len(work_ids) > 0 and int(work_ids[0]) == eos_token_id:
            work_ids = work_ids[1:]
            dropped_leading_bos_like_eos = 1

        if len(work_ids) > 0 and int(work_ids[-1]) == eos_token_id:
            work_ids = work_ids[:-1]
            dropped_trailing_synthetic_eos = 1

        # Keep internal EOS tokens exactly as recovered.
        gpt2_ids = work_ids

        # For padded tensor output, keep a placeholder EOS if empty.
        # True content length is tracked separately in debug stats.
        padded_row = gpt2_ids if len(gpt2_ids) > 0 else [eos_token_id]

        rows.append(padded_row)
        max_len = max(max_len, len(padded_row))

        if return_debug_stats:
            debug_rows.append(
                {
                    "row_idx": int(i),
                    "code_n_input": int(code_stats["n_input"]),
                    "code_n_kept": int(code_stats["n_kept"]),
                    "code_n_invalid_skipped": int(code_stats["n_invalid_skipped"]),
                    "code_hit_pad": int(code_stats["hit_pad"]),
                    "code_hit_eoseq": int(code_stats["hit_eoseq"]),
                    "code_n_unk_kept": int(code_stats["n_unk_kept"]),
                    "code_stop_idx": int(code_stats["stop_idx_code"]),
                    "pua_len": int(len(pua_string)),
                    "gpt2_len_before_edge_strip": int(len(gpt2_ids_full)),
                    "dropped_leading_bos_like_eos": int(dropped_leading_bos_like_eos),
                    "dropped_trailing_synthetic_eos": int(dropped_trailing_synthetic_eos),
                    "gpt2_len_after_edge_strip": int(len(gpt2_ids)),
                }
            )

    out = torch.full((len(rows), max_len), eos_token_id, dtype=torch.long)
    for i, row in enumerate(rows):
        out[i, : len(row)] = torch.tensor(row, dtype=torch.long)

    if return_debug_stats:
        return out, debug_rows
    return out

@torch.no_grad()
def code_ids_to_gpt2_token_id_lists_for_eval(
    code_ids: torch.Tensor,
    *,
    gpt2_tokenizer: Any,
    code_tokenizer: Any,
    code_meta: Dict[str, Any],
    return_debug_stats: bool = False,
):
    gpt2_ids_padded, debug_rows = code_ids_to_gpt2_token_ids_for_eval(
        code_ids,
        gpt2_tokenizer=gpt2_tokenizer,
        code_tokenizer=code_tokenizer,
        code_meta=code_meta,
        return_debug_stats=True,
    )

    rows: List[List[int]] = []
    n_rows = int(gpt2_ids_padded.size(0))

    for i in range(n_rows):
        true_len = int(debug_rows[i].get("gpt2_len_after_edge_strip", -1))
        if true_len < 0:
            row_ids = gpt2_ids_padded[i].tolist()
        else:
            row_ids = gpt2_ids_padded[i, :true_len].tolist()
        rows.append(row_ids)

    if return_debug_stats:
        return rows, debug_rows
    return rows
    
@torch.no_grad()
def code_ids_to_text_for_eval(
    code_ids: torch.Tensor,
    *,
    gpt2_tokenizer: Any,
    code_tokenizer: Any,
    code_meta: Dict[str, Any],
    return_debug_stats: bool = False,
):
    """
    Decode code-token sequences to text for evaluation.

    Important:
      - code_ids_to_gpt2_token_ids_for_eval returns a padded [B, T_max] tensor
      - we must decode each row only up to its true recovered length, otherwise
        EOS padding leaks into the final text as repeated <|endoftext|>
    """
    if isinstance(code_ids, list):
        code_ids = torch.tensor(code_ids, dtype=torch.long)

    batch_size = 1 if code_ids.dim() == 1 else int(code_ids.size(0))

    gpt2_ids, debug_stats = code_ids_to_gpt2_token_ids_for_eval(
        code_ids,
        gpt2_tokenizer=gpt2_tokenizer,
        code_tokenizer=code_tokenizer,
        code_meta=code_meta,
        return_debug_stats=True,
    )

    if gpt2_ids.numel() == 0:
        texts = [""] * batch_size
        return (texts, debug_stats) if return_debug_stats else texts

    texts: List[str] = []
    n_rows = int(gpt2_ids.size(0))

    for i in range(n_rows):
        row = gpt2_ids[i]

        true_len = None
        if i < len(debug_stats):
            true_len = int(debug_stats[i].get("gpt2_len_after_edge_strip", -1))

        if true_len is None or true_len < 0:
            row_ids = row.tolist()
        else:
            row_ids = row[:true_len].tolist()

        try:
            text = gpt2_tokenizer.decode(row_ids, skip_special_tokens=False)
        except TypeError:
            text = gpt2_tokenizer.decode(row_ids)

        texts.append("" if text is None else str(text))

    if return_debug_stats:
        return texts, debug_stats
    return texts

@torch.no_grad()
def debug_gpt2id_bpe16_decode_batch(
    code_ids: Any,
    *,
    gpt2_tokenizer: Any,
    code_tokenizer: Any,
    code_meta: Dict[str, Any],
    max_rows: int = 8,
    preview_code_tokens: int = 48,
    preview_gpt2_tokens: int = 64,
) -> str:
    def _coerce_to_long_2d(x: Any) -> torch.Tensor:
        if x is None:
            return torch.empty((0, 0), dtype=torch.long)

        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().to(torch.long)

        elif isinstance(x, (list, tuple)):
            if len(x) == 0:
                return torch.empty((0, 0), dtype=torch.long)

            first = x[0]

            if isinstance(first, (int, float, bool)):
                t = torch.tensor([list(x)], dtype=torch.long)

            elif isinstance(first, (list, tuple, torch.Tensor)):
                rows: List[List[int]] = []
                max_len_local = 0

                for row in x:
                    if isinstance(row, torch.Tensor):
                        row_list = row.detach().cpu().view(-1).to(torch.long).tolist()
                    elif isinstance(row, (list, tuple)):
                        row_list = [int(v) for v in row]
                    else:
                        raise TypeError(f"Unsupported row type inside code_ids list: {type(row)}")

                    rows.append(row_list)
                    max_len_local = max(max_len_local, len(row_list))

                if max_len_local == 0:
                    return torch.empty((len(rows), 0), dtype=torch.long)

                t = torch.zeros((len(rows), max_len_local), dtype=torch.long)
                for i, row in enumerate(rows):
                    if len(row) > 0:
                        t[i, : len(row)] = torch.tensor(row, dtype=torch.long)

            else:
                raise TypeError(f"Unsupported code_ids list element type: {type(first)}")

        else:
            raise TypeError(f"Unsupported code_ids type: {type(x)}")

        if t.dim() == 0:
            t = t.view(1, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(0)
        elif t.dim() != 2:
            raise ValueError(
                f"debug_gpt2id_bpe16_decode_batch expects [T] or [B,T], got shape {tuple(t.shape)}"
            )

        return t

    try:
        code_ids_t = _coerce_to_long_2d(code_ids)
    except Exception as e:
        return (
            "debug_gpt2id_bpe16_decode_batch: failed to coerce input to a 2D long tensor.\n"
            f"input_type={type(code_ids)}\n"
            f"error={repr(e)}"
        )

    if code_ids_t.numel() == 0 or int(code_ids_t.size(0)) == 0:
        return "debug_gpt2id_bpe16_decode_batch: empty input."

    shown_rows = min(int(code_ids_t.size(0)), max(1, int(max_rows)))
    code_ids_t = code_ids_t[:shown_rows].contiguous()

    try:
        gpt2_ids, dbg = code_ids_to_gpt2_token_ids_for_eval(
            code_ids_t,
            gpt2_tokenizer=gpt2_tokenizer,
            code_tokenizer=code_tokenizer,
            code_meta=code_meta,
            return_debug_stats=True,
        )
    except Exception as e:
        lines = [
            "debug_gpt2id_bpe16_decode_batch: decode failed inside code_ids_to_gpt2_token_ids_for_eval.",
            f"input_shape={tuple(code_ids_t.shape)}",
            f"input_dtype={code_ids_t.dtype}",
            f"error={repr(e)}",
            "",
            "Raw input preview:",
        ]
        for i in range(shown_rows):
            row = code_ids_t[i].tolist()
            lines.append(f"  sample {i} raw_code_ids[:{preview_code_tokens}]: {row[:preview_code_tokens]}")
        return "\n".join(lines)

    texts: List[str] = []
    for i in range(int(gpt2_ids.size(0))):
        row = gpt2_ids[i]
        true_len = int(dbg[i].get("gpt2_len_after_semantic_trunc", -1))
        row_ids = row[:true_len].tolist() if true_len >= 0 else row.tolist()

        try:
            text = gpt2_tokenizer.decode(row_ids, skip_special_tokens=False)
        except TypeError:
            text = gpt2_tokenizer.decode(row_ids)

        texts.append("" if text is None else str(text))

    lines: List[str] = []
    lines.append(
        f"gpt2id_bpe16 debug dump: batch_shape={tuple(code_ids_t.shape)}, "
        f"shown_rows={shown_rows}"
    )
    lines.append("")

    n_rows = min(shown_rows, len(dbg), len(texts), int(gpt2_ids.size(0)))
    for i in range(n_rows):
        row = code_ids_t[i].tolist()
        row_dbg = dbg[i]

        true_len = int(row_dbg.get("gpt2_len_after_semantic_trunc", -1))
        row_gpt2 = gpt2_ids[i][:true_len].tolist() if true_len >= 0 else gpt2_ids[i].tolist()

        lines.append(f"Sample {i}:")
        lines.append(f"  raw_code_ids[:{preview_code_tokens}]: {row[:preview_code_tokens]}")
        lines.append(
            f"  code_stop: idx={row_dbg.get('code_stop_idx', -1)} "
            f"hit_pad={row_dbg.get('code_hit_pad', 0)} "
            f"hit_eoseq={row_dbg.get('code_hit_eoseq', 0)} "
            f"invalid_skipped={row_dbg.get('code_n_invalid_skipped', 0)} "
            f"unk_kept={row_dbg.get('code_n_unk_kept', 0)}"
        )
        lines.append(
            f"  pua_len={row_dbg.get('pua_len', -1)} "
            f"gpt2_len_before={row_dbg.get('gpt2_len_before_semantic_trunc', -1)} "
            f"raw_first_eos_idx={row_dbg.get('gpt2_raw_first_eos_idx', -1)} "
            f"dropped_leading_bos_like_eos={row_dbg.get('dropped_leading_bos_like_eos', 0)} "
            f"first_content_eos_idx={row_dbg.get('gpt2_first_content_eos_idx', -1)} "
            f"gpt2_len_after={row_dbg.get('gpt2_len_after_semantic_trunc', -1)} "
            f"semantic_trunc_applied={row_dbg.get('semantic_trunc_applied', 0)}"
        )
        lines.append(f"  gpt2_ids[:{preview_gpt2_tokens}]: {row_gpt2[:preview_gpt2_tokens]}")
        lines.append(f"  text: {texts[i]}")
        lines.append("")

    return "\n".join(lines)

@torch.no_grad()
def decode_token_sequences_to_token_ids_for_eval(
    cfg,
    samples_tokens: torch.Tensor,
    *,
    dataset_obj: Optional[Any] = None,
) -> torch.Tensor:
    """
    Convert model token sequences to tokenizer token ids suitable for evaluation.

    Returns a [B, T_eval] long tensor.

    Notes:
      - For ordinary tokenizer-id token spaces, this is basically identity.
      - For semantic-rank token spaces, this maps ranks back to tokenizer ids.
      - For OpenWebText sequence_codec='gpt2id_bpe16', this decodes code ids
        back to GPT-2 token ids.
    """
    data_cfg = cfg.data
    dataset_name_raw = str(getattr(data_cfg, "dataset", ""))
    dataset_name = _norm_ds(dataset_name_raw)
    sequence_codec = str(getattr(data_cfg, "sequence_codec", "base")).lower().strip()

    samples_tokens = _coerce_token_id_tensor(samples_tokens)

    # ---------------------- OWT code-token space ----------------------
    if dataset_name == "openwebtext" and sequence_codec == "gpt2id_bpe16":
        root = Path(getattr(data_cfg, "root", "./datasets/openwebtext"))
        tokenizer_name = str(getattr(data_cfg, "tokenizer_name", "gpt2"))
        code_tokenizer_path = str(getattr(data_cfg, "code_tokenizer_path"))
        code_tokenizer_meta_path = getattr(data_cfg, "code_tokenizer_meta_path", None)

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

        return code_ids_to_gpt2_token_ids_for_eval(
            samples_tokens,
            gpt2_tokenizer=gpt2_tok,
            code_tokenizer=code_tok,
            code_meta=code_meta,
        )

    # ---------------------- Generic token-space branch ----------------------
    ids = samples_tokens.long().cpu()
    token_space = str(getattr(data_cfg, "token_space", "tokenizer_id")).lower()

    if token_space not in {"semantic_rank", "semantic", "rank"}:
        return ids

    # Need semantic inverse map.
    new_to_old = None
    tokenizer = None

    if dataset_obj is not None:
        new_to_old = extract_dataset_attr(dataset_obj, "new_to_old")
        if new_to_old is None:
            new_to_old = extract_dataset_attr(dataset_obj, "new_to_old_map")
        tokenizer = extract_dataset_attr(dataset_obj, "tokenizer")
        if tokenizer is None:
            tokenizer = extract_dataset_attr(dataset_obj, "bpe_tokenizer")

    if dataset_name == "lm1b":
        root = Path(getattr(data_cfg, "root", "./datasets/lm1b"))
        tokenizer_name = str(getattr(data_cfg, "tokenizer_name", "bert-base-uncased"))
        if tokenizer is None:
            tokenizer = load_lm1b_tokenizer_only(root=root, tokenizer_name=tokenizer_name)
        if new_to_old is None:
            _, new_to_old = load_lm1b_semantic_assets(root=root, tokenizer_name=tokenizer_name)

    elif dataset_name in {"openwebtext2", "openwebtext", "wikitext103", "wikitext"}:
        root = Path(getattr(data_cfg, "root", "./datasets/openwebtext"))
        tokenizer_name = str(getattr(data_cfg, "tokenizer_name", "gpt2"))

        if dataset_name == "openwebtext":
            if tokenizer is None:
                tokenizer = load_openwebtext_tokenizer_only(root=root, tokenizer_name=tokenizer_name)
            if new_to_old is None:
                _, new_to_old = load_openwebtext_semantic_assets(root=root, tokenizer_name=tokenizer_name)
        elif "wikitext" in dataset_name:
            if tokenizer is None or new_to_old is None:
                tokenizer, new_to_old = load_wikitext_semantic_assets(root=Path("./datasets/wikitext-103"))
        else:
            if tokenizer is None or new_to_old is None:
                tokenizer, new_to_old = load_owt2_semantic_assets(root=Path("./datasets/openwebtext2"))

    elif dataset_name == "text8":
        if tokenizer is None or new_to_old is None:
            tokenizer, new_to_old = load_text8_semantic_assets()

    if new_to_old is None:
        raise RuntimeError(
            f"decode_token_sequences_to_token_ids_for_eval: missing new_to_old map for dataset={dataset_name_raw}"
        )

    unk_id = _get_unk_id(tokenizer, fallback=1) if tokenizer is not None else 1
    return _semantic_ranks_to_token_ids(ids, new_to_old, unk_id)

# -----------------------------------------------------------------------------
# Legacy scalar semantic decoder
# -----------------------------------------------------------------------------
@torch.no_grad()
def bits_to_text_semantic_generic(
    bits_1d: torch.Tensor,
    tokenizer: Any,
    new_to_old_map: Dict[int, int],
    *,
    bits_per_token: int,
    cfg: Optional[Any] = None,
    highlight_ecc_corrections: bool = False,
    prompt_limit_bits: int = 0,
    normalize_text8: bool = False,
) -> str:
    del highlight_ecc_corrections, prompt_limit_bits

    bits_1d = bits_1d.to(torch.long).view(-1)
    S = int(bits_1d.numel())
    if S <= 0:
        return ""

    ecc = ecc_from_cfg(cfg) if cfg is not None else None
    use_ecc = bool(ecc is not None and ecc.enabled)

    unk_id = _get_unk_id(tokenizer, fallback=1)
    token_ids: List[int] = []

    if use_ecc:
        L = ecc_chunk_len(ecc)
        if S < L:
            return ""
        S = (S // L) * L
        bits_1d = bits_1d[:S]

        for i in range(0, S, L):
            chunk = bits_1d[i : i + L]
            data_bits, _, uncorrectable = ecc_decode_token_bits(chunk.unsqueeze(0), ecc)

            is_bad = bool(uncorrectable.item() if isinstance(uncorrectable, torch.Tensor) else uncorrectable)
            if is_bad:
                token_ids.append(int(unk_id))
                continue

            db = data_bits.view(-1)
            gray_val = 0
            for b in db.tolist():
                gray_val = (gray_val << 1) | int(b)

            rank = gray_to_int(gray_val)
            tid = new_to_old_map.get(rank, None)
            token_ids.append(int(tid) if tid is not None else int(unk_id))
    else:
        if S < bits_per_token:
            return ""
        S = (S // bits_per_token) * bits_per_token
        bits_1d = bits_1d[:S]

        for i in range(0, S, bits_per_token):
            chunk = bits_1d[i : i + bits_per_token]
            gray_val = 0
            for b in chunk.tolist():
                gray_val = (gray_val << 1) | int(b)

            rank = gray_to_int(gray_val)
            tid = new_to_old_map.get(rank, None)
            token_ids.append(int(tid) if tid is not None else int(unk_id))

    text = tokenizer.decode(token_ids, skip_special_tokens=False) if token_ids else ""

    if normalize_text8:
        text = text.replace("_", " ")
    return text


# -----------------------------------------------------------------------------
# Asset loaders
# -----------------------------------------------------------------------------
def load_text8_semantic_assets(
    root: Path = Path("./datasets/text8"),
) -> Tuple[Any, Dict[int, int]]:
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise ImportError("Text8 semantic decoding requires `pip install tokenizers`.") from e

    tok_path = root / "tokenizer_bpe_16k.json"
    map_path = root / "semantic_mapping_16k.json"
    if not tok_path.exists() or not map_path.exists():
        raise RuntimeError(f"Missing Text8 semantic assets under {root}.")

    tokenizer = Tokenizer.from_file(str(tok_path))
    with open(map_path, "r") as f:
        maps = json.load(f)
    new_to_old = {int(k): int(v) for k, v in maps["new_to_old"].items()}
    return tokenizer, new_to_old


def load_wikitext_semantic_assets(
    root: Path = Path("./datasets/wikitext-103"),
) -> Tuple[Any, Dict[int, int]]:
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise ImportError("WikiText semantic decoding requires `pip install tokenizers`.") from e

    tok_path = root / "tokenizer_wiki_65k.json"
    map_path = root / "semantic_mapping_wiki_65k.json"
    if not tok_path.exists() or not map_path.exists():
        raise RuntimeError(f"Missing WikiText semantic assets under {root}.")

    tokenizer = Tokenizer.from_file(str(tok_path))
    with open(map_path, "r") as f:
        maps = json.load(f)
    new_to_old = {int(k): int(v) for k, v in maps["new_to_old"].items()}
    return tokenizer, new_to_old


def load_owt2_semantic_assets(
    root: Path = Path("./datasets/openwebtext2"),
) -> Tuple[Any, Dict[int, int]]:
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise ImportError("OpenWebText2 semantic decoding requires `pip install tokenizers`.") from e

    tok_path = root / "tokenizer_bpe_32k.json"
    map_path = root / "semantic_mapping_32k.json"
    if not tok_path.exists() or not map_path.exists():
        raise RuntimeError(f"Missing OpenWebText2 semantic assets under {root}.")

    tokenizer = Tokenizer.from_file(str(tok_path))
    with open(map_path, "r") as f:
        maps = json.load(f)
    new_to_old = {int(k): int(v) for k, v in maps["new_to_old"].items()}
    return tokenizer, new_to_old


def load_lm1b_tokenizer_only(
    root: Path = Path("./datasets/lm1b"),
    tokenizer_name: str = "bert-base-uncased",
):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError("LM1B decoding requires `pip install transformers`.") from e

    del root
    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def load_lm1b_semantic_assets(
    root: Path = Path("./datasets/lm1b"),
    tokenizer_name: str = "bert-base-uncased",
) -> Tuple[Any, Dict[int, int]]:
    tokenizer = load_lm1b_tokenizer_only(root=root, tokenizer_name=tokenizer_name)

    map_path = root / "semantic_mapping_bert_base_uncased.json"
    if not map_path.exists():
        raise RuntimeError(f"Missing LM1B semantic map under {root}: {map_path}")

    with open(map_path, "r") as f:
        maps = json.load(f)

    new_to_old = {int(k): int(v) for k, v in maps["new_to_old"].items()}
    return tokenizer, new_to_old


def load_openwebtext_tokenizer_only(
    root: Path = Path("./datasets/openwebtext"),
    tokenizer_name: str = "gpt2",
):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError("OpenWebText decoding requires `pip install transformers`.") from e

    del root
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_openwebtext_semantic_assets(
    root: Path = Path("./datasets/openwebtext"),
    tokenizer_name: str = "gpt2",
) -> Tuple[Any, Dict[int, int]]:
    tokenizer = load_openwebtext_tokenizer_only(root=root, tokenizer_name=tokenizer_name)

    map_path = root / f"semantic_mapping_{_safe_name(tokenizer_name)}.json"
    if not map_path.exists():
        raise RuntimeError(f"Missing OpenWebText semantic map under {root}: {map_path}")

    with open(map_path, "r", encoding="utf-8") as f:
        maps = json.load(f)

    new_to_old = {int(k): int(v) for k, v in maps["new_to_old"].items()}
    return tokenizer, new_to_old


def normalize_text8_for_char_lm(s: str) -> str:
    s = s.lower().replace("_", " ")
    out: List[str] = []
    for ch in s:
        if ch == " " or ("a" <= ch <= "z"):
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


# -----------------------------------------------------------------------------
# Pretty-print helpers
# -----------------------------------------------------------------------------
def bits_preview(bits_row: torch.Tensor, *, max_bits: int = 128) -> str:
    b = bits_row.detach().cpu().to(torch.long).view(-1)
    s = "".join(str(int(x)) for x in b[:max_bits].tolist())
    if b.numel() > max_bits:
        s += "..."
    return f"{s} (len={b.numel()})"


def semantic_token_bit_chunks(
    bits_row: torch.Tensor,
    *,
    bits_per_token: int,
    max_tokens: int = 8,
) -> str:
    b = bits_row.detach().cpu().to(torch.long).view(-1)
    T = min(max_tokens, b.numel() // bits_per_token)
    chunks = []
    for t in range(T):
        chunk = b[t * bits_per_token : (t + 1) * bits_per_token]
        chunks.append("".join(str(int(x)) for x in chunk.tolist()))
    suffix = " ..." if (b.numel() // bits_per_token) > T else ""
    return " | ".join(chunks) + suffix


@torch.no_grad()
def format_decoded_block(
    cfg,
    samples: torch.Tensor,
    *,
    dataset_obj: Optional[Any],
    tag: str,
    epoch: int,
    max_samples: int = 8,
    include_token_ids: bool = True,
    include_bits: bool = True,
    bits_preview_len: int = 128,
    semantic_bits_per_token_default: int = 14,
    normalize_text8: bool = True,
    prefix_bits: Optional[torch.Tensor] = None,
    prefix_len_bits: int = 0,
    prefix_mask: Optional[torch.Tensor] = None,
    prompt_len_tokens: Optional[torch.Tensor] = None,
    bold_prefix: bool = True,
    show_prompt_suffix_lines: bool = True,
    highlight_ecc_corrections: bool = False,
) -> str:
    del prefix_bits, highlight_ecc_corrections

    repr_mode = str(getattr(cfg.data, "representation", "tokens")).lower()
    if repr_mode == "tokens":
        samples = _coerce_token_id_tensor(samples)
    elif samples.dim() == 1:
        samples = samples.unsqueeze(0)

    B = int(samples.size(0))
    n = min(B, int(max_samples))

    data_cfg = cfg.data
    binarization = str(getattr(data_cfg, "binarization", "")).lower()

    prompt_lens_bits = torch.zeros(n, dtype=torch.long, device=samples.device)
    if prefix_mask is not None:
        if prefix_mask.dim() == 1:
            prompt_lens_bits[:] = int(prefix_mask.sum().item())
        else:
            prompt_lens_bits[:] = prefix_mask[:n].sum(dim=1)
    elif int(prefix_len_bits) > 0:
        prompt_lens_bits[:] = int(prefix_len_bits)

    prompt_lens_tokens = torch.zeros(n, dtype=torch.long, device=samples.device)
    if prompt_len_tokens is not None:
        pt = prompt_len_tokens.view(-1)
        if pt.numel() == 1 and n > 1:
            pt = pt.expand(n)
        prompt_lens_tokens[: min(n, pt.numel())] = pt[: min(n, pt.numel())].to(samples.device)

    decoded_full = decode_samples_to_text(
        cfg,
        samples[:n],
        dataset_obj=dataset_obj,
        normalize_text8=normalize_text8,
        highlight_ecc_corrections=False,
        prompt_lens_bits=None,
    )

    lines: List[str] = []
    for i in range(n):
        lines.append(f"Sample {i}:")

        if repr_mode == "binary":
            bits_row = samples[i].cpu()
            if bits_row.is_floating_point():
                bits_row = (bits_row > 0.5).long()
            bits_row = bits_row.view(-1)

            if include_bits:
                lines.append(f"  bits: {bits_preview(bits_row, max_bits=bits_preview_len)}")
                if binarization == "semantic":
                    ecc = ecc_from_cfg(cfg)
                    if ecc.enabled:
                        bpt = int(ecc_chunk_len(ecc))
                    else:
                        bpt = getattr(data_cfg, "bits_per_token", None)
                        if bpt is None:
                            bpt = int(getattr(data_cfg, "bits_per_char", semantic_bits_per_token_default))
                        else:
                            bpt = int(bpt)
                    if bpt > 0:
                        lines.append(
                            f"  semantic chunks ({bpt} bits/token): "
                            f"{semantic_token_bit_chunks(bits_row, bits_per_token=bpt, max_tokens=8)}"
                        )

            full_text = decoded_full[i]
            cL_i = int(prompt_lens_bits[i].item())

            if cL_i > 0:
                prompt_bits_i = bits_row[:cL_i]
                prompt_text = decode_samples_to_text(
                    cfg,
                    prompt_bits_i.unsqueeze(0),
                    dataset_obj=dataset_obj,
                    normalize_text8=normalize_text8,
                )[0]

                if isinstance(full_text, str) and isinstance(prompt_text, str) and full_text.startswith(prompt_text):
                    suffix_text = full_text[len(prompt_text):]
                else:
                    suffix_bits = bits_row[cL_i:]
                    suffix_text = decode_samples_to_text(
                        cfg,
                        suffix_bits.unsqueeze(0),
                        dataset_obj=dataset_obj,
                        normalize_text8=normalize_text8,
                    )[0]

                if bold_prefix and prompt_text:
                    stripped_prompt = prompt_text.lstrip()
                    leading_space = prompt_text[: len(prompt_text) - len(stripped_prompt)]
                    combined_markdown = f"{leading_space}**{stripped_prompt}**{suffix_text}"
                else:
                    combined_markdown = f"{prompt_text}{suffix_text}"

                if show_prompt_suffix_lines:
                    lines.append(f"  prompt: {repr(prompt_text)}")
                    lines.append(f"  text: {combined_markdown}")
                else:
                    lines.append(f"  text: {combined_markdown}")
            else:
                lines.append(f"  text: {full_text}")

            lines.append("")

        else:
            row = _coerce_token_id_tensor(samples[i]).view(-1).detach().cpu().to(torch.long)
            full_text = decoded_full[i]
            cL_tok = int(prompt_lens_tokens[i].item())

            if include_token_ids:
                lines.append(f"  tokens: {row.tolist()}")

            if cL_tok > 0:
                prompt_tokens_i = row[:cL_tok]
                prompt_text = decode_samples_to_text(
                    cfg,
                    prompt_tokens_i.unsqueeze(0),
                    dataset_obj=dataset_obj,
                    normalize_text8=normalize_text8,
                )[0]

                if isinstance(full_text, str) and isinstance(prompt_text, str) and full_text.startswith(prompt_text):
                    suffix_text = full_text[len(prompt_text):]
                else:
                    suffix_tokens_i = row[cL_tok:]
                    suffix_text = decode_samples_to_text(
                        cfg,
                        suffix_tokens_i.unsqueeze(0),
                        dataset_obj=dataset_obj,
                        normalize_text8=normalize_text8,
                    )[0]

                if bold_prefix and prompt_text:
                    stripped_prompt = prompt_text.lstrip()
                    leading_space = prompt_text[: len(prompt_text) - len(stripped_prompt)]
                    combined_markdown = f"{leading_space}**{stripped_prompt}**{suffix_text}"
                else:
                    combined_markdown = f"{prompt_text}{suffix_text}"

                if show_prompt_suffix_lines:
                    lines.append(f"  prompt: {repr(prompt_text)}")
                    lines.append(f"  text: {combined_markdown}")
                else:
                    lines.append(f"  text: {combined_markdown}")
            else:
                lines.append(f"  text: {repr(full_text)}")

            lines.append("")

    header = f"[{tag}] epoch={epoch}"
    return header + "\n\n" + "\n".join(lines)


# -----------------------------------------------------------------------------
# Token-space decoding helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def _semantic_ranks_to_token_ids(
    ranks: torch.Tensor,
    new_to_old: Dict[int, int],
    unk_id: int,
) -> torch.Tensor:
    ranks = ranks.long().cpu()
    if len(new_to_old) == 0:
        return torch.full_like(ranks, int(unk_id))

    max_rank = max(int(k) for k in new_to_old.keys())
    table = torch.full((max_rank + 2,), int(unk_id), dtype=torch.long)
    k = torch.tensor(list(new_to_old.keys()), dtype=torch.long)
    v = torch.tensor(list(new_to_old.values()), dtype=torch.long)
    table[k] = v
    ranks = ranks.clamp(0, max_rank + 1)
    return table[ranks]


@torch.no_grad()
def decode_token_sequences_for_eval(
    cfg,
    samples_tokens: torch.Tensor,
    *,
    prompt_len_tokens: Optional[torch.Tensor] = None,
    mode: str = "full",
    dataset_obj: Optional[Any] = None,
    normalize_text8: bool = True,
) -> List[str]:
    mode = str(mode).lower().strip()
    if mode == "prefix":
        mode = "prompt"

    if mode not in {"full", "suffix", "prompt"}:
        raise ValueError(
            f"decode_token_sequences_for_eval: mode must be 'full', 'suffix', 'prompt', or 'prefix' "
            f"(got {mode!r})"
        )

    samples_tokens = _coerce_token_id_tensor(samples_tokens)

    decoded_full = decode_samples_to_text(
        cfg,
        samples_tokens,
        dataset_obj=dataset_obj,
        normalize_text8=normalize_text8,
    )

    if mode == "full":
        return decoded_full

    if prompt_len_tokens is None:
        raise ValueError("decode_token_sequences_for_eval(mode='suffix'/'prompt') requires prompt_len_tokens")

    B = int(samples_tokens.size(0))
    if isinstance(prompt_len_tokens, int):
        pl = torch.full((B,), int(prompt_len_tokens), dtype=torch.long)
    else:
        pl = prompt_len_tokens.detach().view(-1).to(dtype=torch.long).cpu()
        if pl.numel() == 1 and B > 1:
            pl = pl.expand(B)

    out: List[str] = []
    for i in range(B):
        cL = int(pl[i].item())
        if cL <= 0:
            out.append("" if mode == "prompt" else decoded_full[i])
            continue

        prompt_tokens = samples_tokens[i, :cL]
        prompt_text = decode_samples_to_text(
            cfg,
            prompt_tokens.unsqueeze(0),
            dataset_obj=dataset_obj,
            normalize_text8=normalize_text8,
        )[0]

        if mode == "prompt":
            out.append(prompt_text)
            continue

        full_text = decoded_full[i]
        if isinstance(full_text, str) and isinstance(prompt_text, str) and full_text.startswith(prompt_text):
            out.append(full_text[len(prompt_text):])
        else:
            suffix_tokens = samples_tokens[i, cL:]
            suffix_text = decode_samples_to_text(
                cfg,
                suffix_tokens.unsqueeze(0),
                dataset_obj=dataset_obj,
                normalize_text8=normalize_text8,
            )[0]
            out.append(suffix_text)

    return out


@torch.no_grad()
def decode_samples_to_text(
    cfg,
    samples: torch.Tensor,
    *,
    dataset_obj: Optional[Any] = None,
    normalize_text8: bool = True,
    highlight_ecc_corrections: bool = False,
    prompt_lens_bits: Optional[torch.Tensor] = None,
) -> List[str]:
    del highlight_ecc_corrections, prompt_lens_bits

    data_cfg = cfg.data
    dataset_name_raw = str(getattr(data_cfg, "dataset", ""))
    dataset_name = _norm_ds(dataset_name_raw)
    repr_mode = str(getattr(data_cfg, "representation", "tokens")).lower()
    sequence_codec = str(getattr(data_cfg, "sequence_codec", "base")).lower().strip()

    if repr_mode == "tokens":
        samples = _coerce_token_id_tensor(samples)
    elif samples.dim() == 1:
        samples = samples.unsqueeze(0)

    bits: Optional[torch.Tensor] = None
    if repr_mode == "binary":
        if samples.is_floating_point():
            bits = (samples.clamp(0, 1) > 0.5).long()
        else:
            bits = (samples != 0).long()

    # ---------------------- TEXT8 ----------------------
    if dataset_name == "text8":
        from data.text8 import (
            bits_to_ascii_string,
            bits_to_text_fixed,
            bits_to_text_huffman,
            token_ids_to_text,
        )

        if repr_mode == "tokens":
            ids = samples.long().cpu()
            texts = [token_ids_to_text(ids[i]) for i in range(ids.size(0))]
            return [normalize_text8_for_char_lm(t) for t in texts] if normalize_text8 else texts

        binarization = str(getattr(data_cfg, "binarization", "ascii")).lower()

        if binarization == "semantic":
            tokenizer = None
            new_to_old = None
            if dataset_obj is not None:
                tokenizer = extract_dataset_attr(dataset_obj, "tokenizer")
                new_to_old = extract_dataset_attr(dataset_obj, "new_to_old")

            if tokenizer is None or new_to_old is None:
                tokenizer, new_to_old = load_text8_semantic_assets()

            bpt = getattr(data_cfg, "bits_per_token", None)
            if bpt is None:
                bpt = int(getattr(data_cfg, "bits_per_char", 14))
            else:
                bpt = int(bpt)

            return bits_to_text_semantic_vectorized(
                bits.cpu(),
                tokenizer,
                new_to_old,
                bits_per_token=bpt,
                cfg=cfg,
                normalize_text8=normalize_text8,
            )

        bits_per_char = int(getattr(data_cfg, "bits_per_char", 8))
        assert bits is not None

        if binarization == "ascii":
            texts = [bits_to_ascii_string(bits[i].cpu(), bits_per_char=bits_per_char) for i in range(bits.size(0))]
            return [normalize_text8_for_char_lm(t) for t in texts] if normalize_text8 else texts

        if binarization == "fixed5":
            texts = [bits_to_text_fixed(bits[i].cpu(), bits_per_char=bits_per_char) for i in range(bits.size(0))]
            return [normalize_text8_for_char_lm(t) for t in texts] if normalize_text8 else texts

        if binarization == "huffman":
            codes = None
            if dataset_obj is not None:
                codes = extract_dataset_attr(dataset_obj, "huffman_codes")
            if codes is None:
                code_path = Path("./datasets/text8") / "huffman_codes.json"
                if not code_path.exists():
                    raise RuntimeError("Text8 Huffman codes not found.")
                with open(code_path, "r") as f:
                    codes = json.load(f)

            texts = [bits_to_text_huffman(bits[i].cpu(), codes) for i in range(bits.size(0))]
            return [normalize_text8_for_char_lm(t) for t in texts] if normalize_text8 else texts

        raise ValueError(f"Unsupported Text8 binarization: {binarization}")

    # ---------------------- WIKITEXT / OWT2 / OWT ----------------------
    if dataset_name in {
        "openwebtext2",
        "openwebtext",
        "openwebtext",
        "wikitext103",
        "wikitext",
    }:
        tokenizer = None
        new_to_old = None
        if dataset_obj is not None:
            tokenizer = extract_dataset_attr(dataset_obj, "tokenizer")
            new_to_old = extract_dataset_attr(dataset_obj, "new_to_old")
            if new_to_old is None:
                new_to_old = extract_dataset_attr(dataset_obj, "new_to_old_map")
            if tokenizer is None:
                tokenizer = extract_dataset_attr(dataset_obj, "bpe_tokenizer")

        root = Path(getattr(data_cfg, "root", "./datasets/openwebtext"))
        tokenizer_name = str(getattr(data_cfg, "tokenizer_name", "gpt2"))

        if dataset_name == "openwebtext" and sequence_codec == "gpt2id_bpe16":
            code_tokenizer_path = str(getattr(data_cfg, "code_tokenizer_path"))
            code_tokenizer_meta_path = getattr(data_cfg, "code_tokenizer_meta_path", None)

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

            if repr_mode == "tokens":
                code_ids = _coerce_token_id_tensor(samples)
            else:
                assert bits is not None
                code_ids = bitstreams_to_token_ids_raw_binary(
                    bits,
                    bits_per_token=int(getattr(data_cfg, "bits_per_token", 16)),
                    cfg=cfg,
                )

            return code_ids_to_text_for_eval(
                code_ids,
                gpt2_tokenizer=gpt2_tok,
                code_tokenizer=code_tok,
                code_meta=code_meta,
            )

        if repr_mode == "tokens":
            if tokenizer is None:
                if dataset_name == "openwebtext":
                    tokenizer = load_openwebtext_tokenizer_only(root=root, tokenizer_name=tokenizer_name)
                elif "wikitext" in dataset_name:
                    tokenizer, _ = load_wikitext_semantic_assets(root=Path("./datasets/wikitext-103"))
                else:
                    tokenizer, _ = load_owt2_semantic_assets(root=Path("./datasets/openwebtext2"))

            token_space = str(getattr(data_cfg, "token_space", "tokenizer_id")).lower()
            ids = samples.long().cpu()

            if token_space in {"semantic_rank", "semantic", "rank"}:
                if new_to_old is None:
                    if dataset_name == "openwebtext":
                        _, new_to_old = load_openwebtext_semantic_assets(
                            root=root,
                            tokenizer_name=tokenizer_name,
                        )
                    elif "wikitext" in dataset_name:
                        _, new_to_old = load_wikitext_semantic_assets(root=Path("./datasets/wikitext-103"))
                    else:
                        _, new_to_old = load_owt2_semantic_assets(root=Path("./datasets/openwebtext2"))

                unk_id = _get_unk_id(tokenizer, fallback=1)
                ids = _semantic_ranks_to_token_ids(ids, new_to_old, unk_id)

            return _tokenizer_batch_decode(tokenizer, ids.tolist(), skip_special_tokens=False)

        assert bits is not None
        binarization = str(getattr(data_cfg, "binarization", "semantic")).lower()
        bpt = getattr(data_cfg, "bits_per_token", None)

        if bpt is None:
            bpt = 16 if dataset_name == "openwebtext" else 15
        else:
            bpt = int(bpt)

        if tokenizer is None:
            if dataset_name == "openwebtext":
                if binarization == "semantic":
                    tokenizer, new_to_old = load_openwebtext_semantic_assets(
                        root=root,
                        tokenizer_name=tokenizer_name,
                    )
                else:
                    tokenizer = load_openwebtext_tokenizer_only(
                        root=root,
                        tokenizer_name=tokenizer_name,
                    )
            elif "wikitext" in dataset_name:
                tokenizer, new_to_old = load_wikitext_semantic_assets(root=Path("./datasets/wikitext-103"))
            else:
                tokenizer, new_to_old = load_owt2_semantic_assets(root=Path("./datasets/openwebtext2"))

        if binarization == "semantic":
            if new_to_old is None:
                if dataset_name == "openwebtext":
                    _, new_to_old = load_openwebtext_semantic_assets(
                        root=root,
                        tokenizer_name=tokenizer_name,
                    )
                elif "wikitext" in dataset_name:
                    _, new_to_old = load_wikitext_semantic_assets(root=Path("./datasets/wikitext-103"))
                else:
                    _, new_to_old = load_owt2_semantic_assets(root=Path("./datasets/openwebtext2"))

            return bits_to_text_semantic_vectorized(
                bits.cpu(),
                tokenizer,
                new_to_old,
                bits_per_token=bpt,
                cfg=cfg,
                normalize_text8=False,
            )

        if binarization == "raw_binary":
            return bits_to_text_raw_binary_vectorized(
                bits.cpu(),
                tokenizer,
                bits_per_token=bpt,
                cfg=cfg,
            )

        raise ValueError(f"Unsupported OpenWebText-style binarization: {binarization}")

    # ---------------------- LM1B ----------------------
    if dataset_name == "lm1b":
        tokenizer = None
        new_to_old = None

        if dataset_obj is not None:
            tokenizer = extract_dataset_attr(dataset_obj, "tokenizer")
            new_to_old = extract_dataset_attr(dataset_obj, "new_to_old")

        tokenizer_name = str(getattr(data_cfg, "tokenizer_name", "bert-base-uncased"))
        root = Path(getattr(data_cfg, "root", "./datasets/lm1b"))

        if repr_mode == "tokens":
            if tokenizer is None:
                tokenizer = load_lm1b_tokenizer_only(root=root, tokenizer_name=tokenizer_name)

            token_space = str(getattr(data_cfg, "token_space", "tokenizer_id")).lower()
            ids = samples.long().cpu()

            if token_space in {"semantic_rank", "semantic", "rank"}:
                if new_to_old is None:
                    _, new_to_old = load_lm1b_semantic_assets(root=root, tokenizer_name=tokenizer_name)
                unk_id = _get_unk_id(tokenizer, fallback=1)
                ids = _semantic_ranks_to_token_ids(ids, new_to_old, unk_id)

            return _tokenizer_batch_decode(tokenizer, ids.tolist(), skip_special_tokens=False)

        assert bits is not None
        binarization = str(getattr(data_cfg, "binarization", "semantic")).lower()
        bpt = getattr(data_cfg, "bits_per_token", None)
        bpt = 15 if bpt is None else int(bpt)

        if tokenizer is None:
            if binarization == "semantic":
                tokenizer, new_to_old = load_lm1b_semantic_assets(root=root, tokenizer_name=tokenizer_name)
            else:
                tokenizer = load_lm1b_tokenizer_only(root=root, tokenizer_name=tokenizer_name)

        if binarization == "semantic":
            if new_to_old is None:
                _, new_to_old = load_lm1b_semantic_assets(root=root, tokenizer_name=tokenizer_name)

            return bits_to_text_semantic_vectorized(
                bits.cpu(),
                tokenizer,
                new_to_old,
                bits_per_token=bpt,
                cfg=cfg,
                normalize_text8=False,
            )

        if binarization == "raw_binary":
            return bits_to_text_raw_binary_vectorized(
                bits.cpu(),
                tokenizer,
                bits_per_token=bpt,
                cfg=cfg,
            )

        raise ValueError(f"Unsupported LM1B binarization: {binarization}")

    raise ValueError(f"Unsupported dataset for decode_samples_to_text: {dataset_name_raw}")


@torch.no_grad()
def decode_bitstreams_to_token_ids_for_eval(
    cfg,
    samples_bits: torch.Tensor,
    *,
    dataset_obj: Optional[Any] = None,
) -> torch.Tensor:
    """
    Decode model bitstreams to tokenizer token ids and return a [B, T] long tensor.

    This is meant for FLM-style sample entropy computation, which operates on
    token ids before any text decoding / normalization.
    """
    if samples_bits.dim() == 1:
        samples_bits = samples_bits.unsqueeze(0)
    if samples_bits.dim() != 2:
        raise ValueError(
            f"decode_bitstreams_to_token_ids_for_eval expects [B,S] or [S], got {tuple(samples_bits.shape)}"
        )

    data_cfg = cfg.data
    dataset_name_raw = str(getattr(data_cfg, "dataset", ""))
    dataset_name = _norm_ds(dataset_name_raw)
    binarization = str(getattr(data_cfg, "binarization", "raw_binary")).lower()
    sequence_codec = str(getattr(data_cfg, "sequence_codec", "base")).lower().strip()

    if dataset_name not in {
        "openwebtext",
        "openwebtext",
        "openwebtext2",
        "wikitext103",
        "wikitext",
        "lm1b",
    }:
        raise ValueError(f"Unsupported dataset for token-id decoding: {dataset_name_raw}")

    tokenizer = None
    new_to_old = None
    if dataset_obj is not None:
        tokenizer = extract_dataset_attr(dataset_obj, "tokenizer")
        new_to_old = extract_dataset_attr(dataset_obj, "new_to_old")
        if new_to_old is None:
            new_to_old = extract_dataset_attr(dataset_obj, "new_to_old_map")
        if tokenizer is None:
            tokenizer = extract_dataset_attr(dataset_obj, "bpe_tokenizer")

    root = Path(getattr(data_cfg, "root", "./datasets/openwebtext"))
    tokenizer_name = str(getattr(data_cfg, "tokenizer_name", "gpt2"))

    if dataset_name == "openwebtext":
        if sequence_codec == "gpt2id_bpe16":
            code_tokenizer_path = str(getattr(data_cfg, "code_tokenizer_path"))
            code_tokenizer_meta_path = getattr(data_cfg, "code_tokenizer_meta_path", None)

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

            code_ids = bitstreams_to_token_ids_raw_binary(
                samples_bits,
                bits_per_token=int(getattr(data_cfg, "bits_per_token", 16)),
                cfg=cfg,
            )

            return code_ids_to_gpt2_token_ids_for_eval(
                code_ids,
                gpt2_tokenizer=gpt2_tok,
                code_tokenizer=code_tok,
                code_meta=code_meta,
            )

        if tokenizer is None:
            tokenizer = load_openwebtext_tokenizer_only(root=root, tokenizer_name=tokenizer_name)
        if binarization == "semantic" and new_to_old is None:
            _, new_to_old = load_openwebtext_semantic_assets(root=root, tokenizer_name=tokenizer_name)

    elif dataset_name == "lm1b":
        if tokenizer is None:
            tokenizer = load_lm1b_tokenizer_only(root=root, tokenizer_name=tokenizer_name)
        if binarization == "semantic" and new_to_old is None:
            _, new_to_old = load_lm1b_semantic_assets(root=root, tokenizer_name=tokenizer_name)

    elif "wikitext" in dataset_name:
        if tokenizer is None:
            tokenizer, maybe_map = load_wikitext_semantic_assets(root=Path("./datasets/wikitext-103"))
            if new_to_old is None:
                new_to_old = maybe_map

    else:
        if tokenizer is None:
            tokenizer, maybe_map = load_owt2_semantic_assets(root=Path("./datasets/openwebtext2"))
            if new_to_old is None:
                new_to_old = maybe_map

    ecc = ecc_from_cfg(cfg)
    use_ecc = bool(ecc is not None and ecc.enabled)

    bits = samples_bits.long()

    if use_ecc:
        data_bits_flat, _, uncorrectable_mask = ecc_decode_batch_bitstream(bits, ecc)
        m = int(ecc.data_bits)
        B = int(bits.size(0))
        T = data_bits_flat.size(1) // m
        data_bits = data_bits_flat.view(B, T, m)
    else:
        bpt = getattr(data_cfg, "bits_per_token", None)
        if bpt is None:
            raise ValueError("cfg.data.bits_per_token must be set for non-ECC bit decoding")
        m = int(bpt)
        B, S_total = bits.shape
        T = S_total // m
        data_bits = bits[:, : T * m].view(B, T, m)
        uncorrectable_mask = None

    device = data_bits.device
    powers = 2 ** torch.arange(m - 1, -1, -1, device=device)
    unk_id = _get_unk_id(tokenizer, fallback=1)

    if binarization == "raw_binary":
        token_ids = (data_bits * powers).sum(dim=-1).to(torch.long)
        vocab_size = _get_tokenizer_vocab_size(tokenizer)
        if vocab_size is not None and vocab_size > 0:
            # --- PATCHED: Modulo wrap-around for redundancy ---
            token_ids = token_ids % int(vocab_size)

    elif binarization == "semantic":
        gray_vals = (data_bits * powers).sum(dim=-1).to(torch.long)
        ranks = gray_to_int_vectorized(gray_vals)

        if not new_to_old:
            raise RuntimeError("semantic bit decoding requested but new_to_old map is missing")

        max_rank = max(int(k) for k in new_to_old.keys())
        lookup_table = torch.full((max_rank + 2,), int(unk_id), dtype=torch.long, device=device)
        k_t = torch.tensor(list(new_to_old.keys()), dtype=torch.long, device=device)
        v_t = torch.tensor(list(new_to_old.values()), dtype=torch.long, device=device)
        lookup_table[k_t] = v_t

        ranks = ranks.clamp(0, max_rank + 1)
        token_ids = lookup_table[ranks]

    else:
        raise ValueError(f"Unsupported binarization for token-id decoding: {binarization}")

    if use_ecc and uncorrectable_mask is not None:
        token_ids[uncorrectable_mask] = int(unk_id)

    return token_ids.cpu()
    
@torch.no_grad()
def decode_bitstreams_for_eval(
    cfg,
    samples_bits: torch.Tensor,
    *,
    prompt_len_bits: Optional[torch.Tensor] = None,
    mode: str = "full",
    dataset_obj: Optional[Any] = None,
    normalize_text8: bool = True,
) -> List[str]:
    mode = str(mode).lower().strip()
    if mode == "prefix":
        mode = "prompt"

    if mode not in {"full", "suffix", "prompt"}:
        raise ValueError(
            f"decode_bitstreams_for_eval: mode must be 'full', 'suffix', 'prompt', or 'prefix' "
            f"(got {mode!r})"
        )

    if samples_bits.dim() == 1:
        samples_bits = samples_bits.unsqueeze(0)
    if samples_bits.dim() != 2:
        raise ValueError(f"decode_bitstreams_for_eval expects [B,S] or [S], got {tuple(samples_bits.shape)}")

    B = int(samples_bits.size(0))
    decoded_full = decode_samples_to_text(
        cfg,
        samples_bits,
        dataset_obj=dataset_obj,
        normalize_text8=normalize_text8,
        highlight_ecc_corrections=False,
        prompt_lens_bits=None,
    )

    if mode == "full":
        return decoded_full

    if prompt_len_bits is None:
        raise ValueError("decode_bitstreams_for_eval(mode='suffix'/'prompt') requires prompt_len_bits.")

    if isinstance(prompt_len_bits, int):
        pl = torch.full((B,), int(prompt_len_bits), dtype=torch.long)
    elif isinstance(prompt_len_bits, torch.Tensor):
        pl = prompt_len_bits.detach().view(-1).to(dtype=torch.long).cpu()
        if pl.numel() == 1 and B > 1:
            pl = pl.expand(B)
    else:
        raise TypeError(f"prompt_len_bits must be int or torch.Tensor, got {type(prompt_len_bits)}")

    out: List[str] = []
    for i in range(B):
        full_text = decoded_full[i]
        cL = int(pl[i].item())

        if cL <= 0:
            out.append("" if mode == "prompt" else full_text)
            continue

        bits_prompt = samples_bits[i, :cL]
        prompt_text = decode_samples_to_text(
            cfg,
            bits_prompt.unsqueeze(0),
            dataset_obj=dataset_obj,
            normalize_text8=normalize_text8,
            highlight_ecc_corrections=False,
            prompt_lens_bits=None,
        )[0]

        if mode == "prompt":
            out.append(prompt_text)
            continue

        if isinstance(full_text, str) and isinstance(prompt_text, str) and full_text.startswith(prompt_text):
            out.append(full_text[len(prompt_text):])
        else:
            bits_suffix = samples_bits[i, cL:]
            suffix_text = decode_samples_to_text(
                cfg,
                bits_suffix.unsqueeze(0),
                dataset_obj=dataset_obj,
                normalize_text8=normalize_text8,
                highlight_ecc_corrections=False,
                prompt_lens_bits=None,
            )[0]
            out.append(suffix_text)

    return out