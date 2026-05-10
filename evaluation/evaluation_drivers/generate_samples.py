# evaluation/evaluation_drivers/generate_samples.py
import copy
import json
import os
import re
import math
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from evaluation.generation_driver import GenerationDriver
from evaluation.distributed import barrier
from evaluation.utils import load_checkpoint

from utils.text_decode import decode_bitstreams_for_eval
from diffusion.continuous.processes import ContinuousForwardProcess
from diffusion.discrete.processes import DiscreteForwardProcess

from utils.ecc_secded import ecc_from_cfg, ecc_chunk_len

_STEP_RE = re.compile(r"step=(\d+)\.pt$")


# -----------------------------------------------------------------------------
# Helpers: checkpoint resolution (mirrors mauve.py behavior)
# -----------------------------------------------------------------------------
def _is_glob_pattern(s: str) -> bool:
    return any(ch in s for ch in ["*", "?", "["])


def _ckpt_step(p: Path) -> int:
    m = _STEP_RE.search(p.name)
    return int(m.group(1)) if m else -1


def _as_list(spec) -> List[str]:
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        return [str(x) for x in spec]
    return [str(spec)]


def _resolve_ckpts(spec, base_ckpt_dir: Path) -> List[Path]:
    spec_list = _as_list(spec)
    out: List[Path] = []

    for s in spec_list:
        s = str(s)
        p = Path(s)

        if p.is_absolute():
            out.append(p)
            continue

        if _is_glob_pattern(s):
            matches = [Path(x) for x in glob(str(base_ckpt_dir / s))]
            out.extend(matches)
        else:
            out.append(base_ckpt_dir / p)

    out = [p for p in out if p.exists() and p.is_file()]

    uniq = {}
    for p in out:
        uniq[str(p.resolve())] = p
    out = list(uniq.values())

    out.sort(key=lambda p: (_ckpt_step(p) if _ckpt_step(p) >= 0 else 10**18, p.name))
    return out


def _sanitize_filename(s: str) -> str:
    s = str(s)
    return "".join(ch if (ch.isalnum() or ch in "._-+") else "_" for ch in s)


# -----------------------------------------------------------------------------
# Helpers: token-faithful split prompt/completion (best effort)
# -----------------------------------------------------------------------------
def _get_tokenizer(dataset_obj):
    if dataset_obj is None:
        return None
    return getattr(dataset_obj, "tokenizer", None)


def _tokenizer_encode(tok, text: str) -> Optional[List[int]]:
    if tok is None:
        return None
    try:
        enc = tok.encode(text)
        if hasattr(enc, "ids"):
            return list(enc.ids)
        if isinstance(enc, (list, tuple)):
            return list(enc)
    except Exception:
        return None
    return None


def _tokenizer_decode(tok, ids: List[int]) -> Optional[str]:
    if tok is None:
        return None
    try:
        return tok.decode(ids)
    except Exception:
        return None


def _split_by_tokens(full_text: str, prompt_len_tokens: int, dataset_obj=None) -> Tuple[str, str]:
    """
    Split full_text into (prompt, completion) using tokenizer if available.
    Fallback: whitespace split (less faithful, but robust).
    """
    full_text = "" if full_text is None else str(full_text)
    L = int(max(0, prompt_len_tokens))

    tok = _get_tokenizer(dataset_obj)
    ids = _tokenizer_encode(tok, full_text)
    if ids is not None:
        p_ids = ids[:L]
        c_ids = ids[L:]
        p_txt = _tokenizer_decode(tok, p_ids)
        c_txt = _tokenizer_decode(tok, c_ids)
        if p_txt is not None and c_txt is not None:
            return p_txt, c_txt

    # fallback
    toks = full_text.split()
    return " ".join(toks[:L]), " ".join(toks[L:])


# -----------------------------------------------------------------------------
# Helpers: writing
# -----------------------------------------------------------------------------
def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write_tex(path: Path, tex: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex, encoding="utf-8")


def _bpt_model(cfg) -> int:
    ecc = ecc_from_cfg(cfg)
    if ecc is not None and bool(getattr(ecc, "enabled", False)):
        return int(ecc_chunk_len(ecc))
    return int(getattr(getattr(cfg, "data", None), "bits_per_token", 1))


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------
def evaluate_generate_samples(
    args,
    cfg,
    model,
    ema,
    use_ema: bool,
    test_loader,
    device,
    rank0: bool,
    ddp_active: bool,
    run_meta: Dict[str, Any],
    is_text_dataset: bool,
):
    if not is_text_dataset:
        if rank0:
            print("\n⚠️  Skipping generate_samples for non-text dataset.")
        return
    if test_loader is None:
        if rank0:
            print("\n⚠️  generate_samples requested but test_loader is None.")
        return

    eval_cfg = getattr(cfg, "evaluation", None)
    if eval_cfg is None:
        raise ValueError("cfg.evaluation missing; cannot run generate_samples.")

    gs_cfg = getattr(eval_cfg, "generate_samples", None)
    if gs_cfg is None:
        if rank0:
            print("\n⚠️  cfg.evaluation.generate_samples missing; skipping.")
        return

    # -------------------------
    # Resolve parameters (CLI overrides config if provided)
    # -------------------------
    seed = int(getattr(gs_cfg, "seed", 123))
    num_samples = int(getattr(gs_cfg, "num_samples", 32))
    micro_bs = int(getattr(gs_cfg, "micro_batch_size", 256))

    samplers = list(getattr(gs_cfg, "samplers", ["heun_karras"]))
    terminal_sigmas = list(map(float, getattr(gs_cfg, "terminal_sigmas", [0.185])))
    guidance_scales = list(map(float, getattr(gs_cfg, "guidance_scales", [2.0])))
    num_steps = int(getattr(gs_cfg, "num_sampling_steps", 64))

    sigma_max = getattr(gs_cfg, "sigma_max", None)
    sigma_max = None if sigma_max is None else float(sigma_max)

    prompt_len_tokens = int(getattr(gs_cfg, "prompt_len_tokens", 128))
    include_reference = bool(getattr(gs_cfg, "include_reference", True))

    write_tex = bool(getattr(gs_cfg, "write_tex", True))
    write_jsonl = bool(getattr(gs_cfg, "write_jsonl", True))

    # Optional CLI overrides (safe)
    if getattr(args, "generate_samples_seed", None) is not None:
        seed = int(args.generate_samples_seed)
    if getattr(args, "generate_samples_num_samples", None) is not None:
        num_samples = int(args.generate_samples_num_samples)
    if getattr(args, "generate_samples_micro_batch_size", None) is not None:
        micro_bs = int(args.generate_samples_micro_batch_size)
    if getattr(args, "generate_samples_prompt_len_tokens", None) is not None:
        prompt_len_tokens = int(args.generate_samples_prompt_len_tokens)

    if getattr(args, "generate_samples_samplers", None) is not None:
        samplers = list(args.generate_samples_samplers)
    if getattr(args, "generate_samples_terminal_sigmas", None) is not None:
        terminal_sigmas = list(map(float, args.generate_samples_terminal_sigmas))
    if getattr(args, "generate_samples_guidance_scales", None) is not None:
        guidance_scales = list(map(float, args.generate_samples_guidance_scales))
    if getattr(args, "generate_samples_steps", None) is not None:
        num_steps = int(args.generate_samples_steps)

    if getattr(args, "generate_samples_sigma_max", None) is not None:
        sigma_max = None if args.generate_samples_sigma_max == "none" else float(args.generate_samples_sigma_max)

    # Checkpoints
    checkpoints_spec = getattr(gs_cfg, "checkpoints", None)
    if getattr(args, "generate_samples_checkpoints", None) is not None:
        checkpoints_spec = args.generate_samples_checkpoints

    base_ckpt_dir = Path(cfg.evaluation.checkpoint_path).parent
    ckpts = _resolve_ckpts(checkpoints_spec, base_ckpt_dir) if checkpoints_spec else []
    spec_provided = checkpoints_spec is not None and len(_as_list(checkpoints_spec)) > 0

    # Backward compatible: if no spec provided, use currently loaded checkpoint once
    if (not spec_provided) and len(ckpts) == 0:
        ckpts = [None]

    # Output directory
    samples_dir = Path(getattr(cfg.evaluation, "samples_dir", getattr(cfg.evaluation, "out_dir", "runs"))).resolve()
    out_root = samples_dir / "generate_samples"
    if rank0:
        out_root.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Print a sanity check (rank0)
    # -------------------------
    if rank0:
        print("\n── generate_samples (paper appendix) ──")
        print(f"   experiment:          {cfg.experiment}")
        print(f"   framework:           {cfg.framework}")
        print(f"   num_samples:         {num_samples}")
        print(f"   prompt_len_tokens:   {prompt_len_tokens} (fixed)")
        print(f"   micro_batch_size:    {micro_bs}")
        print(f"   samplers:            {samplers}")
        print(f"   terminal_sigmas:     {terminal_sigmas}")
        print(f"   guidance_scales:     {guidance_scales}")
        print(f"   num_sampling_steps:  {num_steps}")
        print(f"   sigma_max:           {sigma_max}")
        print(f"   use_ema:             {use_ema}")
        print(f"   checkpoints:         {('current only' if ckpts == [None] else len(ckpts))}")
        print(f"   output_dir:          {out_root}")

    # -------------------------
    # Construct process (CRITICAL: cannot be None)
    # -------------------------
    fw = str(cfg.framework).lower()
    if fw == "continuous_score":
        proc = ContinuousForwardProcess(cfg)
        is_discrete = False
    elif fw == "discrete_sedd":
        proc = DiscreteForwardProcess(cfg)
        is_discrete = True
    else:
        raise ValueError(f"Unknown framework for generate_samples: {cfg.framework}")

    if is_discrete:
        # You can add discrete decoding later; for now you’re using continuous text bitstreams.
        if rank0:
            print("⚠️  generate_samples currently supports continuous_score only (text bitstreams).")
        return

    # -------------------------
    # Temporarily force fixed prompt length via cfg.cond (restore afterwards)
    # -------------------------
    cond_cfg = getattr(cfg, "cond", None)
    if cond_cfg is None:
        raise ValueError("cfg.cond missing; cannot control prompt length.")

    orig_cond = dict(
        sample_prompt_len=bool(getattr(cond_cfg, "sample_prompt_len", False)),
        cond_len_tokens=int(getattr(cond_cfg, "cond_len_tokens", 0)),
        cond_len_tokens_min=int(getattr(cond_cfg, "cond_len_tokens_min", 0)),
        cond_len_tokens_max=int(getattr(cond_cfg, "cond_len_tokens_max", 0)),
    )

    cond_cfg.sample_prompt_len = False
    cond_cfg.cond_len_tokens = int(prompt_len_tokens)
    cond_cfg.cond_len_tokens_min = int(prompt_len_tokens)
    cond_cfg.cond_len_tokens_max = int(prompt_len_tokens)

    driver = GenerationDriver(cfg)
    dataset_obj = getattr(test_loader, "dataset", None)
    bpt = _bpt_model(cfg)

    try:
        # -------------------------
        # Loop checkpoints
        # -------------------------
        for ckpt_path in ckpts:
            # Everyone must participate in any DDP collectives
            curr_meta = copy.deepcopy(run_meta)

            if ckpt_path is not None:
                ckpt_name = ckpt_path.name
                if rank0:
                    print(f"\n   -> loading checkpoint: {ckpt_name}")
                load_checkpoint(model, ema, ckpt_path, device, apply_ema=use_ema)
                model.eval()
            else:
                ckpt_name = Path(cfg.evaluation.checkpoint_path).name
                if rank0:
                    print(f"\n   -> using already-loaded checkpoint: {ckpt_name}")

            curr_meta["checkpoint"] = ckpt_name

            # Generate (all ranks)
            batches = driver.generate_prompt_completion(
                model=model,
                proc=proc,
                device=device,
                loader=test_loader,
                num_samples=int(num_samples),
                sampler_names=list(samplers),
                terminal_sigmas=list(map(float, terminal_sigmas)),
                guidance_scales=list(map(float, guidance_scales)),
                num_steps=int(num_steps),
                seed=int(seed),
                use_amp=True,
                amp_dtype="auto",
                micro_batch_size=int(micro_bs),
                sigma_max=sigma_max,
                gather_to_rank0=True,
            )

            if ddp_active:
                barrier()

            # Only rank0 decodes/writes
            if not rank0:
                if ddp_active:
                    barrier()
                continue

            # Build one TeX file per checkpoint with sections per tag
            tex_lines: List[str] = []
            tex_lines.append("% AUTO-GENERATED (generate_samples)")
            tex_lines.append("% Requires in your preamble:")
            tex_lines.append("%   \\usepackage[most]{tcolorbox}")
            tex_lines.append("%   \\tcbuselibrary{breakable}")
            tex_lines.append("")
            tex_lines.append(f"% experiment: {cfg.experiment}")
            tex_lines.append(f"% checkpoint: {ckpt_name}")
            tex_lines.append(f"% prompt_len_tokens: {prompt_len_tokens}")
            tex_lines.append(f"% num_samples: {num_samples}")
            tex_lines.append(f"% use_ema: {use_ema}")
            tex_lines.append("")

            # Optional: a small header section
            tex_lines.append(f"\\subsection*{{Samples: \\texttt{{{cfg.experiment}}}}}")
            tex_lines.append(f"\\noindent\\textbf{{Checkpoint}}: \\texttt{{{ckpt_name}}}\\\\")
            tex_lines.append(f"\\textbf{{Prompt length}}: {prompt_len_tokens} tokens\\\\")
            tex_lines.append(f"\\textbf{{N}}: {num_samples}\\\\")
            tex_lines.append("")

            rows_jsonl: List[Dict[str, Any]] = []

            for tag, batch in batches.items():
                if batch.gen_bits is None or batch.ref_bits is None:
                    continue

                # Decode full sequences
                gen_texts = decode_bitstreams_for_eval(cfg, batch.gen_bits, mode="full", dataset_obj=dataset_obj)
                ref_texts = decode_bitstreams_for_eval(cfg, batch.ref_bits, mode="full", dataset_obj=dataset_obj)

                # Prompt length per sample (tokens) from returned prompt_len_bits (should match fixed, but keep robust)
                if batch.prompt_len_bits is not None:
                    pl_tok = (batch.prompt_len_bits.to(torch.long) // max(1, bpt)).tolist()
                else:
                    pl_tok = [prompt_len_tokens for _ in range(len(gen_texts))]

                tex_lines.append(f"\\subsubsection*{{Tag: \\texttt{{{_sanitize_filename(tag)}}}}}")
                tex_lines.append("")

                # One box per sample
                for i in range(min(num_samples, len(gen_texts), len(ref_texts))):
                    pL = int(pl_tok[i])

                    # Split prompt and completion token-faithfully when tokenizer exists
                    prompt_ref, ref_cont = _split_by_tokens(ref_texts[i], pL, dataset_obj=dataset_obj)
                    prompt_gen, gen_cont = _split_by_tokens(gen_texts[i], pL, dataset_obj=dataset_obj)

                    # Use REF prompt (faithful real prefix), and GEN completion
                    prompt = prompt_ref
                    completion = gen_cont

                    row = dict(
                        **curr_meta,
                        metric="generate_samples",
                        tag=str(tag),
                        sample_index=int(i),
                        prompt_len_tokens=int(pL),
                        prompt=str(prompt),
                        completion=str(completion),
                        reference_continuation=str(ref_cont) if include_reference else None,
                        sampler_names=list(samplers),
                        terminal_sigmas=list(map(float, terminal_sigmas)),
                        guidance_scales=list(map(float, guidance_scales)),
                        num_sampling_steps=int(num_steps),
                        sigma_max=sigma_max,
                        seed=int(seed),
                        use_ema=bool(use_ema),
                    )
                    rows_jsonl.append(row)

                    tex_lines.append(
                        "\\begin{tcolorbox}[breakable,colback=gray!3!white,colframe=gray!60!black,"
                        f"title={{Sample {i+1:03d}}}]"
                    )
                    tex_lines.append("\\begin{verbatim}")
                    tex_lines.append("[PROMPT]")
                    tex_lines.append(prompt.strip())
                    tex_lines.append("")
                    tex_lines.append("[MODEL COMPLETION]")
                    tex_lines.append(completion.strip())
                    if include_reference:
                        tex_lines.append("")
                        tex_lines.append("[REFERENCE CONTINUATION]")
                        tex_lines.append(ref_cont.strip())
                    tex_lines.append("\\end{verbatim}")
                    tex_lines.append("\\end{tcolorbox}")
                    tex_lines.append("")

            # Write outputs
            stem = f"samples_{_sanitize_filename(cfg.experiment)}_{_sanitize_filename(ckpt_name)}_p{prompt_len_tokens}_N{num_samples}_seed{seed}"
            tex_path = out_root / f"{stem}.tex"
            jsonl_path = out_root / f"{stem}.jsonl"

            if write_tex:
                _write_tex(tex_path, "\n".join(tex_lines) + "\n")
                print(f"   ✓ wrote LaTeX:  {tex_path}")

            if write_jsonl:
                _write_jsonl(jsonl_path, rows_jsonl)
                print(f"   ✓ wrote JSONL:  {jsonl_path}")

            if ddp_active:
                barrier()

    finally:
        # Restore conditioning config
        cond_cfg.sample_prompt_len = orig_cond["sample_prompt_len"]
        cond_cfg.cond_len_tokens = orig_cond["cond_len_tokens"]
        cond_cfg.cond_len_tokens_min = orig_cond["cond_len_tokens_min"]
        cond_cfg.cond_len_tokens_max = orig_cond["cond_len_tokens_max"]