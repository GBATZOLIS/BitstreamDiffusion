#evaluation/external_perplexity.py
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _resolve_torch_dtype(name: str):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    return "auto"


@torch.no_grad()
def _score_ids_with_mask(
    model,
    ids: torch.Tensor,
    score_mask: torch.Tensor,
    *,
    device: torch.device,
    max_seq_len: int,
    stride: int,
    use_amp: bool,
) -> Tuple[float, int]:
    """
    ids: [L]
    score_mask: [L] bool, True where target token should contribute to loss
    """
    L = int(ids.numel())
    if L < 2:
        return 0.0, 0

    amp_enabled = bool(use_amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16

    total_nll_nats = 0.0
    total_scored = 0
    prev_end = 1

    for end in range(stride, L + stride, stride):
        end = min(end, L)
        begin = max(0, end - max_seq_len)

        window = ids[begin:end].unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out = model(window)
            logits = out.logits if hasattr(out, "logits") else out

        losses = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            window[:, 1:].reshape(-1),
            reduction="none",
        )

        trg_start = max(prev_end, begin + 1)
        trg_end = end
        if trg_end > trg_start:
            local_start = trg_start - (begin + 1)
            local_end = trg_end - (begin + 1)
            local_losses = losses[local_start:local_end]
            local_mask = score_mask[trg_start:trg_end].to(device)
            if local_mask.any():
                total_nll_nats += local_losses[local_mask].sum().item()
                total_scored += int(local_mask.sum().item())

        prev_end = end
        if end >= L:
            break

    return total_nll_nats, total_scored


class HFExternalPerplexityEvaluator:
    """
    Public, transparent external evaluator using Hugging Face causal-LM weights.
    Default should be openai-community/gpt2-large.
    """

    def __init__(
        self,
        *,
        model_name: str = "openai-community/gpt2-large",
        revision: Optional[str] = None,
        device: Optional[torch.device] = None,
        torch_dtype: str = "bfloat16",
        attn_implementation: Optional[str] = "sdpa",
        use_amp: bool = True,
        max_seq_len: Optional[int] = None,
        stride: Optional[int] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(use_amp)
        self.model_name = str(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=revision,
            use_fast=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = dict(
            revision=revision,
            torch_dtype=_resolve_torch_dtype(torch_dtype),
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs).to(self.device)
        self.model.eval()

        cfg_max = getattr(self.model.config, "max_position_embeddings", None)
        tok_max = getattr(self.tokenizer, "model_max_length", None)
        candidates = [x for x in [cfg_max, tok_max, max_seq_len] if isinstance(x, int) and 0 < x < 10**9]
        self.max_seq_len = min(candidates) if candidates else 2048
        self.stride = int(stride or self.max_seq_len)

    @torch.no_grad()
    def score_texts(self, texts: List[str]) -> Dict[str, float]:
        total_nll_nats = 0.0
        total_scored = 0

        for txt in texts:
            if not txt:
                continue
            enc = self.tokenizer(txt, add_special_tokens=False)
            ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=self.device)
            if ids.numel() < 2:
                continue
            mask = torch.ones_like(ids, dtype=torch.bool)
            mask[0] = False

            nll_nats, n_scored = _score_ids_with_mask(
                self.model,
                ids,
                mask,
                device=self.device,
                max_seq_len=self.max_seq_len,
                stride=self.stride,
                use_amp=self.use_amp,
            )
            total_nll_nats += nll_nats
            total_scored += n_scored

        if total_scored == 0:
            return {"external_bpt": float("inf"), "external_ppl": float("inf")}

        nll_per_tok = total_nll_nats / total_scored
        return {
            "external_bpt": float(nll_per_tok / math.log(2.0)),
            "external_ppl": float(math.exp(nll_per_tok)),
        }

    @torch.no_grad()
    def score_prompt_completion_pairs(
        self,
        prompt_texts: List[str],
        completion_texts: List[str],
    ) -> Dict[str, float]:
        total_nll_nats = 0.0
        total_scored = 0

        for prompt, completion in zip(prompt_texts, completion_texts):
            prompt = prompt or ""
            completion = completion or ""
            full = prompt + completion
            if not full:
                continue

            enc = self.tokenizer(
                full,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=self.device)
            offsets = enc["offset_mapping"]
            if ids.numel() < 2:
                continue

            prompt_chars = len(prompt)
            # count only targets whose token span starts in the completion region
            score_mask = torch.tensor(
                [bool(start >= prompt_chars) for (start, end) in offsets],
                dtype=torch.bool,
                device=self.device,
            )
            score_mask[0] = False

            nll_nats, n_scored = _score_ids_with_mask(
                self.model,
                ids,
                score_mask,
                device=self.device,
                max_seq_len=self.max_seq_len,
                stride=self.stride,
                use_amp=self.use_amp,
            )
            total_nll_nats += nll_nats
            total_scored += n_scored

        if total_scored == 0:
            return {"external_bpt": float("inf"), "external_ppl": float("inf")}

        nll_per_tok = total_nll_nats / total_scored
        return {
            "external_bpt": float(nll_per_tok / math.log(2.0)),
            "external_ppl": float(math.exp(nll_per_tok)),
        }