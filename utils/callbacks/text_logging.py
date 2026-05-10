from __future__ import annotations
from typing import Any
import torch

try:
    import wandb
except ImportError:
    wandb = None

from utils.text_decode import format_decoded_block
from utils.ecc_secded import ecc_from_cfg
from .dataset_utils import is_text8


def log_text_sequences(
    trainer,
    seq: torch.Tensor,
    tag: str,
    epoch: int,
    *,
    max_samples: int = 8,
    prefix_bits: torch.Tensor | None = None,
    prefix_len_bits: int = 0,
    prefix_mask: torch.Tensor | None = None,
):
    ds_for_decode = (
        trainer.val_loader.dataset
        if getattr(trainer, "val_loader", None) is not None
        else trainer.train_loader.dataset
    )

    # Detect if ECC is enabled in the config
    ecc = ecc_from_cfg(trainer.cfg)
    do_ecc_highlight = bool(ecc.enabled)

    # Note: We pass prefix_mask/prefix_len_bits directly. 
    # format_decoded_block now handles calculating the length and applying the bolding.
    block = format_decoded_block(
        trainer.cfg,
        seq,
        dataset_obj=ds_for_decode,
        tag=tag,
        epoch=epoch,
        max_samples=max_samples,
        normalize_text8=is_text8(trainer),
        prefix_bits=prefix_bits,
        prefix_len_bits=int(prefix_len_bits),
        prefix_mask=prefix_mask,              
        bold_prefix=True,
        show_prompt_suffix_lines=True,
        highlight_ecc_corrections=do_ecc_highlight, 
    )
    
    # TensorBoard supports Markdown, so **bold** and _italics_ will render.
    trainer.writer.add_text(f"generated_text/{tag}", block, epoch)

    if getattr(trainer, "use_wandb", False) and wandb is not None:
        # WandB also supports Markdown nicely
        trainer._log_wandb({f"generated_text/{tag}": wandb.Html(f"<pre>{block}</pre>")})