# models/sedd_wrapper.py
import torch.nn as nn
from omegaconf import OmegaConf
from .backbones.official_sedd import SEDD

class OfficialSEDDWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        is_discrete = (cfg.framework == "discrete_sedd")

        if is_discrete:
            graph_type = cfg.diffusion.discrete.q_matrix_type
            if graph_type == "absorb":
                tokens = cfg.data.vocab_size - 1
                scale_by_sigma = cfg.model.get("scale_by_sigma", True)
            else:
                tokens = cfg.data.vocab_size
                scale_by_sigma = False
        else:
            graph_type = "none"
            tokens = 1
            scale_by_sigma = False

        # io_mode: "discrete" | "continuous" | "dual"
        io_mode = str(cfg.model.get("io_mode", "discrete")).lower()

        sedd_config = OmegaConf.create({
            "graph": {"type": graph_type},
            "tokens": tokens,
            "model": {
                "hidden_size": cfg.model.embed_dim,
                "cond_dim":    cfg.model.get("cond_dim", 128),
                "n_heads":     cfg.model.n_heads,
                "n_blocks":    cfg.model.n_blocks,
                "dropout":     cfg.model.get("dropout", 0.05),
                "scale_by_sigma": scale_by_sigma,
                "io_mode":     io_mode,
                "arch_version": int(cfg.model.get("arch_version", 2)),
            }
        })
        self.model = SEDD(sedd_config)

    def forward(self, x, sigma):
        return self.model(x, sigma)
