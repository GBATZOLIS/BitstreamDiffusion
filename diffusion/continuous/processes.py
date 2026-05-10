# diffusion/continuous/processes.py
import torch
import math

class ContinuousForwardProcess:
    """Manages the noise schedule for continuous diffusion models."""
    def __init__(self, cfg):
        c_cfg = cfg.diffusion.continuous
        self.sigma_min = c_cfg.sigma_min
        self.sigma_max = c_cfg.sigma_max
        self.rho = c_cfg.rho
        self.device = torch.device(cfg.device)

    def sample_sigma(self, bsz: int, strategy: str = 'log-uniform', p_mean=-1.2, p_std=1.2) -> torch.Tensor:
        """Draws noise levels σ for a batch."""
        if strategy == "log-normal":
            sigma = torch.empty(bsz, device=self.device)
            mask = torch.ones(bsz, device=self.device, dtype=torch.bool)

            while mask.any():
                n = mask.sum().item()
                s = torch.exp(p_mean + p_std * torch.randn(n, device=self.device))
                ok = (s >= self.sigma_min) & (s <= self.sigma_max)
                sigma[mask.nonzero(as_tuple=True)[0][ok]] = s[ok]
                mask[mask.nonzero(as_tuple=True)[0][ok]] = False

            return sigma
            
        elif strategy == "log-uniform":
            return torch.exp(
                torch.empty(bsz, device=self.device).uniform_(
                    math.log(self.sigma_min),
                    math.log(self.sigma_max)
                )
            )
        else:
            raise ValueError(f"Unknown sigma sampling strategy: {strategy}")

    def get_karras_schedule(self, num_steps: int) -> torch.Tensor:
        inv_rho = 1.0 / self.rho
        i = torch.linspace(0, 1, num_steps, device=self.device)
        sigmas = (
            self.sigma_max**inv_rho
            + i * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        )**self.rho
        return sigmas