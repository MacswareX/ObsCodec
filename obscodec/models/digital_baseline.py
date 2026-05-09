"""Fixed-bitrate scalar quantization baseline."""

import torch
import torch.nn as nn


class DigitalQuantizationBaseline(nn.Module):
    """Project, uniformly quantize, and reconstruct a compact observation code."""
    
    def __init__(
        self, obs_dim: int, latent_dim: int, bits_per_dim: int = 8
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.bits_per_dim = bits_per_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim),
        )
    
    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Uniform quantization with a straight-through estimator."""
        n_levels = 2 ** self.bits_per_dim
        step = 2.0 / (n_levels - 1)
        x_q = torch.round(x / step) * step
        return x + (x_q - x).detach()
    
    def forward(self, x):
        z_cont = self.encoder(x)
        z = self._quantize(z_cont)
        x_hat = self.decoder(z)
        return x_hat, z
    
    def loss(self, x, x_hat):
        return nn.functional.mse_loss(x_hat, x)
    
    @property
    def equivalent_bandwidth(self) -> float:
        return self.latent_dim * self.bits_per_dim
