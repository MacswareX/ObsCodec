"""Standard autoencoder baseline for nonlinear observation compression."""

from __future__ import annotations

import torch
import torch.nn as nn

from obscodec.config import BITS_PER_FLOAT32


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class StandardAE(nn.Module):
    """Plain MSE autoencoder without an explicit rate penalty."""

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.encoder = _mlp(obs_dim, hidden_dim, latent_dim)
        self.decoder = _mlp(latent_dim, hidden_dim, obs_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(x_hat, x)

    @property
    def equivalent_bandwidth(self) -> float:
        return self.latent_dim * BITS_PER_FLOAT32
