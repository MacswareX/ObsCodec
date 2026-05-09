"""VQ-VAE discrete latent codec with codebook-usage diagnostics."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQVAE(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        codebook_size: int = 256,
        commitment_cost: float = 0.25,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(
            self.codebook.weight,
            -1.0 / codebook_size,
            1.0 / codebook_size,
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    # ── public diagnostics (no stale side-effect state) ──────────────
    def codebook_usage(self) -> float:
        """Fraction of codebook entries used on the most recent batch."""
        if not hasattr(self, "_last_indices"):
            return 0.0
        indices = self._last_indices
        return len(torch.unique(indices)) / self.codebook_size

    def _quantize(
        self, z_e: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = torch.sum(
            (z_e.unsqueeze(1) - self.codebook.weight.unsqueeze(0)) ** 2, dim=-1,
        )
        indices = torch.argmin(distances, dim=-1)
        z_q = self.codebook(indices)

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        z_q_ste = z_e + (z_q - z_e).detach()

        # Store indices so codebook_usage() can read them without a
        # second forward pass.
        self._last_indices = indices

        return z_q_ste, indices, vq_loss

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        z_q, _, _ = self._quantize(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, z_q

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        recon_loss = F.mse_loss(x_hat, x)
        z_e = self.encoder(x)
        _, _, vq_loss = self._quantize(z_e)
        return recon_loss + vq_loss

    @property
    def equivalent_bandwidth(self) -> int:
        """Bits needed to transmit one selected codebook index."""
        return math.ceil(math.log2(self.codebook_size))
