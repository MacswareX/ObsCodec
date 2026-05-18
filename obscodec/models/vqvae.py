"""VQ-VAE discrete latent codec with EMA codebook + dead-entry reset.

Architecture decisions:
- EMA codebook updates (van den Oord et al. 2017) prevent dead codebook entries
- Periodic dead-entry reset re-initialises entries that haven't been used recently
- Decoder capacity constrained relative to encoder to avoid trivial reconstruction
"""

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
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay

        enc_hidden = hidden_dim
        dec_hidden = max(int(hidden_dim * 0.5), 32)

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, enc_hidden),
            nn.ReLU(),
            nn.Linear(enc_hidden, enc_hidden),
            nn.ReLU(),
            nn.Linear(enc_hidden, latent_dim),
        )

        codebook = torch.randn(codebook_size, latent_dim)
        codebook = codebook / codebook.norm(dim=-1, keepdim=True) * 0.1
        self.register_buffer("codebook", codebook)
        self.register_buffer("_ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("_ema_sum", codebook.clone())
        self.register_buffer("_usage_count", torch.zeros(codebook_size, dtype=torch.long))

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, dec_hidden),
            nn.ReLU(),
            nn.Linear(dec_hidden, dec_hidden),
            nn.ReLU(),
            nn.Linear(dec_hidden, obs_dim),
        )

        self._last_indices: torch.Tensor | None = None

    def codebook_usage(self) -> float:
        if self._last_indices is None:
            return 0.0
        return len(torch.unique(self._last_indices)) / self.codebook_size

    def reset_dead_entries(self) -> int:
        """Re-initialise codebook entries that have never been used.
        Returns the number of entries reset."""
        dead_mask = self._usage_count == 0
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0
        noise = torch.randn(n_dead, self.codebook.shape[1], device=self.codebook.device)
        noise = noise / noise.norm(dim=-1, keepdim=True) * 0.1
        self.codebook[dead_mask] = noise  # type: ignore[index]
        self._ema_cluster_size[dead_mask] = 0.0
        self._ema_sum[dead_mask] = self.codebook[dead_mask].clone()  # type: ignore[index]
        self._usage_count[dead_mask] = 0
        return n_dead

    def _quantize_ema(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = z_e.reshape(-1, self.latent_dim)
        distances = torch.sum(
            (flat.unsqueeze(1) - self.codebook.unsqueeze(0)) ** 2, dim=-1,
        )
        indices = torch.argmin(distances, dim=-1)
        z_q = F.embedding(indices, self.codebook)

        if self.training:
            one_hot = F.one_hot(indices, self.codebook_size).float()
            cluster_size = one_hot.sum(dim=0)
            self._ema_cluster_size.mul_(self.ema_decay).add_(
                cluster_size, alpha=1 - self.ema_decay,
            )
            embed_sum = flat.t() @ one_hot
            self._ema_sum.mul_(self.ema_decay).add_(
                embed_sum.t(), alpha=1 - self.ema_decay,
            )
            n = cluster_size.sum()
            cluster_size = cluster_size.clamp(min=0.1) / n * (n + 1e-5) / (cluster_size + 1e-5).sum() * n
            self.codebook.copy_(
                (self._ema_sum + 1e-8) / (self._ema_cluster_size.unsqueeze(1) + 1e-8),
            )

            self._usage_count[indices.unique()] += 1

        z_q_ste = z_e + (z_q - z_e).detach()
        self._last_indices = indices
        return z_q_ste, indices

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        z_q, _ = self._quantize_ema(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, z_q

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        recon_loss = F.mse_loss(x_hat, x)
        z_e = self.encoder(x)
        z_q, _ = self._quantize_ema(z_e)
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        return recon_loss + self.commitment_cost * commitment_loss

    @property
    def equivalent_bandwidth(self) -> int:
        return math.ceil(math.log2(self.codebook_size))
