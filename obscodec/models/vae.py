"""β-VAE codec for multi-agent observation compression.

Architecture decisions (Higgins et al. 2017, Burgess et al. 2018):
- Encoder: BatchNorm → 2-layer MLP → mu (linear), logvar (clamped for stability)
- Decoder: capacity-constrained (hidden_dim/2) to prevent premature posterior collapse
- KL annealing: linear warmup over configurable burn-in epochs (Bowman et al. 2016)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from obscodec.config import BITS_PER_FLOAT32, NATS_TO_BITS


def _kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu,σ²) || N(0,I)) — batched, returns per-sample nats."""
    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


class VAEEncoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.bn = nn.BatchNorm1d(obs_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.bn(x)
        h = self.net(h)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -8.0, 4.0)
        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BetaVAE(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        beta: float = 1.0,
        hidden_dim: int = 128,
        kl_warmup_epochs: int = 50,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.kl_warmup_epochs = kl_warmup_epochs
        self.encoder = VAEEncoder(obs_dim, latent_dim, hidden_dim)
        self.decoder = VAEDecoder(latent_dim, obs_dim, hidden_dim=int(hidden_dim * 0.5))

        self._current_epoch: int = 0
        self._mu: torch.Tensor | None = None
        self._logvar: torch.Tensor | None = None

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch

    @property
    def kl_weight(self) -> float:
        if self.kl_warmup_epochs <= 0:
            return self.beta
        ratio = min(1.0, self._current_epoch / self.kl_warmup_epochs)
        return self.beta * ratio

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        self._mu = mu
        self._logvar = logvar
        return mu, logvar

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self._encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, z

    def loss(
        self, x: torch.Tensor, x_hat: torch.Tensor,
    ) -> torch.Tensor:
        recon_loss = F.mse_loss(x_hat, x)
        if self._mu is None or self._logvar is None:
            self._encode(x)
        kl_per_dim = -0.5 * (
            1.0 + self._logvar - self._mu.pow(2) - self._logvar.exp()
        )
        free_bits = 0.1
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        kl = kl_per_dim.sum(dim=-1).mean()
        return recon_loss + self.kl_weight * kl

    @property
    def last_kl(self) -> float:
        if self._mu is None or self._logvar is None:
            return 0.0
        with torch.no_grad():
            return _kl_normal(self._mu, self._logvar).mean().item()

    def kl_nats(self, x: torch.Tensor) -> float:
        """Mean KL divergence in nats for a batch — no side effects."""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            return _kl_normal(mu, logvar).mean().item()

    def get_rate_estimate(
        self, data_loader: DataLoader, device: str | torch.device,
    ) -> float:
        """Effective KL rate in bits averaged over a DataLoader."""
        self.eval()
        total_rate = 0.0
        n = 0
        with torch.no_grad():
            for (x_batch,) in data_loader:
                x_batch = x_batch.to(device)
                mu, logvar = self.encoder(x_batch)
                kl_nats = _kl_normal(mu, logvar).mean()
                total_rate += (kl_nats * NATS_TO_BITS).item() * x_batch.size(0)
                n += x_batch.size(0)
        return total_rate / n

    @property
    def equivalent_bandwidth(self) -> float:
        return self.latent_dim * BITS_PER_FLOAT32
