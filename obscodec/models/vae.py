"""β-VAE with KL warmup, optional free-bits, and asymmetric decoder.

Phase 2a observed: 30-dim data pushes collapse cliff from β≈0.5 to β≈0.7-1.0.
To reach β≈2.0-4.0 (needed for semantic communication), we add:

  1. Free bits (Kingma et al. 2016): per-dimension KL floor λ keeps each latent
     dimension active. Only KL above λ nats/dim is penalized.
  2. Asymmetric decoder: larger decoder than encoder reduces encoder pressure
     toward determinism, allowing richer stochastic latents.

Both default to off (backward compatible with Phase 1/2a checkpoints).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import NATS_TO_BITS


class VAEEncoder(nn.Module):
    """Deterministic encoder backbone — outputs μ and log σ²."""

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.shared(x)
        return self.mu(h), self.logvar(h)


class VAEDecoder(nn.Module):
    """Decoder — full hidden_dim (no halving, per Phase 1 lesson)."""

    def __init__(self, latent_dim: int, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, z):
        return self.net(z)


class BetaVAE(nn.Module):
    """β-VAE with KL warmup (Bowman et al. 2016) and no free bits.

    The KL warmup linearly ramps beta from 0 to target over `kl_warmup_epochs`.
    free_bits=0.0 by default — Phase 1 showed that any free_bits floor
    creates artificial flatlines that obscure the true KL-vs-β relationship.

    Args:
        obs_dim: Observation dimensionality.
        latent_dim: Latent code dimensionality.
        beta: KL weight (Lagrangian multiplier in rate-distortion view).
        hidden_dim: Hidden layer width for both encoder and decoder.
        kl_warmup_epochs: Number of epochs to linearly ramp beta from 0.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        beta: float = 1.0,
        hidden_dim: int = 128,
        kl_warmup_epochs: int = 150,
        free_bits: float = 0.0,
        decoder_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.hidden_dim = hidden_dim
        self.kl_warmup_epochs = kl_warmup_epochs
        self.free_bits = free_bits

        dec_hidden = decoder_hidden_dim if decoder_hidden_dim is not None else hidden_dim
        self.encoder = VAEEncoder(obs_dim, latent_dim, hidden_dim)
        self.decoder = VAEDecoder(latent_dim, obs_dim, dec_hidden)

        self._current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update epoch counter for KL warmup schedule."""
        self._current_epoch = epoch

    @property
    def beta_current(self) -> float:
        """Linearly ramped beta based on warmup progress."""
        if self.kl_warmup_epochs <= 0:
            return self.beta
        progress = min(self._current_epoch / self.kl_warmup_epochs, 1.0)
        return self.beta * progress

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, mu, logvar

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Per-sample KL divergence in nats: KL(q(z|x) || N(0,I))."""
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl_per_dim.sum(dim=-1).mean()

    def training_step(self, x):
        self.train()
        x_hat, z, mu, logvar = self.forward(x)
        recon = F.mse_loss(x_hat, x)
        kl = self.kl_divergence(mu, logvar)
        # Free bits: per-dimension KL floor — only penalize KL above λ nats/dim
        if self.free_bits > 0:
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_per_dim = kl_per_dim.mean(dim=0)  # (latent_dim,)
            kl_active = torch.clamp(kl_per_dim - self.free_bits, min=0).sum()
            loss = recon + self.beta_current * kl_active
        else:
            loss = recon + self.beta_current * kl
        return loss, recon, kl

    def validation_step(self, x):
        self.eval()
        with torch.no_grad():
            x_hat, z, mu, logvar = self.forward(x)
            recon = F.mse_loss(x_hat, x)
            kl = self.kl_divergence(mu, logvar)
            loss = recon + self.beta * kl  # validation uses target beta
        return loss, recon, kl

    def kl_nats(self, x: torch.Tensor) -> float:
        """Compute KL divergence in nats for a test batch."""
        self.eval()
        with torch.no_grad():
            _, mu, logvar = self.encode(x)
            kl = self.kl_divergence(mu, logvar)
        return float(kl.item())

    def get_rate_estimate(self, dataloader: DataLoader, device) -> float:
        """Monte-Carlo estimate of information rate in bits."""
        self.eval()
        total_kl = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)
                _, mu, logvar = self.encode(x)
                kl = self.kl_divergence(mu, logvar)
                total_kl += kl * x.size(0)
                total_samples += x.size(0)
        return float(total_kl / max(total_samples, 1) * NATS_TO_BITS)

    @property
    def equivalent_bandwidth(self) -> float:
        return float(self.latent_dim * 32)  # bits per float32
