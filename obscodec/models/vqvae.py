"""VQ-VAE with EMA codebook updates and dead-entry reset (van den Oord et al. 2017)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import NATS_TO_BITS


class VQEncoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class VQDecoder(nn.Module):
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


class VectorQuantizer(nn.Module):
    """EMA-updated vector quantizer with dead-entry reset."""

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float = 0.25, decay: float = 0.99,
                 epsilon: float = 1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embed = torch.randn(num_embeddings, embedding_dim)
        embed = embed / embed.norm(dim=-1, keepdim=True) * 0.1
        self.register_buffer("embedding", embed)
        self.register_buffer("_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("_ema_embed", embed.clone())

    def forward(self, z: torch.Tensor):
        # z: (B, D), embedding: (K, D)
        # distances: (B, K)
        z_flat = z.reshape(-1, self.embedding_dim)
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.embedding.T
            + self.embedding.pow(2).sum(dim=1)
        )
        encoding_indices = distances.argmin(dim=1)
        z_q = self.embedding[encoding_indices].reshape(z.shape)

        # Update EMA if training
        if self.training:
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            self._cluster_size.data.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )
            dw = encodings.T @ z_flat
            self._ema_embed.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            n = self._cluster_size.sum()
            cluster_size = (
                (self._cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            embed_normalized = self._ema_embed / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)

            # Dead entry reset
            dead = self._cluster_size < 2
            if dead.any():
                n_dead = dead.sum().item()
                alive = z_flat[torch.randint(0, z_flat.size(0), (n_dead,))]
                self.embedding.data[dead] = alive
                self._ema_embed.data[dead] = alive
                self._cluster_size.data[dead] = 1.0

        # VQ loss: commitment + straight-through
        loss = self.commitment_cost * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()  # straight-through estimator
        return z_q, loss.unsqueeze(0), encoding_indices

    @property
    def usage(self) -> float:
        """Fraction of codebook entries with non-zero usage."""
        return float((self._cluster_size > 0).float().mean().item())


class VQVAE(nn.Module):
    """Vector-quantized VAE (VQ-VAE) with EMA codebook."""

    def __init__(self, obs_dim: int, latent_dim: int,
                 num_embeddings: int = 512, commitment_cost: float = 0.25,
                 hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        self.encoder = VQEncoder(obs_dim, latent_dim, hidden_dim)
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = VQDecoder(latent_dim, obs_dim, hidden_dim)

    def encode(self, x):
        z_e = self.encoder(x)
        z_q, _, indices = self.vq(z_e)
        return z_q, indices

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, z_q, vq_loss, indices

    def training_step(self, x):
        self.train()
        x_hat, z_q, vq_loss, _ = self.forward(x)
        recon = F.mse_loss(x_hat, x)
        loss = recon + vq_loss.mean()
        return loss, recon, vq_loss.mean()

    def validation_step(self, x):
        self.eval()
        with torch.no_grad():
            x_hat, z_q, vq_loss, _ = self.forward(x)
            recon = F.mse_loss(x_hat, x)
            loss = recon + vq_loss.mean()
        return loss, recon, vq_loss.mean()

    @property
    def codebook_usage(self) -> float:
        return self.vq.usage

    def get_rate_estimate(self, dataloader: DataLoader, device) -> float:
        """Rate in bits: log2(codebook_size), assuming uniform usage."""
        import math
        n_unique = max(1, int(self.vq.usage * self.vq.num_embeddings))
        return float(math.log2(n_unique) * self.latent_dim)

    @property
    def equivalent_bandwidth(self) -> float:
        import math
        return float(math.log2(self.vq.num_embeddings) * self.latent_dim)
