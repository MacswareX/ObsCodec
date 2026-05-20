"""Differentiable channel layers for joint source-channel coding (JSCC) training.

Unlike ``impairments.py`` (evaluation-only, plain classes), these are
``nn.Module`` subclasses that support gradient flow via reparameterization.

Modules:
  - ``DiffAWGN``: reparameterized z + sigma * eps
  - ``DiffErasure``: straight-through Bernoulli dropout
  - ``DiffBlockErasure``: per-agent-block erasure
  - ``DiffRayleighProxy``: simplified Rayleigh (noise gradient only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffAWGN(nn.Module):
    """Differentiable AWGN channel via reparameterization trick.

    ``y = z + sigma * eps``  where ``eps ~ N(0, I)`` and sigma is derived
    from the SNR. Gradients flow through z; eps is pure noise.

    SNR can be:
      - Fixed: ``snr_db`` scalar
      - Random per batch: pass ``snr_min``/``snr_max`` and set ``training=True``
      - Passed per call: ``forward(z, snr_db=value)``
    """

    def __init__(self, snr_db: float = 10.0, snr_min: float = -5, snr_max: float = 20):
        super().__init__()
        self.snr_db = snr_db
        self.snr_min = snr_min
        self.snr_max = snr_max

    def _sigma(self, z: torch.Tensor, snr: float) -> torch.Tensor:
        if snr == float("inf"):
            return torch.tensor(0.0, device=z.device)
        signal_power = z.pow(2).mean()
        noise_power = signal_power / (10 ** (snr / 10))
        return torch.sqrt(noise_power + 1e-20)

    def forward(self, z: torch.Tensor, snr_db: float | None = None) -> torch.Tensor:
        if snr_db is None:
            if self.training and self.snr_min < self.snr_max:
                snr_db = self.snr_min + (self.snr_max - self.snr_min) * torch.rand(1, device=z.device).item()
            else:
                snr_db = self.snr_db
        if snr_db is None or snr_db == float("inf"):
            return z
        sigma = self._sigma(z, snr_db)
        eps = torch.randn_like(z)
        return z + sigma * eps


class DiffErasure(nn.Module):
    """Differentiable packet erasure via straight-through estimator.

    Randomly zeroes ``loss_rate`` fraction of latent dimensions.
    Surviving dims are scaled by ``1/(1-loss_rate)`` (like dropout).
    Gradients flow via straight-through — the erasure mask is detached.
    """

    def __init__(self, loss_rate: float = 0.1):
        super().__init__()
        self.loss_rate = loss_rate

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.loss_rate <= 0 or not self.training:
            return z
        mask = torch.bernoulli(torch.full_like(z, 1 - self.loss_rate))
        scale = 1.0 / (1.0 - self.loss_rate + 1e-8)
        return z * mask * scale


class DiffBlockErasure(nn.Module):
    """Differentiable per-agent-block erasure.

    Drops contiguous blocks of size ``block_size`` from the latent vector.
    Useful for simulating per-agent message loss in multi-agent systems.
    """

    def __init__(self, loss_rate: float = 0.1, block_size: int = 6):
        super().__init__()
        self.loss_rate = loss_rate
        self.block_size = max(block_size, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.loss_rate <= 0 or not self.training:
            return z
        batch, dim = z.shape
        n_blocks = dim // self.block_size
        if n_blocks == 0:
            return z
        mask = torch.ones(batch, dim, device=z.device)
        block_mask = torch.bernoulli(torch.full((batch, n_blocks), 1 - self.loss_rate, device=z.device))
        for i in range(n_blocks):
            start = i * self.block_size
            end = start + self.block_size
            mask[:, start:end] = block_mask[:, i:i + 1]
        scale = 1.0 / (1.0 - self.loss_rate + 1e-8)
        return z * mask * scale


class DiffRayleighProxy(nn.Module):
    """Simplified Rayleigh fading for JSCC training.

    ``y = h * z + n`` where ``h`` is sampled from |N(0,1)| (detached, no h-gradient)
    and ``n`` is AWGN (reparameterized). Only the noise gradient flows —
    this is a compromise between realism and training stability.
    """

    def __init__(self, snr_db: float = 10.0):
        super().__init__()
        self.snr_db = snr_db
        self._awgn = DiffAWGN(snr_db=snr_db)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.abs(torch.randn_like(z)).detach()
        z_faded = z * h
        sigma = self._awgn._sigma(z_faded, self.snr_db)
        eps = torch.randn_like(z)
        return z_faded + sigma * eps
