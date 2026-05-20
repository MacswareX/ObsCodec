"""Digital quantization codec — uniform scalar quantization baseline.

Simulates a conventional digital communication pipeline:
    1. Normalize to [-1, 1] per dimension
    2. Quantize to N bits per dimension via uniform scalar quantization
    3. Dequantize and denormalize
"""

import torch
import torch.nn as nn


class DigitalCodec(nn.Module):
    """Uniform scalar quantization — digital transmission baseline."""

    def __init__(self, obs_dim: int, bits_per_dim: int = 8):
        super().__init__()
        self.obs_dim = obs_dim
        self.bits_per_dim = bits_per_dim
        self.levels = 2 ** bits_per_dim
        self.register_buffer("_min", None, persistent=True)
        self.register_buffer("_max", None, persistent=True)

    def fit(self, x: torch.Tensor):
        """Record per-dimension min/max from training data."""
        self._min = x.min(dim=0).values.detach()
        self._max = x.max(dim=0).values.detach()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize to discrete levels, return discrete indices."""
        assert self._min is not None, "Must fit before encoding"
        # Normalize to [0, 1]
        x_norm = (x - self._min) / (self._max - self._min + 1e-8)
        x_norm = x_norm.clamp(0, 1)
        # Quantize to integer levels [0, levels-1]
        indices = (x_norm * (self.levels - 1)).round().long()
        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize back to continuous values."""
        x_norm = indices.float() / (self.levels - 1)
        return x_norm * (self._max - self._min + 1e-8) + self._min

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    @property
    def total_bits(self) -> int:
        return self.obs_dim * self.bits_per_dim
