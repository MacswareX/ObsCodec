"""PCA codec — linear dimensionality reduction baseline."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


class PCACodec(nn.Module):
    """PCA-based codec. Fit on training data, then encode/decode via projection."""

    def __init__(self, obs_dim: int, latent_dim: int, whiten: bool = False):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.pca = PCA(n_components=latent_dim, whiten=whiten)
        self._fitted = False

    def fit(self, data: np.ndarray):
        """Fit PCA on training data of shape (N, obs_dim)."""
        self.pca.fit(data)
        self._fitted = True

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        assert self._fitted, "PCA must be fit before encoding"
        x_np = x.detach().cpu().numpy()
        z_np = self.pca.transform(x_np)
        return torch.from_numpy(z_np.astype(np.float32)).to(x.device)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert self._fitted, "PCA must be fit before decoding"
        z_np = z.detach().cpu().numpy()
        x_np = self.pca.inverse_transform(z_np)
        return torch.from_numpy(x_np.astype(np.float32)).to(z.device)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z
