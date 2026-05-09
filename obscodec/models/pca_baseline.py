"""PCA baseline for linear observation compression."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCABaseline:
    """Fit PCA on training observations and reconstruct from a low-dimensional code."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.obs_dim = None
    
    def fit(self, data: np.ndarray):
        """Fit scaler and PCA on data shaped as (n_samples, obs_dim)."""
        self.obs_dim = data.shape[1]
        data_scaled = self.scaler.fit_transform(data)
        self.pca.fit(data_scaled)
        return self
    
    def compress(self, data: np.ndarray) -> np.ndarray:
        """Map observations to principal-component coordinates."""
        data_scaled = self.scaler.transform(data)
        return self.pca.transform(data_scaled)
    
    def reconstruct(self, compressed: np.ndarray) -> np.ndarray:
        """Map principal-component coordinates back to observation space."""
        data_scaled = self.pca.inverse_transform(compressed)
        return self.scaler.inverse_transform(data_scaled)
    
    def forward(self, data: np.ndarray) -> tuple:
        """Return reconstructed observations and MSE."""
        compressed = self.compress(data)
        reconstructed = self.reconstruct(compressed)
        mse = np.mean((data - reconstructed) ** 2)
        return reconstructed, mse
    
    @property
    def equivalent_bandwidth(self) -> float:
        """Nominal bits per observation if each component is sent as float32."""
        return self.n_components * 32
