"""ObsCodec centralized configuration and shared constants."""
from dataclasses import dataclass
import math
import torch

# -- Physical constants ----------------------------------------------
BITS_PER_FLOAT32: int = 32
NATS_TO_BITS: float = 1.0 / math.log(2)   # ~ 1.442695
COLLAPSE_KL_THRESHOLD: float = 0.05
EPS: float = 1e-8
MIN_COMPRESSED_BITS: float = 0.01


@dataclass
class Config:
    data_path: str = "data/mpe_observations.npy"
    n_samples: int = 50_000
    n_agents: int = 3
    max_steps: int = 25
    env_name: str = "simple_spread_v3"
    obs_dim: int = 18
    hidden_dim: int = 128
    batch_size: int = 256
    epochs: int = 200
    lr: float = 1e-3
    patience: int = 20
    val_split: float = 0.2
    test_split: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    assets_dir: str = "assets"

cfg = Config()
