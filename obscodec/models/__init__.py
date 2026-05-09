"""Model registry for all ObsCodec codecs."""

from obscodec.models.pca_baseline import PCABaseline
from obscodec.models.ae_baseline import StandardAE
from obscodec.models.digital_baseline import DigitalQuantizationBaseline
from obscodec.models.vae import BetaVAE
from obscodec.models.vqvae import VQVAE

__all__ = [
    "PCABaseline",
    "StandardAE",
    "DigitalQuantizationBaseline",
    "BetaVAE",
    "VQVAE",
]
