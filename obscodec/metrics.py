"""Metric helpers for reconstruction and rate-distortion reporting."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from obscodec.config import COLLAPSE_KL_THRESHOLD, EPS, MIN_COMPRESSED_BITS


def mse_to_psnr(mse: float, max_val: float = 1.0) -> float:
    """Convert mean squared error to PSNR in dB."""
    if mse < 1e-12:
        return float("inf")
    return 10.0 * np.log10(max_val ** 2 / mse)


def rate_distortion_efficiency(mse: float, bandwidth: float) -> float:
    """Return a compact 1/(distortion * bandwidth) score."""
    return 1.0 / (mse * bandwidth + EPS)


def compression_ratio(raw_bits: float, compressed_bits: float) -> float:
    """Compute raw_bits / compressed_bits with a small floor."""
    return raw_bits / max(compressed_bits, MIN_COMPRESSED_BITS)


def posterior_collapse_ratio(
    kl_values: Iterable[float], threshold: float = COLLAPSE_KL_THRESHOLD,
) -> float:
    """Fraction of KL values below a collapse threshold."""
    values = list(kl_values)
    if not values:
        return 0.0
    return sum(1 for k in values if k < threshold) / len(values)
