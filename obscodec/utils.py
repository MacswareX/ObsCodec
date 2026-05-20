"""Miscellaneous utilities."""

import numpy as np
import torch
from pathlib import Path
import json


def load_mpe_data(data_dir: str | Path = None) -> dict[str, np.ndarray]:
    """Load all collected MPE datasets as numpy arrays.

    Returns:
        Dict mapping scenario_name -> np.ndarray of shape (N, obs_dim).
    """
    from .config import DATA_DIR
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    results = {}
    for np_path in sorted(data_dir.glob("*_obs.npy")):
        name = np_path.stem.replace("_obs", "")
        results[name] = np.load(np_path)
    return results


def get_unified_dataset(datasets: dict[str, np.ndarray] | None = None,
                        pad: bool = True) -> np.ndarray:
    """Concatenate all scenario datasets into one array.

    If pad=True, pads each scenario's observations to the max dimension
    across all scenarios (with zeros).
    """
    if datasets is None:
        datasets = load_mpe_data()

    if not pad:
        return np.concatenate(list(datasets.values()), axis=0)

    max_dim = max(arr.shape[1] for arr in datasets.values())
    padded = []
    for arr in datasets.values():
        if arr.shape[1] < max_dim:
            p = np.zeros((arr.shape[0], max_dim), dtype=arr.dtype)
            p[:, :arr.shape[1]] = arr
            padded.append(p)
        else:
            padded.append(arr)
    return np.concatenate(padded, axis=0)


def print_summary_table(results: list[dict]):
    """Print a formatted summary table of codec results."""
    print(f"{'name':<30} {'MSE':>8} {'KL':>8} {'Rate(b)':>8} {'Regime':<18}")
    print("-" * 75)
    for r in results:
        name = r.get("name", "")[:28]
        mse = f"{r.get('mse', 0):.4f}"
        kl = f"{r.get('kl', 0):.4f}" if r.get("kl") else "  N/A"
        rate = f"{r.get('rate_bits', 0):.1f}" if r.get("rate_bits") else "  N/A"
        regime = r.get("regime", "unknown")[:16]
        print(f"{name:<30} {mse:>8} {kl:>8} {rate:>8} {regime:<18}")
