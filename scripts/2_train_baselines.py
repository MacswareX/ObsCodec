"""Step 2: Train baseline codecs (PCA, AE, Digital) on simple_spread (30-dim).

PCA:   sweep LD in [2, 4, 8, 16, 32, obs_dim]
AE:    sweep LD in [2, 4, 8, 16, 32]
Digital: sweep bits_per_dim in [1, 2, 4, 6, 8, 12, 16]

Usage: python scripts/2_train_baselines.py [--pca] [--ae] [--digital] [--all]
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from obscodec.config import ASSETS_DIR, CHECKPOINT_DIR
from obscodec.models.pca_baseline import PCACodec
from obscodec.models.ae_baseline import AECodec
from obscodec.models.digital_baseline import DigitalCodec
from obscodec.trainer import train_ae_codec
from obscodec.metrics import evaluate_codec
from obscodec.utils import load_mpe_data, get_unified_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_pca(datasets):
    """PCA sweep across LD values."""
    print("\n" + "=" * 50)
    print("PCA Baseline")
    print("=" * 50)
    results = []

    for scenario_name, data in datasets.items():
        obs_dim = data.shape[1]
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        test_t = torch.FloatTensor(test)

        latent_dims = [d for d in [2, 4, 8, 16, 32] if d <= obs_dim]
        if obs_dim not in latent_dims:
            latent_dims.append(obs_dim)

        for ld in latent_dims:
            name = f"PCA-{scenario_name}-LD{ld}"
            print(f"  {name} (obs_dim={obs_dim})")
            pca = PCACodec(obs_dim, ld)
            pca.fit(train)
            r = evaluate_codec(pca, test_t, DEVICE)
            r["name"] = name
            r["latent_dim"] = ld
            r["scenario"] = scenario_name
            r["obs_dim"] = obs_dim
            results.append(r)
            print(f"    MSE={r['mse']:.4f}")

    # Unified (padded) dataset
    unified = get_unified_dataset(datasets, pad=True)
    obs_dim = unified.shape[1]
    train, test = train_test_split(unified, test_size=0.2, random_state=42)
    test_t = torch.FloatTensor(test)

    for ld in [d for d in [2, 4, 8, 16, 32, 64] if d <= obs_dim]:
        name = f"PCA-unified-LD{ld}"
        print(f"  {name} (obs_dim={obs_dim})")
        pca = PCACodec(obs_dim, ld)
        pca.fit(train)
        r = evaluate_codec(pca, test_t, DEVICE)
        r["name"] = name
        r["latent_dim"] = ld
        r["scenario"] = "unified"
        r["obs_dim"] = obs_dim
        results.append(r)
        print(f"    MSE={r['mse']:.4f}")

    with open(ASSETS_DIR / "pca_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {len(results)} PCA results")


def train_ae(datasets):
    """AE sweep across LD values."""
    print("\n" + "=" * 50)
    print("AE Baseline")
    print("=" * 50)
    results = []

    for scenario_name, data in datasets.items():
        obs_dim = data.shape[1]
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        train, val = train_test_split(train, test_size=0.125, random_state=42)
        test_t = torch.FloatTensor(test).to(DEVICE)

        for ld in [d for d in [2, 4, 8, 16, 32] if d <= obs_dim]:
            name = f"AE-{scenario_name}-LD{ld}"
            ckpt = CHECKPOINT_DIR / f"{name}.pt"
            if ckpt.exists():
                print(f"  [SKIP] {name}")
                ae = AECodec(obs_dim, ld, hidden_dim=128)
                ae.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model_state"])
                ae = ae.to(DEVICE).eval()
            else:
                print(f"  {name} (obs_dim={obs_dim})")
                ae = AECodec(obs_dim, ld, hidden_dim=128)
                out = train_ae_codec(ae, train, val, epochs=200, batch_size=256,
                                     device=DEVICE, model_name=name, lr=1e-3, patience=30)
                ae = out["model"].to(DEVICE).eval()

            with torch.no_grad():
                x_hat, _ = ae(test_t)
                mse = float(F.mse_loss(x_hat, test_t).item())
            r = {"name": name, "latent_dim": ld, "scenario": scenario_name,
                 "obs_dim": obs_dim, "mse": mse}
            results.append(r)
            print(f"    MSE={mse:.4f}")

    # Unified
    unified = get_unified_dataset(datasets, pad=True)
    obs_dim = unified.shape[1]
    train, test = train_test_split(unified, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.125, random_state=42)

    for ld in [d for d in [2, 4, 8, 16, 32, 64] if d <= obs_dim]:
        name = f"AE-unified-LD{ld}"
        ckpt = CHECKPOINT_DIR / f"{name}.pt"
        if ckpt.exists():
            print(f"  [SKIP] {name}")
            ae = AECodec(obs_dim, ld, hidden_dim=128)
            ae.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model_state"])
            ae = ae.to(DEVICE).eval()
        else:
            print(f"  {name} (obs_dim={obs_dim})")
            ae = AECodec(obs_dim, ld, hidden_dim=128)
            out = train_ae_codec(ae, train, val, epochs=200, batch_size=256,
                                 device=DEVICE, model_name=name, lr=1e-3, patience=30)
            ae = out["model"].to(DEVICE).eval()

        with torch.no_grad():
            x_hat, _ = ae(torch.FloatTensor(test).to(DEVICE))
            mse = float(F.mse_loss(x_hat, torch.FloatTensor(test).to(DEVICE)).item())
        r = {"name": name, "latent_dim": ld, "scenario": "unified",
             "obs_dim": obs_dim, "mse": mse}
        results.append(r)
        print(f"    MSE={mse:.4f}")

    with open(ASSETS_DIR / "ae_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {len(results)} AE results")


def train_digital(datasets):
    """Digital quantization sweep across bit depths."""
    print("\n" + "=" * 50)
    print("Digital Baseline")
    print("=" * 50)
    results = []
    bit_depths = [1, 2, 4, 6, 8, 12, 16]

    for scenario_name, data in datasets.items():
        obs_dim = data.shape[1]
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        train_t = torch.FloatTensor(train)
        test_t = torch.FloatTensor(test)

        for bpd in bit_depths:
            name = f"Digital-{scenario_name}-B{bpd}"
            print(f"  {name} (obs_dim={obs_dim}, total_bits={obs_dim*bpd})")
            dig = DigitalCodec(obs_dim, bits_per_dim=bpd)
            dig.fit(train_t)
            r = evaluate_codec(dig, test_t, DEVICE)
            r["name"] = name
            r["bits_per_dim"] = bpd
            r["total_bits"] = obs_dim * bpd
            r["scenario"] = scenario_name
            r["obs_dim"] = obs_dim
            results.append(r)
            print(f"    MSE={r['mse']:.4f}")

    # Unified
    unified = get_unified_dataset(datasets, pad=True)
    obs_dim = unified.shape[1]
    train, test = train_test_split(unified, test_size=0.2, random_state=42)
    train_t = torch.FloatTensor(train)
    test_t = torch.FloatTensor(test)

    for bpd in bit_depths:
        name = f"Digital-unified-B{bpd}"
        print(f"  {name} (obs_dim={obs_dim}, total_bits={obs_dim*bpd})")
        dig = DigitalCodec(obs_dim, bits_per_dim=bpd)
        dig.fit(train_t)
        r = evaluate_codec(dig, test_t, DEVICE)
        r["name"] = name
        r["bits_per_dim"] = bpd
        r["total_bits"] = obs_dim * bpd
        r["scenario"] = "unified"
        r["obs_dim"] = obs_dim
        results.append(r)
        print(f"    MSE={r['mse']:.4f}")

    with open(ASSETS_DIR / "digital_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {len(results)} Digital results")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train baseline codecs")
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--ae", action="store_true")
    parser.add_argument("--digital", action="store_true")
    parser.add_argument("--all", action="store_true", default=True, help="Train all (default)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    datasets = load_mpe_data()
    print(f"Datasets: {list(datasets.keys())}")

    do_all = args.all or (not args.pca and not args.ae and not args.digital)
    if do_all or args.pca:
        train_pca(datasets)
    if do_all or args.ae:
        train_ae(datasets)
    if do_all or args.digital:
        train_digital(datasets)
