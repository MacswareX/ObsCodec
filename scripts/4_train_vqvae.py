"""VQ-VAE full ablation with EMA codebook + dead-entry reset."""
import numpy as np
import torch
import torch.nn.functional as F
import json
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obscodec.models.vqvae import VQVAE
from obscodec.trainer import train_model


def model_done(name):
    """Skip if checkpoint already exists."""
    path = f"checkpoints/vqvae_{name}.pt"
    if os.path.exists(path):
        print(f"  SKIP (exists): {name}")
        return True
    return False


def eval_and_record(vqvae, test_data, name, device, extra_info=None):
    """Evaluate and return result dictionary."""
    vqvae = vqvae.to(device)
    vqvae.eval()
    test_t = torch.FloatTensor(test_data).to(device)
    with torch.no_grad():
        x_hat, _ = vqvae(test_t)
        mse = F.mse_loss(x_hat, test_t).item()
    r = {
        "name": name,
        "latent_dim": vqvae.latent_dim,
        "codebook_size": vqvae.codebook_size,
        "commitment_cost": vqvae.commitment_cost,
        "mse": float(mse),
        "bandwidth": float(vqvae.equivalent_bandwidth),
        "codebook_usage": float(vqvae.codebook_usage()),
    }
    if extra_info:
        r.update(extra_info)
    torch.save(vqvae.state_dict(), f"checkpoints/vqvae_{name}.pt")
    return r


def upsert_result(results, record):
    """Keep one result per named configuration."""
    for idx, existing in enumerate(results):
        if existing["name"] == record["name"]:
            results[idx] = record
            return
    results.append(record)


def main():
    data_path = "data/mpe_observations.npy"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            "Run 'python scripts/1_collect_data.py' first."
        )
    data = np.load(data_path)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.125, random_state=42)
    obs_dim = data.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, obs_dim={obs_dim}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    results = []

    # Phase 1: Codebook Size × Latent Dim grid
    print("\n" + "=" * 60)
    print("Phase 1: Codebook Size x Latent Dim (cc=0.25, EMA)")
    for cs in [32, 64, 128, 256, 512]:
        for ld in [2, 4, 8, 16]:
            name = f"cb{cs}_ld{ld}_cc0.25"
            if model_done(name):
                vqvae = VQVAE(obs_dim=obs_dim, latent_dim=ld,
                              codebook_size=cs, commitment_cost=0.25)
                state = torch.load(f"checkpoints/vqvae_{name}.pt",
                                   map_location='cpu', weights_only=True)
                vqvae.load_state_dict(state)
                r = eval_and_record(vqvae, test, name, device)
                upsert_result(results, r)
                print(f"  Evaluated: MSE={r['mse']:.4f} BW={r['bandwidth']:.0f}b "
                      f"usage={r['codebook_usage']:.2%}")
                continue

            print(f"\n  TRAIN: {name}")
            vqvae = VQVAE(obs_dim=obs_dim, latent_dim=ld,
                          codebook_size=cs, commitment_cost=0.25)
            out = train_model(vqvae, train, val, epochs=200, batch_size=256,
                              device=device, model_name=name, vq_reset_interval=30)
            vqvae = out["model"].to(device)
            r = eval_and_record(vqvae, test, name, device)
            upsert_result(results, r)
            print(f"  Done: MSE={r['mse']:.4f} BW={r['bandwidth']:.0f}b "
                  f"usage={r['codebook_usage']:.2%}")

    # Phase 2: Commitment Cost sweep
    print("\n" + "=" * 60)
    print("Phase 2: Commitment Cost Sweep (LD=4, CB=128)")
    for cc in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        name = f"cb128_ld4_cc{cc}"
        if model_done(name):
            vqvae = VQVAE(obs_dim=obs_dim, latent_dim=4,
                          codebook_size=128, commitment_cost=cc)
            state = torch.load(f"checkpoints/vqvae_{name}.pt",
                               map_location='cpu', weights_only=True)
            vqvae.load_state_dict(state)
            r = eval_and_record(vqvae, test, name, device)
            upsert_result(results, r)
            print(f"  Evaluated: cc={cc} MSE={r['mse']:.4f} usage={r['codebook_usage']:.2%}")
            continue

        print(f"\n  TRAIN: {name}")
        vqvae = VQVAE(obs_dim=obs_dim, latent_dim=4,
                      codebook_size=128, commitment_cost=cc)
        out = train_model(vqvae, train, val, epochs=200, batch_size=256,
                          device=device, model_name=name, vq_reset_interval=30)
        vqvae = out["model"].to(device)
        r = eval_and_record(vqvae, test, name, device)
        upsert_result(results, r)
        print(f"  Done: cc={cc} MSE={r['mse']:.4f} usage={r['codebook_usage']:.2%}")

    with open("assets/vqvae_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFinished {len(results)} unique VQ-VAE runs -> assets/vqvae_results.json")


if __name__ == "__main__":
    main()
