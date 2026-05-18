"""β-VAE training + β × latent_dim ablation with KL annealing."""
import numpy as np
import torch
import torch.nn.functional as F
import json
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obscodec.models.vae import BetaVAE
from obscodec.trainer import train_model


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

    latent_dims = [2, 4, 8, 16, 32]
    betas = [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 4.0, 5.0, 8.0, 10.0]
    results = []

    for ld in latent_dims:
        for beta in betas:
            name = f"VAE-LD{ld}-B{beta}"
            print(f"\n{'='*50}\n{name}")

            vae = BetaVAE(obs_dim=obs_dim, latent_dim=ld, beta=beta,
                          kl_warmup_epochs=50)
            out = train_model(vae, train, val, epochs=200, batch_size=256,
                              device=device, model_name=name)

            vae = out["model"].to(device)
            vae.eval()
            test_t = torch.FloatTensor(test).to(device)
            with torch.no_grad():
                x_hat, _ = vae(test_t)
                mse = F.mse_loss(x_hat, test_t).item()

            test_loader = DataLoader(TensorDataset(torch.FloatTensor(test)),
                                     batch_size=256)
            rate_bits = vae.get_rate_estimate(test_loader, device)
            kl_val = vae.kl_nats(test_t)

            r = {
                "latent_dim": ld, "beta": beta,
                "mse": float(mse), "bandwidth": float(vae.equivalent_bandwidth),
                "rate_bits": float(rate_bits), "kl": float(kl_val),
            }
            results.append(r)
            print(f"  MSE={mse:.4f} BW={vae.equivalent_bandwidth:.0f}b "
                  f"Rate={rate_bits:.2f}b KL={kl_val:.4f}")

            if beta in [0.01, 1.0, 4.0]:
                torch.save(vae.state_dict(), f"checkpoints/vae_ld{ld}_b{beta}.pt")

    with open("assets/vae_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFinished {len(results)} VAE runs -> assets/vae_results.json")


if __name__ == "__main__":
    main()
