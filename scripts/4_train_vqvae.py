"""Step 4: VQ-VAE training — codebook sweep + multi-scenario + channel robustness.

Codebook sizes: [128, 256, 512]
Commitment costs: [0.25, 1.0]
Scenarios: simple_spread (30-dim), spread_xhd (90-dim)
Channel robustness: AWGN + Rayleigh fading at test time

Usage: python scripts/4_train_vqvae.py [--scenario simple_spread|spread_xhd]
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from sklearn.model_selection import train_test_split

from obscodec.config import DATA_DIR, ASSETS_DIR, CHECKPOINT_DIR
from obscodec.models.vqvae import VQVAE
from obscodec.trainer import train_model
from obscodec.metrics import evaluate_codec
from obscodec.channel.impairments import (AWGNChannel, RayleighFadingChannel,
    evaluate_channel_sweep, get_agent_boundaries)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 200
BATCH_SIZE = 256
PATIENCE = 30
LR = 1e-3
HIDDEN_DIM = 128
CB_SIZES = [128, 256, 512]
COMMITMENT_COSTS = [0.25, 1.0]
LDS = [8, 16]


def load_data(name):
    path = DATA_DIR / f"{name}_obs.npy"
    if not path.exists():
        raise FileNotFoundError(f"No data: {path}")
    data = np.load(str(path)).astype(np.float32)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.125, random_state=42)
    return train, val, test, data.shape[1]


def run_vqvae_scenario(scenario_name):
    train, val, test, obs_dim = load_data(scenario_name)
    test_t = torch.FloatTensor(test).to(DEVICE)
    results = []

    for ld in LDS:
        for cb in CB_SIZES:
            for cc in COMMITMENT_COSTS:
                name = f"VQVAE-{scenario_name}-LD{ld}-CB{cb}-CC{cc}"
                ckpt_path = CHECKPOINT_DIR / f"{name}.pt"

                if ckpt_path.exists():
                    print(f"  [SKIP] {name}")
                    ckpt = torch.load(ckpt_path, map_location=DEVICE)
                    model = VQVAE(obs_dim, ld, num_embeddings=cb, commitment_cost=cc,
                                  hidden_dim=HIDDEN_DIM)
                    model.load_state_dict(ckpt["model_state"])
                    model = model.to(DEVICE).eval()
                else:
                    print(f"\n  {name} (obs_dim={obs_dim})")
                    model = VQVAE(obs_dim, ld, num_embeddings=cb, commitment_cost=cc,
                                  hidden_dim=HIDDEN_DIM)
                    out = train_model(model, train, val, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                     device=DEVICE, model_name=name, lr=LR, patience=PATIENCE)
                    model = out["model"].to(DEVICE).eval()

                r = evaluate_codec(model, test_t, DEVICE)
                r["name"] = name
                r["latent_dim"] = ld
                r["codebook_size"] = cb
                r["commitment_cost"] = cc
                r["scenario"] = scenario_name
                r["obs_dim"] = obs_dim

                # Codebook utilization
                if hasattr(model, "codebook_usage"):
                    r["codebook_usage"] = float(model.codebook_usage())

                results.append(r)
                cb_use = r.get("codebook_usage", "N/A")
                print(f"    MSE={r['mse']:.4f}  CB_usage={cb_use}")

    return results


def run_channel_robustness(scenario_name):
    """Evaluate AWGN + Rayleigh fading on best VQ-VAE checkpoint."""
    print(f"\n{'='*50}")
    print(f"Channel Robustness: {scenario_name}")
    print(f"{'='*50}")

    _, _, test, obs_dim = load_data(scenario_name)
    test_t = torch.FloatTensor(test).to(DEVICE)

    # Load best VQ-VAE model (LD=16, CB=512)
    for ld in [16]:
        for cb in [512]:
            for cc in [0.25]:
                name = f"VQVAE-{scenario_name}-LD{ld}-CB{cb}-CC{cc}"
                ckpt_path = CHECKPOINT_DIR / f"{name}.pt"
                if not ckpt_path.exists():
                    continue
                ckpt = torch.load(ckpt_path, map_location=DEVICE)
                model = VQVAE(obs_dim, ld, num_embeddings=cb, commitment_cost=cc,
                              hidden_dim=HIDDEN_DIM)
                model.load_state_dict(ckpt["model_state"])
                model = model.to(DEVICE).eval()

                print(f"\n  Model: {name}")
                print(f"  Baseline MSE: {evaluate_codec(model, test_t, DEVICE)['mse']:.4f}")

                # AWGN sweep
                print("\n  AWGN Channel:")
                awgn = evaluate_channel_sweep(model, test_t, DEVICE,
                    lambda v: AWGNChannel(v), "SNR", [20, 15, 10, 5, 0, -5])
                for k, v in awgn.items():
                    print(f"    {k}: MSE={v['mse']:.4f}")

                # Rayleigh fading
                print("\n  Rayleigh Fading:")
                fading = evaluate_channel_sweep(model, test_t, DEVICE,
                    lambda v: RayleighFadingChannel(v, fading_mode="block"),
                    "fading_SNR", [20, 15, 10, 5, 0])
                for k, v in fading.items():
                    print(f"    {k}: MSE={v['mse']:.4f}")

    return {}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VQ-VAE training + channel robustness")
    parser.add_argument("--scenario", choices=["simple_spread", "spread_xhd"],
                        help="Which scenario to run")
    parser.add_argument("--channel", action="store_true", help="Run channel robustness eval")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    scenarios = [args.scenario] if args.scenario else ["simple_spread", "spread_xhd"]
    all_results = []

    for scenario in scenarios:
        print(f"\n{'#'*60}")
        print(f"VQ-VAE: {scenario}")
        print(f"{'#'*60}")
        results = run_vqvae_scenario(scenario)
        all_results.extend(results)

    out_path = ASSETS_DIR / "vqvae_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} VQ-VAE results to {out_path}")

    if args.channel:
        for scenario in scenarios:
            run_channel_robustness(scenario)
