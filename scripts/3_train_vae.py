"""beta-VAE training — standard sweep + high-dim scaling + anti-collapse.

Sub-experiments:
  standard:          Standard beta-VAE sweep on simple_spread (30-dim), LD=8,16
  hd_scaling:        High-dim sweep on 4 datasets (40-90 dim), LD=16
  anti_collapse:     Free-bits + decoder expansion anti-collapse sweep on spread_xhd
  cross_validation:  FB=0.1 cross-scenario validation on tag_hd + comm_hd

Usage: python scripts/3_train_vae.py [--phase standard|hd_scaling|anti_collapse|cross_validation] [--all]
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from sklearn.model_selection import train_test_split

from obscodec.config import DATA_DIR, ASSETS_DIR, CHECKPOINT_DIR
from obscodec.models.vae import BetaVAE
from obscodec.trainer import train_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 250
BATCH_SIZE = 256
PATIENCE = 25
LR = 5e-5
WARMUP = 150
HIDDEN_DIM = 128


def load_data(name):
    path = DATA_DIR / f"{name}_obs.npy"
    if not path.exists():
        raise FileNotFoundError(f"No data: {path}")
    data = np.load(str(path)).astype(np.float32)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.125, random_state=42)
    return train, val, test, data.shape[1]


def evaluate_model(model, test_t):
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encoder(test_t)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_nats = float(kl_per_dim.sum(dim=-1).mean().item())
        z = model.reparameterize(mu, logvar)
        x_hat = model.decode(z)
        mse = float(torch.nn.functional.mse_loss(x_hat, test_t).item())
    return kl_nats, mse


def run_configs(dataset_name, obs_dim, train, val, test, configs, ld=16):
    test_t = torch.FloatTensor(test).to(DEVICE)
    results = []

    for cfg in configs:
        beta = cfg["beta"]
        fb = cfg.get("free_bits", 0.0)
        dec_mult = cfg.get("decoder_mult", 1)
        latent_dim = cfg.get("latent_dim", ld)
        dec_hidden = HIDDEN_DIM * dec_mult

        name = f"VAE-{dataset_name}-LD{latent_dim}-B{beta}-FB{fb}-DM{dec_mult}"

        ckpt_path = CHECKPOINT_DIR / f"{name}.pt"
        if ckpt_path.exists():
            print(f"  [SKIP] {name}")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model = BetaVAE(obs_dim, latent_dim, beta=beta, hidden_dim=HIDDEN_DIM,
                           kl_warmup_epochs=WARMUP, free_bits=fb,
                           decoder_hidden_dim=dec_hidden)
            model.load_state_dict(ckpt["model_state"])
            model = model.to(DEVICE).eval()
        else:
            print(f"\n  {'='*48}")
            print(f"  {name}  obs_dim={obs_dim}")
            print(f"  {'='*48}")
            model = BetaVAE(obs_dim, latent_dim, beta=beta, hidden_dim=HIDDEN_DIM,
                           kl_warmup_epochs=WARMUP, free_bits=fb,
                           decoder_hidden_dim=dec_hidden)
            out = train_model(model, train, val, epochs=EPOCHS, batch_size=BATCH_SIZE,
                             device=DEVICE, model_name=name, lr=LR, patience=PATIENCE)
            model = out["model"].to(DEVICE).eval()

        kl, mse = evaluate_model(model, test_t)
        regime = "OK" if kl > 0.1 else ("LOW" if kl > 0.01 else "COLLAPSED")
        entry = {
            "name": name, "scenario": dataset_name, "obs_dim": obs_dim,
            "latent_dim": latent_dim, "beta": beta, "free_bits": fb,
            "decoder_mult": dec_mult, "kl": kl, "mse": mse,
            "rate_bits": kl * 1.442695, "regime": regime,
        }
        results.append(entry)
        print(f"  KL={kl:.4f} MSE={mse:.4f} [{regime}]")

    return results


# ═══════════════════════════════════════════════════════════════════
# Standard beta-VAE sweep on simple_spread (30-dim)
# ═══════════════════════════════════════════════════════════════════

def run_standard_sweep():
    betas = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 4.0, 10.0]
    lds = [8, 16]

    train, val, test, obs_dim = load_data("simple_spread")
    print(f"\nStandard sweep: simple_spread (obs_dim={obs_dim}, N={train.shape[0]})")

    configs = []
    for ld in lds:
        for b in betas:
            configs.append({"beta": b, "free_bits": 0.0, "decoder_mult": 1, "latent_dim": ld})

    results = run_configs("simple_spread", obs_dim, train, val, test, configs)

    out_path = ASSETS_DIR / "vae_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} standard sweep results to {out_path}")


# ═══════════════════════════════════════════════════════════════════
# High-dimensional beta-VAE sweep (40-90 dim)
# ═══════════════════════════════════════════════════════════════════

def run_hd_scaling():
    betas = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 4.0, 10.0]
    datasets = ["tag_hd", "spread_hd", "comm_hd", "spread_xhd"]
    all_results = []

    for ds in datasets:
        train, val, test, obs_dim = load_data(ds)
        print(f"\n{'#'*60}")
        print(f"HD scaling: {ds} (obs_dim={obs_dim}, N={train.shape[0]})")
        print(f"{'#'*60}")

        configs = [{"beta": b, "free_bits": 0.0, "decoder_mult": 1} for b in betas]
        results = run_configs(ds, obs_dim, train, val, test, configs)
        all_results.extend(results)

    out_path = ASSETS_DIR / "vae_hd_scaling_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} HD scaling results to {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Free-bits + decoder expansion anti-collapse sweep on spread_xhd
# ═══════════════════════════════════════════════════════════════════

def run_anti_collapse_sweep():
    free_bits_list = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
    dec_mult_list = [1, 2]
    betas = [0.1, 0.5, 1.0, 2.0, 4.0, 10.0]

    train, val, test, obs_dim = load_data("spread_xhd")
    print(f"\nAnti-collapse sweep: spread_xhd (obs_dim={obs_dim}, N={train.shape[0]})")

    configs = []
    for fb in free_bits_list:
        for b in betas:
            configs.append({"beta": b, "free_bits": fb, "decoder_mult": 1})

    for fb in [0.0, 0.25, 0.5, 1.0]:
        for b in betas:
            configs.append({"beta": b, "free_bits": fb, "decoder_mult": 2})

    # Remove duplicates
    seen = set()
    unique = []
    for c in configs:
        key = (c["beta"], c["free_bits"], c["decoder_mult"])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    print(f"Configs: {len(unique)}")
    results = run_configs("spread_xhd", obs_dim, train, val, test, unique)

    out_path = ASSETS_DIR / "collapse_barrier_full_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} anti-collapse sweep results to {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Cross-scenario FB=0.1 validation on tag_hd + comm_hd
# ═══════════════════════════════════════════════════════════════════

def run_cross_validation():
    betas = [0.5, 1.0, 2.0, 4.0, 10.0]
    all_results = []

    for scenario in ["tag_hd", "comm_hd"]:
        train, val, test, obs_dim = load_data(scenario)
        print(f"\n{'#'*60}")
        print(f"Cross-scenario: {scenario} (obs_dim={obs_dim}, N={train.shape[0]})")
        print(f"{'#'*60}")

        configs = []
        for fb in [0.0, 0.1]:
            for b in betas:
                configs.append({"beta": b, "free_bits": fb, "decoder_mult": 1})

        results = run_configs(scenario, obs_dim, train, val, test, configs)
        all_results.extend(results)

    out_path = ASSETS_DIR / "fb_cross_scenario_validation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} cross-scenario results to {out_path}")

    # Summary
    for scenario in ["tag_hd", "comm_hd"]:
        collapsed_fb0 = sum(1 for r in all_results if r["scenario"] == scenario and r["free_bits"] == 0.0 and r["kl"] < 0.1)
        collapsed_fb01 = sum(1 for r in all_results if r["scenario"] == scenario and r["free_bits"] == 0.1 and r["kl"] < 0.1)
        total = len([r for r in all_results if r["scenario"] == scenario and r["free_bits"] == 0.0])
        print(f"  {scenario}: FB=0.0 collapse={collapsed_fb0}/{total}, FB=0.1 collapse={collapsed_fb01}/{total}")


# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="beta-VAE training pipeline")
    parser.add_argument("--phase", choices=["standard", "hd_scaling", "anti_collapse", "cross_validation"],
                        help="Which experiment to run")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    if args.all:
        run_standard_sweep()
        run_hd_scaling()
        run_anti_collapse_sweep()
        run_cross_validation()
    elif args.phase == "standard":
        run_standard_sweep()
    elif args.phase == "hd_scaling":
        run_hd_scaling()
    elif args.phase == "anti_collapse":
        run_anti_collapse_sweep()
    elif args.phase == "cross_validation":
        run_cross_validation()
    else:
        print("Available experiments: --phase standard | hd_scaling | anti_collapse | cross_validation")
        print("Use --all to run all experiments")
        print("\nRunning standard sweep by default...")
        run_standard_sweep()
