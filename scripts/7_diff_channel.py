"""Phase 3.1 — Differentiable channel layer benchmark.

Train JSCC-BetaVAE codecs with differentiable AWGN and erasure channels,
comparing channel-in-the-loop training vs. post-hoc channel evaluation.

Usage: python scripts/7_diff_channel.py [--scenario simple_spread]
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from obscodec.config import DATA_DIR, ASSETS_DIR, CHECKPOINT_DIR, RANDOM_SEED
from obscodec.config import JSCC_SCENARIOS, JSCC_BETAS, JSCC_LATENT_DIM
from obscodec.config import DIFF_AWGN_SNR_TRAIN, DIFF_ERASURE_RATES, VAE_EPOCHS
from obscodec.models import BetaVAE, JSCCWrapper
from obscodec.channel.diff_channel import DiffAWGN, DiffErasure
from obscodec.channel.impairments import AWGNChannel, RayleighFadingChannel
from obscodec.trainer import train_model
from obscodec.metrics import evaluate_codec


def load_data(scenario: str):
    path = DATA_DIR / f"{scenario}_obs.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} — run scripts/1_collect_data.py first")
    return np.load(str(path))


def build_model(obs_dim, codec_type, beta, channel, device):
    if codec_type == "vae":
        base = BetaVAE(obs_dim, JSCC_LATENT_DIM, beta=beta,
                       free_bits=0.1, kl_warmup_epochs=150).to(device)
    else:
        raise ValueError(f"Unknown codec: {codec_type}")

    if channel is not None:
        return JSCCWrapper(base, channel).to(device)
    return base


def main():
    parser = argparse.ArgumentParser(description="Phase 3.1 — Diff channel benchmark")
    parser.add_argument("--scenario", default="simple_spread", choices=JSCC_SCENARIOS)
    parser.add_argument("--epochs", type=int, default=VAE_EPOCHS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    data = load_data(args.scenario)
    obs_dim = data.shape[1]
    train_n = int(len(data) * 0.8)
    val_n = int(len(data) * 0.1)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(data))
    train_data = data[idx[:train_n]]
    val_data = data[idx[train_n:train_n + val_n]]
    test_data = data[idx[train_n + val_n:]]

    print(f"Scenario: {args.scenario} ({obs_dim} dim)")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"Device: {device}")

    results = []

    # ── Baseline: no channel ──
    print("\n=== Baseline (no channel) ===")
    model = build_model(obs_dim, "vae", 2.0, None, device)
    ckpt_name = f"jscc_diff_baseline_{args.scenario}"
    ckpt_path = CHECKPOINT_DIR / f"{ckpt_name}.pt"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        train_model(model, train_data, val_data, device, epochs=args.epochs)
        torch.save({"model_state": model.state_dict(), "history": {}}, ckpt_path)

    r = evaluate_codec(model, test_data, device)
    r.update({"scenario": args.scenario, "channel": "none", "snr": "clean",
              "codec": "vae", "beta": 2.0})
    results.append(r)
    print(f"  Clean: MSE={r['mse']:.4f}, KL={r.get('kl', 0):.4f}")

    # ── DiffAWGN training ──
    for snr in [20, 10, 5, 0]:
        print(f"\n=== DiffAWGN SNR={snr}dB ===")
        diff_awgn = DiffAWGN(snr_db=snr)
        model = build_model(obs_dim, "vae", 2.0, diff_awgn, device)
        ckpt_name = f"jscc_diff_awgn{snr}_{args.scenario}"
        ckpt_path = CHECKPOINT_DIR / f"{ckpt_name}.pt"

        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            print(f"  Loaded checkpoint: {ckpt_path}")
        else:
            train_model(model, train_data, val_data, device, epochs=args.epochs)
            torch.save({"model_state": model.state_dict(), "history": {}}, ckpt_path)

        # Test on clean + matched channel + mismatched (Rayleigh)
        for test_snr in ["clean", 20, 10, 5, 0]:
            model.eval()
            test_t = torch.FloatTensor(test_data).to(device)
            with torch.no_grad():
                if test_snr == "clean":
                    x_hat = model.base.forward(test_t)[0]
                elif isinstance(test_snr, (int, float)):
                    # Use evaluation AWGN for fair comparison
                    eval_ch = AWGNChannel(snr_db=test_snr)
                    z, _, _ = model.base.encode(test_t)
                    z_noisy = eval_ch(z)
                    x_hat = model.base.decode(z_noisy)
                mse = float(torch.nn.functional.mse_loss(x_hat, test_t).item())

            r = {"scenario": args.scenario, "channel": "diff_awgn",
                 "train_snr": snr, "test_snr": test_snr, "mse": mse,
                 "codec": "jscc_vae", "beta": 2.0}
            results.append(r)
            print(f"  Test SNR={test_snr}: MSE={mse:.4f}")

    # ── DiffErasure training ──
    for loss_rate in [0.1, 0.3]:
        print(f"\n=== DiffErasure rate={loss_rate} ===")
        diff_eras = DiffErasure(loss_rate=loss_rate)
        model = build_model(obs_dim, "vae", 2.0, diff_eras, device)
        ckpt_name = f"jscc_diff_eras{int(loss_rate*100)}_{args.scenario}"
        ckpt_path = CHECKPOINT_DIR / f"{ckpt_name}.pt"

        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            print(f"  Loaded checkpoint: {ckpt_path}")
        else:
            train_model(model, train_data, val_data, device, epochs=args.epochs)
            torch.save({"model_state": model.state_dict(), "history": {}}, ckpt_path)

        # Test on clean + matched + mismatched loss rates
        for test_rate in [0.0, 0.1, 0.3]:
            model.eval()
            test_t = torch.FloatTensor(test_data).to(device)
            with torch.no_grad():
                if test_rate == 0.0:
                    x_hat = model.base.forward(test_t)[0]
                else:
                    z, _, _ = model.base.encode(test_t)
                    mask = torch.bernoulli(torch.full_like(z, 1 - test_rate))
                    x_hat = model.base.decode(z * mask / (1 - test_rate + 1e-8))
                mse = float(torch.nn.functional.mse_loss(x_hat, test_t).item())

            r = {"scenario": args.scenario, "channel": "diff_erasure",
                 "train_loss_rate": loss_rate, "test_loss_rate": test_rate,
                 "mse": mse, "codec": "jscc_vae", "beta": 2.0}
            results.append(r)
            print(f"  Test loss_rate={test_rate}: MSE={mse:.4f}")

    # ── Save ──
    out_path = ASSETS_DIR / "diff_channel_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path} ({len(results)} entries)")


if __name__ == "__main__":
    main()
