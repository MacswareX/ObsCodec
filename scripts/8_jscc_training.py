"""Phase 3.2 — Joint Source-Channel Coding (JSCC) training benchmark.

Train JSCC-wrapped codecs with differentiable channels in the loop,
comparing BetaVAE (β=0.1, 2.0) and VQVAE across scenarios and channel types.
Tests clean, matched, and mismatched channel conditions.

Usage: python scripts/8_jscc_training.py [--scenario simple_spread]
"""

import sys, json, argparse, itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import numpy as np

from obscodec.config import (DATA_DIR, ASSETS_DIR, CHECKPOINT_DIR, RANDOM_SEED,
                              JSCC_SCENARIOS, JSCC_BETAS, JSCC_LATENT_DIM,
                              DIFF_AWGN_SNR_TRAIN, DIFF_ERASURE_RATES, VAE_EPOCHS,
                              VQ_CODEBOOK_SIZE)
from obscodec.models import BetaVAE, VQVAE, JSCCWrapper
from obscodec.channel.diff_channel import DiffAWGN, DiffErasure
from obscodec.channel.impairments import AWGNChannel
from obscodec.trainer import train_model
from obscodec.data.synthetic import (SYNTHETIC_GENERATORS, SYNTHETIC_GENERATORS_HD,
                                     collect_synthetic_dataset_hd)


def load_data(scenario: str) -> np.ndarray:
    """Load scenario data, generating HD variants on demand."""
    path = DATA_DIR / f"{scenario}_obs.npy"
    if not path.exists():
        if scenario in SYNTHETIC_GENERATORS_HD:
            print(f"  Generating {scenario} data on demand...")
            collect_synthetic_dataset_hd([scenario], save=True)
        elif scenario in SYNTHETIC_GENERATORS:
            raise FileNotFoundError(
                f"{path} — run scripts/1_collect_data.py first")
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    return np.load(str(path))


def build_model_and_channel(obs_dim, codec_type, beta_or_cb, free_bits,
                            channel_type, channel_param, device):
    """Build the base codec and optional channel. Returns (model, channel_obj_or_None)."""
    # Build base codec
    if codec_type == "vae":
        base = BetaVAE(obs_dim, JSCC_LATENT_DIM, beta=beta_or_cb,
                       free_bits=free_bits, kl_warmup_epochs=150).to(device)
    elif codec_type == "vqvae":
        base = VQVAE(obs_dim, JSCC_LATENT_DIM,
                     num_embeddings=beta_or_cb).to(device)
    else:
        raise ValueError(f"Unknown codec: {codec_type}")

    # Build channel (None = no channel / baseline)
    channel = None
    if channel_type == "awgn":
        channel = DiffAWGN(snr_db=channel_param)
    elif channel_type == "erasure":
        channel = DiffErasure(loss_rate=channel_param)

    model = JSCCWrapper(base, channel) if channel is not None else base
    return model, channel


def make_checkpoint_name(scenario, codec_type, beta_or_cb, free_bits,
                         channel_type, channel_param):
    """Deterministic checkpoint name for a config."""
    parts = ["jscc", scenario, str(codec_type)]
    if codec_type == "vae":
        parts.append(f"b{beta_or_cb}")
    else:
        parts.append(f"cb{beta_or_cb}")
    parts.append(f"fb{free_bits}")
    if channel_type and channel_param is not None:
        parts.append(f"{channel_type}{channel_param}")
    else:
        parts.append("clean")
    return "_".join(str(p) for p in parts)


def evaluate_on_channel(model, test_data, device, channel_type, channel_param):
    """Evaluate model (JSCCWrapper or bare) on a specific channel condition."""
    model.eval()
    test_t = torch.FloatTensor(test_data).to(device)

    with torch.no_grad():
        if channel_type == "clean":
            out = model(test_t)
            x_hat = out[0] if isinstance(out, tuple) else out
        elif channel_type == "awgn":
            # Decode through evaluation AWGN
            eval_ch = AWGNChannel(snr_db=channel_param)
            if hasattr(model, "base") and hasattr(model, "channel"):
                # JSCCWrapper: encode, apply eval channel, decode
                if hasattr(model.base, "reparameterize"):
                    mu, logvar = model.base.encoder(test_t)
                    z = model.base.reparameterize(mu, logvar)
                    z_noisy = eval_ch(z)
                    x_hat = model.base.decode(z_noisy)
                elif hasattr(model.base, "vq"):
                    z_e = model.base.encoder(test_t)
                    z_q, _, _ = model.base.vq(z_e)
                    z_noisy = eval_ch(z_q)
                    x_hat = model.base.decode(z_noisy)
                else:
                    z = model.base.encode(test_t)
                    if isinstance(z, tuple):
                        z = z[0]
                    z_noisy = eval_ch(z)
                    x_hat = model.base.decode(z_noisy)
            elif hasattr(model, "reparameterize"):
                mu, logvar = model.encoder(test_t)
                z = model.reparameterize(mu, logvar)
                z_noisy = eval_ch(z)
                x_hat = model.decode(z_noisy)
            else:
                z = model.encode(test_t)
                if isinstance(z, tuple):
                    z = z[0]
                z_noisy = eval_ch(z)
                x_hat = model.decode(z_noisy)
        elif channel_type == "erasure":
            if hasattr(model, "base"):
                inner = model.base
            else:
                inner = model
            if hasattr(inner, "reparameterize"):
                mu, logvar = inner.encoder(test_t)
                z = inner.reparameterize(mu, logvar)
            elif hasattr(inner, "vq"):
                z_e = inner.encoder(test_t)
                z, _, _ = inner.vq(z_e)
            else:
                z = inner.encode(test_t)
                if isinstance(z, tuple):
                    z = z[0]
            mask = torch.bernoulli(torch.full_like(z, 1 - channel_param))
            z_noisy = z * mask / (1 - channel_param + 1e-8)
            x_hat = inner.decode(z_noisy)
        else:
            raise ValueError(f"Unknown test channel: {channel_type}")

        mse = float(F.mse_loss(x_hat, test_t).item())

    # KL
    kl = float("nan")
    inner = model.base if hasattr(model, "base") else model
    if hasattr(inner, "kl_nats"):
        try:
            kl = inner.kl_nats(test_t)
        except Exception:
            pass

    return mse, kl


def main():
    parser = argparse.ArgumentParser(description="Phase 3.2 — JSCC training benchmark")
    parser.add_argument("--scenario", default=None,
                        choices=JSCC_SCENARIOS + [None],
                        help="Single scenario or all (default)")
    parser.add_argument("--epochs", type=int, default=VAE_EPOCHS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    device = args.device
    scenarios = [args.scenario] if args.scenario else JSCC_SCENARIOS

    # Config grid
    codec_configs = [
        ("vae", 0.1, "β=0.1"),
        ("vae", 2.0, "β=2.0"),
        ("vqvae", VQ_CODEBOOK_SIZE, "VQVAE"),
    ]
    free_bits_list = [0.0, 0.1]
    # Train channels: (type, param, label)
    train_channels = [
        (None, None, "clean"),
        ("awgn", 20, "AWGN 20dB"),
        ("awgn", 10, "AWGN 10dB"),
        ("awgn", 0, "AWGN 0dB"),
        ("erasure", 0.1, "Erasure 10%"),
        ("erasure", 0.3, "Erasure 30%"),
    ]
    # Test channels: clean + matched + mismatched
    test_channels = [
        ("clean", None, "clean"),
        ("awgn", 20, "AWGN 20dB"),
        ("awgn", 10, "AWGN 10dB"),
        ("awgn", 0, "AWGN 0dB"),
        ("awgn", -5, "AWGN -5dB"),
        ("erasure", 0.0, "Erasure 0%"),
        ("erasure", 0.1, "Erasure 10%"),
        ("erasure", 0.3, "Erasure 30%"),
    ]

    all_results = []

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario}")
        print(f"{'='*60}")

        try:
            data = load_data(scenario)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        obs_dim = data.shape[1]
        train_n = int(len(data) * 0.8)
        val_n = int(len(data) * 0.1)
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.permutation(len(data))
        train_data = data[idx[:train_n]]
        val_data = data[idx[train_n:train_n + val_n]]
        test_data = data[idx[train_n + val_n:]]

        print(f"  Obs dim: {obs_dim}, Train: {len(train_data)}, "
              f"Val: {len(val_data)}, Test: {len(test_data)}")

        # Precompute baseline reconstructions for relative metrics
        test_tensor = torch.FloatTensor(test_data).to(device)
        data_variance = float(test_tensor.var().item())

        for (codec_type, beta_or_cb, codec_label), fb in \
                itertools.product(codec_configs, free_bits_list):

            # Skip VQVAE + free_bits (FB doesn't apply to VQ-VAE)
            if codec_type == "vqvae" and fb > 0:
                continue

            for ch_type, ch_param, ch_label in train_channels:
                ckpt_name = make_checkpoint_name(
                    scenario, codec_type, beta_or_cb, fb, ch_type, ch_param)
                ckpt_path = CHECKPOINT_DIR / f"{ckpt_name}.pt"

                config_tag = (f"{codec_label} FB={fb} "
                              f"train_ch={ch_label}")

                print(f"\n--- {config_tag} ---")

                # Build and train
                try:
                    model, _ = build_model_and_channel(
                        obs_dim, codec_type, beta_or_cb, fb,
                        ch_type, ch_param, device)
                except Exception as e:
                    print(f"  SKIP build: {e}")
                    continue

                if ckpt_path.exists() and args.skip_existing:
                    ckpt = torch.load(ckpt_path, map_location=device)
                    model.load_state_dict(ckpt["model_state"])
                    print(f"  Loaded: {ckpt_path}")
                else:
                    train_model(model, train_data, val_data, device,
                                epochs=args.epochs, model_name=ckpt_name,
                                verbose=(scenario == scenarios[0]))
                    # re-save in expected format
                    torch.save({"model_state": model.state_dict(),
                                "history": {}}, ckpt_path)

                # Evaluate across test channels
                for test_ch, test_ch_param, test_ch_label in test_channels:
                    mse, kl = evaluate_on_channel(
                        model, test_data, device, test_ch, test_ch_param)

                    r = {
                        "scenario": scenario,
                        "obs_dim": obs_dim,
                        "codec": codec_type,
                        "codec_label": codec_label,
                        "beta": beta_or_cb if codec_type == "vae" else None,
                        "codebook_size": beta_or_cb if codec_type == "vqvae" else None,
                        "free_bits": fb,
                        "train_channel": ch_label,
                        "train_channel_type": ch_type,
                        "train_channel_param": ch_param,
                        "test_channel": test_ch_label,
                        "test_channel_type": test_ch,
                        "test_channel_param": test_ch_param,
                        "mse": mse,
                        "kl": kl,
                        "data_variance": data_variance,
                        "nmse": mse / (data_variance + 1e-8),
                    }
                    all_results.append(r)
                    print(f"  Test {test_ch_label:14s}: MSE={mse:.4f}  "
                          f"KL={kl:.3f}")

    # Save results
    out_path = ASSETS_DIR / "jscc_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path} ({len(all_results)} entries)")


if __name__ == "__main__":
    main()
