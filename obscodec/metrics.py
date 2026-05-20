"""Codec evaluation metrics: MSE, KL, rate, codebook usage, regime classification."""

import torch
import torch.nn.functional as F

from .config import COLLAPSE_KL_THRESHOLD, NATS_TO_BITS


def evaluate_codec(model, test_data, device, test_loader=None):
    """Evaluate a codec on test data and return a metrics dict."""
    model.eval()
    test_t = test_data.to(device) if isinstance(test_data, torch.Tensor) else torch.FloatTensor(test_data).to(device)

    with torch.no_grad():
        out = model(test_t)
        if isinstance(out, tuple):
            x_hat = out[0]
        else:
            x_hat = out

        mse = float(F.mse_loss(x_hat, test_t).item())

    result = {"mse": mse}

    # KL divergence (VAE)
    if hasattr(model, "kl_nats"):
        try:
            result["kl"] = float(model.kl_nats(test_t))
        except Exception:
            result["kl"] = float("nan")

    # Rate estimate
    if hasattr(model, "get_rate_estimate") and test_loader is not None:
        try:
            result["rate_bits"] = float(model.get_rate_estimate(test_loader, device))
        except Exception:
            result["rate_bits"] = float("nan")

    # Codebook usage (VQ-VAE)
    if hasattr(model, "codebook_usage"):
        try:
            result["codebook_usage"] = float(model.codebook_usage())
        except Exception:
            pass

    # Regime classification
    kl = result.get("kl", float("nan"))
    if kl < 0.01:
        result["regime"] = "COLLAPSED"
    elif kl < 0.1:
        result["regime"] = "LOW"
    else:
        result["regime"] = "OK"

    return result


def compute_rate_distortion(codecs, test_data, device):
    """Evaluate a dict of codecs and return rate-distortion pairs sorted by rate.

    Args:
        codecs: dict of {name: model}
        test_data: FloatTensor (N, obs_dim)
        device: torch device

    Returns:
        list of {name, mse, rate_bits, kl, regime} sorted by rate_bits
    """
    results = []
    for name, model in codecs.items():
        r = evaluate_codec(model, test_data, device)
        r["name"] = name
        results.append(r)

    results.sort(key=lambda x: x.get("rate_bits", x.get("total_bits", float("inf"))))
    return results


def benchmark_all_codecs(codec_dict, test_data, device):
    """Full benchmark: MSE, rate, AWGN + packet-loss channel robustness.

    Returns:
        dict with keys: mse, rate, awgn, packet_loss
    """
    from .channel.impairments import evaluate_channel_robustness

    results = {}

    for name, model in codec_dict.items():
        entry = evaluate_codec(model, test_data, device)

        # AWGN robustness
        awgn = evaluate_channel_robustness(model, test_data, device, "awgn")
        entry["awgn"] = awgn

        # Packet loss robustness
        pl = evaluate_channel_robustness(model, test_data, device, "packet_loss")
        entry["packet_loss"] = pl

        results[name] = entry

    return results
