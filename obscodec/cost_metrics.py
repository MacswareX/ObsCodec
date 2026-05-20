"""Cost and resource metrics: parameters, FLOPs, latency, throughput.

Provides model-agnostic measurement functions that work across BetaVAE,
VQ-VAE, standard AE, PCA, and digital quantization codecs.
"""

import time
import torch
import numpy as np
from typing import Dict, List


def count_parameters(model: torch.nn.Module) -> Dict:
    """Count total and trainable parameters.

    Returns:
        dict with keys: total_params, trainable_params, encoder_params, decoder_params
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    result = {"total_params": total, "trainable_params": trainable}

    # Per-component breakdown for codec models
    if hasattr(model, "encoder"):
        enc = sum(p.numel() for p in model.encoder.parameters())
        result["encoder_params"] = enc
    if hasattr(model, "decoder"):
        dec = sum(p.numel() for p in model.decoder.parameters())
        result["decoder_params"] = dec
    if hasattr(model, "vq"):
        vq = sum(p.numel() for p in model.vq.parameters())
        result["codebook_params"] = vq

    return result


def estimate_flops(
    model: torch.nn.Module, input_dim: int, batch_size: int = 1
) -> Dict:
    """Estimate MACs and FLOPs for a forward pass via PyTorch profiler.

    Uses torch.profiler for accurate op-level counting. Falls back to
    analytic estimation for Linear/Conv layers if profiling is unavailable.

    Returns:
        dict with keys: macs, flops
    """
    try:
        from torch.profiler import profile, ProfilerActivity

        device = next(model.parameters()).device
        x = torch.randn(batch_size, input_dim, device=device)

        model.eval()
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                _ = model(x)

        total_macs = 0
        for evt in prof.key_averages():
            if hasattr(evt, "flops"):
                total_macs += evt.flops

        if total_macs > 0:
            return {"macs": total_macs / batch_size, "flops": 2 * total_macs / batch_size}
    except Exception:
        pass

    # Analytic fallback: count Linear layer MACs
    total_macs = _analytic_linear_macs(model, input_dim, batch_size)
    return {"macs": total_macs / batch_size, "flops": 2 * total_macs / batch_size}


def _analytic_linear_macs(model: torch.nn.Module, input_dim: int, batch_size: int) -> int:
    """Estimate MACs for feed-forward Linear layers only."""
    total = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            total += module.in_features * module.out_features * batch_size
            # Bias add
            if module.bias is not None:
                total += module.out_features * batch_size
        # BatchNorm: ~2 operations per element (mean + variance)
        if isinstance(module, torch.nn.BatchNorm1d):
            total += module.num_features * batch_size * 2
    return total


def measure_inference_latency(
    model: torch.nn.Module,
    test_input: torch.Tensor,
    n_warmup: int = 50,
    n_measure: int = 200,
    device: str = "cpu",
) -> Dict:
    """Measure inference latency (mean, std, p50, p95, p99).

    Args:
        model: The codec model (must be on target device).
        test_input: A single sample tensor of shape (1, obs_dim).
        n_warmup: Warmup iterations before timing.
        n_measure: Measurement iterations.
        device: Target device string.

    Returns:
        dict with keys: latency_ms_mean, latency_ms_std, latency_ms_p50,
                        latency_ms_p95, latency_ms_p99
    """
    model = model.to(device)
    model.eval()
    x = test_input.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)

    # Synchronize before timing (CUDA only)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    timings: List[float] = []
    with torch.no_grad():
        for _ in range(n_measure):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000)  # ms

    arr = np.array(timings)
    return {
        "latency_ms_mean": float(np.mean(arr)),
        "latency_ms_std": float(np.std(arr)),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_p99": float(np.percentile(arr, 99)),
    }


def measure_throughput(
    model: torch.nn.Module,
    test_input: torch.Tensor,
    batch_sizes: List[int] = None,
    n_repeats: int = 10,
    device: str = "cpu",
) -> Dict:
    """Measure throughput (samples/sec) across batch sizes.

    Args:
        model: The codec model.
        test_input: A single sample tensor of shape (1, obs_dim).
        batch_sizes: List of batch sizes to test. Default: [1, 8, 32, 128].
        n_repeats: Measurement repeats per batch size.
        device: Target device string.

    Returns:
        dict mapping batch_size label to {samples_per_second, batch_size, total_time_s}
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 128, 256]

    model = model.to(device)
    model.eval()
    obs_dim = test_input.shape[1]

    results = {}
    for bs in batch_sizes:
        x = test_input.expand(bs, obs_dim).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)

        if device.startswith("cuda"):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_repeats):
                _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_samples = bs * n_repeats
        total_time = t1 - t0
        results[f"bs{bs}"] = {
            "batch_size": bs,
            "samples_per_second": float(total_samples / total_time),
            "total_time_s": float(total_time),
        }

    return results


def profile_all(
    model: torch.nn.Module,
    test_input: torch.Tensor,
    device: str = "cpu",
    batch_sizes: List[int] = None,
) -> Dict:
    """Run full cost profile: params, FLOPs, latency, throughput.

    Args:
        model: Trained codec model.
        test_input: Single sample tensor (1, input_dim).
        device: Device to profile on.
        batch_sizes: Throughput batch sizes.

    Returns:
        Combined dict with all cost metrics.
    """
    obs_dim = test_input.shape[1]

    result = {}

    # Parameters
    result["params"] = count_parameters(model)

    # FLOPs
    result["flops"] = estimate_flops(model, obs_dim, batch_size=1)

    # Latency
    result["latency"] = measure_inference_latency(
        model, test_input, device=device
    )

    # Throughput
    if batch_sizes is not None:
        result["throughput"] = measure_throughput(
            model, test_input, batch_sizes=batch_sizes, device=device
        )

    return result
