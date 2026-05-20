"""Channel impairment simulations for semantic communication benchmarking.

Models (applied to latent representation z):
  - AWGN: y = z + n,  n ~ N(0, σ²)
  - Rayleigh fading: y = h·z + n,  |h| ~ Rayleigh(1)
  - Packet loss: random dimension dropout
  - Heterogeneous: different SNR per agent-slot in observation
  - Composite: chain multiple impairments (e.g. fading + AWGN)
"""

import torch
import torch.nn.functional as F

from ..config import AWGN_SNR_RANGE, PACKET_LOSS_RATES


# ═══════════════════════════════════════════════════════════════════
# Basic channel models
# ═══════════════════════════════════════════════════════════════════

class AWGNChannel:
    """Additive White Gaussian Noise.

    SNR in dB. σ² = signal_power / 10^(SNR/10).
    """

    def __init__(self, snr_db: float = 10.0):
        self.snr_db = snr_db

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        if self.snr_db is None or self.snr_db == float("inf"):
            return z
        signal_power = z.pow(2).mean()
        noise_power = signal_power / (10 ** (self.snr_db / 10))
        noise = torch.randn_like(z) * torch.sqrt(noise_power + 1e-20)
        return z + noise


class RayleighFadingChannel:
    """Multiplicative Rayleigh fading + AWGN — simulates multi-path.

    y = h·z + n
    |h| ~ Rayleigh(scale=1.0)  i.e. h² ~ Exponential(1)
    SNR defined as E[|h·z|²] / E[|n|²].

    fading_mode:
      - "iid": each dimension fades independently (worst case)
      - "block": all dims share same fade coefficient (coherent fade)
      - "agent_blocks": each agent's latent block shares a fade coefficient
    """

    def __init__(self, snr_db: float = 10.0, fading_mode: str = "block"):
        self.snr_db = snr_db
        self.fading_mode = fading_mode

    def __call__(self, z: torch.Tensor, agent_boundaries: list[int] | None = None) -> torch.Tensor:
        if self.snr_db is None or self.snr_db == float("inf"):
            return z

        B, D = z.shape

        # Generate fading coefficients
        if self.fading_mode == "iid":
            h = torch.randn(B, D).abs() / (2 ** 0.5)  # Rayleigh(1) per dim
        elif self.fading_mode == "agent_blocks" and agent_boundaries is not None:
            h = torch.ones(B, D, device=z.device)
            for i in range(len(agent_boundaries) - 1):
                start, end = agent_boundaries[i], agent_boundaries[i + 1]
                fade = torch.randn(B, 1).abs() / (2 ** 0.5)
                h[:, start:end] = fade
        else:  # "block" — all dims fade together
            fade = torch.randn(B, 1).abs() / (2 ** 0.5)
            h = fade.expand(B, D)

        h = h.to(z.device)

        # Generate AWGN
        signal_power = (h * z).pow(2).mean()
        noise_power = signal_power / (10 ** (self.snr_db / 10))
        noise = torch.randn_like(z) * torch.sqrt(noise_power + 1e-20)

        return h * z + noise


class PacketLossChannel:
    """Randomly zeros out a fraction of latent dimensions.

    Surviving dims scaled by 1/(1-rate) for unbiased energy.
    """

    def __init__(self, loss_rate: float = 0.1):
        self.loss_rate = loss_rate

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        if self.loss_rate == 0.0:
            return z
        mask = torch.bernoulli(torch.full_like(z, 1 - self.loss_rate))
        return z * mask / (1 - self.loss_rate + 1e-8)


class BurstPacketLossChannel:
    """Burst packet loss: entire contiguous blocks are lost together.

    Simulates multi-agent scenarios where an entire agent's message is lost
    (e.g. agent goes out of range) rather than random bit flips.
    """

    def __init__(self, loss_rate: float = 0.1, block_size: int = 6):
        self.loss_rate = loss_rate
        self.block_size = block_size

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        if self.loss_rate == 0.0:
            return z
        B, D = z.shape
        n_blocks = max(1, D // self.block_size)
        mask = torch.ones(B, D, device=z.device)
        for b in range(n_blocks):
            if torch.rand(1).item() < self.loss_rate:
                start = b * self.block_size
                end = min(start + self.block_size, D)
                mask[:, start:end] = 0
        return z * mask / (1 - self.loss_rate + 1e-8)


# ═══════════════════════════════════════════════════════════════════
# Heterogeneous per-agent channel
# ═══════════════════════════════════════════════════════════════════

class HeterogeneousChannel:
    """Different channel quality for different agent observation slots.

    Each agent's portion of the observation (or latent code) experiences
    a different SNR, simulating varying distances/base-station links.

    Args:
        agent_dims: List of (start, end) index pairs for each agent's slot.
        snr_range: (min_snr, max_snr) in dB — SNRs are linearly spaced
                   across agents. First agent gets max_snr, last gets min_snr.
    """

    def __init__(self, agent_dims: list[tuple[int, int]],
                 snr_range: tuple[float, float] = (0.0, 20.0)):
        self.agent_dims = agent_dims
        self.snr_range = snr_range
        n = len(agent_dims)
        self.agent_snrs = [
            snr_range[1] - (snr_range[1] - snr_range[0]) * i / max(n - 1, 1)
            for i in range(n)
        ]

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """Apply per-agent SNR to each segment of the latent code."""
        z_noisy = z.clone()
        for (start, end), snr in zip(self.agent_dims, self.agent_snrs):
            if start >= z.shape[1]:
                break
            actual_end = min(end, z.shape[1])
            segment = z[:, start:actual_end]
            signal_power = segment.pow(2).mean() + 1e-20
            noise_power = signal_power / (10 ** (snr / 10))
            noise = torch.randn_like(segment) * torch.sqrt(noise_power)
            z_noisy[:, start:actual_end] = segment + noise
        return z_noisy


# ═══════════════════════════════════════════════════════════════════
# Composite channel
# ═══════════════════════════════════════════════════════════════════

class CompositeChannel:
    """Chain multiple impairments in sequence.

    Example: fading → AWGN → packet_loss
    """

    def __init__(self, impairments: list):
        self.impairments = impairments

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        for imp in self.impairments:
            z = imp(z)
        return z


# ═══════════════════════════════════════════════════════════════════
# Evaluation utilities
# ═══════════════════════════════════════════════════════════════════

def get_agent_boundaries(obs_dim: int, agent_spec: list[int]) -> list[int]:
    """Compute agent slot boundaries from per-agent dimension counts.

    Args:
        obs_dim: Total observation dimension.
        agent_spec: List of dimension counts per agent.
                    e.g. [6, 6, 6, 6, 6] for 5 equal agents.

    Returns:
        List of boundary indices [0, 6, 12, 18, 24, 30].
    """
    boundaries = [0]
    for d in agent_spec:
        boundaries.append(boundaries[-1] + d)
    assert boundaries[-1] <= obs_dim, f"Agent spec sum {boundaries[-1]} > obs_dim {obs_dim}"
    return boundaries


def evaluate_channel_sweep(model, test_data: torch.Tensor, device,
                           channel_builder, param_name: str,
                           param_values: list,
                           agent_boundaries: list[int] | None = None) -> dict:
    """Generic sweep over a channel parameter.

    Args:
        model: Codec model.
        test_data: Float tensor (N, obs_dim).
        device: torch device.
        channel_builder: fn(param_value) -> channel callable.
        param_name: e.g. "SNR", "fading_SNR", "PLR".
        param_values: List of parameter values to sweep.
        agent_boundaries: Optional list for agent-aware channels.

    Returns:
        {f"{param_name}_{val}": {"mse": float, "kl": float | None}, ...}
    """
    model.eval()
    results = {}
    test_data = test_data.to(device)

    for val in param_values:
        channel = channel_builder(val)
        with torch.no_grad():
            out = model.encode(test_data)
            if isinstance(out, tuple):
                z = out[0]
            else:
                z = out

            if agent_boundaries and hasattr(channel, '__call__'):
                import inspect
                sig = inspect.signature(channel.__call__)
                if 'agent_boundaries' in sig.parameters:
                    z_noisy = channel(z, agent_boundaries=agent_boundaries)
                else:
                    z_noisy = channel(z)
            else:
                z_noisy = channel(z)

            x_hat = model.decode(z_noisy)
            mse = F.mse_loss(x_hat, test_data).item()

        entry = {"mse": float(mse)}
        if hasattr(model, "kl_nats"):
            entry["kl"] = model.kl_nats(test_data)
        results[f"{param_name}_{val}"] = entry

    return results


def evaluate_channel_robustness(model, test_data, device,
                                channel_type="awgn",
                                snr_values=None, loss_rates=None,
                                agent_boundaries=None):
    """Backward-compatible wrapper — now with agent support."""
    if channel_type == "awgn":
        values = snr_values or AWGN_SNR_RANGE
        return evaluate_channel_sweep(
            model, test_data, device,
            lambda v: AWGNChannel(v), "SNR", values,
            agent_boundaries=agent_boundaries,
        )
    elif channel_type == "fading":
        values = snr_values or AWGN_SNR_RANGE
        return evaluate_channel_sweep(
            model, test_data, device,
            lambda v: RayleighFadingChannel(v, fading_mode="agent_blocks"),
            "fading_SNR", values,
            agent_boundaries=agent_boundaries,
        )
    elif channel_type == "packet_loss":
        values = loss_rates or PACKET_LOSS_RATES
        return evaluate_channel_sweep(
            model, test_data, device,
            lambda v: PacketLossChannel(v), "PLR", values,
            agent_boundaries=agent_boundaries,
        )
    elif channel_type == "burst_loss":
        values = loss_rates or PACKET_LOSS_RATES
        block_size = agent_boundaries[1] - agent_boundaries[0] if agent_boundaries else 6
        return evaluate_channel_sweep(
            model, test_data, device,
            lambda v: BurstPacketLossChannel(v, block_size=block_size),
            "burst_PLR", values,
            agent_boundaries=agent_boundaries,
        )
    elif channel_type == "heterogeneous":
        # Sweep the snr_range width
        results = {}
        for snr_lo in [20, 15, 10, 5, 0, -5]:
            ch = HeterogeneousChannel(
                [(b, agent_boundaries[i+1]) for i, b in enumerate(agent_boundaries[:-1])],
                snr_range=(snr_lo, 20),
            )
            with torch.no_grad():
                out = model.encode(test_data.to(device))
                z = out[0] if isinstance(out, tuple) else out
                z_noisy = ch(z)
                x_hat = model.decode(z_noisy)
                mse = F.mse_loss(x_hat, test_data.to(device)).item()
            results[f"het_SNR_{snr_lo}_20"] = {"mse": float(mse)}
        return results
    else:
        raise ValueError(f"Unknown channel_type: {channel_type}")
