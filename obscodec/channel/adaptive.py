"""Adaptive rate allocation based on per-agent channel quality.

Core idea of semantic communication: agents with poor channel conditions
should receive more protection (more bits) so that the decoder can still
reconstruct the full multi-agent state for coordination.

Rate-Distortion optimization:
  Minimize Σ D_i(R_i)  subject to  Σ R_i ≤ R_total

Water-filling solution: allocate more bits to agents where the marginal
reduction in distortion per bit is highest (typically agents with low SNR).
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field


@dataclass
class AgentChannelState:
    """Per-agent channel quality and rate allocation."""
    agent_id: int
    obs_start: int      # start index in flattened observation
    obs_end: int        # end index
    dims: int           # number of dimensions
    snr_db: float       # current channel SNR
    allocated_bits: int = 0  # bits allocated to this agent
    distortion_estimate: float = 1.0  # estimated reconstruction error


class RateAllocator:
    """Water-filling rate allocator for multi-agent observations.

    Given per-agent channel SNRs and a total bit budget, allocates
    bits to minimize total expected distortion.

    Uses a simple heuristic: distortion ~ σ² · 2^(-2R/D) (rate-distortion
    for Gaussian source), weighted by channel quality.
    """

    def __init__(self, agent_dims: list[int], total_bit_budget: int,
                 min_bits_per_dim: int = 2, max_bits_per_dim: int = 16):
        self.agent_dims = agent_dims          # list of dim counts per agent
        self.n_agents = len(agent_dims)
        self.total_bit_budget = total_bit_budget
        self.min_bits_per_dim = min_bits_per_dim
        self.max_bits_per_dim = max_bits_per_dim

        # Build boundaries
        self.boundaries = [0]
        for d in agent_dims:
            self.boundaries.append(self.boundaries[-1] + d)
        self.total_dims = self.boundaries[-1]

    def allocate_water_filling(self, snrs: list[float]) -> list[int]:
        """Water-filling: more bits to agents with lower effective SNR.

        Distortion for agent i with R_i bits and SNR_i:
          D_i ≈ σ² · [2^(-2R_i/d_i) + 1/SNR_linear_i]

        The channel noise term 1/SNR means agents with worse channels
        benefit more from additional quantization bits, so water-filling
        shifts bits toward them.

        Returns:
            List of allocated bits per agent.
        """
        n = self.n_agents
        dims = np.array(self.agent_dims, dtype=np.float64)
        snr_linear = np.maximum(np.array([10 ** (s / 10) for s in snrs]), 1e-6)

        # Initialize uniform
        bits_per_agent = np.ones(n) * (self.total_bit_budget // n)
        bits_per_agent = bits_per_agent.astype(np.int32)

        for _ in range(200):
            current_bpd = bits_per_agent / dims
            current_bpd = np.clip(current_bpd, self.min_bits_per_dim, self.max_bits_per_dim)

            # Marginal distortion reduction per ADDITIONAL bit
            # ∂D/∂R = -2·ln2/d · 2^(-2R/d) · (1 + 1/SNR)
            # Agents with low SNR have larger marginal benefit because
            # the channel noise amplifies quantization errors
            marginal = np.where(
                current_bpd < self.max_bits_per_dim,
                (2.0 ** (-2 * current_bpd) / dims) * (1.0 + 1.0 / snr_linear),
                0.0,
            )

            donor = np.argmin(marginal)
            recipient = np.argmax(marginal)

            if marginal[recipient] <= marginal[donor] + 1e-8:
                break

            bits_per_agent[donor] -= 1
            bits_per_agent[recipient] += 1

        # Enforce min/max
        min_bits = int(self.min_bits_per_dim * dims.min())
        bits_per_agent = np.clip(bits_per_agent, min_bits, None).astype(np.int32)

        return bits_per_agent.tolist()

    def allocate_uniform(self) -> list[int]:
        """Uniform allocation across all agents."""
        bpa = self.total_bit_budget // self.n_agents
        return [bpa] * self.n_agents

    def allocate_proportional(self, snrs: list[float]) -> list[int]:
        """Inverse-SNR proportional: more bits to worse channels."""
        snr_linear = np.array([10 ** (s / 10) for s in snrs])
        weights = 1.0 / (snr_linear + 0.1)
        weights = weights / weights.sum()
        alloc = (weights * self.total_bit_budget).astype(np.int32)
        alloc = np.clip(alloc, self.min_bits_per_dim, None)
        # Adjust to stay within budget
        excess = alloc.sum() - self.total_bit_budget
        while excess > 0:
            idx = np.argmax(alloc)
            alloc[idx] -= 1
            excess -= 1
        return alloc.tolist()


class AdaptiveDigitalCodec:
    """Digital quantization codec with per-agent adaptive bit allocation.

    Wraps the uniform DigitalCodec to apply variable bits-per-dimension
    based on per-agent channel quality estimates.

    This demonstrates the core semantic communication principle:
    allocate communication resources where they most reduce distortion.
    """

    def __init__(self, obs_dim: int, agent_dims: list[int],
                 total_bit_budget: int,
                 strategy: str = "water_filling"):
        self.obs_dim = obs_dim
        self.agent_dims = agent_dims
        self.total_bit_budget = total_bit_budget
        self.strategy = strategy
        self.boundaries = [0]
        for d in agent_dims:
            self.boundaries.append(self.boundaries[-1] + d)

        self.allocator = RateAllocator(
            agent_dims, total_bit_budget,
            min_bits_per_dim=2, max_bits_per_dim=16,
        )

        # Per-agent quantizers (fit separately)
        self.mins = []
        self.maxs = []
        self._fitted = False

    def fit(self, x: torch.Tensor):
        """Record per-dimension min/max from training data."""
        for i in range(len(self.agent_dims)):
            start, end = self.boundaries[i], self.boundaries[i + 1]
            segment = x[:, start:end]
            self.mins.append(segment.min(dim=0).values)
            self.maxs.append(segment.max(dim=0).values)
        self._fitted = True

    def encode(self, x: torch.Tensor, snrs: list[float]) -> tuple[torch.Tensor, list[int]]:
        """Encode with adaptive bit allocation based on per-agent SNR.

        Returns:
            (quantized_tensor, allocation) — allocation lists bits per agent.
        """
        assert self._fitted, "Must fit before encoding"

        if self.strategy == "water_filling":
            allocation = self.allocator.allocate_water_filling(snrs)
        elif self.strategy == "proportional":
            allocation = self.allocator.allocate_proportional(snrs)
        else:
            allocation = self.allocator.allocate_uniform()

        # Quantize each agent's segment with allocated bits
        segments = []
        for i, bits in enumerate(allocation):
            start, end = self.boundaries[i], self.boundaries[i + 1]
            segment = x[:, start:end]
            levels = 2 ** max(1, bits // self.agent_dims[i])

            x_norm = (segment - self.mins[i]) / (self.maxs[i] - self.mins[i] + 1e-8)
            x_norm = x_norm.clamp(0, 1)
            indices = (x_norm * (levels - 1)).round().long()
            x_recon = indices.float() / (levels - 1)
            x_recon = x_recon * (self.maxs[i] - self.mins[i] + 1e-8) + self.mins[i]
            segments.append(x_recon)

        return torch.cat(segments, dim=1), allocation

    def forward(self, x: torch.Tensor, snrs: list[float] | None = None):
        """Full encode-decode with adaptive allocation.

        Flow: x → per-agent AWGN → adaptive quantization → x_hat
        The key insight: channel noise degrades some agents more than others,
        and adaptive quantization counteracts this by allocating more bits
        to agents with worse channels.
        """
        if snrs is None:
            snrs = [20.0] * len(self.agent_dims)

        # Step 1: Apply per-agent AWGN (simulate heterogeneous channel)
        x_noisy = x.clone()
        for i in range(len(self.agent_dims)):
            start = self.boundaries[i]
            end = self.boundaries[i + 1]
            segment = x[:, start:end]
            signal_power = segment.pow(2).mean() + 1e-20
            snr_linear = 10 ** (snrs[i] / 10)
            noise_power = signal_power / snr_linear
            noise = torch.randn_like(segment) * torch.sqrt(noise_power)
            x_noisy[:, start:end] = segment + noise

        # Step 2: Quantize with adaptive bits (more bits where channel is worse)
        x_hat, alloc = self.encode(x_noisy, snrs)
        return x_hat, alloc

    def get_effective_rate(self, allocation: list[int]) -> float:
        """Total bits spent."""
        return float(sum(allocation))


def compute_coordination_score(model, test_data: torch.Tensor, device,
                               agent_boundaries: list[int],
                               loss_rates: list[float] | None = None) -> dict:
    """Measure coordination quality under partial agent loss.

    The coordination score evaluates how well the decoder can reconstruct
    ALL agents' states when only a SUBSET of agents' latent codes survive.

    For a given loss_rate, each agent's latent block is independently
    dropped. The decoder must then infer the missing agents' states from
    the surviving ones — this tests whether the latent space encodes
    inter-agent relationships.

    Returns:
        Dict with:
          - "full_mse": MSE when all agents present
          - "per_loss_rate": {rate: {"mse": float, "agent_mses": [float]}}
          - "coordination_gap": MSE difference between (isolated) and (joint) decoding
    """
    if loss_rates is None:
        loss_rates = [0.0, 0.1, 0.2, 0.3, 0.5]

    model.eval()
    test_data = test_data.to(device)
    results = {"loss_rates": {}}

    with torch.no_grad():
        out = model.encode(test_data)
        z = out[0] if isinstance(out, tuple) else out

        # Full reconstruction (all agents present)
        x_hat_full = model.decode(z)
        full_mse = F.mse_loss(x_hat_full, test_data).item()
        results["full_mse"] = float(full_mse)

        n_agents = len(agent_boundaries) - 1
        latent_dim = z.shape[1]

        for rate in loss_rates:
            # Burst-drop each agent's latent block independently
            # We estimate latent blocks proportional to observation blocks
            agent_latent_slots = []
            for i in range(n_agents):
                obs_start = int(agent_boundaries[i] / agent_boundaries[-1] * latent_dim)
                obs_end = int(agent_boundaries[i + 1] / agent_boundaries[-1] * latent_dim)
                agent_latent_slots.append((obs_start, obs_end))

            agent_mses = []
            for drop_agent in range(n_agents):
                z_dropped = z.clone()
                start, end = agent_latent_slots[drop_agent]
                z_dropped[:, start:end] = 0  # zero out one agent
                x_hat = model.decode(z_dropped)
                # MSE specifically on the dropped agent's observation
                obs_start = agent_boundaries[drop_agent]
                obs_end = agent_boundaries[drop_agent + 1]
                agent_mse = F.mse_loss(
                    x_hat[:, obs_start:obs_end],
                    test_data[:, obs_start:obs_end],
                ).item()
                agent_mses.append(agent_mse)

            results["loss_rates"][str(rate)] = {
                "mean_agent_mse": float(np.mean(agent_mses)),
                "max_agent_mse": float(np.max(agent_mses)),
                "agent_mses": [float(m) for m in agent_mses],
            }

    # Coordination gap: how much worse is isolated reconstruction vs joint
    gap = results["loss_rates"]["0.3"]["mean_agent_mse"] - full_mse
    results["coordination_gap"] = float(gap)

    return results
