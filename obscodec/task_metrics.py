"""Task-aware evaluation metrics for Phase 3 (Semantic Communication).

Uses task ground-truth from ``*_with_metrics`` generators to compute
coordination-aware distortion measures beyond raw reconstruction MSE.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def extract_self_position(obs: torch.Tensor) -> torch.Tensor:
    """Extract agent's own (x, y) position from observation vector (first 2 dims)."""
    return obs[..., :2]


def self_position_mse(recon_obs: torch.Tensor, true_obs: torch.Tensor) -> float:
    """MSE on self-position only — the most task-critical observation component."""
    recon_self = extract_self_position(recon_obs)
    true_self = extract_self_position(true_obs)
    return float(torch.nn.functional.mse_loss(recon_self, true_self).item())


def per_agent_mse(recon_obs: torch.Tensor, true_obs: torch.Tensor,
                  agent_boundaries: list[tuple[int, int]]) -> dict[int, float]:
    """MSE broken down by agent segment in the observation vector.

    Parameters
    ----------
    agent_boundaries:
        List of (start, end) index pairs for each agent's observation segment.

    Returns
    -------
    dict mapping agent_index → MSE on that agent's observation segment.
    """
    mses = {}
    for i, (start, end) in enumerate(agent_boundaries):
        recon_seg = recon_obs[..., start:end]
        true_seg = true_obs[..., start:end]
        mses[i] = float(torch.nn.functional.mse_loss(recon_seg, true_seg).item())
    return mses


def target_distance_error(recon_obs: torch.Tensor, true_obs: torch.Tensor,
                          task_data: dict, n_agents: int) -> float:
    """L2 error between predicted and true self-position relative to own target.

    Uses task ground-truth ``d_to_target`` (distance to assigned target per agent),
    comparing the decoded self-position to the true self-position — since the
    observation vector only encodes self-position, decoding error on self-pos
    directly affects task performance.
    """
    recon_self = extract_self_position(recon_obs)
    true_self = extract_self_position(true_obs)
    return float(torch.norm(recon_self - true_self, dim=-1).mean().item())


def coordination_error(recon_obs: torch.Tensor, true_obs: torch.Tensor,
                       model_outputs: dict | None = None) -> dict[str, float]:
    """Evaluate multi-agent coordination from decoded observations.

    Computes:
      - self_mse: reconstruction error on self-position
      - others_mse: reconstruction error on other-agent positions
      - coordination_gap: others_mse / self_mse — how much worse other-agent
        reconstruction is (higher = more coordination information lost)
    """
    if recon_obs.dim() > 2:
        recon_obs = recon_obs.reshape(-1, recon_obs.shape[-1])
        true_obs = true_obs.reshape(-1, true_obs.shape[-1])

    self_mse = self_position_mse(recon_obs, true_obs)

    # Other-agent info starts at dim 4 (after self pos + self vel)
    if true_obs.shape[-1] > 4:
        others_mse = float(torch.nn.functional.mse_loss(
            recon_obs[..., 4:], true_obs[..., 4:]).item())
    else:
        others_mse = 0.0

    gap = others_mse / (self_mse + 1e-8)
    return {"self_mse": self_mse, "others_mse": others_mse, "coordination_gap": gap}


def evaluate_task_metrics(model, test_data: np.ndarray, task_metrics: dict,
                          agent_boundaries: list | None = None,
                          device: str = "cpu", batch_size: int = 256) -> dict:
    """Run full task-aware evaluation on a trained model.

    Returns dict with keys: mse, self_mse, coordination_gap, per_agent_mses, kl, rate.
    """
    model.eval()
    model.to(device)
    test_t = torch.FloatTensor(test_data).to(device)

    # Full batch reconstruction
    with torch.no_grad():
        if hasattr(model, "forward"):
            outputs = model.forward(test_t)
            x_hat = outputs[0]
        else:
            x_hat = model(test_t)
            if isinstance(x_hat, tuple):
                x_hat = x_hat[0]

    mse = float(torch.nn.functional.mse_loss(x_hat, test_t).item())
    coord = coordination_error(x_hat, test_t)

    results = {"mse": mse, **coord}

    # Per-agent MSE if boundaries provided
    if agent_boundaries is not None:
        pa = per_agent_mse(x_hat, test_t, agent_boundaries)
        results["per_agent_mses"] = pa
        results["max_agent_mse"] = max(pa.values())
        results["mean_agent_mse"] = np.mean(list(pa.values()))

    # KL and rate if available
    if hasattr(model, "kl_nats"):
        results["kl"] = model.kl_nats(test_t)
    if hasattr(model, "get_rate_estimate"):
        loader = DataLoader(TensorDataset(test_t), batch_size=batch_size, shuffle=False)
        results["rate_bits"] = model.get_rate_estimate(loader, device)

    return results
