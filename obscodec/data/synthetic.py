"""Synthetic multi-agent trajectory data generator.

Generates realistic multi-agent observation data that mimics MPE-like scenarios
without requiring PettingZoo MPE dependencies. Agents move in 2D with
stochastic dynamics, observe themselves, other agents, and landmarks.

Scenarios:
  - spread: agents spread out to cover landmarks (mimics simple_spread)
  - tag: predator-prey pursuit (mimics simple_tag)
  - comm: agents move in formation with communication targets (mimics simple_world_comm)

Each produces per-agent observation vectors of configurable dimensionality.
"""

import numpy as np
from ..config import SCENARIOS, SAMPLES_PER_SCENARIO, DATA_DIR, RANDOM_SEED


def _reflect_agents(pos, vel, bounds=(-2.0, 2.0)):
    """Bounce agents off boundaries."""
    lo, hi = bounds
    for d in range(2):
        outside_lo = pos[:, d] < lo
        outside_hi = pos[:, d] > hi
        pos[outside_lo, d] = 2 * lo - pos[outside_lo, d]
        pos[outside_hi, d] = 2 * hi - pos[outside_hi, d]
        vel[outside_lo, d] *= -0.8
        vel[outside_hi, d] *= -0.8
    return pos, vel


def _agent_obs(agent_idx, pos, vel, landmarks):
    """Build per-agent observation vector: self + others + landmarks."""
    n = len(pos)
    obs_parts = [pos[agent_idx], vel[agent_idx]]  # self: 4 dims

    # Other agents (relative pos + vel)
    other_mask = np.ones(n, dtype=bool)
    other_mask[agent_idx] = False
    obs_parts.append((pos[other_mask] - pos[agent_idx]).ravel())
    obs_parts.append(vel[other_mask].ravel())

    # Landmarks (relative pos)
    if landmarks is not None:
        obs_parts.append((landmarks - pos[agent_idx]).ravel())

    return np.concatenate(obs_parts).astype(np.float32)


def generate_spread(n_agents=5, n_landmarks=5, n_samples=SAMPLES_PER_SCENARIO,
                    seed=RANDOM_SEED, dt=0.1, steps_per_ep=100) -> np.ndarray:
    """Agents navigate toward assigned landmarks with stochastic drift.

    Observation per agent: self(4) + other_agents(4*(N-1)) + landmarks(2*L)
    With N=5, L=5: 4 + 16 + 10 = 30 dims.
    """
    rng = np.random.default_rng(seed)
    all_samples = []
    collected = 0

    while collected < n_samples:
        pos = rng.uniform(-1.5, 1.5, (n_agents, 2))
        vel = rng.normal(0, 0.1, (n_agents, 2))
        landmarks = rng.uniform(-2.0, 2.0, (n_landmarks, 2))

        # Assign each agent to a random landmark
        targets = landmarks[rng.permutation(n_landmarks)[:n_agents] % n_landmarks]

        for step in range(steps_per_ep):
            for i in range(n_agents):
                # Attraction toward target + repulsion from other agents + noise
                to_target = targets[i] - pos[i]
                dist_target = np.linalg.norm(to_target) + 1e-4
                f_target = 0.5 * to_target / dist_target

                # Inter-agent repulsion
                f_repel = np.zeros(2)
                for j in range(n_agents):
                    if j != i:
                        diff = pos[i] - pos[j]
                        dist = np.linalg.norm(diff) + 1e-3
                        f_repel += 0.2 * diff / (dist ** 2 + 0.1)

                acc = f_target + f_repel + rng.normal(0, 0.05, 2)
                vel[i] += acc * dt
                vel[i] *= 0.95  # damping

            pos += vel * dt
            pos, vel = _reflect_agents(pos, vel)

            # Collect observations every 5 steps
            if step % 5 == 0:
                for i in range(n_agents):
                    obs = _agent_obs(i, pos, vel, landmarks)
                    all_samples.append(obs)
                    collected += 1
                    if collected >= n_samples:
                        break
            if collected >= n_samples:
                break

    return np.stack(all_samples[:n_samples], axis=0)


def generate_tag(n_predators=3, n_prey=3, n_samples=SAMPLES_PER_SCENARIO,
                 seed=RANDOM_SEED + 100, dt=0.1, steps_per_ep=100) -> np.ndarray:
    """Predator-prey pursuit: predators chase nearest prey, prey evade.

    Total agents = n_predators + n_prey.
    Observation per agent: self(4) + other_agents(4*(N-1))
    With 3+3=6 agents: 4 + 20 = 24 dims.
    """
    rng = np.random.default_rng(seed)
    n_total = n_predators + n_prey
    all_samples = []
    collected = 0

    while collected < n_samples:
        pos = rng.uniform(-2.0, 2.0, (n_total, 2))
        vel = rng.normal(0, 0.15, (n_total, 2))
        is_predator = np.array([i < n_predators for i in range(n_total)])

        for step in range(steps_per_ep):
            for i in range(n_total):
                if is_predator[i]:
                    # Chase nearest prey
                    prey_positions = pos[~is_predator]
                    prey_idx = np.argmin(np.linalg.norm(prey_positions - pos[i], axis=1))
                    to_target = prey_positions[prey_idx] - pos[i]
                    dist = np.linalg.norm(to_target) + 1e-4
                    f_goal = 0.8 * to_target / dist
                else:
                    # Evade nearest predator
                    predator_positions = pos[is_predator]
                    pred_idx = np.argmin(np.linalg.norm(predator_positions - pos[i], axis=1))
                    away = pos[i] - predator_positions[pred_idx]
                    dist = np.linalg.norm(away) + 1e-4
                    f_goal = 0.6 * away / (dist ** 2 + 0.05)

                acc = f_goal + rng.normal(0, 0.08, 2)
                vel[i] += acc * dt
                vel[i] *= 0.97

            pos += vel * dt
            pos, vel = _reflect_agents(pos, vel)

            if step % 5 == 0:
                for i in range(n_total):
                    obs = _agent_obs(i, pos, vel, None)
                    all_samples.append(obs)
                    collected += 1
                    if collected >= n_samples:
                        break
            if collected >= n_samples:
                break

    return np.stack(all_samples[:n_samples], axis=0)


def generate_comm(n_agents=6, n_food=4, n_forests=2,
                  n_samples=SAMPLES_PER_SCENARIO, seed=RANDOM_SEED + 200,
                  dt=0.1, steps_per_ep=100) -> np.ndarray:
    """Communication scenario: agents navigate toward food, avoid forests.

    Landmarks = food + forests combined.
    Observation: self(4) + others(4*(N-1)) + landmarks(2*(food+forests))
    With N=6, food=4, forests=2: 4 + 20 + 12 = 36 dims.
    """
    rng = np.random.default_rng(seed)
    n_landmarks = n_food + n_forests
    all_samples = []
    collected = 0

    while collected < n_samples:
        pos = rng.uniform(-2.0, 2.0, (n_agents, 2))
        vel = rng.normal(0, 0.1, (n_agents, 2))

        # Place food and forests
        food_pos = rng.uniform(-2.0, 2.0, (n_food, 2))
        forest_pos = rng.uniform(-2.0, 2.0, (n_forests, 2))
        landmarks = np.vstack([food_pos, forest_pos])  # (n_food+n_forests, 2)
        is_food = np.array([True] * n_food + [False] * n_forests)

        for step in range(steps_per_ep):
            for i in range(n_agents):
                # Attraction to nearest food
                d_food = np.linalg.norm(food_pos - pos[i], axis=1)
                nearest_food = food_pos[np.argmin(d_food)]
                to_food = nearest_food - pos[i]
                dist_food = np.linalg.norm(to_food) + 1e-4
                f_food = 0.4 * to_food / dist_food

                # Repulsion from nearest forest
                d_forest = np.linalg.norm(forest_pos - pos[i], axis=1)
                nearest_forest = forest_pos[np.argmin(d_forest)]
                away_forest = pos[i] - nearest_forest
                dist_forest = np.linalg.norm(away_forest) + 1e-4
                f_forest = 0.3 * away_forest / (dist_forest ** 2 + 0.1)

                # Cohesion with other agents
                f_cohere = np.zeros(2)
                for j in range(n_agents):
                    if j != i:
                        diff = pos[j] - pos[i]
                        dist = np.linalg.norm(diff) + 1e-3
                        f_cohere += 0.1 * diff * np.clip(dist - 0.5, 0, 1)

                acc = f_food + f_forest + f_cohere + rng.normal(0, 0.05, 2)
                vel[i] += acc * dt
                vel[i] *= 0.96

            pos += vel * dt
            pos, vel = _reflect_agents(pos, vel)

            if step % 5 == 0:
                for i in range(n_agents):
                    obs = _agent_obs(i, pos, vel, landmarks)
                    all_samples.append(obs)
                    collected += 1
                    if collected >= n_samples:
                        break
            if collected >= n_samples:
                break

    return np.stack(all_samples[:n_samples], axis=0)


# ── High-dimensional variants (Step B) ──────────────────────────

def generate_spread_hd(n_samples=SAMPLES_PER_SCENARIO, seed=RANDOM_SEED + 300):
    """spread with N=8, L=8 → obs_dim=48."""
    return generate_spread(n_agents=8, n_landmarks=8, n_samples=n_samples, seed=seed)


def generate_tag_hd(n_samples=SAMPLES_PER_SCENARIO, seed=RANDOM_SEED + 400):
    """tag with 5 predators + 5 prey → N=10, obs_dim=40."""
    return generate_tag(n_predators=5, n_prey=5, n_samples=n_samples, seed=seed)


def generate_comm_hd(n_samples=SAMPLES_PER_SCENARIO, seed=RANDOM_SEED + 500):
    """comm with N=10, food=6, forest=4 → obs_dim=60."""
    return generate_comm(n_agents=10, n_food=6, n_forests=4,
                         n_samples=n_samples, seed=seed)


def generate_spread_xhd(n_samples=SAMPLES_PER_SCENARIO, seed=RANDOM_SEED + 600):
    """spread extreme: N=15, L=15 → obs_dim=90."""
    return generate_spread(n_agents=15, n_landmarks=15, n_samples=n_samples, seed=seed)


# Registry mapping scenario names to generator functions
SYNTHETIC_GENERATORS = {
    "simple_spread": generate_spread,
    "simple_tag": generate_tag,
    "simple_world_comm": generate_comm,
}

# Step B high-dimensional registry
SYNTHETIC_GENERATORS_HD = {
    "spread_hd": generate_spread_hd,       # 48 dim
    "tag_hd": generate_tag_hd,             # 40 dim
    "comm_hd": generate_comm_hd,           # 60 dim
    "spread_xhd": generate_spread_xhd,     # 90 dim
}


# ═══════════════════════════════════════════════════════════════════
# Task-aware generators (Phase 3) — return observations + ground-truth
# ═══════════════════════════════════════════════════════════════════

def generate_spread_with_metrics(n_agents=5, n_landmarks=5, n_samples=SAMPLES_PER_SCENARIO,
                                 seed=RANDOM_SEED, dt=0.1, steps_per_ep=100):
    """Like generate_spread but also returns task ground-truth dict.

    Returns
    -------
    obs: np.ndarray (n_samples, obs_dim)
    metrics: dict with keys:
        positions    — (n_samples, n_agents, 2)  absolute positions
        velocities   — (n_samples, n_agents, 2)  velocities
        landmarks    — (n_samples, n_landmarks, 2)  landmark positions
        targets      — (n_samples, n_agents, 2)  assigned target positions
        d_to_target  — (n_samples, n_agents)       distance to assigned target
    """
    rng = np.random.default_rng(seed)
    obs_list, pos_list, vel_list, lm_list, target_list, d_list = [], [], [], [], [], []
    collected = 0

    while collected < n_samples:
        pos = rng.uniform(-1.5, 1.5, (n_agents, 2))
        vel = rng.normal(0, 0.1, (n_agents, 2))
        landmarks = rng.uniform(-2.0, 2.0, (n_landmarks, 2))
        targets = landmarks[rng.permutation(n_landmarks)[:n_agents] % n_landmarks]

        for step in range(steps_per_ep):
            for i in range(n_agents):
                to_target = targets[i] - pos[i]
                dist_target = np.linalg.norm(to_target) + 1e-4
                f_target = 0.5 * to_target / dist_target
                f_repel = np.zeros(2)
                for j in range(n_agents):
                    if j != i:
                        diff = pos[i] - pos[j]
                        dist = np.linalg.norm(diff) + 1e-3
                        f_repel += 0.2 * diff / (dist ** 2 + 0.1)
                acc = f_target + f_repel + rng.normal(0, 0.05, 2)
                vel[i] += acc * dt
                vel[i] *= 0.95

            pos += vel * dt
            pos, vel = _reflect_agents(pos, vel)

            if step % 5 == 0:
                for i in range(n_agents):
                    obs = _agent_obs(i, pos, vel, landmarks)
                    obs_list.append(obs)
                    pos_list.append(pos.copy())
                    vel_list.append(vel.copy())
                    lm_list.append(landmarks.copy())
                    target_list.append(targets.copy())
                    d_list.append(np.linalg.norm(pos - targets, axis=1))
                    collected += 1
                    if collected >= n_samples:
                        break
            if collected >= n_samples:
                break

    n = min(len(obs_list), n_samples)
    metrics = {
        "positions": np.stack(pos_list[:n]),
        "velocities": np.stack(vel_list[:n]),
        "landmarks": np.stack(lm_list[:n]),
        "targets": np.stack(target_list[:n]),
        "d_to_target": np.stack(d_list[:n]),
    }
    return np.stack(obs_list[:n]), metrics


def generate_tag_with_metrics(n_predators=3, n_prey=3, n_samples=SAMPLES_PER_SCENARIO,
                              seed=RANDOM_SEED + 100, dt=0.1, steps_per_ep=100):
    """Like generate_tag but returns task ground-truth dict.

    Returns
    -------
    obs: np.ndarray (n_samples, obs_dim)
    metrics: dict with keys:
        positions   — (n_samples, n_total, 2)
        velocities  — (n_samples, n_total, 2)
        is_predator — (n_total,) bool mask
    """
    rng = np.random.default_rng(seed)
    n_total = n_predators + n_prey
    obs_list, pos_list, vel_list = [], [], []
    collected = 0

    while collected < n_samples:
        pos = rng.uniform(-2.0, 2.0, (n_total, 2))
        vel = rng.normal(0, 0.15, (n_total, 2))
        is_predator = np.array([i < n_predators for i in range(n_total)])

        for step in range(steps_per_ep):
            for i in range(n_total):
                if is_predator[i]:
                    prey_positions = pos[~is_predator]
                    prey_idx = np.argmin(np.linalg.norm(prey_positions - pos[i], axis=1))
                    to_target = prey_positions[prey_idx] - pos[i]
                    dist = np.linalg.norm(to_target) + 1e-4
                    f_goal = 0.8 * to_target / dist
                else:
                    predator_positions = pos[is_predator]
                    pred_idx = np.argmin(np.linalg.norm(predator_positions - pos[i], axis=1))
                    away = pos[i] - predator_positions[pred_idx]
                    dist = np.linalg.norm(away) + 1e-4
                    f_goal = 0.6 * away / (dist ** 2 + 0.05)
                acc = f_goal + rng.normal(0, 0.08, 2)
                vel[i] += acc * dt
                vel[i] *= 0.97

            pos += vel * dt
            pos, vel = _reflect_agents(pos, vel)

            if step % 5 == 0:
                for i in range(n_total):
                    obs = _agent_obs(i, pos, vel, None)
                    obs_list.append(obs)
                    pos_list.append(pos.copy())
                    vel_list.append(vel.copy())
                    collected += 1
                    if collected >= n_samples:
                        break
            if collected >= n_samples:
                break

    n = min(len(obs_list), n_samples)
    metrics = {
        "positions": np.stack(pos_list[:n]),
        "velocities": np.stack(vel_list[:n]),
        "is_predator": is_predator,
    }
    return np.stack(obs_list[:n]), metrics


def generate_comm_with_metrics(n_agents=6, n_food=4, n_forests=2,
                                n_samples=SAMPLES_PER_SCENARIO, seed=RANDOM_SEED + 200,
                                dt=0.1, steps_per_ep=100):
    """Like generate_comm but returns task ground-truth dict.

    Returns
    -------
    obs: np.ndarray (n_samples, obs_dim)
    metrics: dict with keys:
        positions   — (n_samples, n_agents, 2)
        velocities  — (n_samples, n_agents, 2)
        landmarks   — (n_samples, n_landmarks, 2)
        is_food     — (n_landmarks,) bool mask (True=food, False=forest)
    """
    rng = np.random.default_rng(seed)
    n_landmarks = n_food + n_forests
    obs_list, pos_list, vel_list, lm_list = [], [], [], []
    collected = 0

    while collected < n_samples:
        pos = rng.uniform(-2.0, 2.0, (n_agents, 2))
        vel = rng.normal(0, 0.1, (n_agents, 2))
        food_pos = rng.uniform(-2.0, 2.0, (n_food, 2))
        forest_pos = rng.uniform(-2.0, 2.0, (n_forests, 2))
        landmarks = np.vstack([food_pos, forest_pos])

        for step in range(steps_per_ep):
            for i in range(n_agents):
                d_food = np.linalg.norm(food_pos - pos[i], axis=1)
                nearest_food = food_pos[np.argmin(d_food)]
                to_food = nearest_food - pos[i]
                dist_food = np.linalg.norm(to_food) + 1e-4
                f_food = 0.4 * to_food / dist_food

                d_forest = np.linalg.norm(forest_pos - pos[i], axis=1)
                nearest_forest = forest_pos[np.argmin(d_forest)]
                away_forest = pos[i] - nearest_forest
                dist_forest = np.linalg.norm(away_forest) + 1e-4
                f_forest = 0.3 * away_forest / (dist_forest ** 2 + 0.1)

                f_cohere = np.zeros(2)
                for j in range(n_agents):
                    if j != i:
                        diff = pos[j] - pos[i]
                        dist = np.linalg.norm(diff) + 1e-3
                        f_cohere += 0.1 * diff * np.clip(dist - 0.5, 0, 1)

                acc = f_food + f_forest + f_cohere + rng.normal(0, 0.05, 2)
                vel[i] += acc * dt
                vel[i] *= 0.96

            pos += vel * dt
            pos, vel = _reflect_agents(pos, vel)

            if step % 5 == 0:
                for i in range(n_agents):
                    obs = _agent_obs(i, pos, vel, landmarks)
                    obs_list.append(obs)
                    pos_list.append(pos.copy())
                    vel_list.append(vel.copy())
                    lm_list.append(landmarks.copy())
                    collected += 1
                    if collected >= n_samples:
                        break
            if collected >= n_samples:
                break

    n = min(len(obs_list), n_samples)
    is_food = np.array([True] * n_food + [False] * n_forests)
    metrics = {
        "positions": np.stack(pos_list[:n]),
        "velocities": np.stack(vel_list[:n]),
        "landmarks": np.stack(lm_list[:n]),
        "is_food": is_food,
    }
    return np.stack(obs_list[:n]), metrics


# Registry for task-aware generators
SYNTHETIC_GENERATORS_METRICS = {
    "simple_spread": generate_spread_with_metrics,
    "simple_tag": generate_tag_with_metrics,
    "simple_world_comm": generate_comm_with_metrics,
}


def collect_synthetic_dataset_hd(scenarios: list[str] | None = None,
                                 save: bool = True) -> dict[str, np.ndarray]:
    """Generate high-dimensional datasets for Step B."""
    if scenarios is None:
        scenarios = list(SYNTHETIC_GENERATORS_HD.keys())

    results = {}
    for name in scenarios:
        gen = SYNTHETIC_GENERATORS_HD[name]
        print(f"Generating {name} (target {SAMPLES_PER_SCENARIO} samples)...")
        data = gen()
        obs_dim = data.shape[1]
        print(f"  {name}: {data.shape[0]} samples, obs_dim={obs_dim}")
        results[name] = data

        if save:
            path = DATA_DIR / f"{name}_obs.npy"
            np.save(path, data)
            print(f"  Saved to {path}")

    if save:
        meta = {name: {"shape": list(arr.shape), "obs_dim": int(arr.shape[1])}
                for name, arr in results.items()}
        import json
        with open(DATA_DIR / "mpe_metadata_hd.json", "w") as f:
            json.dump(meta, f, indent=2)

    return results


def collect_synthetic_dataset(scenarios: list[str] | None = None,
                              save: bool = True) -> dict[str, np.ndarray]:
    """Generate synthetic multi-agent observation data.

    Same interface as collector.collect_mpe_dataset() — drop-in replacement.
    """
    if scenarios is None:
        scenarios = list(SYNTHETIC_GENERATORS.keys())

    results = {}
    for name in scenarios:
        gen = SYNTHETIC_GENERATORS[name]
        print(f"Generating {name} (target {SAMPLES_PER_SCENARIO} samples)...")
        data = gen()
        obs_dim = data.shape[1]
        print(f"  {name}: {data.shape[0]} samples, obs_dim={obs_dim}")
        results[name] = data

        if save:
            path = DATA_DIR / f"{name}_obs.npy"
            np.save(path, data)
            print(f"  Saved to {path}")

    if save:
        meta = {name: {"shape": list(arr.shape), "obs_dim": int(arr.shape[1])}
                for name, arr in results.items()}
        import json
        with open(DATA_DIR / "mpe_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    return results


if __name__ == "__main__":
    collect_synthetic_dataset()
