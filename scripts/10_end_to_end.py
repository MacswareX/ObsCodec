"""Phase 3.4 — End-to-end semantic communication prototype.

Closes the loop: observation → encode → channel → decode → heuristic policy → action.

Trains a JSCC-BetaVAE on synthetic data, then runs closed-loop simulation
comparing three conditions:
  1. No compression (raw obs, no channel)
  2. Clean channel (JSCC codec, no noise)
  3. Noisy channel (JSCC codec, AWGN at configurable SNR)

Measures: final distance to targets, path efficiency, collision rate.

Usage: python scripts/10_end_to_end.py [--snr 10] [--rollout-steps 200]
"""

import sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import numpy as np

from obscodec.config import (DATA_DIR, ASSETS_DIR, CHECKPOINT_DIR, RANDOM_SEED,
                              JSCC_LATENT_DIM, VAE_EPOCHS, E2E_ROLLOUT_STEPS,
                              E2E_SNR_RANGE)
from obscodec.models import BetaVAE, JSCCWrapper
from obscodec.channel.diff_channel import DiffAWGN
from obscodec.channel.impairments import AWGNChannel
from obscodec.trainer import train_model
from obscodec.data.synthetic import generate_spread, _reflect_agents


# ── Simulator (reuses dynamics from synthetic.py) ──

class SpreadSimulator:
    """Minimal spread simulator for closed-loop evaluation."""

    def __init__(self, n_agents=5, n_landmarks=5, seed=None):
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.rng = np.random.default_rng(seed if seed is not None else RANDOM_SEED)
        self.reset()

    def reset(self):
        self.pos = self.rng.uniform(-1.5, 1.5, (self.n_agents, 2))
        self.vel = self.rng.normal(0, 0.1, (self.n_agents, 2))
        self.landmarks = self.rng.uniform(-2.0, 2.0, (self.n_landmarks, 2))
        self.targets = self.landmarks[
            self.rng.permutation(self.n_landmarks)[:self.n_agents] % self.n_landmarks
        ].copy()
        self.step_count = 0

    def get_observations(self):
        """Return (n_agents, obs_dim) observation matrix."""
        obs_list = []
        for i in range(self.n_agents):
            parts = [self.pos[i], self.vel[i]]  # self: 4 dims
            other_mask = np.ones(self.n_agents, dtype=bool)
            other_mask[i] = False
            parts.append((self.pos[other_mask] - self.pos[i]).ravel())
            parts.append(self.vel[other_mask].ravel())
            parts.append((self.landmarks - self.pos[i]).ravel())
            obs_list.append(np.concatenate(parts).astype(np.float32))
        return np.stack(obs_list)

    def step(self, actions):
        """Apply (n_agents, 2) velocity deltas, step physics."""
        dt = 0.1
        for i in range(self.n_agents):
            acc = actions[i] + self.rng.normal(0, 0.05, 2)
            self.vel[i] += acc * dt
            self.vel[i] *= 0.95
        self.pos += self.vel * dt
        self.pos, self.vel = _reflect_agents(self.pos, self.vel, bounds=(-2.0, 2.0))
        self.step_count += 1

    def distance_to_targets(self):
        return np.linalg.norm(self.pos - self.targets, axis=1)

    def inter_agent_distance(self):
        dists = []
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dists.append(np.linalg.norm(self.pos[i] - self.pos[j]))
        return np.mean(dists) if dists else 0.0


def heuristic_policy(obs, landmarks=None):
    """Extract self-position, move toward nearest landmark (or origin fallback).

    Args:
        obs: (n_agents, obs_dim) decoded observations
        landmarks: optional (n_landmarks, 2) — if None, extracts from obs

    Returns:
        actions: (n_agents, 2) velocity adjustments
    """
    n_agents = obs.shape[0]
    actions = np.zeros((n_agents, 2))

    for i in range(n_agents):
        self_pos = obs[i, :2]  # first 2 dims are self position

        if landmarks is not None:
            targets = landmarks
        else:
            # Infer landmarks from observation structure
            # simple_spread: self(4) + others(4*(N-1)) + landmarks(2*L)
            # obs_dim=30 with N=5, L=5: landmarks at index 20:30
            n = n_agents
            lm_start = 4 + 4 * (n - 1)
            lm_dim = obs.shape[1] - lm_start
            n_lm = lm_dim // 2
            if n_lm > 0:
                targets = obs[i, lm_start:lm_start + 2 * n_lm].reshape(n_lm, 2)
                targets = targets + self_pos[np.newaxis, :]  # relative → absolute
            else:
                targets = np.zeros((1, 2))

        # Move toward nearest target
        dists = np.linalg.norm(targets - self_pos[np.newaxis, :], axis=1)
        nearest = targets[np.argmin(dists)]
        direction = nearest - self_pos
        norm = np.linalg.norm(direction) + 1e-4
        actions[i] = 0.5 * direction / norm

    return actions


def run_rollout(sim, model, device, snr_db=None, n_steps=E2E_ROLLOUT_STEPS):
    """Run a closed-loop rollout and return metrics.

    Args:
        sim: SpreadSimulator instance (will be reset)
        model: JSCCWrapper or BetaVAE (None = use raw obs / no compression)
        device: torch device
        snr_db: AWGN SNR in dB (None = clean channel)
        n_steps: number of simulation steps

    Returns:
        dict with: final_d_to_target, mean_d_to_target, path_efficiency,
                   collisions, trajectory_distances
    """
    sim.reset()
    distances = []
    min_distances = []
    collisions = 0

    for step in range(n_steps):
        raw_obs = sim.get_observations()  # (n_agents, obs_dim)

        if model is not None:
            # Encode → channel → decode
            obs_t = torch.FloatTensor(raw_obs).to(device)
            with torch.no_grad():
                if hasattr(model, "base") and hasattr(model, "channel"):
                    inner = model.base
                    if hasattr(inner, "reparameterize"):
                        mu, logvar = inner.encoder(obs_t)
                        z = inner.reparameterize(mu, logvar)
                        if snr_db is not None:
                            eval_ch = AWGNChannel(snr_db=snr_db)
                            z = eval_ch(z)
                        obs_dec = inner.decode(z)
                    else:
                        z = inner.encode(obs_t)
                        if isinstance(z, tuple):
                            z = z[0]
                        if snr_db is not None:
                            eval_ch = AWGNChannel(snr_db=snr_db)
                            z = eval_ch(z)
                        obs_dec = inner.decode(z)
                elif hasattr(model, "reparameterize"):
                    mu, logvar = model.encoder(obs_t)
                    z = model.reparameterize(mu, logvar)
                    if snr_db is not None:
                        eval_ch = AWGNChannel(snr_db=snr_db)
                        z = eval_ch(z)
                    obs_dec = model.decode(z)
                else:
                    # Fallback: just forward
                    out = model(obs_t)
                    obs_dec = out[0] if isinstance(out, tuple) else out

            decoded = obs_dec.cpu().numpy()
        else:
            decoded = raw_obs  # no compression

        actions = heuristic_policy(decoded, sim.landmarks)
        sim.step(actions)

        d = sim.distance_to_targets()
        distances.append(np.mean(d))
        min_distances.append(np.min(d))

        # Count "collisions" — agents within 0.05 of each other
        for i in range(sim.n_agents):
            for j in range(i + 1, sim.n_agents):
                if np.linalg.norm(sim.pos[i] - sim.pos[j]) < 0.05:
                    collisions += 1

    total_dist = float(np.sum(np.sqrt(
        np.sum(np.diff(np.array(distances)) ** 2) + 0
    ))) if len(distances) > 1 else 0.0

    return {
        "final_d_to_target": float(np.mean(distances[-10:])),  # avg last 10 steps
        "mean_d_to_target_early": float(np.mean(distances[:50])),  # avg first 50
        "mean_d_to_target_late": float(np.mean(distances[-50:])),
        "path_efficiency": total_dist / (n_steps + 1e-8),
        "collisions": collisions,
        "min_distance_ever": float(np.min(min_distances)),
        "trajectory_distances": distances[-1:] if distances else [],  # last step
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3.4 — End-to-end prototype")
    parser.add_argument("--scenario", default="simple_spread")
    parser.add_argument("--n-agents", type=int, default=5)
    parser.add_argument("--n-landmarks", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=E2E_ROLLOUT_STEPS)
    parser.add_argument("--n-episodes", type=int, default=10,
                        help="Number of episodes per condition")
    parser.add_argument("--snr", type=float, default=10,
                        help="AWGN SNR (dB) for noisy condition")
    parser.add_argument("--epochs", type=int, default=VAE_EPOCHS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    obs_dim = 4 + 4 * (args.n_agents - 1) + 2 * args.n_landmarks
    print(f"E2E Prototype: {args.n_agents} agents, {args.n_landmarks} landmarks")
    print(f"  Obs dim: {obs_dim}, Rollout steps: {args.rollout_steps}")
    print(f"  Episodes per condition: {args.n_episodes}")

    # ── Step 1: Generate training data ──
    print("\n[1/4] Generating training data...")
    train_obs = generate_spread(n_agents=args.n_agents, n_landmarks=args.n_landmarks,
                                n_samples=20000, seed=RANDOM_SEED)
    train_n = int(len(train_obs) * 0.8)
    val_n = int(len(train_obs) * 0.1)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(train_obs))
    train_data = train_obs[idx[:train_n]]
    val_data = train_obs[idx[train_n:train_n + val_n]]

    # ── Step 2: Train JSCC codec ──
    print("[2/4] Training JSCC-BetaVAE...")
    base = BetaVAE(obs_dim, JSCC_LATENT_DIM, beta=2.0,
                   free_bits=0.1, kl_warmup_epochs=150).to(device)
    diff_awgn = DiffAWGN(snr_db=10.0)
    model = JSCCWrapper(base, diff_awgn).to(device)

    ckpt_name = f"e2e_{args.scenario}_b2.0_fb0.1_awgn10"
    ckpt_path = CHECKPOINT_DIR / f"{ckpt_name}.pt"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded: {ckpt_path}")
    else:
        train_model(model, train_data, val_data, device, epochs=args.epochs,
                    model_name=ckpt_name)
        torch.save({"model_state": model.state_dict(), "history": {}}, ckpt_path)

    # ── Step 3: Run rollouts ──
    print("\n[3/4] Running closed-loop rollouts...")

    conditions = {
        "no_compression": {"model": None, "snr": None},
        "jscc_clean": {"model": model, "snr": None},
        "jscc_noisy": {"model": model, "snr": args.snr},
    }

    all_results = []

    for cond_name, cond_cfg in conditions.items():
        print(f"\n  --- {cond_name} ---")
        ep_results = []
        for ep in range(args.n_episodes):
            sim = SpreadSimulator(n_agents=args.n_agents,
                                  n_landmarks=args.n_landmarks,
                                  seed=RANDOM_SEED + ep * 100)
            r = run_rollout(sim, cond_cfg["model"], device,
                            snr_db=cond_cfg["snr"],
                            n_steps=args.rollout_steps)
            ep_results.append(r)

            if ep == 0 or (ep + 1) % 5 == 0:
                print(f"    Ep {ep + 1}/{args.n_episodes}: "
                      f"final_d={r['final_d_to_target']:.3f}  "
                      f"collisions={r['collisions']}")

        # Aggregate
        agg = {
            "condition": cond_name,
            "snr": cond_cfg["snr"],
            "n_episodes": args.n_episodes,
            "mean_final_d": float(np.mean([e["final_d_to_target"] for e in ep_results])),
            "std_final_d": float(np.std([e["final_d_to_target"] for e in ep_results])),
            "mean_early_d": float(np.mean([e["mean_d_to_target_early"] for e in ep_results])),
            "mean_late_d": float(np.mean([e["mean_d_to_target_late"] for e in ep_results])),
            "mean_collisions": float(np.mean([e["collisions"] for e in ep_results])),
            "mean_path_efficiency": float(np.mean([e["path_efficiency"] for e in ep_results])),
            "per_episode": [{k: v for k, v in e.items() if k != "trajectory_distances"}
                            for e in ep_results],
        }
        all_results.append(agg)
        print(f"    Avg: final_d={agg['mean_final_d']:.3f} ± {agg['std_final_d']:.3f}, "
              f"early={agg['mean_early_d']:.3f}, late={agg['mean_late_d']:.3f}")

    # ── Step 4: Compare & save ──
    print("\n[4/4] Comparison:")
    print(f"{'Condition':<20s} {'Final d':>8s}  {'Early d':>8s}  {'Late d':>8s}  "
          f"{'Collisions':>10s}")
    print("-" * 62)
    for agg in all_results:
        print(f"{agg['condition']:<20s} {agg['mean_final_d']:8.3f}  "
              f"{agg['mean_early_d']:8.3f}  {agg['mean_late_d']:8.3f}  "
              f"{agg['mean_collisions']:10.1f}")

    # Check if JSCC clean is comparable to no-compression
    nc = all_results[0]
    jscc_c = all_results[1]
    jscc_n = all_results[2]
    degradation = jscc_c["mean_final_d"] / (nc["mean_final_d"] + 1e-8)
    noise_impact = jscc_n["mean_final_d"] / (jscc_c["mean_final_d"] + 1e-8)
    print(f"\n  JSCC degradation (vs no-comp): {degradation:.3f}x")
    print(f"  Noise impact (vs JSCC clean):  {noise_impact:.3f}x "
          f"at SNR={args.snr}dB")

    out_path = ASSETS_DIR / "e2e_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
