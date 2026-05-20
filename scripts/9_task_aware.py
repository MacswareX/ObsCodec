"""Phase 3.3 — Task-aware loss experiment.

Train BetaVAE codecs with task-aware additive loss terms:
  - "none": standard β-VAE loss (recon + β·KL)
  - "self_only": MSE on self-position (first 2 dims) added to loss
  - "weighted": 0.7·self-MSE + 0.3·others-MSE added to loss

Tests whether task-aware loss changes KL collapse behavior and whether
it can substitute for free-bits in maintaining latent activity.

Usage: python scripts/9_task_aware.py [--scenario simple_spread]
"""

import sys, json, argparse, itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import numpy as np

from obscodec.config import (DATA_DIR, ASSETS_DIR, CHECKPOINT_DIR, RANDOM_SEED,
                              JSCC_LATENT_DIM, VAE_EPOCHS, ANTI_COLLAPSE_BETAS,
                              TASK_WEIGHTS, TASK_LOSS_TYPES)
from obscodec.models import BetaVAE
from obscodec.trainer import train_model
from obscodec.task_metrics import (evaluate_task_metrics, self_position_mse,
                                   coordination_error, extract_self_position)
from obscodec.data.synthetic import generate_spread_with_metrics


def load_or_generate_data(scenario: str):
    """Load scenario data with metrics, generating if needed."""
    obs_path = DATA_DIR / f"{scenario}_obs.npy"
    metrics_path = DATA_DIR / f"{scenario}_metrics.npz"

    if obs_path.exists() and metrics_path.exists():
        obs = np.load(str(obs_path))
        metrics = dict(np.load(str(metrics_path), allow_pickle=True))
        return obs, metrics

    print(f"  Generating {scenario} with task metrics...")
    if scenario == "simple_spread":
        obs, metrics = generate_spread_with_metrics()
    elif scenario == "simple_tag":
        from obscodec.data.synthetic import generate_tag_with_metrics
        obs, metrics = generate_tag_with_metrics()
    elif scenario == "simple_world_comm":
        from obscodec.data.synthetic import generate_comm_with_metrics
        obs, metrics = generate_comm_with_metrics()
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    np.save(str(obs_path), obs)
    np.savez_compressed(str(metrics_path), **{k: v for k, v in metrics.items()
                                               if isinstance(v, np.ndarray)})
    return obs, metrics


def train_task_aware(obs_dim, beta, free_bits, task_loss_type, task_weight,
                     train_data, val_data, device, epochs, ckpt_name):
    """Train a BetaVAE with optional task-aware loss."""
    model = BetaVAE(obs_dim, JSCC_LATENT_DIM, beta=beta,
                    free_bits=free_bits, kl_warmup_epochs=150,
                    task_weight=task_weight,
                    task_loss_type=task_loss_type).to(device)

    ckpt_path = CHECKPOINT_DIR / f"{ckpt_name}.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        return model

    train_model(model, train_data, val_data, device, epochs=epochs,
                model_name=ckpt_name, verbose=False)
    torch.save({"model_state": model.state_dict(), "history": {}}, ckpt_path)
    return model


def main():
    parser = argparse.ArgumentParser(description="Phase 3.3 — Task-aware loss")
    parser.add_argument("--scenario", default="simple_spread",
                        choices=["simple_spread", "simple_tag", "simple_world_comm"])
    parser.add_argument("--epochs", type=int, default=VAE_EPOCHS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device

    print(f"Scenario: {args.scenario}")
    obs, task_metrics = load_or_generate_data(args.scenario)
    obs_dim = obs.shape[1]
    print(f"  Obs dim: {obs_dim}, Samples: {len(obs)}")

    # Train/test split
    train_n = int(len(obs) * 0.8)
    val_n = int(len(obs) * 0.1)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(obs))
    train_data = obs[idx[:train_n]]
    val_data = obs[idx[train_n:train_n + val_n]]
    test_data = obs[idx[train_n + val_n:]]

    # Also split task metrics for test evaluation
    test_metrics = {}
    for k, v in task_metrics.items():
        if isinstance(v, np.ndarray) and len(v) == len(obs):
            test_metrics[k] = v[idx[train_n + val_n:]]
        else:
            test_metrics[k] = v

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Agent boundaries for coordination metrics
    # simple_spread: self(4) + 4 others(4 each) + 5 landmarks(2 each) = 30
    if args.scenario == "simple_spread":
        agent_boundaries = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)]
    elif args.scenario == "simple_tag":
        agent_boundaries = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
    elif args.scenario == "simple_world_comm":
        agent_boundaries = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
    else:
        agent_boundaries = None

    # Config sweep
    betas = [0.01, 0.1, 1.0]
    free_bits = 0.1  # fixed — already proven effective
    task_loss_types = ["none", "self_only", "weighted"]
    task_weights = [0.0, 0.01, 0.1, 0.5, 1.0]

    results = []
    total_configs = len(betas) * len(task_loss_types) * len(task_weights)
    n_done = 0

    for beta, loss_type, tw in itertools.product(betas, task_loss_types, task_weights):
        # "none" type only makes sense with weight=0
        if loss_type == "none" and tw > 0:
            continue
        # non-none loss type with weight=0 is equivalent to "none" weight=0
        if loss_type != "none" and tw == 0:
            continue

        n_done += 1
        tag = f"β={beta:.2f} {loss_type} w={tw:.2f}"
        print(f"\n[{n_done}/{total_configs}] {tag}")

        ckpt_name = (f"task_{args.scenario}_b{beta}_fb{free_bits}_"
                     f"{loss_type}_w{tw}")

        model = train_task_aware(
            obs_dim, beta, free_bits, loss_type, tw,
            train_data, val_data, device, args.epochs, ckpt_name)

        # Evaluate
        task_eval = evaluate_task_metrics(
            model, test_data, test_metrics, agent_boundaries, device)

        r = {
            "scenario": args.scenario,
            "obs_dim": obs_dim,
            "beta": beta,
            "free_bits": free_bits,
            "task_loss_type": loss_type,
            "task_weight": tw,
            "mse": task_eval.get("mse", float("nan")),
            "self_mse": task_eval.get("self_mse", float("nan")),
            "others_mse": task_eval.get("others_mse", float("nan")),
            "coordination_gap": task_eval.get("coordination_gap", float("nan")),
            "kl": task_eval.get("kl", float("nan")),
            "rate_bits": task_eval.get("rate_bits", float("nan")),
            "per_agent_mses": task_eval.get("per_agent_mses", {}),
            "regime": (task_eval.get("regime") or
                       ("COLLAPSED" if task_eval.get("kl", 1) < 0.01 else "OK")),
        }
        results.append(r)
        print(f"  MSE={r['mse']:.4f}  Self={r['self_mse']:.4f}  "
              f"Others={r['others_mse']:.4f}  Gap={r['coordination_gap']:.2f}  "
              f"KL={r['kl']:.3f}  Regime={r['regime']}")

    # Summary: does task loss help?
    baseline_mses = [r for r in results if r["task_loss_type"] == "none"]
    best_task_mses = {}
    for r in results:
        if r["task_loss_type"] != "none":
            key = (r["beta"], r["task_loss_type"])
            if key not in best_task_mses or r["self_mse"] < best_task_mses[key]["self_mse"]:
                best_task_mses[key] = r

    print("\n=== Summary ===")
    for b in betas:
        bl = [r for r in baseline_mses if r["beta"] == b]
        if bl:
            print(f"β={b:.2f} baseline: MSE={bl[0]['mse']:.4f}  "
                  f"Self={bl[0]['self_mse']:.4f}  KL={bl[0]['kl']:.3f}")
        for (beta_k, lt), r in sorted(best_task_mses.items()):
            if beta_k == b:
                print(f"  + {lt} w={r['task_weight']:.2f}: "
                      f"Self={r['self_mse']:.4f}  KL={r['kl']:.3f}")

    out_path = ASSETS_DIR / "task_aware_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path} ({len(results)} entries)")


if __name__ == "__main__":
    main()
