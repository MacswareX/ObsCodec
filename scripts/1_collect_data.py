"""收集MPE多智能体观测 — PettingZoo 1.24 适配"""
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from tqdm import tqdm
import os

N_AGENTS = 3
N_SAMPLES = 50_000
MAX_STEPS = 25

def collect_observations(n_samples: int = N_SAMPLES) -> np.ndarray:
    env = simple_spread_v3.parallel_env(
        N=N_AGENTS, max_cycles=MAX_STEPS, continuous_actions=True
    )
    
    obs_dict, _ = env.reset()
    first_agent = list(obs_dict.keys())[0]
    obs_dim = obs_dict[first_agent].shape[0]
    print(f"环境: simple_spread, {N_AGENTS} agents, obs_dim={obs_dim}")
    
    all_obs = np.zeros((n_samples, obs_dim), dtype=np.float32)
    idx = 0
    pbar = tqdm(total=n_samples, desc="收集观测")
    
    while idx < n_samples:
        obs_dict, _ = env.reset()
        for _ in range(MAX_STEPS):
            actions = {
                agent: env.action_space(agent).sample()
                for agent in env.agents
            }
            obs_dict, _, terms, truncs, _ = env.step(actions)
            for agent in env.agents:
                if idx < n_samples:
                    all_obs[idx] = obs_dict[agent]
                    idx += 1
                    pbar.update(1)
                else:
                    break
            if all(terms.values()) or all(truncs.values()):
                break
    
    pbar.close()
    env.close()
    
    os.makedirs("data", exist_ok=True)
    np.save("data/mpe_observations.npy", all_obs)
    print(f"Saved {idx} samples -> data/mpe_observations.npy")
    print(f"  统计: mean={all_obs.mean():.4f}, std={all_obs.std():.4f}")
    return all_obs

if __name__ == "__main__":
    collect_observations()
