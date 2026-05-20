"""Agent-count scaling study — spread scenario with N=3,5,7,10,12,15. Test if FB=0.1 works at all scales."""
import json, os, sys, time
os.chdir(r'C:\Users\Administrator\ObsCodec')
sys.path.insert(0, '.')

import torch
import numpy as np
from obscodec.data.synthetic import generate_spread
from obscodec.models.vae import BetaVAE
from obscodec.trainer import train_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = 'checkpoints'
DATA_DIR = 'data'
ASSETS_DIR = 'assets'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

AGENT_COUNTS = [3, 5, 7, 10, 12, 15]
FB = 0.1
BETA = 2.0
LD = 16
EPOCHS = 200
SAMPLES = 33333

print(f'Agent-count scaling: N={AGENT_COUNTS}')
print(f'FB={FB}, beta={BETA}, LD={LD}, epochs={EPOCHS}')
print(f'Device: {DEVICE}')

results = []

for n_agents in AGENT_COUNTS:
    obs_dim = n_agents * 6  # 6 features per agent: pos(2), vel(2), target_pos(2)
    data_path = os.path.join(DATA_DIR, f'spread_N{n_agents}_obs.npy')

    # Generate data if needed
    if not os.path.exists(data_path):
        print(f'\n=== N={n_agents} — Generating data ({obs_dim}-dim) ===')
        obs = generate_spread(n_agents=n_agents, n_landmarks=n_agents,
                               n_samples=SAMPLES, seed=42 + n_agents)
        np.save(data_path, obs)
    else:
        print(f'\n=== N={n_agents} — Data exists ({obs_dim}-dim) ===')
        obs = np.load(data_path)

    n = len(obs); n_train = int(0.8 * n); n_val = int(0.1 * n)
    train_data = obs[:n_train]; val_data = obs[n_train:n_train + n_val]
    test_data = obs[n_train + n_val:]

    # Train with FB=0.0 (baseline) and FB=0.1
    for fb_val in [0.0, 0.1]:
        name = f'VAE-spread_N{n_agents}-LD{LD}-B{BETA}-FB{fb_val}-DM1'
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'{name}.pt')

        if os.path.exists(ckpt_path):
            print(f'  FB={fb_val} — SKIP (checkpoint)')
            model = BetaVAE(obs_dim=obs_dim, latent_dim=LD, free_bits=fb_val, beta=BETA).to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(ckpt['model_state'])
        else:
            print(f'  FB={fb_val} — TRAINING...')
            model = BetaVAE(obs_dim=obs_dim, latent_dim=LD, free_bits=fb_val, beta=BETA).to(DEVICE)
            t0 = time.time()
            out = train_model(model, train_data, val_data, epochs=EPOCHS, device=DEVICE,
                              model_name=name, lr=1e-3)
            model = out['model']
            print(f'  Time: {time.time() - t0:.0f}s')

        model.eval()
        test_t = torch.FloatTensor(test_data).to(DEVICE)
        with torch.no_grad():
            x_hat, z, mu, logvar = model(test_t)
            mse = float(torch.nn.functional.mse_loss(x_hat, test_t))
            kl = float(model.kl_nats(test_t))

        regime = 'OK' if kl >= 0.1 else ('LOW' if kl >= 0.01 else 'COLLAPSED')
        print(f'  MSE={mse:.4f}, KL={kl:.4f}, regime={regime}')
        results.append({'n_agents': n_agents, 'obs_dim': obs_dim, 'fb': fb_val,
                        'mse': mse, 'kl': kl, 'regime': regime})

# Save
out_path = os.path.join(ASSETS_DIR, 'agent_scaling_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved: {out_path}')

# Summary table
print('\n=== Agent Scaling Summary ===')
print(f'{"N":>4}  {"Dim":>4}  {"FB=0.0 KL":>10}  {"FB=0.0 Regime":>14}  {"FB=0.1 KL":>10}  {"FB=0.1 Regime":>14}  {"MSE Delta":>10}')
for n_agents in AGENT_COUNTS:
    r0 = [r for r in results if r['n_agents'] == n_agents and r['fb'] == 0.0][0]
    r1 = [r for r in results if r['n_agents'] == n_agents and r['fb'] == 0.1][0]
    mse_delta = (r1['mse'] - r0['mse']) / r0['mse'] * 100 if r0['mse'] > 0 else 0
    print(f'{n_agents:>4}  {r0["obs_dim"]:>4}  {r0["kl"]:>10.4f}  {r0["regime"]:>14}  {r1["kl"]:>10.4f}  {r1["regime"]:>14}  {mse_delta:>+9.1f}%')
