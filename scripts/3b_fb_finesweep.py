"""FB fine-sweep on spread_xhd at beta=2.0 — find minimum effective free-bits dose."""
import json, os, sys, time
os.chdir(r'C:\Users\Administrator\ObsCodec')
sys.path.insert(0, '.')

import torch
import numpy as np
from obscodec.models.vae import BetaVAE
from obscodec.trainer import train_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = 'checkpoints'
ASSETS_DIR = 'assets'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

FB_VALUES = [0.0, 0.02, 0.05, 0.07, 0.10, 0.12, 0.15, 0.17, 0.20, 0.25]
BETA = 2.0
LD = 16
EPOCHS = 200

data = np.load('data/spread_xhd_obs.npy')
obs_dim = data.shape[1]; n = len(data)
n_train = int(0.8 * n); n_val = int(0.1 * n)
train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:]

print(f'spread_xhd: {n} samples, obs_dim={obs_dim}')
print(f'Device: {DEVICE}')
print(f'FB sweep: {FB_VALUES}')
print(f'beta={BETA}, LD={LD}, epochs={EPOCHS}')

results = []
test_t = torch.FloatTensor(test_data).to(DEVICE)

for fb in FB_VALUES:
    name = f'VAE-spread_xhd-LD{LD}-B{BETA}-FB{fb}-DM1'
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{name}.pt')

    if os.path.exists(ckpt_path):
        print(f'\n=== FB={fb} — SKIP (checkpoint exists) ===')
        model = BetaVAE(obs_dim=obs_dim, latent_dim=LD, free_bits=fb, beta=BETA).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt['model_state'])
    else:
        print(f'\n=== FB={fb} — TRAINING ===')
        model = BetaVAE(obs_dim=obs_dim, latent_dim=LD, free_bits=fb, beta=BETA).to(DEVICE)
        t0 = time.time()
        out = train_model(model, train_data, val_data, epochs=EPOCHS, device=DEVICE,
                          model_name=name, lr=1e-3)
        model = out['model']
        elapsed = time.time() - t0
        print(f'  Train time: {elapsed:.0f}s')

    model.eval()
    with torch.no_grad():
        x_hat, z, mu, logvar = model(test_t)
        mse = float(torch.nn.functional.mse_loss(x_hat, test_t))
        kl = float(model.kl_nats(test_t))

    regime = 'OK' if kl >= 0.1 else ('LOW' if kl >= 0.01 else 'COLLAPSED')
    print(f'  MSE={mse:.4f}, KL={kl:.4f}, regime={regime}')
    results.append({'fb': fb, 'mse': mse, 'kl': kl, 'regime': regime})

# Save results
out_path = os.path.join(ASSETS_DIR, 'fb_finesweep_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved: {out_path}')

# Report
print('\n=== FB Fine-Sweep Results ===')
print(f'{"FB":>8}  {"MSE":>8}  {"KL":>8}  {"Regime":>10}')
for r in results:
    print(f'{r["fb"]:>8.2f}  {r["mse"]:>8.4f}  {r["kl"]:>8.4f}  {r["regime"]:>10}')

collapsed_fbs = [r['fb'] for r in results if r['regime'] == 'COLLAPSED']
min_ok_fb = min([r['fb'] for r in results if r['regime'] == 'OK'], default=None)
print(f'\nCollapsed at FB: {collapsed_fbs}')
print(f'Minimum effective FB dose: {min_ok_fb}')
