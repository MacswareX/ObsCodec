"""Cross-scenario unified codec — train one model on all scenarios, test per-scenario."""
import json, os, sys, time
os.chdir(r'C:\Users\Administrator\ObsCodec')
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
from obscodec.models.vae import BetaVAE
from obscodec.trainer import train_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = 'checkpoints'
ASSETS_DIR = 'assets'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

SCENARIOS = ['simple_spread', 'spread_hd', 'spread_xhd']
FB = 0.1
BETA = 2.0
LD = 16
EPOCHS = 300  # More epochs for multi-scenario data

print(f'Cross-Scenario Unified Codec')
print(f'Scenarios: {SCENARIOS}')
print(f'FB={FB}, beta={BETA}, LD={LD}, epochs={EPOCHS}')
print(f'Device: {DEVICE}')

# Load and combine all scenarios
all_train, all_val = [], []
test_sets = {}
obs_dims = {}

for scenario in SCENARIOS:
    data_path = f'data/{scenario}_obs.npy'
    data = np.load(data_path)
    obs_dim = data.shape[1]
    n = len(data)
    n_train = int(0.8 * n); n_val = int(0.1 * n)

    all_train.append(data[:n_train])
    all_val.append(data[n_train:n_train + n_val])
    test_sets[scenario] = data[n_train + n_val:]
    obs_dims[scenario] = obs_dim
    print(f'  {scenario}: {obs_dim}-dim, train={n_train}, val={n_val}, test={n - n_train - n_val}')

# Check all have same obs_dim for unified codec
unique_dims = set(obs_dims.values())
if len(unique_dims) > 1:
    print(f'\nWARNING: Inconsistent obs_dims: {obs_dims}')
    print('Padding all to max_dim...')
    max_dim = max(unique_dims)

    def pad_data(data_list, target_dim):
        padded = []
        for d in data_list:
            if d.shape[1] < target_dim:
                p = np.zeros((d.shape[0], target_dim), dtype=np.float32)
                p[:, :d.shape[1]] = d
                padded.append(p)
            else:
                padded.append(d.astype(np.float32))
        return padded

    all_train = pad_data(all_train, max_dim)
    all_val = pad_data(all_val, max_dim)
    for s in SCENARIOS:
        if test_sets[s].shape[1] < max_dim:
            p = np.zeros((test_sets[s].shape[0], max_dim), dtype=np.float32)
            p[:, :test_sets[s].shape[1]] = test_sets[s]
            test_sets[s] = p
    obs_dim = max_dim
else:
    obs_dim = unique_dims.pop()

train_data = np.concatenate(all_train, axis=0)
val_data = np.concatenate(all_val, axis=0)
np.random.default_rng(42).shuffle(train_data)
np.random.default_rng(42).shuffle(val_data)

print(f'\nCombined: {train_data.shape[0]} train, {val_data.shape[0]} val, obs_dim={obs_dim}')

# Train unified model
name = f'VAE-unified-LD{LD}-B{BETA}-FB{FB}-DM1'
ckpt_path = os.path.join(CHECKPOINT_DIR, f'{name}.pt')

if os.path.exists(ckpt_path):
    print(f'Unified model checkpoint exists — loading')
    model = BetaVAE(obs_dim=obs_dim, latent_dim=LD, free_bits=FB, beta=BETA).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state'])
else:
    print(f'Training unified model...')
    model = BetaVAE(obs_dim=obs_dim, latent_dim=LD, free_bits=FB, beta=BETA).to(DEVICE)
    t0 = time.time()
    out = train_model(model, train_data, val_data, epochs=EPOCHS, device=DEVICE,
                      model_name=name, lr=1e-3)
    model = out['model']
    print(f'Total train time: {time.time() - t0:.0f}s')

# Per-scenario evaluation
results = []
model.eval()

print(f'\n=== Per-Scenario Results ===')
print(f'{"Scenario":>20}  {"Dim":>4}  {"MSE":>8}  {"KL":>8}  {"Regime":>10}')

for scenario in SCENARIOS:
    test_t = torch.FloatTensor(test_sets[scenario]).to(DEVICE)
    with torch.no_grad():
        x_hat, z, mu, logvar = model(test_t)
        mse = float(F.mse_loss(x_hat, test_t))
        kl = float(model.kl_nats(test_t))
    regime = 'OK' if kl >= 0.1 else ('LOW' if kl >= 0.01 else 'COLLAPSED')
    print(f'{scenario:>20}  {obs_dims[scenario]:>4}  {mse:>8.4f}  {kl:>8.4f}  {regime:>10}')
    results.append({'scenario': scenario, 'obs_dim': obs_dims[scenario],
                    'mse': mse, 'kl': kl, 'regime': regime})

# Also test unified model on combined test set
all_test = np.concatenate([test_sets[s] for s in SCENARIOS], axis=0)
test_t = torch.FloatTensor(all_test).to(DEVICE)
with torch.no_grad():
    x_hat, z, mu, logvar = model(test_t)
    unified_mse = float(F.mse_loss(x_hat, test_t))
    unified_kl = float(model.kl_nats(test_t))
print(f'\nUnified (all test): MSE={unified_mse:.4f}, KL={unified_kl:.4f}')
results.append({'scenario': 'unified_all', 'obs_dim': obs_dim,
                'mse': unified_mse, 'kl': unified_kl,
                'regime': 'OK' if unified_kl >= 0.1 else ('LOW' if unified_kl >= 0.01 else 'COLLAPSED')})

# Compare with per-scenario models
print(f'\n=== Comparison: Unified vs Per-Scenario ===')
for scenario in SCENARIOS:
    scenario_model_path = os.path.join(CHECKPOINT_DIR, f'VAE-{scenario}-LD{LD}-B{BETA}-FB{FB}-DM1.pt')
    if os.path.exists(scenario_model_path):
        per_model = BetaVAE(obs_dim=obs_dims[scenario], latent_dim=LD,
                            free_bits=FB, beta=BETA).to(DEVICE)
        ckpt = torch.load(scenario_model_path, map_location=DEVICE, weights_only=True)
        per_model.load_state_dict(ckpt['model_state'])
        per_model.eval()
        test_t = torch.FloatTensor(test_sets[scenario]).to(DEVICE)
        with torch.no_grad():
            x_hat_p, z_p, mu_p, logvar_p = per_model(test_t)
            mse_per = float(F.mse_loss(x_hat_p, test_t))
            kl_per = float(per_model.kl_nats(test_t))
        uni_r = [r for r in results if r['scenario'] == scenario][0]
        mse_delta = (uni_r['mse'] - mse_per) / mse_per * 100
        print(f'{scenario}: per-model MSE={mse_per:.4f}, unified MSE={uni_r["mse"]:.4f} ({mse_delta:+.1f}%)')
    else:
        print(f'{scenario}: no per-scenario model to compare')

out_path = os.path.join(ASSETS_DIR, 'unified_codec_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved: {out_path}')
