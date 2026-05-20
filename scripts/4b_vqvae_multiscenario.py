"""VQ-VAE multi-scenario + channel robustness integration."""
import json, os, sys, time
os.chdir(r'C:\Users\Administrator\ObsCodec')
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
from obscodec.models.vqvae import VQVAE
from obscodec.trainer import train_model
from obscodec.channel.impairments import AWGNChannel, RayleighFadingChannel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = 'checkpoints'
ASSETS_DIR = 'assets'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

SCENARIOS = ['simple_spread', 'spread_hd', 'spread_xhd']
CB_SIZES = [128, 256, 512]
COMMITMENT_COSTS = [0.25, 1.0]
LD = 16
EPOCHS = 200

SNR_VALUES = [None, 20, 10, 5, 0]  # None = clean

print(f'VQ-VAE multi-scenario + channel integration')
print(f'Scenarios: {SCENARIOS}')
print(f'Codebook sizes: {CB_SIZES}, commitment costs: {COMMITMENT_COSTS}')
print(f'LD={LD}, epochs={EPOCHS}')
print(f'Device: {DEVICE}')

results = []

for scenario in SCENARIOS:
    data_path = f'data/{scenario}_obs.npy'
    data = np.load(data_path)
    obs_dim = data.shape[1]
    n = len(data); n_train = int(0.8 * n); n_val = int(0.1 * n)
    train_data = data[:n_train]; val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    print(f'\n{"="*60}')
    print(f'Scenario: {scenario} ({obs_dim}-dim, {n} samples)')

    for cb_size in CB_SIZES:
        for cc in COMMITMENT_COSTS:
            name = f'VQVAE-{scenario}-LD{LD}-CB{cb_size}-CC{cc}'
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'{name}.pt')

            if os.path.exists(ckpt_path):
                print(f'  CB={cb_size}, CC={cc} — SKIP (checkpoint)')
                model = VQVAE(obs_dim=obs_dim, latent_dim=LD,
                              num_embeddings=cb_size, commitment_cost=cc).to(DEVICE)
                ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
                model.load_state_dict(ckpt['model_state'])
            else:
                print(f'  CB={cb_size}, CC={cc} — TRAINING...')
                model = VQVAE(obs_dim=obs_dim, latent_dim=LD,
                              num_embeddings=cb_size, commitment_cost=cc).to(DEVICE)
                t0 = time.time()
                out = train_model(model, train_data, val_data, epochs=EPOCHS, device=DEVICE,
                                  model_name=name, lr=1e-3)
                model = out['model']
                print(f'  Time: {time.time() - t0:.0f}s')

            # Evaluate clean + channel-impaired
            model.eval()
            test_t = torch.FloatTensor(test_data).to(DEVICE)

            for snr in SNR_VALUES:
                if snr is None:
                    # Clean
                    with torch.no_grad():
                        x_hat, z_q, indices, _ = model(test_t)
                        mse = float(F.mse_loss(x_hat, test_t))
                    cb = model.codebook_usage
                    cb_usage = float(cb() if callable(cb) else cb)
                    tag = 'clean'
                else:
                    # AWGN channel on latent
                    awgn = AWGNChannel(snr_db=snr)
                    with torch.no_grad():
                        z_e = model.encoder(test_t)
                        z_noisy = awgn(z_e)
                        x_hat = model.decoder(z_noisy)
                        mse = float(F.mse_loss(x_hat, test_t))
                    cb = model.codebook_usage
                    cb_usage = float(cb() if callable(cb) else cb)
                    tag = f'AWGN_{snr}dB'

                results.append({
                    'scenario': scenario, 'obs_dim': obs_dim,
                    'num_embeddings': cb_size, 'commitment_cost': cc,
                    'snr': tag, 'mse': mse, 'codebook_usage': cb_usage
                })
                print(f'    {tag}: MSE={mse:.4f}, CB_usage={cb_usage:.3f}')

            # Rayleigh fading test
            rayleigh = RayleighFadingChannel(snr_db=10)
            with torch.no_grad():
                z_e = model.encoder(test_t)
                z_faded = rayleigh(z_e)
                x_hat = model.decoder(z_faded)
                mse_ray = float(F.mse_loss(x_hat, test_t))
            results.append({
                'scenario': scenario, 'obs_dim': obs_dim,
                'num_embeddings': cb_size, 'commitment_cost': cc,
                'snr': 'Rayleigh_10dB', 'mse': mse_ray,
                'codebook_usage': float(cb() if callable(cb) else cb)
            })
            print(f'    Rayleigh_10dB: MSE={mse_ray:.4f}')

out_path = os.path.join(ASSETS_DIR, 'vqvae_multiscenario_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved: {out_path}')

# Best config summary
print('\n=== Best VQ-VAE per scenario ===')
for scenario in SCENARIOS:
    scenario_results = [r for r in results if r['scenario'] == scenario and r['snr'] == 'clean']
    if scenario_results:
        best = min(scenario_results, key=lambda r: r['mse'])
        print(f'{scenario}: CB={best["num_embeddings"]}, CC={best["commitment_cost"]}, MSE={best["mse"]:.4f}')
