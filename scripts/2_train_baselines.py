"""
scripts/2_train_baselines.py
一键训练所有baseline
"""
import numpy as np
import torch
import json
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obscodec.models.pca_baseline import PCABaseline
from obscodec.models.ae_baseline import StandardAE
from obscodec.models.digital_baseline import DigitalQuantizationBaseline
from obscodec.trainer import train_model


def main():
    # 加载数据
    data_path = "data/mpe_observations.npy"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            "Run 'python scripts/1_collect_data.py' first."
        )
    data = np.load(data_path)
    print(f"数据: {data.shape}")
    
    # 划分
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=42)
    
    obs_dim = data.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, obs_dim={obs_dim}")
    
    results = {}
    
    # ─── Baseline 1: PCA（多组） ───
    print("\n" + "="*60)
    print("Baseline 1: PCA")
    pca_results = []
    for n_comp in [2, 4, 8, 16, 32]:
        if n_comp > obs_dim:
            break
        pca = PCABaseline(n_components=n_comp).fit(train_data)
        _, mse = pca.forward(test_data)
        bw = pca.equivalent_bandwidth
        pca_results.append({"n_components": n_comp, "mse": float(mse), "bandwidth": float(bw)})
        print(f"  PCA-{n_comp:2d}: MSE={mse:.4f}, BW={bw:.0f} bits")
    results["pca"] = pca_results
    
    # ─── Baseline 2: Standard AE（多组） ───
    print("\n" + "="*60)
    print("Baseline 2: Standard Autoencoder")
    ae_results = []
    for latent_dim in [2, 4, 8, 16, 32]:
        ae = StandardAE(obs_dim=obs_dim, latent_dim=latent_dim)
        out = train_model(
            ae, train_data, val_data, epochs=200, device=device,
            model_name=f"AE-{latent_dim}"
        )
        # 在测试集上评估
        ae.eval()
        ae = ae.to(device)
        with torch.no_grad():
            test_t = torch.FloatTensor(test_data).to(device)
            x_hat, _ = ae(test_t)
            mse = torch.nn.functional.mse_loss(x_hat, test_t).item()
        bw = ae.equivalent_bandwidth
        ae_results.append({"latent_dim": latent_dim, "mse": float(mse), "bandwidth": float(bw)})
        print(f"  AE-{latent_dim:2d}: MSE={mse:.4f}, BW={bw:.0f} bits")
        # 保存最好的模型
        torch.save(ae.state_dict(), f"checkpoints/ae_ld{latent_dim}.pt")
    results["ae"] = ae_results
    
    # ─── Baseline 3: Digital量化 ───
    print("\n" + "="*60)
    print("Baseline 3: Digital Quantization")
    dig_results = []
    for bits in [2, 4, 6, 8]:
        for ld in [4, 8, 16]:
            dig = DigitalQuantizationBaseline(
                obs_dim=obs_dim, latent_dim=ld, bits_per_dim=bits
            )
            out = train_model(
                dig, train_data, val_data, epochs=200, device=device,
                model_name=f"Dig-LD{ld}-B{bits}"
            )
            dig.eval()
            dig = dig.to(device)
            with torch.no_grad():
                test_t = torch.FloatTensor(test_data).to(device)
                x_hat, _ = dig(test_t)
                mse = torch.nn.functional.mse_loss(x_hat, test_t).item()
            bw = dig.equivalent_bandwidth
            dig_results.append({
                "latent_dim": ld, "bits_per_dim": bits, "mse": float(mse), "bandwidth": float(bw)
            })
            print(f"  Dig-LD{ld}-B{bits}: MSE={mse:.4f}, BW={bw:.0f} bits")
            torch.save(dig.state_dict(), f"checkpoints/dig_ld{ld}_b{bits}.pt")
    results["digital"] = dig_results
    
    # 保存结果
    os.makedirs("assets", exist_ok=True)
    with open("assets/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nBaseline results saved to assets/baseline_results.json")


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()
