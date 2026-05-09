"""ObsCodec visualization helpers for rate-distortion analysis."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from obscodec import dedupe_by_name

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
    "legend.fontsize": 9, "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

METHOD_COLORS: dict[str, str] = {
    "pca": "#6c757d",
    "ae": "#1f77b4",
    "digital": "#f59f00",
    "vae": "#8b0000",
    "vqvae": "#087f5b",
}


# ═══ Figure 1: Rate-Distortion ═══
def plot_rate_distortion(
    pca_results=None, ae_results=None, digital_results=None,
    vae_results=None, vqvae_results=None,
    save_path="assets/rate_distortion.png",
):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    if pca_results:
        bw = [r["bandwidth"] for r in pca_results]
        ax.plot(bw, [r["mse"] for r in pca_results], 's--', color=METHOD_COLORS["pca"],
                label='PCA', lw=2, ms=8)
    if ae_results:
        bw = [r["bandwidth"] for r in ae_results]
        ax.plot(bw, [r["mse"] for r in ae_results], '^--', color=METHOD_COLORS["ae"],
                label='Standard AE', lw=2, ms=8)
    if digital_results:
        ax.scatter([r["bandwidth"] for r in digital_results],
                   [r["mse"] for r in digital_results],
                   marker='D', color=METHOD_COLORS["digital"], alpha=0.55, s=40,
                   label='Digital Quant.')
    if vae_results:
        betas = sorted(set(r["beta"] for r in vae_results))
        colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(betas)))
        for beta, c in zip(betas, colors):
            s = [r for r in vae_results if r["beta"] == beta]
            ax.plot([r["bandwidth"] for r in s], [r["mse"] for r in s],
                    'o-', color=c, alpha=0.85, ms=5, lw=1.2,
                    label=f'β-VAE β={beta}')
    if vqvae_results:
        # VQ-VAE: 只画 commitment_cost=0.25 的点（避免图太乱）
        vq_default = [r for r in vqvae_results
                      if r.get("commitment_cost", 0.25) == 0.25]
        ax.scatter([r["bandwidth"] for r in vq_default],
                   [r["mse"] for r in vq_default],
                   marker='*', color=METHOD_COLORS["vqvae"], s=110,
                   label='VQ-VAE', zorder=10, edgecolors='black', linewidths=0.4)
    ax.set_xlabel("Equivalent Bandwidth (bits/obs)", fontsize=12)
    ax.set_ylabel("Reconstruction MSE", fontsize=12)
    ax.set_title("Rate-Distortion: Multi-Agent Observation Compression", fontsize=13)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"wrote {save_path}")


# ═══ Figure 2: β-VAE Ablation Heatmap ═══
def plot_ablation_heatmap(vae_results, save_path="assets/ablation_heatmap.png"):
    latent_dims = sorted(set(r["latent_dim"] for r in vae_results))
    betas = sorted(set(r["beta"] for r in vae_results))
    lookup = {(r["latent_dim"], r["beta"]): r["mse"] for r in vae_results}
    matrix = np.zeros((len(latent_dims), len(betas)))
    for i, ld in enumerate(latent_dims):
        for j, b in enumerate(betas):
            matrix[i, j] = lookup.get((ld, b), np.nan)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(range(len(betas))); ax.set_xticklabels(betas)
    ax.set_yticks(range(len(latent_dims))); ax.set_yticklabels(latent_dims)
    ax.set_xlabel("β (KL weight)", fontsize=12)
    ax.set_ylabel("Latent Dimension", fontsize=12)
    ax.set_title("β-VAE Ablation: Latent Dim × β → MSE", fontsize=13)
    for i in range(len(latent_dims)):
        for j in range(len(betas)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha='center', va='center',
                        fontsize=8, color='white' if v > 0.3 else 'black')
    plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"wrote {save_path}")


# ═══ Figure 3: KL Collapse ═══
def plot_kl_collapse(vae_results, save_path="assets/kl_collapse.png"):
    kl_by_beta = defaultdict(list)
    mse_by_beta = defaultdict(list)
    for r in vae_results:
        kl_by_beta[r["beta"]].append(r["kl"])
        mse_by_beta[r["beta"]].append(r["mse"])
    betas = sorted(kl_by_beta.keys())
    kl_means = [np.mean(kl_by_beta[b]) for b in betas]
    kl_stds  = [np.std(kl_by_beta[b]) for b in betas]
    mse_means = [np.mean(mse_by_beta[b]) for b in betas]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.set_xlabel('β (KL weight)', fontsize=12)
    ax1.set_ylabel('KL Divergence (nats)', color='tab:red', fontsize=12)
    ax1.errorbar(betas, kl_means, yerr=kl_stds, marker='o', color='tab:red',
                 capsize=4, lw=2, label='KL Divergence')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_xscale('log')
    ax1.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5,
                label='collapse threshold')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Reconstruction MSE', color='tab:blue', fontsize=12)
    ax2.plot(betas, mse_means, 'D--', color='tab:blue', lw=1.5, ms=6,
             label='Reconstruction MSE')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='center left', fontsize=10, framealpha=0.9)
    plt.title('β-VAE: KL Collapse vs Reconstruction Quality', fontsize=13)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"wrote {save_path}")


# ═══ Figure 4: Latent Space ═══
def plot_latent_space(vae_checkpoint="checkpoints/vae_ld8_b1.0.pt",
                      data_path="data/mpe_observations.npy",
                      obs_dim=18, latent_dim=8, beta=1.0,
                      save_path="assets/latent_space.png", n_samples=2000):
    import torch
    from obscodec.models.vae import BetaVAE
    data = np.load(data_path)
    idx = np.random.choice(len(data), n_samples, replace=False)
    sample = data[idx]
    vae = BetaVAE(obs_dim=obs_dim, latent_dim=latent_dim, beta=beta)
    state = torch.load(vae_checkpoint, map_location='cpu', weights_only=True)
    vae.load_state_dict(state); vae.eval()
    with torch.no_grad():
        mu, _ = vae.encoder(torch.FloatTensor(sample))
    latents = mu.numpy()
    latents_2d = PCA(n_components=2).fit_transform(latents)
    colors = PCA(n_components=1).fit_transform(sample)[:, 0]
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(latents_2d[:, 0], latents_2d[:, 1],
                    c=colors, cmap='plasma', alpha=0.6, s=8)
    plt.colorbar(sc, ax=ax, label='Obs PC1')
    ax.set_xlabel("Latent PC1"); ax.set_ylabel("Latent PC2")
    ax.set_title(f'β-VAE Latent Space (LD={latent_dim}, β={beta}, PCA)', fontsize=13)
    ax.set_aspect('equal')
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"wrote {save_path}")


# ═══ Figure 5: Reconstruction Comparison ═══
def plot_reconstruction_comparison(data_path="data/mpe_observations.npy",
                                   save_path="assets/reconstruction_comparison.png"):
    import torch
    from obscodec.models.ae_baseline import StandardAE
    from obscodec.models.vae import BetaVAE
    data = np.load(data_path)
    idx = np.random.choice(len(data), 500, replace=False)
    test = data[idx]
    obs_dim = data.shape[1]
    recon_dict = {}
    try:
        ae = StandardAE(obs_dim=obs_dim, latent_dim=8)
        ae.load_state_dict(torch.load("checkpoints/ae_ld8.pt",
                          map_location='cpu', weights_only=True))
        ae.eval()
        with torch.no_grad(): xh, _ = ae(torch.FloatTensor(test))
        recon_dict["AE (LD=8)"] = xh.numpy()
    except Exception as e: print(f"  AE skip: {e}")
    try:
        vae = BetaVAE(obs_dim=obs_dim, latent_dim=8, beta=1.0)
        vae.load_state_dict(torch.load("checkpoints/vae_ld8_b1.0.pt",
                            map_location='cpu', weights_only=True))
        vae.eval()
        with torch.no_grad(): xh, _ = vae(torch.FloatTensor(test))
        recon_dict["β-VAE (LD=8,β=1)"] = xh.numpy()
    except Exception as e: print(f"  VAE skip: {e}")
    if not recon_dict: return
    n_methods = len(recon_dict) + 1; n_s = 5
    samp = np.random.choice(len(test), n_s, replace=False)
    fig, axes = plt.subplots(n_methods, n_s, figsize=(2.8*n_s, 2.5*n_methods))
    if n_methods == 1: axes = axes.reshape(1, -1)
    for j in range(n_s):
        axes[0,j].bar(range(obs_dim), test[samp[j]], color='steelblue', alpha=0.8)
        if j==0: axes[0,j].set_ylabel("Original", fontsize=10)
    for i,(name,recon) in enumerate(recon_dict.items()):
        for j in range(n_s):
            axes[i+1,j].bar(range(obs_dim), recon[samp[j]], color='coral', alpha=0.8)
            if j==0: axes[i+1,j].set_ylabel(name, fontsize=10)
    plt.suptitle("Reconstruction Comparison", fontsize=12)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"wrote {save_path}")


# ═══ Figure 6: VQ-VAE Commitment Cost Sweep ═══
def plot_vqvae_commitment_sweep(vqvae_results, save_path="assets/vqvae_commitment.png"):
    """Plot the VQ-VAE commitment-cost sweep."""
    cc_data = [r for r in vqvae_results if r.get("commitment_cost") is not None
               and r.get("codebook_size") == 256 and r.get("latent_dim") == 8]
    if len(cc_data) < 3:
        print("  VQ commitment sweep skipped: insufficient data")
        return

    cc_data = sorted(cc_data, key=lambda x: x["commitment_cost"])
    ccs = [r["commitment_cost"] for r in cc_data]
    mses = [r["mse"] for r in cc_data]
    usages = [r.get("codebook_usage", 0) for r in cc_data]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.set_xlabel('Commitment Cost (VQ-VAE "β")', fontsize=12)
    ax1.set_ylabel('Reconstruction MSE', color='tab:blue', fontsize=12)
    ax1.plot(ccs, mses, 'D-', color='tab:blue', lw=2, ms=8, label='MSE')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xscale('log')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Codebook Usage (%)', color='tab:green', fontsize=12)
    ax2.plot(ccs, [u*100 for u in usages], 's--', color='tab:green',
             lw=1.5, ms=8, label='Codebook Usage %')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.set_ylim(-5, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='center left', fontsize=10, framealpha=0.9)
    plt.title('VQ-VAE: Commitment Cost vs MSE & Codebook Usage (LD=8, CB=256)', fontsize=13)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"wrote {save_path}")


# ═══ Figure 7: VQ-VAE Codebook Usage Heatmap ═══
def plot_vqvae_usage_heatmap(vqvae_results, save_path="assets/vqvae_usage_heatmap.png"):
    """Plot which VQ-VAE configurations use their codebooks."""
    default = [r for r in vqvae_results
               if r.get("commitment_cost", 0.25) == 0.25]
    if len(default) < 5:
        print("  VQ usage heatmap skipped: insufficient data")
        return

    lds = sorted(set(r["latent_dim"] for r in default))
    cbs = sorted(set(r["codebook_size"] for r in default))
    lookup = {(r["latent_dim"], r["codebook_size"]): r.get("codebook_usage", 0)
              for r in default}
    matrix = np.zeros((len(lds), len(cbs)))
    for i, ld in enumerate(lds):
        for j, cb in enumerate(cbs):
            matrix[i, j] = lookup.get((ld, cb), np.nan)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(cbs))); ax.set_xticklabels(cbs)
    ax.set_yticks(range(len(lds))); ax.set_yticklabels(lds)
    ax.set_xlabel("Codebook Size", fontsize=12)
    ax.set_ylabel("Latent Dimension", fontsize=12)
    ax.set_title("VQ-VAE Codebook Usage (commitment_cost=0.25)", fontsize=13)
    for i in range(len(lds)):
        for j in range(len(cbs)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1%}", ha='center', va='center',
                        fontsize=9, color='white' if v < 0.5 else 'black')
    plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"wrote {save_path}")


# ═══ Figure 8: Pareto Frontier ═══
def plot_pareto_frontier(
    pca_results=None, ae_results=None, digital_results=None,
    vae_results=None, vqvae_results=None,
    save_path="assets/pareto_frontier.png",
):
    """Plot the budgeted lower envelope for each method."""
    fig, ax = plt.subplots(figsize=(9, 6))

    def pareto_xy(results, bw_key="bandwidth", mse_key="mse"):
        """Best MSE available with bandwidth <= each observed budget."""
        bw_groups = defaultdict(list)
        for r in results:
            bw_groups[r[bw_key]].append(r[mse_key])
        bws = sorted(bw_groups.keys())
        mses = []
        best_so_far = float("inf")
        for bandwidth in bws:
            best_so_far = min(best_so_far, min(bw_groups[bandwidth]))
            mses.append(best_so_far)
        return bws, mses

    if pca_results:
        bx, mx = pareto_xy(pca_results)
        ax.plot(bx, mx, 's-', color=METHOD_COLORS["pca"], label='PCA', lw=2, ms=7)
    if ae_results:
        bx, mx = pareto_xy(ae_results)
        ax.plot(bx, mx, '^-', color=METHOD_COLORS["ae"], label='Standard AE', lw=2, ms=7)
    if digital_results:
        bx, mx = pareto_xy(digital_results)
        ax.plot(bx, mx, 'D-', color=METHOD_COLORS["digital"], label='Digital Quant.', lw=1.8, ms=6)
    if vae_results:
        bx, mx = pareto_xy(vae_results)
        ax.plot(bx, mx, 'o-', color=METHOD_COLORS["vae"],
                label='β-VAE (budgeted best)', lw=2, ms=7)
    if vqvae_results:
        bx, mx = pareto_xy(vqvae_results)
        ax.plot(bx, mx, '*-', color=METHOD_COLORS["vqvae"],
                label='VQ-VAE (budgeted best)', lw=2, ms=10)

    # 标注原始带宽（576 bits = 18 dim × 32 bits）
    ax.axvline(x=576, color='black', linestyle=':', alpha=0.4, linewidth=1)
    ax.text(576, ax.get_ylim()[1]*0.95, 'Raw (576b)', ha='right', fontsize=9, alpha=0.5)

    ax.set_xlabel("Equivalent Bandwidth (bits/obs)", fontsize=12)
    ax.set_ylabel("Best Reconstruction MSE Within Budget", fontsize=12)
    ax.set_title("Budgeted Rate-Distortion Frontier per Method", fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"wrote {save_path}")
    
    
    # ═══ 统一入口 ═══
def generate_all() -> None:
    os.makedirs("assets", exist_ok=True)

    def load(path: str) -> Any:
        if not os.path.exists(path):
            print(f"  (skip) {path} not found")
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  (skip) {path}: {e}")
            return None

    baseline = load("assets/baseline_results.json") or {}
    vae_r    = load("assets/vae_results.json")
    vqvae_r  = dedupe_by_name(load("assets/vqvae_results.json") or [])

    plot_rate_distortion(
        pca_results=baseline.get("pca"),
        ae_results=baseline.get("ae"),
        digital_results=baseline.get("digital"),
        vae_results=vae_r,
        vqvae_results=vqvae_r,
    )
    if vae_r:
        plot_ablation_heatmap(vae_r)
        plot_kl_collapse(vae_r)
    if (
        os.path.exists("checkpoints/vae_ld8_b1.0.pt")
        and os.path.exists("data/mpe_observations.npy")
    ):
        plot_latent_space()
    if os.path.exists("data/mpe_observations.npy"):
        plot_reconstruction_comparison()
    else:
        print("  Reconstruction comparison skipped: data/mpe_observations.npy missing")
    if vqvae_r:
        plot_vqvae_commitment_sweep(vqvae_r)
        plot_vqvae_usage_heatmap(vqvae_r)
    plot_pareto_frontier(
        pca_results=baseline.get("pca"),
        ae_results=baseline.get("ae"),
        digital_results=baseline.get("digital"),
        vae_results=vae_r,
        vqvae_results=vqvae_r,
    )

    print("\nall available figures generated")
    print("  numeric metrics: assets/results_summary.md")


if __name__ == "__main__":
    generate_all()
