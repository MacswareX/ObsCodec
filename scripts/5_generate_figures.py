"""Step 5: Generate all figures and run cross-codec evaluation.

Figures produced:
  1. Rate-distortion curves (per scenario)
  2. KL vs beta across scenarios (Step B)
  3. Collapse barrier comprehensive figure (Step C, 6-panel)
  4. Cross-scenario FB validation figure (3-panel)
  5. Channel robustness figure

Reads all results JSONs from assets/. Saves all PNGs to assets/.

Usage: python scripts/5_generate_figures.py [--all]
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from obscodec.config import ASSETS_DIR
from obscodec.visualize import (set_style, save_figure, collapse_threshold_line,
    target_zone, scenario_color, scenario_marker, scenario_label, CODEC_COLORS)

set_style()


def load_json(name):
    path = ASSETS_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════
# Figure 1: Rate-Distortion curves per scenario
# ═══════════════════════════════════════════════════════════════════

def figure_rate_distortion():
    pca = load_json("pca_results.json") or []
    ae = load_json("ae_results.json") or []
    digital = load_json("digital_results.json") or []
    vae = load_json("vae_results.json") or []
    vqvae = load_json("vqvae_results.json") or [] if Path(ASSETS_DIR / "vqvae_results.json").exists() else load_json("vqvae_stepA_results.json") or []

    scenarios = sorted(set(
        r.get("scenario", "") for r in (pca + ae + digital + vae + vqvae)
    ))

    for scenario in scenarios:
        fig, ax = plt.subplots(figsize=(8, 5.5))

        # PCA
        items = sorted([r for r in pca if r.get("scenario") == scenario],
                       key=lambda x: x.get("latent_dim", 0))
        if items:
            ax.plot([r["latent_dim"] for r in items], [r["mse"] for r in items],
                    "o-", color=CODEC_COLORS["PCA"], linewidth=2, markersize=7, label="PCA")

        # AE
        items = sorted([r for r in ae if r.get("scenario") == scenario],
                       key=lambda x: x.get("latent_dim", 0))
        if items:
            ax.plot([r["latent_dim"] for r in items], [r["mse"] for r in items],
                    "s-", color=CODEC_COLORS["AE"], linewidth=2, markersize=7, label="AE")

        # Digital
        items = sorted([r for r in digital if r.get("scenario") == scenario],
                       key=lambda x: x.get("total_bits", 0))
        if items:
            ax.plot([r["total_bits"] for r in items], [r["mse"] for r in items],
                    "^-", color=CODEC_COLORS["Digital"], linewidth=2, markersize=7, label="Digital")

        # VAE
        items = sorted([r for r in vae if r.get("scenario") == scenario],
                       key=lambda x: x.get("rate_bits", 0))
        if items:
            ax.plot([r.get("rate_bits", r.get("latent_dim", 0)) for r in items],
                    [r["mse"] for r in items],
                    "D-", color=CODEC_COLORS["VAE"], linewidth=2, markersize=7, label="VAE")

        # VQ-VAE
        items = sorted([r for r in vqvae if r.get("scenario") == scenario],
                       key=lambda x: x.get("rate_bits", 0))
        if items:
            ax.plot([r.get("rate_bits", r.get("latent_dim", 0)) for r in items],
                    [r["mse"] for r in items],
                    "p-", color=CODEC_COLORS["VQ-VAE"], linewidth=2, markersize=7, label="VQ-VAE")

        ax.set_xlabel("Rate (bits)", fontsize=11)
        ax.set_ylabel("MSE", fontsize=11)
        ax.set_title(f"Rate-Distortion: {scenario}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        safe_name = scenario.replace("/", "_")
        save_figure(fig, f"rate_distortion_{safe_name}.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 2: KL vs beta across all scenarios + anti-collapse
# ═══════════════════════════════════════════════════════════════════

def figure_kl_vs_beta():
    stepb = load_json("vae_stepB_results.json") or []
    fb_full = load_json("collapse_barrier_full_results.json") or []
    fb_cross = load_json("fb_cross_scenario_validation.json") or []

    if not stepb:
        print("  SKIP: no Step B results")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Step B baselines (FB=0)
    by_scenario = {}
    for r in stepb:
        by_scenario.setdefault(r["scenario"], []).append(r)
    for name in sorted(by_scenario.keys()):
        items = sorted(by_scenario[name], key=lambda x: x["beta"])
        betas = [r["beta"] for r in items]
        kls = [max(r["kl"], 1e-5) for r in items]
        ax.loglog(betas, kls, marker=scenario_marker(name), color=scenario_color(name),
                  linewidth=1.5, markersize=7, alpha=0.5, linestyle="--",
                  label=f"{scenario_label(name)} (FB=0)")

    # FB=0.1 overlays
    fb01_spread = sorted([r for r in fb_full if r.get("decoder_mult") == 1 and r.get("free_bits") == 0.1],
                         key=lambda x: x["beta"])
    if fb01_spread:
        ax.loglog([r["beta"] for r in fb01_spread], [max(r["kl"], 1e-5) for r in fb01_spread],
                  marker="D", color=scenario_color("spread_xhd"), linewidth=2.2, markersize=9,
                  label="spread_xhd (FB=0.1)")

    for scenario in ["tag_hd", "comm_hd"]:
        fb01 = sorted([r for r in fb_cross if r.get("scenario") == scenario and r.get("free_bits") == 0.1],
                      key=lambda x: x["beta"])
        if fb01:
            ax.loglog([r["beta"] for r in fb01], [max(r["kl"], 1e-5) for r in fb01],
                      marker=scenario_marker(scenario), color=scenario_color(scenario),
                      linewidth=2.2, markersize=9,
                      label=f"{scenario_label(scenario)} (FB=0.1)")

    collapse_threshold_line(ax)
    target_zone(ax)
    ax.set_xlabel("beta (KL weight)", fontsize=11)
    ax.set_ylabel("KL Divergence (nats)", fontsize=11)
    ax.set_title("FB=0.1 Universal Anti-Collapse Across All Scenarios", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.3, 12)

    save_figure(fig, "kl_vs_beta_all_scenarios.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 3: Full collapse-barrier figure (6-panel)
# ═══════════════════════════════════════════════════════════════════

def figure_collapse_barrier_full():
    fb_full = load_json("collapse_barrier_full_results.json")
    if not fb_full:
        print("  SKIP: no collapse_barrier_full_results.json")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel 1: KL vs FB for each beta (DM=1)
    ax = axes[0, 0]
    betas = sorted(set(r["beta"] for r in fb_full if r["decoder_mult"] == 1))
    colors_beta = plt.cm.viridis(np.linspace(0.1, 0.9, len(betas)))

    for beta, c in zip(betas, colors_beta):
        items = sorted([r for r in fb_full if r["decoder_mult"] == 1 and abs(r["beta"] - beta) < 0.001],
                       key=lambda x: x["free_bits"])
        ax.plot([r["free_bits"] for r in items], [r["kl"] for r in items],
                "o-", color=c, linewidth=1.5, markersize=6, label=f"beta={beta:.1f}")

    ax.axhline(y=0.1, color="red", linestyle=":", alpha=0.5)
    ax.set_xlabel("Free Bits lambda (nats/dim)", fontsize=10)
    ax.set_ylabel("KL Divergence (nats)", fontsize=10)
    ax.set_title("KL vs Free Bits (DM=1)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: MSE vs FB
    ax = axes[0, 1]
    for beta, c in zip(betas, colors_beta):
        items = sorted([r for r in fb_full if r["decoder_mult"] == 1 and abs(r["beta"] - beta) < 0.001],
                       key=lambda x: x["free_bits"])
        ax.plot([r["free_bits"] for r in items], [r["mse"] for r in items],
                "o-", color=c, linewidth=1.5, markersize=6, label=f"beta={beta:.1f}")

    ax.set_xlabel("Free Bits lambda (nats/dim)", fontsize=10)
    ax.set_ylabel("MSE", fontsize=10)
    ax.set_title("MSE vs Free Bits (DM=1)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3: DM comparison
    ax = axes[0, 2]
    from matplotlib.lines import Line2D
    key_betas = [0.1, 1.0, 2.0, 4.0, 10.0]
    for beta in key_betas:
        for dm in [1, 2]:
            items = sorted([r for r in fb_full if r["decoder_mult"] == dm and abs(r["beta"] - beta) < 0.001],
                           key=lambda x: x["free_bits"])
            ax.plot([r["free_bits"] for r in items], [r["kl"] for r in items],
                    marker="o" if dm == 1 else "s", color="#e74c3c" if dm == 1 else "#3498db",
                    linestyle="-" if dm == 1 else "--", linewidth=1.5, markersize=6)

    legend_elements = [
        Line2D([0], [0], color="#e74c3c", lw=1.5, label="DM=1"),
        Line2D([0], [0], color="#3498db", lw=1.5, linestyle="--", label="DM=2"),
    ]
    ax.legend(handles=legend_elements, fontsize=7)
    ax.axhline(y=0.1, color="red", linestyle=":", alpha=0.5)
    ax.set_xlabel("Free Bits lambda (nats/dim)", fontsize=10)
    ax.set_ylabel("KL Divergence (nats)", fontsize=10)
    ax.set_title("Decoder Expansion Effect on KL", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel 4: KL heatmap
    ax = axes[1, 0]
    dm1 = sorted([r for r in fb_full if r["decoder_mult"] == 1],
                 key=lambda x: (x["free_bits"], x["beta"]))
    fb_vals = sorted(set(r["free_bits"] for r in dm1))
    beta_vals = sorted(set(r["beta"] for r in dm1))

    kl_matrix = np.zeros((len(fb_vals), len(beta_vals)))
    for i, fb in enumerate(fb_vals):
        for j, beta in enumerate(beta_vals):
            match = [r for r in dm1 if r["free_bits"] == fb and abs(r["beta"] - beta) < 0.001]
            kl_matrix[i, j] = match[0]["kl"] if match else np.nan

    im = ax.pcolormesh(beta_vals, fb_vals, np.log10(np.maximum(kl_matrix, 1e-5)),
                        cmap="RdYlGn", shading="auto")
    ax.contour(beta_vals, fb_vals, kl_matrix, levels=[0.1, 1.0, 5.0],
               colors=["white", "black", "black"], linewidths=[2, 1, 1],
               linestyles=["-", "--", "--"])
    plt.colorbar(im, ax=ax).set_label("log10(KL nats)", fontsize=9)
    ax.set_xlabel("beta (KL weight)", fontsize=10)
    ax.set_ylabel("Free Bits lambda (nats/dim)", fontsize=10)
    ax.set_title("KL Heatmap: DM=1 (collapse frontier)", fontsize=11, fontweight="bold")

    # Panel 5: Phase diagram
    ax = axes[1, 1]
    beta_list = sorted(set(r["beta"] for r in fb_full if r["decoder_mult"] == 1))
    min_fb_ok = []
    for beta in beta_list:
        dm1_items = sorted([r for r in fb_full if r["decoder_mult"] == 1 and abs(r["beta"] - beta) < 0.001],
                           key=lambda x: x["free_bits"])
        min_fb = next((r["free_bits"] for r in dm1_items if r["kl"] >= 0.1), 2.0)
        min_fb_ok.append(min_fb)

    ax.fill_between(beta_list, 0, min_fb_ok, alpha=0.1, color="red")
    ax.step(beta_list, min_fb_ok, "r-o", where="post", linewidth=2, markersize=6,
            label="Min FB for KL >= 0.1")
    ax.set_xlabel("beta (KL weight)", fontsize=10)
    ax.set_ylabel("Minimum Free Bits lambda (nats/dim)", fontsize=10)
    ax.set_title("Anti-Collapse Phase Diagram (DM=1)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.05, 2.5)

    # Panel 6: Rate-Distortion frontier
    ax = axes[1, 2]
    for fb in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        items = sorted([r for r in fb_full if r["decoder_mult"] == 1 and r["free_bits"] == fb],
                       key=lambda x: x["beta"])
        if not items:
            continue
        ax.plot([r["kl"] for r in items], [r["mse"] for r in items],
                "o-", linewidth=1.8, markersize=7, label=f"FB={fb:.2f}")

    baseline = sorted([r for r in fb_full if r["decoder_mult"] == 1 and r["free_bits"] == 0.0],
                      key=lambda x: x["beta"])
    if baseline:
        ax.plot([r["kl"] for r in baseline], [r["mse"] for r in baseline],
                "ko-", linewidth=2.5, markersize=9, zorder=10, label="Baseline (FB=0)")

    ax.set_xlabel("KL Divergence (nats)", fontsize=10)
    ax.set_ylabel("MSE", fontsize=10)
    ax.set_title("Rate-Distortion: Free Bits Frontier", fontsize=11, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.suptitle("Step C: Collapse Barrier — Free Bits + Decoder Expansion on spread_xhd (90-dim)",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    save_figure(fig, "stepC_collapse_barrier_full.png")


# ═══════════════════════════════════════════════════════════════════
# Figure 4: Cross-scenario FB validation
# ═══════════════════════════════════════════════════════════════════

def figure_cross_scenario():
    cross = load_json("fb_cross_scenario_validation.json")
    fb_full = load_json("collapse_barrier_full_results.json")
    stepb = load_json("vae_stepB_results.json")
    if not cross:
        print("  SKIP: no cross-scenario results")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    scenarios = ["tag_hd", "comm_hd", "spread_xhd"]

    # Panel 1: KL vs beta
    ax = axes[0]
    for scenario in scenarios:
        bl_items = sorted([r for r in (stepb or []) if r["scenario"] == scenario], key=lambda x: x["beta"])
        if bl_items:
            ax.loglog([r["beta"] for r in bl_items], [max(r["kl"], 1e-5) for r in bl_items],
                      marker=scenario_marker(scenario), color=scenario_color(scenario),
                      linestyle="--", linewidth=1.5, markersize=7, alpha=0.5,
                      label=f"{scenario_label(scenario)} (FB=0)")

        if scenario == "spread_xhd":
            fb01_items = sorted([r for r in (fb_full or []) if r.get("decoder_mult") == 1 and r.get("free_bits") == 0.1],
                                key=lambda x: x["beta"])
        else:
            fb01_items = sorted([r for r in cross if r["scenario"] == scenario and r["free_bits"] == 0.1],
                                key=lambda x: x["beta"])
        if fb01_items:
            ax.loglog([r["beta"] for r in fb01_items], [max(r["kl"], 1e-5) for r in fb01_items],
                      marker=scenario_marker(scenario), color=scenario_color(scenario),
                      linestyle="-", linewidth=2.2, markersize=9,
                      label=f"{scenario_label(scenario)} (FB=0.1)")

    collapse_threshold_line(ax)
    target_zone(ax)
    ax.set_xlabel("beta (KL weight)", fontsize=11)
    ax.set_ylabel("KL Divergence (nats)", fontsize=11)
    ax.set_title("FB=0.1 Universal Anti-Collapse", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.3, 12)

    # Panel 2: Collapse rate comparison
    ax2 = axes[1]
    scenario_labels_list = ["tag_hd\n(40-dim)", "comm_hd\n(60-dim)", "spread_xhd\n(90-dim)"]
    collapse_fb0 = []
    collapse_fb01 = []

    for scenario in scenarios:
        if scenario == "spread_xhd":
            fb0 = [r for r in (fb_full or []) if r.get("decoder_mult") == 1 and r.get("free_bits") == 0.0]
            fb01 = [r for r in (fb_full or []) if r.get("decoder_mult") == 1 and r.get("free_bits") == 0.1]
        else:
            fb0 = [r for r in cross if r["scenario"] == scenario and r["free_bits"] == 0.0]
            fb01 = [r for r in cross if r["scenario"] == scenario and r["free_bits"] == 0.1]
        collapse_fb0.append(sum(1 for r in fb0 if r["kl"] < 0.1) / max(len(fb0), 1) * 100)
        collapse_fb01.append(sum(1 for r in fb01 if r["kl"] < 0.1) / max(len(fb01), 1) * 100)

    x = np.arange(len(scenario_labels_list))
    width = 0.35
    bars1 = ax2.bar(x - width/2, collapse_fb0, width, color="#e74c3c", edgecolor="white", label="FB=0.0")
    bars2 = ax2.bar(x + width/2, collapse_fb01, width, color="#27ae60", edgecolor="white", label="FB=0.1")
    for bar, val in zip(bars1, collapse_fb0):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}%",
                 ha="center", fontsize=10, fontweight="bold", color="#e74c3c")
    for bar, val in zip(bars2, collapse_fb01):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}%",
                 ha="center", fontsize=10, fontweight="bold", color="#27ae60")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_labels_list)
    ax2.set_ylabel("Collapse Rate (KL < 0.1)", fontsize=11)
    ax2.set_title("Collapse Eliminated Across All Scenarios", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 120)
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: MSE improvement at beta=2.0
    ax3 = axes[2]
    beta_focus = 2.0
    for i, scenario in enumerate(scenarios):
        if scenario == "spread_xhd":
            fb0_r = next((r for r in (fb_full or []) if r.get("decoder_mult") == 1 and r.get("free_bits") == 0.0 and abs(r.get("beta", 0) - beta_focus) < 0.001), None)
            fb01_r = next((r for r in (fb_full or []) if r.get("decoder_mult") == 1 and r.get("free_bits") == 0.1 and abs(r.get("beta", 0) - beta_focus) < 0.001), None)
        else:
            fb0_r = next((r for r in cross if r["scenario"] == scenario and r["free_bits"] == 0.0 and abs(r["beta"] - beta_focus) < 0.001), None)
            fb01_r = next((r for r in cross if r["scenario"] == scenario and r["free_bits"] == 0.1 and abs(r["beta"] - beta_focus) < 0.001), None)
        if fb0_r and fb01_r:
            mse_improve = (fb0_r["mse"] - fb01_r["mse"]) / fb0_r["mse"] * 100
            ax3.bar(i - 0.2, fb0_r["mse"], 0.35, color="#e74c3c", edgecolor="white", label="FB=0.0" if i == 0 else "")
            ax3.bar(i + 0.2, fb01_r["mse"], 0.35, color="#27ae60", edgecolor="white", label="FB=0.1" if i == 0 else "")
            ax3.text(i + 0.2, fb01_r["mse"] + 0.02, f"-{mse_improve:.1f}%",
                     ha="center", fontsize=9, fontweight="bold", color="#27ae60")
    ax3.set_xticks(range(len(scenarios)))
    ax3.set_xticklabels(scenario_labels_list)
    ax3.set_ylabel("MSE", fontsize=11)
    ax3.set_title(f"MSE Improvement at beta={beta_focus}", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.suptitle("FB=0.1 Cross-Scenario Validation: Universal Posterior Collapse Prevention",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_figure(fig, "fb_cross_scenario_validation.png")


# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating figures...\n")

    print("1/4 Rate-Distortion curves...")
    figure_rate_distortion()

    print("\n2/4 KL vs Beta (all scenarios)...")
    figure_kl_vs_beta()

    print("\n3/4 Collapse Barrier Full (6-panel)...")
    figure_collapse_barrier_full()

    print("\n4/4 Cross-Scenario FB Validation (3-panel)...")
    figure_cross_scenario()

    print("\nAll figures generated.")
