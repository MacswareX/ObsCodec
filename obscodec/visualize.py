"""Figure generation utilities — consistent styling, palettes, and save helpers."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ASSETS_DIR

# ── Style ──
SCENARIO_COLORS = {
    "simple_spread": "#e74c3c",
    "simple_tag": "#e67e22",
    "simple_world_comm": "#f1c40f",
    "tag_hd": "#e74c3c",
    "spread_hd": "#3498db",
    "comm_hd": "#2ecc71",
    "spread_xhd": "#9b59b6",
    "unified": "#95a5a6",
}

SCENARIO_MARKERS = {
    "simple_spread": "o",
    "simple_tag": "s",
    "simple_world_comm": "^",
    "tag_hd": "s",
    "spread_hd": "o",
    "comm_hd": "^",
    "spread_xhd": "D",
    "unified": "p",
}

SCENARIO_LABELS = {
    "simple_spread": "simple_spread (30-dim)",
    "simple_tag": "simple_tag (24-dim)",
    "simple_world_comm": "simple_world_comm (36-dim)",
    "tag_hd": "tag_hd (40-dim)",
    "spread_hd": "spread_hd (48-dim)",
    "comm_hd": "comm_hd (60-dim)",
    "spread_xhd": "spread_xhd (90-dim)",
    "unified": "unified (padded)",
}

CODEC_COLORS = {
    "PCA": "#e74c3c",
    "AE": "#3498db",
    "Digital": "#2ecc71",
    "VAE": "#9b59b6",
    "VQ-VAE": "#f39c12",
}


def set_style():
    """Apply consistent matplotlib style."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 7.5,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })


def save_figure(fig, name):
    """Save figure to ASSETS_DIR with consistent settings."""
    path = ASSETS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {path}")


def collapse_threshold_line(ax):
    """Add KL=0.1 collapse threshold line."""
    ax.axhline(y=0.1, color="red", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(0.3, 0.13, "KL=0.1 (collapse threshold)", fontsize=7, color="red", alpha=0.7)


def target_zone(ax):
    """Add beta=2-4 target zone shading."""
    ax.axvspan(2.0, 4.0, alpha=0.08, color="green")
    ax.text(3.0, 8e-5, "Target\nbeta=2-4", fontsize=8, color="green", ha="center", fontweight="bold")


def scenario_color(name):
    """Get consistent color for a scenario."""
    return SCENARIO_COLORS.get(name, "#333333")


def scenario_marker(name):
    """Get consistent marker for a scenario."""
    return SCENARIO_MARKERS.get(name, "o")


def scenario_label(name):
    """Get human-readable label for a scenario."""
    return SCENARIO_LABELS.get(name, name)
