"""ObsCodec integrity check."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
errors = []

expected = [
    "README.md", "README_zh.md", "TECHNICAL_REPORT.md", "LICENSE",
    ".gitignore", "requirements.txt", "setup.py",
    "obscodec/__init__.py", "obscodec/config.py", "obscodec/metrics.py",
    "obscodec/trainer.py", "obscodec/visualize.py",
    "obscodec/models/__init__.py",
    "obscodec/models/pca_baseline.py", "obscodec/models/ae_baseline.py",
    "obscodec/models/digital_baseline.py", "obscodec/models/vae.py",
    "obscodec/models/vqvae.py",
    "scripts/1_collect_data.py", "scripts/2_train_baselines.py",
    "scripts/3_train_vae.py", "scripts/4_train_vqvae.py",
    "scripts/5_generate_figures.py", "scripts/6_summary_table.py",
]
for f in expected:
    if not os.path.exists(os.path.join(root, f)):
        errors.append(f"MISSING: {f}")

assets = [
    "ablation_heatmap.png", "kl_collapse.png", "latent_space.png",
    "pareto_frontier.png", "rate_distortion.png", "reconstruction_comparison.png",
    "vqvae_commitment.png", "vqvae_usage_heatmap.png",
    "baseline_results.json", "vae_results.json", "vqvae_results.json",
    "results_summary.md",
]
for a in assets:
    if not os.path.exists(os.path.join(root, "assets", a)):
        errors.append(f"MISSING: assets/{a}")

imports = [
    "obscodec.config", "obscodec.trainer", "obscodec.visualize",
    "obscodec.models.pca_baseline", "obscodec.models.ae_baseline",
    "obscodec.models.digital_baseline", "obscodec.models.vae",
    "obscodec.models.vqvae",
]
for mod in imports:
    try:
        __import__(mod)
    except Exception as e:
        errors.append(f"IMPORT FAIL: {mod}: {e}")

try:
    from obscodec.models import PCABaseline, StandardAE, BetaVAE, VQVAE, DigitalQuantizationBaseline
except ImportError as e:
    errors.append(f"REGISTRY FAIL: {e}")

print("=" * 60)
if errors:
    print(f"\nFAIL: {len(errors)} issue(s):")
    for e in errors: print(f"  {e}")
    sys.exit(1)
else:
    print("All checks passed")
    print(f"   {len(expected)} source files present")
    print(f"   {len(assets)} assets present")
    print("   All imports successful - ObsCodec v1.0 ready")
