"""Check data, checkpoint, and results integrity.

Usage: python scripts/0_check_integrity.py
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obscodec.config import DATA_DIR, ASSETS_DIR, CHECKPOINT_DIR


def check_data():
    """Verify all .npy data files exist and have correct shapes."""
    print("=== Data Files ===")
    expected = {
        "simple_spread_obs.npy": (33333, 30),
        "simple_tag_obs.npy": (33333, 24),
        "simple_world_comm_obs.npy": (33333, 36),
        "tag_hd_obs.npy": (33333, 40),
        "spread_hd_obs.npy": (33333, 48),
        "comm_hd_obs.npy": (33333, 60),
        "spread_xhd_obs.npy": (33333, 90),
        "spread_N3_obs.npy": (33333, 18),
        "spread_N5_obs.npy": (33333, 30),
        "spread_N7_obs.npy": (33333, 42),
        "spread_N10_obs.npy": (33333, 60),
        "spread_N12_obs.npy": (33333, 72),
        "spread_N15_obs.npy": (33333, 90),
    }
    all_ok = True
    for fname, expected_shape in expected.items():
        path = DATA_DIR / fname
        if path.exists():
            data = np.load(str(path))
            shape_ok = len(data.shape) == 2 and data.shape[0] == expected_shape[0] and data.shape[1] == expected_shape[1]
            status = "OK" if shape_ok else f"BAD SHAPE: {data.shape} expected {expected_shape}"
            if not shape_ok:
                all_ok = False
        else:
            status = "MISSING"
            all_ok = False
        print(f"  {fname:<30} {status}")
    return all_ok


def check_results():
    """Verify all results JSON files are valid."""
    print("\n=== Results JSONs ===")
    expected = [
        "pca_results.json", "ae_results.json", "digital_results.json",
        "vae_results.json", "vae_stepB_results.json",
        "vqvae_stepA_results.json", "vae_stepA_results.json",
        "collapse_barrier_results.json", "collapse_barrier_full_results.json",
        "fb_cross_scenario_validation.json", "pilot_channel_results.json",
        "fb_finesweep_results.json", "agent_scaling_results.json",
        "vqvae_multiscenario_results.json", "unified_codec_results.json",
    ]
    all_ok = True
    for fname in expected:
        path = ASSETS_DIR / fname
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            status = f"OK ({len(data)} entries)" if isinstance(data, list) else "OK"
        else:
            status = "MISSING"
            all_ok = False
        print(f"  {fname:<40} {status}")
    return all_ok


def check_checkpoints():
    """Verify checkpoints directory exists."""
    print("\n=== Checkpoints ===")
    if CHECKPOINT_DIR.exists():
        pts = list(CHECKPOINT_DIR.glob("*.pt"))
        print(f"  {len(pts)} .pt files in {CHECKPOINT_DIR}")
        return True
    else:
        print(f"  DIRECTORY MISSING: {CHECKPOINT_DIR}")
        return False


def check_figures():
    """Verify figure PNGs exist."""
    print("\n=== Figures ===")
    expected = [
        "rate_distortion_simple_spread.png",
        "rate_distortion_simple_tag.png",
        "rate_distortion_simple_world_comm.png",
        "rate_distortion_unified.png",
        "kl_beta_simple_spread.png",
        "stepB_kl_vs_beta.png",
        "stepC_collapse_barrier.png",
        "stepC_collapse_barrier_full.png",
        "fb_cross_scenario_validation.png",
    ]
    for fname in expected:
        path = ASSETS_DIR / fname
        status = "OK" if path.exists() else "MISSING"
        print(f"  {fname:<45} {status}")


if __name__ == "__main__":
    print("ObsCodec Integrity Check\n")
    d = check_data()
    r = check_results()
    c = check_checkpoints()
    check_figures()

    print(f"\n{'='*50}")
    if d and r and c:
        print("OVERALL: PASS")
    else:
        print("OVERALL: ISSUES FOUND")
        if not d:
            print("  - Data files missing/malformed")
        if not r:
            print("  - Results JSONs missing")
        if not c:
            print("  - Checkpoints directory missing")
