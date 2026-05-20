"""Generate synthetic multi-agent observation data for all 7 scenarios.

Generates: simple_spread (30-dim), simple_tag (24-dim), simple_world_comm (36-dim),
           tag_hd (40-dim), spread_hd (48-dim), comm_hd (60-dim), spread_xhd (90-dim)

Usage: python scripts/1_collect_data.py [--hd] [--xhd]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obscodec.data.synthetic import (collect_synthetic_dataset,
                                       collect_synthetic_dataset_hd,
                                       SYNTHETIC_GENERATORS,
                                       SYNTHETIC_GENERATORS_HD)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic observation data")
    parser.add_argument("--hd", action="store_true", help="Also generate high-dim scenarios")
    parser.add_argument("--xhd", action="store_true", help="Also generate extreme-dim spread")
    parser.add_argument("--all", action="store_true", default=True, help="Generate all 7 scenarios (default)")
    args = parser.parse_args()

    # Standard scenarios (30/24/36 dim)
    print("=" * 60)
    print("Generating standard scenarios (low-dim)")
    print("=" * 60)
    datasets = collect_synthetic_dataset(save=True)
    for name, data in datasets.items():
        print(f"  {name}: {data.shape}")

    # High-dimensional scenarios (40/48/60/90 dim)
    if args.hd or args.xhd or args.all:
        print(f"\n{'=' * 60}")
        print("Generating high-dimensional scenarios")
        print("=" * 60)
        hd_datasets = collect_synthetic_dataset_hd(save=True)
        for name, data in hd_datasets.items():
            print(f"  {name}: {data.shape}")

    print(f"\n=== Data Collection Summary ===")
    print(f"Base scenarios: {list(SYNTHETIC_GENERATORS.keys())}")
    print(f"HD scenarios:   {list(SYNTHETIC_GENERATORS_HD.keys())}")
    print(f"Total: {len(SYNTHETIC_GENERATORS) + len(SYNTHETIC_GENERATORS_HD)} scenarios")
