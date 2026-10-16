"""
CLI script to generate synthetic dataset fixtures for pipeline testing.

Generates:
- Close-up leaf images with disease lesions across multiple farms and capture sessions.
- Aerial canopy orthophoto with stress regions and annotations.
- Full metadata.csv and dataset NOTICE.txt.
"""

import argparse
import sys
from pathlib import Path

# Safeguard to allow running directly from repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agriscan.data.synthetic import generate_synthetic_dataset, SYNTHETIC_DISCLAIMER


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset fixtures for AgriScan testing and smoke tests."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Base directory for synthetic data output (default: 'data')",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=8,
        help="Number of synthetic samples per disease class (default: 8)",
    )
    parser.add_argument(
        "--num-farms",
        type=int,
        default=3,
        help="Number of distinct farms to generate for leakage prevention testing (default: 3)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("AGRISCAN SYNTHETIC DATASET GENERATOR")
    print("=" * 80)
    print(SYNTHETIC_DISCLAIMER)
    print("=" * 80 + "\n")

    summary = generate_synthetic_dataset(
        base_dir=args.output_dir,
        num_samples_per_class=args.samples_per_class,
        num_farms=args.num_farms,
    )

    print(f"Generated {summary['num_records']} close-up leaf images in: {summary['close_up_dir']}")
    print(f"Classes: {', '.join(summary['classes'])}")
    print(f"Simulated Farms: {', '.join(summary['farms'])}")
    print(f"Generated aerial fixture in: {summary['aerial_dir']}")
    print("Synthetic fixture preparation completed successfully.\n")


if __name__ == "__main__":
    main()
