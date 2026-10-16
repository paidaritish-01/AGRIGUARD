"""
CLI script to validate an agricultural dataset for AgriScan.

Performs:
- Corrupt/unreadable image identification.
- Content hash (SHA-256) duplicate file detection.
- Class balance calculation and imbalance ratio reporting.
- Group-split leakage verification across farms/sessions.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agriscan.data.dataset import load_dataset_records, split_records_by_group
from agriscan.data.validator import DatasetValidator


def main():
    parser = argparse.ArgumentParser(description="Validate agricultural dataset integrity.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/close_up",
        help="Path to dataset root folder",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default=None,
        help="Optional path to metadata CSV file",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default="farm_id",
        help="Column used for group-based train/val/test splitting to prevent data leakage",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print("\n" + "=" * 80)
    print(f"VALIDATING DATASET: {data_dir.resolve()}")
    print("=" * 80)

    if not data_dir.exists():
        print(f"Error: Dataset directory does not exist: {data_dir}")
        sys.exit(1)

    validator = DatasetValidator(dataset_dir=data_dir, metadata_file=args.metadata_file)
    records = load_dataset_records(data_dir, metadata_file=args.metadata_file)

    if not records:
        print("Error: No image records found in dataset directory or metadata file.")
        sys.exit(1)

    print(f"Loaded {len(records)} total records. Scanning files...")
    results = validator.validate_metadata_records(records)

    print("\n--- VALIDATION RESULTS ---")
    print(f"Valid Records:   {results['valid_records']} / {results['total_records']}")
    print(f"Corrupt Images:  {len(results['corrupt_images'])}")
    if results["corrupt_images"]:
        for c in results["corrupt_images"][:5]:
            print(f"   - {c['file']}: {c['error']}")
    print(f"Missing Labels:  {len(results['missing_labels'])}")
    print(f"Duplicate Files: {len(results['duplicates'])}")
    if results["duplicates"]:
        for orig, dupes in list(results["duplicates"].items())[:3]:
            print(f"   - Original: {orig} -> Duplicate: {dupes[0]}")

    print("\n--- CLASS DISTRIBUTION ---")
    for cls_name, count in sorted(results["class_counts"].items()):
        pct = (count / max(results["valid_records"], 1)) * 100.0
        print(f"  {cls_name:<25}: {count:>4} samples ({pct:>5.1f}%)")
    print(f"Imbalance Ratio (Max/Min): {results['imbalance_ratio']:.2f}")

    print("\n--- GROUP DISJOINTNESS (LEAKAGE CHECK) ---")
    print(f"Farms/Sessions Identified: {', '.join(results['farms_or_sessions'])}")

    train_recs, val_recs, test_recs = split_records_by_group(
        records, group_key=args.group_col, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    leakage_check = validator.validate_split_leakage(
        train_recs, val_recs, test_recs, group_key=args.group_col
    )
    print(f"Train Records: {len(train_recs)} | Val Records: {len(val_recs)} | Test Records: {len(test_recs)}")
    print(f"Data Leakage Detected: {leakage_check['leakage_detected']}")
    if leakage_check["leakage_detected"]:
        print(f"WARNING: Group overlaps found:")
        print(f"  Train-Val:  {leakage_check['train_val_overlap']}")
        print(f"  Train-Test: {leakage_check['train_test_overlap']}")
        print(f"  Val-Test:   {leakage_check['val_test_overlap']}")
    else:
        print("Zero group data leakage verified. Farm/session splits are strictly disjoint.")

    print("\nValidation completed successfully.\n")


if __name__ == "__main__":
    main()
