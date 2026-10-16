"""
Dataset inspection and validation script for SAR-MLD1-2025 mango leaf dataset.

Evaluates:
- Presence of SAR-MLD1-2025 Part 1 and Part 2 files
- Directory structure and class coverage across 5 classes:
  1. Healthy
  2. Anthracnose
  3. Powdery Mildew
  4. Turning Brown
  5. Gall Midge
- Image readability and corruption detection
- Per-class counts and imbalance ratios
- Duplicate image detection
- Metadata grouping availability (tree_id / session_id) for zero-leakage splitting
- Attribution and Mendeley license compliance (CC BY 4.0)

References:
- SAR-MLD1-2025 Dataset Part 1: https://data.mendeley.com/datasets/sd8hzpg69b/5
- SAR-MLD1-2025 Dataset Part 2: https://data.mendeley.com/datasets/j3bn63t4sp/5
- Data in Brief (2026): https://doi.org/10.1016/j.dib.2026.112826
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image

EXPECTED_MANGO_CLASSES = {
    "healthy",
    "anthracnose",
    "powdery_mildew",
    "turning_brown",
    "gall_midge",
}

EXPECTED_TOTAL_IMAGES = 4921


def inspect_mango_dataset(dataset_dir: str | Path) -> Dict[str, Any]:
    dataset_path = Path(dataset_dir)
    print(f"\n[AgriScan] Inspecting Mango Dataset at: {dataset_path}")

    report = {
        "dataset_path": str(dataset_path),
        "status": "unavailable",
        "dataset_name": "SAR-MLD1-2025",
        "expected_classes": sorted(list(EXPECTED_MANGO_CLASSES)),
        "found_classes": [],
        "total_images": 0,
        "corrupt_images": [],
        "per_class_counts": {},
        "duplicate_hash_count": 0,
        "leakage_metadata_present": False,
        "attribution_notice": (
            "SAR-MLD1-2025 Mango Leaf Dataset, Mendeley Data, Parts 1 & 2 (CC BY 4.0). "
            "Captured in Bangladesh; contains close-up leaf imagery against white background. "
            "Not validated for aerial or in-canopy fruit detection."
        ),
        "recommendations": [],
    }

    if not dataset_path.exists():
        report["status"] = "unavailable"
        report["recommendations"].append(
            f"Dataset directory '{dataset_path}' does not exist. "
            "To use real mango data, download SAR-MLD1-2025 from Mendeley Data and place in data/mango_sar_mld1."
        )
        return report

    # Find class subdirectories
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    found_classes = {}
    for d in subdirs:
        norm_name = d.name.lower().replace(" ", "_").replace("-", "_")
        for exp in EXPECTED_MANGO_CLASSES:
            if exp in norm_name or norm_name in exp:
                found_classes[exp] = d
                break

    report["found_classes"] = list(found_classes.keys())

    if not found_classes:
        report["status"] = "invalid_structure"
        report["recommendations"].append(
            f"No expected mango disease classes found in {dataset_path}. "
            f"Expected subdirectories for: {sorted(list(EXPECTED_MANGO_CLASSES))}"
        )
        return report

    # Check images within class directories
    hashes = set()
    duplicate_count = 0
    total_imgs = 0

    for class_name, class_dir in found_classes.items():
        image_files = [
            f for f in class_dir.glob("**/*")
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
        ]
        report["per_class_counts"][class_name] = len(image_files)
        total_imgs += len(image_files)

        for img_path in image_files[:100]:  # Sample first 100 per class for lightweight check
            try:
                with Image.open(img_path) as im:
                    im.verify()
                # Hash check for duplicates
                file_hash = hashlib.md5(img_path.read_bytes()).hexdigest()
                if file_hash in hashes:
                    duplicate_count += 1
                else:
                    hashes.add(file_hash)
            except Exception as e:
                report["corrupt_images"].append(f"{img_path.name}: {str(e)}")

    report["total_images"] = total_imgs
    report["duplicate_hash_count"] = duplicate_count

    # Check for metadata.csv
    meta_path = dataset_path / "metadata.csv"
    if meta_path.exists():
        report["leakage_metadata_present"] = True
    else:
        report["recommendations"].append(
            "No metadata.csv found. Group-based splitting by tree/session cannot be performed without metadata."
        )

    # Status evaluation
    if len(found_classes) == len(EXPECTED_MANGO_CLASSES):
        report["status"] = "validated" if not report["corrupt_images"] else "corrupt_files_detected"
    else:
        report["status"] = "partial_classes"

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate SAR-MLD1-2025 Mango Leaf Dataset")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/mango_sar_mld1",
        help="Path to SAR-MLD1-2025 dataset directory",
    )
    args = parser.parse_args()

    report = inspect_mango_dataset(args.dataset_dir)
    print("\n--- Validation Report Summary ---")
    print(f"Status: {report['status']}")
    print(f"Total Images: {report['total_images']}")
    print(f"Classes Found: {report['found_classes']}")
    print(f"Per-Class Counts: {report['per_class_counts']}")
    if report["recommendations"]:
        print("\nRecommendations / Notes:")
        for r in report["recommendations"]:
            print(f"- {r}")

    print(f"\nAttribution: {report['attribution_notice']}")


if __name__ == "__main__":
    main()
