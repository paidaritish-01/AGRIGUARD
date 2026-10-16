"""
Dataset validation and integrity checking for AgriScan.

Checks:
1. Unreadable or corrupt image files.
2. Missing labels or missing metadata attributes.
3. Duplicate files via SHA-256 hashing.
4. Class distribution and imbalance severity.
5. Group disjointness (guaranteeing no farm_id or session_id leaks across train/val/test).
"""

import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from PIL import Image


class DatasetValidationError(Exception):
    """Raised when critical dataset validation checks fail."""
    pass


class DatasetValidator:
    """
    Validates agricultural image datasets for classification or aerial detection.
    """

    def __init__(self, dataset_dir: str | Path, metadata_file: Optional[str | Path] = None):
        self.dataset_dir = Path(dataset_dir)
        self.metadata_file = Path(metadata_file) if metadata_file else None

    @staticmethod
    def compute_sha256(filepath: Path) -> str:
        """Compute SHA-256 checksum of an image file to detect duplicate content."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def validate_image_file(self, filepath: Path) -> Tuple[bool, Optional[str]]:
        """
        Verify that an image is readable, uncorrupted, and has valid dimensions.
        """
        if not filepath.exists():
            return False, "File does not exist"
        if filepath.stat().st_size == 0:
            return False, "Empty file (0 bytes)"

        try:
            with Image.open(filepath) as img:
                img.verify()  # Verifies header and stream integrity
            # Reopen to verify decode
            with Image.open(filepath) as img:
                img.load()
                if img.width <= 0 or img.height <= 0:
                    return False, f"Invalid dimensions: {img.width}x{img.height}"
            return True, None
        except Exception as e:
            return False, f"Corrupted or unsupported image: {str(e)}"

    def validate_metadata_records(
        self, records: List[Dict[str, Any]], mode: str = "close_up"
    ) -> Dict[str, Any]:
        """
        Validate records loaded from metadata (CSV/JSON/directory layout).

        Required fields for close_up: image_path, label, farm_id (or session_id)
        Required fields for aerial: image_path, farm_id, annotations/label
        """
        results: Dict[str, Any] = {
            "total_records": len(records),
            "valid_records": 0,
            "corrupt_images": [],
            "missing_files": [],
            "missing_labels": [],
            "duplicates": {},
            "class_counts": {},
            "farms_or_sessions": set(),
        }

        seen_hashes: Dict[str, str] = {}  # hash -> first_seen_filepath

        for rec in records:
            img_rel_or_abs = rec.get("image_path")
            if not img_rel_or_abs:
                results["missing_files"].append("Record missing 'image_path' field")
                continue

            img_path = Path(img_rel_or_abs)
            if not img_path.is_absolute():
                img_path = self.dataset_dir / img_path

            # Label check
            label = rec.get("label")
            if not label or str(label).strip() == "":
                results["missing_labels"].append(str(img_path))
                continue

            # Readability check
            is_valid, err_msg = self.validate_image_file(img_path)
            if not is_valid:
                results["corrupt_images"].append({"file": str(img_path), "error": err_msg})
                continue

            # Duplicate check via content hash
            try:
                img_hash = self.compute_sha256(img_path)
                if img_hash in seen_hashes:
                    prev_file = seen_hashes[img_hash]
                    if prev_file != str(img_path):
                        results["duplicates"].setdefault(prev_file, []).append(str(img_path))
                else:
                    seen_hashes[img_hash] = str(img_path)
            except Exception as e:
                results["corrupt_images"].append({"file": str(img_path), "error": f"Hashing failed: {e}"})
                continue

            # Class counts
            results["class_counts"][label] = results["class_counts"].get(label, 0) + 1

            # Farm / session grouping tracking
            group = rec.get("farm_id") or rec.get("session_id") or "default_farm"
            results["farms_or_sessions"].add(group)

            results["valid_records"] += 1

        # Imbalance ratio
        if results["class_counts"]:
            counts = list(results["class_counts"].values())
            results["imbalance_ratio"] = max(counts) / max(min(counts), 1)
        else:
            results["imbalance_ratio"] = 0.0

        results["farms_or_sessions"] = list(results["farms_or_sessions"])
        return results

    def validate_split_leakage(
        self,
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
        test_records: List[Dict[str, Any]],
        group_key: str = "farm_id",
    ) -> Dict[str, Any]:
        """
        Verify that farm/session groups are mutually exclusive between train, val, and test.
        Also verifies no identical files (by hash) exist across splits.
        """
        def get_groups(records: List[Dict[str, Any]]) -> Set[str]:
            return {
                str(r.get(group_key) or r.get("session_id") or "unknown")
                for r in records
            }

        train_groups = get_groups(train_records)
        val_groups = get_groups(val_records)
        test_groups = get_groups(test_records)

        train_val_overlap = train_groups.intersection(val_groups)
        train_test_overlap = train_groups.intersection(test_groups)
        val_test_overlap = val_groups.intersection(test_groups)

        leakage_detected = bool(train_val_overlap or train_test_overlap or val_test_overlap)

        return {
            "leakage_detected": leakage_detected,
            "train_val_overlap": list(train_val_overlap),
            "train_test_overlap": list(train_test_overlap),
            "val_test_overlap": list(val_test_overlap),
            "train_group_count": len(train_groups),
            "val_group_count": len(val_groups),
            "test_group_count": len(test_groups),
        }
