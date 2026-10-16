"""
Synthetic dataset generator for AgriScan.

Provides deterministic test fixtures for pipeline smoke tests and CI/CD validation.
IMPORTANT: Synthetic fixtures verify pipeline mechanics, data schemas, and unit tests.
They CANNOT validate real-world biological accuracy or agricultural efficacy.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw
import numpy as np


SYNTHETIC_DISCLAIMER = (
    "DISCLAIMER: This is a synthetic dataset fixture created solely for software "
    "pipeline verification, test suites, and smoke tests. It CANNOT validate real-world "
    "agricultural performance, disease discrimination, or diagnostic efficacy."
)


def create_synthetic_leaf_image(
    filepath: Path,
    label: str,
    size: Tuple[int, int] = (256, 256),
    seed: int = 42,
) -> None:
    """
    Generate a synthetic leaf-like image with simulated color and lesion textures.
    """
    rng = np.random.default_rng(seed)
    w, h = size

    # Base leaf green background with slight gradient
    base = np.zeros((h, w, 3), dtype=np.uint8)
    # Green variations (R: 30-70, G: 120-190, B: 30-70)
    base[:, :, 0] = rng.integers(30, 70, (h, w), dtype=np.uint8)
    base[:, :, 1] = rng.integers(120, 190, (h, w), dtype=np.uint8)
    base[:, :, 2] = rng.integers(30, 70, (h, w), dtype=np.uint8)

    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)

    # Add leaf vein structures
    draw.line([(w // 2, 10), (w // 2, h - 10)], fill=(70, 210, 70), width=3)
    for y_offset in range(30, h - 30, 25):
        draw.line([(w // 2, y_offset), (w // 4, y_offset - 15)], fill=(65, 195, 65), width=2)
        draw.line([(w // 2, y_offset), (3 * w // 4, y_offset - 15)], fill=(65, 195, 65), width=2)

    # Add disease-specific lesions
    if "early_blight" in label:
        # Concentric dark brown rings
        for cx, cy in [(w // 3, h // 3), (2 * w // 3, 2 * h // 3)]:
            draw.ellipse([cx - 18, cy - 18, cx + 18, cy + 18], fill=(90, 50, 20), outline=(130, 80, 30), width=2)
            draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(50, 25, 10))
    elif "late_blight" in label:
        # Irregular water-soaked dark patches
        draw.polygon([(w // 2 - 20, h // 2 - 20), (w // 2 + 30, h // 2 - 10), (w // 2 + 10, h // 2 + 35)], fill=(40, 45, 30))
        draw.polygon([(w // 4, h // 2), (w // 4 + 25, h // 2 - 15), (w // 4 + 15, h // 2 + 20)], fill=(45, 50, 35))
    elif "bacterial_spot" in label:
        # Small speckled dark dots with yellow halo
        for _ in range(8):
            bx = int(rng.integers(30, w - 30))
            by = int(rng.integers(30, h - 30))
            draw.ellipse([bx - 6, by - 6, bx + 6, by + 6], fill=(160, 150, 40))  # yellow halo
            draw.ellipse([bx - 3, by - 3, bx + 3, by + 3], fill=(30, 20, 10))   # necrotic center
    # "healthy" leaves have no lesions

    filepath.parent.mkdir(parents=True, exist_ok=True)
    img.save(filepath, format="JPEG", quality=90)


def create_synthetic_aerial_image(
    filepath: Path,
    size: Tuple[int, int] = (1024, 1024),
    seed: int = 100,
) -> List[Dict[str, Any]]:
    """
    Generate a synthetic aerial canopy image with crop rows and stressed patches.
    Returns simulated ground truth bounding boxes.
    """
    rng = np.random.default_rng(seed)
    w, h = size

    # Soil background (brownish)
    img_arr = np.zeros((h, w, 3), dtype=np.uint8)
    img_arr[:, :, 0] = rng.integers(110, 140, (h, w), dtype=np.uint8)
    img_arr[:, :, 1] = rng.integers(85, 110, (h, w), dtype=np.uint8)
    img_arr[:, :, 2] = rng.integers(60, 80, (h, w), dtype=np.uint8)

    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)

    # Draw dense green crop rows
    row_spacing = 40
    for x in range(20, w, row_spacing):
        draw.rectangle([x, 0, x + 24, h], fill=(35, 140, 45))

    # Add visible crop stress regions (yellowing / drought patches)
    annotations = [
        {"x_min": 120, "y_min": 150, "x_max": 280, "y_max": 310, "label": "visible_crop_stress"},
        {"x_min": 600, "y_min": 520, "x_max": 780, "y_max": 700, "label": "abnormal_canopy_void"},
    ]

    for ann in annotations:
        if ann["label"] == "visible_crop_stress":
            draw.rectangle([ann["x_min"], ann["y_min"], ann["x_max"], ann["y_max"]], fill=(160, 150, 40))
        elif ann["label"] == "abnormal_canopy_void":
            draw.rectangle([ann["x_min"], ann["y_min"], ann["x_max"], ann["y_max"]], fill=(95, 75, 55))

    filepath.parent.mkdir(parents=True, exist_ok=True)
    img.save(filepath, format="JPEG", quality=92)
    return annotations


def generate_synthetic_dataset(
    base_dir: str | Path,
    num_samples_per_class: int = 6,
    num_farms: int = 3,
) -> Dict[str, Any]:
    """
    Create a complete self-contained synthetic fixture for close-up and aerial pipelines.
    Includes:
    - Multiple disease classes + healthy
    - Multiple farm IDs and session IDs
    - metadata.csv
    - Aerial image with annotations.json
    - NOTICE.txt documenting synthetic nature and licensing
    """
    base_path = Path(base_dir)
    close_up_dir = base_path / "close_up"
    aerial_dir = base_path / "aerial"

    close_up_dir.mkdir(parents=True, exist_ok=True)
    aerial_dir.mkdir(parents=True, exist_ok=True)

    classes = [
        "tomato_healthy",
        "tomato_early_blight",
        "tomato_late_blight",
        "tomato_bacterial_spot",
    ]
    farms = [f"farm_{chr(65 + i)}" for i in range(num_farms)]  # farm_A, farm_B, farm_C
    sessions = ["session_morning", "session_afternoon"]

    records = []
    sample_idx = 0

    for label in classes:
        for i in range(num_samples_per_class):
            farm = farms[i % len(farms)]
            session = sessions[(i // len(farms)) % len(sessions)]
            filename = f"{farm}_{session}_{label}_{i:03d}.jpg"
            img_rel = Path(label) / filename
            img_path = close_up_dir / img_rel

            create_synthetic_leaf_image(
                img_path,
                label=label,
                size=(224, 224),
                seed=sample_idx * 17 + 42,
            )

            # Simulated field coordinates (e.g. Salinas Valley, CA)
            lat = round(36.6777 + (i * 0.001), 6)
            lon = round(-121.6555 + (i * 0.001), 6)

            records.append({
                "image_path": str(img_rel).replace("\\", "/"),
                "label": label,
                "crop": "tomato",
                "farm_id": farm,
                "session_id": session,
                "gps_lat": lat,
                "gps_lon": lon,
            })
            sample_idx += 1

    # Write metadata.csv
    meta_csv = close_up_dir / "metadata.csv"
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_path", "label", "crop", "farm_id", "session_id", "gps_lat", "gps_lon"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Generate synthetic aerial orthophoto
    aerial_img_path = aerial_dir / "ortho_farm_A_session_01.jpg"
    aerial_annotations = create_synthetic_aerial_image(aerial_img_path, size=(1024, 1024), seed=42)
    aerial_meta = {
        "image_file": "ortho_farm_A_session_01.jpg",
        "farm_id": "farm_A",
        "session_id": "flight_session_01",
        "flight_altitude_m": 45.0,
        "gps_lat": 36.6780,
        "gps_lon": -121.6560,
        "annotations": aerial_annotations,
    }
    with open(aerial_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(aerial_meta, f, indent=2)

    # Write notice
    with open(base_path / "NOTICE.txt", "w", encoding="utf-8") as f:
        f.write(SYNTHETIC_DISCLAIMER + "\n")

    return {
        "close_up_dir": str(close_up_dir),
        "aerial_dir": str(aerial_dir),
        "num_records": len(records),
        "classes": classes,
        "farms": farms,
    }
