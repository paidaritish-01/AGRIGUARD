"""
Unit tests for label mappings and persistence.
"""

import json
from pathlib import Path
import pytest
from agriscan.data.dataset import AgriCloseUpDataset


def test_label_mapping_generation(tmp_path):
    records = [
        {"image_path": "dummy1.jpg", "label": "tomato_healthy"},
        {"image_path": "dummy2.jpg", "label": "tomato_early_blight"},
        {"image_path": "dummy3.jpg", "label": "tomato_late_blight"},
        {"image_path": "dummy4.jpg", "label": "tomato_healthy"},
    ]

    dataset = AgriCloseUpDataset(records, dataset_dir=tmp_path)
    assert len(dataset.label_to_idx) == 3
    assert dataset.label_to_idx["tomato_early_blight"] == 0
    assert dataset.label_to_idx["tomato_healthy"] == 1
    assert dataset.label_to_idx["tomato_late_blight"] == 2

    # Check inverse mapping
    for label, idx in dataset.label_to_idx.items():
        assert dataset.idx_to_label[idx] == label


def test_label_mapping_save_and_load(tmp_path):
    records = [
        {"image_path": "dummy1.jpg", "label": "corn_blight"},
        {"image_path": "dummy2.jpg", "label": "corn_rust"},
    ]

    dataset = AgriCloseUpDataset(records, dataset_dir=tmp_path)
    json_path = tmp_path / "label_mapping.json"
    dataset.save_label_mapping(json_path)

    assert json_path.exists()
    l2i, i2l = AgriCloseUpDataset.load_label_mapping(json_path)

    assert l2i == dataset.label_to_idx
    assert i2l == dataset.idx_to_label
