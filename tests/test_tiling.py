"""
Unit tests for aerial tiling engine and NMS detection merging.
"""

from PIL import Image
import pytest
from agriscan.data.tiling import AerialTilingEngine, DetectionBox


def test_tiling_slice_dimensions():
    engine = AerialTilingEngine(tile_size=256, overlap=0.20)
    # 600x600 image should produce multiple overlapping tiles
    img = Image.new("RGB", (600, 600), color=(100, 150, 50))
    tiles = engine.slice_image(img)

    assert len(tiles) > 1
    for t in tiles:
        assert t.x_max <= 600
        assert t.y_max <= 600
        assert t.x_min >= 0
        assert t.y_min >= 0
        assert t.image.size == (256, 256)


def test_tiling_small_image_no_slice():
    engine = AerialTilingEngine(tile_size=512, overlap=0.20)
    img = Image.new("RGB", (300, 300), color=(80, 140, 60))
    tiles = engine.slice_image(img)

    assert len(tiles) == 1
    assert tiles[0].x_min == 0
    assert tiles[0].y_min == 0
    assert tiles[0].x_max == 300
    assert tiles[0].y_max == 300


def test_nms_merging_overlapping_boxes():
    engine = AerialTilingEngine(tile_size=512, overlap=0.20, nms_iou_threshold=0.40)

    # Two almost identical overlapping boxes for the same class
    box1 = DetectionBox(x_min=100.0, y_min=100.0, x_max=200.0, y_max=200.0, confidence=0.90, label="visible_crop_stress")
    box2 = DetectionBox(x_min=105.0, y_min=105.0, x_max=205.0, y_max=205.0, confidence=0.75, label="visible_crop_stress")
    # One separate box far away
    box3 = DetectionBox(x_min=400.0, y_min=400.0, x_max=500.0, y_max=500.0, confidence=0.85, label="visible_crop_stress")

    merged = engine.merge_detections([box1, box2, box3])
    # Box2 should be suppressed by box1 due to high IoU (> 0.40)
    assert len(merged) == 2
    assert merged[0].confidence == 0.90
    assert merged[1].confidence == 0.85


def test_nms_preserves_different_classes():
    engine = AerialTilingEngine(tile_size=512, overlap=0.20, nms_iou_threshold=0.40)

    # Identical bounding boxes but different classes
    box1 = DetectionBox(x_min=100.0, y_min=100.0, x_max=200.0, y_max=200.0, confidence=0.88, label="visible_crop_stress")
    box2 = DetectionBox(x_min=100.0, y_min=100.0, x_max=200.0, y_max=200.0, confidence=0.85, label="abnormal_canopy_void")

    merged = engine.merge_detections([box1, box2])
    # Different classes should NOT suppress each other
    assert len(merged) == 2
