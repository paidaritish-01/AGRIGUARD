"""
Unit tests for inference engine and GPS metadata extraction.
"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import io
import pytest

from agriscan.inference.engine import AgriScanInferenceEngine
from agriscan.inference.exif_utils import extract_exif_gps


def test_inference_unsupported_crop():
    """Verify that unsupported crops yield status 'unsupported' and score 0.0."""
    engine = AgriScanInferenceEngine(close_up_checkpoint=None)
    img = Image.new("RGB", (224, 224), color=(50, 150, 50))
    result = engine.run_close_up_inference(img, crop="exotic_dragonfruit")

    assert result["status"] == "unsupported"
    assert result["predicted_label"] is None
    assert result["model_score"] == 0.0
    assert "exotic_dragonfruit" in result["message"]


def test_inference_demo_mode_explicit_labeling():
    """Verify that demo mode is explicitly declared and not silently substituted."""
    engine = AgriScanInferenceEngine(close_up_checkpoint=None)
    assert engine.is_demo_mode is True

    img = Image.new("RGB", (224, 224), color=(50, 150, 50))
    result = engine.run_close_up_inference(img, crop="tomato")

    assert result["is_demo_mode"] is True
    assert "demo" in result["model_version"].lower()
    assert any("DEMO MODE" in lim for lim in result["relevant_limitations"])


def test_aerial_inference_runs_and_produces_boxes():
    """Verify aerial inference operates on large canvas and produces bounding boxes."""
    engine = AgriScanInferenceEngine(close_up_checkpoint=None)
    img = Image.new("RGB", (600, 600), color=(120, 90, 60))
    result = engine.run_aerial_inference(img, crop="general_crop")

    assert result["analysis_mode"] == "aerial"
    assert "bounding_boxes" in result
    assert "tiling_summary" in result
    assert result["tiling_summary"]["total_tiles_analyzed"] > 0
