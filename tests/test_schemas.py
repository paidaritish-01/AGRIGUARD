"""
Unit tests for Pydantic API response and request schemas.
"""

import pytest
from pydantic import ValidationError
from agriscan.api.schemas import (
    PredictResponse,
    HealthResponse,
    ModelInfoResponse,
    GPSMetadataSchema,
    BoundingBoxSchema,
)


def test_valid_predict_response_schema():
    payload = {
        "image_id": "test_img_01.jpg",
        "analysis_mode": "close_up",
        "model_version": "1.0.0",
        "supported_crop": "tomato",
        "predicted_label": "tomato_early_blight",
        "model_score": 0.892,
        "score_type": "calibrated_probability",
        "status": "success",
        "message": "Diagnosis complete",
        "bounding_boxes": None,
        "gps_metadata": {
            "latitude": 36.678,
            "longitude": -121.656,
            "altitude_m": 45.0,
            "has_gps": True,
            "note": "Sensor position only",
        },
        "processing_time_ms": 32.5,
        "is_demo_mode": False,
        "relevant_limitations": ["No pesticide recommendations"],
    }
    resp = PredictResponse(**payload)
    assert resp.status == "success"
    assert resp.predicted_label == "tomato_early_blight"
    assert resp.gps_metadata.has_gps is True


def test_invalid_status_rejected_by_schema():
    payload = {
        "image_id": "test_img_02.jpg",
        "analysis_mode": "close_up",
        "model_version": "1.0.0",
        "supported_crop": "tomato",
        "predicted_label": None,
        "model_score": 0.0,
        "score_type": "not_applicable",
        "status": "invalid_status_value",  # Invalid
        "message": "Failed",
        "bounding_boxes": None,
        "gps_metadata": {"note": "Test note"},
        "processing_time_ms": 10.0,
        "is_demo_mode": True,
        "relevant_limitations": [],
    }
    with pytest.raises(ValidationError):
        PredictResponse(**payload)


def test_bounding_box_schema():
    box = BoundingBoxSchema(bbox=[10.0, 20.0, 100.0, 150.0], confidence=0.85, label="visible_crop_stress")
    assert box.bbox == [10.0, 20.0, 100.0, 150.0]
    assert box.confidence == 0.85
