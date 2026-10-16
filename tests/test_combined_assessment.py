"""
Unit tests for the Combined Assessment coordinator.
Verifies multi-source synthesis, concordance/discordance reasoning,
partial evidence handling, and contract schema adherence.
"""

import pytest
from agriscan.assessment.combined import CombinedAssessmentCoordinator
from agriscan.sensors.schema import (
    SensorAnalysisResponse,
    DataQualityReport,
    SensorDerivedFeatures,
    EnvironmentalRiskResult,
)


@pytest.fixture
def coordinator():
    return CombinedAssessmentCoordinator()


def test_combined_mode_image_only(coordinator):
    image_result = {
        "image_id": "mango_leaf_01.jpg",
        "analysis_mode": "close_up",
        "model_version": "1.0.0",
        "supported_crop": "mango",
        "predicted_label": "mango_anthracnose",
        "model_score": 0.88,
        "score_type": "calibrated_probability",
        "status": "success",
    }
    result = coordinator.synthesize(
        image_id="mango_leaf_01.jpg",
        crop="mango",
        image_result=image_result,
        sensor_result=None,
    )

    assert result["assessment_status"] == "partial_evidence"
    assert result["combined_assessment"]["evidence_concordance"] == "single_source"
    assert result["image_analysis"]["status"] == "success"
    assert result["sensor_analysis"]["status"] == "not_assessed"
    assert "solely on visual close-up" in result["combined_assessment"]["explanation"]


def test_combined_mode_sensor_only(coordinator):
    sensor_result = SensorAnalysisResponse(
        status="assessed",
        plot_id="plot_mango_01",
        window_hours=72.0,
        data_quality=DataQualityReport(record_count=36, valid_count=36),
        derived_features=SensorDerivedFeatures(
            window_hours=72.0,
            measurement_count=36,
            coverage_pct=100.0,
            temperature_mean_c=26.5,
            humidity_mean_pct=89.0,
        ),
        risk_assessment=[
            EnvironmentalRiskResult(
                disease_or_condition="Mango Anthracnose",
                pathogen_or_stressor="Colletotrichum gloeosporioides",
                risk_level="high",
                favorable_conditions_met=True,
                explanation="High humidity and optimal temperature.",
                required_variables=["air_temperature_c", "relative_humidity_pct"],
                source_citation="Dodd et al. (1992)",
                evidence_type="literature-informed",
                assumptions_and_limitations="Microclimate favorability only.",
            )
        ],
    )

    result = coordinator.synthesize(
        crop="mango",
        image_result=None,
        sensor_result=sensor_result,
    )

    assert result["assessment_status"] == "partial_evidence"
    assert result["combined_assessment"]["evidence_concordance"] == "single_source"
    assert result["image_analysis"]["status"] == "not_assessed"
    assert result["sensor_analysis"]["status"] == "assessed"
    assert "solely on environmental microclimate" in result["combined_assessment"]["explanation"]


def test_combined_mode_concordant_anthracnose(coordinator):
    image_result = {
        "image_id": "mango_leaf_02.jpg",
        "status": "success",
        "predicted_label": "mango_anthracnose",
        "model_score": 0.91,
        "score_type": "calibrated_probability",
        "model_version": "1.0.0",
    }
    sensor_result = SensorAnalysisResponse(
        status="assessed",
        plot_id="plot_mango_01",
        window_hours=72.0,
        data_quality=DataQualityReport(record_count=36, valid_count=36),
        risk_assessment=[
            EnvironmentalRiskResult(
                disease_or_condition="Mango Anthracnose",
                pathogen_or_stressor="Colletotrichum gloeosporioides",
                risk_level="high",
                favorable_conditions_met=True,
                explanation="High favorability.",
                required_variables=["air_temperature_c", "relative_humidity_pct"],
                source_citation="Dodd et al. (1992)",
                evidence_type="literature-informed",
                assumptions_and_limitations="Test note",
            )
        ],
    )

    result = coordinator.synthesize(
        image_id="mango_leaf_02.jpg",
        crop="mango",
        image_result=image_result,
        sensor_result=sensor_result,
    )

    assert result["assessment_status"] == "completed"
    assert result["combined_assessment"]["evidence_concordance"] == "concordant"
    assert "Concordant positive finding" in result["combined_assessment"]["explanation"]
    assert any("Dieback" in act or "agronomist" in act.lower() for act in result["combined_assessment"]["recommended_actions"])


def test_combined_mode_discordant_healthy_leaf_high_risk(coordinator):
    image_result = {
        "image_id": "mango_leaf_clean.jpg",
        "status": "success",
        "predicted_label": "mango_healthy",
        "model_score": 0.95,
        "score_type": "calibrated_probability",
        "model_version": "1.0.0",
    }
    sensor_result = SensorAnalysisResponse(
        status="assessed",
        plot_id="plot_mango_01",
        window_hours=72.0,
        data_quality=DataQualityReport(record_count=36, valid_count=36),
        risk_assessment=[
            EnvironmentalRiskResult(
                disease_or_condition="Mango Anthracnose",
                pathogen_or_stressor="Colletotrichum gloeosporioides",
                risk_level="high",
                favorable_conditions_met=True,
                explanation="Optimal conditions for conidial germination.",
                required_variables=["air_temperature_c", "relative_humidity_pct"],
                source_citation="Dodd et al. (1992)",
                evidence_type="literature-informed",
                assumptions_and_limitations="Test note",
            )
        ],
    )

    result = coordinator.synthesize(
        image_id="mango_leaf_clean.jpg",
        crop="mango",
        image_result=image_result,
        sensor_result=sensor_result,
    )

    assert result["assessment_status"] == "completed"
    assert result["combined_assessment"]["evidence_concordance"] == "discordant"
    assert "Discordant preventive finding" in result["combined_assessment"]["explanation"]
    assert any("Scout the orchard" in act for act in result["combined_assessment"]["recommended_actions"])


def test_combined_mode_neither_available(coordinator):
    result = coordinator.synthesize(
        image_id=None,
        crop="mango",
        image_result=None,
        sensor_result=None,
    )

    assert result["assessment_status"] == "insufficient_data"
    assert result["combined_assessment"]["evidence_concordance"] == "insufficient"
    assert "adequate sensor history is not available" in result["combined_assessment"]["explanation"]


def test_combined_response_schema_contract(coordinator):
    """Verify that all Section 9 required fields exist in the synthesized output."""
    result = coordinator.synthesize(
        image_id="contract_test_img",
        crop="mango",
        image_result=None,
        sensor_result=None,
    )

    required_keys = [
        "image_id",
        "crop",
        "assessment_status",
        "image_analysis",
        "sensor_analysis",
        "combined_assessment",
        "limitations",
        "processing_time_ms",
    ]
    for key in required_keys:
        assert key in result, f"Missing required top-level key: {key}"

    assert "status" in result["image_analysis"]
    assert "predicted_class" in result["image_analysis"]
    assert "scores" in result["image_analysis"]
    assert "model_version" in result["image_analysis"]

    assert "status" in result["sensor_analysis"]
    assert "data_quality" in result["sensor_analysis"]
    assert "observations" in result["sensor_analysis"]
    assert "risk_assessment" in result["sensor_analysis"]

    assert "status" in result["combined_assessment"]
    assert "explanation" in result["combined_assessment"]
    assert "recommended_actions" in result["combined_assessment"]
