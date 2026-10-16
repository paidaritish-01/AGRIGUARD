"""
End-to-end smoke test for the Mango AI crop-health assessment pipeline in AgriScan.

Executes complete lifecycle:
1. Validates synthetic sensor history CSV fixture.
2. Exercises SensorHistoryService to ingest, filter by plot and time-window, and extract derived microclimate features.
3. Evaluates literature-informed disease favorability rules (Anthracnose and Powdery Mildew).
4. Verifies AgriScanInferenceEngine handling of mango (honestly reporting model_unavailable when no checkpoint is present).
5. Exercises CombinedAssessmentCoordinator to synthesize image analysis with sensor analysis and produce a Section 9 compliant response contract.
"""

from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

from agriscan.sensors.validator import SensorDataValidator
from agriscan.sensors.service import SensorHistoryService
from agriscan.inference.engine import AgriScanInferenceEngine
from agriscan.assessment.combined import CombinedAssessmentCoordinator


def test_mango_pipeline_end_to_end_smoke(tmp_path):
    # 1. Prepare synthetic sensor CSV in temp directory
    sensor_dir = tmp_path / "sensor_storage"
    sensor_dir.mkdir(parents=True, exist_ok=True)

    csv_content = """# SYNTHETIC MANGO SENSOR FIXTURE - END TO END SMOKE TEST ONLY
timestamp,plot_id,crop,air_temperature_c,relative_humidity_pct,soil_moisture_pct,rainfall_mm,leaf_wetness_hours
2026-10-10T00:00:00Z,plot_mango_e2e,mango,25.5,88.0,42.0,0.0,2.0
2026-10-10T04:00:00Z,plot_mango_e2e,mango,25.0,91.0,42.5,0.0,2.0
2026-10-10T08:00:00Z,plot_mango_e2e,mango,26.5,87.0,42.5,0.0,1.5
2026-10-10T12:00:00Z,plot_mango_e2e,mango,28.0,85.5,41.5,0.0,1.0
2026-10-10T16:00:00Z,plot_mango_e2e,mango,27.2,88.5,42.0,1.5,2.0
2026-10-10T20:00:00Z,plot_mango_e2e,mango,25.8,92.0,43.0,0.0,2.0
2026-10-11T00:00:00Z,plot_mango_e2e,mango,25.2,93.0,43.0,0.0,2.0
2026-10-11T04:00:00Z,plot_mango_e2e,mango,24.8,94.0,43.5,0.0,2.0
"""
    test_csv = tmp_path / "e2e_sensor.csv"
    test_csv.write_text(csv_content, encoding="utf-8")

    # 2. Validate CSV using SensorDataValidator
    validator = SensorDataValidator()
    records, quality_report = validator.parse_and_validate_csv(test_csv)
    assert len(records) == 8
    assert quality_report.valid_count == 8
    assert quality_report.is_synthetic_fixture is True
    assert len(quality_report.missing_fields) == 0

    # 3. Query sensor history via SensorHistoryService
    service = SensorHistoryService(storage_dir=sensor_dir, validator=validator)
    sensor_response = service.assess_sensor_history(
        plot_id="plot_mango_e2e",
        target_timestamp=datetime(2026, 10, 11, 4, 0, 0, tzinfo=timezone.utc),
        window_hours=48.0,
        records=records,
    )

    assert sensor_response.status == "assessed"
    assert sensor_response.plot_id == "plot_mango_e2e"
    assert sensor_response.derived_features is not None
    assert sensor_response.derived_features.measurement_count == 8
    assert sensor_response.derived_features.hours_high_humidity_above_85 >= 12.0

    # Verify Anthracnose risk evaluated as High
    anth_risk = next((r for r in sensor_response.risk_assessment if "Anthracnose" in r.disease_or_condition), None)
    assert anth_risk is not None
    assert anth_risk.risk_level == "high"
    assert anth_risk.evidence_type == "literature-informed"

    # 4. Exercise AgriScanInferenceEngine on mango image
    # Note: No mango checkpoint exists in repo; must honestly declare model_unavailable
    engine = AgriScanInferenceEngine(close_up_checkpoint=None)
    test_leaf = Image.new("RGB", (224, 224), color=(60, 160, 60))
    image_result = engine.run_close_up_inference(test_leaf, crop="mango", image_id="e2e_mango_leaf.jpg")

    assert image_result["status"] == "model_unavailable"
    assert image_result["predicted_label"] is None
    assert image_result["model_score"] == 0.0
    assert "No validated mango close-up" in image_result["message"]

    # 5. Synthesize Combined Assessment
    coordinator = CombinedAssessmentCoordinator()
    combined_result = coordinator.synthesize(
        image_id="e2e_mango_leaf.jpg",
        crop="mango",
        image_result=image_result,
        sensor_result=sensor_response,
    )

    # Verify Section 9 contract compliance
    assert combined_result["image_id"] == "e2e_mango_leaf.jpg"
    assert combined_result["crop"] == "mango"
    assert combined_result["assessment_status"] == "partial_evidence"
    assert combined_result["image_analysis"]["status"] == "model_unavailable"
    assert combined_result["sensor_analysis"]["status"] == "assessed"
    assert combined_result["combined_assessment"]["evidence_concordance"] == "single_source"
    assert "solely on environmental microclimate" in combined_result["combined_assessment"]["explanation"]
    assert len(combined_result["combined_assessment"]["recommended_actions"]) > 0
    assert any("pesticide" in lim.lower() for lim in combined_result["limitations"])
    assert combined_result["processing_time_ms"] is not None
