"""
Unit and integration tests for the AgriScan sensor-history module.
Tests:
- CSV schema validation (required/optional columns, type parsing).
- Data quality checks (out-of-order timestamps, duplicates, sampling gaps, implausible bounds).
- Derived statistical features (mean, min, max, trends, condition durations).
- Literature-grounded risk rules (Anthracnose, Powdery Mildew, Heat/Moisture stress).
- Insufficient sensor history handling.
- Synthetic fixture marker detection.
"""

import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from agriscan.sensors.schema import SensorRecord, SensorDerivedFeatures
from agriscan.sensors.validator import SensorDataValidator
from agriscan.sensors.service import SensorHistoryService
from agriscan.sensors.risk_rules import evaluate_mango_environmental_risks


def test_validator_parses_valid_csv():
    csv_data = """# SYNTHETIC SENSOR FIXTURE - TEST ONLY
timestamp,plot_id,crop,air_temperature_c,relative_humidity_pct,soil_moisture_pct,rainfall_mm,leaf_wetness_hours
2026-10-10T00:00:00Z,plot_01,mango,26.5,88.0,40.0,0.0,1.0
2026-10-10T01:00:00Z,plot_01,mango,27.0,89.5,40.5,0.5,1.0
2026-10-10T02:00:00Z,plot_01,mango,26.8,91.0,41.0,1.2,1.0
"""
    validator = SensorDataValidator()
    records, report = validator.parse_and_validate_csv(csv_data)

    assert len(records) == 3
    assert report.valid_count == 3
    assert report.record_count == 3
    assert report.is_synthetic_fixture is True
    assert len(report.missing_fields) == 0
    assert records[0].plot_id == "plot_01"
    assert records[0].air_temperature_c == 26.5
    assert records[0].relative_humidity_pct == 88.0


def test_validator_rejects_missing_mandatory_headers():
    bad_csv = """timestamp,plot_id,air_temperature_c
2026-10-10T00:00:00Z,plot_01,26.5
"""
    validator = SensorDataValidator()
    records, report = validator.parse_and_validate_csv(bad_csv)

    assert len(records) == 0
    assert "relative_humidity_pct" in report.missing_fields
    assert "soil_moisture_pct" in report.missing_fields
    assert any("mandatory columns" in w for w in report.warnings)


def test_validator_sorts_out_of_order_and_dedupes():
    csv_data = """timestamp,plot_id,air_temperature_c,relative_humidity_pct,soil_moisture_pct
2026-10-10T02:00:00Z,plot_01,25.0,80.0,35.0
2026-10-10T00:00:00Z,plot_01,24.0,85.0,35.0
2026-10-10T00:00:00Z,plot_01,24.2,85.5,35.0
2026-10-10T01:00:00Z,plot_01,24.5,82.0,35.0
"""
    validator = SensorDataValidator()
    records, report = validator.parse_and_validate_csv(csv_data)

    assert len(records) == 3  # Duplicate at 00:00:00 removed
    assert records[0].timestamp.hour == 0
    assert records[1].timestamp.hour == 1
    assert records[2].timestamp.hour == 2
    assert any("sorting automatically" in w for w in report.warnings)
    assert any("Duplicate reading" in w for w in report.warnings)


def test_validator_detects_sampling_gaps():
    csv_data = """timestamp,plot_id,air_temperature_c,relative_humidity_pct,soil_moisture_pct
2026-10-10T00:00:00Z,plot_01,25.0,80.0,35.0
2026-10-10T01:00:00Z,plot_01,25.2,80.5,35.0
2026-10-10T06:00:00Z,plot_01,26.0,82.0,35.0
"""
    validator = SensorDataValidator(max_sampling_gap_hours=3.0)
    records, report = validator.parse_and_validate_csv(csv_data)

    assert len(records) == 3
    assert any("Data gap detected" in w for w in report.warnings)


def test_validator_flags_implausible_physical_values():
    csv_data = """timestamp,plot_id,air_temperature_c,relative_humidity_pct,soil_moisture_pct,rainfall_mm
2026-10-10T00:00:00Z,plot_01,75.0,85.0,35.0,0.0
2026-10-10T01:00:00Z,plot_01,25.0,120.0,35.0,0.0
2026-10-10T02:00:00Z,plot_01,25.0,85.0,35.0,-5.0
"""
    validator = SensorDataValidator()
    records, report = validator.parse_and_validate_csv(csv_data)

    assert any("outside plausible range" in w for w in report.warnings)
    assert any("outside valid bounds" in w for w in report.warnings)
    assert any("Negative rainfall" in w for w in report.warnings)


def test_service_query_and_derived_features(tmp_path):
    service = SensorHistoryService(storage_dir=tmp_path / "sensor_storage")

    # Generate 12 hourly readings for plot_A
    base_ts = datetime(2026, 10, 10, 0, 0, 0, tzinfo=timezone.utc)
    records = []
    for i in range(12):
        ts = base_ts + timedelta(hours=i)
        rec = SensorRecord(
            timestamp=ts,
            plot_id="plot_A",
            air_temperature_c=25.0 + (i * 0.2),  # Increasing temp trend
            relative_humidity_pct=88.0,          # High RH >= 85
            soil_moisture_pct=40.0,
            rainfall_mm=0.5,
            leaf_wetness_hours=1.0,
        )
        records.append(rec)

    response = service.assess_sensor_history(
        plot_id="plot_A",
        target_timestamp=base_ts + timedelta(hours=11),
        window_hours=24.0,
        records=records,
    )

    assert response.status == "assessed"
    assert response.plot_id == "plot_A"
    assert response.derived_features is not None
    assert response.derived_features.measurement_count == 12
    assert response.derived_features.hours_high_humidity_above_85 >= 10.0
    assert response.derived_features.temperature_trend_c_per_hour is not None
    assert response.derived_features.temperature_trend_c_per_hour > 0.0

    # Anthracnose risk should be high due to 25°C-27°C and high RH
    anth_risk = next((r for r in response.risk_assessment if "Anthracnose" in r.disease_or_condition), None)
    assert anth_risk is not None
    assert anth_risk.risk_level == "high"
    assert anth_risk.evidence_type == "literature-informed"


def test_risk_rules_insufficient_history():
    # Only 2 observations (< 6 threshold)
    features = SensorDerivedFeatures(
        window_hours=2.0,
        measurement_count=2,
        coverage_pct=10.0,
        temperature_mean_c=25.0,
        humidity_mean_pct=85.0,
    )
    risks, stresses = evaluate_mango_environmental_risks(features)

    assert len(risks) == 2
    assert all(r.risk_level == "insufficient_data" for r in risks)
    assert any("Insufficient temporal history" in r.explanation for r in risks)


def test_powdery_mildew_high_risk_diurnal_swing():
    features = SensorDerivedFeatures(
        window_hours=48.0,
        measurement_count=24,
        coverage_pct=100.0,
        temperature_mean_c=23.5,  # Within 20-26°C optimal
        humidity_mean_pct=68.0,
        humidity_min_pct=52.0,    # <= 65% dry day
        humidity_max_pct=79.0,    # >= 75% humid night
        total_rainfall_mm=0.0,    # No rain
        hours_high_humidity_above_85=0.0,
    )
    risks, stresses = evaluate_mango_environmental_risks(features)

    pm_risk = next((r for r in risks if "Powdery Mildew" in r.disease_or_condition), None)
    assert pm_risk is not None
    assert pm_risk.risk_level == "high"
    assert pm_risk.favorable_conditions_met is True
    assert "optimal temperature" in pm_risk.explanation.lower()


def test_abiotic_stress_indicators():
    features = SensorDerivedFeatures(
        window_hours=24.0,
        measurement_count=12,
        coverage_pct=100.0,
        temperature_mean_c=36.0,
        temperature_max_c=39.5,   # >= 38.0 heat stress
        soil_moisture_mean_pct=14.5, # < 18.0 drought stress
        humidity_mean_pct=30.0,
    )
    risks, stresses = evaluate_mango_environmental_risks(features)

    assert len(stresses) == 2
    assert any("Extreme ambient heat stress" in s for s in stresses)
    assert any("Severe soil moisture deficit" in s for s in stresses)
