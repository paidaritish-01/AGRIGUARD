"""
Service layer for sensor-history file management, plot-based time window queries,
statistical feature derivation, and environmental risk assessment.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from agriscan.sensors.schema import (
    SensorRecord,
    DataQualityReport,
    SensorDerivedFeatures,
    SensorAnalysisResponse,
)
from agriscan.sensors.validator import SensorDataValidator
from agriscan.sensors.risk_rules import evaluate_mango_environmental_risks


STANDARD_SENSOR_LIMITATIONS = [
    "Environmental risk assessments indicate pathogen microclimate favorability only, NOT biological infection or disease confirmation.",
    "Sensor records represent localized environmental readings and may not capture canopy microclimate variations.",
    "Automated findings are for screening and scouting guidance only. Consult an agronomist before taking action.",
    "Strict safety policy: AgriScan does not generate pesticide recommendations.",
]


class SensorHistoryService:
    """
    Manages local storage and querying of historical sensor readings.
    """

    def __init__(
        self,
        storage_dir: str | Path = "data/sensor_history",
        validator: Optional[SensorDataValidator] = None,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.validator = validator or SensorDataValidator()

    def save_csv(self, content: str, filename: str) -> Tuple[Path, DataQualityReport, int]:
        """
        Saves uploaded CSV string to storage directory after preliminary validation.
        """
        clean_name = Path(filename).name
        if not clean_name.lower().endswith(".csv"):
            clean_name = f"{clean_name}.csv"

        dest_path = self.storage_dir / clean_name
        dest_path.write_text(content, encoding="utf-8")

        records, report = self.validator.parse_and_validate_csv(dest_path)
        return dest_path, report, len(records)

    def load_records_from_csv(self, csv_path: str | Path) -> Tuple[List[SensorRecord], DataQualityReport]:
        """Load and parse records from a specified CSV file."""
        path = Path(csv_path)
        if not path.exists():
            # Check relative to storage_dir
            alt_path = self.storage_dir / path.name
            if alt_path.exists():
                path = alt_path
            else:
                raise FileNotFoundError(f"Sensor CSV file not found at: {csv_path}")

        return self.validator.parse_and_validate_csv(path)

    def assess_sensor_history(
        self,
        plot_id: str,
        target_timestamp: Optional[datetime | str] = None,
        window_hours: float = 72.0,
        csv_file: Optional[str | Path] = None,
        records: Optional[List[SensorRecord]] = None,
        data_quality: Optional[DataQualityReport] = None,
    ) -> SensorAnalysisResponse:
        """
        Filters sensor history for the given plot_id and time window, computes derived
        statistical features, and evaluates literature-grounded risk rules.
        """
        clean_plot_id = plot_id.strip() if plot_id else ""
        if not clean_plot_id:
            return SensorAnalysisResponse(
                status="insufficient_data",
                plot_id=None,
                target_timestamp=None,
                window_hours=window_hours,
                data_quality=DataQualityReport(
                    warnings=["No plot_id provided for sensor assessment."]
                ),
                observations=[],
                derived_features=None,
                risk_assessment=[],
                general_stress_indicators=[],
                limitations=STANDARD_SENSOR_LIMITATIONS,
            )

        # 1. Ingest records
        all_records: List[SensorRecord] = []
        quality_report = data_quality or DataQualityReport()

        if records is not None:
            all_records = records
        elif csv_file is not None:
            try:
                all_records, quality_report = self.load_records_from_csv(csv_file)
            except Exception as e:
                return SensorAnalysisResponse(
                    status="insufficient_data",
                    plot_id=clean_plot_id,
                    target_timestamp=None,
                    window_hours=window_hours,
                    data_quality=DataQualityReport(
                        warnings=[f"Failed to load sensor history CSV: {str(e)}"]
                    ),
                    observations=[],
                    derived_features=None,
                    risk_assessment=[],
                    general_stress_indicators=[],
                    limitations=STANDARD_SENSOR_LIMITATIONS,
                )
        else:
            # Check default synthetic demo file if exists
            default_demo = Path("data/synthetic_sensors/sample_mango_sensor_history.csv")
            if default_demo.exists():
                all_records, quality_report = self.load_records_from_csv(default_demo)
            else:
                return SensorAnalysisResponse(
                    status="not_assessed",
                    plot_id=clean_plot_id,
                    target_timestamp=None,
                    window_hours=window_hours,
                    data_quality=DataQualityReport(
                        warnings=["No sensor CSV file or records provided and default demo file not found."]
                    ),
                    observations=[],
                    derived_features=None,
                    risk_assessment=[],
                    general_stress_indicators=[],
                    limitations=STANDARD_SENSOR_LIMITATIONS,
                )

        # 2. Filter records by plot_id
        plot_records = [
            r for r in all_records
            if r.plot_id.lower().strip() == clean_plot_id.lower().strip()
        ]

        if not plot_records:
            return SensorAnalysisResponse(
                status="insufficient_data",
                plot_id=clean_plot_id,
                target_timestamp=None,
                window_hours=window_hours,
                data_quality=quality_report,
                observations=[],
                derived_features=None,
                risk_assessment=[],
                general_stress_indicators=[],
                limitations=[
                    *STANDARD_SENSOR_LIMITATIONS,
                    f"No records found matching plot_id '{clean_plot_id}'. Available records contain other plots.",
                ],
            )

        # 3. Determine target timestamp & time window
        target_dt: datetime
        if target_timestamp:
            if isinstance(target_timestamp, str):
                target_dt = datetime.fromisoformat(target_timestamp.replace("Z", "+00:00"))
            else:
                target_dt = target_timestamp
            if target_dt.tzinfo is None:
                target_dt = target_dt.replace(tzinfo=timezone.utc)
        else:
            target_dt = max(r.timestamp for r in plot_records)

        window_start = target_dt - timedelta(hours=window_hours)

        # 4. Filter window records [window_start, target_dt]
        window_records = [
            r for r in plot_records
            if window_start <= r.timestamp <= target_dt
        ]
        window_records.sort(key=lambda r: r.timestamp)

        if not window_records:
            return SensorAnalysisResponse(
                status="insufficient_data",
                plot_id=clean_plot_id,
                target_timestamp=target_dt.isoformat(),
                window_hours=window_hours,
                data_quality=quality_report,
                observations=[],
                derived_features=None,
                risk_assessment=[],
                general_stress_indicators=[],
                limitations=[
                    *STANDARD_SENSOR_LIMITATIONS,
                    f"No sensor observations found in plot '{clean_plot_id}' between {window_start.isoformat()} and {target_dt.isoformat()}.",
                ],
            )

        # 5. Derive window features
        derived = self._calculate_derived_features(window_records, window_hours, window_start, target_dt)

        # 6. Evaluate literature-grounded risk rules
        risks, stress_indicators = evaluate_mango_environmental_risks(derived)

        # 7. Convert observations to compact dict format (limit to last 20 for payload compactness)
        obs_dicts = [
            {
                "timestamp": r.timestamp.isoformat(),
                "plot_id": r.plot_id,
                "air_temperature_c": r.air_temperature_c,
                "relative_humidity_pct": r.relative_humidity_pct,
                "soil_moisture_pct": r.soil_moisture_pct,
                "rainfall_mm": r.rainfall_mm,
                "leaf_wetness_hours": r.leaf_wetness_hours,
            }
            for r in window_records[-20:]
        ]

        # Status: assessed if features derived, else insufficient_data
        eval_status = "assessed" if derived.measurement_count >= 6 else "insufficient_data"

        return SensorAnalysisResponse(
            status=eval_status,
            plot_id=clean_plot_id,
            target_timestamp=target_dt.isoformat(),
            window_hours=window_hours,
            data_quality=quality_report,
            observations=obs_dicts,
            derived_features=derived,
            risk_assessment=risks,
            general_stress_indicators=stress_indicators,
            limitations=STANDARD_SENSOR_LIMITATIONS,
        )

    def _calculate_derived_features(
        self,
        records: List[SensorRecord],
        window_hours: float,
        window_start: datetime,
        window_end: datetime,
    ) -> SensorDerivedFeatures:
        """
        Computes window statistics, trends, and condition duration counts.
        """
        temps = [r.air_temperature_c for r in records]
        rhs = [r.relative_humidity_pct for r in records]
        soils = [r.soil_moisture_pct for r in records]
        rainfalls = [r.rainfall_mm for r in records if r.rainfall_mm is not None]
        leaf_wets = [r.leaf_wetness_hours for r in records if r.leaf_wetness_hours is not None]

        # Measurement duration & average interval
        if len(records) >= 2:
            time_span_hrs = max(0.1, (records[-1].timestamp - records[0].timestamp).total_seconds() / 3600.0)
            avg_interval_hrs = time_span_hrs / (len(records) - 1)
        else:
            time_span_hrs = window_hours
            avg_interval_hrs = 1.0

        coverage_pct = round(min(100.0, (time_span_hrs / max(1.0, window_hours)) * 100.0), 1)

        # Trends using linear regression slope (units / hour)
        temp_trend = None
        rh_trend = None
        if len(records) >= 3:
            hours_from_start = np.array([
                (r.timestamp - records[0].timestamp).total_seconds() / 3600.0 for r in records
            ])
            if np.std(hours_from_start) > 1e-4:
                temp_slope, _ = np.polyfit(hours_from_start, temps, 1)
                temp_trend = round(float(temp_slope), 3)
                rh_slope, _ = np.polyfit(hours_from_start, rhs, 1)
                rh_trend = round(float(rh_slope), 3)

        # Condition duration counts (hours)
        high_rh_count = sum(1 for r in records if r.relative_humidity_pct >= 85.0)
        hours_high_rh = round(high_rh_count * avg_interval_hrs, 1)

        fungal_temp_count = sum(1 for r in records if 24.0 <= r.air_temperature_c <= 30.0)
        hours_fungal_temp = round(fungal_temp_count * avg_interval_hrs, 1)

        return SensorDerivedFeatures(
            window_hours=window_hours,
            measurement_count=len(records),
            coverage_pct=coverage_pct,
            temperature_mean_c=round(float(np.mean(temps)), 2),
            temperature_min_c=round(float(np.min(temps)), 2),
            temperature_max_c=round(float(np.max(temps)), 2),
            temperature_trend_c_per_hour=temp_trend,
            humidity_mean_pct=round(float(np.mean(rhs)), 2),
            humidity_min_pct=round(float(np.min(rhs)), 2),
            humidity_max_pct=round(float(np.max(rhs)), 2),
            humidity_trend_pct_per_hour=rh_trend,
            hours_high_humidity_above_85=hours_high_rh,
            hours_temp_favorable_fungal_24_30=hours_fungal_temp,
            soil_moisture_mean_pct=round(float(np.mean(soils)), 2),
            soil_moisture_min_pct=round(float(np.min(soils)), 2),
            total_rainfall_mm=round(float(sum(rainfalls)), 2) if rainfalls else None,
            total_leaf_wetness_hours=round(float(sum(leaf_wets)), 2) if leaf_wets else None,
        )
