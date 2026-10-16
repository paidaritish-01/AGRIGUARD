"""
Validation and data-quality inspection for environmental sensor records and CSV files.
"""

import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from agriscan.sensors.schema import SensorRecord, DataQualityReport


REQUIRED_CSV_COLUMNS = {
    "timestamp",
    "plot_id",
    "air_temperature_c",
    "relative_humidity_pct",
    "soil_moisture_pct",
}

OPTIONAL_CSV_COLUMNS = {
    "tree_id",
    "crop",
    "rainfall_mm",
    "leaf_wetness_hours",
    "sensor_id",
    "quality_flag",
}


class SensorDataValidator:
    """
    Validates sensor CSV formatting, data types, physical plausibility,
    and temporal continuity.
    """

    def __init__(
        self,
        max_sampling_gap_hours: float = 3.0,
        temp_min_c: float = -10.0,
        temp_max_c: float = 60.0,
    ):
        self.max_sampling_gap_hours = max_sampling_gap_hours
        self.temp_min_c = temp_min_c
        self.temp_max_c = temp_max_c

    def parse_and_validate_csv(
        self,
        csv_source: str | Path | io.StringIO,
    ) -> Tuple[List[SensorRecord], DataQualityReport]:
        """
        Parses a CSV file or string, validates every row, sorts by timestamp,
        checks for duplicates/gaps, and flags data-quality issues.
        """
        warnings: List[str] = []
        missing_fields: List[str] = []
        is_synthetic = False

        if isinstance(csv_source, (str, Path)) and Path(str(csv_source)).exists():
            with open(csv_source, "r", encoding="utf-8") as f:
                content = f.read()
        elif isinstance(csv_source, io.StringIO):
            content = csv_source.getvalue()
        elif isinstance(csv_source, str):
            content = csv_source
        else:
            raise ValueError(f"Unsupported CSV source type: {type(csv_source)}")

        # Check for synthetic header markers
        lines = content.splitlines()
        clean_lines = []
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("#"):
                if "synthetic" in line_stripped.lower():
                    is_synthetic = True
                continue
            if line_stripped:
                clean_lines.append(line_stripped)

        if not clean_lines:
            report = DataQualityReport(
                record_count=0,
                valid_count=0,
                warnings=["Empty sensor CSV provided."],
                is_synthetic_fixture=is_synthetic,
            )
            return [], report

        reader = csv.DictReader(clean_lines)
        if reader.fieldnames is None:
            report = DataQualityReport(
                record_count=0,
                valid_count=0,
                warnings=["Failed to parse CSV header."],
                is_synthetic_fixture=is_synthetic,
            )
            return [], report

        fieldnames = {f.strip() for f in reader.fieldnames if f}
        missing_required = REQUIRED_CSV_COLUMNS - fieldnames
        if missing_required:
            missing_fields.extend(sorted(list(missing_required)))
            warnings.append(f"CSV missing mandatory columns: {missing_fields}")
            report = DataQualityReport(
                record_count=0,
                valid_count=0,
                missing_fields=missing_fields,
                warnings=warnings,
                is_synthetic_fixture=is_synthetic,
            )
            return [], report

        parsed_records: List[Tuple[datetime, SensorRecord, int]] = []
        total_rows = 0

        for row_idx, row in enumerate(reader, start=1):
            total_rows += 1
            row_clean = {k.strip(): v.strip() for k, v in row.items() if k is not None and v is not None}

            # Check timestamp format
            ts_raw = row_clean.get("timestamp")
            if not ts_raw:
                warnings.append(f"Row {row_idx}: Missing timestamp.")
                continue

            try:
                # Support ISO 8601 parsing
                ts_dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                if ts_dt.tzinfo is None:
                    # Treat naive timestamps as UTC per convention
                    ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            except Exception as e:
                warnings.append(f"Row {row_idx}: Unparseable timestamp '{ts_raw}' ({e}).")
                continue

            plot_id = row_clean.get("plot_id")
            if not plot_id:
                warnings.append(f"Row {row_idx}: Missing plot_id.")
                continue

            # Parse numeric variables
            try:
                temp_c = float(row_clean.get("air_temperature_c", 0.0))
                rh_pct = float(row_clean.get("relative_humidity_pct", 0.0))
                soil_m = float(row_clean.get("soil_moisture_pct", 0.0))
            except ValueError as e:
                warnings.append(f"Row {row_idx}: Non-numeric value in mandatory metrics ({e}).")
                continue

            # Range plausibility checks
            if temp_c < self.temp_min_c or temp_c > self.temp_max_c:
                warnings.append(
                    f"Row {row_idx}: Air temperature {temp_c}°C outside plausible range [{self.temp_min_c}, {self.temp_max_c}]."
                )
            if rh_pct < 0.0 or rh_pct > 100.0:
                warnings.append(
                    f"Row {row_idx}: Relative humidity {rh_pct}% outside valid bounds [0, 100]."
                )
                continue
            if soil_m < 0.0 or soil_m > 100.0:
                warnings.append(
                    f"Row {row_idx}: Soil moisture {soil_m}% outside valid bounds [0, 100]."
                )
                continue

            # Optional fields
            rainfall_mm = None
            if row_clean.get("rainfall_mm"):
                try:
                    val = float(row_clean["rainfall_mm"])
                    if val < 0.0:
                        warnings.append(f"Row {row_idx}: Negative rainfall {val} mm flagged.")
                    rainfall_mm = max(0.0, val)
                except ValueError:
                    warnings.append(f"Row {row_idx}: Invalid rainfall value '{row_clean['rainfall_mm']}'.")

            leaf_wetness = None
            if row_clean.get("leaf_wetness_hours"):
                try:
                    val = float(row_clean["leaf_wetness_hours"])
                    if val < 0.0:
                        warnings.append(f"Row {row_idx}: Negative leaf wetness {val} hours flagged.")
                    leaf_wetness = max(0.0, val)
                except ValueError:
                    warnings.append(f"Row {row_idx}: Invalid leaf_wetness_hours value.")

            record = SensorRecord(
                timestamp=ts_dt,
                plot_id=plot_id,
                tree_id=row_clean.get("tree_id") or None,
                crop=row_clean.get("crop", "mango"),
                air_temperature_c=temp_c,
                relative_humidity_pct=rh_pct,
                soil_moisture_pct=soil_m,
                rainfall_mm=rainfall_mm,
                leaf_wetness_hours=leaf_wetness,
                sensor_id=row_clean.get("sensor_id") or None,
                quality_flag=row_clean.get("quality_flag", "valid"),
            )
            parsed_records.append((ts_dt, record, row_idx))

        if not parsed_records:
            report = DataQualityReport(
                record_count=total_rows,
                valid_count=0,
                warnings=warnings,
                is_synthetic_fixture=is_synthetic,
            )
            return [], report

        # Check for out-of-order timestamps
        is_sorted = all(parsed_records[i][0] <= parsed_records[i + 1][0] for i in range(len(parsed_records) - 1))
        if not is_sorted:
            warnings.append("Sensor records were not in chronological order; sorting automatically.")
            parsed_records.sort(key=lambda x: x[0])

        # Check for duplicate timestamps per plot
        seen_keys = set()
        deduped_records: List[SensorRecord] = []
        for ts, rec, row_num in parsed_records:
            key = (rec.plot_id, ts.isoformat())
            if key in seen_keys:
                warnings.append(
                    f"Duplicate reading detected for plot '{rec.plot_id}' at {ts.isoformat()} (row {row_num}). Preserving first occurrence."
                )
                continue
            seen_keys.add(key)
            deduped_records.append(rec)

        # Check sampling frequency and gaps
        dur_hours = 0.0
        median_interval_min = None
        if len(deduped_records) >= 2:
            dur_seconds = (deduped_records[-1].timestamp - deduped_records[0].timestamp).total_seconds()
            dur_hours = round(dur_seconds / 3600.0, 2)

            intervals = [
                (deduped_records[i + 1].timestamp - deduped_records[i].timestamp).total_seconds() / 60.0
                for i in range(len(deduped_records) - 1)
            ]
            median_interval_min = round(float(np.median(intervals)), 1)

            # Flag any interval larger than max gap
            for i, interval_min in enumerate(intervals):
                gap_hrs = interval_min / 60.0
                if gap_hrs > self.max_sampling_gap_hours:
                    t_start = deduped_records[i].timestamp.isoformat()
                    t_end = deduped_records[i + 1].timestamp.isoformat()
                    warnings.append(
                        f"Data gap detected: {gap_hrs:.1f} hours between {t_start} and {t_end}."
                    )

        report = DataQualityReport(
            record_count=total_rows,
            valid_count=len(deduped_records),
            earliest_timestamp=deduped_records[0].timestamp.isoformat() if deduped_records else None,
            latest_timestamp=deduped_records[-1].timestamp.isoformat() if deduped_records else None,
            duration_hours=dur_hours,
            sampling_interval_minutes=median_interval_min,
            missing_fields=missing_fields,
            warnings=warnings,
            is_synthetic_fixture=is_synthetic,
        )

        return deduped_records, report
