"""
Pydantic schemas and dataclasses for environmental sensor records and analysis.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


class SensorRecord(BaseModel):
    """
    Standard schema for a single environmental sensor observation.
    Timestamps are handled in ISO 8601 with documented UTC convention.
    """
    timestamp: datetime = Field(
        ...,
        description="Observation timestamp (ISO 8601 with timezone or UTC)."
    )
    plot_id: str = Field(
        ...,
        description="Unique identifier for the orchard plot or farm sector."
    )
    tree_id: Optional[str] = Field(
        default=None,
        description="Optional tree or canopy observation identifier."
    )
    crop: str = Field(
        default="mango",
        description="Target crop species (default 'mango')."
    )
    air_temperature_c: float = Field(
        ...,
        description="Ambient air temperature in degrees Celsius (°C)."
    )
    relative_humidity_pct: float = Field(
        ...,
        description="Relative air humidity in percentage (% RH, 0-100)."
    )
    soil_moisture_pct: float = Field(
        ...,
        description="Volumetric soil water content or sensor percentage (0-100)."
    )
    rainfall_mm: Optional[float] = Field(
        default=None,
        description="Accumulated rainfall over preceding sampling interval (mm)."
    )
    leaf_wetness_hours: Optional[float] = Field(
        default=None,
        description="Leaf wetness duration in hours (0.0 to interval duration)."
    )
    sensor_id: Optional[str] = Field(
        default=None,
        description="Hardware identifier for the recording sensor node."
    )
    quality_flag: Optional[str] = Field(
        default="valid",
        description="Hardware or upstream pipeline quality flag (e.g. 'valid', 'suspect')."
    )

    @field_validator("relative_humidity_pct")
    @classmethod
    def check_rh_range(cls, v: float) -> float:
        if v < 0.0 or v > 100.0:
            raise ValueError(f"Relative humidity must be between 0 and 100%, got {v}")
        return v

    @field_validator("soil_moisture_pct")
    @classmethod
    def check_soil_moisture_range(cls, v: float) -> float:
        if v < 0.0 or v > 100.0:
            raise ValueError(f"Soil moisture percentage must be between 0 and 100%, got {v}")
        return v


class DataQualityReport(BaseModel):
    """
    Quality diagnostics and coverage summary for a collection of sensor records.
    """
    record_count: int = 0
    valid_count: int = 0
    earliest_timestamp: Optional[str] = None
    latest_timestamp: Optional[str] = None
    duration_hours: float = 0.0
    sampling_interval_minutes: Optional[float] = None
    missing_fields: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    is_synthetic_fixture: bool = False


class SensorDerivedFeatures(BaseModel):
    """
    Window-based statistical aggregations and trend features derived from raw sensor history.
    """
    window_hours: float
    measurement_count: int
    coverage_pct: float
    temperature_mean_c: Optional[float] = None
    temperature_min_c: Optional[float] = None
    temperature_max_c: Optional[float] = None
    temperature_trend_c_per_hour: Optional[float] = None
    humidity_mean_pct: Optional[float] = None
    humidity_min_pct: Optional[float] = None
    humidity_max_pct: Optional[float] = None
    humidity_trend_pct_per_hour: Optional[float] = None
    hours_high_humidity_above_85: Optional[float] = None
    hours_temp_favorable_fungal_24_30: Optional[float] = None
    soil_moisture_mean_pct: Optional[float] = None
    soil_moisture_min_pct: Optional[float] = None
    total_rainfall_mm: Optional[float] = None
    total_leaf_wetness_hours: Optional[float] = None


class EnvironmentalRiskResult(BaseModel):
    """
    Literature-grounded microclimatic favorability assessment for a specific disease or stress.
    """
    disease_or_condition: str
    pathogen_or_stressor: str
    risk_level: Literal["low", "moderate", "high", "insufficient_data", "not_assessed"]
    favorable_conditions_met: bool
    explanation: str
    required_variables: List[str]
    source_citation: str
    evidence_type: Literal["literature-informed", "locally-calibrated", "experimentally-validated"]
    assumptions_and_limitations: str


class SensorAnalysisResponse(BaseModel):
    """
    Complete output from the sensor-history analysis module.
    Strictly separates raw measurements, derived features, and literature-based risk rules.
    """
    status: Literal["assessed", "insufficient_data", "not_assessed"]
    plot_id: Optional[str] = None
    target_timestamp: Optional[str] = None
    window_hours: float = 72.0
    data_quality: DataQualityReport
    observations: List[Dict[str, Any]] = Field(default_factory=list)
    derived_features: Optional[SensorDerivedFeatures] = None
    risk_assessment: List[EnvironmentalRiskResult] = Field(default_factory=list)
    general_stress_indicators: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
