"""
AgriScan Sensor Module.
Handles ingestion, validation, time-window aggregations, and literature-informed
environmental risk assessments for crop disease monitoring.
"""

from agriscan.sensors.schema import (
    SensorRecord,
    DataQualityReport,
    SensorDerivedFeatures,
    EnvironmentalRiskResult,
    SensorAnalysisResponse,
)
from agriscan.sensors.validator import SensorDataValidator
from agriscan.sensors.service import SensorHistoryService
from agriscan.sensors.risk_rules import evaluate_mango_environmental_risks

__all__ = [
    "SensorRecord",
    "DataQualityReport",
    "SensorDerivedFeatures",
    "EnvironmentalRiskResult",
    "SensorAnalysisResponse",
    "SensorDataValidator",
    "SensorHistoryService",
    "evaluate_mango_environmental_risks",
]
