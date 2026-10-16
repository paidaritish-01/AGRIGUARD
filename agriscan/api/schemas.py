"""
Pydantic schemas for AgriScan REST API requests and responses.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
    model_version: str = "1.0.0"
    device: str = "cpu"
    cuda_available: bool = False
    close_up_model_loaded: bool = False
    aerial_model_loaded: bool = False
    is_demo_mode: bool = False


class ModelInfoResponse(BaseModel):
    service_name: str = "AgriScan Crop Health Monitoring API"
    model_version: str = "1.0.0"
    supported_analysis_modes: List[str] = ["close_up", "aerial"]
    supported_crops: List[str] = ["mango", "tomato", "corn", "wheat", "apple", "general_crop"]
    close_up_classes: List[str] = Field(default_factory=list)
    aerial_classes: List[str] = Field(default_factory=list)
    mango_classes: List[str] = Field(
        default_factory=lambda: ["healthy", "anthracnose", "powdery_mildew", "turning_brown", "gall_midge"]
    )
    sensor_risk_rules: List[str] = Field(
        default_factory=lambda: [
            "Mango Anthracnose (Colletotrichum gloeosporioides) - Dodd et al. (1992)",
            "Mango Powdery Mildew (Oidium mangiferae) - Palti et al. (1974)",
            "General Crop Heat Stress and Soil Moisture Deficit",
        ]
    )
    default_abstention_threshold: float = 0.70
    score_calibration: Dict[str, Any] = Field(default_factory=dict)
    operational_limitations: List[str] = Field(default_factory=list)


class GPSMetadataSchema(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_m: Optional[float] = None
    has_gps: bool = False
    note: str


class BoundingBoxSchema(BaseModel):
    bbox: List[float] = Field(description="[x_min, y_min, x_max, y_max] in image pixel coordinates")
    confidence: float
    label: str


class PredictResponse(BaseModel):
    image_id: str
    analysis_mode: Literal["close_up", "aerial"]
    model_version: str
    supported_crop: str
    predicted_label: Optional[str] = None
    model_score: float
    score_type: str
    status: Literal["success", "uncertain", "unsupported", "failed", "model_unavailable", "not_assessed"]
    message: str
    bounding_boxes: Optional[List[BoundingBoxSchema]] = None
    gps_metadata: GPSMetadataSchema
    processing_time_ms: float
    is_demo_mode: bool
    relevant_limitations: List[str]


class SensorUploadResponse(BaseModel):
    status: str = "success"
    filename: str
    saved_path: str
    valid_records: int
    data_quality: Dict[str, Any]
    message: str


class SensorAssessRequest(BaseModel):
    plot_id: str = Field(..., description="Orchard plot identifier to query")
    target_timestamp: Optional[str] = Field(None, description="Anchor observation timestamp (ISO 8601 UTC)")
    window_hours: float = Field(72.0, description="Preceding time window in hours (e.g. 24, 48, 72)")
    sensor_history_file: Optional[str] = Field(None, description="Optional filename of previously uploaded sensor CSV")
    records: Optional[List[Dict[str, Any]]] = Field(None, description="Optional inline sensor observation records")


class CombinedAssessmentResponse(BaseModel):
    """
    Contract for multi-source crop health assessment adhering to Section 9.
    Separates visible image classification, sensor data quality, observations,
    and synthesized concordance.
    """
    image_id: str
    crop: str
    assessment_status: Literal["completed", "partial_evidence", "insufficient_data"]
    image_analysis: Dict[str, Any]
    sensor_analysis: Dict[str, Any]
    combined_assessment: Dict[str, Any]
    limitations: List[str]
    processing_time_ms: Optional[float] = None

