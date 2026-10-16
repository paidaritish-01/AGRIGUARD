"""
FastAPI REST service for AgriScan crop health monitoring.

Provides:
- GET /health: Service health, model load status, GPU availability.
- GET /model-info: Model version, supported crops, disease classes, and limitations.
- POST /predict: Multipart image upload for close-up classification or aerial stress detection.

Adheres to:
- Startup lifespan model loading (loaded once at application startup).
- Strict input validation (file size limits, corrupt image handling, mode validation).
- Explicit status, clear wording ("suspected disease", "visible crop stress").
- Explicit demo mode reporting when no checkpoint is present.
"""

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
import torch

from agriscan.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
    SensorUploadResponse,
    SensorAssessRequest,
    CombinedAssessmentResponse,
)
from agriscan.inference.engine import AgriScanInferenceEngine, STANDARD_LIMITATIONS
from agriscan.sensors import SensorHistoryService, SensorAnalysisResponse
from agriscan.assessment import CombinedAssessmentCoordinator


# Maximum allowed upload size (25 MB)
MAX_IMAGE_FILE_SIZE_BYTES = 25 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/tiff", "image/webp"}

# Global engine reference loaded once during lifespan
inference_engine: Optional[AgriScanInferenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize model engine once at application startup.
    Supports environment variable overrides for checkpoint paths.
    """
    engine = get_engine()
    if engine.is_demo_mode:
        print("[AgriScan Startup] Notice: Checkpoint not found; running in labelled DEMO mode.")
    else:
        print("[AgriScan Startup] Close-up model loaded successfully.")

    yield
    print("[AgriScan Shutdown] Cleaning up inference engine resources.")


def get_engine() -> AgriScanInferenceEngine:
    """Helper to retrieve or lazily initialize the inference engine."""
    global inference_engine
    if inference_engine is None:
        close_up_ckpt = os.getenv("CLOSE_UP_CHECKPOINT", "checkpoints/best_model.pth")
        aerial_ckpt = os.getenv("AERIAL_CHECKPOINT", "checkpoints/aerial_model.pth")
        device_pref = os.getenv("DEVICE", "cpu")
        device = "cuda" if (device_pref != "cpu" and torch.cuda.is_available()) else "cpu"
        inference_engine = AgriScanInferenceEngine(
            close_up_checkpoint=close_up_ckpt if Path(close_up_ckpt).exists() else None,
            aerial_checkpoint=aerial_ckpt if Path(aerial_ckpt).exists() else None,
            device=device,
        )
    return inference_engine


app = FastAPI(
    title="AgriScan Crop Health Monitoring API",
    description=(
        "Dual-mode ML inference service for drone-assisted crop health monitoring. "
        "Supports close-up pathology classification and aerial canopy stress detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Check service health, model load status, and hardware acceleration."""
    engine = get_engine()
    return HealthResponse(
        status="healthy",
        model_version=engine.model_version,
        device=str(engine.device),
        cuda_available=torch.cuda.is_available(),
        close_up_model_loaded=(engine.close_up_model is not None),
        aerial_model_loaded=(
            engine.aerial_model is not None and engine.aerial_model.is_trained
        ),
        is_demo_mode=engine.is_demo_mode,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    """Retrieve active model metadata, supported classes, thresholds, and limitations."""
    engine = get_engine()

    close_up_classes = []
    if engine.close_up_model is not None:
        close_up_classes = [
            engine.close_up_model.idx_to_label[i]
            for i in sorted(engine.close_up_model.idx_to_label.keys())
        ]
    else:
        close_up_classes = [
            "tomato_healthy",
            "tomato_early_blight",
            "tomato_late_blight",
            "tomato_bacterial_spot",
        ]

    aerial_classes = [
        "healthy_canopy",
        "visible_crop_stress",
        "abnormal_canopy_void",
    ]

    return ModelInfoResponse(
        service_name="AgriScan Crop Health Monitoring API",
        model_version=engine.model_version,
        supported_analysis_modes=["close_up", "aerial"],
        supported_crops=engine.supported_crops,
        close_up_classes=close_up_classes,
        aerial_classes=aerial_classes,
        default_abstention_threshold=engine.default_abstention_threshold,
        score_calibration={
            "method": "temperature_scaling",
            "is_calibrated": getattr(engine.close_up_model, "is_calibrated", False),
            "temperature": float(
                getattr(engine.close_up_model.temperature_scaler.temperature, "item", lambda: 1.0)()
            ) if engine.close_up_model else 1.0,
        },
        operational_limitations=STANDARD_LIMITATIONS,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_crop_health(
    file: UploadFile = File(..., description="Crop image file (JPEG, PNG, TIFF)"),
    analysis_mode: str = Form("close_up", description="'close_up' or 'aerial'"),
    crop: str = Form("tomato", description="Target crop (e.g., 'tomato', 'corn', 'wheat')"),
    abstention_threshold: Optional[float] = Form(None, description="Custom confidence threshold for abstention"),
    use_calibration: bool = Form(True, description="Whether to apply temperature scaling if calibrated"),
) -> PredictResponse:
    """
    Execute crop health diagnosis on an uploaded image.
    Validates file integrity, parses EXIF metadata, and returns structured diagnosis.
    """
    engine = get_engine()

    # 1. Validate analysis mode
    mode = analysis_mode.strip().lower()
    if mode not in {"close_up", "aerial"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid analysis_mode '{analysis_mode}'. Must be 'close_up' or 'aerial'.",
        )

    # 2. Validate file size and read bytes
    content = await file.read()
    if len(content) > MAX_IMAGE_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_IMAGE_FILE_SIZE_BYTES // (1024*1024)} MB.",
        )
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    # 3. Validate image readability and decode
    try:
        pil_image = Image.open(io.BytesIO(content))
        pil_image.verify()
        # Re-open after verify() to load into memory
        pil_image = Image.open(io.BytesIO(content))
        pil_image.load()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupt or unreadable image file: {str(e)}",
        )

    # 4. Dispatch to appropriate analysis mode
    if mode == "close_up":
        result = engine.run_close_up_inference(
            image=pil_image,
            crop=crop,
            image_id=file.filename or "uploaded_image",
            abstention_threshold=abstention_threshold,
            use_calibration=use_calibration,
        )
    else:  # "aerial"
        result = engine.run_aerial_inference(
            image=pil_image,
            crop=crop,
            image_id=file.filename or "uploaded_aerial_image",
            confidence_threshold=abstention_threshold,
        )

    return PredictResponse(**result)


# Services for sensor analysis and combined assessment
sensor_service = SensorHistoryService()
combined_coordinator = CombinedAssessmentCoordinator()


@app.post("/sensors/upload", response_model=SensorUploadResponse)
async def upload_sensor_csv(
    file: UploadFile = File(..., description="Historical sensor observations CSV file"),
) -> SensorUploadResponse:
    """
    Ingest, validate, and persist a CSV containing historical microclimatic readings.
    Validates mandatory columns, checks physical plausibility, detects timestamp gaps/duplicates,
    and returns a data quality summary.
    """
    if not (file.filename and file.filename.lower().endswith(".csv")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a .csv extension.",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV file is empty (0 bytes).",
        )
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Sensor CSV exceeds maximum allowed size of 10 MB.",
        )

    try:
        csv_text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV file must be UTF-8 encoded text.",
        )

    dest_path, report, valid_count = sensor_service.save_csv(csv_text, file.filename)

    return SensorUploadResponse(
        status="success" if valid_count > 0 else "warning",
        filename=Path(dest_path).name,
        saved_path=str(dest_path),
        valid_records=valid_count,
        data_quality=report.model_dump(),
        message=f"Parsed {valid_count} valid records from {file.filename}.",
    )


@app.post("/sensors/assess", response_model=SensorAnalysisResponse)
async def assess_sensors(payload: SensorAssessRequest) -> SensorAnalysisResponse:
    """
    Query environmental sensor history for a specific orchard plot and time window.
    Derives statistical aggregations and evaluates literature-grounded disease risk rules.
    """
    resp = sensor_service.assess_sensor_history(
        plot_id=payload.plot_id,
        target_timestamp=payload.target_timestamp,
        window_hours=payload.window_hours,
        csv_file=payload.sensor_history_file,
    )
    return resp


@app.post("/assess", response_model=CombinedAssessmentResponse)
async def assess_crop_health_combined(
    file: Optional[UploadFile] = File(None, description="Optional crop leaf image (JPEG, PNG, TIFF)"),
    crop: str = Form("mango", description="Target crop (default 'mango')"),
    plot_id: Optional[str] = Form(None, description="Orchard plot identifier for sensor matching"),
    timestamp: Optional[str] = Form(None, description="Target observation timestamp (ISO 8601 UTC)"),
    window_hours: float = Form(72.0, description="Sensor history window in hours"),
    sensor_history_file: Optional[str] = Form(None, description="Optional filename of previously uploaded sensor CSV"),
    abstention_threshold: Optional[float] = Form(None, description="Custom confidence threshold for abstention"),
    use_calibration: bool = Form(True, description="Whether to apply temperature scaling if calibrated"),
) -> CombinedAssessmentResponse:
    """
    Execute combined multi-source crop health assessment.
    Integrates visual leaf classification with microclimatic sensor risk analysis.
    Handles image-only, sensor-only, and multi-source modes cleanly.
    """
    engine = get_engine()
    image_result = None
    image_id = "unspecified_image"

    # 1. Process image if provided
    if file is not None and file.filename:
        image_id = file.filename
        content = await file.read()
        if len(content) > MAX_IMAGE_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image exceeds maximum size of {MAX_IMAGE_FILE_SIZE_BYTES // (1024*1024)} MB.",
            )
        if len(content) > 0:
            try:
                pil_image = Image.open(io.BytesIO(content))
                pil_image.verify()
                pil_image = Image.open(io.BytesIO(content))
                pil_image.load()
                image_result = engine.run_close_up_inference(
                    image=pil_image,
                    crop=crop,
                    image_id=image_id,
                    abstention_threshold=abstention_threshold,
                    use_calibration=use_calibration,
                )
            except Exception as e:
                image_result = {
                    "image_id": image_id,
                    "analysis_mode": "close_up",
                    "model_version": engine.model_version,
                    "supported_crop": crop,
                    "predicted_label": None,
                    "model_score": 0.0,
                    "score_type": "not_applicable",
                    "status": "failed",
                    "message": f"Corrupt or unreadable image file: {str(e)}",
                    "bounding_boxes": None,
                    "gps_metadata": {"has_gps": False, "note": "Failed image read"},
                    "processing_time_ms": 0.0,
                    "is_demo_mode": False,
                    "relevant_limitations": STANDARD_LIMITATIONS,
                }

    # 2. Process sensor history if plot_id provided
    sensor_result = None
    if plot_id:
        sensor_result = sensor_service.assess_sensor_history(
            plot_id=plot_id,
            target_timestamp=timestamp,
            window_hours=window_hours,
            csv_file=sensor_history_file,
        )

    # 3. Synthesize combined assessment
    combined = combined_coordinator.synthesize(
        image_id=image_id,
        crop=crop,
        image_result=image_result,
        sensor_result=sensor_result,
    )

    return CombinedAssessmentResponse(**combined)
