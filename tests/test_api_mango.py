"""
Integration tests for Mango AI endpoints in FastAPI REST service:
- POST /sensors/upload
- POST /sensors/assess
- POST /assess (combined multi-modal assessment)
- GET /model-info
"""

import io
from PIL import Image
import pytest
from httpx import AsyncClient, ASGITransport

from agriscan.api.main import app


SAMPLE_CSV = """# SYNTHETIC TEST FIXTURE
timestamp,plot_id,crop,air_temperature_c,relative_humidity_pct,soil_moisture_pct,rainfall_mm,leaf_wetness_hours
2026-10-10T00:00:00Z,plot_test_01,mango,26.0,90.0,42.0,1.0,2.0
2026-10-10T02:00:00Z,plot_test_01,mango,25.5,91.0,42.0,0.5,2.0
2026-10-10T04:00:00Z,plot_test_01,mango,25.0,92.0,42.5,0.0,2.0
2026-10-10T06:00:00Z,plot_test_01,mango,25.5,91.5,42.5,0.0,2.0
2026-10-10T08:00:00Z,plot_test_01,mango,27.0,88.0,42.0,0.0,1.5
2026-10-10T10:00:00Z,plot_test_01,mango,28.0,86.0,41.5,0.0,1.0
2026-10-10T12:00:00Z,plot_test_01,mango,28.5,85.5,41.5,0.0,1.0
"""


@pytest.mark.anyio
async def test_sensor_upload_valid_csv():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("test_upload.csv", SAMPLE_CSV.encode("utf-8"), "text/csv")}
        resp = await client.post("/sensors/upload", files=files)

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["valid_records"] == 7
        assert "filename" in data


@pytest.mark.anyio
async def test_sensor_upload_invalid_extension():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("test_upload.txt", b"not,a,csv", "text/plain")}
        resp = await client.post("/sensors/upload", files=files)
        assert resp.status_code == 400
        assert "must have a .csv extension" in resp.json()["detail"]


@pytest.mark.anyio
async def test_sensor_assess_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "plot_id": "plot_mango_01",
            "window_hours": 72.0,
            "target_timestamp": "2026-10-12T22:00:00Z",
        }
        resp = await client.post("/sensors/assess", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in {"assessed", "insufficient_data"}
        assert "risk_assessment" in data


@pytest.mark.anyio
async def test_assess_combined_sensor_only():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        data = {
            "crop": "mango",
            "plot_id": "plot_mango_01",
            "timestamp": "2026-10-12T22:00:00Z",
            "window_hours": 72.0,
        }
        resp = await client.post("/assess", data=data)
        assert resp.status_code == 200
        result = resp.json()
        assert result["crop"] == "mango"
        assert result["assessment_status"] in {"partial_evidence", "completed"}
        assert result["image_analysis"]["status"] == "not_assessed"
        assert result["sensor_analysis"]["status"] == "assessed"


@pytest.mark.anyio
async def test_assess_combined_neither_provided():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        data = {"crop": "mango"}
        resp = await client.post("/assess", data=data)
        assert resp.status_code == 200
        result = resp.json()
        assert result["assessment_status"] == "insufficient_data"
        assert result["combined_assessment"]["evidence_concordance"] == "insufficient"


@pytest.mark.anyio
async def test_model_info_reports_mango():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert "mango" in [c.lower() for c in data["supported_crops"]]
        assert "mango_classes" in data
        assert "anthracnose" in data["mango_classes"]
        assert "sensor_risk_rules" in data
