"""
Integration tests for FastAPI endpoints (/health, /model-info, /predict).
"""

import io
from PIL import Image
import pytest
from httpx import AsyncClient, ASGITransport

from agriscan.api.main import app


@pytest.mark.anyio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in {"healthy", "unhealthy"}
        assert "model_version" in data
        assert "device" in data


@pytest.mark.anyio
async def test_model_info_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert "supported_analysis_modes" in data
        assert "supported_crops" in data
        assert "operational_limitations" in data


@pytest.mark.anyio
async def test_predict_endpoint_close_up():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create a valid test image
        img = Image.new("RGB", (224, 224), color=(50, 160, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        files = {"file": ("test_leaf.jpg", buf, "image/jpeg")}
        data = {"analysis_mode": "close_up", "crop": "tomato"}

        resp = await client.post("/predict", files=files, data=data)
        assert resp.status_code == 200
        result = resp.json()
        assert result["analysis_mode"] == "close_up"
        assert result["status"] in {"success", "uncertain", "unsupported"}
        assert "gps_metadata" in result
        assert "relevant_limitations" in result


@pytest.mark.anyio
async def test_predict_endpoint_invalid_mode():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        img = Image.new("RGB", (100, 100), color=(50, 50, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        files = {"file": ("test.jpg", buf, "image/jpeg")}
        data = {"analysis_mode": "invalid_mode_xyz", "crop": "tomato"}

        resp = await client.post("/predict", files=files, data=data)
        assert resp.status_code == 400
        assert "Invalid analysis_mode" in resp.json()["detail"]


@pytest.mark.anyio
async def test_predict_endpoint_empty_file():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        data = {"analysis_mode": "close_up", "crop": "tomato"}

        resp = await client.post("/predict", files=files, data=data)
        assert resp.status_code == 400
        assert "0 bytes" in resp.json()["detail"]
