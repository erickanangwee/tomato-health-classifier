"""
Unit tests for the FastAPI application.
Run with: pytest tests/ -v
"""
import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image


# ── Fixtures ──────────────────────────────────────────────────────────────


def make_fake_image(color=(100, 160, 80), size=(224, 224)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    """Create a test client with mocked ML components."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])                    # UNHEALTHY
    mock_model.predict_proba.return_value = np.array([[0.12, 0.88]])

    mock_scaler = MagicMock()
    mock_scaler.transform.side_effect = lambda x: x

    mock_guard = MagicMock()
    mock_guard.is_tomato.return_value = (True, 0.82)
    mock_guard.threshold = 0.65

    mock_embedder = MagicMock()
    mock_embedder.return_value = np.zeros((1, 1280), dtype=np.float32)

    with (
        patch("api.main.get_model", return_value=mock_model),
        patch("api.main.get_scaler", return_value=mock_scaler),
        patch("api.main.TomatoGuard", return_value=mock_guard),
        patch("api.main.extract_features", mock_embedder),
        patch("api.main._guard", mock_guard),
        patch("api.main._embedder_model", MagicMock()),
        patch("api.main._embedder_transform", MagicMock()),
    ):
        from api.main import app
        yield TestClient(app)


# ── Tests ─────────────────────────────────────────────────────────────────


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_classes_endpoint(client):
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert "HEALTHY" in data["classes"]
    assert "UNHEALTHY" in data["classes"]


def test_predict_valid_image(client):
    img_bytes = make_fake_image()
    response = client.post(
        "/predict",
        files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ("HEALTHY", "UNHEALTHY")
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["is_tomato"] is True


def test_predict_invalid_content_type(client):
    response = client.post(
        "/predict",
        files={"file": ("doc.pdf", b"PDF content", "application/pdf")},
    )
    assert response.status_code == 415


def test_predict_non_tomato_rejected(client):
    """When guard returns is_tomato=False, endpoint must return 422."""
    img_bytes = make_fake_image(color=(200, 100, 50))

    with patch("api.main._guard") as mock_g:
        mock_g.is_tomato.return_value = (False, 0.21)
        mock_g.threshold = 0.65
        response = client.post(
            "/predict",
            files={"file": ("not_a_leaf.jpg", img_bytes, "image/jpeg")},
        )
    assert response.status_code == 422
