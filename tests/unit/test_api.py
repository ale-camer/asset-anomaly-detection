"""Unit tests for the FastAPI inference service."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, app_state

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_app_state() -> Generator[None, None, None]:
    """Reset the global app state before and after each test."""
    app_state["model"] = None
    app_state["model_version"] = None
    yield
    app_state["model"] = None
    app_state["model_version"] = None


def test_health_check_no_model() -> None:
    """Test health check when model fails to load."""
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


def test_health_check_with_model() -> None:
    """Test health check when model is loaded successfully."""
    app_state["model"] = MagicMock()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint() -> None:
    """Test the /predict endpoint with a valid payload."""
    mock_model = MagicMock()
    # Mocking standard predict and score_samples arrays
    mock_model.predict.return_value = [1, 0]
    mock_model.score_samples.return_value = [0.85, 0.12]
    app_state["model"] = mock_model
    app_state["model_version"] = "v1"

    payload = {
        "features": [
            {
                "close": 50000.0,
                "volume": 1200.5,
                "close_rolling_mean_7": 49000.0,
                "close_rolling_std_7": 500.0,
                "volatility_std_7": 0.02,
                "momentum_rsi_14": 55.0,
                "price_velocity_roc_1": 0.01,
            },
            {
                "close": 51000.0,
                "volume": 900.0,
                "close_rolling_mean_7": 50000.0,
                "close_rolling_std_7": 400.0,
                "volatility_std_7": 0.01,
                "momentum_rsi_14": 45.0,
                "price_velocity_roc_1": -0.01,
            }
        ]
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["model_version"] == "v1"
    assert len(data["predictions"]) == 2

    assert data["predictions"][0]["is_anomaly"] is True
    assert data["predictions"][0]["anomaly_score"] == 0.85

    assert data["predictions"][1]["is_anomaly"] is False
    assert data["predictions"][1]["anomaly_score"] == 0.12


def test_predict_invalid_payload() -> None:
    """Test the /predict endpoint with missing required fields."""
    app_state["model"] = MagicMock()

    # Payload missing 'features' root key
    payload = {"wrong_key": []}  # type: ignore

    response = client.post("/predict", json=payload)
    # FastAPI returns 422 Unprocessable Entity for Pydantic validation errors
    assert response.status_code == 422


@patch("src.api.main.load_model_from_mlflow")
def test_app_lifespan(mock_load_model: MagicMock) -> None:
    """Test that the lifespan manager loads the model successfully."""
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    with TestClient(app):
        # Startup event fired
        assert app_state["model"] is mock_model
        assert app_state["model_version"] == "latest"

    # Shutdown event fired
    assert app_state["model"] is None
