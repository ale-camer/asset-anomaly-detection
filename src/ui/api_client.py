"""HTTP Client for interacting with the FastAPI inference service."""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def check_api_health(base_url: str = "http://localhost:8000", timeout: float = 3.0) -> dict[str, Any]:
    """Check the health and model status of the FastAPI service.

    Args:
        base_url: Base URL of the FastAPI application.
        timeout: Request timeout in seconds.

    Returns:
        dict: Health status containing 'status', 'model_loaded', and optional 'detail'.
    """
    clean_url = base_url.rstrip("/")
    endpoint = f"{clean_url}/health"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(endpoint)
            data = response.json()
            if response.status_code == 200:
                return {
                    "status": "online",
                    "model_loaded": data.get("model_loaded", False),
                    "raw": data,
                }
            return {
                "status": "degraded",
                "model_loaded": data.get("model_loaded", False),
                "detail": data.get("detail", f"HTTP {response.status_code}"),
            }
    except httpx.ConnectError:
        logger.warning("Could not connect to FastAPI at %s", endpoint)
        return {
            "status": "offline",
            "model_loaded": False,
            "detail": f"Connection refused at {clean_url}",
        }
    except httpx.TimeoutException:
        logger.warning("Timeout connecting to FastAPI at %s", endpoint)
        return {
            "status": "timeout",
            "model_loaded": False,
            "detail": f"Request timed out after {timeout}s",
        }
    except Exception as e:
        logger.error("Unexpected error checking API health: %s", e)
        return {
            "status": "error",
            "model_loaded": False,
            "detail": str(e),
        }


def predict_anomalies(
    features: list[dict[str, float]],
    base_url: str = "http://localhost:8000",
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Send feature vector(s) to FastAPI /predict for anomaly inference.

    Args:
        features: List of feature dictionaries (each dictionary represents a record).
        base_url: Base URL of the FastAPI application.
        timeout: Request timeout in seconds.

    Returns:
        dict: Prediction response containing 'predictions' and 'model_version'.

    Raises:
        ValueError: If features list is empty.
        httpx.HTTPStatusError: If API returns an HTTP 4xx or 5xx error.
        httpx.RequestError: If connection or timeout fails.
    """
    if not features:
        raise ValueError("Features list cannot be empty for prediction.")

    clean_url = base_url.rstrip("/")
    endpoint = f"{clean_url}/predict"
    payload = {"features": features}

    with httpx.Client(timeout=timeout) as client:
        response = client.post(endpoint, json=payload)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data
