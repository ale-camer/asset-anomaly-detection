"""Unit tests for the Streamlit UI module and API client."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from src.ui.api_client import check_api_health, predict_anomalies
from src.ui.components import (
    calculate_alert_severity,
    format_timeseries_data,
    generate_demo_timeseries,
    render_drift_report_html,
)


# ==========================================
# API Client Tests
# ==========================================
def test_check_api_health_success() -> None:
    """Test health check when FastAPI responds with 200 OK."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok", "model_loaded": True}

    with patch("httpx.Client.get", return_value=mock_response):
        result = check_api_health("http://localhost:8000")
        assert result["status"] == "online"
        assert result["model_loaded"] is True


def test_check_api_health_degraded() -> None:
    """Test health check when model is not loaded (503 Service Unavailable)."""
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.json.return_value = {"status": "unavailable", "model_loaded": False, "detail": "No model"}

    with patch("httpx.Client.get", return_value=mock_response):
        result = check_api_health("http://localhost:8000")
        assert result["status"] == "degraded"
        assert result["model_loaded"] is False


def test_check_api_health_connection_error() -> None:
    """Test health check when the server is offline or unreachable."""
    with patch("httpx.Client.get", side_effect=httpx.ConnectError("Connection refused")):
        result = check_api_health("http://localhost:8000")
        assert result["status"] == "offline"
        assert result["model_loaded"] is False


def test_check_api_health_timeout() -> None:
    """Test health check when request times out."""
    with patch("httpx.Client.get", side_effect=httpx.TimeoutException("Timed out")):
        result = check_api_health("http://localhost:8000")
        assert result["status"] == "timeout"
        assert result["model_loaded"] is False


def test_predict_anomalies_success() -> None:
    """Test sending feature vector to /predict endpoint."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    expected_data = {
        "predictions": [{"is_anomaly": True, "anomaly_score": 0.88}],
        "model_version": "v1",
    }
    mock_response.json.return_value = expected_data

    features = [{"close": 60000.0, "volume": 1500.0}]
    with patch("httpx.Client.post", return_value=mock_response):
        result = predict_anomalies(features=features, base_url="http://localhost:8000")
        assert result == expected_data
        assert result["predictions"][0]["is_anomaly"] is True


def test_predict_anomalies_empty_features() -> None:
    """Test error when empty features list is provided."""
    with pytest.raises(ValueError, match="Features list cannot be empty"):
        predict_anomalies(features=[], base_url="http://localhost:8000")


def test_predict_anomalies_http_error() -> None:
    """Test that HTTP status errors are properly raised."""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Bad Request", request=MagicMock(), response=mock_response
    )

    with patch("httpx.Client.post", return_value=mock_response):
        with pytest.raises(httpx.HTTPStatusError):
            predict_anomalies(features=[{"close": 100.0}], base_url="http://localhost:8000")


# ==========================================
# Components & Alerting Logic Tests
# ==========================================
@pytest.mark.parametrize(
    "score,threshold,expected_level",
    [
        (0.20, 0.50, "Low"),
        (0.55, 0.50, "Medium"),
        (0.70, 0.50, "High"),
        (0.95, 0.50, "Critical"),
    ],
)
def test_calculate_alert_severity(score: float, threshold: float, expected_level: str) -> None:
    """Test alert level categorizations against configured thresholds."""
    alert = calculate_alert_severity(anomaly_score=score, threshold=threshold)
    assert alert["level"] == expected_level
    assert "status" in alert
    assert "badge" in alert
    assert "description" in alert


def test_format_timeseries_data_valid() -> None:
    """Test formatting valid DataFrame with timestamp and close price."""
    df = pd.DataFrame({
        "timestamp": ["2026-01-01 00:00:00", "2026-01-01 01:00:00"],
        "close": [50000.0, 51000.0],
    })
    result = format_timeseries_data(df)
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
    assert "anomaly_score" in result.columns
    assert "is_anomaly" in result.columns


def test_format_timeseries_data_missing_close() -> None:
    """Test ValueError when 'close' column is missing."""
    df = pd.DataFrame({"open": [50000.0], "volume": [100.0]})
    with pytest.raises(ValueError, match="DataFrame must contain a 'close' price column"):
        format_timeseries_data(df)


def test_format_timeseries_data_empty() -> None:
    """Test ValueError when DataFrame is empty."""
    with pytest.raises(ValueError, match="Cannot format an empty DataFrame"):
        format_timeseries_data(pd.DataFrame())


def test_render_drift_report_html(tmp_path: Path) -> None:
    """Test reading HTML drift report from disk."""
    # Test non-existent file
    missing_path = tmp_path / "non_existent.html"
    assert render_drift_report_html(missing_path) is None

    # Test existing HTML file
    sample_file = tmp_path / "report.html"
    sample_content = "<html><body>Evidently Report</body></html>"
    sample_file.write_text(sample_content, encoding="utf-8")

    loaded_html = render_drift_report_html(sample_file)
    assert loaded_html == sample_content


def test_generate_demo_timeseries() -> None:
    """Test synthetic demo dataset generation."""
    df = generate_demo_timeseries(n_samples=50)
    assert len(df) == 50
    assert "close" in df.columns
    assert "timestamp" in df.columns
    assert "anomaly_score" in df.columns
    assert "is_anomaly" in df.columns
    assert df["is_anomaly"].sum() > 0
