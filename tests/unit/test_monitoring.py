"""Unit tests for the observability and monitoring integrations."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, app_state
from src.api.metrics import ANOMALIES_DETECTED, PREDICTION_REQUESTS
from src.monitoring.drift import generate_drift_report

client = TestClient(app)



def test_prometheus_metrics_endpoint() -> None:
    """Test that the /metrics endpoint exposes expected Prometheus counters."""
    # Ensure model is mocked so we can hit /predict
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_model.score_samples.return_value = [0.99]
    app_state["model"] = mock_model
    app_state["model_version"] = "test-monitoring"

    # Snapshot counter values before the request
    requests_before = PREDICTION_REQUESTS._value.get()
    anomalies_before = ANOMALIES_DETECTED._value.get()

    # Send a request to trigger metric increments
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
            }
        ]
    }
    client.post("/predict", json=payload)

    # Fetch metrics endpoint
    response = client.get("/metrics")
    assert response.status_code == 200

    metrics_text = response.text
    # Verify our custom metrics are registered and appear in output
    assert "anomaly_prediction_requests_total" in metrics_text
    assert "anomalies_detected_total" in metrics_text
    assert "anomaly_score_distribution" in metrics_text

    # Verify counters were incremented compared to before the request
    assert PREDICTION_REQUESTS._value.get() == requests_before + 1.0
    # Mock predicted [1] → is_anomaly = True, so anomalies counter should increment by 1
    assert ANOMALIES_DETECTED._value.get() == anomalies_before + 1.0


def test_evidently_drift_report(tmp_path: Path) -> None:
    """Test the generation of a Data Drift report using Evidently AI."""
    np.random.seed(42)

    # Base reference data (e.g. from training)
    ref_df = pd.DataFrame(
        {
            "close": np.random.normal(50000, 1000, 100),
            "volume": np.random.normal(2000, 500, 100),
        }
    )

    # Current inference data (with slight drift in mean)
    curr_df = pd.DataFrame(
        {
            "close": np.random.normal(55000, 1000, 100),
            "volume": np.random.normal(1500, 500, 100),
        }
    )

    out_dir = tmp_path / "reports"
    report_file = generate_drift_report(
        reference_df=ref_df,
        current_df=curr_df,
        report_name="test_drift",
        output_dir=out_dir,
    )

    assert report_file.exists()
    assert report_file.suffix == ".html"
    assert "test_drift.html" == report_file.name

    # Check if empty dataframe raises ValueError
    with pytest.raises(ValueError):
        generate_drift_report(
            reference_df=pd.DataFrame(),
            current_df=curr_df,
            output_dir=out_dir,
        )
