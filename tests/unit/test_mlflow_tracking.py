import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from src.models.autoencoder import AutoencoderAnomalyDetector
from src.models.baseline import IsolationForestDetector, LOFDetector


@pytest.fixture
def sample_train_data() -> np.ndarray:
    """Generate synthetic training data."""
    np.random.seed(42)
    return np.random.normal(loc=0.0, scale=1.0, size=(50, 4))


def test_autoencoder_mlflow_tracking(sample_train_data: np.ndarray) -> None:
    """Verify parameters and metrics are logged to MLflow during Autoencoder training."""
    client = MlflowClient()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        detector = AutoencoderAnomalyDetector(
            hidden_dim=8,
            latent_dim=2,
            lr=1e-3,
            epochs=3,
            batch_size=16,
            threshold_std_factor=2.5,
            random_seed=42,
        )
        detector.fit(sample_train_data)

    run_data = client.get_run(run_id).data

    # Check parameters
    assert run_data.params["model_type"] == "AutoencoderAnomalyDetector"
    assert run_data.params["hidden_dim"] == "8"
    assert run_data.params["latent_dim"] == "2"
    assert run_data.params["epochs"] == "3"
    assert run_data.params["batch_size"] == "16"
    assert run_data.params["threshold_std_factor"] == "2.5"

    # Check metrics
    assert "train_loss" in run_data.metrics
    assert "anomaly_threshold" in run_data.metrics
    assert "reconstruction_error_mean" in run_data.metrics
    assert "reconstruction_error_std" in run_data.metrics

    # Check metric history for train_loss across epochs
    metric_history = client.get_metric_history(run_id, "train_loss")
    assert len(metric_history) == 3
    assert [m.step for m in metric_history] == [0, 1, 2]


def test_isolation_forest_mlflow_tracking(sample_train_data: np.ndarray) -> None:
    """Verify parameters are logged to MLflow during IsolationForest training."""
    client = MlflowClient()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        detector = IsolationForestDetector(n_estimators=50, contamination=0.05, random_state=42)
        detector.fit(sample_train_data)

    run_data = client.get_run(run_id).data

    assert run_data.params["model_type"] == "IsolationForestDetector"
    assert run_data.params["n_estimators"] == "50"
    assert run_data.params["contamination"] == "0.05"
    assert run_data.params["random_state"] == "42"


def test_lof_mlflow_tracking(sample_train_data: np.ndarray) -> None:
    """Verify parameters are logged to MLflow during LOF training."""
    client = MlflowClient()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        detector = LOFDetector(n_neighbors=15, contamination=0.1)
        detector.fit(sample_train_data)

    run_data = client.get_run(run_id).data

    assert run_data.params["model_type"] == "LOFDetector"
    assert run_data.params["n_neighbors"] == "15"
    assert run_data.params["contamination"] == "0.1"


def test_models_fit_without_active_mlflow_run(sample_train_data: np.ndarray) -> None:
    """Verify models execute normally when no MLflow run is active."""
    assert mlflow.active_run() is None

    ae = AutoencoderAnomalyDetector(epochs=2, batch_size=16)
    ae.fit(sample_train_data)
    assert ae.is_fitted
    assert ae.predict(sample_train_data).shape == (len(sample_train_data),)

    iforest = IsolationForestDetector(n_estimators=10)
    iforest.fit(sample_train_data)
    assert iforest.is_fitted
    assert iforest.predict(sample_train_data).shape == (len(sample_train_data),)

    lof = LOFDetector(n_neighbors=10)
    lof.fit(sample_train_data)
    assert lof.is_fitted
    assert lof.predict(sample_train_data).shape == (len(sample_train_data),)
