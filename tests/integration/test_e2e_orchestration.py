"""End-to-end integration tests for ML orchestration pipelines."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import mlflow
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from dags.ingestion_feature_pipeline import (
    compute_features,
    ingest_market_data,
    validate_raw_data,
)
from dags.model_retraining_pipeline import (
    calculate_thresholds,
    fetch_training_data,
    train_anomaly_model,
    validate_and_register_model,
)
from src.features.models import FeatureSetRecord
from src.utils.config import get_settings


@pytest.fixture(autouse=True)
def setup_e2e_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Set up temporary isolated environment variables for E2E testing."""
    raw_dir = tmp_path / "data" / "raw"
    features_dir = tmp_path / "data" / "features"
    processed_dir = tmp_path / "data" / "processed"
    models_dir = tmp_path / "models"

    for d in (raw_dir, features_dir, processed_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DATA_RAW_DIR", str(raw_dir))
    monkeypatch.setenv("DATA_FEATURES_DIR", str(features_dir))
    monkeypatch.setenv("DATA_PROCESSED_DIR", str(processed_dir))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow_e2e.db")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "e2e_anomaly_test")

    # Clear config cache so new env vars are loaded by Pydantic
    get_settings.cache_clear()


def test_end_to_end_orchestration_pipeline() -> None:
    """Test full E2E orchestration from ingestion to model registry promotion."""
    settings = get_settings()

    # 1. Setup MLflow Tracking (simulating Local/SQLite Tracking Server)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    # 2. Mock Market Data API
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    # Generate 100 days of history
    times = [now_ms - i * 86400000 for i in range(100)][::-1]

    mock_response = {
        "prices": [[t, 50000.0 + i * 10] for i, t in enumerate(times)],
        "market_caps": [[t, 1000000.0 + i * 100] for i, t in enumerate(times)],
        "total_volumes": [[t, 1000.0 + i * 5] for i, t in enumerate(times)],
    }

    with patch("src.ingestion.coingecko.CoinGeckoConnector._make_request", return_value=mock_response):

        # ==========================================
        # FASE 1: Ingesta y Validación (DAG 1)
        # ==========================================
        raw_paths = ingest_market_data(symbols=["bitcoin"], days_back=100)
        assert len(raw_paths) == 1
        assert Path(raw_paths[0]).exists()

        valid_paths = validate_raw_data(file_paths=raw_paths)
        assert len(valid_paths) == 1
        assert Path(valid_paths[0]).exists()

        # ==========================================
        # FASE 2: Transformación de Features (DAG 1)
        # ==========================================
        feature_dest_dir = compute_features(file_paths=valid_paths)
        assert Path(feature_dest_dir).exists()

        # Healthcheck: Verify feature dataset schema
        feature_files = list(Path(feature_dest_dir).rglob("*.parquet"))
        assert len(feature_files) > 0

        features_df = pd.read_parquet(feature_dest_dir)
        assert not features_df.empty

        # Pydantic schema validation (nan replaced with None for Optionals)
        for _, row in features_df.replace({np.nan: None}).iterrows():
            try:
                FeatureSetRecord(**row.to_dict())
            except ValidationError as e:
                pytest.fail(f"End-to-End Feature schema validation failed: {e}")

        # ==========================================
        # FASE 3: Re-entrenamiento (DAG 2)
        # ==========================================
        train_data_path = fetch_training_data(feature_dir=feature_dest_dir)
        assert Path(train_data_path).exists()

        model_meta = train_anomaly_model(
            dataset_path=train_data_path,
            contamination=0.05,
        )
        assert Path(model_meta["model_path"]).exists()
        assert Path(model_meta["scores_path"]).exists()

        # ==========================================
        # FASE 4: Cálculo de Umbrales (DAG 2)
        # ==========================================
        thresholds = calculate_thresholds(scores_path=model_meta["scores_path"])
        assert "selected_threshold" in thresholds
        assert "threshold_evt" in thresholds

        # ==========================================
        # FASE 5: Validación y Registro (DAG 2)
        # ==========================================
        registry_result = validate_and_register_model(
            model_path=model_meta["model_path"],
            thresholds=thresholds,
            registered_model_name="e2e-anomaly-detector",
        )

        assert registry_result["status"] == "promoted"
        assert registry_result["registered_model_name"] == "e2e-anomaly-detector"
        prod_model_path = Path(registry_result["production_model_path"])
        assert prod_model_path.exists()
        assert prod_model_path.name == "production_model.pkl"

        # Healthcheck: Verify MLflow actually logged the model
        client = mlflow.tracking.MlflowClient()
        registered_models = client.search_registered_models()
        model_names = [rm.name for rm in registered_models]
        assert "e2e-anomaly-detector" in model_names
