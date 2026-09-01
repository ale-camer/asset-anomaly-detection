from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from dags.model_retraining_pipeline import (
    AIRFLOW_AVAILABLE,
    calculate_thresholds,
    dag,
    fetch_training_data,
    train_anomaly_model,
    validate_and_register_model,
)
from src.models.baseline import IsolationForestDetector
from src.models.registry import save_model_artifact


@pytest.mark.skipif(not AIRFLOW_AVAILABLE, reason="Apache Airflow not installed in local environment")
def test_retraining_dag_structure() -> None:
    """Verify retraining DAG structure and task dependency flow."""
    assert dag is not None
    assert dag.dag_id == "model_retraining_pipeline"
    expected_tasks = {
        "fetch_training_data",
        "train_anomaly_model",
        "calculate_thresholds",
        "validate_and_register_model",
    }
    assert set(dag.task_dict.keys()) == expected_tasks

    fetch_task = dag.get_task("fetch_training_data")
    train_task = dag.get_task("train_anomaly_model")
    calc_task = dag.get_task("calculate_thresholds")
    val_task = dag.get_task("validate_and_register_model")

    assert train_task in fetch_task.downstream_list
    assert calc_task in train_task.downstream_list
    assert val_task in calc_task.downstream_list


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Create sample feature dataset for retraining tests."""
    np.random.seed(42)
    n_samples = 50
    return pd.DataFrame(
        {
            "close": np.random.uniform(50000, 60000, n_samples),
            "volume": np.random.uniform(1000, 5000, n_samples),
            "close_rolling_mean_7": np.random.uniform(50000, 60000, n_samples),
            "close_rolling_std_7": np.random.uniform(100, 500, n_samples),
            "volatility_std_7": np.random.uniform(0.01, 0.05, n_samples),
            "momentum_rsi_14": np.random.uniform(30, 70, n_samples),
            "price_velocity_roc_1": np.random.uniform(-0.05, 0.05, n_samples),
        }
    )


def test_fetch_training_data(tmp_path: Path, sample_feature_df: pd.DataFrame) -> None:
    """Test consolidating partition files into a single training parquet file."""
    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True)
    file1 = features_dir / "part1.parquet"
    file2 = features_dir / "part2.parquet"
    sample_feature_df.to_parquet(file1, index=False)
    sample_feature_df.to_parquet(file2, index=False)

    out_file = tmp_path / "training_output.parquet"
    result = fetch_training_data(feature_dir=features_dir, output_path=out_file)

    assert result == str(out_file)
    assert out_file.exists()
    consolidated_df = pd.read_parquet(out_file)
    assert len(consolidated_df) == len(sample_feature_df) * 2


def test_train_anomaly_model(tmp_path: Path, sample_feature_df: pd.DataFrame) -> None:
    """Test training anomaly model and saving candidate artifacts."""
    data_file = tmp_path / "train_data.parquet"
    sample_feature_df.to_parquet(data_file, index=False)

    model_out = tmp_path / "candidate_model.pkl"
    meta = train_anomaly_model(
        dataset_path=str(data_file),
        model_output_path=str(model_out),
        contamination=0.05,
    )

    assert meta["model_path"] == str(model_out)
    assert Path(meta["model_path"]).exists()
    assert Path(meta["scores_path"]).exists()
    assert meta["n_samples"] == len(sample_feature_df)


def test_calculate_thresholds(tmp_path: Path) -> None:
    """Test calculation of dynamic thresholds from saved scores array."""
    scores = np.random.exponential(scale=0.5, size=100)
    scores_file = tmp_path / "model.scores.npy"
    np.save(scores_file, scores)

    thresholds = calculate_thresholds(scores_path=str(scores_file))

    assert "threshold_percentile_95" in thresholds
    assert "threshold_percentile_99" in thresholds
    assert "threshold_std_3" in thresholds
    assert "threshold_evt" in thresholds
    assert "selected_threshold" in thresholds
    assert thresholds["selected_threshold"] > 0


def test_validate_and_register_model(tmp_path: Path, sample_feature_df: pd.DataFrame) -> None:
    """Test validation and promotion to MLflow Model Registry."""
    detector = IsolationForestDetector()
    detector.fit(sample_feature_df)

    candidate_path = tmp_path / "candidate_detector.pkl"
    save_model_artifact(detector, candidate_path)

    thresholds = {"selected_threshold": 0.75}

    with patch("dags.model_retraining_pipeline.log_model_to_mlflow") as mock_log:
        mock_info = MagicMock()
        mock_info.model_uri = "models:/asset-anomaly-detector/1"
        mock_log.return_value = mock_info

        summary = validate_and_register_model(
            model_path=str(candidate_path),
            thresholds=thresholds,
            registered_model_name="asset-anomaly-detector",
        )

        assert summary["status"] == "promoted"
        assert summary["registered_model_name"] == "asset-anomaly-detector"
        assert summary["thresholds"] == thresholds
        mock_log.assert_called_once()
