from datetime import UTC
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from dags.ingestion_feature_pipeline import (
    AIRFLOW_AVAILABLE,
    compute_features,
    dag,
    ingest_market_data,
    validate_raw_data,
)


@pytest.mark.skipif(not AIRFLOW_AVAILABLE, reason="Apache Airflow not installed in local environment")
def test_dag_structure() -> None:
    """Verify DAG structure, task presence, and dependency graph."""
    assert dag is not None
    assert dag.dag_id == "ingestion_feature_pipeline"
    assert set(dag.task_dict.keys()) == {
        "ingest_market_data",
        "validate_raw_data",
        "compute_features",
    }

    ingest_task = dag.get_task("ingest_market_data")
    validate_task = dag.get_task("validate_raw_data")
    compute_features_task = dag.get_task("compute_features")

    assert validate_task in ingest_task.downstream_list
    assert compute_features_task in validate_task.downstream_list
    assert dag.schedule_interval == "@daily" or getattr(dag, "schedule", None) == "@daily"


@pytest.fixture
def mock_raw_dataframe() -> pd.DataFrame:
    """Generate sample market data."""
    dates = pd.date_range(start="2026-01-01", periods=10, freq="D", tz=UTC)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["BTC"] * 10,
            "open": [100.0] * 10,
            "high": [110.0] * 10,
            "low": [90.0] * 10,
            "close": np.linspace(100, 110, 10),
            "volume": [1000.0] * 10,
            "market_cap": [1000000.0] * 10,
        }
    )


def test_ingest_market_data(tmp_path: Path, mock_raw_dataframe: pd.DataFrame) -> None:
    """Test ingestion function calls connector and returns saved paths."""
    out_file = tmp_path / "btc_sample.parquet"
    mock_raw_dataframe.to_parquet(out_file, index=False)

    with patch("dags.ingestion_feature_pipeline.CoinGeckoConnector") as mock_connector_cls:
        instance = MagicMock()
        instance.fetch_data.return_value = mock_raw_dataframe
        instance.save_data.return_value = out_file
        mock_connector_cls.return_value = instance

        paths = ingest_market_data(symbols=["bitcoin"], days_back=5)

        assert len(paths) == 1
        assert paths[0] == str(out_file)
        instance.fetch_data.assert_called_once()
        instance.save_data.assert_called_once()


def test_validate_raw_data(tmp_path: Path, mock_raw_dataframe: pd.DataFrame) -> None:
    """Test raw data validation over parquet files."""
    test_file = tmp_path / "raw_data.parquet"
    mock_raw_dataframe.to_parquet(test_file, index=False)

    with patch("dags.ingestion_feature_pipeline.CoinGeckoConnector") as mock_connector_cls:
        instance = MagicMock()
        instance.validate_data.return_value = mock_raw_dataframe
        mock_connector_cls.return_value = instance

        validated = validate_raw_data(file_paths=[str(test_file)])

        assert len(validated) == 1
        assert validated[0] == str(test_file)
        instance.validate_data.assert_called_once()


def test_compute_features_pipeline(tmp_path: Path, mock_raw_dataframe: pd.DataFrame) -> None:
    """Test feature calculation from raw data and sink storage."""
    test_file = tmp_path / "raw_valid.parquet"
    mock_raw_dataframe.to_parquet(test_file, index=False)

    with patch("dags.ingestion_feature_pipeline.FeatureStoreSink") as mock_sink_cls:
        mock_sink = MagicMock()
        mock_sink.write.return_value = tmp_path / "features"
        mock_sink_cls.return_value = mock_sink

        dest = compute_features(file_paths=[str(test_file)])

        assert dest == str(tmp_path / "features")
        mock_sink.write.assert_called_once()
