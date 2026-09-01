"""Airflow DAG for scheduled cryptocurrency market data ingestion, validation, and feature computation."""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    DAG = Any  # type: ignore[misc,assignment]
    PythonOperator = Any  # type: ignore[misc,assignment]

import pandas as pd

from src.features.store import FeatureStoreSink
from src.features.transformers import (
    MomentumFeatures,
    PriceVelocityFeatures,
    TimeSeriesRollingFeatures,
    VolatilityFeatures,
)
from src.ingestion.coingecko import CoinGeckoConnector
from src.utils.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["bitcoin", "ethereum", "solana"]


def ingest_market_data(
    symbols: list[str] | None = None,
    days_back: int = 30,
    **context: Any,
) -> list[str]:
    """Ingest market data for target cryptocurrency assets.

    Args:
        symbols: List of asset symbols/slugs to ingest. Defaults to DEFAULT_SYMBOLS.
        days_back: Number of past days to query. Defaults to 30.
        **context: Airflow task execution context.

    Returns:
        list[str]: Paths to saved raw Parquet files.
    """
    settings = get_settings()
    connector = CoinGeckoConnector(settings=settings)
    target_symbols = symbols or DEFAULT_SYMBOLS

    end_date = datetime.now(UTC)
    start_date = end_date - timedelta(days=days_back)

    saved_paths: list[str] = []
    for symbol in target_symbols:
        logger.info("Ingesting market data for symbol: %s", symbol)
        df = connector.fetch_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        if not df.empty:
            out_file = connector.save_data(df)
            saved_paths.append(str(out_file))
            logger.info("Saved %d records for %s to %s", len(df), symbol, out_file)
        else:
            logger.warning("No data retrieved for %s", symbol)

    return saved_paths


def validate_raw_data(
    file_paths: list[str] | None = None,
    **context: Any,
) -> list[str]:
    """Validate ingested raw market data files against schema requirements.

    Args:
        file_paths: Optional explicit list of parquet file paths to validate.
        **context: Airflow task context (pulls from upstream XCom if file_paths is None).

    Returns:
        list[str]: List of validated file paths.

    Raises:
        ValueError: If validation fails or no files are provided.
    """
    ti = context.get("ti")
    paths = file_paths
    if not paths and ti:
        paths = ti.xcom_pull(task_ids="ingest_market_data")

    if not paths:
        settings = get_settings()
        raw_dir = Path(settings.data_raw_dir)
        paths = [str(p) for p in raw_dir.glob("*.parquet")]

    if not paths:
        raise ValueError("No raw data files found for validation.")

    settings = get_settings()
    connector = CoinGeckoConnector(settings=settings)
    validated_files: list[str] = []

    for file_str in paths:
        file_path = Path(file_str)
        if not file_path.exists():
            logger.warning("File %s does not exist, skipping validation", file_str)
            continue

        df = pd.read_parquet(file_path)
        validated_df = connector.validate_data(df)
        logger.info("Successfully validated %d rows in %s", len(validated_df), file_path.name)
        validated_files.append(str(file_path))

    return validated_files


def compute_features(
    file_paths: list[str] | None = None,
    **context: Any,
) -> str:
    """Execute feature engineering transformers and persist to the Feature Store.

    Args:
        file_paths: Optional list of validated raw files to compute features from.
        **context: Airflow task context.

    Returns:
        str: Destination directory where feature partitions were stored.

    Raises:
        ValueError: If no data is available to generate features.
    """
    ti = context.get("ti")
    paths = file_paths
    if not paths and ti:
        paths = ti.xcom_pull(task_ids="validate_raw_data")

    if not paths:
        settings = get_settings()
        raw_dir = Path(settings.data_raw_dir)
        paths = [str(p) for p in raw_dir.glob("*.parquet")]

    if not paths:
        raise ValueError("No validated data files available to compute features.")

    dfs = [pd.read_parquet(p) for p in paths if Path(p).exists()]
    if not dfs:
        raise ValueError("All raw data files were missing or empty.")

    combined_df = pd.concat(dfs, ignore_index=True)

    # Execute feature engineering pipeline in sequential order
    pipeline: list[Any] = [
        TimeSeriesRollingFeatures(),
        VolatilityFeatures(),
        MomentumFeatures(),
        PriceVelocityFeatures(),
    ]

    transformed_df = combined_df
    for step in pipeline:
        transformed_df = step.fit_transform(transformed_df)

    settings = get_settings()
    feature_sink = FeatureStoreSink(settings=settings)
    destination = feature_sink.write(transformed_df)

    logger.info(
        "Successfully wrote %d feature records to Feature Store at %s",
        len(transformed_df),
        destination,
    )
    return str(destination)


# Default arguments for DAG execution
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

if AIRFLOW_AVAILABLE:
    # DAG Definition
    dag = DAG(
        dag_id="ingestion_feature_pipeline",
        default_args=default_args,
        description="Automated daily ingestion, validation, and feature computation for anomaly detection",
        schedule="@daily",
        start_date=datetime(2026, 1, 1, tzinfo=UTC),
        catchup=False,
        tags=["ingestion", "features", "anomaly_detection"],
    )

    ingest_task = PythonOperator(
        task_id="ingest_market_data",
        python_callable=ingest_market_data,
        dag=dag,
    )

    validate_task = PythonOperator(
        task_id="validate_raw_data",
        python_callable=validate_raw_data,
        dag=dag,
    )

    compute_features_task = PythonOperator(
        task_id="compute_features",
        python_callable=compute_features,
        dag=dag,
    )

    # Define sequential task dependencies
    ingest_task >> validate_task >> compute_features_task
else:
    dag = None
    ingest_task = None
    validate_task = None
    compute_features_task = None
