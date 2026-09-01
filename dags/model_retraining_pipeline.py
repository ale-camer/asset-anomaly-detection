"""Airflow DAG for automated model retraining, threshold calculation, and registry promotion."""

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

import numpy as np
import pandas as pd

from src.models.baseline import IsolationForestDetector
from src.models.evaluation import compute_dynamic_threshold
from src.models.registry import (
    load_model_artifact,
    log_model_to_mlflow,
    save_model_artifact,
)
from src.utils.config import get_settings

logger = logging.getLogger(__name__)

# Feature columns used for anomaly detection models
NUMERIC_FEATURE_COLS = [
    "close",
    "volume",
    "close_rolling_mean_7",
    "close_rolling_std_7",
    "volatility_std_7",
    "momentum_rsi_14",
    "price_velocity_roc_1",
]


def fetch_training_data(
    feature_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    **context: Any,
) -> str:
    """Extract and consolidate historical feature sets from Feature Store.

    Args:
        feature_dir: Directory containing partitioned feature parquet files.
        output_path: Destination path for consolidated training parquet dataset.
        **context: Airflow task execution context.

    Returns:
        str: Absolute path to consolidated training dataset.

    Raises:
        FileNotFoundError: If feature directory has no valid parquet partitions.
    """
    settings = get_settings()
    src_dir = Path(feature_dir or settings.data_features_dir)
    parquet_files = list(src_dir.rglob("*.parquet"))

    if not parquet_files:
        logger.warning("No feature files found in %s, checking data_raw_dir...", src_dir)
        raw_dir = Path(settings.data_raw_dir)
        parquet_files = list(raw_dir.rglob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet datasets found in {src_dir} or {settings.data_raw_dir}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    dataset = pd.concat(dfs, ignore_index=True)

    dest = Path(output_path or (Path(settings.data_processed_dir) / "training_features.parquet"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(dest, index=False)
    logger.info("Consolidated %d feature rows to %s", len(dataset), dest)

    return str(dest)


def train_anomaly_model(
    dataset_path: str | None = None,
    model_output_path: str | None = None,
    contamination: float = 0.05,
    **context: Any,
) -> dict[str, Any]:
    """Train candidate anomaly detection model on extracted features.

    Args:
        dataset_path: Path to consolidated training parquet dataset.
        model_output_path: Destination path to save trained candidate artifact.
        contamination: Expected anomaly contamination rate. Defaults to 0.05.
        **context: Airflow task execution context.

    Returns:
        dict[str, Any]: Metadata dictionary containing artifact paths and metrics.

    Raises:
        ValueError: If training dataset is missing or empty.
    """
    ti = context.get("ti")
    data_file = dataset_path
    if not data_file and ti:
        data_file = ti.xcom_pull(task_ids="fetch_training_data")

    if not data_file:
        settings = get_settings()
        data_file = str(Path(settings.data_processed_dir) / "training_features.parquet")

    df = pd.read_parquet(data_file)
    if df.empty:
        raise ValueError("Training dataset cannot be empty.")

    # Select existing numeric feature columns
    available_cols = [c for c in NUMERIC_FEATURE_COLS if c in df.columns]
    if not available_cols:
        # Fallback to any numeric columns present in DataFrame
        available_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not available_cols:
        raise ValueError("No numeric features available in dataset for training.")

    X = df[available_cols].fillna(0.0)

    logger.info("Training IsolationForest on %d samples with %d features", len(X), len(available_cols))
    detector = IsolationForestDetector(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        feature_cols=available_cols,
    )
    detector.fit(X)

    # Compute training anomaly scores
    scores = detector.score_samples(X)

    settings = get_settings()
    out_path = Path(model_output_path or (Path(settings.data_raw_dir).parent / "models" / "candidate_detector.pkl"))
    saved_model_path = save_model_artifact(detector, out_path)

    # Save scores array alongside model for downstream thresholding
    scores_file = out_path.with_suffix(".scores.npy")
    np.save(scores_file, scores)

    return {
        "model_path": str(saved_model_path),
        "scores_path": str(scores_file),
        "feature_cols": available_cols,
        "n_samples": int(len(X)),
    }


def calculate_thresholds(
    scores_path: str | None = None,
    **context: Any,
) -> dict[str, float]:
    """Compute optimal dynamic anomaly thresholds from model scores.

    Args:
        scores_path: Optional path to numpy scores array.
        **context: Airflow task execution context.

    Returns:
        dict[str, float]: Calculated dynamic threshold values.

    Raises:
        FileNotFoundError: If scores file cannot be located.
    """
    ti = context.get("ti")
    sc_path = scores_path
    if not sc_path and ti:
        meta = ti.xcom_pull(task_ids="train_anomaly_model")
        if isinstance(meta, dict):
            sc_path = meta.get("scores_path")

    if not sc_path or not Path(sc_path).exists():
        raise FileNotFoundError(f"Scores file not found at: {sc_path}")

    scores: np.ndarray = np.load(sc_path)

    percentile_95 = compute_dynamic_threshold(scores, method="percentile", percentile=95.0)
    percentile_99 = compute_dynamic_threshold(scores, method="percentile", percentile=99.0)
    std_thresh = compute_dynamic_threshold(scores, method="std", std_factor=3.0)
    evt_thresh = compute_dynamic_threshold(scores, method="evt", initial_quantile=0.95, risk_prob=0.01)

    thresholds = {
        "threshold_percentile_95": float(percentile_95),
        "threshold_percentile_99": float(percentile_99),
        "threshold_std_3": float(std_thresh),
        "threshold_evt": float(evt_thresh),
        "selected_threshold": float(evt_thresh),
    }

    logger.info("Calculated dynamic thresholds: %s", thresholds)
    return thresholds


def validate_and_register_model(
    model_path: str | None = None,
    thresholds: dict[str, float] | None = None,
    registered_model_name: str = "asset-anomaly-detector",
    **context: Any,
) -> dict[str, Any]:
    """Validate model and promote/register to MLflow Model Registry.

    Args:
        model_path: Path to serialized candidate detector artifact.
        thresholds: Computed threshold metrics dictionary.
        registered_model_name: MLflow registry model name. Defaults to 'asset-anomaly-detector'.
        **context: Airflow task execution context.

    Returns:
        dict[str, Any]: Summary of registered model and deployment metadata.

    Raises:
        FileNotFoundError: If candidate model artifact is missing.
    """
    ti = context.get("ti")
    m_path = model_path
    thresh_dict = thresholds

    if ti:
        if not m_path:
            meta = ti.xcom_pull(task_ids="train_anomaly_model")
            if isinstance(meta, dict):
                m_path = meta.get("model_path")
        if not thresh_dict:
            thresh_dict = ti.xcom_pull(task_ids="calculate_thresholds")

    if not m_path or not Path(m_path).exists():
        raise FileNotFoundError(f"Candidate model artifact not found at: {m_path}")

    detector = load_model_artifact(m_path)

    # Log candidate model to MLflow with model registry integration
    logger.info("Logging model to MLflow and registering as '%s'", registered_model_name)
    model_info = log_model_to_mlflow(
        model=detector,
        artifact_path="model",
        registered_model_name=registered_model_name,
    )

    # Save finalized production artifact
    settings = get_settings()
    prod_path = Path(settings.data_raw_dir).parent / "models" / "production_model.pkl"
    saved_prod = save_model_artifact(detector, prod_path)

    result_summary = {
        "status": "promoted",
        "registered_model_name": registered_model_name,
        "production_model_path": str(saved_prod),
        "model_uri": getattr(model_info, "model_uri", f"models:/{registered_model_name}/latest"),
        "thresholds": thresh_dict or {},
    }

    logger.info("Model successfully promoted: %s", result_summary)
    return result_summary


# Default execution arguments for Airflow
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
        dag_id="model_retraining_pipeline",
        default_args=default_args,
        description="Automated weekly model retraining, threshold optimization, and registry promotion",
        schedule="@weekly",
        start_date=datetime(2026, 1, 1, tzinfo=UTC),
        catchup=False,
        tags=["retraining", "models", "registry", "anomaly_detection"],
    )

    fetch_data_task = PythonOperator(
        task_id="fetch_training_data",
        python_callable=fetch_training_data,
        dag=dag,
    )

    train_model_task = PythonOperator(
        task_id="train_anomaly_model",
        python_callable=train_anomaly_model,
        dag=dag,
    )

    calculate_thresholds_task = PythonOperator(
        task_id="calculate_thresholds",
        python_callable=calculate_thresholds,
        dag=dag,
    )

    validate_and_register_task = PythonOperator(
        task_id="validate_and_register_model",
        python_callable=validate_and_register_model,
        dag=dag,
    )

    # Sequential retraining pipeline dependencies
    fetch_data_task >> train_model_task >> calculate_thresholds_task >> validate_and_register_task
else:
    dag = None
    fetch_data_task = None
    train_model_task = None
    calculate_thresholds_task = None
    validate_and_register_task = None
