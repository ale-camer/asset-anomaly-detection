"""UI helper functions, metric calculations, and alerting components."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_alert_severity(anomaly_score: float, threshold: float = 0.5) -> dict[str, Any]:
    """Calculate the alert severity level based on anomaly score and decision threshold.

    Args:
        anomaly_score: Model output score (continuous, typically 0.0 to 1.0+).
        threshold: Base decision threshold for binary anomaly classification.

    Returns:
        dict: Alert metadata containing 'level', 'status', 'color', and 'description'.
    """
    if anomaly_score < threshold:
        return {
            "level": "Low",
            "status": "Normal",
            "color": "#10B981",  # Emerald Green
            "badge": "🟢 Normal",
            "description": "Market features are within normal operational regime.",
        }
    elif anomaly_score < threshold + 0.15:
        return {
            "level": "Medium",
            "status": "Warning",
            "color": "#F59E0B",  # Amber
            "badge": "🟡 Warning",
            "description": "Mild deviations detected. Increased volatility or momentum shift.",
        }
    elif anomaly_score < threshold + 0.35:
        return {
            "level": "High",
            "status": "Elevated Risk",
            "color": "#F97316",  # Orange
            "badge": "🟠 High Risk",
            "description": "Significant anomalous behavior. High likelihood of flash movement.",
        }
    else:
        return {
            "level": "Critical",
            "status": "Severe Anomaly",
            "color": "#EF4444",  # Red
            "badge": "🔴 Critical Anomaly",
            "description": "Extreme anomaly detected! Immediate attention required.",
        }


def format_timeseries_data(df: pd.DataFrame) -> pd.DataFrame:
    """Format and validate DataFrame for time series anomaly visualization.

    Args:
        df: Raw or feature DataFrame.

    Returns:
        pd.DataFrame: Formatted DataFrame with datetime index and guaranteed columns.

    Raises:
        ValueError: If input DataFrame is empty or missing vital price columns.
    """
    if df.empty:
        raise ValueError("Cannot format an empty DataFrame.")

    clean_df = df.copy()

    # Normalize timestamp column
    if "timestamp" in clean_df.columns:
        clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"])
        clean_df = clean_df.sort_values("timestamp").reset_index(drop=True)
    elif not isinstance(clean_df.index, pd.DatetimeIndex):
        clean_df["timestamp"] = pd.date_range(end=pd.Timestamp.now(), periods=len(clean_df), freq="1h")

    # Ensure price column exists
    if "close" not in clean_df.columns:
        raise ValueError("DataFrame must contain a 'close' price column.")

    # Guarantee anomaly flag and score columns
    if "anomaly_score" not in clean_df.columns:
        clean_df["anomaly_score"] = 0.0

    if "is_anomaly" not in clean_df.columns:
        clean_df["is_anomaly"] = clean_df["anomaly_score"] > 0.5

    clean_df["is_anomaly"] = clean_df["is_anomaly"].astype(bool)
    clean_df["anomaly_score"] = clean_df["anomaly_score"].astype(float)

    return clean_df


def render_drift_report_html(report_path: str | Path) -> str | None:
    """Read Evidently HTML report from disk for embedding in Streamlit.

    Args:
        report_path: Path to the HTML report file.

    Returns:
        str | None: Raw HTML content string, or None if file does not exist.
    """
    target = Path(report_path)
    if not target.exists() or not target.is_file():
        logger.info("Drift report not found at %s", target)
        return None

    try:
        with open(target, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error("Failed to read drift report at %s: %s", target, e)
        return None


def generate_demo_timeseries(n_samples: int = 150) -> pd.DataFrame:
    """Generate realistic synthetic crypto time-series data with seeded anomalies.

    Args:
        n_samples: Number of hourly timestamps to generate.

    Returns:
        pd.DataFrame: Synthetic dataset with prices, indicators, and anomalies.
    """
    np.random.seed(42)
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq="1h")

    # Base price random walk
    returns = np.random.normal(0.0005, 0.015, n_samples)
    price = 65000.0 * np.exp(np.cumsum(returns))
    volume = np.random.uniform(500, 2500, n_samples)

    df = pd.DataFrame({"timestamp": timestamps, "close": price, "volume": volume})

    # Rolling statistics
    df["close_rolling_mean_7"] = df["close"].rolling(window=7, min_periods=1).mean()
    df["close_rolling_std_7"] = df["close"].rolling(window=7, min_periods=1).std().fillna(100.0)
    df["volatility_parkinson_14"] = np.random.uniform(0.01, 0.04, n_samples)
    df["rsi_14"] = np.random.uniform(30.0, 70.0, n_samples)
    df["price_velocity_1"] = df["close"].pct_change().fillna(0.0)

    # Base anomaly scores
    anomaly_scores = np.random.beta(a=1.5, b=6.0, size=n_samples)

    # Inject distinct anomaly spikes proportionally
    anomaly_indices = [
        int(n_samples * p)
        for p in [0.15, 0.40, 0.65, 0.85]
        if int(n_samples * p) < n_samples
    ]
    for idx in anomaly_indices:
        df.loc[idx, "close"] *= 1.08 if idx % 2 == 0 else 0.91
        df.loc[idx, "volume"] *= 3.5
        df.loc[idx, "volatility_parkinson_14"] = 0.09
        df.loc[idx, "rsi_14"] = 88.0 if idx % 2 == 0 else 14.0
        anomaly_scores[idx] = 0.82 + np.random.uniform(0.05, 0.15)

    df["anomaly_score"] = anomaly_scores
    df["is_anomaly"] = df["anomaly_score"] >= 0.60

    return df
