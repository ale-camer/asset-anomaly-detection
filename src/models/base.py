from abc import ABC, abstractmethod
from typing import Any

import mlflow
import numpy as np
import pandas as pd


class BaseAnomalyDetector(ABC):
    """Abstract base class for asset anomaly detection models."""

    def __init__(self, feature_cols: list[str] | None = None, **kwargs: Any) -> None:
        """Initialize base anomaly detector.

        Args:
            feature_cols: Optional subset of feature column names to use when input is a DataFrame.
            **kwargs: Extra detector-specific arguments.
        """
        self.feature_cols = feature_cols
        self.is_fitted: bool = False

    def _log_mlflow_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow if an active run exists.

        Args:
            params: Dictionary of parameters to log.
        """
        try:
            if mlflow.active_run() is not None:
                filtered_params = {k: v for k, v in params.items() if v is not None}
                mlflow.log_params(filtered_params)
        except Exception:
            pass

    def _log_mlflow_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric to MLflow if an active run exists.

        Args:
            key: Metric name.
            value: Metric value.
            step: Optional step index or epoch.
        """
        try:
            if mlflow.active_run() is not None:
                mlflow.log_metric(key, float(value), step=step)
        except Exception:
            pass

    def _log_mlflow_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics to MLflow if an active run exists.

        Args:
            metrics: Dictionary of metric names and values.
            step: Optional step index or epoch.
        """
        try:
            if mlflow.active_run() is not None:
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)
        except Exception:
            pass

    def _prepare_features(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Extract and validate feature matrix from input.

        Args:
            X: Input DataFrame or numpy array.

        Returns:
            np.ndarray: Prepared 2D numpy array of float features.

        Raises:
            TypeError: If input is not a DataFrame or numpy array.
            ValueError: If input is empty or has invalid shape.
        """
        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("Input DataFrame cannot be empty.")
            if self.feature_cols:
                missing = [c for c in self.feature_cols if c not in X.columns]
                if missing:
                    raise ValueError(f"Missing required feature columns in input: {missing}")
                features = X[self.feature_cols].to_numpy(dtype=float)
            else:
                numeric_df = X.select_dtypes(include=[np.number])
                if numeric_df.empty:
                    raise ValueError("No numeric features found in input DataFrame.")
                features = numeric_df.to_numpy(dtype=float)
        elif isinstance(X, np.ndarray):
            if X.size == 0:
                raise ValueError("Input numpy array cannot be empty.")
            features = X.astype(float)
            if features.ndim == 1:
                features = features.reshape(-1, 1)
        else:
            raise TypeError("Input must be a pandas DataFrame or numpy ndarray.")

        if np.isnan(features).any():
            raise ValueError("Input features contain NaN values. Clean or impute features before modeling.")

        return np.asarray(features, dtype=float)

    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "BaseAnomalyDetector":
        """Fit anomaly detection model on training data.

        Args:
            X: Training feature matrix or DataFrame.
            y: Ignored (unsupervised models).

        Returns:
            self
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict binary anomaly labels (1 for anomaly, 0 for normal).

        Args:
            X: Feature matrix or DataFrame.

        Returns:
            np.ndarray: 1D array with 1 indicating anomaly and 0 indicating inlier.
        """
        ...

    @abstractmethod
    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute continuous anomaly scores (higher values indicate greater anomaly likelihood).

        Args:
            X: Feature matrix or DataFrame.

        Returns:
            np.ndarray: 1D array of continuous anomaly scores.
        """
        ...
