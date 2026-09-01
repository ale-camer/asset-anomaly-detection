from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.models.base import BaseAnomalyDetector


class IsolationForestDetector(BaseAnomalyDetector):
    """Anomaly detector based on Isolation Forest."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        random_state: int | None = 42,
        feature_cols: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Isolation Forest anomaly detector.

        Args:
            n_estimators: Number of base estimators in the ensemble. Defaults to 100.
            contamination: Expected proportion of outliers in the data. Defaults to 'auto'.
            random_state: Random seed for reproducibility. Defaults to 42.
            feature_cols: Optional list of columns to use if input is a DataFrame.
            **kwargs: Additional parameters passed to sklearn IsolationForest.
        """
        super().__init__(feature_cols=feature_cols)
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.extra_kwargs = kwargs
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            **self.extra_kwargs,
        )

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "IsolationForestDetector":
        """Fit Isolation Forest model on training data.

        Args:
            X: Input features.
            y: Ignored.

        Returns:
            self
        """
        features = self._prepare_features(X)
        self._log_mlflow_params(
            {
                "model_type": self.__class__.__name__,
                "n_estimators": self.n_estimators,
                "contamination": str(self.contamination),
                "random_state": self.random_state,
            }
        )
        self._model.fit(features)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict binary anomaly labels (1 for anomaly, 0 for normal).

        Args:
            X: Input features.

        Returns:
            np.ndarray: 1D binary array where 1 = anomaly, 0 = normal.
        """
        if not self.is_fitted:
            raise RuntimeError("IsolationForestDetector is not fitted yet. Call 'fit' before 'predict'.")
        features = self._prepare_features(X)
        raw_preds = self._model.predict(features)
        # Sklearn returns -1 for anomaly, 1 for inlier. Convert to 1 for anomaly, 0 for inlier.
        return np.asarray((raw_preds == -1).astype(int))

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute continuous anomaly score (higher value = more anomalous).

        Args:
            X: Input features.

        Returns:
            np.ndarray: 1D array of anomaly scores.
        """
        if not self.is_fitted:
            raise RuntimeError("IsolationForestDetector is not fitted yet. Call 'fit' before 'score_samples'.")
        features = self._prepare_features(X)
        # Sklearn returns negative anomaly score (lower is more anomalous).
        # Inverting so higher score indicates greater anomaly severity.
        return np.asarray(-self._model.score_samples(features), dtype=float)


class LOFDetector(BaseAnomalyDetector):
    """Anomaly detector based on Local Outlier Factor (LOF) with novelty detection."""

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float | str = "auto",
        feature_cols: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Local Outlier Factor anomaly detector.

        Args:
            n_neighbors: Number of neighbors to use. Defaults to 20.
            contamination: Expected proportion of outliers. Defaults to 'auto'.
            feature_cols: Optional list of columns to use if input is a DataFrame.
            **kwargs: Additional parameters passed to sklearn LocalOutlierFactor.
        """
        super().__init__(feature_cols=feature_cols)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.extra_kwargs = kwargs
        self._model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
            **self.extra_kwargs,
        )

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "LOFDetector":
        """Fit Local Outlier Factor detector on training data.

        Args:
            X: Input features.
            y: Ignored.

        Returns:
            self
        """
        features = self._prepare_features(X)
        self._log_mlflow_params(
            {
                "model_type": self.__class__.__name__,
                "n_neighbors": self.n_neighbors,
                "contamination": str(self.contamination),
            }
        )
        self._model.fit(features)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict binary anomaly labels (1 for anomaly, 0 for normal).

        Args:
            X: Input features.

        Returns:
            np.ndarray: 1D binary array where 1 = anomaly, 0 = normal.
        """
        if not self.is_fitted:
            raise RuntimeError("LOFDetector is not fitted yet. Call 'fit' before 'predict'.")
        features = self._prepare_features(X)
        raw_preds = self._model.predict(features)
        return np.asarray((raw_preds == -1).astype(int))

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute continuous anomaly score (higher value = more anomalous).

        Args:
            X: Input features.

        Returns:
            np.ndarray: 1D array of anomaly scores.
        """
        if not self.is_fitted:
            raise RuntimeError("LOFDetector is not fitted yet. Call 'fit' before 'score_samples'.")
        features = self._prepare_features(X)
        return np.asarray(-self._model.score_samples(features), dtype=float)
