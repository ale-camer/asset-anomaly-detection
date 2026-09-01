"""Evaluation metrics and dynamic thresholding utilities for anomaly detection."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def precision_at_k(
    y_true: np.ndarray | pd.Series | list[int],
    y_scores: np.ndarray | pd.Series | list[float],
    k: int,
) -> float:
    """Compute Precision@k for anomaly detection.

    Proportion of true anomalies among the top-k highest anomaly scores.

    Args:
        y_true: Binary ground truth array (1 for anomaly, 0 for inlier).
        y_scores: Continuous anomaly scores (higher indicates greater anomaly).
        k: Number of top-ranked samples to consider. Must be >= 1.

    Returns:
        float: Precision at rank k (between 0.0 and 1.0).

    Raises:
        ValueError: If k is less than 1 or exceeds dataset size, or if lengths do not match.
    """
    labels = np.asarray(y_true, dtype=int).ravel()
    scores = np.asarray(y_scores, dtype=float).ravel()

    if len(labels) != len(scores):
        raise ValueError(f"Length mismatch: len(y_true)={len(labels)} != len(y_scores)={len(scores)}")
    if len(labels) == 0:
        raise ValueError("Input arrays cannot be empty.")
    if k < 1 or k > len(labels):
        raise ValueError(f"k must be between 1 and {len(labels)}, got {k}")

    # Top-k indices with largest scores
    top_k_indices = np.argsort(scores)[::-1][:k]
    true_positives = np.sum(labels[top_k_indices] == 1)
    return float(true_positives / k)


def compute_dynamic_threshold(
    scores: np.ndarray | pd.Series | list[float],
    method: str = "percentile",
    **kwargs: Any,
) -> float:
    """Compute a dynamic decision threshold for continuous anomaly scores.

    Supported methods:
        - "percentile": Cutoff based on a percentile of the score distribution.
          kwargs: `percentile` (float, default 95.0).
        - "std": Mean plus k standard deviations (Gaussian tail).
          kwargs: `std_factor` or `factor` (float, default 3.0).
        - "evt" / "pot": Extreme Value Theory via Peaks-Over-Threshold (POT).
          kwargs: `initial_quantile` (float, default 0.95), `risk_prob` (float, default 1e-3).

    Args:
        scores: 1D array of anomaly scores.
        method: Method used to compute threshold ("percentile", "std", "evt", "pot").
        **kwargs: Additional method-specific parameters.

    Returns:
        float: Computed threshold value.

    Raises:
        ValueError: If scores array is empty, contains NaNs, or method is unrecognized.
    """
    arr = np.asarray(scores, dtype=float).ravel()

    if arr.size == 0:
        raise ValueError("Cannot compute threshold from empty scores array.")
    if np.isnan(arr).any():
        raise ValueError("Scores array contains NaN values.")

    norm_method = method.lower()

    if norm_method == "percentile":
        q = float(kwargs.get("percentile", kwargs.get("q", 95.0)))
        if not (0.0 <= q <= 100.0):
            raise ValueError(f"Percentile must be between 0 and 100, got {q}")
        return float(np.percentile(arr, q))

    elif norm_method == "std":
        factor = float(kwargs.get("std_factor", kwargs.get("factor", 3.0)))
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        return mean_val + factor * std_val

    elif norm_method in ("evt", "pot"):
        initial_q = float(kwargs.get("initial_quantile", 0.95))
        risk_prob = float(kwargs.get("risk_prob", 1e-3))

        if not (0.0 < initial_q < 1.0):
            raise ValueError(f"initial_quantile must be in (0, 1), got {initial_q}")
        if not (0.0 < risk_prob < 1.0):
            raise ValueError(f"risk_prob must be in (0, 1), got {risk_prob}")

        # Initial threshold t
        t = float(np.quantile(arr, initial_q))
        excesses = arr[arr > t] - t
        n_t = len(excesses)
        n = len(arr)

        if n_t < 2:
            # Fallback to percentile if not enough tail samples
            return float(np.quantile(arr, 1.0 - risk_prob))

        # Fit Generalized Pareto Distribution (GPD) on excesses via method of moments
        mean_excess = float(np.mean(excesses))
        var_excess = float(np.var(excesses, ddof=1)) if n_t > 1 else 0.0

        if var_excess > 0:
            gamma = 0.5 * (1.0 - (mean_excess**2) / var_excess)
            sigma = 0.5 * mean_excess * ((mean_excess**2) / var_excess + 1.0)
        else:
            gamma = 0.0
            sigma = max(mean_excess, 1e-6)

        # Ensure parameters are well-behaved
        if abs(gamma) < 1e-5:
            # Gumbel / Exponential limit
            evt_threshold = t + sigma * np.log((n / n_t) * (1.0 - initial_q) / risk_prob)
        else:
            ratio = (risk_prob * n) / n_t
            if ratio > 0:
                evt_threshold = t + (sigma / gamma) * ((ratio ** (-gamma)) - 1.0)
            else:
                evt_threshold = t + mean_excess

        return float(evt_threshold)

    else:
        raise ValueError(
            f"Unrecognized thresholding method: '{method}'. "
            f"Supported methods are: 'percentile', 'std', 'evt', 'pot'."
        )


def evaluate_anomalies(
    y_true: np.ndarray | pd.Series | list[int],
    y_scores: np.ndarray | pd.Series | list[float],
    k: int | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    """Calculate standard anomaly detection evaluation metrics.

    Metrics calculated:
        - `roc_auc`: Area under the ROC curve.
        - `average_precision`: Area under the Precision-Recall curve (PR-AUC).
        - `precision_at_k`: Precision@k for top anomalies (if k is specified or inferred).
        - `precision`, `recall`, `f1`: Classification metrics if `threshold` is provided.

    Args:
        y_true: Ground truth binary labels (1 = anomaly, 0 = inlier).
        y_scores: Continuous anomaly scores.
        k: Optional rank k for Precision@k calculation.
        threshold: Optional decision threshold to compute binary metrics (precision, recall, f1).

    Returns:
        dict[str, float]: Dictionary mapping metric names to their computed values.

    Raises:
        ValueError: If input dimensions do not match or arrays are empty.
    """
    labels = np.asarray(y_true, dtype=int).ravel()
    scores = np.asarray(y_scores, dtype=float).ravel()

    if len(labels) != len(scores):
        raise ValueError(f"Length mismatch: len(y_true)={len(labels)} != len(y_scores)={len(scores)}")
    if len(labels) == 0:
        raise ValueError("Input arrays cannot be empty.")

    metrics: dict[str, float] = {}

    # ROC-AUC & PR-AUC
    unique_labels = np.unique(labels)
    if len(unique_labels) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["average_precision"] = float(average_precision_score(labels, scores))
    else:
        # Fallback when only one class is present in y_true
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")

    # Precision@k
    if k is None:
        # Default k to number of true anomalies or at least 1
        positives = int(np.sum(labels == 1))
        k = positives if positives > 0 else min(10, len(labels))

    if 1 <= k <= len(labels):
        metrics[f"precision_at_{k}"] = precision_at_k(labels, scores, k=k)

    # Threshold-based binary metrics
    if threshold is not None:
        preds = (scores >= threshold).astype(int)
        metrics["precision"] = float(precision_score(labels, preds, zero_division=0.0))
        metrics["recall"] = float(recall_score(labels, preds, zero_division=0.0))
        metrics["f1"] = float(f1_score(labels, preds, zero_division=0.0))
        metrics["threshold"] = float(threshold)

    return metrics
