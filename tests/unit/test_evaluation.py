import math

import numpy as np
import pandas as pd
import pytest

from src.models.evaluation import (
    compute_dynamic_threshold,
    evaluate_anomalies,
    precision_at_k,
)


@pytest.fixture
def synthetic_scores_and_labels() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic ground truth and scores with clear separation."""
    # 80 normal points with low scores, 20 anomalies with high scores
    np.random.seed(42)
    normal_scores = np.random.normal(loc=0.5, scale=0.1, size=80)
    anomaly_scores = np.random.normal(loc=5.0, scale=0.5, size=20)

    y_true = np.array([0] * 80 + [1] * 20)
    y_scores = np.concatenate([normal_scores, anomaly_scores])
    return y_true, y_scores


def test_precision_at_k_perfect() -> None:
    y_true = [0, 0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.3, 0.9, 0.8]
    # Top 2 scores are 0.9 and 0.8 which correspond to label 1
    assert precision_at_k(y_true, y_scores, k=2) == 1.0


def test_precision_at_k_partial() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_scores = np.array([10.0, 5.0, 2.0, 1.0])
    # Top 2 scores are 10.0 (label 1) and 5.0 (label 0)
    assert precision_at_k(y_true, y_scores, k=2) == 0.5
    # Top 1 score is 10.0 (label 1)
    assert precision_at_k(y_true, y_scores, k=1) == 1.0
    # Top 4 scores: 2 labels are 1
    assert precision_at_k(y_true, y_scores, k=4) == 0.5


def test_precision_at_k_pandas_series() -> None:
    y_true = pd.Series([1, 0, 0, 1])
    y_scores = pd.Series([10.0, 2.0, 1.0, 9.0])
    assert precision_at_k(y_true, y_scores, k=2) == 1.0


def test_precision_at_k_invalid_args() -> None:
    with pytest.raises(ValueError, match="k must be between 1 and"):
        precision_at_k([0, 1], [0.1, 0.9], k=0)

    with pytest.raises(ValueError, match="k must be between 1 and"):
        precision_at_k([0, 1], [0.1, 0.9], k=5)

    with pytest.raises(ValueError, match="Length mismatch"):
        precision_at_k([0, 1], [0.1], k=1)

    with pytest.raises(ValueError, match="cannot be empty"):
        precision_at_k([], [], k=1)


def test_compute_dynamic_threshold_percentile() -> None:
    scores = np.arange(1, 101, dtype=float)
    thresh_95 = compute_dynamic_threshold(scores, method="percentile", percentile=95)
    assert np.isclose(thresh_95, 95.05)

    thresh_50 = compute_dynamic_threshold(scores, method="percentile", q=50)
    assert np.isclose(thresh_50, 50.5)


def test_compute_dynamic_threshold_std() -> None:
    scores = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    thresh = compute_dynamic_threshold(scores, method="std", std_factor=3.0)
    assert np.isclose(thresh, 10.0)

    scores_var = np.array([0.0, 10.0])
    # mean = 5.0, std = 5.0
    thresh_var = compute_dynamic_threshold(scores_var, method="std", std_factor=2.0)
    assert np.isclose(thresh_var, 15.0)


def test_compute_dynamic_threshold_evt() -> None:
    np.random.seed(42)
    scores = np.random.exponential(scale=2.0, size=500)
    thresh_evt = compute_dynamic_threshold(scores, method="evt", initial_quantile=0.9, risk_prob=1e-3)
    thresh_pot = compute_dynamic_threshold(scores, method="pot", initial_quantile=0.9, risk_prob=1e-3)

    assert thresh_evt > float(np.quantile(scores, 0.9))
    assert np.isclose(thresh_evt, thresh_pot)


def test_compute_dynamic_threshold_errors() -> None:
    with pytest.raises(ValueError, match="empty scores"):
        compute_dynamic_threshold(np.array([]))

    with pytest.raises(ValueError, match="contains NaN"):
        compute_dynamic_threshold(np.array([1.0, np.nan, 3.0]))

    with pytest.raises(ValueError, match="Unrecognized thresholding method"):
        compute_dynamic_threshold(np.array([1.0, 2.0]), method="unknown")

    with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
        compute_dynamic_threshold(np.array([1.0, 2.0]), method="percentile", percentile=150)

    with pytest.raises(ValueError, match="initial_quantile must be in"):
        compute_dynamic_threshold(np.array([1.0, 2.0]), method="evt", initial_quantile=1.5)


def test_evaluate_anomalies_perfect(synthetic_scores_and_labels: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_scores = synthetic_scores_and_labels
    metrics = evaluate_anomalies(y_true, y_scores, k=20, threshold=2.0)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["precision_at_20"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["threshold"] == pytest.approx(2.0)


def test_evaluate_anomalies_inverted() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_scores = np.array([0.1, 0.2, 0.9, 0.8])
    metrics = evaluate_anomalies(y_true, y_scores)

    assert metrics["roc_auc"] == 0.0
    assert metrics["precision_at_2"] == 0.0


def test_evaluate_anomalies_single_class() -> None:
    y_true = np.array([0, 0, 0, 0])
    y_scores = np.array([0.1, 0.2, 0.3, 0.4])
    metrics = evaluate_anomalies(y_true, y_scores, threshold=0.25)

    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["average_precision"])
    assert "precision" in metrics


def test_evaluate_anomalies_validation_errors() -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_anomalies([1, 0], [0.5])

    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_anomalies([], [])
