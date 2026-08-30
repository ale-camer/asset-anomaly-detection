import numpy as np
import pandas as pd
import pytest

from src.models.baseline import IsolationForestDetector, LOFDetector


@pytest.fixture
def synthetic_anomaly_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic 2D normal data and a distinct extreme outlier."""
    np.random.seed(42)
    # 100 normal points around origin
    inliers = np.random.normal(loc=0.0, scale=1.0, size=(100, 2))
    # 2 extreme outliers far away
    outliers = np.array([[20.0, 20.0], [-25.0, -25.0]])
    return inliers, outliers


@pytest.mark.parametrize("detector_cls", [IsolationForestDetector, LOFDetector])
def test_baseline_detector_fit_predict(
    detector_cls: type[IsolationForestDetector] | type[LOFDetector],
    synthetic_anomaly_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    inliers, outliers = synthetic_anomaly_dataset
    X_train = inliers
    X_test = np.vstack([inliers[:5], outliers])

    detector = detector_cls(contamination=0.05)
    detector.fit(X_train)

    preds = detector.predict(X_test)
    scores = detector.score_samples(X_test)

    assert len(preds) == len(X_test)
    assert len(scores) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})

    # Outliers should have higher anomaly score than inliers
    inlier_scores = scores[:5]
    outlier_scores = scores[5:]
    assert np.mean(outlier_scores) > np.mean(inlier_scores)


def test_detector_with_dataframe_and_feature_cols() -> None:
    df_train = pd.DataFrame(
        {
            "symbol": ["BTC"] * 100,
            "feature_1": np.random.normal(0, 1, 100),
            "feature_2": np.random.normal(0, 1, 100),
        }
    )
    df_test = pd.DataFrame(
        {
            "symbol": ["BTC", "BTC"],
            "feature_1": [0.1, 50.0],
            "feature_2": [0.2, 50.0],
        }
    )

    detector = IsolationForestDetector(feature_cols=["feature_1", "feature_2"], contamination=0.1)
    detector.fit(df_train)

    preds = detector.predict(df_test)
    assert preds[0] == 0  # Inlier
    assert preds[1] == 1  # Outlier


def test_detector_unfitted_raises() -> None:
    detector = IsolationForestDetector()
    X = np.array([[1.0, 2.0]])

    with pytest.raises(RuntimeError, match="is not fitted yet"):
        detector.predict(X)

    with pytest.raises(RuntimeError, match="is not fitted yet"):
        detector.score_samples(X)


def test_detector_empty_input_raises() -> None:
    detector = LOFDetector()
    with pytest.raises(ValueError, match="Input numpy array cannot be empty"):
        detector.fit(np.array([]))

    with pytest.raises(ValueError, match="Input DataFrame cannot be empty"):
        detector.fit(pd.DataFrame())


def test_detector_nan_input_raises() -> None:
    detector = IsolationForestDetector()
    X_nan = np.array([[1.0, np.nan], [2.0, 3.0]])
    with pytest.raises(ValueError, match="Input features contain NaN"):
        detector.fit(X_nan)
