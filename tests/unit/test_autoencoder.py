import numpy as np
import pandas as pd
import pytest

from src.models.autoencoder import AutoencoderAnomalyDetector, AutoencoderNetwork


@pytest.fixture
def synthetic_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate normal continuous market features and extreme outliers."""
    np.random.seed(42)
    # 200 normal samples with 4 features
    normal_data = np.random.normal(loc=0.0, scale=1.0, size=(200, 4))
    # 2 extreme outliers
    outliers = np.array(
        [
            [15.0, 15.0, 15.0, 15.0],
            [-20.0, -20.0, -20.0, -20.0],
        ]
    )
    return normal_data, outliers


def test_autoencoder_network_forward() -> None:
    import torch

    net = AutoencoderNetwork(input_dim=4, hidden_dim=8, latent_dim=2)
    x = torch.randn(10, 4)
    out = net(x)

    assert out.shape == (10, 4)


def test_autoencoder_fit_predict_scores(
    synthetic_training_data: tuple[np.ndarray, np.ndarray],
) -> None:
    normal_data, outliers = synthetic_training_data
    X_train = normal_data
    X_test = np.vstack([normal_data[:10], outliers])

    detector = AutoencoderAnomalyDetector(
        hidden_dim=8,
        latent_dim=2,
        epochs=30,
        batch_size=32,
        threshold_std_factor=2.0,
        random_seed=42,
    )

    detector.fit(X_train)

    scores = detector.score_samples(X_test)
    preds = detector.predict(X_test)

    assert len(scores) == len(X_test)
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})

    # Outlier reconstruction error must be noticeably higher than normal points
    normal_scores = scores[:10]
    outlier_scores = scores[10:]
    assert np.mean(outlier_scores) > np.mean(normal_scores) * 2.0

    # The extreme outliers should be classified as anomalies (1)
    assert (preds[10:] == 1).all()


def test_autoencoder_dataframe_and_feature_cols() -> None:
    np.random.seed(42)
    df_train = pd.DataFrame(
        {
            "symbol": ["BTC"] * 100,
            "f1": np.random.normal(0, 1, 100),
            "f2": np.random.normal(0, 1, 100),
        }
    )
    df_test = pd.DataFrame(
        {
            "symbol": ["BTC", "BTC"],
            "f1": [0.1, 30.0],
            "f2": [0.2, 30.0],
        }
    )

    detector = AutoencoderAnomalyDetector(
        feature_cols=["f1", "f2"],
        hidden_dim=4,
        latent_dim=2,
        epochs=20,
        threshold_std_factor=1.5,
    )
    detector.fit(df_train)

    preds = detector.predict(df_test)
    assert preds[0] == 0
    assert preds[1] == 1


def test_autoencoder_unfitted_raises() -> None:
    detector = AutoencoderAnomalyDetector()
    X = np.array([[1.0, 2.0]])

    with pytest.raises(RuntimeError, match="is not fitted yet"):
        detector.predict(X)

    with pytest.raises(RuntimeError, match="is not fitted yet"):
        detector.score_samples(X)


def test_autoencoder_empty_input_raises() -> None:
    detector = AutoencoderAnomalyDetector()
    with pytest.raises(ValueError, match="Input numpy array cannot be empty"):
        detector.fit(np.array([]))

    with pytest.raises(ValueError, match="Input DataFrame cannot be empty"):
        detector.fit(pd.DataFrame())
