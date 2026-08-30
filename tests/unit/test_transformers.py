from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features.transformers import TimeSeriesRollingFeatures


@pytest.fixture
def sample_timeseries_df() -> pd.DataFrame:
    """Fixture providing simple predictable sequential market data."""
    dates = [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
        datetime(2024, 1, 4, tzinfo=UTC),
    ]
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["BTC"] * 4,
            "close": [10.0, 20.0, 30.0, 40.0],
            "volume": [100.0, 200.0, 300.0, 400.0],
        }
    )


def test_time_series_rolling_features_single_symbol(
    sample_timeseries_df: pd.DataFrame,
) -> None:
    transformer = TimeSeriesRollingFeatures(windows=[2], columns=["close"])
    df_transformed = transformer.fit_transform(sample_timeseries_df)

    assert "close_rolling_mean_2" in df_transformed.columns
    assert "close_rolling_std_2" in df_transformed.columns
    assert "close_ema_2" in df_transformed.columns

    # Window 2 mean for [10, 20, 30, 40] with min_periods=1: [10, 15, 25, 35]
    expected_means = [10.0, 15.0, 25.0, 35.0]
    np.testing.assert_allclose(df_transformed["close_rolling_mean_2"].values, expected_means)

    # First std should be 0.0, second std([10, 20]) should be ~7.071
    assert df_transformed["close_rolling_std_2"].iloc[0] == 0.0
    assert df_transformed["close_rolling_std_2"].iloc[1] == pytest.approx(np.std([10, 20], ddof=1))


def test_time_series_rolling_features_multiple_symbols() -> None:
    dates = [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
    ]
    df = pd.DataFrame(
        {
            "timestamp": dates + dates,
            "symbol": ["BTC", "BTC", "ETH", "ETH"],
            "close": [100.0, 200.0, 10.0, 20.0],
        }
    )
    transformer = TimeSeriesRollingFeatures(windows=[2], columns=["close"], group_by="symbol")
    df_transformed = transformer.fit_transform(df)

    # Verify grouping prevents cross-symbol leakage
    btc_means = df_transformed[df_transformed["symbol"] == "BTC"]["close_rolling_mean_2"].values
    eth_means = df_transformed[df_transformed["symbol"] == "ETH"]["close_rolling_mean_2"].values

    np.testing.assert_allclose(btc_means, [100.0, 150.0])
    np.testing.assert_allclose(eth_means, [10.0, 15.0])


def test_time_series_rolling_features_empty_df_raises() -> None:
    transformer = TimeSeriesRollingFeatures()
    with pytest.raises(ValueError, match="Input DataFrame cannot be empty"):
        transformer.transform(pd.DataFrame())


def test_time_series_rolling_features_invalid_type_raises() -> None:
    transformer = TimeSeriesRollingFeatures()
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        transformer.transform("invalid_type")  # type: ignore[arg-type]


def test_time_series_rolling_features_pipeline_compatibility(
    sample_timeseries_df: pd.DataFrame,
) -> None:
    pipeline = Pipeline(
        [
            ("rolling", TimeSeriesRollingFeatures(windows=[2, 3])),
        ]
    )
    res = pipeline.fit_transform(sample_timeseries_df)

    assert isinstance(res, pd.DataFrame)
    assert "close_rolling_mean_2" in res.columns
    assert "close_rolling_mean_3" in res.columns
    assert "volume_ema_2" in res.columns
