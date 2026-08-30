from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features.transformers import (
    MomentumFeatures,
    PriceVelocityFeatures,
    TimeSeriesRollingFeatures,
    VolatilityFeatures,
)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Fixture providing 10 days of synthetic OHLCV data."""
    dates = [datetime(2024, 1, i + 1, tzinfo=UTC) for i in range(10)]
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["BTC"] * 10,
            "open": [100.0, 102.0, 101.0, 105.0, 107.0, 106.0, 110.0, 112.0, 111.0, 115.0],
            "high": [105.0, 106.0, 104.0, 108.0, 110.0, 109.0, 114.0, 115.0, 113.0, 118.0],
            "low": [98.0, 100.0, 99.0, 102.0, 104.0, 103.0, 107.0, 108.0, 108.0, 112.0],
            "close": [102.0, 101.0, 104.0, 107.0, 106.0, 108.0, 112.0, 111.0, 114.0, 117.0],
            "volume": [1000.0 * (i + 1) for i in range(10)],
        }
    )


def test_volatility_features_parkinson(sample_ohlcv_df: pd.DataFrame) -> None:
    transformer = VolatilityFeatures(windows=[3, 5])
    df_transformed = transformer.fit_transform(sample_ohlcv_df)

    assert "volatility_parkinson_3" in df_transformed.columns
    assert "volatility_parkinson_5" in df_transformed.columns
    assert (df_transformed["volatility_parkinson_3"] >= 0).all()
    assert (df_transformed["volatility_parkinson_5"] >= 0).all()


def test_volatility_features_missing_columns() -> None:
    transformer = VolatilityFeatures()
    df_invalid = pd.DataFrame({"close": [10.0, 20.0]})
    with pytest.raises(ValueError, match="DataFrame must contain 'high' and 'low'"):
        transformer.transform(df_invalid)


def test_momentum_features_rsi_and_macd(sample_ohlcv_df: pd.DataFrame) -> None:
    transformer = MomentumFeatures(rsi_window=5, macd_fast=3, macd_slow=6, macd_signal=3)
    df_transformed = transformer.fit_transform(sample_ohlcv_df)

    assert "rsi_5" in df_transformed.columns
    assert "macd_line" in df_transformed.columns
    assert "macd_signal" in df_transformed.columns
    assert "macd_hist" in df_transformed.columns

    # RSI must be strictly bounded between 0 and 100
    assert (df_transformed["rsi_5"] >= 0.0).all()
    assert (df_transformed["rsi_5"] <= 100.0).all()

    # MACD Histogram should equal MACD Line minus Signal
    np.testing.assert_allclose(
        df_transformed["macd_hist"].values,
        (df_transformed["macd_line"] - df_transformed["macd_signal"]).values,
    )


def test_price_velocity_features(sample_ohlcv_df: pd.DataFrame) -> None:
    transformer = PriceVelocityFeatures(periods=[1, 2])
    df_transformed = transformer.fit_transform(sample_ohlcv_df)

    assert "price_return_1" in df_transformed.columns
    assert "price_velocity_1" in df_transformed.columns
    assert "price_return_2" in df_transformed.columns
    assert "price_velocity_2" in df_transformed.columns


def test_technical_features_multi_symbol() -> None:
    dates = [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)]
    df = pd.DataFrame(
        {
            "timestamp": dates + dates,
            "symbol": ["BTC", "BTC", "ETH", "ETH"],
            "high": [110.0, 120.0, 11.0, 12.0],
            "low": [100.0, 105.0, 10.0, 10.5],
            "close": [105.0, 115.0, 10.5, 11.5],
        }
    )

    vol = VolatilityFeatures(windows=[2], group_by="symbol")
    df_vol = vol.fit_transform(df)
    assert len(df_vol) == 4
    assert "volatility_parkinson_2" in df_vol.columns

    mom = MomentumFeatures(rsi_window=2, macd_fast=2, macd_slow=3, macd_signal=2, group_by="symbol")
    df_mom = mom.fit_transform(df)
    assert len(df_mom) == 4
    assert "rsi_2" in df_mom.columns


def test_full_feature_pipeline(sample_ohlcv_df: pd.DataFrame) -> None:
    pipeline = Pipeline(
        [
            ("rolling", TimeSeriesRollingFeatures(windows=[3])),
            ("volatility", VolatilityFeatures(windows=[3])),
            ("momentum", MomentumFeatures(rsi_window=3, macd_fast=3, macd_slow=5, macd_signal=2)),
            ("velocity", PriceVelocityFeatures(periods=[1])),
        ]
    )

    res = pipeline.fit_transform(sample_ohlcv_df)
    assert isinstance(res, pd.DataFrame)
    assert "close_rolling_mean_3" in res.columns
    assert "volatility_parkinson_3" in res.columns
    assert "rsi_3" in res.columns
    assert "price_velocity_1" in res.columns
