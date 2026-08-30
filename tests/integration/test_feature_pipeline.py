from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.store import FeatureStoreSink
from src.features.transformers import (
    MomentumFeatures,
    PriceVelocityFeatures,
    TimeSeriesRollingFeatures,
    VolatilityFeatures,
)


def test_feature_pipeline_e2e_integration(tmp_path: Path) -> None:
    """Test the complete feature engineering pipeline from raw data to Feature Store."""

    # 1. Generate realistic raw dataset (synthetic)
    # We need enough data points (e.g. 20) to satisfy the 14-period rolling windows without being all NaNs
    n_periods = 20
    dates = [datetime(2024, 1, i + 1, 12, 0, tzinfo=UTC) for i in range(n_periods)]

    # Create somewhat realistic prices with a bit of variance
    close_prices = np.linspace(100, 120, n_periods) + np.random.normal(0, 1, n_periods)
    high_prices = close_prices + 2.0
    low_prices = close_prices - 2.0

    df_raw = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["BTC"] * n_periods,
            "open": close_prices - 0.5,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": [1000.0] * n_periods,
        }
    )

    # 2. Configure the Pipeline corresponding exactly to the FeatureSetRecord schema requirements
    pipeline = Pipeline(
        [
            ("rolling", TimeSeriesRollingFeatures(windows=[7], columns=["close"])),
            ("volatility", VolatilityFeatures(windows=[14])),
            ("momentum", MomentumFeatures(rsi_window=14)),
            ("velocity", PriceVelocityFeatures(periods=[1])),
        ]
    )

    # 3. Transform the data
    df_features = pipeline.fit_transform(df_raw)

    # 4. Handle NaNs caused by rolling windows
    # For a feature store, we can either drop initial rows with NaNs, or forward/backward fill.
    # We'll drop rows where the longest window (14) didn't have enough data (min_periods=1 allows some,
    # but pct_change requires at least 2). We will just rely on Pydantic's Optional/None handling,
    # but pandas NaNs must be mapped properly.
    # The store sink does `replace({float("nan"): None})` internally so it handles it.

    # 5. Persist to Feature Store Sink
    sink = FeatureStoreSink(base_dir=tmp_path)

    # This write will automatically trigger Pydantic schema validation.
    # If the pipeline failed to create 'rsi_14' or 'volatility_parkinson_14', this would raise ValidationError.
    output_path = sink.write(df_features)

    # 6. Verify outputs
    assert output_path == tmp_path

    # Read back the parquet to ensure it round-trips correctly
    df_loaded = sink.read()

    # Verify expected columns from the schema are present in the loaded dataset
    expected_columns = [
        "timestamp", "symbol",
        "close_rolling_mean_7", "close_rolling_std_7", "close_ema_7",
        "volatility_parkinson_14", "rsi_14",
        "macd_line", "macd_signal", "macd_hist",
        "price_return_1", "price_velocity_1"
    ]

    for col in expected_columns:
        assert col in df_loaded.columns, f"Missing required column {col} in persisted features"

    # Verify that data types are generally float64 for metrics
    assert pd.api.types.is_float_dtype(df_loaded["rsi_14"])
    assert pd.api.types.is_float_dtype(df_loaded["volatility_parkinson_14"])
