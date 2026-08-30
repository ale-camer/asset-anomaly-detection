from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from src.features.store import FeatureStoreSink


@pytest.fixture
def valid_features_df() -> pd.DataFrame:
    """Fixture providing a DataFrame with valid computed features."""
    return pd.DataFrame(
        [
            {
                "timestamp": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
                "symbol": "BTC",
                "close_rolling_mean_7": 42000.0,
                "close_rolling_std_7": 150.5,
                "close_ema_7": 42100.0,
                "volatility_parkinson_14": 0.05,
                "rsi_14": 55.0,
                "macd_line": 100.0,
                "macd_signal": 90.0,
                "macd_hist": 10.0,
                "price_return_1": 0.01,
                "price_velocity_1": -0.005,
            },
        ]
    )


def test_feature_store_valid_write(
    valid_features_df: pd.DataFrame, tmp_path: Path
) -> None:
    sink = FeatureStoreSink(base_dir=tmp_path)
    output_path = sink.write(valid_features_df)

    assert output_path == tmp_path
    partition_dir = tmp_path / "symbol=BTC" / "date=2024-01-01"
    assert partition_dir.exists()
    assert len(list(partition_dir.glob("*.parquet"))) >= 1


def test_feature_store_missing_required_fields(tmp_path: Path) -> None:
    sink = FeatureStoreSink(base_dir=tmp_path)
    # Missing symbol
    invalid_df = pd.DataFrame(
        [{"timestamp": datetime(2024, 1, 1, 12, 0, tzinfo=UTC), "rsi_14": 50.0}]
    )

    with pytest.raises(ValidationError):
        sink.write(invalid_df)


def test_feature_store_invalid_types(
    valid_features_df: pd.DataFrame, tmp_path: Path
) -> None:
    sink = FeatureStoreSink(base_dir=tmp_path)
    invalid_df = valid_features_df.copy()
    invalid_df["rsi_14"] = invalid_df["rsi_14"].astype(object)
    invalid_df.loc[0, "rsi_14"] = "invalid_string"  # type: ignore[call-overload]

    with pytest.raises(ValidationError):
        sink.write(invalid_df)


def test_feature_store_out_of_bounds(
    valid_features_df: pd.DataFrame, tmp_path: Path
) -> None:
    sink = FeatureStoreSink(base_dir=tmp_path)
    invalid_df = valid_features_df.copy()
    # RSI cannot be > 100
    invalid_df.loc[0, "rsi_14"] = 150.0

    with pytest.raises(ValidationError):
        sink.write(invalid_df)

    # Parkinson Volatility cannot be negative
    invalid_df_2 = valid_features_df.copy()
    invalid_df_2.loc[0, "volatility_parkinson_14"] = -0.05

    with pytest.raises(ValidationError):
        sink.write(invalid_df_2)
