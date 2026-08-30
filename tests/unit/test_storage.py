from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.storage.parquet_sink import ParquetStorageSink


@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    """Fixture providing sample market data DataFrame."""
    return pd.DataFrame(
        [
            {
                "timestamp": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
                "symbol": "BTC",
                "close": 42000.0,
                "volume": 1500.0,
            },
            {
                "timestamp": datetime(2024, 1, 1, 13, 0, tzinfo=UTC),
                "symbol": "BTC",
                "close": 42500.0,
                "volume": 1800.0,
            },
            {
                "timestamp": datetime(2024, 1, 2, 12, 0, tzinfo=UTC),
                "symbol": "ETH",
                "close": 2200.0,
                "volume": 8000.0,
            },
        ]
    )


def test_parquet_storage_write_and_read(
    sample_market_df: pd.DataFrame, tmp_path: Path
) -> None:
    sink = ParquetStorageSink(base_dir=tmp_path)
    output_path = sink.write(sample_market_df)

    assert output_path == tmp_path

    # Verify partition directories exist
    btc_dir = tmp_path / "symbol=BTC" / "date=2024-01-01"
    eth_dir = tmp_path / "symbol=ETH" / "date=2024-01-02"

    assert btc_dir.exists()
    assert eth_dir.exists()
    assert len(list(btc_dir.glob("*.parquet"))) >= 1
    assert len(list(eth_dir.glob("*.parquet"))) >= 1

    # Read back and verify row count and values
    loaded_df = sink.read()
    assert len(loaded_df) == 3
    assert set(loaded_df["symbol"].unique()) == {"BTC", "ETH"}
    assert set(loaded_df["date"].unique()) == {"2024-01-01", "2024-01-02"}


def test_parquet_storage_empty_df_raises(tmp_path: Path) -> None:
    sink = ParquetStorageSink(base_dir=tmp_path)
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Cannot write an empty DataFrame"):
        sink.write(empty_df)


def test_parquet_storage_missing_timestamp_for_date_raises(tmp_path: Path) -> None:
    sink = ParquetStorageSink(base_dir=tmp_path)
    df = pd.DataFrame([{"symbol": "BTC", "close": 100.0}])

    with pytest.raises(ValueError, match="must contain 'timestamp' column"):
        sink.write(df)


def test_parquet_storage_missing_partition_col_raises(tmp_path: Path) -> None:
    sink = ParquetStorageSink(base_dir=tmp_path)
    df = pd.DataFrame(
        [{"timestamp": datetime(2024, 1, 1, tzinfo=UTC), "close": 100.0}]
    )

    with pytest.raises(ValueError, match="Partition column 'symbol' is not present"):
        sink.write(df)


def test_parquet_storage_read_non_existent_raises(tmp_path: Path) -> None:
    non_existent = tmp_path / "does_not_exist"
    sink = ParquetStorageSink(base_dir=non_existent)

    with pytest.raises(FileNotFoundError):
        sink.read()
