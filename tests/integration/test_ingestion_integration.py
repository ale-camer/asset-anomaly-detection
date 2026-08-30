from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from src.ingestion.coingecko import CoinGeckoConnector
from src.storage.parquet_sink import ParquetStorageSink
from src.utils.config import Settings


def test_ingestion_to_storage_pipeline_integration(tmp_path: Path) -> None:
    """End-to-end integration test from CoinGecko data ingestion to Parquet storage sink."""
    settings = Settings(
        coingecko_api_base_url="https://api.coingecko.com/api/v3",
        coingecko_api_key="test_mock_key",
        data_raw_dir=tmp_path,
    )

    connector = CoinGeckoConnector(settings=settings, rate_limit_delay=0.0)
    sink = ParquetStorageSink(base_dir=tmp_path, settings=settings)

    mock_payload = {
        "prices": [
            [1704067200000, 42000.0],  # 2024-01-01 00:00:00 UTC
            [1704153600000, 43000.0],  # 2024-01-02 00:00:00 UTC
        ],
        "market_caps": [
            [1704067200000, 800000000000.0],
            [1704153600000, 820000000000.0],
        ],
        "total_volumes": [
            [1704067200000, 20000000000.0],
            [1704153600000, 22000000000.0],
        ],
    }

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_payload

    with patch("httpx.Client.get", return_value=mock_response):
        # 1. Fetch raw data from provider
        df_raw = connector.fetch_data(
            symbol="bitcoin",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 3, tzinfo=UTC),
        )

        # 2. Validate data conforming to schema
        df_validated = connector.validate_data(df_raw)

        # 3. Persist dataset into partitioned Parquet sink
        storage_path = sink.write(df_validated)

    # 4. Assert directory structure and partitions
    assert storage_path == tmp_path
    partition_day_1 = tmp_path / "symbol=BITCOIN" / "date=2024-01-01"
    partition_day_2 = tmp_path / "symbol=BITCOIN" / "date=2024-01-02"

    assert partition_day_1.exists(), f"Expected partition directory {partition_day_1} to exist"
    assert partition_day_2.exists(), f"Expected partition directory {partition_day_2} to exist"

    # 5. Read back from Parquet storage and verify round-trip integrity
    loaded_df = sink.read()
    assert len(loaded_df) == 2
    assert set(loaded_df["symbol"].unique()) == {"BITCOIN"}
    assert set(loaded_df["date"].unique()) == {"2024-01-01", "2024-01-02"}
    assert list(loaded_df.sort_values("date")["close"]) == [42000.0, 43000.0]
