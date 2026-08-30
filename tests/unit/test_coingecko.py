from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from src.ingestion.coingecko import CoinGeckoConnector
from src.utils.config import Settings


@pytest.fixture
def mock_settings() -> Settings:
    """Provide a Settings instance for tests."""
    return Settings(
        coingecko_api_base_url="https://api.coingecko.com/api/v3",
        coingecko_api_key="test_api_key",
    )


@pytest.fixture
def connector(mock_settings: Settings) -> CoinGeckoConnector:
    """Provide a CoinGeckoConnector instance with 0 delay for fast tests."""
    return CoinGeckoConnector(settings=mock_settings, rate_limit_delay=0.0)


def test_coingecko_headers(connector: CoinGeckoConnector) -> None:
    """Verify headers include the demo API key if present in settings."""
    headers = connector._get_headers()
    assert headers["Accept"] == "application/json"
    assert headers["x-cg-demo-api-key"] == "test_api_key"


def test_coingecko_fetch_data_success(connector: CoinGeckoConnector) -> None:
    """Test successful data retrieval and DataFrame formatting."""
    mock_payload = {
        "prices": [[1700000000000, 35000.5], [1700086400000, 36000.0]],
        "market_caps": [[1700000000000, 700000000000.0], [1700086400000, 710000000000.0]],
        "total_volumes": [[1700000000000, 15000000000.0], [1700086400000, 16000000000.0]],
    }

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_payload

    with patch("httpx.Client.get", return_value=mock_response) as mock_get:
        df = connector.fetch_data(
            symbol="bitcoin",
            start_date=datetime(2023, 11, 14, tzinfo=UTC),
            end_date=datetime(2023, 11, 16, tzinfo=UTC),
        )

        assert mock_get.called
        assert len(df) == 2
        assert list(df.columns) == [
            "timestamp",
            "symbol",
            "close",
            "market_cap",
            "volume",
            "open",
            "high",
            "low",
        ]
        assert df["symbol"].iloc[0] == "BITCOIN"
        assert df["close"].iloc[0] == 35000.5
        assert df["market_cap"].iloc[0] == 700000000000.0
        assert df["volume"].iloc[0] == 15000000000.0


def test_coingecko_fetch_data_empty(connector: CoinGeckoConnector) -> None:
    """Test fetch_data when API returns no price points."""
    mock_payload: dict[str, list] = {"prices": [], "market_caps": [], "total_volumes": []}
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_payload

    with patch("httpx.Client.get", return_value=mock_response):
        df = connector.fetch_data(
            symbol="bitcoin",
            start_date="2023-11-14",
            end_date="2023-11-16",
        )
        assert df.empty


def test_coingecko_validate_data(connector: CoinGeckoConnector) -> None:
    """Test schema validation on valid DataFrame."""
    df_raw = pd.DataFrame(
        [
            {
                "timestamp": datetime(2023, 11, 15, 0, 0, tzinfo=UTC),
                "symbol": "BTC",
                "open": None,
                "high": None,
                "low": None,
                "close": 35000.0,
                "volume": 10000.0,
                "market_cap": 500000.0,
            }
        ]
    )

    validated_df = connector.validate_data(df_raw)
    assert not validated_df.empty
    assert "timestamp" in validated_df.columns
    assert validated_df["close"].iloc[0] == 35000.0


def test_coingecko_validate_data_empty_raises(connector: CoinGeckoConnector) -> None:
    """Test that validating an empty DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        connector.validate_data(pd.DataFrame())


def test_coingecko_save_data(connector: CoinGeckoConnector, tmp_path: Path) -> None:
    """Test saving DataFrame to Parquet format."""
    df = pd.DataFrame(
        [
            {
                "timestamp": datetime(2023, 11, 15, 0, 0, tzinfo=UTC),
                "symbol": "BTC",
                "close": 35000.0,
                "volume": 1000.0,
                "market_cap": 50000.0,
            }
        ]
    )
    dest_path = tmp_path / "test_data.parquet"
    result_path = connector.save_data(df, output_path=dest_path)

    assert result_path.exists()
    loaded_df = pd.read_parquet(result_path)
    assert len(loaded_df) == 1
    assert loaded_df["symbol"].iloc[0] == "BTC"


def test_coingecko_retry_on_rate_limit(connector: CoinGeckoConnector) -> None:
    """Test retry behavior when encountering 429 Too Many Requests."""
    mock_req = httpx.Request("GET", "https://api.coingecko.com/api/v3/test")
    error_response = httpx.Response(status_code=429, request=mock_req)
    http_error = httpx.HTTPStatusError("Rate Limit", request=mock_req, response=error_response)

    success_response = MagicMock(spec=httpx.Response)
    success_response.status_code = 200
    success_response.json.return_value = {
        "prices": [[1700000000000, 35000.0]],
        "market_caps": [],
        "total_volumes": [],
    }

    # First attempt raises 429, second succeeds
    with patch(
        "httpx.Client.get",
        side_effect=[http_error, success_response],
    ) as mock_get:
        df = connector.fetch_data(
            symbol="bitcoin",
            start_date="2023-11-14",
            end_date="2023-11-16",
        )
        assert mock_get.call_count == 2
        assert len(df) == 1


def test_coingecko_retry_exhausted(connector: CoinGeckoConnector) -> None:
    """Test that retry error is raised when retries are exhausted."""
    mock_req = httpx.Request("GET", "https://api.coingecko.com/api/v3/test")
    error_response = httpx.Response(status_code=500, request=mock_req)
    http_error = httpx.HTTPStatusError("Server Error", request=mock_req, response=error_response)

    with patch("httpx.Client.get", side_effect=http_error) as mock_get:
        with pytest.raises(httpx.HTTPStatusError):
            connector.fetch_data(
                symbol="bitcoin",
                start_date="2023-11-14",
                end_date="2023-11-16",
            )
        assert mock_get.call_count == 3
