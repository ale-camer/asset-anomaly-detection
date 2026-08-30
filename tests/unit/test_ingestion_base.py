from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from src.ingestion.base import BaseConnector, MarketDataRecord
from src.utils.config import Settings


class MockConnector(BaseConnector):
    """Concrete mock implementation of BaseConnector for testing."""

    def fetch_data(
        self,
        symbol: str,
        start_date: datetime | str,
        end_date: datetime | str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "symbol": symbol,
                    "close": 100.0,
                }
            ]
        )

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def save_data(
        self,
        df: pd.DataFrame,
        output_path: Path | str | None = None,
        **kwargs: Any,
    ) -> Path:
        target = Path(output_path or "dummy.parquet")
        return target


def test_base_connector_cannot_be_instantiated() -> None:
    """Test that the abstract BaseConnector raises TypeError when instantiated directly."""
    with pytest.raises(TypeError):
        BaseConnector()  # type: ignore


def test_concrete_connector_instantiation_and_execution() -> None:
    """Test that a concrete implementation of BaseConnector works as expected."""
    custom_settings = Settings(environment="testing")
    connector = MockConnector(settings=custom_settings)

    assert connector.settings.environment == "testing"

    df = connector.fetch_data("BTC", "2026-01-01", "2026-01-02")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "BTC"

    validated_df = connector.validate_data(df)
    assert len(validated_df) == 1

    saved_path = connector.save_data(validated_df, "test.parquet")
    assert saved_path == Path("test.parquet")


def test_market_data_record_validation() -> None:
    """Test Pydantic validation for MarketDataRecord."""
    # Valid record
    record = MarketDataRecord(
        timestamp=datetime(2026, 1, 1, 12, 0),
        symbol="ETH",
        open=2000.0,
        high=2100.0,
        low=1950.0,
        close=2050.0,
        volume=15000.0,
    )
    assert record.symbol == "ETH"
    assert record.close == 2050.0

    # Invalid record (missing required 'close')
    with pytest.raises(ValidationError):
        MarketDataRecord.model_validate(
            {
                "timestamp": datetime(2026, 1, 1, 12, 0),
                "symbol": "ETH",
            }
        )
