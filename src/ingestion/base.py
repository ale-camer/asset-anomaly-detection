from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from src.utils.config import Settings, get_settings


class MarketDataRecord(BaseModel):
    """Schema model for a single standardized market data record."""

    timestamp: datetime = Field(..., description="Timestamp of the market data point in UTC")
    symbol: str = Field(..., description="Ticker or asset identifier (e.g. BTC, ETH, AAPL)")
    open: float | None = Field(default=None, description="Opening price of the period")
    high: float | None = Field(default=None, description="Highest price of the period")
    low: float | None = Field(default=None, description="Lowest price of the period")
    close: float = Field(..., description="Closing or spot price of the period")
    volume: float | None = Field(default=None, description="Trading volume")
    market_cap: float | None = Field(default=None, description="Market capitalization if available")


class BaseConnector(ABC):
    """Abstract Base Class for market data ingestion connectors."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the connector with application settings.

        Args:
            settings: Application settings instance. If None, uses default settings.
        """
        self.settings = settings or get_settings()

    @abstractmethod
    def fetch_data(
        self,
        symbol: str,
        start_date: datetime | str,
        end_date: datetime | str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch raw market data from the source API.

        Args:
            symbol: Ticker or identifier of the asset.
            start_date: Start date for historical data extraction.
            end_date: End date for historical data extraction.
            **kwargs: Additional provider-specific parameters.

        Returns:
            pd.DataFrame: Raw market data retrieved from the source.
        """
        pass

    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and conform the input DataFrame against the standard market schema.

        Args:
            df: Input DataFrame containing market data.

        Returns:
            pd.DataFrame: Validated DataFrame conforming to schema standards.
        """
        pass

    @abstractmethod
    def save_data(
        self,
        df: pd.DataFrame,
        output_path: Path | str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Persist market data to local or object storage (e.g. Parquet).

        Args:
            df: Validated DataFrame to persist.
            output_path: Destination file or directory path.
            **kwargs: Additional storage parameters.

        Returns:
            Path: Path to the saved file.
        """
        pass
