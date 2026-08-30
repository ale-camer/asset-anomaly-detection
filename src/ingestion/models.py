from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PriceSnapshot(BaseModel):
    """Strict validation schema for raw market price snapshots."""

    model_config = ConfigDict(strict=True, extra="forbid")

    timestamp: datetime = Field(..., description="Timestamp of the price snapshot in UTC")
    symbol: str = Field(..., min_length=1, description="Ticker or asset symbol (e.g. BTC, ETH)")
    price: float = Field(..., gt=0, description="Asset price, must be strictly greater than 0")
    currency: str = Field(default="USD", min_length=1, description="Quote currency")
    source: str | None = Field(default=None, description="Ingestion source name (e.g. coingecko)")

    @field_validator("symbol", mode="after")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.strip().upper()


class VolumeSnapshot(BaseModel):
    """Strict validation schema for raw market volume snapshots."""

    model_config = ConfigDict(strict=True, extra="forbid")

    timestamp: datetime = Field(..., description="Timestamp of the volume snapshot in UTC")
    symbol: str = Field(..., min_length=1, description="Ticker or asset symbol (e.g. BTC, ETH)")
    volume: float = Field(..., ge=0, description="Trading volume, must be non-negative")
    currency: str | None = Field(default=None, description="Quote currency or volume unit")
    source: str | None = Field(default=None, description="Ingestion source name")

    @field_validator("symbol", mode="after")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.strip().upper()


class OrderBookEntry(BaseModel):
    """Single price-quantity level in an order book."""

    model_config = ConfigDict(strict=True, extra="forbid")

    price: float = Field(..., gt=0, description="Order price level, must be greater than 0")
    amount: float = Field(..., ge=0, description="Order quantity or volume at price level")


class OrderBookSnapshot(BaseModel):
    """Strict validation schema for raw market order-book snapshots."""

    model_config = ConfigDict(strict=True, extra="forbid")

    timestamp: datetime = Field(..., description="Timestamp of the snapshot in UTC")
    symbol: str = Field(..., min_length=1, description="Ticker or asset symbol (e.g. BTC, ETH)")
    bids: list[OrderBookEntry] = Field(
        default_factory=list, description="List of buy order levels (bids)"
    )
    asks: list[OrderBookEntry] = Field(
        default_factory=list, description="List of sell order levels (asks)"
    )
    source: str | None = Field(default=None, description="Ingestion source name")

    @field_validator("symbol", mode="after")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.strip().upper()
