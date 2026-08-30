from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FeatureSetRecord(BaseModel):
    """Strict validation schema for computed market features."""

    model_config = ConfigDict(strict=True, extra="allow")

    # Base identifying fields
    timestamp: datetime = Field(..., description="Timestamp of the feature record in UTC")
    symbol: str = Field(..., min_length=1, description="Ticker or asset symbol")

    # Time-Series Rolling Statistics
    close_rolling_mean_7: float | None = Field(default=None, description="7-period rolling mean of close price")
    close_rolling_std_7: float | None = Field(default=None, description="7-period rolling std of close price")
    close_ema_7: float | None = Field(default=None, description="7-period EMA of close price")

    # Volatility
    volatility_parkinson_14: float | None = Field(default=None, ge=0, description="14-period Parkinson volatility")

    # Momentum
    rsi_14: float | None = Field(default=None, ge=0, le=100, description="14-period RSI")
    macd_line: float | None = Field(default=None, description="MACD line")
    macd_signal: float | None = Field(default=None, description="MACD signal line")
    macd_hist: float | None = Field(default=None, description="MACD histogram")

    # Velocity
    price_return_1: float | None = Field(default=None, description="1-period price return")
    price_velocity_1: float | None = Field(default=None, description="1-period price velocity")
