import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.ingestion.base import BaseConnector, MarketDataRecord
from src.utils.config import Settings

logger = logging.getLogger(__name__)


def _is_retryable_exception(exc: BaseException) -> bool:
    """Determine if an exception should trigger a retry."""
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        # Retry on Rate Limit (429) and Server Errors (5xx)
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


class CoinGeckoConnector(BaseConnector):
    """CoinGecko API connector for crypto market data ingestion."""

    def __init__(
        self,
        settings: Settings | None = None,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize CoinGecko connector.

        Args:
            settings: Settings instance.
            rate_limit_delay: Delay in seconds between API calls for rate limiting.
            max_retries: Maximum retry attempts for transient errors.
        """
        super().__init__(settings=settings)
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.base_url = self.settings.coingecko_api_base_url.rstrip("/")

    def _get_headers(self) -> dict[str, str]:
        """Build headers including optional API key."""
        headers = {"Accept": "application/json"}
        if self.settings.coingecko_api_key:
            headers["x-cg-demo-api-key"] = self.settings.coingecko_api_key
        return headers

    def _parse_timestamp(self, dt: datetime | str) -> int:
        """Convert datetime or ISO string to UNIX timestamp (seconds)."""
        if isinstance(dt, str):
            parsed_dt = pd.to_datetime(dt)
            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.tz_localize(UTC)
            return int(parsed_dt.timestamp())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp())

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable_exception),
        reraise=True,
    )
    def _make_request(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        """Perform HTTP GET request with retries and rate limiting."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers()

        time.sleep(self.rate_limit_delay)

        with httpx.Client(timeout=15.0) as client:
            response = client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return cast(dict[str, Any], response.json())

    def fetch_data(
        self,
        symbol: str,
        start_date: datetime | str,
        end_date: datetime | str,
        vs_currency: str = "usd",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch historical market data from CoinGecko API.

        Args:
            symbol: Coin ID (e.g. 'bitcoin', 'ethereum').
            start_date: Start date for historical range.
            end_date: End date for historical range.
            vs_currency: Target currency (default: 'usd').
            **kwargs: Extra parameters passed to the request.

        Returns:
            pd.DataFrame: Raw DataFrame containing timestamp, prices, market_caps, and total_volumes.
        """
        from_timestamp = self._parse_timestamp(start_date)
        to_timestamp = self._parse_timestamp(end_date)

        endpoint = f"coins/{symbol.lower()}/market_chart/range"
        params: dict[str, Any] = {
            "vs_currency": vs_currency.lower(),
            "from": from_timestamp,
            "to": to_timestamp,
            **kwargs,
        }

        logger.info(
            "Fetching CoinGecko data for %s from %s to %s",
            symbol,
            from_timestamp,
            to_timestamp,
        )
        data = self._make_request(endpoint=endpoint, params=params)

        prices = data.get("prices", [])
        market_caps = dict(data.get("market_caps", []))
        total_volumes = dict(data.get("total_volumes", []))

        records: list[dict[str, Any]] = []
        for item in prices:
            ts_ms, price = item[0], item[1]
            records.append(
                {
                    "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True),
                    "symbol": symbol.upper(),
                    "close": float(price),
                    "market_cap": (
                        float(market_caps[ts_ms]) if ts_ms in market_caps else None
                    ),
                    "volume": (
                        float(total_volumes[ts_ms]) if ts_ms in total_volumes else None
                    ),
                    "open": None,
                    "high": None,
                    "low": None,
                }
            )

        if not records:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "market_cap",
                ]
            )

        return pd.DataFrame(records)

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate DataFrame against MarketDataRecord schema.

        Args:
            df: Raw DataFrame returned by fetch_data.

        Returns:
            pd.DataFrame: Validated DataFrame.

        Raises:
            ValueError: If validation fails or DataFrame is empty.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty, validation failed.")

        validated_records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            record_dict = row.to_dict()
            # Convert pandas Timestamp to Python datetime if necessary
            if isinstance(record_dict.get("timestamp"), pd.Timestamp):
                record_dict["timestamp"] = record_dict["timestamp"].to_pydatetime()

            validated_model = MarketDataRecord.model_validate(record_dict)
            validated_records.append(validated_model.model_dump())

        validated_df = pd.DataFrame(validated_records)
        validated_df["timestamp"] = pd.to_datetime(validated_df["timestamp"], utc=True)
        return validated_df

    def save_data(
        self,
        df: pd.DataFrame,
        output_path: Path | str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Save market data DataFrame to Parquet format.

        Args:
            df: Validated market data DataFrame.
            output_path: Target path to save the parquet file.
            **kwargs: Extra arguments to pass to pandas to_parquet.

        Returns:
            Path: Path of the saved Parquet file.
        """
        if output_path is None:
            symbol = df["symbol"].iloc[0] if not df.empty and "symbol" in df else "crypto"
            now_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            target_dir = Path(self.settings.data_raw_dir)
            target_file = target_dir / f"{symbol.lower()}_{now_str}.parquet"
        else:
            target_file = Path(output_path)

        target_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(target_file, index=False, **kwargs)
        logger.info("Saved data to %s", target_file)
        return target_file
