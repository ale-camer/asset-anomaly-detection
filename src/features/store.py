from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from src.features.models import FeatureSetRecord
from src.storage.parquet_sink import ParquetStorageSink
from src.utils.config import Settings, get_settings


class FeatureStoreSink(ParquetStorageSink):
    """Persistence sink for saving computed feature sets with strict schema validation."""

    def __init__(
        self,
        base_dir: Path | str | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the Feature Store sink.

        Args:
            base_dir: Base directory for storing features. Defaults to settings.data_features_dir.
            settings: Application settings instance.
        """
        self.settings = settings or get_settings()
        # Default to data_features_dir instead of data_raw_dir
        storage_dir = Path(base_dir) if base_dir else self.settings.data_features_dir
        super().__init__(base_dir=storage_dir, settings=self.settings)

    def write(
        self,
        df: pd.DataFrame,
        base_dir: Path | str | None = None,
        partition_cols: list[str] | None = None,
        **kwargs: Any,
    ) -> Path:
        """Validate and write a features DataFrame to Parquet storage.

        Args:
            df: DataFrame containing computed features.
            base_dir: Destination base directory. Defaults to self.base_dir.
            partition_cols: Columns used to partition dataset. Defaults to ['symbol', 'date'].
            **kwargs: Extra parameters passed to pandas `to_parquet`.

        Returns:
            Path: Destination base directory where features were written.

        Raises:
            ValueError: If df is empty.
            ValidationError: If any row fails schema validation.
        """
        if df.empty:
            raise ValueError("Cannot write an empty DataFrame to Feature Store.")

        # Validate DataFrame schema
        self._validate_features(df)

        # Call parent write method to handle partitioning and parquet persistence
        return super().write(
            df=df,
            base_dir=base_dir,
            partition_cols=partition_cols,
            **kwargs,
        )

    def _validate_features(self, df: pd.DataFrame) -> None:
        """Validate DataFrame against FeatureSetRecord Pydantic schema."""
        # Fast path for empty checks
        if df.empty:
            return

        # Iterate over records and validate strictly
        # Convert NaNs to None to allow Pydantic to validate Optionals properly
        records = df.replace({float("nan"): None}).to_dict(orient="records")
        for record in records:
            try:
                FeatureSetRecord(**record)
            except ValidationError as e:
                # Re-raise with context about the failing row
                raise ValidationError.from_exception_data(
                    title=f"Feature validation failed for symbol {record.get('symbol')} at {record.get('timestamp')}",
                    line_errors=e.errors(),  # type: ignore[arg-type]
                ) from e
