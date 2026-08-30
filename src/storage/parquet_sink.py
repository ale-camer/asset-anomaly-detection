from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import Settings, get_settings


class ParquetStorageSink:
    """Persistence sink for saving market data DataFrames as partitioned Parquet datasets."""

    def __init__(
        self,
        base_dir: Path | str | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the Parquet storage sink.

        Args:
            base_dir: Base directory for storing parquet partitions. Defaults to settings.data_raw_dir.
            settings: Application settings instance. If None, uses default settings.
        """
        self.settings = settings or get_settings()
        self.base_dir = Path(base_dir) if base_dir else self.settings.data_raw_dir

    def write(
        self,
        df: pd.DataFrame,
        base_dir: Path | str | None = None,
        partition_cols: list[str] | None = None,
        **kwargs: Any,
    ) -> Path:
        """Write a DataFrame to partitioned Parquet files.

        Args:
            df: DataFrame containing market data.
            base_dir: Destination base directory. Defaults to self.base_dir.
            partition_cols: Columns used to partition dataset. Defaults to ['symbol', 'date'].
            **kwargs: Extra parameters passed to pandas `to_parquet`.

        Returns:
            Path: Destination base directory where data was written.

        Raises:
            ValueError: If df is empty or missing required timestamp/symbol fields.
        """
        if df.empty:
            raise ValueError("Cannot write an empty DataFrame to Parquet storage.")

        target_dir = Path(base_dir) if base_dir else self.base_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        data = df.copy()

        # Derive 'date' partition column from 'timestamp' if needed
        cols_to_partition = partition_cols if partition_cols is not None else ["symbol", "date"]

        if "date" in cols_to_partition and "date" not in data.columns:
            if "timestamp" not in data.columns:
                raise ValueError("DataFrame must contain 'timestamp' column to generate 'date' partition.")
            data["date"] = pd.to_datetime(data["timestamp"]).dt.strftime("%Y-%m-%d")

        for col in cols_to_partition:
            if col not in data.columns:
                raise ValueError(f"Partition column '{col}' is not present in DataFrame.")

        data.to_parquet(
            path=target_dir,
            engine="pyarrow",
            partition_cols=cols_to_partition,
            index=False,
            **kwargs,
        )

        return target_dir

    def read(
        self,
        base_dir: Path | str | None = None,
        columns: list[str] | None = None,
        filters: list[Any] | None = None,
    ) -> pd.DataFrame:
        """Read a partitioned Parquet dataset from storage.

        Args:
            base_dir: Destination base directory. Defaults to self.base_dir.
            columns: Specific column names to read.
            filters: PyArrow filters to push down into parquet read.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        target_dir = Path(base_dir) if base_dir else self.base_dir
        if not target_dir.exists():
            raise FileNotFoundError(f"Storage path {target_dir} does not exist.")

        return pd.read_parquet(
            path=target_dir,
            engine="pyarrow",
            columns=columns,
            filters=filters,
        )
