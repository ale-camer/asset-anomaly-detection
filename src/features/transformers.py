from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TimeSeriesRollingFeatures(BaseEstimator, TransformerMixin):
    """Transformer for calculating time-series rolling statistics and EMA on market data."""

    def __init__(
        self,
        windows: list[int] | None = None,
        columns: list[str] | None = None,
        group_by: str | None = "symbol",
        min_periods: int = 1,
    ) -> None:
        """Initialize the time-series rolling features transformer.

        Args:
            windows: List of window sizes (e.g. [7, 14, 30]). Defaults to [7, 14].
            columns: Target numeric column names to compute features on. Defaults to ['close', 'volume'].
            group_by: Column to group by before rolling (e.g. 'symbol'). Defaults to 'symbol'.
            min_periods: Minimum observations required to produce a value. Defaults to 1.
        """
        self.windows = windows if windows is not None else [7, 14]
        self.columns = columns if columns is not None else ["close", "volume"]
        self.group_by = group_by
        self.min_periods = min_periods

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TimeSeriesRollingFeatures":
        """Fit the transformer on the input DataFrame (stateless).

        Args:
            X: Input market DataFrame.
            y: Ignored (scikit-learn compatibility).

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling statistics and exponential moving averages on input DataFrame.

        Args:
            X: Input DataFrame containing market data.

        Returns:
            pd.DataFrame: DataFrame augmented with rolling and EMA features.

        Raises:
            TypeError: If input is not a pandas DataFrame.
            ValueError: If DataFrame is empty.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if X.empty:
            raise ValueError("Input DataFrame cannot be empty.")

        df = X.copy()

        # Sort chronologically if timestamp exists to ensure valid time-series operations
        if "timestamp" in df.columns:
            sort_cols = [self.group_by, "timestamp"] if (self.group_by and self.group_by in df.columns) else ["timestamp"]
            df = df.sort_values(sort_cols).reset_index(drop=True)

        use_group = bool(self.group_by and self.group_by in df.columns)

        for col in self.columns:
            if col not in df.columns:
                continue

            for w in self.windows:
                mean_col = f"{col}_rolling_mean_{w}"
                std_col = f"{col}_rolling_std_{w}"
                ema_col = f"{col}_ema_{w}"

                if use_group and self.group_by:
                    grouped = df.groupby(self.group_by)[col]
                    df[mean_col] = grouped.transform(
                        lambda s, window=w: s.rolling(window, min_periods=self.min_periods).mean()
                    )
                    df[std_col] = (
                        grouped.transform(
                            lambda s, window=w: s.rolling(window, min_periods=self.min_periods).std()
                        )
                        .fillna(0.0)
                    )
                    df[ema_col] = grouped.transform(
                        lambda s, window=w: s.ewm(span=window, adjust=False).mean()
                    )
                else:
                    df[mean_col] = df[col].rolling(w, min_periods=self.min_periods).mean()
                    df[std_col] = df[col].rolling(w, min_periods=self.min_periods).std().fillna(0.0)
                    df[ema_col] = df[col].ewm(span=w, adjust=False).mean()

        return df
