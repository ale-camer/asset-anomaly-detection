from typing import Any

import numpy as np
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


class VolatilityFeatures(BaseEstimator, TransformerMixin):
    """Transformer for calculating Parkinson historical volatility based on High and Low prices."""

    def __init__(
        self,
        windows: list[int] | None = None,
        high_col: str = "high",
        low_col: str = "low",
        group_by: str | None = "symbol",
        min_periods: int = 1,
    ) -> None:
        """Initialize Parkinson volatility feature extractor.

        Args:
            windows: Rolling window sizes in periods. Defaults to [7, 14].
            high_col: High price column name. Defaults to 'high'.
            low_col: Low price column name. Defaults to 'low'.
            group_by: Column to group by before calculating volatility. Defaults to 'symbol'.
            min_periods: Minimum observations for rolling calculation. Defaults to 1.
        """
        self.windows = windows if windows is not None else [7, 14]
        self.high_col = high_col
        self.low_col = low_col
        self.group_by = group_by
        self.min_periods = min_periods

    def fit(self, X: pd.DataFrame, y: Any = None) -> "VolatilityFeatures":
        """Fit transformer (stateless)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate Parkinson volatility over specified rolling windows.

        Args:
            X: Input DataFrame containing high and low price columns.

        Returns:
            pd.DataFrame: DataFrame augmented with Parkinson volatility features.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if X.empty:
            raise ValueError("Input DataFrame cannot be empty.")
        if self.high_col not in X.columns or self.low_col not in X.columns:
            raise ValueError(f"DataFrame must contain '{self.high_col}' and '{self.low_col}' columns.")

        df = X.copy()

        if "timestamp" in df.columns:
            sort_cols = [self.group_by, "timestamp"] if (self.group_by and self.group_by in df.columns) else ["timestamp"]
            df = df.sort_values(sort_cols).reset_index(drop=True)

        use_group = bool(self.group_by and self.group_by in df.columns)

        # Handle cases where high or low might be zero or negative by clipping
        high_vals = np.maximum(df[self.high_col].to_numpy(dtype=float), 1e-8)
        low_vals = np.maximum(df[self.low_col].to_numpy(dtype=float), 1e-8)

        # Parkinson single-period metric: (ln(High / Low))^2 / (4 * ln(2))
        log_hl = np.log(high_vals / low_vals)
        parkinson_instant = (log_hl ** 2) / (4.0 * np.log(2.0))
        df["_parkinson_instant"] = parkinson_instant

        for w in self.windows:
            col_name = f"volatility_parkinson_{w}"
            if use_group and self.group_by:
                grouped = df.groupby(self.group_by)["_parkinson_instant"]
                rolling_metric = grouped.transform(
                    lambda s, window=w: s.rolling(window, min_periods=self.min_periods).mean()
                )
            else:
                rolling_metric = df["_parkinson_instant"].rolling(w, min_periods=self.min_periods).mean()

            df[col_name] = np.sqrt(np.maximum(rolling_metric.to_numpy(dtype=float), 0.0))

        df.drop(columns=["_parkinson_instant"], inplace=True)
        return df


class MomentumFeatures(BaseEstimator, TransformerMixin):
    """Transformer for calculating Relative Strength Index (RSI) and MACD."""

    def __init__(
        self,
        close_col: str = "close",
        rsi_window: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        group_by: str | None = "symbol",
    ) -> None:
        """Initialize Momentum features extractor.

        Args:
            close_col: Close price column name. Defaults to 'close'.
            rsi_window: Period for RSI calculation. Defaults to 14.
            macd_fast: Fast EMA period for MACD. Defaults to 12.
            macd_slow: Slow EMA period for MACD. Defaults to 26.
            macd_signal: Signal line EMA period for MACD. Defaults to 9.
            group_by: Column to group by. Defaults to 'symbol'.
        """
        self.close_col = close_col
        self.rsi_window = rsi_window
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.group_by = group_by

    def fit(self, X: pd.DataFrame, y: Any = None) -> "MomentumFeatures":
        """Fit transformer (stateless)."""
        return self

    def _calc_rsi_series(self, series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(span=self.rsi_window, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_window, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    def _calc_macd_series(self, series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = series.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        return macd_line, macd_signal, macd_hist

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI and MACD momentum indicators.

        Args:
            X: Input DataFrame containing close price column.

        Returns:
            pd.DataFrame: DataFrame augmented with RSI and MACD features.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if X.empty:
            raise ValueError("Input DataFrame cannot be empty.")
        if self.close_col not in X.columns:
            raise ValueError(f"DataFrame must contain '{self.close_col}' column.")

        df = X.copy()

        if "timestamp" in df.columns:
            sort_cols = [self.group_by, "timestamp"] if (self.group_by and self.group_by in df.columns) else ["timestamp"]
            df = df.sort_values(sort_cols).reset_index(drop=True)

        use_group = bool(self.group_by and self.group_by in df.columns)

        if use_group and self.group_by:
            grouped = df.groupby(self.group_by)[self.close_col]
            df[f"rsi_{self.rsi_window}"] = grouped.transform(self._calc_rsi_series)

            # MACD
            ema_fast = grouped.transform(lambda s: s.ewm(span=self.macd_fast, adjust=False).mean())
            ema_slow = grouped.transform(lambda s: s.ewm(span=self.macd_slow, adjust=False).mean())
            macd_line = ema_fast - ema_slow
            df["macd_line"] = macd_line

            df["macd_signal"] = df.groupby(self.group_by)["macd_line"].transform(
                lambda s: s.ewm(span=self.macd_signal, adjust=False).mean()
            )
            df["macd_hist"] = df["macd_line"] - df["macd_signal"]
        else:
            df[f"rsi_{self.rsi_window}"] = self._calc_rsi_series(df[self.close_col])
            macd_line, macd_signal, macd_hist = self._calc_macd_series(df[self.close_col])
            df["macd_line"] = macd_line
            df["macd_signal"] = macd_signal
            df["macd_hist"] = macd_hist

        return df


class PriceVelocityFeatures(BaseEstimator, TransformerMixin):
    """Transformer for calculating returns and price acceleration (velocity of change)."""

    def __init__(
        self,
        periods: list[int] | None = None,
        close_col: str = "close",
        group_by: str | None = "symbol",
    ) -> None:
        """Initialize Price Velocity feature extractor.

        Args:
            periods: List of lag periods for returns and velocity. Defaults to [1, 3, 5].
            close_col: Close price column name. Defaults to 'close'.
            group_by: Column to group by. Defaults to 'symbol'.
        """
        self.periods = periods if periods is not None else [1, 3, 5]
        self.close_col = close_col
        self.group_by = group_by

    def fit(self, X: pd.DataFrame, y: Any = None) -> "PriceVelocityFeatures":
        """Fit transformer (stateless)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate price returns and velocity indicators.

        Args:
            X: Input DataFrame containing close price column.

        Returns:
            pd.DataFrame: DataFrame augmented with return and velocity features.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if X.empty:
            raise ValueError("Input DataFrame cannot be empty.")
        if self.close_col not in X.columns:
            raise ValueError(f"DataFrame must contain '{self.close_col}' column.")

        df = X.copy()

        if "timestamp" in df.columns:
            sort_cols = [self.group_by, "timestamp"] if (self.group_by and self.group_by in df.columns) else ["timestamp"]
            df = df.sort_values(sort_cols).reset_index(drop=True)

        use_group = bool(self.group_by and self.group_by in df.columns)

        for p in self.periods:
            ret_col = f"price_return_{p}"
            vel_col = f"price_velocity_{p}"

            if use_group and self.group_by:
                grouped = df.groupby(self.group_by)[self.close_col]
                df[ret_col] = grouped.transform(lambda s, period=p: s.pct_change(period)).fillna(0.0)
                df[vel_col] = (
                    df.groupby(self.group_by)[ret_col]
                    .transform(lambda s: s.diff())
                    .fillna(0.0)
                )
            else:
                df[ret_col] = df[self.close_col].pct_change(p).fillna(0.0)
                df[vel_col] = df[ret_col].diff().fillna(0.0)

        return df
