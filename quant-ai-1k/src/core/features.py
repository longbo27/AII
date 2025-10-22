"""Core feature engineering utilities for quant-ai-1k."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ensure_series(data: pd.Series | pd.DataFrame, column: str | None = None) -> pd.Series:
    """Ensure the input is a pandas Series.

    Args:
        data: Series or DataFrame containing price data.
        column: Optional column name when ``data`` is a DataFrame.

    Returns:
        Pandas Series extracted from the input data.
    """
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("Column name must be provided when data is a DataFrame.")
        return data[column]
    return data


def returns_5d(close: pd.Series | pd.DataFrame, column: str | None = None) -> pd.Series:
    """Compute 5-day simple returns.

    Args:
        close: Close price Series or DataFrame.
        column: Column name when ``close`` is a DataFrame.

    Returns:
        5-day percentage returns with NaNs for the initial lookback window.
    """
    series = _ensure_series(close, column)
    return series.pct_change(5)


def returns_20d(close: pd.Series | pd.DataFrame, column: str | None = None) -> pd.Series:
    """Compute 20-day simple returns."""
    series = _ensure_series(close, column)
    return series.pct_change(20)


def rsi_14(close: pd.Series | pd.DataFrame, column: str | None = None, window: int = 14) -> pd.Series:
    """Calculate the 14-day Relative Strength Index (RSI).

    The implementation uses Wilder's smoothing method. The initial window is
    backfilled with 50 to avoid extreme starting values.
    """
    series = _ensure_series(close, column)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)
    return rsi


def macd(
    close: pd.Series | pd.DataFrame,
    column: str | None = None,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Compute the Moving Average Convergence Divergence (MACD).

    Args:
        close: Close price Series or DataFrame.
        column: Column name when ``close`` is a DataFrame.
        fast: Fast EMA window.
        slow: Slow EMA window.
        signal: Signal line EMA window.

    Returns:
        DataFrame with ``macd`` (line), ``signal`` (signal line) and ``hist`` (histogram).
    """
    series = _ensure_series(close, column)

    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def atr_14(
    high: pd.Series | pd.DataFrame,
    low: pd.Series | pd.DataFrame,
    close: pd.Series | pd.DataFrame,
    high_col: str | None = None,
    low_col: str | None = None,
    close_col: str | None = None,
    window: int = 14,
) -> pd.Series:
    """Calculate the 14-day Average True Range (ATR).

    Args:
        high: High price Series or DataFrame.
        low: Low price Series or DataFrame.
        close: Close price Series or DataFrame.
        high_col: Column name for high when passing DataFrame.
        low_col: Column name for low when passing DataFrame.
        close_col: Column name for close when passing DataFrame.
        window: ATR window length.

    Returns:
        Series representing the ATR values.
    """
    high_series = _ensure_series(high, high_col)
    low_series = _ensure_series(low, low_col)
    close_series = _ensure_series(close, close_col)

    prev_close = close_series.shift(1)
    tr_components = pd.concat(
        [
            (high_series - low_series).abs(),
            (high_series - prev_close).abs(),
            (low_series - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)

    atr = true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    atr = atr.bfill()
    return atr


__all__ = [
    "returns_5d",
    "returns_20d",
    "rsi_14",
    "macd",
    "atr_14",
]
