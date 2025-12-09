# classes/indicators/bollinger.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from .sma import sma_at_index, sma_last_row, sma_full  # Reuse your SMA

def bollinger_at_index(df: pd.DataFrame, idx: int, window: int = 20, std_mult: float = 2.0, prefix: str = "bb") -> None:
    if df is None or df.empty or idx < 0 or idx >= len(df) or "close" not in df.columns or not is_numeric_dtype(df["close"]):
        return
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonically increasing for Bollinger Bands calculation.")
    mid_col = f"{prefix}_mid_{window}"
    upper_col = f"{prefix}_upper_{window}"
    lower_col = f"{prefix}_lower_{window}"
    width_col = f"{prefix}_width_{window}"
    for c in [mid_col, upper_col, lower_col, width_col]:
        if c not in df.columns:
            df[c] = np.nan
    # Compute SMA mid (reuse)
    sma_at_index(df, idx, window=window, prefix=f"{prefix}_mid")
    mid_val = df.iat[idx, df.columns.get_loc(mid_col)]
    if pd.isna(mid_val):
        return
    # Std dev over window up to idx
    start = max(0, idx - window + 1)
    series = df["close"].iloc[start:idx+1]
    std_val = series.std(ddof=0) if len(series) > 1 else 0.0
    upper = mid_val + (std_val * std_mult)
    lower = mid_val - (std_val * std_mult)
    width = (upper - lower) / mid_val if mid_val != 0 else np.nan
    df.iat[idx, df.columns.get_loc(upper_col)] = upper
    df.iat[idx, df.columns.get_loc(lower_col)] = lower
    df.iat[idx, df.columns.get_loc(width_col)] = width

def bollinger_last_row(df: pd.DataFrame, window: int = 20, std_mult: float = 2.0, prefix: str = "bb") -> None:
    if df is not None and not df.empty:
        bollinger_at_index(df, len(df)-1, window=window, std_mult=std_mult, prefix=prefix)

def bollinger_full(df: pd.DataFrame, window: int = 20, std_mult: float = 2.0, prefix: str = "bb") -> None:
    if df is None or df.empty or "close" not in df.columns or not is_numeric_dtype(df["close"]):
        return
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonically increasing for Bollinger Bands calculation.")
    mid_col = f"{prefix}_mid_{window}"
    upper_col = f"{prefix}_upper_{window}"
    lower_col = f"{prefix}_lower_{window}"
    width_col = f"{prefix}_width_{window}"
    sma_full(df, window=window, prefix=f"{prefix}_mid")
    std = df["close"].rolling(window, min_periods=1).std(ddof=0)
    df[upper_col] = df[mid_col] + (std * std_mult)
    df[lower_col] = df[mid_col] - (std * std_mult)
    df[width_col] = (df[upper_col] - df[lower_col]) / df[mid_col]
