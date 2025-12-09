# classes/indicators/rsi.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def _rsi_series(close: pd.Series, window: int) -> pd.Series:
    """
    Traditional Wilder's RSI using initial SMA followed by exponential smoothing.
    Robust to zero-loss/gain:
      - If avg_loss == 0 and avg_gain > 0 → RSI = 100
      - If avg_gain == 0 and avg_loss > 0 → RSI = 0
      - If both are 0                      → RSI = 50
    Handles NaN deltas as zero change.
    """
    if close.empty or not is_numeric_dtype(close):
        return pd.Series(np.nan, index=close.index, dtype="float64")
    close = close.astype(np.float64)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    rsi = pd.Series(np.nan, index=close.index, dtype="float64")
    if len(close) <= window:
        return rsi
    # Initial averages over first window deltas (positions 1 to window)
    avg_gain = gain.iloc[1:window+1].mean()
    avg_loss = loss.iloc[1:window+1].mean()
    if np.isnan(avg_gain) or np.isnan(avg_loss):
        rsi.iloc[window] = np.nan
    elif avg_loss == 0:
        rsi.iloc[window] = 100.0 if avg_gain > 0 else 50.0
    else:
        rs = avg_gain / avg_loss
        rsi.iloc[window] = 100 - (100 / (1 + rs))
    # Smoothing for subsequent values
    for i in range(window + 1, len(close)):
        curr_gain = gain.iloc[i]
        curr_loss = loss.iloc[i]
        if np.isnan(curr_gain) or np.isnan(curr_loss):
            rsi.iloc[i] = np.nan
            continue
        avg_gain = (avg_gain * (window - 1) + curr_gain) / window
        avg_loss = (avg_loss * (window - 1) + curr_loss) / window
        if avg_loss == 0:
            rsi.iloc[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi.iloc[i] = 100 - (100 / (1 + rs))
    return rsi

# ---------- Full-history compute ----------
def rsi_full(df: pd.DataFrame, window: int = 14, prefix: str = "rsi") -> None:
    if df is None or df.empty or "close" not in df.columns:
        return
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonically increasing for RSI calculation.")
    w = int(window)
    rsi_val = _rsi_series(df["close"], w)
    df.loc[:, f"{prefix}_{w}"] = rsi_val

# ---------- Fast last-row update ----------
def rsi_last_row(df: pd.DataFrame, window: int = 14, prefix: str = "rsi") -> None:
    """
    Recompute only the last row’s RSI using full history up to the last row (no look-ahead).
    """
    if df is None or df.empty or "close" not in df.columns:
        return
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonically increasing for RSI calculation.")
    n = len(df)
    idx = df.index[-1]
    w = int(window)
    sub = df.iloc[0:n]["close"]
    rsi_tail = _rsi_series(sub, w)
    df.at[idx, f"{prefix}_{w}"] = float(rsi_tail.iloc[-1]) if not np.isnan(rsi_tail.iloc[-1]) else np.nan

# ---------- Per-index recompute ----------
def rsi_at_index(df: pd.DataFrame, idx: int, window: int = 14, prefix: str = "rsi") -> None:
    """
    Recompute RSI at a specific index using only bars up to idx (no look-ahead).
    """
    if df is None or df.empty or idx < 0 or idx >= len(df) or "close" not in df.columns:
        return
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonically increasing for RSI calculation.")
    w = int(window)
    sub = df.iloc[0: idx + 1]["close"]
    rsi_here = _rsi_series(sub, w).iloc[-1]
    df.at[df.index[idx], f"{prefix}_{w}"] = float(rsi_here) if not np.isnan(rsi_here) else np.nan
