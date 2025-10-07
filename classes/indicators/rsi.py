# classes/indicators/rsi.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _rsi_series(close: pd.Series, window: int) -> pd.Series:
    """
    Wilder-style RSI via simple rolling means (not EMA), robust to zero-loss/gain:
      - If avg_loss == 0 and avg_gain > 0 → RSI = 100
      - If avg_gain == 0 and avg_loss > 0 → RSI = 0
      - If both are 0                      → RSI = 50
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = pd.Series(np.nan, index=close.index, dtype="float64")
    nonzero = avg_loss != 0
    rs.loc[nonzero] = (avg_gain.loc[nonzero] / avg_loss.loc[nonzero]).astype("float64")

    rsi_val = 100 - (100 / (1 + rs))
    rsi_val = rsi_val.where(
        nonzero,
        np.where(
            avg_gain > 0, 100.0,
            np.where((avg_gain == 0) & (avg_loss == 0), 50.0, 0.0)
        )
    )
    return rsi_val

# ---------- Full-history compute ----------
def rsi_full(df: pd.DataFrame, window: int = 14, prefix: str = "rsi") -> None:
    if df is None or df.empty:
        return
    w = int(window)
    rsi_val = _rsi_series(df["close"], w)
    df.loc[:, f"{prefix}_{w}"] = rsi_val

# ---------- Fast last-row update ----------
def rsi_last_row(df: pd.DataFrame, window: int = 14, prefix: str = "rsi") -> None:
    """
    Recompute only the last row’s RSI using a bounded tail (window+1 bars).
    """
    if df is None or df.empty:
        return
    n = len(df)
    idx = df.index[-1]
    w = int(window)
    start = max(0, n - (w + 1))  # need +1 for diff() warmup
    sub = df.iloc[start:n]["close"]
    rsi_tail = _rsi_series(sub, w)
    df.at[idx, f"{prefix}_{w}"] = float(rsi_tail.iloc[-1])

# ---------- Per-index recompute ----------
def rsi_at_index(df: pd.DataFrame, idx: int, window: int = 14, prefix: str = "rsi") -> None:
    """
    Recompute RSI at a specific index using only bars up to idx (no look-ahead).
    """
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df):
        return
    w = int(window)
    start = max(0, idx - (w + 1) + 1)  # bounded prefix including one extra for diff()
    sub = df.iloc[start: idx + 1]["close"]
    rsi_here = _rsi_series(sub, w).iloc[-1]
    df.at[idx, f"{prefix}_{w}"] = float(rsi_here)

