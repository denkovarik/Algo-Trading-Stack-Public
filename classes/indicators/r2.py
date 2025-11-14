# classes/indicators/r2.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

# --------------------- internals ---------------------

def _rolling_r2_numpy(y: np.ndarray) -> float:
    """
    Compute R^2 for a simple linear regression of y on x, where
    x = [0, 1, ..., len(y) - 1]. Returns NaN if variance is zero.
    """
    n = y.shape[0]
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    # center
    xm = x.mean()
    ym = y.mean()
    x0 = x - xm
    y0 = y - ym
    # correlation r = cov(x,y)/(sx*sy)
    denom = (np.sqrt((x0**2).sum()) * np.sqrt((y0**2).sum()))
    if denom == 0.0:
        return np.nan
    r = (x0 * y0).sum() / denom
    # R^2
    r2 = r * r
    return float(r2)

def _compute_r2_series(close: pd.Series, window: int) -> pd.Series:
    """Vectorized rolling R² over 'close' using numpy in a rolling.apply."""
    if "float" not in str(close.dtype):
        close = close.astype(float)
    r2 = close.rolling(window=window, min_periods=window).apply(
        lambda arr: _rolling_r2_numpy(arr), raw=True
    )
    return r2.astype(float)

def _col_r2(window: int) -> str:
    return f"r2_{int(window)}"

# --------------------- public API ---------------------

def r2_full(
    df: pd.DataFrame,
    window: int = 30,
    prefix: Optional[str] = None,  # kept for API symmetry (unused)
) -> None:
    """
    Compute rolling R² of linear regression of CLOSE vs [0..w-1] over 'window' bars.
    Writes: r2_<window>.
    """
    if "close" not in df.columns:
        raise ValueError("r2_full requires 'close' column in df")
    col = _col_r2(window)
    df[col] = _compute_r2_series(df["close"], window)


def r2_last_row(
    df: pd.DataFrame,
    window: int = 30,
    prefix: Optional[str] = None,
    lookback_factor: int = 3,
) -> None:
    """
    Fast tail update: recompute R² on a tail slice and write it back.
    """
    n = len(df)
    if n == 0:
        return
    lb = max(window * int(lookback_factor), window)
    start = max(0, n - lb)
    tail = df.iloc[start:].copy()
    r2 = _compute_r2_series(tail["close"], window)
    col = _col_r2(window)
    df.loc[tail.index, col] = r2


def r2_at_index(
    df: pd.DataFrame,
    idx: int,
    window: int = 30,
    prefix: Optional[str] = None,
    lookback_factor: int = 3,
) -> None:
    """
    Recompute R² ending at a specific index (inclusive) using a tail slice,
    then write the overlapping values back.
    """
    if idx is None or idx < 0 or idx >= len(df):
        return
    lb = max(window * int(lookback_factor), window)
    start = max(0, idx + 1 - lb)
    block = df.iloc[start:idx + 1].copy()
    r2 = _compute_r2_series(block["close"], window)
    col = _col_r2(window)
    df.loc[block.index, col] = r2

