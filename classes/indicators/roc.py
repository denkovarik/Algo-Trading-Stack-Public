# classes/indicators/roc.py
from __future__ import annotations
import numpy as np
import pandas as pd

def roc_full(df: pd.DataFrame, window: int = 30, prefix: str = "roc") -> None:
    """
    Full-history ROC (% change over N bars): (close / close.shift(N)) - 1
    Writes: {prefix}_{window}
    """
    if df is None or df.empty:
        return
    w = int(window)
    close = df["close"].astype("float64")
    prev = close.shift(w)
    # Avoid div-by-zero: where prev==0 -> NaN
    denom = prev.replace(0, np.nan)
    df.loc[:, f"{prefix}_{w}"] = (close / denom) - 1.0

def roc_last_row(df: pd.DataFrame, window: int = 30, prefix: str = "roc") -> None:
    """
    Fast last-row update (no look-ahead). If n <= window, falls back to full compute.
    """
    if df is None or df.empty:
        return
    n = len(df)
    idx = df.index[-1]
    w = int(window)

    if n <= w:
        # Not enough history to produce a finite ROC; mirror full path behavior.
        roc_full(df, window=w, prefix=prefix)
        return

    close_now = float(df["close"].iloc[-1])
    close_prev = float(df["close"].iloc[-1 - w])
    val = (close_now / close_prev) - 1.0 if close_prev != 0 else np.nan
    df.at[idx, f"{prefix}_{w}"] = val

def roc_at_index(df: pd.DataFrame, idx: int, window: int = 30, prefix: str = "roc") -> None:
    """
    Per-index recompute using only bars up to idx (no look-ahead).
    Writes value at idx; does not touch earlier rows.
    """
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df):
        return
    w = int(window)
    if idx - w < 0:
        # Not enough history at this index → mirror full behavior (NaN)
        df.at[idx, f"{prefix}_{w}"] = np.nan
        return

    close_now = float(df["close"].iloc[idx])
    close_prev = float(df["close"].iloc[idx - w])
    val = (close_now / close_prev) - 1.0 if close_prev != 0 else np.nan
    df.at[idx, f"{prefix}_{w}"] = val

