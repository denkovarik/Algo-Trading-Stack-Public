# classes/indicators/bollinger.py
from __future__ import annotations
import numpy as np
import pandas as pd

# -------- Bollinger bands (MA, upper, lower) --------

def bollinger_full(df: pd.DataFrame, window: int = 20, num_std: float = 2.0, prefix: str = "bb") -> None:
    """
    Vectorized full compute of MA/upper/lower over the whole DataFrame.
    Writes: {prefix}_ma, {prefix}_upper, {prefix}_lower
    """
    if df is None or df.empty:
        return
    win = int(window)
    close = df["close"]
    ma = close.rolling(win).mean()
    sd = close.rolling(win).std()
    df.loc[:, f"{prefix}_ma"]    = ma
    df.loc[:, f"{prefix}_upper"] = ma + float(num_std) * sd
    df.loc[:, f"{prefix}_lower"] = ma - float(num_std) * sd


def bollinger_last_row(df: pd.DataFrame, window: int = 20, num_std: float = 2.0, prefix: str = "bb") -> None:
    """
    Fast last-row update (no look-ahead).
    If there aren't enough rows for the window, falls back to full compute.
    """
    if df is None or df.empty:
        return
    n = len(df)
    idx = df.index[-1]
    win = int(window)
    if n < win:
        bollinger_full(df, window=win, num_std=num_std, prefix=prefix)
        return

    close_tail = df["close"].iloc[n - win : n]
    w = close_tail.to_numpy(dtype="float64", copy=False)
    ma = float(np.nanmean(w))
    sd = float(np.nanstd(w, ddof=1))
    df.at[idx, f"{prefix}_ma"]    = ma
    df.at[idx, f"{prefix}_upper"] = ma + float(num_std) * sd
    df.at[idx, f"{prefix}_lower"] = ma - float(num_std) * sd


def bollinger_at_index(df: pd.DataFrame, idx: int, window: int = 20, num_std: float = 2.0, prefix: str = "bb") -> None:
    """
    Per-index recompute using only bars up to idx (no look-ahead).
    """
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df):
        return
    win = int(window)
    start = max(0, idx - win + 1)
    sub = df.iloc[start : idx + 1]["close"]
    ma = float(sub.rolling(win).mean().iloc[-1])
    sd = float(sub.rolling(win).std().iloc[-1])
    df.at[idx, f"{prefix}_ma"]    = ma
    df.at[idx, f"{prefix}_upper"] = ma + float(num_std) * sd
    df.at[idx, f"{prefix}_lower"] = ma - float(num_std) * sd

# -------- Bollinger bandwidth --------

def bandwidth_full(df: pd.DataFrame, window: int = 60, prefix: str = "bb") -> None:
    """
    Full compute of bandwidth: (upper - lower) / ma
    Ensures bands exist by calling bollinger_full if needed.
    Writes: {prefix}_bw
    """
    ma_col, up_col, lo_col = f"{prefix}_ma", f"{prefix}_upper", f"{prefix}_lower"
    if ma_col not in df.columns or up_col not in df.columns or lo_col not in df.columns:
        bollinger_full(df, window=window, num_std=2.0, prefix=prefix)
    ma = df[ma_col]
    up = df[up_col]
    lo = df[lo_col]
    df.loc[:, f"{prefix}_bw"] = (up - lo) / ma.replace(0, np.nan)

def bandwidth_last_row(df: pd.DataFrame, window: int = 60, prefix: str = "bb") -> None:
    """
    Fast last-row update of bandwidth; ensures bands on last row exist.
    """
    if df is None or df.empty:
        return
    idx = df.index[-1]
    ma_col, up_col, lo_col = f"{prefix}_ma", f"{prefix}_upper", f"{prefix}_lower"
    need_bands = (
        (ma_col not in df.columns) or (up_col not in df.columns) or (lo_col not in df.columns) or
        pd.isna(df.at[idx, ma_col]) or pd.isna(df.at[idx, up_col]) or pd.isna(df.at[idx, lo_col])
    )
    if need_bands:
        bollinger_last_row(df, window=window, num_std=2.0, prefix=prefix)

    ma = df.at[idx, ma_col]
    up = df.at[idx, up_col]
    lo = df.at[idx, lo_col]
    denom = (ma if (ma is not None and not pd.isna(ma) and ma != 0) else np.nan)
    df.at[idx, f"{prefix}_bw"] = (up - lo) / denom if denom == denom else np.nan

def bandwidth_at_index(df: pd.DataFrame, idx: int, window: int = 60, prefix: str = "bb") -> None:
    """
    Per-index recompute of bandwidth; ensures bands at idx exist.
    """
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df):
        return
    ma_col, up_col, lo_col = f"{prefix}_ma", f"{prefix}_upper", f"{prefix}_lower"
    if (ma_col not in df.columns) or pd.isna(df.iloc[idx].get(ma_col, np.nan)):
        bollinger_at_index(df, idx, window=window, num_std=2.0, prefix=prefix)
    ma = df.at[idx, ma_col]
    up = df.at[idx, up_col]
    lo = df.at[idx, lo_col]
    denom = (ma if (ma is not None and not pd.isna(ma) and ma != 0) else np.nan)
    df.at[idx, f"{prefix}_bw"] = (up - lo) / denom if denom == denom else np.nan

