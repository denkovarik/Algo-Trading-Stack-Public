# classes/indicators/donchian.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ========== Donchian Channels (high/low/mid/width) ==========

def donchian_full(df: pd.DataFrame, window: int = 60, prefix: str = "dc") -> None:
    """
    Full, vectorized Donchian channels.
    Writes: {prefix}_high, {prefix}_low, {prefix}_mid, {prefix}_width
    """
    if df is None or df.empty:
        return
    w = int(window)
    hi = df["high"].rolling(w).max()
    lo = df["low"].rolling(w).min()
    mid = (hi + lo) / 2.0
    width = (hi - lo) / mid.replace(0, np.nan)
    df.loc[:, f"{prefix}_high"]  = hi
    df.loc[:, f"{prefix}_low"]   = lo
    df.loc[:, f"{prefix}_mid"]   = mid
    df.loc[:, f"{prefix}_width"] = width

def donchian_last_row(df: pd.DataFrame, window: int = 60, prefix: str = "dc") -> None:
    """
    Fast last-row update; if rows < window, falls back to full compute.
    """
    if df is None or df.empty:
        return
    n = len(df); idx = df.index[-1]; w = int(window)
    if n < w:
        donchian_full(df, window=w, prefix=prefix)
        return
    hiw = float(np.nanmax(df["high"].iloc[n-w:n].to_numpy(dtype="float64", copy=False)))
    loww = float(np.nanmin(df["low" ].iloc[n-w:n].to_numpy(dtype="float64", copy=False)))
    mid = (hiw + loww) / 2.0
    denom = (mid if mid != 0 else np.nan)
    width = (hiw - loww) / denom if denom == denom else np.nan
    df.at[idx, f"{prefix}_high"]  = hiw
    df.at[idx, f"{prefix}_low"]   = loww
    df.at[idx, f"{prefix}_mid"]   = mid
    df.at[idx, f"{prefix}_width"] = width

def donchian_at_index(df: pd.DataFrame, idx: int, window: int = 60, prefix: str = "dc") -> None:
    """
    Per-index recompute (no look-ahead): use only bars up to idx.
    """
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df):
        return
    w = int(window)
    start = max(0, idx - w + 1)
    highs = df["high"].iloc[start:idx+1]
    lows  = df["low" ].iloc[start:idx+1]
    hiw = float(np.nanmax(highs.to_numpy(dtype="float64", copy=False)))
    loww = float(np.nanmin(lows .to_numpy(dtype="float64", copy=False)))
    mid = (hiw + loww) / 2.0
    denom = (mid if mid != 0 else np.nan)
    width = (hiw - loww) / denom if denom == denom else np.nan
    df.at[idx, f"{prefix}_high"]  = hiw
    df.at[idx, f"{prefix}_low"]   = loww
    df.at[idx, f"{prefix}_mid"]   = mid
    df.at[idx, f"{prefix}_width"] = width

# ========== Donchian Position (pos ∈ [0,1]) ==========

def donchian_pos_full(df: pd.DataFrame, window: int = 60, prefix: str = "dc") -> None:
    """
    Full compute of Donchian position: (close - low) / (high - low)
    Requires channels; will call donchian_full if missing.
    Writes: {prefix}_pos
    """
    hi_col, lo_col = f"{prefix}_high", f"{prefix}_low"
    if hi_col not in df.columns or lo_col not in df.columns:
        donchian_full(df, window=window, prefix=prefix)
    hi = df[hi_col]; lo = df[lo_col]
    rng = (hi - lo).replace(0, np.nan)
    df.loc[:, f"{prefix}_pos"] = (df["close"] - lo) / rng

def donchian_pos_last_row(df: pd.DataFrame, window: int = 60, prefix: str = "dc") -> None:
    """
    Fast last-row update of Donchian position; ensures channels exist on last row.
    """
    if df is None or df.empty:
        return
    idx = df.index[-1]
    hi_col, lo_col = f"{prefix}_high", f"{prefix}_low"
    if (hi_col not in df.columns) or (lo_col not in df.columns) \
       or pd.isna(df.iloc[-1].get(hi_col, np.nan)) or pd.isna(df.iloc[-1].get(lo_col, np.nan)):
        donchian_last_row(df, window=window, prefix=prefix)
    hi = float(df.at[idx, hi_col]); lo = float(df.at[idx, lo_col]); c = float(df.at[idx, "close"])
    rng = hi - lo
    df.at[idx, f"{prefix}_pos"] = (c - lo) / rng if rng != 0 else np.nan

def donchian_pos_at_index(df: pd.DataFrame, idx: int, window: int = 60, prefix: str = "dc") -> None:
    """
    Per-index recompute of Donchian position; ensures channels at idx exist.
    """
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df):
        return
    hi_col, lo_col = f"{prefix}_high", f"{prefix}_low"
    if (hi_col not in df.columns) or pd.isna(df.iloc[idx].get(hi_col, np.nan)):
        donchian_at_index(df, idx, window=window, prefix=prefix)
    hi = float(df.at[idx, hi_col]); lo = float(df.at[idx, lo_col]); c = float(df.at[idx, "close"])
    rng = hi - lo
    df.at[idx, f"{prefix}_pos"] = (c - lo) / rng if rng != 0 else np.nan

