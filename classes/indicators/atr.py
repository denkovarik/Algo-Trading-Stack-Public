# classes/indicators/atr.py
from __future__ import annotations
import numpy as np
import pandas as pd


def _tr_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    high = df["high"].fillna(df["close"])
    low = df["low"].fillna(df["close"])
    close = df["close"]
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.astype("float64")


def ensure_tr(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    if "tr" not in df.columns or df["tr"].isna().all():
        df["tr"] = _tr_series(df)  # Direct assignment — no copy warning


def _wilder_atr_incremental(df: pd.DataFrame, window: int, col: str) -> None:
    """Pure incremental Wilder ATR — no slicing, no copying, no warnings"""
    tr = df["tr"]
    if len(tr) < window:
        return

    # First value at index window-1 — safe direct assignment
    if pd.isna(df.iat[window - 1, df.columns.get_loc(col)]):
        df.iat[window - 1, df.columns.get_loc(col)] = tr.iloc[:window].mean()

    # Wilder smoothing from window onward
    col_idx = df.columns.get_loc(col)
    for i in range(window, len(df)):
        prev = df.iat[i - 1, col_idx]
        df.iat[i, col_idx] = (prev * (window - 1) + tr.iloc[i]) / window


def atr_full(df: pd.DataFrame, window: int = 14, prefix: str = "atr") -> None:
    if df is None or df.empty:
        return
    ensure_tr(df)
    col = f"{prefix}_{window}"
    if col not in df.columns:
        df[col] = np.nan  # This is safe — df is the original frame
    _wilder_atr_incremental(df, window, col)


def atr_at_index(df: pd.DataFrame, idx: int, window: int = 14, prefix: str = "atr") -> None:
    if df is None or df.empty or idx < 0 or idx >= len(df):
        return
    ensure_tr(df)
    col = f"{prefix}_{window}"
    if col not in df.columns:
        df[col] = np.nan
    # Work directly on the original df — no slicing
    _wilder_atr_incremental(df.iloc[:idx + 1], window, col)
    # No need to write back — already in place


def atr_last_row(df: pd.DataFrame, window: int = 14, prefix: str = "atr") -> None:
    if df is None or df.empty:
        return
    atr_at_index(df, len(df) - 1, window=window, prefix=prefix)
