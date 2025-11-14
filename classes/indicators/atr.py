# indicators/atr.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ---------- Shared helpers ----------

def _tr_series(df: pd.DataFrame) -> pd.Series:
    """
    True Range per bar, NaN-tolerant for synthetic bars (uses close when H/L missing).
    Writes nothing; returns a Series you can assign to df['tr'] if desired.
    """
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    high = df["high"].fillna(df["close"])
    low  = df["low"].fillna(df["close"])
    close = df["close"]
    prev_close = close.shift(1).fillna(close)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.astype("float64")


def ensure_tr(df: pd.DataFrame) -> None:
    """Ensure df['tr'] exists; compute in-place if missing."""
    if df is None or df.empty:
        return
    if "tr" not in df.columns or df["tr"].isna().all():
        df.loc[:, "tr"] = _tr_series(df)


# ---------- Full ATR (vectorized) ----------

def atr(df: pd.DataFrame, window: int = 14, prefix: str = "atr") -> None:
    """
    Compute TR and ATR(window) for the entire DataFrame (vectorized).
    - Writes df['tr'] and df[f'{prefix}_{window}'] in-place.
    """
    if df is None or df.empty:
        return
    tr = _tr_series(df)
    df.loc[:, "tr"] = tr
    df.loc[:, f"{prefix}_{window}"] = tr.rolling(int(window)).mean()


# ---------- Fast last-row updates (for live/open phase) ----------

def atr_last_row(df: pd.DataFrame, window: int = 14, prefix: str = "atr") -> None:
    """
    Update only the last row’s TR and ATR(window), in-place.
    Mirrors your existing last-row optimized path.
    """
    if df is None or df.empty:
        return
    n = len(df)
    idx = df.index[-1]
    # Guard: need at least 2 rows to form TR for the newest bar
    start = max(0, n - (window + 1))
    sub = df.iloc[start:n].copy()
    atr(sub, window=window, prefix=prefix)
    # Assign back only newest values
    df.at[idx, f"{prefix}_{window}"] = sub[f"{prefix}_{window}"].iloc[-1]
    if "tr" in sub.columns:
        df.at[idx, "tr"] = sub["tr"].iloc[-1]


def atr_at_index(df: pd.DataFrame, idx, window: int = 14, prefix: str = "atr") -> None:
    """
    Recompute ATR(window) for a specific index using only bars up to idx (no look-ahead).
    Writes df[f'{prefix}_{window}'] at idx and (if needed) df['tr'] at idx.
    """
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df):
        return
    start = max(0, idx - (window + 1) + 1)  # include prev bar for TR
    sub = df.iloc[start: idx + 1].copy()
    atr(sub, window=window, prefix=prefix)
    df.at[idx, f"{prefix}_{window}"] = sub[f"{prefix}_{window}"].iloc[-1]
    if "tr" in sub.columns:
        df.at[idx, "tr"] = sub["tr"].iloc[-1]


# ---------- ATR ratios (fast/slow & short/medium) ----------

def atr_ratio(df: pd.DataFrame, fast: int = 90, slow: int = 390, prefix: str = "atrR") -> None:
    """
    ATR fast/slow ratio for the full DataFrame.
    Uses TR internally; no look-ahead (rolling means).
    """
    if df is None or df.empty:
        return
    tr = _tr_series(df)
    atr_f = tr.rolling(int(fast)).mean()
    atr_s = tr.rolling(int(slow)).mean().replace(0, np.nan)
    df.loc[:, f"{prefix}_{fast}_{slow}"] = atr_f / atr_s


def atr_ratio_sm(df: pd.DataFrame, fast: int = 14, slow: int = 60, prefix: str = "atrRsm") -> None:
    """
    Fast/medium ATR ratio (e.g., 14 vs 60) for microstructure reactivity.
    """
    if df is None or df.empty:
        return
    tr = _tr_series(df)
    atr_f = tr.rolling(int(fast)).mean()
    atr_s = tr.rolling(int(slow)).mean().replace(0, np.nan)
    df.loc[:, f"{prefix}_{fast}_{slow}"] = atr_f / atr_s


# ---------- Last-row & at-index helpers for ratios ----------

def atr_ratio_last_row(df: pd.DataFrame, fast: int = 90, slow: int = 390, prefix: str = "atrR") -> None:
    """
    Update only the last row of ATR-ratio (fast/slow).
    """
    if df is None or df.empty:
        return
    n = len(df)
    idx = df.index[-1]
    look = max(int(fast), int(slow)) + 1
    start = max(0, n - look)
    sub = df.iloc[start:n].copy()
    atr_ratio(sub, fast=fast, slow=slow, prefix=prefix)
    df.at[idx, f"{prefix}_{fast}_{slow}"] = sub[f"{prefix}_{fast}_{slow}"].iloc[-1]


def atr_ratio_at_index(df: pd.DataFrame, idx, fast: int = 90, slow: int = 390, prefix: str = "atrR") -> None:
    """
    Recompute ATR-ratio (fast/slow) at a specific index using only past bars.
    """
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df):
        return
    look = max(int(fast), int(slow)) + 1
    start = max(0, idx - look + 1)
    sub = df.iloc[start: idx + 1].copy()
    atr_ratio(sub, fast=fast, slow=slow, prefix=prefix)
    df.at[idx, f"{prefix}_{fast}_{slow}"] = sub[f"{prefix}_{fast}_{slow}"].iloc[-1]


def atr_ratio_sm_last_row(df: pd.DataFrame, fast: int = 14, slow: int = 60, prefix: str = "atrRsm") -> None:
    if df is None or df.empty:
        return
    n = len(df)
    idx = df.index[-1]
    look = max(int(fast), int(slow)) + 1
    start = max(0, n - look)
    sub = df.iloc[start:n].copy()
    atr_ratio_sm(sub, fast=fast, slow=slow, prefix=prefix)
    df.at[idx, f"{prefix}_{fast}_{slow}"] = sub[f"{prefix}_{fast}_{slow}"].iloc[-1]


def atr_ratio_sm_at_index(df: pd.DataFrame, idx, fast: int = 14, slow: int = 60, prefix: str = "atrRsm") -> None:
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df):
        return
    look = max(int(fast), int(slow)) + 1
    start = max(0, idx - look + 1)
    sub = df.iloc[start: idx + 1].copy()
    atr_ratio_sm(sub, fast=fast, slow=slow, prefix=prefix)
    df.at[idx, f"{prefix}_{fast}_{slow}"] = sub[f"{prefix}_{fast}_{slow}"].iloc[-1]

