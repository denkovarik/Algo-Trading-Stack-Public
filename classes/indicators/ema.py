# classes/indicators/ema.py
from __future__ import annotations
import numpy as np
import pandas as pd

def ema_full(df: pd.DataFrame, span: int = 21, prefix: str = "ema") -> None:
    """
    Full-history EMA: writes df[f'{prefix}_{span}'] over the entire frame.
    """
    if df is None or df.empty:
        return
    span = int(span)
    df.loc[:, f"{prefix}_{span}"] = df["close"].ewm(span=span, adjust=False).mean()

def ema_last_row(df: pd.DataFrame, span: int = 21, prefix: str = "ema") -> None:
    """
    Fast last-row update: O(1) online update if prev EMA exists; otherwise seed from tail.
    """
    if df is None or df.empty:
        return
    span = int(span)
    col = f"{prefix}_{span}"
    n = len(df)
    idx = df.index[-1]

    # O(1) update if previous EMA exists
    if col in df.columns and n >= 2 and pd.notna(df.iloc[-2].get(col, np.nan)):
        prev = float(df.iloc[-2][col])
        price = float(df["close"].iloc[-1])
        alpha = 2.0 / (span + 1.0)
        df.at[idx, col] = prev + alpha * (price - prev)
        return

    # Fallback: compute from minimal tail window (<= span)
    start = max(0, n - span)
    tail = df.iloc[start:n]["close"]
    if tail.size:
        df.at[idx, col] = tail.ewm(span=span, adjust=False).mean().iloc[-1]

def ema_at_index(df: pd.DataFrame, idx: int, span: int = 21, prefix: str = "ema") -> None:
    """
    Recompute EMA at a specific index using only bars up to idx (no look-ahead),
    matching pandas ewm(adjust=False) seeding behavior.
    """
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df):
        return
    span = int(span)
    # IMPORTANT: use the full prefix up to idx so the EMA seed matches pandas
    sub = df.iloc[: idx + 1]["close"]
    df.at[idx, f"{prefix}_{span}"] = sub.ewm(span=span, adjust=False).mean().iloc[-1]

# OPTIONAL extras used by trainer if present
def zscore_close_vs_ema(df, ema_span=60, prefix="z_close_ema60"):
    col = f"ema_{ema_span}"
    if col not in df.columns:
        ema(df, span=ema_span, prefix="ema")
    dev = (df["close"] - df[col])
    mu = dev.rolling(390).mean()
    sd = dev.rolling(390).std().replace(0, np.nan)
    df.loc[:, prefix] = (dev - mu) / sd

def zscore_ema60_vs_ema120(df, prefix="z_ema60_ema120"):
    if "ema_60" not in df.columns: ema(df, span=60, prefix="ema")
    if "ema_120" not in df.columns: ema(df, span=120, prefix="ema")
    diff = df["ema_60"] - df["ema_120"]
    mu = diff.rolling(390).mean()
    sd = diff.rolling(390).std().replace(0, np.nan)
    df.loc[:, prefix] = (diff - mu) / sd

def ema_slope_over(df, span1=60, span2=120, prefix="ema_slope_60_120"):
    # ATR-normalized slope over k=span1 bars: (ema_now - ema_{t-k})/(k*ATR)
    if f"ema_{span1}" not in df.columns:
        ema(df, span=span1, prefix="ema")
    k = int(span1)
    ema_now = df[f"ema_{span1}"]
    ema_prev = ema_now.shift(k)
    atr_col = "atr_90" if "atr_90" in df.columns else ("atr_60" if "atr_60" in df.columns else "atr_14")
    if atr_col not in df.columns:
        atr(df, window=14, prefix="atr")
        atr_col = "atr_14"
    slope = (ema_now - ema_prev) / (k * df[atr_col].replace(0, np.nan))
    df.loc[:, prefix] = slope
