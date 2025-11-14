# classes/indicators/ema.py
from __future__ import annotations
import numpy as np
import pandas as pd


# =========================
# Core EMA (existing)
# =========================
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
    sub = df.iloc[: idx + 1]["close"]
    df.at[idx, f"{prefix}_{span}"] = sub.ewm(span=span, adjust=False).mean().iloc[-1]


# =========================
# EMA curvature (NEW)
# =========================
def _ema(series: pd.Series, window: int) -> pd.Series:
    """Standard EMA with span=window."""
    return series.ewm(span=float(window), adjust=False).mean()


def _compute_ema_curve_block(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute second derivative of EMA(window):
      ema        = EMA(close, window)
      slope      = ema.diff(1)
      ema_curve  = slope.diff(1)
    Returns a DataFrame with column: f'ema_curve_{window}'.
    """
    if "close" not in df.columns:
        raise ValueError("ema_curve requires 'close' column")

    c = df["close"].astype(float)
    ema = _ema(c, window)
    slope = ema.diff(1)
    curve = slope.diff(1)

    out = pd.DataFrame({
        f"ema_curve_{int(window)}": curve.astype(float)
    }, index=df.index)
    return out


def ema_curve_full(
    df: pd.DataFrame,
    window: int = 60,
    prefix: str = "ema_curve",   # kept for API symmetry (column name is fixed as ema_curve_<window>)
) -> None:
    """
    Compute and attach EMA curvature (2nd derivative) across the entire DataFrame:
      ema_curve_<window>
    """
    out = _compute_ema_curve_block(df, window)
    for col in out.columns:
        df[col] = out[col]


def ema_curve_last_row(
    df: pd.DataFrame,
    window: int = 60,
    prefix: str = "ema_curve",
    lookback_factor: int = 3,
) -> None:
    """
    Fast tail update. Recompute EMA curvature on a tail slice and write it back.
    """
    if df is None or df.empty:
        return
    n = len(df)
    lb = max(window * int(lookback_factor), window + 3)
    start = max(0, n - lb)
    tail = df.iloc[start:].copy()
    out = _compute_ema_curve_block(tail, window)
    for col in out.columns:
        df.loc[out.index, col] = out[col]


def ema_curve_at_index(
    df: pd.DataFrame,
    idx: int,
    window: int = 60,
    prefix: str = "ema_curve",
    lookback_factor: int = 3,
) -> None:
    """
    Recompute EMA curvature ending at a specific index (inclusive) using a tail slice,
    then write overlapping values back.
    """
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df):
        return
    lb = max(window * int(lookback_factor), window + 3)
    start = max(0, idx + 1 - lb)
    block = df.iloc[start:idx + 1].copy()
    out = _compute_ema_curve_block(block, window)
    for col in out.columns:
        df.loc[out.index, col] = out[col]


# =========================
# OPTIONAL extras used by trainer (kept, with minor fixes)
# =========================
def zscore_close_vs_ema(df, ema_span=60, prefix="z_close_ema60"):
    col = f"ema_{ema_span}"
    if col not in df.columns:
        ema_full(df, span=ema_span, prefix="ema")
    dev = (df["close"] - df[col])
    mu = dev.rolling(390).mean()
    sd = dev.rolling(390).std().replace(0, np.nan)
    df.loc[:, prefix] = (dev - mu) / sd


def zscore_ema60_vs_ema120(df, prefix="z_ema60_ema120"):
    if "ema_60" not in df.columns:
        ema_full(df, span=60, prefix="ema")
    if "ema_120" not in df.columns:
        ema_full(df, span=120, prefix="ema")
    diff = df["ema_60"] - df["ema_120"]
    mu = diff.rolling(390).mean()
    sd = diff.rolling(390).std().replace(0, np.nan)
    df.loc[:, prefix] = (diff - mu) / sd


def ema_slope_over(df, span1=60, span2=120, prefix="ema_slope_60_120"):
    """
    ATR-normalized slope over k=span1 bars:
      slope = (ema_now - ema_{t-k}) / (k * ATR)
    """
    if f"ema_{span1}" not in df.columns:
        ema_full(df, span=span1, prefix="ema")
    k = int(span1)
    ema_now = df[f"ema_{span1}"]
    ema_prev = ema_now.shift(k)

    # Lazy import to avoid circulars at module import time
    try:
        from classes.indicators.atr import atr as _atr_full  # full-history compute
    except Exception:
        _atr_full = None

    # Select an existing ATR column or compute a fallback
    if "atr_90" in df.columns:
        atr_col = "atr_90"
    elif "atr_60" in df.columns:
        atr_col = "atr_60"
    elif "atr_14" in df.columns:
        atr_col = "atr_14"
    else:
        if _atr_full is not None:
            _atr_full(df, window=14, prefix="atr")
            atr_col = "atr_14"
        else:
            # Last resort: avoid crash, normalize by price change scale
            atr_col = None

    denom = (k * (df[atr_col].replace(0, np.nan))) if atr_col is not None else (k * 1.0)
    slope = (ema_now - ema_prev) / denom
    df.loc[:, prefix] = slope

