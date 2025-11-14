# classes/indicators/breakout_age.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd


def _rolling_max_min(series: pd.Series, window: int, is_max: bool) -> pd.Series:
    if is_max:
        return series.rolling(window=window, min_periods=window).max()
    return series.rolling(window=window, min_periods=window).min()


def _compute_breakout_age_block(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute breakout_age_<window>:
      - donch_high = rolling max(high, window)
      - donch_low  = rolling min(low, window)
      - breakout at t if close[t] > donch_high[t-1] or close[t] < donch_low[t-1]
      - age[t] = 0 if breakout else age[t-1] + 1 (NaN until enough history)
    """
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"breakout_age requires {required}, missing: {missing}")

    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    # Donchian components
    donch_high = _rolling_max_min(h, window, True)
    donch_low  = _rolling_max_min(l, window, False)

    # Use previous-bar donchian for breakout test (avoid lookahead)
    prev_high = donch_high.shift(1)
    prev_low  = donch_low.shift(1)

    breakout = (c > prev_high) | (c < prev_low)

    # Age counter that resets on breakout
    age = pd.Series(index=df.index, dtype=float)
    running = np.nan
    for i, brk in enumerate(breakout.values):
        # No signal until we have valid previous donchian bounds
        if not np.isfinite(prev_high.iloc[i]) or not np.isfinite(prev_low.iloc[i]):
            age.iloc[i] = np.nan
            continue
        if brk:
            running = 0.0
        else:
            running = (running + 1.0) if np.isfinite(running) else 1.0
        age.iloc[i] = running

    out = pd.DataFrame({
        f"breakout_age_{int(window)}": age
    }, index=df.index)
    return out


def breakout_age_full(
    df: pd.DataFrame,
    window: int = 20,
    prefix: Optional[str] = None,  # API symmetry
) -> None:
    out = _compute_breakout_age_block(df, window)
    for col in out.columns:
        df[col] = out[col]


def breakout_age_last_row(
    df: pd.DataFrame,
    window: int = 20,
    prefix: Optional[str] = None,
    lookback_factor: int = 5,
) -> None:
    """
    Fast tail update: recompute on a tail slice (to keep continuity of the counter)
    and write it back.
    """
    n = len(df)
    if n == 0:
        return
    lb = max(window * int(lookback_factor), window + 5)
    start = max(0, n - lb)
    tail = df.iloc[start:].copy()
    out = _compute_breakout_age_block(tail, window)
    for col in out.columns:
        df.loc[out.index, col] = out[col]


def breakout_age_at_index(
    df: pd.DataFrame,
    idx: int,
    window: int = 20,
    prefix: Optional[str] = None,
    lookback_factor: int = 5,
) -> None:
    """
    Recompute ending at a specific index (inclusive) using a tail slice, then
    write overlapping values back.
    """
    if idx is None or idx < 0 or idx >= len(df):
        return
    lb = max(window * int(lookback_factor), window + 5)
    start = max(0, idx + 1 - lb)
    block = df.iloc[start:idx + 1].copy()
    out = _compute_breakout_age_block(block, window)
    for col in out.columns:
        df.loc[out.index, col] = out[col]

