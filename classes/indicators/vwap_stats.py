# classes/indicators/vwap_stats.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd


def _compute_vwap_stats_block(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute:
      vwap_revert_rate_<w> : fraction of sign flips of (close - vwap) in the last w bars
      vwap_dist_avg_<w>    : mean absolute distance |close - vwap| over the last w bars
    """
    if not {"close", "vwap"}.issubset(df.columns):
        raise ValueError("vwap_stats requires 'close' and 'vwap' columns")

    close = df["close"].astype(float)
    vwap = df["vwap"].astype(float)

    dev = close - vwap
    sign = np.sign(dev)
    flips = (sign * sign.shift(1) < 0).astype(float)

    revert_rate = flips.rolling(window=window, min_periods=window).mean()
    dist_avg = dev.abs().rolling(window=window, min_periods=window).mean()

    out = pd.DataFrame({
        f"vwap_revert_rate_{int(window)}": revert_rate,
        f"vwap_dist_avg_{int(window)}": dist_avg,
    }, index=df.index)
    return out


def vwap_stats_full(df: pd.DataFrame, window: int = 60, prefix: Optional[str] = None) -> None:
    out = _compute_vwap_stats_block(df, window)
    for c in out.columns:
        df[c] = out[c]


def vwap_stats_last_row(
    df: pd.DataFrame,
    window: int = 60,
    prefix: Optional[str] = None,
    lookback_factor: int = 3,
) -> None:
    n = len(df)
    if n == 0:
        return
    lb = max(window * int(lookback_factor), window + 2)
    start = max(0, n - lb)
    tail = df.iloc[start:].copy()
    out = _compute_vwap_stats_block(tail, window)
    for c in out.columns:
        df.loc[out.index, c] = out[c]


def vwap_stats_at_index(
    df: pd.DataFrame,
    idx: int,
    window: int = 60,
    prefix: Optional[str] = None,
    lookback_factor: int = 3,
) -> None:
    if idx is None or idx < 0 or idx >= len(df):
        return
    lb = max(window * int(lookback_factor), window + 2)
    start = max(0, idx + 1 - lb)
    block = df.iloc[start:idx + 1].copy()
    out = _compute_vwap_stats_block(block, window)
    for c in out.columns:
        df.loc[out.index, c] = out[c]

