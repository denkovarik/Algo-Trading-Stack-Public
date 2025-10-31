# classes/indicators/range_tightness.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd


def _compute_range_tightness_block(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute range_tightness_<w> = (donchian_high_w - donchian_low_w) / atr_w
    Requires columns: dc<w>_high, dc<w>_low, atr_<w>
    """
    hi_col = f"dc{int(window)}_high"
    lo_col = f"dc{int(window)}_low"
    atr_col = f"atr_{int(window)}"

    required = {hi_col, lo_col, atr_col}
    if not required.issubset(df.columns):
        raise ValueError(f"range_tightness requires {required}")

    rng = df[hi_col] - df[lo_col]
    tightness = rng / df[atr_col].replace(0, np.nan)

    out = pd.DataFrame({
        f"range_tightness_{int(window)}": tightness.astype(float)
    }, index=df.index)
    return out


def range_tightness_full(df: pd.DataFrame, window: int = 30, prefix: Optional[str] = None) -> None:
    out = _compute_range_tightness_block(df, window)
    for c in out.columns:
        df[c] = out[c]


def range_tightness_last_row(
    df: pd.DataFrame,
    window: int = 30,
    prefix: Optional[str] = None,
    lookback_factor: int = 3,
) -> None:
    n = len(df)
    if n == 0:
        return
    lb = max(window * int(lookback_factor), window + 2)
    start = max(0, n - lb)
    tail = df.iloc[start:].copy()
    out = _compute_range_tightness_block(tail, window)
    for c in out.columns:
        df.loc[out.index, c] = out[c]


def range_tightness_at_index(
    df: pd.DataFrame,
    idx: int,
    window: int = 30,
    prefix: Optional[str] = None,
    lookback_factor: int = 3,
) -> None:
    if idx is None or idx < 0 or idx >= len(df):
        return
    lb = max(window * int(lookback_factor), window + 2)
    start = max(0, idx + 1 - lb)
    block = df.iloc[start:idx + 1].copy()
    out = _compute_range_tightness_block(block, window)
    for c in out.columns:
        df.loc[out.index, c] = out[c]

