# classes/indicators/volume_stats.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd


def _compute_vol_stats_block(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute rolling volume z-score (vol_z_<w>) and relative volume (rvol_<w>):
      vol_z = (vol - mean(vol, w)) / std(vol, w)
      rvol  = vol / mean(vol, w)
    Uses a population std (ddof=0). Returns NaN until w bars are available.
    """
    if "volume" not in df.columns:
        raise ValueError("volume_stats requires 'volume' column")

    vol = df["volume"].astype(float)
    ma  = vol.rolling(window=window, min_periods=window).mean()
    sd  = vol.rolling(window=window, min_periods=window).std(ddof=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (vol - ma) / sd
        r = vol / ma

    out = pd.DataFrame({
        f"vol_z_{int(window)}": z.astype(float),
        f"rvol_{int(window)}":  r.astype(float),
    }, index=df.index)
    return out


def _consolidate(df: pd.DataFrame) -> None:
    """
    Best-effort consolidation to reduce pandas block fragmentation warnings.
    (Private API on older pandas; no-op on newer managers.)
    """
    try:
        df._consolidate_inplace()  # type: ignore[attr-defined]
    except Exception:
        pass


def _bulk_write(df: pd.DataFrame, out: pd.DataFrame) -> None:
    """
    Write multiple columns at once into a consolidated frame.
    """
    # PRE: consolidate before adding columns so we don't insert into a fragmented frame
    _consolidate(df)

    # Bulk assign both columns at once (fast path; avoids per-column insert)
    df[out.columns] = out

    # POST: keep the frame consolidated for downstream indicator modules
    _consolidate(df)


def vol_stats_full(
    df: pd.DataFrame,
    window: int = 60,
    prefix: Optional[str] = None,   # kept for API symmetry (unused)
) -> None:
    out = _compute_vol_stats_block(df, window)
    _bulk_write(df, out)


def vol_stats_last_row(
    df: pd.DataFrame,
    window: int = 60,
    prefix: Optional[str] = None,
    lookback_factor: int = 3,
) -> None:
    n = len(df)
    if n == 0:
        return
    lb = max(window * int(lookback_factor), window)
    start = max(0, n - lb)
    tail = df.iloc[start:].copy()
    out = _compute_vol_stats_block(tail, window)
    _bulk_write(df, out)


def vol_stats_at_index(
    df: pd.DataFrame,
    idx: int,
    window: int = 60,
    prefix: Optional[str] = None,
    lookback_factor: int = 3,
) -> None:
    if idx is None or idx < 0 or idx >= len(df):
        return
    lb = max(window * int(lookback_factor), window)
    start = max(0, idx + 1 - lb)
    block = df.iloc[start:idx + 1].copy()
    out = _compute_vol_stats_block(block, window)
    _bulk_write(df, out)

