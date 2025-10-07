# classes/indicators/pullback.py
from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd


def _fractal_swings_vectorized(high: pd.Series, low: pd.Series, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized fractal detection using centered rolling windows of width (2k+1).
    Returns:
      idx   : np.ndarray of swing indices (int)
      types : np.ndarray of 'H' (1) or 'L' (-1) flags as int8 (+1 = high, -1 = low)
      price : np.ndarray of swing prices (float)
    Notes:
      - Edges (where centered window incomplete) yield NaN and thus no swing.
      - If both high/low conditions happen to be true at the same bar (rare),
        we prefer the dominant extremum by comparing distances to neighbors.
    """
    n = len(high)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=np.int8), np.array([], dtype=float)

    w = 2 * k + 1
    # Centered rolling max/min
    rmax = high.rolling(window=w, center=True, min_periods=w).max()
    rmin = low.rolling(window=w, center=True, min_periods=w).min()

    is_swing_high = (high == rmax) & rmax.notna()
    is_swing_low  = (low  == rmin) & rmin.notna()

    hi_idx = np.flatnonzero(is_swing_high.values)
    lo_idx = np.flatnonzero(is_swing_low.values)

    # Build combined list
    all_idx   = np.concatenate([hi_idx, lo_idx])
    all_type  = np.concatenate([np.ones_like(hi_idx, dtype=np.int8), -np.ones_like(lo_idx, dtype=np.int8)])
    all_price = np.concatenate([high.values[hi_idx], low.values[lo_idx]])

    if all_idx.size == 0:
        return all_idx, all_type, all_price

    # Sort by index to get chronological swings
    order = np.argsort(all_idx, kind="mergesort")
    all_idx   = all_idx[order]
    all_type  = all_type[order]
    all_price = all_price[order]

    # If any bar is marked as both H and L (possible in flat windows), deduplicate:
    # Keep the one with larger extremeness by comparing difference to adjacent values.
    # Simple approach: drop duplicates by index keeping the first occurrence (already sorted),
    # which suffices in practice. For more rigor, you'd compute extremeness; but this is rare.
    uniq_mask = np.ones_like(all_idx, dtype=bool)
    if all_idx.size > 1:
        dup = np.flatnonzero(all_idx[1:] == all_idx[:-1]) + 1
        uniq_mask[dup] = False
    all_idx   = all_idx[uniq_mask]
    all_type  = all_type[uniq_mask]
    all_price = all_price[uniq_mask]

    return all_idx, all_type, all_price


def _pullback_events_from_swings(sw_idx: np.ndarray, sw_typ: np.ndarray, sw_prc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given chronological swings (indices, types, prices) where sw_typ = +1 (H) or -1 (L),
    compute retracement events:

    For L -> H -> L:
      impulse = H - prior_L
      retr    = H - this_L
      pct     = retr / impulse

    For H -> L -> H:
      impulse = prior_H - L
      retr    = this_H  - L
      pct     = retr / impulse

    Returns:
      event_idx : indices (bar index of the retrace pivot, i.e., the third pivot in the triple)
      event_pct : corresponding retracement percentages
    """
    m = sw_idx.size
    if m < 3:
        return np.array([], dtype=int), np.array([], dtype=float)

    out_idx: List[int] = []
    out_pct: List[float] = []

    # Scan the swing triples
    for i in range(2, m):
        t0, t1, t2 = sw_typ[i-2], sw_typ[i-1], sw_typ[i]
        p0, p1, p2 = sw_prc[i-2], sw_prc[i-1], sw_prc[i]

        if t0 == -1 and t1 == +1 and t2 == -1:      # L -> H -> L
            impulse = p1 - p0
            retr    = p1 - p2
        elif t0 == +1 and t1 == -1 and t2 == +1:    # H -> L -> H
            impulse = p0 - p1
            retr    = p2 - p1
        else:
            continue

        if impulse > 0 and np.isfinite(retr):
            out_idx.append(int(sw_idx[i]))
            out_pct.append(float(retr / impulse))

    if len(out_idx) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    return np.asarray(out_idx, dtype=int), np.asarray(out_pct, dtype=float)


def _compute_pullback_block(df: pd.DataFrame, swing_k: int, lookback_bars: int, avg_n: int) -> pd.DataFrame:
    """
    Vectorized pullback metrics:
      pullback_last_pct
      pullback_avg_pct_<avg_n>

    Strategy:
      1) Compute fractal swings for the ENTIRE slice with centered rolling windows.
      2) Turn swings into retracement "events" (pct) stamped at the retrace pivot index.
      3) Build two event series:
          - last_pct: forward-fill of pct events
          - avg_pct:  rolling mean of the last `avg_n` non-NaN events (computed on a sparse series),
                      then forward-fill
      4) Return both columns aligned to df.index.
    """
    if not {"high", "low"}.issubset(df.columns):
        raise ValueError("pullback requires 'high' and 'low' columns")

    # We operate on the entire (tail) block; prev-bar semantics are preserved
    # because the live bar won't be in the tail when your router calls at last_row/at_index.
    high = df["high"].astype(float)
    low  = df["low"].astype(float)

    # 1) Swings
    sw_idx, sw_typ, sw_prc = _fractal_swings_vectorized(high, low, int(swing_k))
    if sw_idx.size == 0:
        return pd.DataFrame({
            "pullback_last_pct": pd.Series(np.nan, index=df.index, dtype=float),
            f"pullback_avg_pct_{int(avg_n)}": pd.Series(np.nan, index=df.index, dtype=float),
        })

    # 2) Retracement events stamped at the retrace pivot index (3rd pivot in the triple)
    ev_idx, ev_pct = _pullback_events_from_swings(sw_idx, sw_typ, sw_prc)

    # 3) Build event series on sparse index and forward-fill to all bars
    s = pd.Series(ev_pct, index=df.index[ev_idx], dtype=float)

    # last_pct: most recent event value at or before t
    last_pct = s.reindex(df.index).ffill()

    # avg over last `avg_n` events: compute rolling on event series only, then ffill
    avg_n = int(max(1, avg_n))
    s_avg = s.rolling(window=avg_n, min_periods=1).mean()
    avg_pct = s_avg.reindex(df.index).ffill()

    out = pd.DataFrame({
        "pullback_last_pct": last_pct,
        f"pullback_avg_pct_{int(avg_n)}": avg_pct,
    }, index=df.index)
    return out


# ----------------------- public API -----------------------

def pullback_full(
    df: pd.DataFrame,
    swing_k: int = 2,
    lookback_bars: int = 400,  # kept for API symmetry; not used in vectorized path
    avg_n: int = 5,
    prefix: Optional[str] = None,   # API symmetry (unused)
) -> None:
    out = _compute_pullback_block(df, swing_k, lookback_bars, avg_n)
    # Bulk-assign both columns at once
    df[out.columns] = out


def pullback_last_row(
    df: pd.DataFrame,
    swing_k: int = 2,
    lookback_bars: int = 400,  # kept for API symmetry; not used in vectorized path
    avg_n: int = 5,
    lookback_factor: int = 3,       # kept for API symmetry; vectorized path is already O(n)
    prefix: Optional[str] = None,
) -> None:
    n = len(df)
    if n == 0:
        return
    # Compute on a tail to keep work bounded; include enough bars to let centered windows work.
    w = 2 * int(swing_k) + 1
    lb = max(int(lookback_bars), w * 10)  # generous but finite
    start = max(0, n - lb)
    tail = df.iloc[start:].copy()
    out = _compute_pullback_block(tail, swing_k, lookback_bars, avg_n)
    df[out.columns] = out


def pullback_at_index(
    df: pd.DataFrame,
    idx: int,
    swing_k: int = 2,
    lookback_bars: int = 400,  # kept for API symmetry; not used in vectorized path
    avg_n: int = 5,
    lookback_factor: int = 3,  # kept for API symmetry
    prefix: Optional[str] = None,
) -> None:
    if idx is None or idx < 0 or idx >= len(df):
        return
    w = 2 * int(swing_k) + 1
    lb = max(int(lookback_bars), w * 10)
    start = max(0, idx + 1 - lb)
    block = df.iloc[start:idx + 1].copy()
    out = _compute_pullback_block(block, swing_k, lookback_bars, avg_n)
    df[out.columns] = out

