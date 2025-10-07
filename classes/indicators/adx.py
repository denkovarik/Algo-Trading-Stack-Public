# classes/indicators/adx.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

# ---- Wilder helpers ----------------------------------------------------------

def _wilder_rma(s: pd.Series, window: int) -> pd.Series:
    """
    Wilder's RMA (a.k.a. SMMA) via ewm with alpha=1/window (adjust=False).
    This matches the standard ADX smoothing for TR and DM.
    """
    if window <= 0:
        raise ValueError("window must be > 0")
    return s.ewm(alpha=1.0 / float(window), adjust=False, ignore_na=False).mean()

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def _dm_plus_minus(high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
    up = high.diff()
    dn = (-low.diff())
    # Wilder's definition: choose only the dominant movement; negatives are zeroed
    dm_plus  = np.where((up > dn) & (up > 0), up, 0.0)
    dm_minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    return pd.Series(dm_plus, index=high.index), pd.Series(dm_minus, index=high.index)

# ---- Column naming -----------------------------------------------------------

def _col_adx(window: int) -> str:
    return f"adx_{int(window)}"

def _col_di_plus(window: int) -> str:
    return f"di_plus_{int(window)}"

def _col_di_minus(window: int) -> str:
    return f"di_minus_{int(window)}"

# ---- Core computation (vectorized) ------------------------------------------

def _compute_adx_block(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute DI+/DI- and ADX on the full DataFrame (vectorized).
    Returns a DataFrame with columns: di_plus_<w>, di_minus_<w>, adx_<w>.
    """
    if not {"high", "low", "close"} <= set(df.columns):
        raise ValueError("ADX requires 'high','low','close' columns")

    high, low, close = df["high"], df["low"], df["close"]

    tr = _true_range(high, low, close)
    dm_plus_raw, dm_minus_raw = _dm_plus_minus(high, low)

    # Wilder smooth TR and DM
    atr_w = _wilder_rma(tr, window)
    dm_plus_w = _wilder_rma(dm_plus_raw, window)
    dm_minus_w = _wilder_rma(dm_minus_raw, window)

    # DI+
    with np.errstate(divide="ignore", invalid="ignore"):
        di_plus = 100.0 * (dm_plus_w / atr_w)
        di_minus = 100.0 * (dm_minus_w / atr_w)

    # DX and ADX
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus).abs()

    adx = _wilder_rma(dx, window)

    out = pd.DataFrame({
        _col_di_plus(window): di_plus.astype(float),
        _col_di_minus(window): di_minus.astype(float),
        _col_adx(window): adx.astype(float),
    }, index=df.index)
    return out

# ---- Public API: full / last_row / at_index ---------------------------------

def adx_full(
    df: pd.DataFrame,
    window: int = 14,
    prefix: Optional[str] = None,   # kept for consistency with other indicators (unused)
) -> None:
    """
    Compute DI+/DI-/ADX over the entire DataFrame and attach columns:
      di_plus_<window>, di_minus_<window>, adx_<window>
    """
    out = _compute_adx_block(df, window)
    for c in out.columns:
        df[c] = out[c]

def adx_last_row(
    df: pd.DataFrame,
    window: int = 14,
    prefix: Optional[str] = None,
    lookback_factor: int = 5,
) -> None:
    """
    Fast update for the most recent bar.
    Recomputes on a tail slice (lookback_factor * window) and writes results back.
    This avoids requiring prior smoothed seeds to be carried manually.
    """
    n = len(df)
    if n < 2:
        return
    lb = max(window * int(lookback_factor), window + 2)
    start = max(0, n - lb)
    tail = df.iloc[start:].copy()
    out = _compute_adx_block(tail, window)
    # write back into df for just the overlapping index
    for c in out.columns:
        df.loc[out.index, c] = out[c]

def adx_at_index(
    df: pd.DataFrame,
    idx: int,
    window: int = 14,
    prefix: Optional[str] = None,
    lookback_factor: int = 5,
) -> None:
    """
    Recompute ADX for a specific index position (inclusive).
    We recompute on a tail ending at idx to maintain smoothing continuity.
    """
    if idx is None or idx < 0 or idx >= len(df):
        return
    lb = max(window * int(lookback_factor), window + 2)
    start = max(0, idx + 1 - lb)
    # slice up to and including idx
    block = df.iloc[start:idx + 1].copy()
    out = _compute_adx_block(block, window)
    for c in out.columns:
        df.loc[out.index, c] = out[c]

