# classes/indicators/adx.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

# ---- Wilder helpers ----------------------------------------------------------
def _wilder_rma(s: pd.Series, window: int) -> pd.Series:
    """Wilder's RMA (SMMA) via ewm."""
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
    dm_plus = np.where((up > dn) & (up > 0), up, 0.0)
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
    """Compute DI+/DI-/ADX on a DataFrame slice."""
    if not {"high", "low", "close"} <= set(df.columns):
        raise ValueError("ADX requires 'high','low','close' columns")
    high, low, close = df["high"], df["low"], df["close"]
    tr = _true_range(high, low, close)
    dm_plus_raw, dm_minus_raw = _dm_plus_minus(high, low)
    atr_w = _wilder_rma(tr, window)
    dm_plus_w = _wilder_rma(dm_plus_raw, window)
    dm_minus_w = _wilder_rma(dm_minus_raw, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        di_plus = 100.0 * (dm_plus_w / atr_w)
        di_minus = 100.0 * (dm_minus_w / atr_w)
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
    prefix: Optional[str] = None,
) -> None:
    """Full recompute — for backtesting only."""
    out = _compute_adx_block(df, window)
    for c in out.columns:
        df[c] = out[c]

def adx_last_row(
    df: pd.DataFrame,
    window: int = 14,
    prefix: Optional[str] = None,
    lookback_factor: int = 5,
) -> None:
    """Fast: only update last row — NO LOOK-AHEAD"""
    n = len(df)
    if n < 2:
        return
    idx = df.index[-1]
    lb = max(window * int(lookback_factor), window + 2)
    start = max(0, n - lb)
    tail = df.iloc[start:].copy()
    out = _compute_adx_block(tail, window)
    last_idx_in_tail = tail.index[-1]
    for c in out.columns:
        df.at[idx, c] = out.at[last_idx_in_tail, c]  # ← only last row

def adx_at_index(
    df: pd.DataFrame,
    idx: int,
    window: int = 14,
    prefix: Optional[str] = None,
    lookback_factor: int = 5,
) -> None:
    """Update only the target index — NO LOOK-AHEAD"""
    if idx is None or idx < 0 or idx >= len(df):
        return
    lb = max(window * int(lookback_factor), window + 2)
    start = max(0, idx + 1 - lb)
    block = df.iloc[start:idx + 1].copy()
    out = _compute_adx_block(block, window)
    target_idx_in_block = block.index[-1]
    for c in out.columns:
        df.at[df.index[idx], c] = out.at[target_idx_in_block, c]  # ← only target row
