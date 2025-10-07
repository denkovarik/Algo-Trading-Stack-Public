# classes/indicators/keltner.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

# ---- Wilder/EMA helpers ------------------------------------------------------

def _rma(series: pd.Series, window: int) -> pd.Series:
    """Wilder's RMA (SMMA) via ewm(alpha=1/window, adjust=False)."""
    if window <= 0:
        raise ValueError("window must be > 0")
    return series.ewm(alpha=1.0 / float(window), adjust=False).mean()

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def _ema(series: pd.Series, window: int) -> pd.Series:
    """EMA with span=window (standard EMA, not Wilder)."""
    return series.ewm(span=float(window), adjust=False).mean()

# ---- Column names ------------------------------------------------------------

def _col_mid(window: int) -> str:     return f"kelt_mid_{int(window)}"
def _col_up(window: int) -> str:      return f"kelt_up_{int(window)}"
def _col_dn(window: int) -> str:      return f"kelt_dn_{int(window)}"
def _col_squeeze(window: int) -> str: return f"squeeze_{int(window)}"

# ---- Core computation (vectorized) ------------------------------------------

def _compute_keltner_block(
    df: pd.DataFrame,
    window: int,
    k_atr: float,
    bb_std: float,
    use_typical_price: bool = True,
) -> pd.DataFrame:
    """
    Vectorized computation of Keltner mid/up/dn and squeeze flag on df (all rows).
    Keltner:
      mid = EMA(typical_price, window)   where typical = (H+L+C)/3 if use_typical_price else close
      up  = mid + k_atr * ATR(window)    with ATR via Wilder RMA of True Range
      dn  = mid - k_atr * ATR(window)
    Squeeze flag:
      1 if Bollinger( window, bb_std ) bands are fully inside Keltner channel, else 0.
    """
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Keltner requires columns {required}, missing: {missing}")

    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    # ATR (Wilder)
    tr = _true_range(h, l, c)
    atr = _rma(tr, window)

    # Midline: EMA of typical price (or close)
    if use_typical_price:
        typical = (h + l + c) / 3.0
        mid = _ema(typical, window)
    else:
        mid = _ema(c, window)

    up = mid + k_atr * atr
    dn = mid - k_atr * atr

    # Bollinger Bands on close (SMA/STD) for squeeze comparison
    ma = c.rolling(window=window, min_periods=window).mean()
    sd = c.rolling(window=window, min_periods=window).std(ddof=0)
    bb_up = ma + bb_std * sd
    bb_dn = ma - bb_std * sd

    # Squeeze if BB completely inside Keltner: bb_up <= up and bb_dn >= dn
    squeeze = ((bb_up <= up) & (bb_dn >= dn)).astype("int64")

    out = pd.DataFrame({
        _col_mid(window): mid,
        _col_up(window):  up,
        _col_dn(window):  dn,
        _col_squeeze(window): squeeze,
    }, index=df.index).astype({ _col_squeeze(window): "int64" })
    return out

# ---- Public API: full / last_row / at_index ---------------------------------

def keltner_full(
    df: pd.DataFrame,
    window: int = 20,
    k_atr: float = 1.5,
    bb_std: float = 2.0,
    use_typical_price: bool = True,
    prefix: Optional[str] = None,   # kept for API symmetry (unused)
) -> None:
    """
    Compute and attach Keltner & Squeeze columns across the entire DataFrame:
      kelt_mid_<w>, kelt_up_<w>, kelt_dn_<w>, squeeze_<w>.
    """
    out = _compute_keltner_block(df, window, k_atr, bb_std, use_typical_price)
    df = pd.concat([df, out], axis=1)

def keltner_last_row(
    df: pd.DataFrame,
    window: int = 20,
    k_atr: float = 1.5,
    bb_std: float = 2.0,
    use_typical_price: bool = True,
    lookback_factor: int = 5,
    prefix: Optional[str] = None,
) -> None:
    """
    Fast tail update for newest bar(s): recompute on a tail slice and write back.
    """
    n = len(df)
    if n == 0:
        return
    lb = max(window * int(lookback_factor), window + 2)
    start = max(0, n - lb)
    tail = df.iloc[start:].copy()
    out = _compute_keltner_block(tail, window, k_atr, bb_std, use_typical_price)
    df = pd.concat([df, out], axis=1)

def keltner_at_index(
    df: pd.DataFrame,
    idx: int,
    window: int = 20,
    k_atr: float = 1.5,
    bb_std: float = 2.0,
    use_typical_price: bool = True,
    lookback_factor: int = 5,
    prefix: Optional[str] = None,
) -> None:
    """
    Recompute Keltner & Squeeze ending at a specific index (inclusive) using a tail slice,
    then write overlapping values back into df.
    """
    if idx is None or idx < 0 or idx >= len(df):
        return
    lb = max(window * int(lookback_factor), window + 2)
    start = max(0, idx + 1 - lb)
    block = df.iloc[start:idx + 1].copy()
    out = _compute_keltner_block(block, window, k_atr, bb_std, use_typical_price)
    df = pd.concat([df, out], axis=1)


