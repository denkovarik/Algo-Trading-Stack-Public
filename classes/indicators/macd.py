# classes/indicators/macd.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _ema_online(prev: float, x: float, span: int) -> float:
    alpha = 2.0 / (int(span) + 1.0)
    return prev + alpha * (x - prev)

# ---------- Full-history compute ----------
def macd_full(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, prefix: str = "macd") -> None:
    """
    Full MACD:
      line   = EMA_fast(close) - EMA_slow(close)
      signal = EMA_signal(line)
      hist   = line - signal
    Also writes cached EMAs to _tmp_ema_fast_* and _tmp_ema_slow_* so last-row updates are O(1).
    """
    if df is None or df.empty:
        return
    fast = int(fast); slow = int(slow); signal = int(signal)
    price = df["close"]

    ef = price.ewm(span=fast,  adjust=False).mean()
    es = price.ewm(span=slow,  adjust=False).mean()
    line = ef - es
    sig  = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig

    line_col   = f"{prefix}_line"
    sig_col    = f"{prefix}_signal"
    hist_col   = f"{prefix}_hist"
    fast_col   = f"_tmp_ema_fast_{fast}"
    slow_col   = f"_tmp_ema_slow_{slow}"

    df.loc[:, line_col] = line
    df.loc[:, sig_col]  = sig
    df.loc[:, hist_col] = hist
    # cache EMAs for O(1) online updates later
    df.loc[:, fast_col] = ef
    df.loc[:, slow_col] = es

# ---------- Fast last-row update ----------
def macd_last_row(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, prefix: str = "macd") -> None:
    if df is None or df.empty:
        return
    n = len(df)
    if n < 2:
        macd_full(df, fast=fast, slow=slow, signal=signal, prefix=prefix)
        return

    fast = int(fast); slow = int(slow); signal = int(signal)
    idx = df.index[-1]
    price_now = float(df["close"].iloc[-1])

    line_col = f"{prefix}_line"
    sig_col  = f"{prefix}_signal"
    hist_col = f"{prefix}_hist"
    fast_col = f"_tmp_ema_fast_{fast}"
    slow_col = f"_tmp_ema_slow_{slow}"

    have_prev_fast = (fast_col in df.columns) and pd.notna(df.iloc[-2].get(fast_col, np.nan))
    have_prev_slow = (slow_col in df.columns) and pd.notna(df.iloc[-2].get(slow_col, np.nan))
    have_prev_sig  = (sig_col  in df.columns) and pd.notna(df.iloc[-2].get(sig_col,  np.nan))

    if have_prev_fast and have_prev_slow and have_prev_sig:
        # Fast O(1) path
        ef_prev = float(df.iloc[-2][fast_col])
        es_prev = float(df.iloc[-2][slow_col])
        ef = _ema_online(ef_prev, price_now, fast)
        es = _ema_online(es_prev, price_now, slow)
        line_now = float(ef - es)
        sig_prev = float(df.iloc[-2][sig_col])
        sig_now  = _ema_online(sig_prev, line_now, signal)

    else:
        # Exact-parity seed: compute previous EMAs & signal from full prefix (up to n-1),
        # then online-update one step to the last bar.
        close_prefix = df["close"].iloc[: n - 1]
        ef_series = close_prefix.ewm(span=fast, adjust=False).mean()
        es_series = close_prefix.ewm(span=slow, adjust=False).mean()
        ef_prev = float(ef_series.iloc[-1])
        es_prev = float(es_series.iloc[-1])
        ef = _ema_online(ef_prev, price_now, fast)
        es = _ema_online(es_prev, price_now, slow)
        line_now = float(ef - es)

        # Signal: previous signal from full prefix of line, then online-update
        line_prefix = ef_series - es_series
        sig_prev = float(line_prefix.ewm(span=signal, adjust=False).mean().iloc[-1])
        sig_now  = _ema_online(sig_prev, line_now, signal)

    hist_now = float(line_now - sig_now)

    # Write outputs + caches for next step
    df.at[idx, line_col] = line_now
    df.at[idx, sig_col]  = sig_now
    df.at[idx, hist_col] = hist_now
    df.at[idx, fast_col] = ef
    df.at[idx, slow_col] = es

# ---------- Per-index recompute (no look-ahead) ----------
def macd_at_index(df: pd.DataFrame, idx: int, fast: int = 12, slow: int = 26, signal: int = 9, prefix: str = "macd") -> None:
    """
    Recompute MACD at a specific index using only bars up to idx.
    (No temp caches necessary here.)
    """
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df):
        return
    fast = int(fast); slow = int(slow); signal = int(signal)
    start = 0  # use the full prefix so EMAs match pandas seeding exactly
    sub_price = df.iloc[start: idx + 1]["close"]
    ef = sub_price.ewm(span=fast, adjust=False).mean().iloc[-1]
    es = sub_price.ewm(span=slow, adjust=False).mean().iloc[-1]
    line_now = float(ef - es)
    # signal from line prefix up to idx
    line_series = sub_price.ewm(span=fast, adjust=False).mean() - sub_price.ewm(span=slow, adjust=False).mean()
    sig_now  = float(line_series.ewm(span=signal, adjust=False).mean().iloc[-1])
    hist_now = float(line_now - sig_now)

    line_col   = f"{prefix}_line"
    sig_col    = f"{prefix}_signal"
    hist_col   = f"{prefix}_hist"

    df.at[idx, line_col] = line_now
    df.at[idx, sig_col]  = sig_now
    df.at[idx, hist_col] = hist_now

