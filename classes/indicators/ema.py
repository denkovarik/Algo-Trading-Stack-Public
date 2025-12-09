# classes/indicators/ema.py — FINAL + FULL DEBUG VERSION
from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# =========================
# EMA — O(1) incremental + safe fallback
# =========================
def ema_at_index(df: pd.DataFrame, idx: int, span: int = 21, prefix: str = "ema") -> None:
    """Canonical at-index updater — used by everyone (including curve)"""
    if df is None or df.empty or idx < 0 or idx >= len(df) or "close" not in df.columns or not is_numeric_dtype(df["close"]):
        return
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonically increasing for EMA calculation.")
    span = int(span)
    col = f"{prefix}_{span}"
    if col not in df.columns:
        df[col] = np.nan
    price = df.iat[idx, df.columns.get_loc("close")]
    if idx == 0:
        df.iat[idx, df.columns.get_loc(col)] = price
        return
    prev_val = df.iat[idx-1, df.columns.get_loc(col)]
    if pd.isna(prev_val):
        # Safe fallback — only when cold-starting a new EMA
        start = max(0, idx - span*4)
        series = df["close"].iloc[start:idx+1]
        df.iat[idx, df.columns.get_loc(col)] = series.ewm(span=span, adjust=False).mean().iloc[-1]
    else:
        alpha = 2.0 / (span + 1.0)
        df.iat[idx, df.columns.get_loc(col)] = prev_val + alpha * (price - prev_val)

def ema_last_row(df: pd.DataFrame, span: int = 21, prefix: str = "ema") -> None:
    if df is not None and not df.empty:
        ema_at_index(df, len(df)-1, span=span, prefix=prefix)

def ema_full(df: pd.DataFrame, span: int = 21, prefix: str = "ema") -> None:
    if df is None or df.empty or "close" not in df.columns or not is_numeric_dtype(df["close"]):
        return
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonically increasing for EMA calculation.")
    col = f"{prefix}_{span}"
    df[col] = df["close"].ewm(span=span, adjust=False).mean()

# =========================
# EMA Curve — now uses the exact same O(1) path, no slices
# =========================
def ema_curve_at_index(df: pd.DataFrame, idx: int, window: int = 60, prefix: str = "ema_curve") -> None:
    if df is None or df.empty or idx < 2 or "close" not in df.columns or not is_numeric_dtype(df["close"]):
        return
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonically increasing for EMA curve calculation.")
    w = int(window)
    ema_col = f"ema_{w}"
    curve_col = f"{prefix}_{w}"
    if ema_col not in df.columns:
        df[ema_col] = np.nan
    if df[ema_col].iloc[:idx+1].isna().all():
        for i in range(idx + 1):
            ema_at_index(df, i, span=w, prefix="ema")
    if curve_col not in df.columns:
        df[curve_col] = np.nan
    # LAST ROW ALWAYS GETS A VALUE — NO EXCUSES
    ema_now = df.iat[idx, df.columns.get_loc(ema_col)]
    ema_prev1 = df.iat[idx-1, df.columns.get_loc(ema_col)]
    ema_prev2 = df.iat[idx-2, df.columns.get_loc(ema_col)]
    # Force fallback to previous valid curvature or 0
    if pd.isna(ema_now) or pd.isna(ema_prev1) or pd.isna(ema_prev2):
        # Copy from previous bar if exists, else 0
        if idx >= 3 and not pd.isna(df.iat[idx-3, df.columns.get_loc(curve_col)]):
            df.iat[idx, df.columns.get_loc(curve_col)] = df.iat[idx-3, df.columns.get_loc(curve_col)]
        else:
            df.iat[idx, df.columns.get_loc(curve_col)] = 0.0
    else:
        df.iat[idx, df.columns.get_loc(curve_col)] = (ema_now - ema_prev1) - (ema_prev1 - ema_prev2)

def ema_curve_last_row(df: pd.DataFrame, window: int = 60, prefix: str = "ema_curve") -> None:
    if df is not None and not df.empty:
        ema_curve_at_index(df, len(df)-1, window=window, prefix=prefix)

def ema_curve_full(df: pd.DataFrame, window: int = 60, prefix: str = "ema_curve") -> None:
    if df is not None and not df.empty:
        ema_full(df, span=window, prefix="ema")
        ema_col = f"ema_{window}"
        curve_col = f"{prefix}_{window}"
        if ema_col in df.columns:
            df[curve_col] = df[ema_col].diff().diff()

# =========================
# OPTIONAL extras – LIVE-STABLE + FULL DEBUG OUTPUT
# =========================
def zscore_close_vs_ema_last_row(df: pd.DataFrame, ema_span: int = 60, prefix: str = "z_close_ema60") -> None:
    """Live-safe last-row z-score with full debug prints"""
    if df is None or df.empty or "close" not in df.columns or not is_numeric_dtype(df["close"]):
        return
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonically increasing for z-score calculation.")
    idx = df.index[-1]
    ema_col = f"ema_{ema_span}"
    if ema_col not in df.columns:
        print(f"[DEBUG z_close] ema_{ema_span} column missing")
        return
    if pd.isna(df.at[idx, ema_col]):
        print(f"[DEBUG z_close] ema_{ema_span} is NaN at last row")
        return
    dev = df["close"] - df[ema_col]
    lookback = min(390, len(dev))
    tail = dev.iloc[-lookback:]
    mu = tail.mean()
    sd = tail.std(ddof=0)
    current_dev = dev.iloc[-1]
    #print(f"[z_close] bar:{len(df)-1:4d} | close:{df['close'].iloc[-1]:8.2f} | "
    # f"ema{ema_span}:{df.at[idx, ema_col]:8.4f} | dev:{current_dev:8.4f} | "
    # f"lookback:{lookback:3d} | mu:{mu:8.4f} | sd:{sd:8.4f}", end=" → ")
    if pd.isna(sd) or sd == 0:
        df.at[idx, prefix] = np.nan
        #print("Z = NaN (sd=0 or NaN)")
    else:
        z = (current_dev - mu) / sd
        df.at[idx, prefix] = z
        #print(f"Z = {z:+8.4f}")

def zscore_ema60_vs_ema120_last_row(df: pd.DataFrame, prefix: str = "z_ema60_ema120") -> None:
    """Live-safe EMA crossover z-score with debug"""
    if df is None or df.empty or "close" not in df.columns or not is_numeric_dtype(df["close"]):
        return
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonically increasing for z-score calculation.")
    idx = df.index[-1]
    if "ema_60" not in df.columns or "ema_120" not in df.columns:
        print("[DEBUG z_ema] ema_60 or ema_120 column missing")
        return
    if pd.isna(df.at[idx, "ema_60"]) or pd.isna(df.at[idx, "ema_120"]):
        print(f"[DEBUG z_ema] ema_60={df.at[idx, 'ema_60']} or ema_120={df.at[idx, 'ema_120']} is NaN")
        return
    diff = df["ema_60"] - df["ema_120"]
    lookback = min(390, len(diff))
    tail = diff.iloc[-lookback:]
    mu = tail.mean()
    sd = tail.std(ddof=0)
    current_diff = diff.iloc[-1]
    #print(f"[z_ema ] bar:{len(df)-1:4d} | ema60:{df.at[idx, 'ema_60']:8.4f} | "
    # f"ema120:{df.at[idx, 'ema_120']:8.4f} | diff:{current_diff:8.4f} | "
    # f"lookback:{lookback:3d} | mu:{mu:8.4f} | sd:{sd:8.4f}", end=" → ")
    if pd.isna(sd) or sd == 0:
        df.at[idx, prefix] = np.nan
        #print("Z = NaN")
    else:
        z = (current_diff - mu) / sd
        df.at[idx, prefix] = z
        #print(f"Z = {z:+8.4f}")

def ema_slope_over_last_row(df: pd.DataFrame, span: int = 60, prefix: str = "ema_slope_60") -> None:
    """Live-safe EMA slope with debug"""
    if df is None or df.empty or "close" not in df.columns or not is_numeric_dtype(df["close"]):
        return
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonically increasing for EMA slope calculation.")
    curr_pos = len(df) - 1
    if curr_pos + 1 < span:
        return
    ema_col = f"ema_{span}"
    if ema_col not in df.columns or pd.isna(df.iat[curr_pos, df.columns.get_loc(ema_col)]):
        return
    ema_now = df.iat[curr_pos, df.columns.get_loc(ema_col)]
    prev_pos = curr_pos - span
    if prev_pos < 0:
        return
    ema_prev = df.iat[prev_pos, df.columns.get_loc(ema_col)]
    if pd.isna(ema_prev):
        return
    # ATR fallback
    atr_val = 1.0
    for col in ["atr_90", "atr_60", "atr_14"]:
        if col in df.columns and not pd.isna(df.iat[curr_pos, df.columns.get_loc(col)]):
            atr_val = df.iat[curr_pos, df.columns.get_loc(col)]
            break
    slope = (ema_now - ema_prev) / (span * atr_val)
    if prefix not in df.columns:
        df[prefix] = np.nan
    df.iat[curr_pos, df.columns.get_loc(prefix)] = slope
    #print(f"[slope ] bar:{len(df)-1:4d} | ema{span}_now:{ema_now:8.4f} | ema{span}_prev:{ema_prev:8.4f} | "
    # f"atr:{atr_val:6.2f} → slope = {slope:+.6f}")
