# classes/indicators/vwap.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ============== Helpers ==============

def _ensure_tz_series(dt_series: pd.Series, tz: str) -> pd.Series:
    """Return a tz-aware datetime series converted to given tz."""
    if dt_series.dtype == "datetime64[ns, UTC]":
        return dt_series.dt.tz_convert(tz)
    # If naive or different tz, coerce to UTC first, then convert
    return pd.to_datetime(dt_series, utc=True).dt.tz_convert(tz)

def _rth_mask_local(local_ts: pd.Series) -> pd.Series:
    """True inside 09:30–16:00 local; False elsewhere."""
    t = local_ts.dt.time
    start = pd.Timestamp("09:30").time()
    end   = pd.Timestamp("16:00").time()
    return (t >= start) & (t <= end)

def _group_key_local_date(local_ts: pd.Series):
    """Group key by local calendar date."""
    return local_ts.dt.date

# ============== VWAP (generic, session-agnostic) ==============

def vwap_full(df: pd.DataFrame, price_col: str = "close", vol_col: str = "volume",
              prefix: str = "vwap") -> None:
    """
    Generic cumulative VWAP over the entire DataFrame (no session filtering).
    Writes {prefix}.
    """
    if df is None or df.empty or price_col not in df.columns or vol_col not in df.columns:
        df.loc[:, prefix] = np.nan
        return
    px = df[price_col].astype("float64")
    vol = df[vol_col].astype("float64")
    pv = (px * vol).cumsum()
    vv = vol.cumsum()
    df.loc[:, prefix] = pv / vv.replace(0.0, np.nan)

def vwap_last_row(df: pd.DataFrame, price_col: str = "close", vol_col: str = "volume",
                  prefix: str = "vwap") -> None:
    """Fast last-row update of generic VWAP (uses cumulative sums up to last row)."""
    # For generic (no per-day reset), full or last-row produce identical final value
    # We can simply compute pv/vv at last row using cumsum:
    if df is None or df.empty or price_col not in df.columns or vol_col not in df.columns:
        return
    px = df[price_col].astype("float64")
    vol = df[vol_col].astype("float64")
    pv = (px * vol).cumsum()
    vv = vol.cumsum()
    df.at[df.index[-1], prefix] = float(pv.iloc[-1] / (vv.iloc[-1] if vv.iloc[-1] != 0 else np.nan))

def vwap_at_index(df: pd.DataFrame, idx: int, price_col: str = "close",
                  vol_col: str = "volume", prefix: str = "vwap") -> None:
    """Per-index recompute of generic VWAP using only rows up to idx."""
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df):
        return
    sub = df.iloc[: idx + 1]
    if price_col not in sub.columns or vol_col not in sub.columns:
        df.at[idx, prefix] = np.nan
        return
    px = sub[price_col].astype("float64")
    vol = sub[vol_col].astype("float64")
    pv = (px * vol).sum()
    vv = vol.sum()
    df.at[idx, prefix] = float(pv / (vv if vv != 0 else np.nan))

# ============== VWAP_RTH (session VWAP within 09:30–16:00 local) ==============

def vwap_rth_full(df: pd.DataFrame, price_col: str = "close", vol_col: str = "volume",
                  prefix: str = "vwap", tz: str = "America/New_York") -> None:
    """
    RTH-only cumulative VWAP per local day. Off-hours rows get NaN.
    Writes {prefix}.
    """
    if df is None or df.empty or price_col not in df.columns or vol_col not in df.columns or "date" not in df.columns:
        df.loc[:, prefix] = np.nan
        return

    local = _ensure_tz_series(df["date"], tz)
    rth = _rth_mask_local(local)
    key = _group_key_local_date(local)

    px = df[price_col].astype("float64")
    vol = df[vol_col].astype("float64").where(rth, 0.0)

    pv = (px * vol).groupby(key).cumsum()
    vv = vol.groupby(key).cumsum()

    vwap_val = pv / vv.replace(0.0, np.nan)
    df.loc[:, prefix] = vwap_val.where(rth, np.nan)

def vwap_rth_last_row(df: pd.DataFrame, price_col: str = "close", vol_col: str = "volume",
                      prefix: str = "vwap", tz: str = "America/New_York") -> None:
    """
    Fast last-row update for RTH VWAP: recompute only for today's local date.
    """
    if df is None or df.empty or price_col not in df.columns or vol_col not in df.columns or "date" not in df.columns:
        return
    idx = df.index[-1]

    local = _ensure_tz_series(df["date"], tz)
    rth = _rth_mask_local(local)
    key = _group_key_local_date(local)

    # Mask subset = rows of today's local date
    today = key.iloc[-1]
    mask_today = (key == today)

    px = df.loc[mask_today, price_col].astype("float64")
    vol = df.loc[mask_today, vol_col].astype("float64").where(rth[mask_today], 0.0)

    pv = (px * vol).cumsum()
    vv = vol.cumsum()
    vwap_today = pv / vv.replace(0.0, np.nan)

    # write last value if last row is in RTH; else NaN
    df.at[idx, prefix] = float(vwap_today.iloc[-1]) if bool(rth.iloc[-1]) else np.nan

def vwap_rth_at_index(df: pd.DataFrame, idx: int, price_col: str = "close",
                      vol_col: str = "volume", prefix: str = "vwap",
                      tz: str = "America/New_York") -> None:
    """
    Per-index recompute at idx for RTH VWAP using only that day's rows up to idx.
    """
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df) or "date" not in df.columns:
        return

    local = _ensure_tz_series(df["date"], tz)
    rth = _rth_mask_local(local)
    key = _group_key_local_date(local)

    day = key.iloc[idx]
    mask_day = (key == day)
    mask_upto = mask_day & (df.index <= idx)

    px = df.loc[mask_upto, price_col].astype("float64")
    vol = df.loc[mask_upto, vol_col].astype("float64").where(rth[mask_upto], 0.0)

    pv = (px * vol).cumsum().iloc[-1]
    vv = vol.cumsum().iloc[-1]
    df.at[idx, prefix] = float(pv / (vv if vv != 0 else np.nan)) if bool(rth.iloc[idx]) else np.nan

# ============== VWAP deviation vs ATR ==============

def vwap_dev_atr_full(df: pd.DataFrame, atr_col: str | None = None, vwap_col: str = "vwap",
                      prefix: str = "vwap_dev_atr") -> None:
    """
    Full compute of (close - vwap)/ATR. If atr_col is None, tries 'atr_90' then any 'atr_*'.
    Writes {prefix}.
    """
    if df is None or df.empty or "close" not in df.columns:
        df.loc[:, prefix] = np.nan
        return

    # Ensure vwap_col exists; if missing, fallback to generic VWAP
    if vwap_col not in df.columns:
        vwap_full(df, prefix=vwap_col)

    vw = df[vwap_col]

    if atr_col and atr_col in df.columns:
        atrv = df[atr_col]
    elif "atr_90" in df.columns:
        atrv = df["atr_90"]
    else:
        # pick the first atr_* column if present
        cand = [c for c in df.columns if isinstance(c, str) and c.startswith("atr_")]
        atrv = df[cand[0]] if cand else pd.Series(np.nan, index=df.index, dtype="float64")

    numer = (df["close"] - vw).astype("float64")
    denom = atrv.replace(0.0, np.nan)
    df.loc[:, prefix] = numer / denom

def vwap_dev_atr_last_row(df: pd.DataFrame, atr_col: str | None = None, vwap_col: str = "vwap",
                          prefix: str = "vwap_dev_atr") -> None:
    """Fast last-row update for (close - vwap)/ATR."""
    if df is None or df.empty or "close" not in df.columns:
        return

    idx = df.index[-1]
    if vwap_col not in df.columns or pd.isna(df.iloc[-1].get(vwap_col, np.nan)):
        # if missing, seed generic vwap at last row
        vwap_last_row(df, prefix=vwap_col)

    if atr_col and atr_col in df.columns:
        atrv = df.at[idx, atr_col]
    elif "atr_90" in df.columns:
        atrv = df.at[idx, "atr_90"]
    else:
        # fallback to any atr_*
        cand = [c for c in df.columns if isinstance(c, str) and c.startswith("atr_")]
        atrv = df.at[idx, cand[0]] if len(cand) else np.nan

    vw = df.at[idx, vwap_col]
    cl = df.at[idx, "close"]
    denom = (atrv if (atrv is not None and not pd.isna(atrv) and atrv != 0) else np.nan)
    df.at[idx, prefix] = (cl - vw) / denom if denom == denom else np.nan

def vwap_dev_atr_at_index(df: pd.DataFrame, idx: int, atr_col: str | None = None, vwap_col: str = "vwap",
                          prefix: str = "vwap_dev_atr") -> None:
    """Per-index recompute for (close - vwap)/ATR at idx."""
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df) or "close" not in df.columns:
        return

    if vwap_col not in df.columns or pd.isna(df.iloc[idx].get(vwap_col, np.nan)):
        # seed vwap at idx with generic vwap up to idx
        vwap_at_index(df, idx, prefix=vwap_col)

    if atr_col and atr_col in df.columns:
        atrv = df.at[idx, atr_col]
    elif "atr_90" in df.columns:
        atrv = df.at[idx, "atr_90"]
    else:
        cand = [c for c in df.columns if isinstance(c, str) and c.startswith("atr_")]
        atrv = df.at[idx, cand[0]] if len(cand) else np.nan

    vw = df.at[idx, vwap_col]
    cl = df.at[idx, "close"]
    denom = (atrv if (atrv is not None and not pd.isna(atrv) and atrv != 0) else np.nan)
    df.at[idx, prefix] = (cl - vw) / denom if denom == denom else np.nan

