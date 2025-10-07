# classes/indicators/session_struct.py
from __future__ import annotations
import numpy as np
import pandas as pd

# Optional: use ATR.ensure_tr if available; otherwise fall back to quick TR (H/L based)
try:
    from classes.indicators.atr import ensure_tr
except Exception:
    ensure_tr = None

# -------------------- Rolling range width --------------------

def rolling_range_width_full(df: pd.DataFrame, window: int = 60, prefix: str = "rw") -> None:
    """Full: mean(H-L, N) / EMA(close, N)."""
    if df is None or df.empty: return
    w = int(window)
    rr = (df["high"] - df["low"]).rolling(w).mean()
    ema_ = df["close"].ewm(span=w, adjust=False).mean().replace(0, np.nan)
    df.loc[:, f"{prefix}_{w}"] = rr / ema_

def rolling_range_width_last_row(df: pd.DataFrame, window: int = 60, prefix: str = "rw") -> None:
    if df is None or df.empty: return
    n = len(df); idx = df.index[-1]; w = int(window)
    hw = df["high"].iloc[max(0, n - w):n]
    lw = df["low" ].iloc[max(0, n - w):n]
    cw = df["close"].iloc[max(0, n - w):n]
    if hw.empty or lw.empty or cw.empty:
        rolling_range_width_full(df, window=w, prefix=prefix); return
    rr = float(np.nanmean((hw - lw).to_numpy(dtype="float64", copy=False)))
    ema_ = pd.Series(cw).ewm(span=w, adjust=False).mean().iloc[-1]
    df.at[idx, f"{prefix}_{w}"] = rr / (ema_ if ema_ != 0 else np.nan)

def rolling_range_width_at_index(df: pd.DataFrame, idx: int, window: int = 60, prefix: str = "rw") -> None:
    if df is None or df.empty or idx < 0 or idx >= len(df): return
    w = int(window)
    hw = df["high"].iloc[max(0, idx - w + 1): idx + 1]
    lw = df["low" ].iloc[max(0, idx - w + 1): idx + 1]
    cw = df["close"].iloc[max(0, idx - w + 1): idx + 1]
    if hw.empty or lw.empty or cw.empty:
        df.at[idx, f"{prefix}_{w}"] = np.nan; return
    rr = float(np.nanmean((hw - lw).to_numpy(dtype="float64", copy=False)))
    ema_ = pd.Series(cw).ewm(span=w, adjust=False).mean().iloc[-1]
    df.at[idx, f"{prefix}_{w}"] = rr / (ema_ if ema_ != 0 else np.nan)

# -------------------- Candle body percent --------------------

def candle_body_pct_full(df: pd.DataFrame, prefix: str = "cb") -> None:
    if df is None or df.empty: return
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    df.loc[:, prefix] = body / rng

def candle_body_pct_last_row(df: pd.DataFrame, prefix: str = "cb") -> None:
    if df is None or df.empty: return
    idx = df.index[-1]
    rng = float(df["high"].iloc[-1] - df["low"].iloc[-1])
    body = float(df["close"].iloc[-1] - df["open"].iloc[-1])
    df.at[idx, prefix] = (abs(body) / rng) if rng != 0 else np.nan

def candle_body_pct_at_index(df: pd.DataFrame, idx: int, prefix: str = "cb") -> None:
    if df is None or df.empty or idx < 0 or idx >= len(df): return
    rng = float(df.at[idx, "high"] - df.at[idx, "low"])
    body = float(df.at[idx, "close"] - df.at[idx, "open"])
    df.at[idx, prefix] = (abs(body) / rng) if rng != 0 else np.nan

# -------------------- minutes since open (RTH ET) --------------------

def mins_since_open_full(df: pd.DataFrame, tz: str = "America/New_York", prefix: str = "mins_since_open") -> None:
    if df is None or df.empty or "date" not in df.columns or df["date"].isna().all():
        df.loc[:, prefix] = np.nan; return
    local = pd.to_datetime(df["date"], utc=True).dt.tz_convert(tz)
    mins = (local - local.dt.normalize() - pd.Timedelta(hours=9, minutes=30)).dt.total_seconds() / 60.0
    t = local.dt.time
    start = pd.Timestamp("09:30").time(); end = pd.Timestamp("16:00").time()
    mask = (t >= start) & (t <= end)
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    out[mask] = mins[mask]
    df.loc[:, prefix] = out

def mins_since_open_last_row(df: pd.DataFrame, tz: str = "America/New_York", prefix: str = "mins_since_open") -> None:
    if df is None or df.empty or "date" not in df.columns or pd.isna(df.iloc[-1]["date"]): return
    idx = df.index[-1]
    local = pd.to_datetime(df.at[idx, "date"], utc=True).tz_convert(tz)
    t = local.time()
    if (t >= pd.Timestamp("09:30").time()) and (t <= pd.Timestamp("16:00").time()):
        mins = (local - pd.Timestamp(local.date(), tz=tz) - pd.Timedelta(hours=9, minutes=30)).total_seconds() / 60.0
        df.at[idx, prefix] = mins
    else:
        df.at[idx, prefix] = np.nan

def mins_since_open_at_index(df: pd.DataFrame, idx: int, tz: str = "America/New_York", prefix: str = "mins_since_open") -> None:
    if df is None or df.empty or idx < 0 or idx >= len(df) or "date" not in df.columns or pd.isna(df.at[idx, "date"]):
        return
    local = pd.to_datetime(df.at[idx, "date"], utc=True).tz_convert(tz)
    t = local.time()
    if (t >= pd.Timestamp("09:30").time()) and (t <= pd.Timestamp("16:00").time()):
        mins = (local - pd.Timestamp(local.date(), tz=tz) - pd.Timedelta(hours=9, minutes=30)).total_seconds() / 60.0
        df.at[idx, prefix] = mins
    else:
        df.at[idx, prefix] = np.nan

# -------------------- ToD cyclical (sin/cos) --------------------

def tod_cyclical_full(df: pd.DataFrame, session_len: int = 390, src_col: str = "mins_since_open", prefix: str = "tod") -> None:
    if src_col not in df.columns:
        mins_since_open_full(df, prefix=src_col)
    m = df[src_col].astype("float64")
    angle = 2 * np.pi * (m / float(session_len))
    df.loc[:, f"{prefix}_sin"] = np.sin(angle)
    df.loc[:, f"{prefix}_cos"] = np.cos(angle)

def tod_cyclical_last_row(df: pd.DataFrame, session_len: int = 390, src_col: str = "mins_since_open", prefix: str = "tod") -> None:
    if src_col not in df.columns:
        mins_since_open_last_row(df, prefix=src_col)
    idx = df.index[-1]
    val = df.at[idx, src_col] if src_col in df.columns else np.nan
    ang = 2 * np.pi * (val / float(session_len)) if pd.notna(val) else np.nan
    df.at[idx, f"{prefix}_sin"] = np.sin(ang) if pd.notna(ang) else np.nan
    df.at[idx, f"{prefix}_cos"] = np.cos(ang) if pd.notna(ang) else np.nan

def tod_cyclical_at_index(df: pd.DataFrame, idx: int, session_len: int = 390, src_col: str = "mins_since_open", prefix: str = "tod") -> None:
    if src_col not in df.columns:
        mins_since_open_at_index(df, idx, prefix=src_col)
    val = df.at[idx, src_col] if src_col in df.columns else np.nan
    ang = 2 * np.pi * (val / float(session_len)) if pd.notna(val) else np.nan
    df.at[idx, f"{prefix}_sin"] = np.sin(ang) if pd.notna(ang) else np.nan
    df.at[idx, f"{prefix}_cos"] = np.cos(ang) if pd.notna(ang) else np.nan

# -------------------- Intraday TR stats (z & pct) --------------------

def intraday_tr_stats_full(df: pd.DataFrame, prefix: str = "trday") -> None:
    if ensure_tr is not None:
        ensure_tr(df)
    elif "tr" not in df.columns:
        df.loc[:, "tr"] = (df["high"].fillna(df["close"]) - df["low"].fillna(df["close"])).abs()

    key = (pd.to_datetime(df['date']).dt.date if 'date' in df.columns else df.index.date)
    tr = df['tr'].astype('float64')
    mu = tr.groupby(key).transform('mean')
    sd = tr.groupby(key).transform('std').replace(0, np.nan)
    df.loc[:, f"{prefix}_z"] = (tr - mu) / sd
    df.loc[:, f"{prefix}_pct"] = tr.groupby(key).transform(lambda s: s.rank(pct=True))

def intraday_tr_stats_last_row(df: pd.DataFrame, prefix: str = "trday") -> None:
    if ensure_tr is not None:
        ensure_tr(df)
    elif "tr" not in df.columns:
        df.loc[:, "tr"] = (df["high"].fillna(df["close"]) - df["low"].fillna(df["close"])).abs()

    idx = df.index[-1]
    day = pd.to_datetime(df.at[idx, 'date']).date() if 'date' in df.columns else df.index[idx].date()
    mask = ((pd.to_datetime(df['date']).dt.date == day) if 'date' in df.columns else (df.index.date == day))
    s = df.loc[mask & (df.index <= idx), 'tr'].astype('float64')
    if len(s):
        mu, sd = s.mean(), s.std()
        z = (s.iloc[-1] - mu) / sd if sd and not np.isnan(sd) and sd != 0 else np.nan
        pct = s.rank(pct=True).iloc[-1]
    else:
        z, pct = np.nan, np.nan
    df.at[idx, f'{prefix}_z'] = z
    df.at[idx, f'{prefix}_pct'] = pct

def intraday_tr_stats_at_index(df: pd.DataFrame, idx: int, prefix: str = "trday") -> None:
    if ensure_tr is not None:
        ensure_tr(df)
    elif "tr" not in df.columns:
        df.loc[:, "tr"] = (df["high"].fillna(df["close"]) - df["low"].fillna(df["close"])).abs()

    day = pd.to_datetime(df.at[idx, 'date']).date() if 'date' in df.columns else df.index[idx].date()
    mask = ((pd.to_datetime(df['date']).dt.date == day) if 'date' in df.columns else (df.index.date == day))
    s = df.loc[mask & (df.index <= idx), 'tr'].astype('float64')
    if len(s):
        mu, sd = s.mean(), s.std()
        z = (s.iloc[-1] - mu) / sd if sd and not np.isnan(sd) and sd != 0 else np.nan
        pct = s.rank(pct=True).iloc[-1]
    else:
        z, pct = np.nan, np.nan
    df.at[idx, f'{prefix}_z'] = z
    df.at[idx, f'{prefix}_pct'] = pct

# -------------------- Range contraction (declines & slope) --------------------

def range_contraction_full(df: pd.DataFrame, k: int = 5, prefix: str = "rc") -> None:
    rng = (df["high"] - df["low"]).astype("float64")
    dec = (rng.diff() < 0)
    grp = (rng.diff() >= 0).cumsum()
    df.loc[:, f"{prefix}_n_declines"] = dec.groupby(grp).cumsum().fillna(0.0)

    def _slope(x):
        n = len(x); idx = np.arange(n, dtype=float)
        xbar = np.nanmean(x); ibar = (n - 1) / 2.0
        num = ((idx - ibar) * (x - xbar)).sum()
        den = ((idx - ibar) ** 2).sum()
        return np.nan if den == 0 else num / den

    df.loc[:, f"{prefix}_slope_{k}"] = rng.rolling(int(k)).apply(_slope, raw=False)

def range_contraction_last_row(df: pd.DataFrame, k: int = 5, prefix: str = "rc") -> None:
    idx = df.index[-1]
    w = int(max(8, k * 4))
    rng = (df["high"] - df["low"]).astype("float64").iloc[max(0, len(df) - w):]
    if rng.size < 2:
        df.at[idx, f"{prefix}_n_declines"] = np.nan
        df.at[idx, f"{prefix}_slope_{k}"]  = np.nan
        return
    diffs = np.diff(rng.to_numpy())
    n_decl = 0
    for d in diffs[::-1]:
        if d < 0: n_decl += 1
        else: break
    df.at[idx, f"{prefix}_n_declines"] = n_decl
    if rng.size >= k:
        y = rng.to_numpy()[-k:]; x = np.arange(len(y), dtype=float)
        ybar, xbar = np.nanmean(y), (len(y)-1)/2.0
        num = ((x - xbar)*(y - ybar)).sum()
        den = ((x - xbar)**2).sum()
        df.at[idx, f"{prefix}_slope_{k}"] = (num/den) if den else np.nan
    else:
        df.at[idx, f"{prefix}_slope_{k}"] = np.nan

def range_contraction_at_index(df: pd.DataFrame, idx: int, k: int = 5, prefix: str = "rc") -> None:
    w = int(max(8, k * 4))
    rng = (df["high"] - df["low"]).astype("float64").iloc[max(0, idx - w + 1): idx + 1]
    if rng.size < 2:
        df.at[idx, f"{prefix}_n_declines"] = np.nan
        df.at[idx, f"{prefix}_slope_{k}"]  = np.nan
        return
    diffs = np.diff(rng.to_numpy())
    n_decl = 0
    for d in diffs[::-1]:
        if d < 0: n_decl += 1
        else: break
    df.at[idx, f"{prefix}_n_declines"] = n_decl
    if rng.size >= k:
        y = rng.to_numpy()[-k:]; x = np.arange(len(y), dtype=float)
        ybar, xbar = np.nanmean(y), (len(y)-1)/2.0
        num = ((x - xbar)*(y - ybar)).sum()
        den = ((x - xbar)**2).sum()
        df.at[idx, f"{prefix}_slope_{k}"] = (num/den) if den else np.nan
    else:
        df.at[idx, f"{prefix}_slope_{k}"] = np.nan

# -------------------- Intraday extremes & position --------------------

def intraday_extremes_full(df: pd.DataFrame, atr_col: str | None = None, prefix: str = "iday") -> None:
    key = (pd.to_datetime(df['date']).dt.date if 'date' in df.columns else df.index.date)
    day_high = df["high"].groupby(key).cummax()
    day_low  = df["low" ].groupby(key).cummin()
    close = df["close"]
    dist_hi = (day_high - close)
    dist_lo = (close - day_low)
    day_rng = (day_high - day_low).replace(0, np.nan)
    df.loc[:, f"{prefix}_dist_to_high"] = dist_hi
    df.loc[:, f"{prefix}_dist_to_low"]  = dist_lo
    df.loc[:, f"{prefix}_pos"]          = (close - day_low) / day_rng
    if atr_col and atr_col in df.columns:
        atrv = df[atr_col].replace(0, np.nan)
        df.loc[:, f"{prefix}_dist_to_high_atr"] = dist_hi / atrv
        df.loc[:, f"{prefix}_dist_to_low_atr"]  = dist_lo / atrv

def intraday_extremes_last_row(df: pd.DataFrame, atr_col: str | None = None, prefix: str = "iday") -> None:
    idx = df.index[-1]
    day = pd.to_datetime(df.at[idx, 'date']).date() if 'date' in df.columns else df.index[idx].date()
    mask = ((pd.to_datetime(df['date']).dt.date == day) if 'date' in df.columns else (df.index.date == day))
    hi_today = float(df.loc[mask & (df.index <= idx), 'high'].cummax().iloc[-1])
    lo_today = float(df.loc[mask & (df.index <= idx), 'low' ].cummin().iloc[-1])
    c = float(df.at[idx, 'close'])
    dist_hi = hi_today - c
    dist_lo = c - lo_today
    rng_day = hi_today - lo_today
    df.at[idx, f'{prefix}_dist_to_high'] = dist_hi
    df.at[idx, f'{prefix}_dist_to_low']  = dist_lo
    df.at[idx, f'{prefix}_pos']          = (c - lo_today) / rng_day if rng_day != 0 else np.nan
    if atr_col and atr_col in df.columns and pd.notna(df.at[idx, atr_col]):
        atrv = float(df.at[idx, atr_col])
        df.at[idx, f'{prefix}_dist_to_high_atr'] = (dist_hi / atrv) if atrv else np.nan
        df.at[idx, f'{prefix}_dist_to_low_atr']  = (dist_lo / atrv) if atrv else np.nan

def intraday_extremes_at_index(df: pd.DataFrame, idx: int, atr_col: str | None = None, prefix: str = "iday") -> None:
    day = pd.to_datetime(df.at[idx, 'date']).date() if 'date' in df.columns else df.index[idx].date()
    mask = ((pd.to_datetime(df['date']).dt.date == day) if 'date' in df.columns else (df.index.date == day))
    hi_today = float(df.loc[mask & (df.index <= idx), 'high'].cummax().iloc[-1])
    lo_today = float(df.loc[mask & (df.index <= idx), 'low' ].cummin().iloc[-1])
    c = float(df.at[idx, 'close'])
    dist_hi = hi_today - c
    dist_lo = c - lo_today
    rng_day = hi_today - lo_today
    df.at[idx, f'{prefix}_dist_to_high'] = dist_hi
    df.at[idx, f'{prefix}_dist_to_low']  = dist_lo
    df.at[idx, f'{prefix}_pos']          = (c - lo_today) / rng_day if rng_day != 0 else np.nan
    if atr_col and atr_col in df.columns and pd.notna(df.at[idx, atr_col]):
        atrv = float(df.at[idx, atr_col])
        df.at[idx, f'{prefix}_dist_to_high_atr'] = (dist_hi / atrv) if atrv else np.nan
        df.at[idx, f'{prefix}_dist_to_low_atr']  = (dist_lo / atrv) if atrv else np.nan

# -------------------- Backward-compatible aliases --------------------
# (Keep old names working; they now refer to the "full" variants.)

def rolling_range_width(df, window=60, prefix="rw"): return rolling_range_width_full(df, window, prefix)
def candle_body_pct(df, prefix="cb"): return candle_body_pct_full(df, prefix)
def mins_since_open(df, tz="America/New_York", prefix="mins_since_open"): return mins_since_open_full(df, tz, prefix)
def tod_cyclical(df, session_len=390, src_col="mins_since_open", prefix="tod"): return tod_cyclical_full(df, session_len, src_col, prefix)
def intraday_tr_stats(df, prefix="trday"): return intraday_tr_stats_full(df, prefix)
def range_contraction(df, k=5, prefix="rc"): return range_contraction_full(df, k, prefix)
def intraday_extremes(df, atr_col=None, prefix="iday"): return intraday_extremes_full(df, atr_col, prefix)

