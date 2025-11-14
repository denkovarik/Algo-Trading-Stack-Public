# classes/indicators/update_router.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Mapping, Iterable, Any

# Import last_row / at_index helpers for every indicator we route
from classes.indicators.bollinger import (
    bollinger_last_row, bandwidth_last_row,
    bollinger_at_index, bandwidth_at_index,
)
from classes.indicators.ema import ema_last_row, ema_at_index
from classes.indicators.rsi import rsi_last_row, rsi_at_index
from classes.indicators.atr import (
    atr_last_row, atr_at_index,
    atr_ratio_last_row, atr_ratio_at_index,
    atr_ratio_sm_last_row, atr_ratio_sm_at_index,
)
from classes.indicators.donchian import (
    donchian_last_row, donchian_at_index,
    donchian_pos_last_row, donchian_pos_at_index,
)
from classes.indicators.macd import macd_last_row, macd_at_index
from classes.indicators.roc import roc_last_row, roc_at_index
from classes.indicators.vwap import (
    vwap_last_row, vwap_at_index,
    vwap_rth_last_row, vwap_rth_at_index,
    vwap_dev_atr_last_row, vwap_dev_atr_at_index, 
)
from classes.indicators.session_struct import (
    rolling_range_width_last_row, rolling_range_width_at_index,
    candle_body_pct_last_row,  candle_body_pct_at_index,
    mins_since_open_last_row,  mins_since_open_at_index,
    tod_cyclical_last_row,     tod_cyclical_at_index,
    intraday_tr_stats_last_row, intraday_tr_stats_at_index,
    range_contraction_last_row, range_contraction_at_index,
    intraday_extremes_last_row, intraday_extremes_at_index,
)

def update_indicators_last_row(
    df: pd.DataFrame,
    indicator_config: Iterable[Mapping[str, Any]],
    registry: Mapping[str, Any],   # INDICATOR_REGISTRY (for any fall-through names)
) -> None:
    """
    Fast last-row update: avoid DataFrame copies and compute just the newest values.
    Behavior mirrors the env's previous implementation; decisions about which
    indicators to compute (and their params) are driven by `indicator_config`.
    """
    if df is None or df.empty:
        return

    idx = df.index[-1]
    n = len(df)

    # convenience views (kept for parity; some branches used them previously)
    def view(series, window):
        start = max(0, n - window)
        return series.iloc[start:n].to_numpy(dtype="float64", copy=False)

    close = df["close"]
    open_  = df["open"]  if "open"  in df.columns else pd.Series(index=df.index, dtype="float64")
    high   = df["high"]  if "high"  in df.columns else pd.Series(index=df.index, dtype="float64")
    low    = df["low"]   if "low"   in df.columns else pd.Series(index=df.index, dtype="float64")

    for ind in indicator_config:
        name = ind['name']
        p = ind.get('params', {})

        if name == 'bollinger_bands':
            win = int(p.get('window', 20)); prefix = p.get('prefix', 'bb'); num_std = float(p.get('num_std', 2))
            bollinger_last_row(df, window=win, num_std=num_std, prefix=prefix)

        elif name == 'bb_bandwidth':
            win = int(p.get('window', 60)); prefix = p.get('prefix', 'bb60')
            bandwidth_last_row(df, window=win, prefix=prefix)

        elif name == 'ema':
            span = int(p.get('span', 21)); prefix = p.get('prefix', 'ema')
            ema_last_row(df, span=span, prefix=prefix)

        elif name == 'rsi':
            window = int(p.get('window', 14)); prefix = p.get('prefix', 'rsi')
            rsi_last_row(df, window=window, prefix=prefix)

        elif name == 'atr':
            win = int(p.get('window', 14)); prefix = p.get('prefix', 'atr')
            atr_last_row(df, window=win, prefix=prefix)

        elif name == 'atr_ratio':
            fast = int(p.get('fast', 90)); slow = int(p.get('slow', 390)); prefix = p.get('prefix', 'atrR')
            atr_ratio_last_row(df, fast=fast, slow=slow, prefix=prefix)

        elif name == 'atr_ratio_sm':
            fast = int(p.get('fast', 14)); slow = int(p.get('slow', 60)); prefix = p.get('prefix', 'atrRsm')
            atr_ratio_sm_last_row(df, fast=fast, slow=slow, prefix=prefix)

        elif name == 'donchian':
            window = int(p.get('window', 60)); prefix = p.get('prefix', 'dc')
            donchian_last_row(df, window=window, prefix=prefix)

        elif name == 'donchian_pos':
            window = int(p.get('window', 60)); prefix = p.get('prefix', 'dc')
            donchian_pos_last_row(df, window=window, prefix=prefix)

        elif name == 'macd':
            fast  = int(p.get('fast', 12)); slow  = int(p.get('slow', 26)); signal= int(p.get('signal', 9))
            prefix= p.get('prefix', 'macd')
            macd_last_row(df, fast=fast, slow=slow, signal=signal, prefix=prefix)

        elif name == 'roc':
            window = int(p.get('window', 30)); prefix = p.get('prefix', 'roc')
            roc_last_row(df, window=window, prefix=prefix)

        elif name == 'range_width':
            window = int(p.get('window', 60)); prefix = p.get('prefix', 'rw')
            rolling_range_width_last_row(df, window=window, prefix=prefix)

        elif name == 'candle_body_pct':
            prefix = p.get('prefix', 'cb')
            candle_body_pct_last_row(df, prefix=prefix)

        elif name == 'mins_since_open':
            tz = p.get('tz', "America/New_York"); prefix = "mins_since_open"
            mins_since_open_last_row(df, tz=tz, prefix=prefix)

        elif name == 'tod_cyclical':
            session_len = int(p.get('session_len', 390)); src = p.get('src_col', 'mins_since_open'); prefix = p.get('prefix','tod')
            tod_cyclical_last_row(df, session_len=session_len, src_col=src, prefix=prefix)

        elif name == 'intraday_tr_stats':
            prefix = p.get('prefix','trday')
            intraday_tr_stats_last_row(df, prefix=prefix)

        elif name == 'range_contraction':
            k = int(p.get('k',5)); prefix = p.get('prefix','rc')
            range_contraction_last_row(df, k=k, prefix=prefix)

        elif name == 'intraday_extremes':
            atr_col = p.get('atr_col', None); prefix = p.get('prefix','iday')
            intraday_extremes_last_row(df, atr_col=atr_col, prefix=prefix)

        elif name == 'vwap_rth':
            pcol = p.get('price_col', 'close'); prefix = p.get('prefix', 'vwap'); tz = p.get('tz', "America/New_York")
            vwap_rth_last_row(df, price_col=pcol, vol_col='volume', prefix=prefix, tz=tz)

        elif name == 'vwap_dev_atr':
            atr_col = p.get('atr_col', 'atr_90'); vcol = p.get('vwap_col', 'vwap'); prefix = p.get('prefix', 'vwap_dev_atr')
            if vcol not in df.columns or pd.isna(df.iloc[-1].get(vcol, np.nan)):
                vwap_last_row(df, prefix=vcol)
            vwap_dev_atr_last_row(df, atr_col=atr_col, vwap_col=vcol, prefix=prefix)

        else:
            # Fall-through: allow registry functions (rare) that only need df+params at last row
            func = registry.get(name)
            if callable(func):
                try:
                    func(df, **p)
                except Exception:
                    # keep last-row router robust
                    pass


def update_indicators_at_index(
    df: pd.DataFrame,
    idx: int,
    indicator_config: Iterable[Mapping[str, Any]],
    registry: Mapping[str, Any],   # INDICATOR_REGISTRY
) -> None:
    """
    Compute indicators for a specific index `idx` using only past data up to idx.
    Mirrors env's previous implementation; any unknown names delegate to `registry`.
    """
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df):
        return

    for ind in indicator_config:
        name   = ind['name']
        params = ind.get('params', {})

        if name == 'bollinger_bands':
            window = int(params.get('window', 20)); prefix = params.get('prefix', 'bb'); num_std = float(params.get('num_std', 2))
            bollinger_at_index(df, idx, window=window, num_std=num_std, prefix=prefix)

        elif name == 'bb_bandwidth':
            window = int(params.get('window', 60)); prefix = params.get('prefix', 'bb60')
            bandwidth_at_index(df, idx, window=window, prefix=prefix)

        elif name == 'ema':
            span = int(params.get('span', 21)); prefix = params.get('prefix', 'ema')
            ema_at_index(df, idx, span=span, prefix=prefix)

        elif name == 'rsi':
            window = int(params.get('window', 14)); prefix = params.get('prefix', 'rsi')
            rsi_at_index(df, idx, window=window, prefix=prefix)

        elif name == 'atr':
            window = int(params.get('window', 14)); prefix = params.get('prefix', 'atr')
            atr_at_index(df, idx, window=window, prefix=prefix)

        elif name == 'atr_ratio':
            fast = int(params.get('fast', 90)); slow = int(params.get('slow', 390)); prefix = params.get('prefix', 'atrR')
            atr_ratio_at_index(df, idx, fast=fast, slow=slow, prefix=prefix)

        elif name == 'atr_ratio_sm':
            fast = int(params.get('fast', 14)); slow = int(params.get('slow', 60)); prefix = params.get('prefix', 'atrRsm')
            atr_ratio_sm_at_index(df, idx, fast=fast, slow=slow, prefix=prefix)

        elif name == 'donchian':
            window = int(params.get('window', 60)); prefix = params.get('prefix', 'dc')
            donchian_at_index(df, idx, window=window, prefix=prefix)

        elif name == 'donchian_pos':
            window = int(params.get('window', 60)); prefix = params.get('prefix', 'dc')
            donchian_pos_at_index(df, idx, window=window, prefix=prefix)

        elif name == 'macd':
            fast = int(params.get('fast', 12)); slow = int(params.get('slow', 26)); signal = int(params.get('signal', 9)); prefix = params.get('prefix', 'macd')
            macd_at_index(df, idx, fast=fast, slow=slow, signal=signal, prefix=prefix)

        elif name == 'roc':
            window = int(params.get('window', 30)); prefix = params.get('prefix', 'roc')
            roc_at_index(df, idx, window=window, prefix=prefix)

        elif name == 'vwap_rth':
            pcol = params.get('price_col', 'close'); prefix = params.get('prefix', 'vwap'); tz = params.get('tz', "America/New_York")
            vwap_rth_at_index(df, idx, price_col=pcol, vol_col='volume', prefix=prefix, tz=tz)

        elif name == 'vwap_dev_atr':
            atr_col = params.get('atr_col', 'atr_90'); vcol = params.get('vwap_col', 'vwap'); prefix = params.get('prefix', 'vwap_dev_atr')
            if vcol not in df.columns or pd.isna(df.iloc[idx].get(vcol, np.nan)):
                vwap_at_index(df, idx, prefix=vcol)
            vwap_dev_atr_at_index(df, idx, atr_col=atr_col, vwap_col=vcol, prefix=prefix)

        elif name == 'range_width':
            window = int(params.get('window', 60)); prefix = params.get('prefix', 'rw')
            rolling_range_width_at_index(df, idx, window=window, prefix=prefix)

        elif name == 'candle_body_pct':
            prefix = params.get('prefix', 'cb')
            candle_body_pct_at_index(df, idx, prefix=prefix)

        elif name == 'mins_since_open':
            tz = params.get('tz', "America/New_York"); prefix = "mins_since_open"
            mins_since_open_at_index(df, idx, tz=tz, prefix=prefix)

        elif name == 'tod_cyclical':
            session_len = int(params.get('session_len', 390)); src = params.get('src_col', 'mins_since_open'); prefix = params.get('prefix','tod')
            tod_cyclical_at_index(df, idx, session_len=session_len, src_col=src, prefix=prefix)

        elif name == 'intraday_tr_stats':
            prefix = params.get('prefix','trday')
            intraday_tr_stats_at_index(df, idx, prefix=prefix)

        elif name == 'range_contraction':
            k = int(params.get('k',5)); prefix = params.get('prefix','rc')
            range_contraction_at_index(df, idx, k=k, prefix=prefix)

        elif name == 'intraday_extremes':
            atr_col = params.get('atr_col', None); prefix = params.get('prefix','iday')
            intraday_extremes_at_index(df, idx, atr_col=atr_col, prefix=prefix)

        elif name in ('z_close_ema60', 'z_ema60_ema120', 'ema_slope_60_120'):
            # Delegate to registry (uses full functions that can safely compute at idx)
            func = registry.get(name)
            if callable(func):
                try:
                    func(df, **params)
                except Exception:
                    pass

