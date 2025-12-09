# classes/indicators/update_router.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Mapping, Iterable, Any

from classes.indicators.ema import (
    ema_full, ema_at_index, ema_last_row,
    zscore_close_vs_ema_last_row, zscore_ema60_vs_ema120_last_row, ema_slope_over_last_row,
)
from classes.indicators.rsi import rsi_last_row, rsi_at_index
from classes.indicators.atr import atr_last_row, atr_at_index
from classes.indicators.macd import macd_last_row, macd_at_index
from classes.indicators.adx import adx_last_row, adx_at_index
from classes.indicators.sma import (
    sma_last_row, sma_at_index,
    ema_sma_crossover_last_row, ema_sma_crossover_at_index,
)
from classes.indicators.bollinger import bollinger_last_row, bollinger_at_index
from classes.indicators.rsi_divergence import rsi_divergence_last_row, rsi_divergence_at_index
from classes.indicators.psar import psar_last_row, psar_at_index

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

        if name == 'ema':
            span = int(p.get('span', 21))
            prefix = p.get('prefix', 'ema')
            ema_last_row(df, span=span, prefix=prefix)

        elif name == 'z_close_ema60':
            zscore_close_vs_ema_last_row(df, **p)
            
        elif name == 'z_ema60_ema120':
            zscore_ema60_vs_ema120_last_row(df, **p)
            
        elif name == 'ema_slope_60_120':
            ema_slope_over_last_row(df, **p)
            
        elif name == 'rsi':
            window = int(p.get('window', 14))
            prefix = p.get('prefix', 'rsi')
            rsi_last_row(df, window=window, prefix=prefix)

        elif name == 'atr':
            win = int(p.get('window', 14))
            prefix = p.get('prefix', 'atr')
            atr_last_row(df, window=win, prefix=prefix)

        elif name == 'macd':
            fast = int(p.get('fast', 12))
            slow = int(p.get('slow', 26))
            signal = int(p.get('signal', 9))
            prefix = p.get('prefix', 'macd')
            macd_last_row(df, fast=fast, slow=slow, signal=signal, prefix=prefix)

        elif name == 'adx':
            window = int(p.get('window', 14))
            prefix = p.get('prefix', 'adx')
            adx_last_row(df, window=window, prefix=prefix)

        elif name == 'sma':
            window = int(p.get('window', 20))
            prefix = p.get('prefix', 'sma')
            sma_last_row(df, window=window, prefix=prefix)

        elif name == 'ema_sma_crossover':
            ema_span = int(p.get('ema_span', 10))
            sma_window = int(p.get('sma_window', 20))
            prefix = p.get('prefix', 'crossover')
            ema_sma_crossover_last_row(df, ema_span=ema_span, sma_window=sma_window, prefix=prefix)

        elif name == 'bollinger_bands':
            window = int(p.get('window', 20))
            std_mult = float(p.get('std_mult', 2.0))
            prefix = p.get('prefix', 'bb')
            bollinger_last_row(df, window=window, std_mult=std_mult, prefix=prefix)

        elif name == 'rsi_divergence':
            rsi_window = int(p.get('rsi_window', 14))
            lookback = int(p.get('lookback', 5))
            prefix = p.get('prefix', 'rsi_div')
            rsi_divergence_last_row(df, rsi_window=rsi_window, lookback=lookback, prefix=prefix)

        elif name == 'psar':
            af_start = float(p.get('af_start', 0.02))
            af_step = float(p.get('af_step', 0.02))
            af_max = float(p.get('af_max', 0.2))
            prefix = p.get('prefix', 'psar')
            psar_last_row(df, af_start=af_start, af_step=af_step, af_max=af_max, prefix=prefix)

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
    registry: Mapping[str, Any], # INDICATOR_REGISTRY
) -> None:
    """
    Compute indicators for a specific index `idx` using only past data up to idx.
    Mirrors env's previous implementation; any unknown names delegate to `registry`.
    """
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df):
        return
    for ind in indicator_config:
        name = ind['name']
        params = ind.get('params', {})
        if name == 'ema':
            span = int(params.get('span', 21))
            prefix = params.get('prefix', 'ema')
            ema_at_index(df, idx, span=span, prefix=prefix)

        elif name == 'rsi':
            window = int(params.get('window', 14))
            prefix = params.get('prefix', 'rsi')
            rsi_at_index(df, idx, window=window, prefix=prefix)

        elif name == 'atr':
            window = int(params.get('window', 14))
            prefix = params.get('prefix', 'atr')
            atr_at_index(df, idx, window=window, prefix=prefix)

        elif name == 'macd':
            fast = int(params.get('fast', 12))
            slow = int(params.get('slow', 26))
            signal = int(params.get('signal', 9))
            prefix = params.get('prefix', 'macd')
            macd_at_index(df, idx, fast=fast, slow=slow, signal=signal, prefix=prefix)

        elif name == 'adx':
            window = int(params.get('window', 14))
            prefix = params.get('prefix', 'adx')
            adx_at_index(df, idx, window=window, prefix=prefix)

        elif name == 'sma':
            window = int(params.get('window', 20))
            prefix = params.get('prefix', 'sma')
            sma_at_index(df, idx, window=window, prefix=prefix)

        elif name == 'ema_sma_crossover':
            ema_span = int(params.get('ema_span', 10))
            sma_window = int(params.get('sma_window', 20))
            prefix = params.get('prefix', 'crossover')
            ema_sma_crossover_at_index(df, idx, ema_span=ema_span, sma_window=sma_window, prefix=prefix)

        elif name == 'bollinger_bands':
            window = int(params.get('window', 20))
            std_mult = float(params.get('std_mult', 2.0))
            prefix = params.get('prefix', 'bb')
            bollinger_at_index(df, idx, window=window, std_mult=std_mult, prefix=prefix)

        elif name == 'rsi_divergence':
            rsi_window = int(params.get('rsi_window', 14))
            lookback = int(params.get('lookback', 5))
            prefix = params.get('prefix', 'rsi_div')
            rsi_divergence_at_index(df, idx, rsi_window=rsi_window, lookback=lookback, prefix=prefix)

        elif name == 'psar':
            af_start = float(params.get('af_start', 0.02))
            af_step = float(params.get('af_step', 0.02))
            af_max = float(params.get('af_max', 0.2))
            prefix = params.get('prefix', 'psar')
            psar_at_index(df, idx, af_start=af_start, af_step=af_step, af_max=af_max, prefix=prefix)

        else:
            # Fall-through: allow registry functions (rare) that only need df+params at idx
            func = registry.get(name)
            if callable(func):
                try:
                    func(df, **params)
                except Exception:
                    # keep at-index router robust
                    pass
