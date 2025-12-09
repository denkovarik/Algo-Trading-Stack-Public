# PYTHONPATH=. pytest testing/unit_tests/Trading_Environment/test_indicator_helpers.py

import pytest
import pandas as pd
from classes.Trading_Environment import (
    update_indicators_last_row,
    update_indicators_at_index,
    DEFAULT_INDICATOR_CONFIG,
)

def _df_with_len(n=30, start=100.0):
    close = pd.Series([start + i * 0.1 for i in range(n)], name="close")
    high = close + 0.2
    low = close - 0.2
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": 1.0})

def test_update_indicators_last_row_small_windows_ok():
    df = _df_with_len(10)  # smaller than some defaults (e.g., EMA 50)
    # Use a custom, tiny config that fits into 10 rows to avoid all-NaN
    cfg = [
        {"name": "bollinger_bands", "params": {"window": 5, "num_std": 2, "prefix": "bb"}},
        {"name": "ema", "params": {"span": 5, "prefix": "ema"}},
        {"name": "rsi", "params": {"window": 5, "prefix": "rsi"}},
        {"name": "atr", "params": {"window": 5, "prefix": "atr"}},
    ]
    update_indicators_last_row(df, indicator_config=cfg)
    last = df.iloc[-1]
    assert not pd.isna(last.get("bb_ma"))
    assert not pd.isna(last.get("ema_5"))
    assert not pd.isna(last.get("rsi_5"))
    assert not pd.isna(last.get("atr_5"))
    assert not pd.isna(last.get("tr"))

def test_update_indicators_at_index_bounds_and_partial_history():
    df = _df_with_len(20)
    # Index before windows warm up should still work due to bounded slicing
    update_indicators_at_index(df, 0)   # should not crash
    update_indicators_at_index(df, 5)
    update_indicators_at_index(df, 19)
    # spot-check that last write produced values (some may be NaN early, but keys exist)
    assert "ema_21" in df.columns or "ema_50" in df.columns or "ema_5" in df.columns

