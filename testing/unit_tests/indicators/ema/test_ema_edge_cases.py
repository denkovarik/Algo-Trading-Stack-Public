# testing/unit_tests/indicators/ema/test_ema_edge_cases.py
import numpy as np
import pandas as pd

from classes.indicators.ema import (
    ema_full, ema_last_row, ema_at_index
)

def make_constant_df(n=30, value=100.0):
    close = pd.Series(np.full(n, value, dtype=float), name="close")
    ts = pd.date_range("2025-01-01 09:30:00", periods=n, freq="min", tz="UTC")
    return pd.DataFrame({"date": ts, "open": close, "high": close, "low": close, "close": close})

def test_ema_constant_series_equals_constant():
    df = make_constant_df(50, 123.45)
    span = 20
    ema_full(df, span=span, prefix="ema")
    assert np.allclose(df[f"ema_{span}"].dropna().to_numpy(), 123.45)

def test_ema_at_index_boundaries():
    df = make_constant_df(15, 101.0)
    span = 5
    # At index 0
    ema_at_index(df, 0, span=span, prefix="ema")
    assert np.isfinite(df.loc[0, f"ema_{span}"])
    # At last index
    ema_at_index(df, len(df)-1, span=span, prefix="ema")
    assert np.isfinite(df.loc[len(df)-1, f"ema_{span}"])

def test_ema_last_row_seeds_from_tail_when_prev_missing():
    df = make_constant_df(8, 200.0)
    span = 6
    # No prior EMA column — call last_row to seed
    ema_last_row(df, span=span, prefix="ema")
    assert np.isfinite(df[f"ema_{span}"].iloc[-1])
    # Now that prior exists, another last_row call should be identical (no drift)
    prev = df[f"ema_{span}"].iloc[-1]
    ema_last_row(df, span=span, prefix="ema")
    assert df[f"ema_{span}"].iloc[-1] == prev

