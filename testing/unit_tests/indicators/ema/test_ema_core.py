# testing/unit_tests/indicators/ema/test_ema_core.py
import numpy as np
import pandas as pd

from classes.indicators.ema import (
    ema_full, ema_last_row, ema_at_index
)

def make_df(n=60, seed=0, start=100.0):
    rng = np.random.default_rng(seed)
    close = pd.Series(np.cumsum(rng.normal(0, 1, n)) + start, name="close")
    high  = close + 0.3  # not used by EMA; present to look like OHLC
    low   = close - 0.3
    open_ = close.shift(1).fillna(close)
    ts = pd.date_range("2025-01-01 09:30:00", periods=n, freq="min", tz="UTC")
    return pd.DataFrame({"date": ts, "open": open_, "high": high, "low": low, "close": close})

def panda_ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False).mean()

def test_ema_full_matches_pandas_ewm():
    df = make_df(120)
    for span in [1, 2, 5, 14, 60]:
        ema_full(df, span=span, prefix="ema")
        col = f"ema_{span}"
        expected = panda_ema(df["close"], span)
        pd.testing.assert_series_equal(
            df[col].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

def test_ema_last_row_equals_full_vectorized_at_tail():
    df_full = make_df(100)
    span = 14
    # Full baseline
    ema_full(df_full, span=span, prefix="ema")
    last_full = df_full[f"ema_{span}"].iloc[-1]

    # Incremental path: compute up to n-1, append one row, then last-row update
    df_inc = df_full.iloc[:-1].copy()
    ema_full(df_inc, span=span, prefix="ema")
    df_inc = pd.concat([df_inc, df_full.iloc[[-1]]], ignore_index=True)
    ema_last_row(df_inc, span=span, prefix="ema")
    last_inc = df_inc[f"ema_{span}"].iloc[-1]

    assert np.isfinite(last_full) and np.isfinite(last_inc)
    np.testing.assert_allclose(last_inc, last_full, rtol=1e-12, atol=1e-12)

def test_ema_at_index_no_lookahead_parity():
    df = make_df(80)
    span = 21
    # Reference full column
    ema_full(df, span=span, prefix="ema")
    ref = df[f"ema_{span}"].copy()

    # Recompute at random interior indices using only past bars
    rng = np.random.default_rng(123)
    for idx in rng.choice(np.arange(5, len(df)-5), size=5, replace=False):
        df2 = df.drop(columns=[f"ema_{span}"]).copy()
        ema_at_index(df2, idx, span=span, prefix="ema")
        assert np.isfinite(ref.iloc[idx])
        np.testing.assert_allclose(df2.loc[idx, f"ema_{span}"], ref.iloc[idx], rtol=1e-12, atol=1e-12)

def test_ema_idempotent_multiple_calls():
    df = make_df(70)
    span = 30
    ema_full(df, span=span, prefix="ema")
    first = df[f"ema_{span}"].copy()
    # Call again; values should not drift
    ema_full(df, span=span, prefix="ema")
    pd.testing.assert_series_equal(first, df[f"ema_{span}"])

def test_ema_last_row_online_update_path_used_when_prev_exists():
    df = make_df(40)
    span = 10
    # Seed the column (prev EMA exists)
    ema_full(df, span=span, prefix="ema")
    prev = df[f"ema_{span}"].iloc[-1]
    # Append one new row with a deterministic price, update only last row
    new_row = df.iloc[[-1]].copy()
    new_row["close"] = new_row["close"] + 5.0  # push price
    new_row["open"] = new_row["close"]
    new_row["high"] = new_row["close"] + 0.3
    new_row["low"] = new_row["close"] - 0.3
    new_row["date"] = new_row["date"] + pd.Timedelta(minutes=1)
    df2 = pd.concat([df, new_row], ignore_index=True)

    ema_last_row(df2, span=span, prefix="ema")
    # Closed-form O(1) update: ema_t = ema_{t-1} + alpha*(price - ema_{t-1})
    alpha = 2.0 / (span + 1.0)
    expected = prev + alpha * (float(df2["close"].iloc[-1]) - prev)
    np.testing.assert_allclose(df2[f"ema_{span}"].iloc[-1], expected, rtol=1e-12, atol=1e-12)

def test_ema_supports_custom_prefix():
    df = make_df(25)
    span = 7
    ema_full(df, span=span, prefix="myema")
    assert f"myema_{span}" in df.columns
    ema_last_row(df, span=span, prefix="myema")  # should not error

def test_ema_handles_empty_and_tiny_frames():
    # Empty
    df0 = pd.DataFrame(columns=["close"])
    ema_full(df0, span=5, prefix="ema")
    ema_last_row(df0, span=5, prefix="ema")
    # Tiny (n < span)
    df1 = make_df(3)
    ema_full(df1, span=5, prefix="ema")
    ema_last_row(df1, span=5, prefix="ema")
    # Ensure column exists and last value is finite if span >= 1 and n >= 1
    assert f"ema_5" in df1.columns
    assert np.isfinite(df1["ema_5"].iloc[-1])

