# testing/unit_tests/indicators/roc/test_roc_core.py
import numpy as np
import pandas as pd

from classes.indicators.roc import roc_full, roc_last_row, roc_at_index

def make_df(n=100, seed=0, start=100.0):
    rng = np.random.default_rng(seed)
    close = pd.Series(np.cumsum(rng.normal(0, 1, n)) + start, name="close")
    ts = pd.date_range("2025-01-01 09:30:00", periods=n, freq="min", tz="UTC")
    return pd.DataFrame({
        "date": ts,
        "open": close.shift(1).fillna(close),
        "high": close + 0.4,
        "low": close - 0.4,
        "close": close
    })

def panda_roc(close: pd.Series, window: int):
    prev = close.shift(window).replace(0, np.nan)
    return (close / prev) - 1.0

def test_roc_full_matches_pandas():
    df = make_df(150)
    for w in [1, 2, 5, 14, 30]:
        roc_full(df, window=w, prefix="roc")
        expected = panda_roc(df["close"], w)
        pd.testing.assert_series_equal(
            df[f"roc_{w}"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

def test_roc_last_row_equals_full_at_tail():
    df = make_df(120)
    w = 14
    # Full baseline
    df_full = df.copy()
    roc_full(df_full, window=w, prefix="roc")
    last_full = df_full[f"roc_{w}"].iloc[-1]

    # Incremental: compute up to n-1, append last bar, then last-row update
    df_inc = df.iloc[:-1].copy()
    roc_full(df_inc, window=w, prefix="roc")
    df_inc = pd.concat([df_inc, df.iloc[[-1]]], ignore_index=True)
    roc_last_row(df_inc, window=w, prefix="roc")
    last_inc = df_inc[f"roc_{w}"].iloc[-1]

    np.testing.assert_allclose(last_inc, last_full, rtol=1e-12, atol=1e-12)

def test_roc_at_index_no_lookahead_parity():
    df = make_df(110)
    w = 20
    roc_full(df, window=w, prefix="roc")
    ref = df[f"roc_{w}"].copy()
    for idx in [25, 50, 80, 100]:
        df2 = df.drop(columns=[f"roc_{w}"], errors="ignore").copy()
        roc_at_index(df2, idx, window=w, prefix="roc")
        np.testing.assert_allclose(df2.loc[idx, f"roc_{w}"], ref.loc[idx], rtol=1e-12, atol=1e-12)

def test_roc_handles_short_series_and_zero_prev():
    # Short series (n <= window) -> NaNs (consistent with shift behavior)
    df = make_df(5)
    w = 10
    roc_full(df, window=w, prefix="roc")
    assert df[f"roc_{w}"].isna().all()

    # Explicit zero previous value case
    df2 = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=6, freq="min", tz="UTC"),
        "open":  [1,1,1,1,1,1],
        "high":  [1,1,1,1,1,1],
        "low":   [1,1,1,1,1,1],
        "close": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    })
    w2 = 1
    roc_full(df2, window=w2, prefix="roc")
    # At index 1, prev close is 0 -> NaN
    assert np.isnan(df2.loc[1, f"roc_{w2}"])
    # At index 2, prev=1 -> finite
    assert np.isfinite(df2.loc[2, f"roc_{w2}"])

def test_roc_custom_prefix_and_last_row_seed():
    df = make_df(40)
    w = 7
    roc_full(df, window=w, prefix="myroc")
    assert f"myroc_{w}" in df.columns

    # last_row with enough history should not error and match full at tail
    df_full = df.copy()
    roc_full(df_full, window=w, prefix="myroc")
    last_full = df_full[f"myroc_{w}"].iloc[-1]

    df_inc = df.iloc[:-1].copy()
    roc_full(df_inc, window=w, prefix="myroc")
    df_inc = pd.concat([df_inc, df.iloc[[-1]]], ignore_index=True)
    roc_last_row(df_inc, window=w, prefix="myroc")
    last_inc = df_inc[f"myroc_{w}"].iloc[-1]
    np.testing.assert_allclose(last_inc, last_full, rtol=1e-12, atol=1e-12)

