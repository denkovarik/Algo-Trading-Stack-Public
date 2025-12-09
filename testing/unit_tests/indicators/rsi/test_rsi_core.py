import numpy as np, pandas as pd
from classes.indicators.rsi import rsi_full, rsi_last_row, rsi_at_index

def make_df(n=80, seed=0):
    rng = np.random.default_rng(seed)
    close = pd.Series(np.cumsum(rng.normal(0,1,n))+100, name="close")
    ts = pd.date_range("2025-01-01 09:30:00", periods=n, freq="min", tz="UTC")
    return pd.DataFrame({"date": ts, "open": close, "high": close+0.4, "low": close-0.4, "close": close})

def test_rsi_full_monotonic_up_is_100():
    df = make_df(40)
    df["close"] = np.linspace(100, 120, len(df))
    rsi_full(df, window=5, prefix="rsi")
    assert np.isclose(df["rsi_5"].iloc[-1], 100.0, equal_nan=False)

def test_rsi_full_monotonic_down_is_0():
    df = make_df(40)
    df["close"] = np.linspace(120, 100, len(df))
    rsi_full(df, window=5, prefix="rsi")
    assert np.isclose(df["rsi_5"].iloc[-1], 0.0, equal_nan=False)

def test_rsi_full_flat_is_50():
    df = make_df(40)
    df["close"] = 100.0
    rsi_full(df, window=5, prefix="rsi")
    assert np.isclose(df["rsi_5"].iloc[-1], 50.0, equal_nan=False)

def test_rsi_last_row_equals_full_at_tail():
    df = make_df(80)
    w = 14
    df_full = df.copy()
    rsi_full(df_full, window=w, prefix="rsi")
    last_full = df_full[f"rsi_{w}"].iloc[-1]

    df_inc = df.iloc[:-1].copy()
    rsi_full(df_inc, window=w, prefix="rsi")
    df_inc = pd.concat([df_inc, df.iloc[[-1]]], ignore_index=True)
    rsi_last_row(df_inc, window=w, prefix="rsi")
    last_inc = df_inc[f"rsi_{w}"].iloc[-1]

    np.testing.assert_allclose(last_inc, last_full, rtol=1e-12, atol=1e-12)

def test_rsi_at_index_no_lookahead_parity():
    df = make_df(90)
    w = 10
    rsi_full(df, window=w, prefix="rsi")
    ref = df[f"rsi_{w}"].copy()
    for idx in [15, 30, 60, 80]:
        df2 = df.drop(columns=[f"rsi_{w}"]).copy()
        rsi_at_index(df2, idx, window=w, prefix="rsi")
        np.testing.assert_allclose(df2.loc[idx, f"rsi_{w}"], ref.loc[idx], rtol=1e-12, atol=1e-12)

