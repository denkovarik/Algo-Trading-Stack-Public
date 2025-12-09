import numpy as np, pandas as pd
from classes.indicators.bollinger import (
    bollinger_full, bollinger_last_row, bollinger_at_index,
    bandwidth_full, bandwidth_last_row, bandwidth_at_index
)

def make_df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    close = pd.Series(np.cumsum(rng.normal(0,1,n)) + 100, name="close")
    ts = pd.date_range("2025-01-01 09:30:00", periods=n, freq="min", tz="UTC")
    return pd.DataFrame({"date": ts, "open": close, "high": close+0.4, "low": close-0.4, "close": close})

def test_bollinger_full_matches_manual():
    df = make_df()
    w, s, p = 20, 2.0, "bb"
    bollinger_full(df, window=w, num_std=s, prefix=p)
    ma = df["close"].rolling(w).mean()
    sd = df["close"].rolling(w).std()
    np.testing.assert_allclose(df[f"{p}_ma"].to_numpy(), ma.to_numpy(), equal_nan=True)
    np.testing.assert_allclose(df[f"{p}_upper"].to_numpy(), (ma + s*sd).to_numpy(), equal_nan=True)
    np.testing.assert_allclose(df[f"{p}_lower"].to_numpy(), (ma - s*sd).to_numpy(), equal_nan=True)

def test_bollinger_last_row_equals_full_at_tail():
    df = make_df(80)
    w, s, p = 20, 2.0, "bb"
    df_full = df.copy()
    bollinger_full(df_full, window=w, num_std=s, prefix=p)
    last_ma, last_up, last_lo = df_full[f"{p}_ma"].iloc[-1], df_full[f"{p}_upper"].iloc[-1], df_full[f"{p}_lower"].iloc[-1]
    df_inc = df.iloc[:-1].copy()
    bollinger_full(df_inc, window=w, num_std=s, prefix=p)
    df_inc = pd.concat([df_inc, df.iloc[[-1]]], ignore_index=True)
    bollinger_last_row(df_inc, window=w, num_std=s, prefix=p)
    np.testing.assert_allclose(df_inc[f"{p}_ma"].iloc[-1],    last_ma, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(df_inc[f"{p}_upper"].iloc[-1], last_up, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(df_inc[f"{p}_lower"].iloc[-1], last_lo, rtol=1e-12, atol=1e-12)

def test_bollinger_at_index_no_lookahead_parity():
    df = make_df(90)
    w, s, p = 20, 2.0, "bb"
    bollinger_full(df, window=w, num_std=s, prefix=p)
    ref_ma  = df[f"{p}_ma"].copy()
    ref_up  = df[f"{p}_upper"].copy()
    ref_low = df[f"{p}_lower"].copy()
    for idx in [30, 40, 50, 60]:
        df2 = df.drop(columns=[f"{p}_ma", f"{p}_upper", f"{p}_lower"]).copy()
        bollinger_at_index(df2, idx, window=w, num_std=s, prefix=p)
        np.testing.assert_allclose(df2.loc[idx, f"{p}_ma"],    ref_ma.loc[idx],  rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(df2.loc[idx, f"{p}_upper"], ref_up.loc[idx],  rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(df2.loc[idx, f"{p}_lower"], ref_low.loc[idx], rtol=1e-12, atol=1e-12)

def test_bandwidth_variants_parity():
    df = make_df(100)
    p = "bb60"; w = 60
    bollinger_full(df, window=w, num_std=2.0, prefix=p)
    bandwidth_full(df, window=w, prefix=p)
    bw_full = df[f"{p}_bw"].copy()
    # last-row
    df_inc = df.iloc[:-1].copy()
    bollinger_full(df_inc, window=w, num_std=2.0, prefix=p)
    df_inc = pd.concat([df_inc, df.iloc[[-1]]], ignore_index=True)
    bandwidth_last_row(df_inc, window=w, prefix=p)
    np.testing.assert_allclose(df_inc[f"{p}_bw"].iloc[-1], bw_full.iloc[-1], rtol=1e-12, atol=1e-12)
    # at-index
    df2 = df.drop(columns=[f"{p}_bw"]).copy()
    bandwidth_at_index(df2, 70, window=w, prefix=p)
    np.testing.assert_allclose(df2.loc[70, f"{p}_bw"], bw_full.loc[70], rtol=1e-12, atol=1e-12)

