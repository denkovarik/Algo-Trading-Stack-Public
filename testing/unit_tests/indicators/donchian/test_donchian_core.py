import numpy as np, pandas as pd
from classes.indicators.donchian import (
    donchian_full, donchian_last_row, donchian_at_index,
    donchian_pos_full, donchian_pos_last_row, donchian_pos_at_index,
)

def make_df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    close = pd.Series(np.cumsum(rng.normal(0,1,n)) + 100, name="close")
    high  = close + rng.uniform(0.2, 0.6, n)
    low   = close - rng.uniform(0.2, 0.6, n)
    ts = pd.date_range("2025-01-01 09:30:00", periods=n, freq="min", tz="UTC")
    return pd.DataFrame({"date": ts, "open": close, "high": high, "low": low, "close": close})

def test_donchian_full_matches_manual():
    df = make_df(150); w=60; p="dc60"
    donchian_full(df, window=w, prefix=p)
    hi = df["high"].rolling(w).max()
    lo = df["low" ].rolling(w).min()
    mid = (hi + lo)/2.0
    width = (hi - lo) / mid.replace(0, np.nan)
    np.testing.assert_allclose(df[f"{p}_high"].to_numpy(),  hi.to_numpy(),  equal_nan=True)
    np.testing.assert_allclose(df[f"{p}_low"].to_numpy(),   lo.to_numpy(),  equal_nan=True)
    np.testing.assert_allclose(df[f"{p}_mid"].to_numpy(),   mid.to_numpy(), equal_nan=True)
    np.testing.assert_allclose(df[f"{p}_width"].to_numpy(), width.to_numpy(), equal_nan=True)

def test_donchian_last_row_equals_full_at_tail():
    df = make_df(100); w=60; p="dc60"
    ref = df.copy(); donchian_full(ref, window=w, prefix=p)
    last = (ref[f"{p}_high"].iloc[-1], ref[f"{p}_low"].iloc[-1], ref[f"{p}_mid"].iloc[-1], ref[f"{p}_width"].iloc[-1])

    inc = df.iloc[:-1].copy(); donchian_full(inc, window=w, prefix=p)
    inc = pd.concat([inc, df.iloc[[-1]]], ignore_index=True)
    donchian_last_row(inc, window=w, prefix=p)

    np.testing.assert_allclose(inc[f"{p}_high"].iloc[-1],  last[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(inc[f"{p}_low"].iloc[-1],   last[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(inc[f"{p}_mid"].iloc[-1],   last[2], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(inc[f"{p}_width"].iloc[-1], last[3], rtol=1e-12, atol=1e-12)

def test_donchian_at_index_no_lookahead_parity():
    df = make_df(110); w=60; p="dc60"
    donchian_full(df, window=w, prefix=p)
    for idx in [70, 80, 90, 100]:
        df2 = df.drop(columns=[f"{p}_high", f"{p}_low", f"{p}_mid", f"{p}_width"]).copy()
        donchian_at_index(df2, idx, window=w, prefix=p)
        for col in ["high","low","mid","width"]:
            np.testing.assert_allclose(df2.loc[idx, f"{p}_{col}"], df.loc[idx, f"{p}_{col}"], rtol=1e-12, atol=1e-12)

def test_donchian_pos_variants_parity():
    df = make_df(100); w=60; p="dc60"
    donchian_full(df, window=w, prefix=p); donchian_pos_full(df, window=w, prefix=p)
    ref = df[f"{p}_pos"].copy()

    # last-row
    inc = df.iloc[:-1].copy()
    donchian_full(inc, window=w, prefix=p)
    inc = pd.concat([inc, df.iloc[[-1]]], ignore_index=True)
    donchian_pos_last_row(inc, window=w, prefix=p)
    np.testing.assert_allclose(inc[f"{p}_pos"].iloc[-1], ref.iloc[-1], rtol=1e-12, atol=1e-12)

    # at-index
    df2 = df.drop(columns=[f"{p}_pos"]).copy()
    donchian_pos_at_index(df2, 80, window=w, prefix=p)
    np.testing.assert_allclose(df2.loc[80, f"{p}_pos"], ref.loc[80], rtol=1e-12, atol=1e-12)

