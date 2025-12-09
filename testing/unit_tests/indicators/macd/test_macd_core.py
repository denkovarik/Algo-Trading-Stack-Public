# testing/unit_tests/indicators/macd/test_macd_core.py
import numpy as np
import pandas as pd

from classes.indicators.macd import (
    macd_full, macd_last_row, macd_at_index
)

def make_df(n=150, seed=0, start=100.0):
    rng = np.random.default_rng(seed)
    close = pd.Series(np.cumsum(rng.normal(0, 1.0, n)) + start, name="close")
    ts = pd.date_range("2025-01-01 09:30:00", periods=n, freq="min", tz="UTC")
    # MACD doesn't use H/L, but include typical OHLC shape
    return pd.DataFrame({
        "date": ts, "open": close.shift(1).fillna(close),
        "high": close + 0.4, "low": close - 0.4, "close": close
    })

def panda_macd(close: pd.Series, fast: int, slow: int, signal: int):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    line = ef - es
    sig  = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def test_macd_full_matches_pandas_ewm():
    df = make_df(240)
    for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (8, 17, 5)]:
        macd_full(df, fast=fast, slow=slow, signal=signal, prefix="macd")
        line, sig, hist = panda_macd(df["close"], fast, slow, signal)
        np.testing.assert_allclose(df["macd_line"].to_numpy(),   line.to_numpy(),   equal_nan=True)
        np.testing.assert_allclose(df["macd_signal"].to_numpy(), sig.to_numpy(),    equal_nan=True)
        np.testing.assert_allclose(df["macd_hist"].to_numpy(),   (line - sig).to_numpy(), equal_nan=True)

def test_macd_last_row_equals_full_at_tail():
    # Seed the caches with a full compute up to n-1; then append and last-row update
    df_full = make_df(200)
    fast, slow, signal = 12, 26, 9
    macd_full(df_full, fast=fast, slow=slow, signal=signal, prefix="macd")
    last_line  = df_full["macd_line"].iloc[-1]
    last_sig   = df_full["macd_signal"].iloc[-1]
    last_hist  = df_full["macd_hist"].iloc[-1]

    # Incremental path
    df_inc = df_full.iloc[:-1].copy()
    macd_full(df_inc, fast=fast, slow=slow, signal=signal, prefix="macd")  # seed caches up to n-1
    df_inc = pd.concat([df_inc, df_full.iloc[[-1]]], ignore_index=True)
    macd_last_row(df_inc, fast=fast, slow=slow, signal=signal, prefix="macd")

    np.testing.assert_allclose(df_inc["macd_line"].iloc[-1],   last_line, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(df_inc["macd_signal"].iloc[-1], last_sig,  rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(df_inc["macd_hist"].iloc[-1],   last_hist, rtol=1e-12, atol=1e-12)

def test_macd_at_index_no_lookahead_parity():
    df = make_df(180)
    fast, slow, signal = 12, 26, 9
    macd_full(df, fast=fast, slow=slow, signal=signal, prefix="macd")
    ref_line  = df["macd_line"].copy()
    ref_sig   = df["macd_signal"].copy()
    ref_hist  = df["macd_hist"].copy()

    # Random interior indices to check prefix-only recompute parity
    rng = np.random.default_rng(123)
    for idx in rng.choice(np.arange(30, len(df)-5), size=5, replace=False):
        df2 = df.drop(columns=["macd_line", "macd_signal", "macd_hist",
                               f"_tmp_ema_fast_{fast}", f"_tmp_ema_slow_{slow}"], errors="ignore").copy()
        macd_at_index(df2, idx, fast=fast, slow=slow, signal=signal, prefix="macd")

        np.testing.assert_allclose(df2.loc[idx, "macd_line"],   ref_line.loc[idx],  rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(df2.loc[idx, "macd_signal"], ref_sig.loc[idx],   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(df2.loc[idx, "macd_hist"],   ref_hist.loc[idx],  rtol=1e-12, atol=1e-12)

def test_macd_last_row_seeds_when_caches_missing():
    # When caches are missing, last_row should seed from short lookback and still be close to full
    df = make_df(120)
    fast, slow, signal = 12, 26, 9

    # Full baseline
    df_full = df.copy()
    macd_full(df_full, fast=fast, slow=slow, signal=signal, prefix="macd")
    last_full = df_full[["macd_line", "macd_signal", "macd_hist"]].iloc[-1].to_numpy()

    # last_row with no prior caches in df2
    df2 = df.copy()
    macd_last_row(df2, fast=fast, slow=slow, signal=signal, prefix="macd")
    last_inc = df2[["macd_line", "macd_signal", "macd_hist"]].iloc[-1].to_numpy()

    # Not exact (seeded), but should be very close
    np.testing.assert_allclose(last_inc, last_full, rtol=1e-6, atol=1e-6)

def test_macd_supports_custom_prefix():
    df = make_df(60)
    macd_full(df, fast=8, slow=17, signal=5, prefix="m")
    assert {"m_line", "m_signal", "m_hist"} <= set(df.columns)
    # last-row should update without error
    macd_last_row(df, fast=8, slow=17, signal=5, prefix="m")

def test_macd_handles_empty_and_tiny_frames():
    # Empty
    df0 = pd.DataFrame(columns=["close"])
    macd_full(df0, prefix="macd")
    macd_last_row(df0, prefix="macd")
    macd_at_index(df0, 0, prefix="macd")

    # Tiny (n < slow)
    df1 = make_df(5)
    macd_full(df1, prefix="macd")
    macd_last_row(df1, prefix="macd")
    macd_at_index(df1, len(df1)-1, prefix="macd")
    # Columns exist
    assert {"macd_line", "macd_signal", "macd_hist"} <= set(df1.columns)

