# testing/unit_tests/indicators/atr/test_atr_core.py
import numpy as np
import pandas as pd
import pytest

from classes.indicators.atr import (
    _tr_series, ensure_tr, atr, atr_last_row, atr_at_index
)

def make_df(n=20, seed=0):
    rng = np.random.default_rng(seed)
    close = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100.0)
    high  = close + rng.uniform(0.1, 0.5, n)
    low   = close - rng.uniform(0.1, 0.5, n)
    open_ = close.shift(1).fillna(close)  # not used by ATR but typical OHLC
    # NOTE: 'T' deprecated; use 'min'
    ts = pd.date_range("2025-01-01 09:30:00", periods=n, freq="min", tz="UTC")
    df = pd.DataFrame({"date": ts, "open": open_, "high": high, "low": low, "close": close})
    return df

def test_tr_series_matches_max_of_three_forms():
    df = make_df(50)
    tr = _tr_series(df)
    # Manual check: TR >= (high-low), (high-prev_close), (low-prev_close) element-wise
    high = df["high"].fillna(df["close"])
    low  = df["low"].fillna(df["close"])
    close = df["close"]
    prev_close = close.shift(1).fillna(close)
    a = (high - low).abs()
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    manual = pd.concat([a, b, c], axis=1).max(axis=1).astype("float64")
    pd.testing.assert_series_equal(tr.reset_index(drop=True), manual.reset_index(drop=True), check_names=False)

def test_ensure_tr_adds_tr_when_missing():
    df = make_df(30)
    assert "tr" not in df.columns
    ensure_tr(df)
    assert "tr" in df.columns
    assert df["tr"].notna().sum() >= 1

@pytest.mark.parametrize("window", [1, 2, 5, 14])
def test_atr_vectorized_window(window):
    df = make_df(40)
    atr(df, window=window, prefix="atr")
    assert f"atr_{window}" in df.columns
    # Expected: rolling mean of TR
    tr = _tr_series(df)
    expected = tr.rolling(window).mean()
    pd.testing.assert_series_equal(
        df[f"atr_{window}"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False
    )

def test_atr_handles_nan_high_low_as_close():
    df = make_df(25)
    # Introduce synthetic bars with NaN H/L
    df.loc[5:7, "high"] = np.nan
    df.loc[5:7, "low"]  = np.nan
    atr(df, window=5, prefix="atr")
    # No NaNs forced by our substitution in TR at those rows beyond the initial warmup
    assert df.loc[7:, "atr_5"].isna().sum() < 3  # allow early warmup NaNs, but not all

def test_atr_last_row_equals_full_vectorized():
    # Build df, compute full ATR, then simulate last-row path
    df_full = make_df(60)
    atr(df_full, window=14, prefix="atr")
    last_full = df_full["atr_14"].iloc[-1]

    df_inc = df_full.iloc[:-1].copy()
    atr(df_inc, window=14, prefix="atr")     # compute up to n-1
    # Append the last row (simulate a new bar), then update only last row
    df_inc = pd.concat([df_inc, df_full.iloc[[-1]]], ignore_index=True)
    atr_last_row(df_inc, window=14, prefix="atr")
    last_inc = df_inc["atr_14"].iloc[-1]

    assert np.isfinite(last_full) and np.isfinite(last_inc)
    np.testing.assert_allclose(last_inc, last_full, rtol=1e-12, atol=1e-12)

def test_atr_at_index_uses_past_only_no_lookahead():
    df = make_df(30)
    window = 10
    atr(df, window=window, prefix="atr")  # reference

    # Pick an interior index and recompute via at_index (using only past bars)
    idx = 20
    df2 = df.copy()
    df2.drop(columns=[f"atr_{window}"], inplace=True, errors="ignore")
    atr_at_index(df2, idx, window=window, prefix="atr")

    exp = df.loc[idx, f"atr_{window}"]
    got = df2.loc[idx, f"atr_{window}"]
    assert np.isfinite(exp) and np.isfinite(got)
    np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-12)

# -----------------------------
# Extra high-value tests
# -----------------------------

def test_atr_warmup_nan_window_boundary():
    n, win = 10, 5
    df = pd.DataFrame({
        "close": np.arange(n, dtype=float) + 100,
        "high":  np.arange(n, dtype=float) + 100.5,
        "low":   np.arange(n, dtype=float) + 99.5,
    })
    atr(df, window=win, prefix="atr")
    col = f"atr_{win}"
    # Exactly win-1 NaNs, then finite values
    assert df[col].isna().sum() == win - 1
    assert df[col].iloc[win-1:].notna().all()

def test_atr_all_synthetic_bars_ok():
    n = 20
    close = pd.Series(np.linspace(100, 110, n))
    df = pd.DataFrame({"close": close, "high": np.nan, "low": np.nan})
    atr(df, window=5, prefix="atr")
    # TR should be fully populated; ATR has warmup NaNs only
    assert df["tr"].notna().sum() == n
    assert df["atr_5"].isna().sum() == 4  # warmup only

def test_atr_idempotent_multiple_calls():
    n = 50
    rng = np.random.default_rng(0)
    close = np.cumsum(rng.normal(0, 1, n)) + 100
    df = pd.DataFrame({
        "close": close,
        "high":  close + 0.3,
        "low":   close - 0.3,
    })
    atr(df, window=14, prefix="atr")
    first = df["atr_14"].copy()
    # Re-run should not change values
    atr(df, window=14, prefix="atr")
    pd.testing.assert_series_equal(first, df["atr_14"])

