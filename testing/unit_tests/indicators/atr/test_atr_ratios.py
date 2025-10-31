# testing/unit_tests/indicators/atr/test_atr_ratios.py
import numpy as np
import pandas as pd

from classes.indicators.atr import (
    _tr_series,
    atr_ratio, atr_ratio_last_row, atr_ratio_at_index,
    atr_ratio_sm, atr_ratio_sm_last_row, atr_ratio_sm_at_index
)

def make_df(n=80, seed=123):
    rng = np.random.default_rng(seed)
    close = pd.Series(np.cumsum(rng.normal(0, 1.2, n)) + 50.0)
    high  = close + rng.uniform(0.2, 0.7, n)
    low   = close - rng.uniform(0.2, 0.7, n)
    # NOTE: 'T' deprecated; use 'min'
    ts = pd.date_range("2025-02-01 09:30:00", periods=n, freq="min", tz="UTC")
    return pd.DataFrame({"date": ts, "high": high, "low": low, "close": close})

def manual_ratio(df, fast, slow):
    tr = _tr_series(df)
    a_f = tr.rolling(int(fast)).mean()
    a_s = tr.rolling(int(slow)).mean().replace(0, np.nan)
    return a_f / a_s

def test_atr_ratio_vectorized_matches_manual():
    df = make_df(120)
    fast, slow = 9, 21
    atr_ratio(df, fast=fast, slow=slow, prefix="atrR")
    expected = manual_ratio(df, fast, slow)
    col = f"atrR_{fast}_{slow}"
    pd.testing.assert_series_equal(
        df[col].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False
    )

def test_atr_ratio_last_row_equals_full():
    df_full = make_df(150)
    fast, slow = 14, 60
    atr_ratio(df_full, fast=fast, slow=slow, prefix="atrR")
    last_full = df_full[f"atrR_{fast}_{slow}"].iloc[-1]

    df_inc = df_full.iloc[:-1].copy()
    atr_ratio(df_inc, fast=fast, slow=slow, prefix="atrR")
    df_inc = pd.concat([df_inc, df_full.iloc[[-1]]], ignore_index=True)
    atr_ratio_last_row(df_inc, fast=fast, slow=slow, prefix="atrR")
    last_inc = df_inc[f"atrR_{fast}_{slow}"].iloc[-1]

    assert np.isfinite(last_full) and np.isfinite(last_inc)
    np.testing.assert_allclose(last_inc, last_full, rtol=1e-12, atol=1e-12)

def test_atr_ratio_at_index_no_lookahead():
    df = make_df(90)
    fast, slow = 7, 20
    atr_ratio(df, fast=fast, slow=slow, prefix="atrR")  # reference
    idx = 40
    df2 = df.copy()
    df2.drop(columns=[f"atrR_{fast}_{slow}"], inplace=True, errors="ignore")
    atr_ratio_at_index(df2, idx, fast=fast, slow=slow, prefix="atrR")
    exp = df.loc[idx, f"atrR_{fast}_{slow}"]
    got = df2.loc[idx, f"atrR_{fast}_{slow}"]
    np.testing.assert_allclose(got, exp, rtol=1e-12, atol=1e-12)

def test_atr_ratio_sm_variants_behave_like_ratio():
    df = make_df(120)
    fast, slow = 14, 60
    atr_ratio_sm(df, fast=fast, slow=slow, prefix="atrRsm")
    expected = manual_ratio(df, fast, slow)
    col = f"atrRsm_{fast}_{slow}"
    pd.testing.assert_series_equal(
        df[col].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False
    )

def test_atr_ratio_sm_last_row_and_at_index():
    df = make_df(120)
    fast, slow = 14, 60
    # Full
    atr_ratio_sm(df, fast=fast, slow=slow, prefix="atrRsm")
    last_full = df[f"atrRsm_{fast}_{slow}"].iloc[-1]

    # last-row
    df_inc = df.iloc[:-1].copy()
    atr_ratio_sm(df_inc, fast=fast, slow=slow, prefix="atrRsm")
    df_inc = pd.concat([df_inc, df.iloc[[-1]]], ignore_index=True)
    atr_ratio_sm_last_row(df_inc, fast=fast, slow=slow, prefix="atrRsm")
    last_inc = df_inc[f"atrRsm_{fast}_{slow}"].iloc[-1]
    np.testing.assert_allclose(last_inc, last_full, rtol=1e-12, atol=1e-12)

    # at-index
    df2 = df.copy()
    col = f"atrRsm_{fast}_{slow}"
    df2.drop(columns=[col], inplace=True, errors="ignore")
    idx = len(df2) - 5
    atr_ratio_sm_at_index(df2, idx, fast=fast, slow=slow, prefix="atrRsm")
    np.testing.assert_allclose(df2.loc[idx, col], df.loc[idx, col], rtol=1e-12, atol=1e-12)

# -----------------------------
# Extra high-value tests
# -----------------------------

def test_atr_ratio_warmup_nan_boundaries():
    # The ratio is defined only once BOTH rolling windows have values.
    # So warmup NaNs should be max(fast, slow) - 1.
    df = make_df(100)
    fast, slow = 8, 21
    col = f"atrR_{fast}_{slow}"
    atr_ratio(df, fast=fast, slow=slow, prefix="atrR")
    warmup_nans = df[col].isna().sum()
    assert warmup_nans == max(fast, slow) - 1
    assert df[col].iloc[max(fast, slow)-1:].notna().all()

def test_atr_ratio_all_synthetic_bars_ok():
    # All H/L NaN → TR falls back to close-based deltas; ratio still defined post warmup.
    n = 120
    close = pd.Series(np.linspace(100, 130, n))
    df = pd.DataFrame({"close": close, "high": np.nan, "low": np.nan})
    fast, slow = 9, 21
    atr_ratio(df, fast=fast, slow=slow, prefix="atrR")
    col = f"atrR_{fast}_{slow}"
    # Warmup only NaNs; plenty of finite values afterwards.
    assert df[col].isna().sum() == max(fast, slow) - 1
    assert df[col].iloc[max(fast, slow)-1:].notna().all()
    # Parity with manual
    expected = manual_ratio(df, fast, slow)
    pd.testing.assert_series_equal(
        df[col].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False
    )

def test_atr_ratio_idempotent_multiple_calls():
    df = make_df(140)
    fast, slow = 10, 30
    col = f"atrR_{fast}_{slow}"
    atr_ratio(df, fast=fast, slow=slow, prefix="atrR")
    first = df[col].copy()
    # Re-run should not change values
    atr_ratio(df, fast=fast, slow=slow, prefix="atrR")
    pd.testing.assert_series_equal(first, df[col])

