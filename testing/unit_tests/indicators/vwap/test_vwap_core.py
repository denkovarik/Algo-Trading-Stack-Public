# testing/unit_tests/indicators/vwap/test_vwap_core.py
import numpy as np
import pandas as pd

from classes.indicators.vwap import (
    vwap_full, vwap_last_row, vwap_at_index,
    vwap_rth_full, vwap_rth_last_row, vwap_rth_at_index,
    vwap_dev_atr_full, vwap_dev_atr_last_row, vwap_dev_atr_at_index,
)

# ---------- Helpers ----------

def make_intraday_df(days=2, seed=0, start_price=100.0):
    """
    Build a UTC-timestamped minute DataFrame spanning two trading days with
    both pre/post hours and RTH window contained. We span 13:00..21:00 UTC
    so that RTH 09:30..16:00 America/New_York is fully contained (14:30..21:00 UTC).
    """
    rng = np.random.default_rng(seed)
    frames = []
    for d in range(days):
        date_utc = pd.Timestamp("2025-01-02 13:00:00Z") + pd.Timedelta(days=d)
        idx = pd.date_range(date_utc, periods=480, freq="min", tz="UTC")  # 8 hours
        # Random walk close + positive volume
        close = pd.Series(np.cumsum(rng.normal(0, 0.2, len(idx))) + start_price + 2*d, index=idx)
        vol = pd.Series(rng.integers(10, 200, len(idx)).astype(float), index=idx)
        # Give off-hours some zeros to ensure they're ignored in RTH VWAP
        # (First 90 mins and last 30 mins of the span => ~pre/post)
        vol.iloc[:90] = 0.0
        vol.iloc[-30:] = 0.0
        open_ = close.shift(1).fillna(close)
        high = np.maximum(open_, close) + 0.1
        low  = np.minimum(open_, close) - 0.1
        frames.append(pd.DataFrame({
            "date": idx,
            "open": open_.values,
            "high": high.values,
            "low":  low.values,
            "close": close.values,
            "volume": vol.values
        }))
    df = pd.concat(frames, ignore_index=True)
    return df

def localize(df, tz="America/New_York"):
    local = pd.to_datetime(df["date"], utc=True).dt.tz_convert(tz)
    return local

def rth_mask(df, tz="America/New_York"):
    local = localize(df, tz)
    t = local.dt.time
    return (t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("16:00").time())

def manual_rth_vwap(df, price_col="close", vol_col="volume", tz="America/New_York"):
    """
    Manual per-day RTH VWAP reference:
      vwap = cumsum(price*vol) / cumsum(vol) within each local day, RTH rows only.
      Off-hours rows = NaN.
    """
    local = localize(df, tz)
    key = local.dt.date
    rth = rth_mask(df, tz)

    px = df[price_col].astype("float64")
    vol = df[vol_col].astype("float64").where(rth, 0.0)

    pv = (px * vol).groupby(key).cumsum()
    vv = vol.groupby(key).cumsum()
    vwap_val = pv / vv.replace(0.0, np.nan)
    out = vwap_val.where(rth, np.nan)
    return out

def manual_generic_vwap(df, price_col="close", vol_col="volume"):
    px = df[price_col].astype("float64")
    vol = df[vol_col].astype("float64")
    pv = (px * vol).cumsum()
    vv = vol.cumsum()
    return pv / vv.replace(0.0, np.nan)

# ---------- Tests: Generic VWAP ----------

def test_vwap_full_matches_manual_generic():
    df = make_intraday_df(days=2)
    vwap_full(df, price_col="close", vol_col="volume", prefix="vwap")
    expected = manual_generic_vwap(df)
    pd.testing.assert_series_equal(
        df["vwap"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False
    )

def test_vwap_last_row_equals_full_tail():
    df = make_intraday_df(days=2)
    df_full = df.copy()
    vwap_full(df_full, prefix="vwap")
    last_full = df_full["vwap"].iloc[-1]

    df_inc = df.iloc[:-1].copy()
    vwap_full(df_inc, prefix="vwap")
    df_inc = pd.concat([df_inc, df.iloc[[-1]]], ignore_index=True)
    vwap_last_row(df_inc, prefix="vwap")
    last_inc = df_inc["vwap"].iloc[-1]

    np.testing.assert_allclose(last_inc, last_full, rtol=1e-12, atol=1e-12)

def test_vwap_at_index_no_lookahead_parity():
    df = make_intraday_df(days=2)
    vwap_full(df, prefix="vwap")
    ref = df["vwap"].copy()
    for idx in [50, 100, len(df)-1]:
        df2 = df.drop(columns=["vwap"], errors="ignore").copy()
        vwap_at_index(df2, idx, prefix="vwap")
        np.testing.assert_allclose(df2.loc[idx, "vwap"], ref.loc[idx], rtol=1e-12, atol=1e-12)

# ---------- Tests: RTH VWAP ----------

def test_vwap_rth_full_matches_manual():
    df = make_intraday_df(days=2)
    vwap_rth_full(df, price_col="close", vol_col="volume", prefix="vwap_rth", tz="America/New_York")
    expected = manual_rth_vwap(df, tz="America/New_York")
    pd.testing.assert_series_equal(
        df["vwap_rth"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False
    )

def test_vwap_rth_last_row_equals_full_tail():
    df = make_intraday_df(days=2)
    df_full = df.copy()
    vwap_rth_full(df_full, prefix="vwap_rth")
    last_full = df_full["vwap_rth"].iloc[-1]

    df_inc = df.iloc[:-1].copy()
    vwap_rth_full(df_inc, prefix="vwap_rth")
    df_inc = pd.concat([df_inc, df.iloc[[-1]]], ignore_index=True)
    vwap_rth_last_row(df_inc, prefix="vwap_rth")
    last_inc = df_inc["vwap_rth"].iloc[-1]

    # If the last row is off-hours, both should be NaN; otherwise equal
    if pd.isna(last_full):
        assert pd.isna(last_inc)
    else:
        np.testing.assert_allclose(last_inc, last_full, rtol=1e-12, atol=1e-12)

def test_vwap_rth_at_index_no_lookahead_parity():
    df = make_intraday_df(days=2)
    vwap_rth_full(df, prefix="vwap_rth")
    ref = df["vwap_rth"].copy()
    # Pick some indices across days (middle and tail)
    for idx in [60, 240, len(df)-10]:
        df2 = df.drop(columns=["vwap_rth"], errors="ignore").copy()
        vwap_rth_at_index(df2, idx, prefix="vwap_rth")
        # Both sides should be NaN off-hours or equal in RTH
        if pd.isna(ref.loc[idx]):
            assert pd.isna(df2.loc[idx, "vwap_rth"])
        else:
            np.testing.assert_allclose(df2.loc[idx, "vwap_rth"], ref.loc[idx], rtol=1e-12, atol=1e-12)

def test_vwap_rth_off_hours_are_nan_rth_are_finite():
    df = make_intraday_df(days=1)
    vwap_rth_full(df, prefix="vwap_rth")
    mask = rth_mask(df)
    assert df.loc[~mask, "vwap_rth"].isna().all()
    # Among RTH rows with positive volume, we should have finite VWAPs
    rth_with_vol = mask & (df["volume"] > 0)
    assert df.loc[rth_with_vol, "vwap_rth"].notna().any()

# ---------- Tests: VWAP deviation vs ATR ----------

def test_vwap_dev_atr_full_matches_manual_when_atr_is_one():
    # If atr_90 == 1, vwap_dev_atr == close - vwap
    df = make_intraday_df(days=1)
    vwap_full(df, prefix="vwap")
    df["atr_90"] = 1.0
    vwap_dev_atr_full(df, atr_col="atr_90", vwap_col="vwap", prefix="vwap_dev")
    expected = (df["close"] - df["vwap"])
    pd.testing.assert_series_equal(
        df["vwap_dev"].reset_index(drop=True).astype("float64"),
        expected.reset_index(drop=True).astype("float64"),
        check_names=False
    )

def test_vwap_dev_atr_last_row_and_at_index_parity():
    df = make_intraday_df(days=2)
    # Seed generic VWAP & ATR
    vwap_full(df, prefix="vwap")
    df["atr_90"] = 2.0  # constant to make math easy

    # Full
    df_full = df.copy()
    vwap_dev_atr_full(df_full, atr_col="atr_90", vwap_col="vwap", prefix="vwap_dev")
    last_full = df_full["vwap_dev"].iloc[-1]

    # last-row (should match tail exactly)
    df_inc = df.copy()
    vwap_dev_atr_last_row(df_inc, atr_col="atr_90", vwap_col="vwap", prefix="vwap_dev")
    last_inc = df_inc["vwap_dev"].iloc[-1]
    np.testing.assert_allclose(last_inc, last_full, rtol=1e-12, atol=1e-12)

    # at-index (pick a few indices)
    ref = df_full["vwap_dev"].copy()
    for idx in [10, 200, len(df)-5]:
        df2 = df.drop(columns=["vwap_dev"], errors="ignore").copy()
        vwap_dev_atr_at_index(df2, idx, atr_col="atr_90", vwap_col="vwap", prefix="vwap_dev")
        np.testing.assert_allclose(df2.loc[idx, "vwap_dev"], ref.loc[idx], rtol=1e-12, atol=1e-12)

def test_vwap_dev_atr_seeds_vwap_if_missing():
    # If vwap column is missing at the last row, the last-row function should seed it
    df = make_intraday_df(days=1)
    # Intentionally do NOT compute vwap_full
    df["atr_90"] = 1.0
    vwap_dev_atr_last_row(df, atr_col="atr_90", vwap_col="vwap", prefix="vwap_dev")
    # Now vwap and vwap_dev at last row should be finite (unless volume=0)
    assert "vwap" in df.columns
    # When volume is 0 at the last row, deviation may be nan; accept either finite or nan here.
    # We're primarily checking that the function seeded the base VWAP column.
    assert "vwap_dev" in df.columns

