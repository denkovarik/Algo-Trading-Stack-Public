# classes/Trading_Environment.py
from PyQt5 import QtCore
import pandas as pd
import numpy as np
from datetime import time as _time

# ----------------------------------------------------------------------
# Indicator Imports (unchanged – keep all your existing ones)
# ----------------------------------------------------------------------
from classes.indicators.atr import (
    atr, atr_last_row, atr_at_index,
    atr_ratio, atr_ratio_last_row, atr_ratio_at_index,
    atr_ratio_sm, atr_ratio_sm_last_row, atr_ratio_sm_at_index,
    ensure_tr,
)
from classes.indicators.ema import (
    ema_full, ema_last_row, ema_at_index,
    zscore_close_vs_ema,
    zscore_ema60_vs_ema120,
    ema_slope_over,
    ema_curve_full,
    ema_curve_last_row,
    ema_curve_at_index,
)
from classes.indicators.bollinger import (
    bollinger_full, bollinger_last_row, bollinger_at_index,
    bandwidth_full, bandwidth_last_row, bandwidth_at_index,
)
from classes.indicators.rsi import (
    rsi_full, rsi_last_row, rsi_at_index,
)
from classes.indicators.donchian import (
    donchian_full, donchian_last_row, donchian_at_index,
    donchian_pos_full, donchian_pos_last_row, donchian_pos_at_index,
)
from classes.indicators.macd import (
    macd_full, macd_last_row, macd_at_index,
)
from classes.indicators.roc import (
    roc_full, roc_last_row, roc_at_index,
)
from classes.indicators.vwap import (
    vwap_full, vwap_last_row, vwap_at_index,
    vwap_rth_full, vwap_rth_last_row, vwap_rth_at_index,
    vwap_dev_atr_full, vwap_dev_atr_last_row, vwap_dev_atr_at_index,
)
from classes.indicators.session_struct import (
    rolling_range_width_full, rolling_range_width_last_row, rolling_range_width_at_index,
    candle_body_pct_full, candle_body_pct_last_row, candle_body_pct_at_index,
    mins_since_open_full, mins_since_open_last_row, mins_since_open_at_index,
    tod_cyclical_full, tod_cyclical_last_row, tod_cyclical_at_index,
    intraday_tr_stats_full, intraday_tr_stats_last_row, intraday_tr_stats_at_index,
    range_contraction_full, range_contraction_last_row, range_contraction_at_index,
    intraday_extremes_full, intraday_extremes_last_row, intraday_extremes_at_index,
)
from classes.indicators.update_router import (
    update_indicators_last_row as _router_update_last_row,
    update_indicators_at_index as _router_update_at_index,
)
from classes.indicators.adx import adx_full, adx_last_row, adx_at_index
from classes.indicators.r2 import r2_full, r2_last_row, r2_at_index
from classes.indicators.keltner import (
    keltner_full, keltner_last_row, keltner_at_index,
)
from classes.indicators.volume_stats import (
    vol_stats_full, vol_stats_last_row, vol_stats_at_index
)
from classes.indicators.pullback import (
    pullback_full, pullback_last_row, pullback_at_index
)
from classes.indicators.breakout_age import (
    breakout_age_full, breakout_age_last_row, breakout_age_at_index,
)
from classes.indicators.vwap_stats import (
    vwap_stats_full, vwap_stats_last_row, vwap_stats_at_index
)
from classes.indicators.range_tightness import (
    range_tightness_full, range_tightness_last_row, range_tightness_at_index
)
from classes.wavelets.wavelets import (
    add_daubechies_wavelets, add_daubechies_wavelets_last_row, add_daubechies_wavelets_at_index,
    add_morlet_cwt, add_morlet_cwt_last_row, add_morlet_cwt_at_index,
    add_haar_denoising, add_haar_denoising_last_row, add_haar_denoising_at_index,
    add_mexican_hat, add_mexican_hat_last_row, add_mexican_hat_at_index,
    add_symlet_wavelets, add_symlet_wavelets_last_row, add_symlet_wavelets_at_index
)
from classes.indicators.fft import add_fft_full, add_fft_last_row, add_fft_at_index

# ----------------------------------------------------------------------
# Indicator Registry & Configs (unchanged)
# ----------------------------------------------------------------------
INDICATOR_REGISTRY = {    
    # ATR
    "atr": atr,
    "atr_last_row": atr_last_row,
    "atr_at_index": atr_at_index,
    "atr_ratio": atr_ratio,
    "atr_ratio_last_row": atr_ratio_last_row,
    "atr_ratio_at_index": atr_ratio_at_index,
    "atr_ratio_sm": atr_ratio_sm,
    "atr_ratio_sm_last_row": atr_ratio_sm_last_row,
    "atr_ratio_sm_at_index": atr_ratio_sm_at_index,
    # EMA
    "ema": ema_full,
    "ema_last_row": ema_last_row,
    "ema_at_index": ema_at_index,
    "ema_curve": ema_curve_full,
    "ema_curve_last_row": ema_curve_last_row,
    "ema_curve_at_index": ema_curve_at_index,
    "z_close_ema60": zscore_close_vs_ema,
    "z_ema60_ema120": zscore_ema60_vs_ema120,
    "ema_slope_60_120": ema_slope_over,
    # Bollinger Bands
    "bollinger_bands": bollinger_full,
    "bollinger_bands_last_row": bollinger_last_row,
    "bollinger_bands_at_index": bollinger_at_index,
    "bb_bandwidth": bandwidth_full,
    "bb_bandwidth_last_row": bandwidth_last_row,
    "bb_bandwidth_at_index": bandwidth_at_index,
    # RSI
    "rsi": rsi_full,
    "rsi_last_row": rsi_last_row,
    "rsi_at_index": rsi_at_index,
    "donchian": donchian_full,
    "donchian_last_row": donchian_last_row,
    "donchian_at_index": donchian_at_index,
    "donchian_pos": donchian_pos_full,
    "donchian_pos_last_row": donchian_pos_last_row,
    "donchian_pos_at_index": donchian_pos_at_index,
    "macd": macd_full,
    "macd_last_row": macd_last_row,
    "macd_at_index": macd_at_index,
    "roc": roc_full,
    "roc_last_row": roc_last_row,
    "roc_at_index": roc_at_index,
    "range_width": rolling_range_width_full,
    "range_width_last_row": rolling_range_width_last_row,
    "range_width_at_index": rolling_range_width_at_index,
    "candle_body_pct": candle_body_pct_full,
    "candle_body_pct_last_row": candle_body_pct_last_row,
    "candle_body_pct_at_index": candle_body_pct_at_index,
    "mins_since_open": mins_since_open_full,
    "mins_since_open_last_row": mins_since_open_last_row,
    "mins_since_open_at_index": mins_since_open_at_index,
    "tod_cyclical": tod_cyclical_full,
    "tod_cyclical_last_row": tod_cyclical_last_row,
    "tod_cyclical_at_index": tod_cyclical_at_index,
    "intraday_tr_stats": intraday_tr_stats_full,
    "intraday_tr_stats_last_row": intraday_tr_stats_last_row,
    "intraday_tr_stats_at_index": intraday_tr_stats_at_index,
    "range_contraction": range_contraction_full,
    "range_contraction_last_row": range_contraction_last_row,
    "range_contraction_at_index": range_contraction_at_index,
    "intraday_extremes": intraday_extremes_full,
    "intraday_extremes_last_row": intraday_extremes_last_row,
    "intraday_extremes_at_index": intraday_extremes_at_index,
    # Vol regime ratios
    "vwap": vwap_full,
    "vwap_last_row": vwap_last_row,
    "vwap_at_index": vwap_at_index,
    "vwap_rth": vwap_rth_full,
    "vwap_rth_last_row": vwap_rth_last_row,
    "vwap_rth_at_index": vwap_rth_at_index,
    "vwap_dev_atr": vwap_dev_atr_full,
    "vwap_dev_atr_last_row": vwap_dev_atr_last_row,
    "vwap_dev_atr_at_index": vwap_dev_atr_at_index,
    "vwap_stats": vwap_stats_full,
    "vwap_stats_last_row": vwap_stats_last_row,
    "vwap_stats_at_index": vwap_stats_at_index,
    # Structure / ranges / momentum
    "adx": adx_full,
    "adx_last_row": adx_last_row,
    "adx_at_index": adx_at_index,
    "r2": r2_full,
    "r2_last_row": r2_last_row,
    "r2_at_index": r2_at_index,
    "keltner": keltner_full,
    "keltner_last_row": keltner_last_row,
    "keltner_at_index": keltner_at_index,
    "vol_stats": vol_stats_full,
    "vol_stats_last_row": vol_stats_last_row,
    "vol_stats_at_index": vol_stats_at_index,
    "breakout_age": breakout_age_full,
    "breakout_age_last_row": breakout_age_last_row,
    "breakout_age_at_index": breakout_age_at_index,
    'range_tightness': range_tightness_full,
    'range_tightness_last_row': range_tightness_last_row,
    'range_tightness_at_index': range_tightness_at_index,
    # Wavelets
    "daubechies_wavelets": add_daubechies_wavelets,
    "daubechies_wavelets_last_row": add_daubechies_wavelets_last_row,
    "daubechies_wavelets_at_index": add_daubechies_wavelets_at_index,
    "morlet_cwt": add_morlet_cwt,
    "morlet_cwt_last_row": add_morlet_cwt_last_row,
    "morlet_cwt_at_index": add_morlet_cwt_at_index,
    "haar_denoising": add_haar_denoising,
    "haar_denoising_last_row": add_haar_denoising_last_row,
    "haar_denoising_at_index": add_haar_denoising_at_index,
    "mexican_hat": add_mexican_hat,
    "mexican_hat_last_row": add_mexican_hat_last_row,
    "mexican_hat_at_index": add_mexican_hat_at_index,
    "symlet_wavelets": add_symlet_wavelets,
    "symlet_wavelets_last_row": add_symlet_wavelets_last_row,
    "symlet_wavelets_at_index": add_symlet_wavelets_at_index,
    # Frequency domain
    'fft': add_fft_full,
    'fft_last_row': add_fft_last_row,
    'fft_at_index': add_fft_at_index,
}

DEFAULT_INDICATOR_CONFIG = [
    {'name': 'bollinger_bands', 'params': {'window': 20, 'num_std': 2, 'prefix': 'bb'}},
    {'name': 'ema', 'params': {'span': 21, 'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 50, 'prefix': 'ema'}},
    {'name': 'rsi', 'params': {'window': 14, 'prefix': 'rsi'}},
    {'name': 'atr', 'params': {'window': 14, 'prefix': 'atr'}},
]

MINUTE_AUGMENTED_INDICATOR_CONFIG = [
    # === ATRs (volatility regime) ===
    {'name': 'atr', 'params': {'window': 14, 'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 30, 'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 60, 'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 90, 'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 120, 'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 150, 'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 180, 'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 210, 'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 240, 'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 390, 'prefix': 'atr'}},
    {'name': 'atr_ratio', 'params': {'fast': 90, 'slow': 390, 'prefix': 'atrR'}},
    {'name': 'atr_ratio_sm', 'params': {'fast': 14, 'slow': 60, 'prefix': 'atrRsm'}},
    # === EMAs (trend & slope) ===
    {'name': 'ema', 'params': {'span': 14, 'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 21, 'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 30, 'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 45, 'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 60, 'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 120, 'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 210, 'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 240, 'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 390, 'prefix': 'ema'}},
    # === EMA-DERIVED FEATURES (after deps)
    {'name': 'z_close_ema60', 'params': {'ema_span': 60}},
    {'name': 'z_ema60_ema120', 'params': {}},
    {'name': 'ema_slope_60_120', 'params': {'span1': 60, 'span2': 120}},
    {'name': 'ema_curve', 'params': {'window': 60}},
    # === RSI (slower intraday momentum) ===
    {'name': 'rsi', 'params': {'window': 30, 'prefix': 'rsi'}},
    {'name': 'rsi', 'params': {'window': 60, 'prefix': 'rsi'}},
    # === Bollinger bandwidths ===
    {'name': 'bb_bandwidth', 'params': {'window': 20, 'prefix': 'bb20'}},
    {'name': 'bb_bandwidth', 'params': {'window': 60, 'prefix': 'bb60'}},
    {'name': 'bb_bandwidth', 'params': {'window': 90, 'prefix': 'bb90'}},
    {'name': 'bb_bandwidth', 'params': {'window': 120, 'prefix': 'bb120'}},
    {'name': 'bollinger_bands', 'params': {'window': 390, 'num_std': 2, 'prefix': 'bb390'}},
    # === Donchian ===
    {'name': 'donchian', 'params': {'window': 30, 'prefix': 'dc30'}},
    {'name': 'donchian_pos', 'params': {'window': 30, 'prefix': 'dc30'}},
    {'name': 'donchian', 'params': {'window': 60, 'prefix': 'dc60'}},
    {'name': 'donchian_pos', 'params': {'window': 60, 'prefix': 'dc60'}},
    {'name': 'donchian', 'params': {'window': 120, 'prefix': 'dc120'}},
    {'name': 'donchian_pos', 'params': {'window': 120, 'prefix': 'dc120'}},
    # === Momentum & structure ===
    {'name': 'macd', 'params': {'fast': 12, 'slow': 26, 'signal': 9, 'prefix': 'macd'}},
    {'name': 'roc', 'params': {'window': 30, 'prefix': 'roc'}},
    {'name': 'roc', 'params': {'window': 60, 'prefix': 'roc'}},
    # === Session context & extras ===
    {'name': 'range_width', 'params': {'window': 60, 'prefix': 'rw'}},
    {'name': 'candle_body_pct', 'params': {'prefix': 'cb'}},
    {'name': 'mins_since_open', 'params': {}},
    {'name': 'tod_cyclical', 'params': {'session_len': 390, 'src_col': 'mins_since_open', 'prefix': 'tod'}},
    {'name': 'intraday_tr_stats', 'params': {'prefix': 'trday'}},
    {'name': 'range_contraction', 'params': {'k': 5, 'prefix': 'rc'}},
    {'name': 'intraday_extremes', 'params': {'atr_col': 'atr_90', 'prefix': 'iday'}},
    # === VWAP anchors ===
    {'name': 'vwap_rth', 'params': {'price_col': 'close', 'prefix': 'vwap'}},
    {'name': 'vwap_dev_atr', 'params': {'atr_col': 'atr_90', 'vwap_col': 'vwap', 'prefix': 'vwap_dev_atr'}},
    # === ADX ===
    {'name': 'adx', 'params': {'window': 14}},
    {'name': 'adx', 'params': {'window': 30}},
    # === R2 ===
    {'name': 'r2', 'params': {'window': 30}},
    {'name': 'r2', 'params': {'window': 60}},
    # === Keltner ===
    {'name': 'keltner', 'params': {'window': 20, 'k_atr': 1.5, 'bb_std': 2.0}},
    {'name': 'vol_stats', 'params': {'window': 60}},
    {'name': 'breakout_age', 'params': {'window': 20}},
    {'name': 'breakout_age', 'params': {'window': 60}},
    {'name': 'vwap_stats', 'params': {'window': 60}},
    {'name': 'range_tightness', 'params': {'window': 30}},
    # === Wavelets ===
    {'name': 'daubechies_wavelets', 'params': {'levels': 3}},
    {'name': 'morlet_cwt', 'params': {}},
    {'name': 'haar_denoising', 'params': {'level': 2, 'threshold': 0.1}},
    {'name': 'mexican_hat', 'params': {}},
    {'name': 'symlet_wavelets', 'params': {'levels': 3}},
    # === FFT ===
    {'name': 'fft', 'params': {'nperseg': 64, 'noverlap': 32, 'top_k': 3}},
]

# ----------------------------------------------------------------------
# Helper Functions (unchanged)
# ----------------------------------------------------------------------
def update_indicators_last_row(df, indicator_config=None):
    if indicator_config is None:
        indicator_config = DEFAULT_INDICATOR_CONFIG
    _router_update_last_row(df, indicator_config, INDICATOR_REGISTRY)

def update_indicators_at_index(df, idx, indicator_config=None):
    if indicator_config is None:
        indicator_config = DEFAULT_INDICATOR_CONFIG
    _router_update_at_index(df, idx, indicator_config, INDICATOR_REGISTRY)

_INDICATOR_PREFIXES = (
    "bb_", "ema_", "rsi_", "atr_", "tr", "dc", "macd_", "roc_", "rw_", "cb",
    "atrR_", "atrRsm_",
    "mins_since_open",
    "tod_", "trday_", "rc_", "iday_",
    "z_close_ema60", "z_ema60_ema120", "ema_slope_60_120",
    # Wavelet prefixes
    "db4_", "morlet_", "haar_", "mexh_", "sym4_",
    "fft_",
)

def _ensure_indicator_dependencies(df, indicator_config):
    names = {c["name"] for c in indicator_config}
    if "tod_cyclical" in names and "mins_since_open" not in df.columns:
        INDICATOR_REGISTRY["mins_since_open"](df)
    if "intraday_tr_stats" in names:
        ensure_tr(df)
    for cfg in indicator_config:
        if cfg.get("name") == "range_tightness":
            params = cfg.get("params", {}) or {}
            w = int(params.get("window", 30))
            atr_col = f"atr_{w}"
            if atr_col not in df.columns:
                INDICATOR_REGISTRY["atr"](df, window=w, prefix="atr")
            dc_prefix = f"dc{w}_"
            if not any(c.startswith(dc_prefix) for c in df.columns):
                INDICATOR_REGISTRY["donchian"](df, window=w, prefix=f"dc{w}")

def _safe_div(a, b, eps=1e-9):
    return np.where(np.abs(b) > eps, a / b, 0.0)

def _bars_since(series_bool: pd.Series) -> pd.Series:
    idx = np.where(series_bool.to_numpy(), np.arange(len(series_bool)), -1)
    last = np.maximum.accumulate(idx)
    out = (np.arange(len(series_bool), dtype=float) - last)
    out[last < 0] = np.nan
    return pd.Series(out, index=series_bool.index)

def _compute_derived_policy_features(df: pd.DataFrame, tz="America/New_York") -> None:
    if df is None or df.empty:
        return
    CLOSED = -1.0

    # --- 1. Collect all new columns in a dict ---
    new_cols = {}

    # mins_since_open
    if "mins_since_open" not in df.columns:
        ts = pd.to_datetime(df.get("timestamp", df.get("date")), utc=True, errors="coerce")
        try:
            ts_local = ts.dt.tz_convert(tz)
        except Exception:
            ts_local = ts.dt.tz_localize("UTC").dt.tz_convert(tz)
        rth_open = ts_local.dt.floor('D') + pd.Timedelta(hours=9, minutes=30)
        mins = (ts_local - rth_open).dt.total_seconds() / 60.0
        rth_mask = ts_local.dt.time.between(_time(9, 30), _time(16, 0))
        rth_mask = rth_mask.fillna(False)
        out = pd.Series(CLOSED, index=df.index, dtype=float)
        out[rth_mask] = mins[rth_mask]
        new_cols["mins_since_open"] = out

    # atr ratios
    if "atrR_90_390" not in df.columns and {"atr_90", "atr_390"}.issubset(df.columns):
        new_cols["atrR_90_390"] = _safe_div(df["atr_90"], df["atr_390"])
    if "atrRsm_14_60" not in df.columns and {"atr_14", "atr_60"}.issubset(df.columns):
        new_cols["atrRsm_14_60"] = _safe_div(df["atr_14"] - df["atr_60"], df["atr_60"])

    # donchian pos
    for w in (30, 60, 120):
        low = f"dc{w}_low"
        wid = f"dc{w}_width"
        pos = f"dc{w}_pos"
        if pos not in df.columns and {low, wid}.issubset(df.columns):
            new_cols[pos] = _safe_div(df["close"] - df[low], df[wid]).clip(0, 1)

    # roc_60
    if "roc_60" not in df.columns and "close" in df.columns:
        new_cols["roc_60"] = df["close"].pct_change(60)

    # breakout_age_60
    if "breakout_age_60" not in df.columns and "close" in df.columns:
        roll_hi = df["close"].rolling(60, min_periods=60).max().shift(1)
        roll_lo = df["close"].rolling(60, min_periods=60).min().shift(1)
        new_hi = df["close"] > roll_hi
        new_lo = df["close"] < roll_lo
        trig = (new_hi | new_lo).fillna(False)
        new_cols["breakout_age_60"] = _bars_since(trig)

    # intraday extremes
    if {"high", "low", "close", "atr_14"}.issubset(df.columns):
        ts = pd.to_datetime(df.get("timestamp", df.get("date")), utc=True, errors="coerce")
        try:
            ts_local = ts.dt.tz_convert(tz)
        except Exception:
            ts_local = ts.dt.tz_localize("UTC").dt.tz_convert(tz)
        day = ts_local.dt.date
        day_high = df.groupby(day)["high"].cummax()
        day_low = df.groupby(day)["low"].cummin()
        if "iday_dist_to_high_atr" not in df.columns:
            new_cols["iday_dist_to_high_atr"] = _safe_div(day_high - df["close"], df["atr_14"])
        if "iday_dist_to_low_atr" not in df.columns:
            new_cols["iday_dist_to_low_atr"] = _safe_div(df["close"] - day_low, df["atr_14"])

    # --- 2. Apply all at once with pd.concat ---
    if new_cols:
        new_df = pd.DataFrame(new_cols, index=df.index)
        df.update(new_df)  # fast in-place for existing cols
        # Add any new columns not already in df
        missing = [c for c in new_df.columns if c not in df.columns]
        if missing:
            df[missing] = new_df[missing]

    # --- 3. Final defragment (only once per bar) ---
    df._mgr.consolidate() 

# ----------------------------------------------------------------------
# TradingEnvironment – MAIN CLASS (FIXED)
# ----------------------------------------------------------------------
class TradingEnvironment(QtCore.QObject):
    bar_advanced = QtCore.pyqtSignal()
    backtest_finished = QtCore.pyqtSignal()

    def __init__(self, precompute_indicators: bool = False):
        super().__init__()
        self.api = None
        self.bot = None
        self._indicator_cache = {}
        self._precompute = bool(precompute_indicators)

    def set_api(self, api):
        self.api = api
        cfg = {}
        try:
            cfg = getattr(api, "config", {}) or {}
        except Exception:
            cfg = {}
        if bool(cfg.get("precompute_indicators", False)):
            self._precompute = True
        assume_precomputed = bool(cfg.get("assume_indicators_precomputed", False))
        debug_indicators = bool(cfg.get("indicator_debug", False))

        for name, handler in [
            ("bar_advanced", self.bar_advanced.emit),
            ("bar_updated", lambda _df: self.on_bar_advanced(compute_indicators=True)),
            ("bar_closed", self._recompute_closed_bar_indicators),
            ("backtest_finished", self.backtest_finished.emit),
        ]:
            if hasattr(api, name):
                sig = getattr(api, name)
                try:
                    sig.disconnect()
                except Exception:
                    pass
                if self._precompute and name == "bar_closed":
                    continue
                sig.connect(handler)

        for symbol in self.get_asset_list():
            df = self.api.get_asset_data(symbol)
            if df is None or df.empty:
                continue
            cfg_list = self._indicator_config_for_symbol(symbol)
            print(f"[INDICATOR CONFIG] {symbol}: {len(cfg_list)} indicators")
            print(f"[INDICATOR CONFIG] Sample: {[c['name'] for c in cfg_list[:5]]}")
            if assume_precomputed:
                print("Assumed Precomputed")
                print(f"[PRECOMPUTE] Computing {len(cfg_list)} indicators for {symbol}...")
                if debug_indicators:
                    print(f"[INDDBG] {symbol}: assume_indicators_precomputed=True → skipping compute for all indicators.")
                _compute_derived_policy_features(df)
                self._indicator_cache[symbol] = df.index[-1]
                self._defragment_asset_df(symbol)
                continue
            if self._precompute:
                print("Precomputing Indicators")
                for ind in cfg_list:
                    name = ind.get("name")
                    params = ind.get("params", {}) or {}
                    pref = params.get("prefix", None)
                    def _exists(col: str) -> bool:
                        return (col in df.columns)
                    present = False
                    if name == "ema":
                        span = int(params.get("span", 0))
                        if span > 0:
                            expected = f"{pref or 'ema'}_{span}"
                            present = _exists(expected)
                    elif name == "atr":
                        w = int(params.get("window", 0))
                        if w > 0:
                            expected = f"{pref or 'atr'}_{w}"
                            present = _exists(expected)
                    elif name == "rsi":
                        w = int(params.get("window", 0))
                        if w > 0:
                            expected = f"{pref or 'rsi'}_{w}"
                            present = _exists(expected)
                    elif name == "adx":
                        w = int(params.get("window", 0) or 14)
                        expected_cols = [f"adx_{w}", f"di_plus_{w}", f"di_minus_{w}"]
                        present = all(_exists(c) for c in expected_cols)
                    elif name == "r2":
                        w = int(params.get("window", 0) or 0)
                        if w > 0:
                            expected = f"r2_{w}"
                            present = _exists(expected)
                        else:
                            present = any(isinstance(c, str) and c.startswith("r2_") for c in df.columns)
                    else:
                        if pref:
                            present = any(isinstance(c, str) and c.startswith(pref) for c in df.columns)
                        else:
                            present = any(isinstance(c, str) and c.startswith(name) for c in df.columns)
                    if present:
                        continue
                    _ensure_indicator_dependencies(df, [ind])
                    if debug_indicators:
                        import time as _t
                        t0 = _t.time()
                        print(f"[INDDBG] {symbol}: computing indicator '{name}' with params={params} …")
                        INDICATOR_REGISTRY[name](df, **params)
                        dt = _t.time() - t0
                        print(f"[INDDBG] {symbol}: computed '{name}' in {dt:.3f}s; new cols added: "
                              f"{[c for c in df.columns if (pref and c.startswith(pref)) or (not pref and c.startswith(name))][:6]}…")
                    else:
                        INDICATOR_REGISTRY[name](df, **params)
                _compute_derived_policy_features(df)
                self._indicator_cache[symbol] = df.index[-1]
                self._defragment_asset_df(symbol)
            else:
                self._compute_indicators(symbol, df, indicator_config=cfg_list)

    def set_bot(self, bot):
        self.bot = bot

    # ------------------------------------------------------------------
    # FIXED: on_bar_advanced() – RESTORED ORIGINAL CALL
    # ------------------------------------------------------------------
    def on_bar_advanced(self, _ignored=None, compute_indicators=True):
        if not self.api or not getattr(self.api, "is_running", False):
            return
        compute = (False if self._precompute else bool(compute_indicators))
        for symbol in self.get_asset_list():
            df = self.api.get_asset_data(symbol)
            if compute and df is not None and not df.empty:
                cfg = self._indicator_config_for_symbol(symbol)
                self._compute_indicators(symbol, df, indicator_config=cfg)

        # <<< THIS LINE WAS MISSING — NOW RESTORED >>>
        if self.bot is not None:
            self.bot.on_bar(self)
        # <<< END OF FIX >>>

    def _recompute_closed_bar_indicators(self, _df=None):
        if self._precompute or not self.api:
            return
        idx = getattr(self.api, "current_index", None)
        if idx is None:
            return
        for symbol in self.get_asset_list():
            df = self.api.get_asset_data(symbol)
            if df is None or df.empty or idx >= len(df):
                continue
            if self._indicator_cache.get(symbol) == idx:
                continue
            cfg = self._indicator_config_for_symbol(symbol)
            self._defragment_asset_df(symbol)
            _router_update_at_index(df, idx, cfg, INDICATOR_REGISTRY)
            _compute_derived_policy_features(df)
            self._defragment_asset_df(symbol)
            self._indicator_cache[symbol] = idx

    # ------------------------------------------------------------------
    # Passthroughs (unchanged)
    # ------------------------------------------------------------------
    def get_portfolio(self): return self.api.get_portfolio() if self.api else {}
    def get_positions(self): return self.api.get_positions() if self.api else {}
    def get_total_pnl(self): return self.api.get_total_pnl() if self.api else {'realized': 0.0, 'unrealized': 0.0, 'total': 0.0}
    def get_asset_list(self): return self.api.get_asset_list() if self.api else []
    def get_asset_data(self, symbol):
        if not self.api: return None
        df = self.api.get_asset_data(symbol)
        if df is None or df.empty or 'close' not in df.columns: return df
        return df
    def get_latest_data(self, symbol, window_size=256):
        return self.api.get_latest_data(symbol, window_size) if self.api else pd.DataFrame()
    def get_orders(self, symbol=None):
        if not hasattr(self.api, "trade_log"): return []
        return self.api.trade_log if symbol is None else [o for o in self.api.trade_log if o['symbol'] == symbol]
    def place_order(self, order):
        return self.api.place_order(order) if self.api else None
    def get_order_status(self, order_id): return self.api.get_order_status(order_id) if self.api else None
    def cancel_order(self, order_id):
        if self.api: self.api.cancel_order(order_id)
        
    def modify_stop_loss(self, symbol, new_value):
        if self.api: 
            return self.api.modify_stop_loss(symbol, new_value)
        
    def modify_take_profit(self, symbol, new_value):
        if self.api: self.api.modify_take_profit(symbol, new_value)
    def connect(self): return self.api.connect() if self.api else None
    def disconnect(self): return self.api.disconnect() if self.api else None

    def reset_indicators(self):
        if self._precompute:
            return
        self._indicator_cache.clear()
        for symbol in self.get_asset_list():
            df = self.api.get_asset_data(symbol)
            if df is not None and not df.empty:
                df.drop(
                    columns=[col for col in df.columns
                             if any(col.startswith(p) for p in _INDICATOR_PREFIXES)
                             or col.startswith("_tmp_")],
                    inplace=True, errors="ignore"
                )

    def _drop_indicator_columns(self, df):
        cols_to_drop = [col for col in df.columns
                        if any(col.startswith(p) for p in _INDICATOR_PREFIXES)
                        or col.startswith("_tmp_")]
        return df.drop(columns=cols_to_drop, errors='ignore')

    def _has_indicator_columns(self, df):
        return any(col.startswith(p) for p in _INDICATOR_PREFIXES for col in df.columns)

    def _compute_indicators(self, symbol, df, indicator_config=None):
        if self._precompute:
            return
        if indicator_config is None:
            indicator_config = DEFAULT_INDICATOR_CONFIG
        current_idx = df.index[-1]
        indicator_cols = [col for col in df.columns
                          if any(col.startswith(p) for p in _INDICATOR_PREFIXES)]
        last_idx = self._indicator_cache.get(symbol)
        if not indicator_cols or (len(indicator_cols) > 0 and df[indicator_cols].isna().all().all()):
            _ensure_indicator_dependencies(df, indicator_config)
            for cfg in indicator_config:
                INDICATOR_REGISTRY[cfg["name"]](df, **cfg.get("params", {}))
            _compute_derived_policy_features(df)
            self._defragment_asset_df(symbol)
            self._indicator_cache[symbol] = current_idx
        elif last_idx != current_idx:
            self._defragment_asset_df(symbol)
            _router_update_last_row(df, indicator_config, INDICATOR_REGISTRY)
            _compute_derived_policy_features(df)
            self._defragment_asset_df(symbol)
            self._indicator_cache[symbol] = current_idx

    def _indicator_config_for_symbol(self, symbol: str):
        tf = None
        try:
            tf = self.api.get_symbol_timeframe(symbol) or self.api.get_timeframe()
        except Exception:
            tf = None
        if isinstance(tf, str) and tf.endswith('m'):
            return DEFAULT_INDICATOR_CONFIG + MINUTE_AUGMENTED_INDICATOR_CONFIG
        return DEFAULT_INDICATOR_CONFIG

    def _defragment_asset_df(self, symbol: str) -> None:
        if not getattr(self.api, "assets", None):
            return
        for asset in self.api.assets:
            if asset.get("symbol") == symbol:
                df = asset.get("data")
                if df is None:
                    break
                asset["data"] = df.copy()
                break
