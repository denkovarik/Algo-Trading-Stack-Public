from PyQt5 import QtCore
import pandas as pd
import numpy as np
from classes.indicators.atr import (
    atr, atr_last_row, atr_at_index,
    atr_ratio, atr_ratio_last_row, atr_ratio_at_index,
    atr_ratio_sm, atr_ratio_sm_last_row, atr_ratio_sm_at_index,
    ensure_tr,
)
from classes.indicators.ema import (
    ema_full,      # full-history compute for initial/precompute phase
    ema_last_row,  # fast last-row update
    ema_at_index,  # per-index recompute
    zscore_close_vs_ema,
    zscore_ema60_vs_ema120,
    ema_slope_over
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
    candle_body_pct_full,  candle_body_pct_last_row,  candle_body_pct_at_index,
    mins_since_open_full,  mins_since_open_last_row,  mins_since_open_at_index,
    tod_cyclical_full,     tod_cyclical_last_row,     tod_cyclical_at_index,
    intraday_tr_stats_full, intraday_tr_stats_last_row, intraday_tr_stats_at_index,
    range_contraction_full, range_contraction_last_row, range_contraction_at_index,
    intraday_extremes_full, intraday_extremes_last_row, intraday_extremes_at_index,
)
from classes.indicators.update_router import (
    update_indicators_last_row as _router_update_last_row,
    update_indicators_at_index as _router_update_at_index,
)

# ---- Indicator Registry & Configs ----

INDICATOR_REGISTRY = {
    # Core
    'bollinger_bands': bollinger_full,
    'bb_bandwidth': bandwidth_full,
    'ema': ema_full,
    'rsi': rsi_full,
    'atr': atr,

    # Structure / ranges / momentum
    'donchian': donchian_full,
    'donchian_pos': donchian_pos_full,
    'macd': macd_full,
    'roc': roc_full,

    # Vol regime ratios
    'atr_ratio': atr_ratio,          # (fast/slow) e.g., 90/390
    'atr_ratio_sm': atr_ratio_sm,    # (short/medium) e.g., 14/60

    # Optional trainer extras
    'z_close_ema60': zscore_close_vs_ema,
    'z_ema60_ema120': zscore_ema60_vs_ema120,
    'ema_slope_60_120': ema_slope_over,
    
    'vwap_rth': vwap_rth_full,
    'vwap_dev_atr': vwap_dev_atr_full,
    
    # Session/ToD + extras
    'range_width':       rolling_range_width_full,
    'candle_body_pct':   candle_body_pct_full,
    'mins_since_open':   mins_since_open_full,
    'tod_cyclical':      tod_cyclical_full,
    'intraday_tr_stats': intraday_tr_stats_full,
    'range_contraction': range_contraction_full,
    'intraday_extremes': intraday_extremes_full,
}

# Default indicators
DEFAULT_INDICATOR_CONFIG = [
    {'name': 'bollinger_bands', 'params': {'window': 20, 'num_std': 2, 'prefix': 'bb'}},
    {'name': 'ema', 'params': {'span': 21, 'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 50, 'prefix': 'ema'}},
    {'name': 'rsi', 'params': {'window': 14, 'prefix': 'rsi'}},
    {'name': 'atr', 'params': {'window': 14, 'prefix': 'atr'}},
]

# Minute-timeframe companions (augmented)
MINUTE_AUGMENTED_INDICATOR_CONFIG = [
    # === EMAs (trend & slope) ===
    {'name': 'ema', 'params': {'span': 14,  'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 21,  'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 30,  'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 45,  'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 60,  'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 120, 'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 210, 'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 240, 'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 390, 'prefix': 'ema'}}, 

    # === ATRs (volatility regime) ===
    {'name': 'atr', 'params': {'window': 14,  'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 30,  'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 60,  'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 90,  'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 120, 'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 150, 'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 180, 'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 210, 'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 240, 'prefix': 'atr'}},
    {'name': 'atr', 'params': {'window': 390, 'prefix': 'atr'}},
    # day-context ratio
    {'name': 'atr_ratio', 'params': {'fast': 90, 'slow': 390, 'prefix': 'atrR'}},
    # reactive micro ratio
    {'name': 'atr_ratio_sm', 'params': {'fast': 14, 'slow': 60, 'prefix': 'atrRsm'}},

    # === RSI (slower intraday momentum) ===
    {'name': 'rsi', 'params': {'window': 30, 'prefix': 'rsi'}},
    {'name': 'rsi', 'params': {'window': 60, 'prefix': 'rsi'}},

    # === Bollinger bandwidths ===
    {'name': 'bb_bandwidth', 'params': {'window': 60,  'prefix': 'bb60'}},
    {'name': 'bb_bandwidth', 'params': {'window': 90,  'prefix': 'bb90'}},
    {'name': 'bb_bandwidth', 'params': {'window': 120, 'prefix': 'bb120'}},
    {'name': 'bollinger_bands', 'params': {'window': 390, 'num_std': 2, 'prefix': 'bb390'}},

    # === Donchian ===
    {'name': 'donchian',     'params': {'window': 60,  'prefix': 'dc60'}},
    {'name': 'donchian_pos', 'params': {'window': 60,  'prefix': 'dc60'}},
    {'name': 'donchian',     'params': {'window': 120, 'prefix': 'dc120'}},
    {'name': 'donchian_pos', 'params': {'window': 120, 'prefix': 'dc120'}},

    # === Momentum & structure ===
    {'name': 'macd',  'params': {'fast': 12, 'slow': 26, 'signal': 9, 'prefix': 'macd'}},
    {'name': 'roc',   'params': {'window': 30, 'prefix': 'roc'}},
    {'name': 'roc',   'params': {'window': 60, 'prefix': 'roc'}},
    {'name': 'range_width',     'params': {'window': 60, 'prefix': 'rw'}},
    {'name': 'candle_body_pct', 'params': {'prefix': 'cb'}},

    # === Session context & extras ===
    {'name': 'mins_since_open',   'params': {}},
    {'name': 'tod_cyclical',      'params': {'session_len': 390, 'src_col': 'mins_since_open', 'prefix': 'tod'}},
    {'name': 'intraday_tr_stats', 'params': {'prefix': 'trday'}},
    {'name': 'range_contraction', 'params': {'k': 5, 'prefix': 'rc'}},
    {'name': 'intraday_extremes', 'params': {'atr_col': 'atr_90', 'prefix': 'iday'}},

    # Optional trainer extras
    {'name': 'z_close_ema60',    'params': {'ema_span': 60}},
    {'name': 'z_ema60_ema120',   'params': {}},
    {'name': 'ema_slope_60_120', 'params': {'span1': 60, 'span2': 120}},
 
    # === VWAP anchors ===
    {'name': 'vwap_rth',     'params': {'price_col': 'close', 'prefix': 'vwap'}},
    {'name': 'vwap_dev_atr', 'params': {'atr_col': 'atr_90', 'vwap_col': 'vwap', 'prefix': 'vwap_dev_atr'}},
]

def update_indicators_last_row(df, indicator_config=None):
    if indicator_config is None:
        indicator_config = DEFAULT_INDICATOR_CONFIG
    _router_update_last_row(df, indicator_config, INDICATOR_REGISTRY)

def update_indicators_at_index(df, idx, indicator_config=None):
    if indicator_config is None:
        indicator_config = DEFAULT_INDICATOR_CONFIG
    _router_update_at_index(df, idx, indicator_config, INDICATOR_REGISTRY)

# ---- Helpers (prefixes + deps) ----

_INDICATOR_PREFIXES = (
    "bb_", "ema_", "rsi_", "atr_", "tr", "dc", "macd_", "roc_", "rw_", "cb",
    "atrR_", "atrRsm_",
    "mins_since_open",
    "tod_", "trday_", "rc_", "iday_",
    "z_close_ema60", "z_ema60_ema120", "ema_slope_60_120",
)

def _ensure_indicator_dependencies(df, indicator_config):
    """
    Guarantee minimal prerequisites so minute-timeframe indicators
    never fail when configs are reordered or trimmed.
    """
    names = {c["name"] for c in indicator_config}

    # tod_cyclical needs mins_since_open
    if "tod_cyclical" in names and "mins_since_open" not in df.columns:
        INDICATOR_REGISTRY["mins_since_open"](df)

    # intraday_tr_stats needs TR
    if "intraday_tr_stats" in names:
        ensure_tr(df)  # ensures 'tr' exists

# ---- Trading Environment ----

class TradingEnvironment(QtCore.QObject):
    bar_advanced = QtCore.pyqtSignal()
    backtest_finished = QtCore.pyqtSignal()

    def __init__(self, precompute_indicators: bool = False):
        super().__init__()
        self.api = None
        self.bot = None
        self._indicator_cache = {}  # symbol -> last index
        self._precompute = bool(precompute_indicators)

    def set_api(self, api):
        self.api = api

        # Optionally pick up YAML flag from engine config:
        try:
            cfg_flag = bool(getattr(api, "config", {}).get("precompute_indicators", False))
            if cfg_flag:
                self._precompute = True
        except Exception:
            pass

        # Disconnect any previous slots to avoid double-wiring
        for name, handler in [
            ("bar_advanced", self.bar_advanced.emit),
            ("bar_updated", lambda _df: self.on_bar_advanced(compute_indicators=False)),
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

        # === Initial indicator compute ===
        for symbol in self.get_asset_list():
            df = self.api.get_asset_data(symbol)
            if df is None or df.empty:
                continue
            cfg = self._indicator_config_for_symbol(symbol)

            if self._precompute:
                # Purge then compute full history by calling each indicator once
                df.drop(
                    columns=[col for col in df.columns
                             if any(col.startswith(p) for p in _INDICATOR_PREFIXES)
                             or col.startswith("_tmp_")],
                    inplace=True, errors="ignore"
                )
                _ensure_indicator_dependencies(df, cfg)
                for ind in cfg:
                    INDICATOR_REGISTRY[ind["name"]](df, **ind.get("params", {}))
                self._indicator_cache[symbol] = df.index[-1]
            else:
                self._compute_indicators(symbol, df, indicator_config=cfg)

    def set_bot(self, bot):
        self.bot = bot

    def on_bar_advanced(self, _ignored=None, compute_indicators=True):
        if not self.api or not getattr(self.api, "is_running", False):
            return

        compute = (False if self._precompute else bool(compute_indicators))

        for symbol in self.get_asset_list():
            df = self.api.get_asset_data(symbol)
            if compute and df is not None and not df.empty:
                cfg = self._indicator_config_for_symbol(symbol)
                self._compute_indicators(symbol, df, indicator_config=cfg)

        if self.bot is not None:
            self.bot.on_bar(self)

    def _recompute_closed_bar_indicators(self, _df=None):
        """Recompute indicators ONLY for the just-closed index on all symbols."""
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
            update_indicators_at_index(df, idx, indicator_config=cfg)
            self._indicator_cache[symbol] = idx

    # Passthroughs
    def get_portfolio(self):      return self.api.get_portfolio() if self.api else {}
    def get_positions(self):      return self.api.get_positions() if self.api else {}
    def get_total_pnl(self):      return self.api.get_total_pnl() if self.api else {'realized': 0.0, 'unrealized': 0.0, 'total': 0.0}
    def get_asset_list(self):     return self.api.get_asset_list() if self.api else []
    def get_asset_data(self, symbol):
        if not self.api: return None
        df = self.api.get_asset_data(symbol)
        if df is None or df.empty or 'close' not in df.columns: return df
        return df
    def get_latest_data(self, symbol): return self.api.get_latest_data(symbol) if self.api else pd.DataFrame()
    def get_orders(self, symbol=None):
        if not hasattr(self.api, "trade_log"): return []
        return self.api.trade_log if symbol is None else [o for o in self.api.trade_log if o['symbol'] == symbol]
    def place_order(self, order):       return self.api.place_order(order) if self.api else None
    def get_order_status(self, order_id): return self.api.get_order_status(order_id) if self.api else None
    def cancel_order(self, order_id):
        if self.api: self.api.cancel_order(order_id)
    def modify_stop_loss(self, symbol, new_value):
        if self.api: self.api.modify_stop_loss(symbol, new_value)
    def modify_take_profit(self, symbol, new_value):
        if self.api: self.api.modify_take_profit(symbol, new_value)
    def connect(self):    return self.api.connect() if self.api else None
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
        """Full compute once; afterwards, only update the newest row when appending."""
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
            self._indicator_cache[symbol] = current_idx
        elif last_idx != current_idx:
            update_indicators_last_row(df, indicator_config=indicator_config)
            self._indicator_cache[symbol] = current_idx

    # -------- per-symbol indicator config chooser --------
    def _indicator_config_for_symbol(self, symbol: str):
        """
        If timeframe is minute-based, return DEFAULT + augmented minute indicators.
        Otherwise return DEFAULT only.
        """
        tf = None
        try:
            tf = self.api.get_symbol_timeframe(symbol) or self.api.get_timeframe()
        except Exception:
            tf = None

        if isinstance(tf, str) and tf.endswith('m'):
            return DEFAULT_INDICATOR_CONFIG + MINUTE_AUGMENTED_INDICATOR_CONFIG
        return DEFAULT_INDICATOR_CONFIG

