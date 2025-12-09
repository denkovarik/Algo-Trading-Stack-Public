# classes/Trading_Environment.py
from PyQt5 import QtCore
import pandas as pd
import numpy as np
from datetime import time as _time
from tqdm import tqdm

# ----------------------------------------------------------------------
# Indicator imports
# ----------------------------------------------------------------------
from classes.indicators.atr import (
    atr_full, atr_last_row, atr_at_index,
    ensure_tr,
)
from classes.indicators.ema import (
    ema_full, ema_last_row, ema_at_index,
    zscore_close_vs_ema_last_row,
    zscore_ema60_vs_ema120_last_row,
    ema_slope_over_last_row,
    ema_curve_full,
    ema_curve_last_row,
    ema_curve_at_index,
)
from classes.indicators.rsi import rsi_full, rsi_last_row, rsi_at_index
from classes.indicators.macd import macd_full, macd_last_row, macd_at_index
from classes.indicators.adx import adx_full, adx_last_row, adx_at_index
from classes.indicators.sma import (
    sma_full, sma_last_row, sma_at_index,
    ema_sma_crossover_full, ema_sma_crossover_last_row, ema_sma_crossover_at_index,
)
from classes.indicators.bollinger import bollinger_full, bollinger_last_row, bollinger_at_index
from classes.indicators.rsi_divergence import rsi_divergence_full, rsi_divergence_last_row, rsi_divergence_at_index
from classes.indicators.psar import psar_full, psar_last_row, psar_at_index
from classes.indicators.update_router import (
    update_indicators_last_row as _real_last_row,
    update_indicators_at_index as _real_at_index,
)

# ----------------------------------------------------------------------
# Registry & configs
# ----------------------------------------------------------------------
INDICATOR_REGISTRY = {
    "atr": atr_full,
    "atr_last_row": atr_last_row,
    "atr_at_index": atr_at_index,
    "ema": ema_full,
    "ema_last_row": ema_last_row,
    "ema_at_index": ema_at_index,
    "rsi": rsi_full,
    "rsi_last_row": rsi_last_row,
    "rsi_at_index": rsi_at_index,
    "macd": macd_full,
    "macd_last_row": macd_last_row,
    "macd_at_index": macd_at_index,
    "adx": adx_full,
    "adx_last_row": adx_last_row,
    "adx_at_index": adx_at_index,
    "sma": sma_full,
    "sma_last_row": sma_last_row,
    "sma_at_index": sma_at_index,
    "ema_sma_crossover": ema_sma_crossover_full,
    "ema_sma_crossover_last_row": ema_sma_crossover_last_row,
    "ema_sma_crossover_at_index": ema_sma_crossover_at_index,
    "bollinger_bands": bollinger_full,
    "rsi_divergence": rsi_divergence_full,
    "psar": psar_full,
    "ema_curve": ema_curve_full,  
}

DEFAULT_INDICATOR_CONFIG = [
    {'name': 'ema', 'params': {'span': 10, 'prefix': 'ema'}},
    {'name': 'rsi', 'params': {'window': 14, 'prefix': 'rsi'}},
    {'name': 'macd', 'params': {'fast': 12, 'slow': 26, 'signal': 9, 'prefix': 'macd'}},
    {'name': 'adx', 'params': {'window': 14}},
    {'name': 'atr', 'params': {'window': 14, 'prefix': 'atr'}},
    {'name': 'sma', 'params': {'window': 20, 'prefix': 'sma'}},
    {'name': 'ema_sma_crossover', 'params': {'ema_span': 10, 'sma_window': 20, 'prefix': 'crossover'}},
    {'name': 'bollinger_bands', 'params': {'window': 20, 'std_mult': 2.0, 'prefix': 'bb'}},
    {'name': 'rsi_divergence', 'params': {'rsi_window': 14, 'lookback': 5, 'prefix': 'rsi_div'}},
    {'name': 'psar', 'params': {'af_start': 0.02, 'af_step': 0.02, 'af_max': 0.2, 'prefix': 'psar'}},
]

MINUTE_AUGMENTED = [
    {'name': 'atr', 'params': {'window': 14, 'prefix': 'atr'}},
    {'name': 'ema', 'params': {'span': 10, 'prefix': 'ema'}},
    {'name': 'rsi', 'params': {'window': 14, 'prefix': 'rsi'}},
    {'name': 'macd', 'params': {'fast': 12, 'slow': 26, 'signal': 9, 'prefix': 'macd'}},
    {'name': 'adx', 'params': {'window': 14}},
    {'name': 'sma', 'params': {'window': 20, 'prefix': 'sma'}},
    {'name': 'ema_sma_crossover', 'params': {'ema_span': 10, 'sma_window': 20, 'prefix': 'crossover'}},
    {'name': 'bollinger_bands', 'params': {'window': 20, 'std_mult': 2.0, 'prefix': 'bb'}},
    {'name': 'rsi_divergence', 'params': {'rsi_window': 14, 'lookback': 5, 'prefix': 'rsi_div'}},
    {'name': 'psar', 'params': {'af_start': 0.02, 'af_step': 0.02, 'af_max': 0.2, 'prefix': 'psar'}},
]

# ----------------------------------------------------------------------
# PUBLIC helper functions – test-friendly (registry optional)
# ----------------------------------------------------------------------
def update_indicators_last_row(df, indicator_config=None, registry=None):
    if indicator_config is None:
        indicator_config = DEFAULT_INDICATOR_CONFIG
    if registry is None:
        registry = INDICATOR_REGISTRY
    _real_last_row(df, indicator_config, registry)

def update_indicators_at_index(df, idx, indicator_config=None, registry=None):
    if indicator_config is None:
        indicator_config = DEFAULT_INDICATOR_CONFIG
    if registry is None:
        registry = INDICATOR_REGISTRY
    _real_at_index(df, idx, indicator_config, registry)

# ----------------------------------------------------------------------
# Private routers – used internally (always pass registry)
# ----------------------------------------------------------------------
def _router_update_last_row(df, config, registry):
    _real_last_row(df, config, registry)

def _router_update_at_index(df, idx, config, registry):
    _real_at_index(df, idx, config, registry)

# ----------------------------------------------------------------------
# Column prefixes
# ----------------------------------------------------------------------
_INDICATOR_PREFIXES = (
    "ema_", "rsi_", "atr_", "macd_", "atrR_", "sma_", "crossover_", "tr",
    "bb_", "rsi_div_", "psar_", "adx_",  # Added new prefixes for reset
)

def _safe_div(a, b, default=np.nan):
    try:
        if b == 0 or b is None or pd.isna(b):
            return default
        return a / b
    except Exception:
        return default

# ----------------------------------------------------------------------
# MAIN CLASS
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
        cfg = getattr(api, "config", {}) or {}
        if cfg.get("precompute_indicators", False):
            self._precompute = True

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
            if cfg.get("assume_indicators_precomputed", False):
                self._indicator_cache[symbol] = df.index[-1]
                continue
            if self._precompute:
                for ind in cfg_list:
                    INDICATOR_REGISTRY[ind["name"]](df, **ind.get("params", {}))
                self._indicator_cache[symbol] = df.index[-1]
            else:
                self._compute_indicators(symbol, df, indicator_config=cfg_list)

    def set_bot(self, bot):
        self.bot = bot

    def on_bar_advanced(self, _ignored=None, compute_indicators=True):
        if not self.api or not getattr(self.api, "is_running", False):
            return
        compute = not self._precompute and compute_indicators
        for symbol in self.get_asset_list():
            df = self.api.get_asset_data(symbol)
            if compute and df is not None and not df.empty:
                cfg = self._indicator_config_for_symbol(symbol)
                self._compute_indicators(symbol, df, indicator_config=cfg)
        if self.bot:
            self.bot.on_bar(self)

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
            _router_update_at_index(df, idx, cfg, INDICATOR_REGISTRY)
            self._indicator_cache[symbol] = idx

    def _compute_indicators(self, symbol, df, indicator_config=None):
        if self._precompute:
            return
        if indicator_config is None:
            indicator_config = self._indicator_config_for_symbol(symbol)
        current_idx = df.index[-1]
        last_cached = self._indicator_cache.get(symbol)

        if last_cached is None:
            for cfg in indicator_config:
                INDICATOR_REGISTRY[cfg["name"]](df, **cfg.get("params", {}))
            self._indicator_cache[symbol] = current_idx
        else:
            for idx in df.index[last_cached + 1:]:
                _router_update_at_index(df, idx, indicator_config, INDICATOR_REGISTRY)
            self._indicator_cache[symbol] = current_idx

    def _indicator_config_for_symbol(self, symbol: str):
        tf = None
        try:
            tf = self.api.get_symbol_timeframe(symbol) or self.api.get_timeframe()
        except Exception:
            tf = None
        if isinstance(tf, str) and tf.endswith('m'):
            return DEFAULT_INDICATOR_CONFIG + MINUTE_AUGMENTED
        return DEFAULT_INDICATOR_CONFIG

    def reset_indicators(self):
        if self._precompute:
            return
        self._indicator_cache.clear()
        for symbol in self.get_asset_list():
            df = self.api.get_asset_data(symbol)
            if df is not None and not df.empty:
                df.drop(
                    columns=[c for c in df.columns
                             if any(c.startswith(p) for p in _INDICATOR_PREFIXES)
                             or c.startswith("_tmp_")],
                    inplace=True,
                    errors="ignore",
                )

    # ------------------------------------------------------------------
    # Public passthroughs
    # ------------------------------------------------------------------
    def get_portfolio(self): return self.api.get_portfolio() if self.api else {}
    def get_positions(self): return self.api.get_positions() if self.api else {}
    def get_total_pnl(self):
        return self.api.get_total_pnl() if self.api else {"realized":0.0,"unrealized":0.0,"total":0.0}
    def get_asset_list(self): return self.api.get_asset_list() if self.api else []
    def get_asset_data(self, symbol):
        return self.api.get_asset_data(symbol) if self.api else None
    def get_latest_data(self, symbol, window_size=256):
        return self.api.get_latest_data(symbol, window_size) if self.api else pd.DataFrame()
    def get_orders(self, symbol=None):
        if not hasattr(self.api, "trade_log"):
            return []
        if symbol is None:
            return self.api.trade_log
        return [o for o in self.api.trade_log if o['symbol'] == symbol]
    def place_order(self, order):
        return self.api.place_order(order) if self.api else None
    def get_order_status(self, order_id):
        return self.api.get_order_status(order_id) if self.api else None
    def cancel_order(self, order_id):
        if self.api: self.api.cancel_order(order_id)
    def modify_stop_loss(self, symbol, new_value):
        if self.api: self.api.modify_stop_loss(symbol, new_value)
    def modify_take_profit(self, symbol, new_value):
        if self.api: self.api.modify_take_profit(symbol, new_value)
    def connect(self): return self.api.connect() if self.api else None
    def disconnect(self): return self.api.disconnect() if self.api else None

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
