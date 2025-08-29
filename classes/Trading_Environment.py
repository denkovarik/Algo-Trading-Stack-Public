from PyQt5 import QtCore
import pandas as pd

# ---- Indicator Functions & Registry ----

def bollinger_bands(df, window=20, num_std=2, prefix='bb'):
    df.loc[:, f'{prefix}_ma'] = df['close'].rolling(window).mean()
    df.loc[:, f'{prefix}_upper'] = df[f'{prefix}_ma'] + num_std * df['close'].rolling(window).std()
    df.loc[:, f'{prefix}_lower'] = df[f'{prefix}_ma'] - num_std * df['close'].rolling(window).std()

def ema(df, span=21, prefix='ema'):
    df.loc[:, f'{prefix}_{span}'] = df['close'].ewm(span=span, adjust=False).mean()

def rsi(df, window=14, prefix='rsi'):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    df.loc[:, f'{prefix}_{window}'] = 100 - (100 / (1 + rs))

def atr(df, window=14, prefix='atr'):
    # NaN-tolerant TR: use close when H/L are missing (synthetic bars)
    high = df['high'].fillna(df['close'])
    low = df['low'].fillna(df['close'])
    close = df['close']
    prev_close = close.shift(1).fillna(close)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    df.loc[:, 'tr'] = tr
    df.loc[:, f'{prefix}_{window}'] = tr.rolling(window).mean()


INDICATOR_REGISTRY = {
    'bollinger_bands': bollinger_bands,
    'ema': ema,
    'rsi': rsi,
    'atr': atr,
}

DEFAULT_INDICATOR_CONFIG = [
    {'name': 'bollinger_bands', 'params': {'window': 20, 'num_std': 2, 'prefix': 'bb'}},
    {'name': 'ema', 'params': {'span': 21, 'prefix': 'ema'}},
    {'name': 'ema', 'params': {'span': 50, 'prefix': 'ema'}},
    {'name': 'rsi', 'params': {'window': 14, 'prefix': 'rsi'}},
    {'name': 'atr', 'params': {'window': 14, 'prefix': 'atr'}},
]

def update_indicators_last_row(df, indicator_config=None):
    """(Used for full recompute or when appending) Compute indicators for the last row of df."""
    if indicator_config is None:
        indicator_config = DEFAULT_INDICATOR_CONFIG
    idx = df.index[-1]
    for ind in indicator_config:
        func = INDICATOR_REGISTRY[ind['name']]
        name = ind['name']
        params = ind.get('params', {})
        if name == 'bollinger_bands':
            window = params.get('window', 20)
            sub_df = df.iloc[-window:].copy()
            func(sub_df, **params)
            for col in [f"{params.get('prefix','bb')}_ma",
                        f"{params.get('prefix','bb')}_upper",
                        f"{params.get('prefix','bb')}_lower"]:
                df.at[idx, col] = sub_df[col].iloc[-1]
        elif name == 'ema':
            span = params.get('span', 21)
            sub_df = df.iloc[-span:].copy()
            func(sub_df, **params)
            col = f"{params.get('prefix','ema')}_{span}"
            df.at[idx, col] = sub_df[col].iloc[-1]
        elif name == 'rsi':
            window = params.get('window', 14)
            sub_df = df.iloc[-(window+1):].copy()
            func(sub_df, **params)
            col = f"{params.get('prefix','rsi')}_{window}"
            df.at[idx, col] = sub_df[col].iloc[-1]
        elif name == 'atr':
            window = params.get('window', 14)
            sub_df = df.iloc[-(window+1):].copy()
            func(sub_df, **params)
            col = f"{params.get('prefix','atr')}_{window}"
            df.at[idx, col] = sub_df[col].iloc[-1]
            if 'tr' in sub_df.columns:
                df.at[idx, 'tr'] = sub_df['tr'].iloc[-1]

def update_indicators_at_index(df, idx, indicator_config=None):
    """Compute indicators for a specific index `idx` using only past data up to idx."""
    if indicator_config is None:
        indicator_config = DEFAULT_INDICATOR_CONFIG
    if df is None or df.empty or idx is None or idx < 0 or idx >= len(df):
        return
    for ind in indicator_config:
        func = INDICATOR_REGISTRY[ind['name']]
        name = ind['name']
        params = ind.get('params', {})
        if name == 'bollinger_bands':
            window = params.get('window', 20)
            start = max(0, idx - window + 1)
            sub_df = df.iloc[start: idx + 1].copy()
            func(sub_df, **params)
            prefix = params.get('prefix', 'bb')
            for col in [f"{prefix}_ma", f"{prefix}_upper", f"{prefix}_lower"]:
                df.at[idx, col] = sub_df[col].iloc[-1]
        elif name == 'ema':
            span = params.get('span', 21)
            start = max(0, idx - span + 1)
            sub_df = df.iloc[start: idx + 1].copy()
            func(sub_df, **params)
            col = f"{params.get('prefix','ema')}_{span}"
            df.at[idx, col] = sub_df[col].iloc[-1]
        elif name == 'rsi':
            window = params.get('window', 14)
            start = max(0, idx - (window + 1) + 1)
            sub_df = df.iloc[start: idx + 1].copy()
            func(sub_df, **params)
            col = f"{params.get('prefix','rsi')}_{window}"
            df.at[idx, col] = sub_df[col].iloc[-1]
        elif name == 'atr':
            window = params.get('window', 14)
            start = max(0, idx - (window + 1) + 1)
            sub_df = df.iloc[start: idx + 1].copy()
            func(sub_df, **params)
            col = f"{params.get('prefix','atr')}_{window}"
            df.at[idx, col] = sub_df[col].iloc[-1]
            if 'tr' in sub_df.columns:
                df.at[idx, 'tr'] = sub_df['tr'].iloc[-1]


# ---- Trading Environment ----

class TradingEnvironment(QtCore.QObject):
    bar_advanced = QtCore.pyqtSignal()
    backtest_finished = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.api = None
        self.bot = None
        self._indicator_cache = {}  # symbol -> last index

    def set_api(self, api):
        self.api = api

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
                sig.connect(handler)

        # Initial full-history indicator compute
        for symbol in self.get_asset_list():
            df = self.api.get_asset_data(symbol)
            if df is not None and not df.empty:
                self._compute_indicators(symbol, df)

    def set_bot(self, bot):
        self.bot = bot

    def on_bar_advanced(self, _ignored=None, compute_indicators=True):
        if not self.api.is_running:
            return

        # ✅ Don’t let unexpected objects (like DataFrames) trip the boolean check
        compute = compute_indicators if isinstance(compute_indicators, bool) else False

        for symbol in self.get_asset_list():
            df = self.api.get_asset_data(symbol)
            if compute and df is not None and not df.empty:
                self._compute_indicators(symbol, df)

        if self.bot is not None:
            # Bot runs here at OPEN when called from bar_updated with compute=False
            self.bot.on_bar(self)

    def _recompute_closed_bar_indicators(self, _df=None):
        """Recompute indicators ONLY for the just-closed index on all symbols."""
        if not self.api:
            return
        idx = getattr(self.api, "current_index", None)
        if idx is None:
            return
        for symbol in self.get_asset_list():
            df = self.api.get_asset_data(symbol)
            if df is None or df.empty or idx >= len(df):
                continue
            # Avoid redundant recompute if we've already done this index
            if self._indicator_cache.get(symbol) == idx:
                continue
            update_indicators_at_index(df, idx)
            self._indicator_cache[symbol] = idx

    def get_portfolio(self):
        return self.api.get_portfolio() if self.api else {}

    def get_positions(self):
        return self.api.get_positions() if self.api else {}

    def get_total_pnl(self):
        return self.api.get_total_pnl() if self.api else {'realized': 0.0, 'unrealized': 0.0, 'total': 0.0}

    def get_asset_list(self):
        return self.api.get_asset_list() if self.api else []

    def get_asset_data(self, symbol):
        if not self.api:
            return None
        df = self.api.get_asset_data(symbol)
        if df is None or df.empty or 'close' not in df.columns:
            return df
        return df

    def get_latest_data(self, symbol):
        df = self.api.get_latest_data(symbol) if self.api else pd.DataFrame()
        return df
        
    def get_orders(self, symbol=None):
        if not hasattr(self.api, "trade_log"):
            return []
        if symbol is None:
            return self.api.trade_log
        else:
            return [o for o in self.api.trade_log if o['symbol'] == symbol]

    def place_order(self, order):
        return self.api.place_order(order) if self.api else None

    def get_order_status(self, order_id):
        return self.api.get_order_status(order_id) if self.api else None

    def cancel_order(self, order_id):
        if self.api:
            self.api.cancel_order(order_id)

    def modify_stop_loss(self, symbol, new_value):
        if self.api:
            self.api.modify_stop_loss(symbol, new_value)

    def modify_take_profit(self, symbol, new_value):
        if self.api:
            self.api.modify_take_profit(symbol, new_value)

    def connect(self):
        return self.api.connect() if self.api else None

    def disconnect(self):
        return self.api.disconnect() if self.api else None
        
    def reset_indicators(self):
        self._indicator_cache.clear()
        # Remove indicator columns from all asset DataFrames
        for symbol in self.get_asset_list():
            df = self.api.get_asset_data(symbol)
            if df is not None and not df.empty:
                # Remove indicator columns IN-PLACE
                df.drop(
                    columns=[
                        col for col in df.columns
                        if col.startswith("bb_")
                        or col.startswith("ema_")
                        or col.startswith("rsi_")
                        or col.startswith("atr_")
                        or col.startswith("tr")
                    ],
                    inplace=True, errors="ignore"
                )
        
    def _drop_indicator_columns(self, df):
        cols_to_drop = [col for col in df.columns
                        if col.startswith(("bb_", "ema_", "rsi_", "atr_", "tr"))]
        return df.drop(columns=cols_to_drop, errors='ignore')

    def _has_indicator_columns(self, df):
        return any(col.startswith(("bb_", "ema_", "rsi_", "atr_", "tr")) for col in df.columns)

    def _compute_indicators(self, symbol, df):
        """Full compute once; afterwards, only update the newest row when appending."""
        current_idx = df.index[-1]
        indicator_cols = [col for col in df.columns if col.startswith(("bb_", "ema_", "rsi_", "atr_", "tr"))]
        last_idx = self._indicator_cache.get(symbol)

        if not indicator_cols or df[indicator_cols].isna().all().all():
            for cfg in DEFAULT_INDICATOR_CONFIG:
                INDICATOR_REGISTRY[cfg["name"]](df, **cfg.get("params", {}))
            self._indicator_cache[symbol] = current_idx

        elif last_idx != current_idx:
            update_indicators_last_row(df)
            self._indicator_cache[symbol] = current_idx

