# PYTHONPATH=. pytest testing/unit_tests/Trading_Environment/test_env_core.py

import pytest
import pandas as pd
from PyQt5 import QtCore
from classes.Trading_Environment import TradingEnvironment


def _mk_df(n=40, start=100.0):
    # price scaffolding
    close = pd.Series([start + i*0.5 for i in range(n)], name="close")
    high = close + 0.2
    low = close - 0.2
    df = pd.DataFrame({"close": close, "open": close, "high": high, "low": low, "volume": 1})

    # add time columns to match the engine’s data contract
    # Use 9:30am New York (14:30 UTC) as the session start and step by 1 minute.
    date = pd.date_range("2024-01-02 14:30:00Z", periods=n, freq="min")
    df["date"] = pd.to_datetime(date, utc=True)
    df["timestamp"] = df["date"].astype("int64") // 10**9

    return df.reset_index(drop=True)


class DummyAPI(QtCore.QObject):
    bar_advanced = QtCore.pyqtSignal()
    bar_updated = QtCore.pyqtSignal(pd.DataFrame)
    bar_closed = QtCore.pyqtSignal(pd.DataFrame)
    backtest_finished = QtCore.pyqtSignal()

    def __init__(self, df_map):
        super().__init__()
        self._df_map = df_map
        self.is_running = False
        self.current_index = len(next(iter(df_map.values()))) - 1
        self.trade_log = []
        self._connected = False

    # --- methods TradingEnvironment uses ---
    def get_asset_list(self):
        return list(self._df_map.keys())

    def get_asset_data(self, symbol):
        return self._df_map[symbol]

    def get_latest_data(self, symbol, window_size=None):
        # For tests we just return the whole df; env handles open/closed itself.
        # Accept window_size for compatibility with env passthrough.
        return self._df_map[symbol]

    def get_portfolio(self):
        return {"total_equity": 123.45}

    def get_positions(self):
        return {"SYM": {"qty": 1}}

    def get_total_pnl(self):
        return {"realized": 1.0, "unrealized": 2.0, "total": 3.0}

    def place_order(self, order):
        oid = len(self.trade_log) + 1
        self.trade_log.append({**order, "order_id": oid})
        return oid

    def get_order_status(self, oid):
        return "filled"

    def cancel_order(self, oid):
        self.trade_log.append({"cancel": oid})

    def modify_stop_loss(self, symbol, v): ...
    def modify_take_profit(self, symbol, v): ...
    def connect(self):
        self._connected = True
    def disconnect(self):
        self._connected = False

@pytest.fixture
def env_with_api():
    df = _mk_df(60)  # long enough for all indicators
    api = DummyAPI({"NQ": df})
    env = TradingEnvironment()
    env.set_api(api)
    return env, api

def _has_any_indicators(df):
    return any(c.startswith(("bb_", "ema_", "rsi_", "atr_", "tr")) for c in df.columns)

def test_set_api_triggers_initial_indicator_full_compute(env_with_api):
    env, api = env_with_api
    df = api.get_asset_data("NQ")
    assert _has_any_indicators(df), "Initial full-history compute should add indicator columns"

def test_on_bar_advanced_updates_only_last_row_when_running(env_with_api):
    env, api = env_with_api
    api.is_running = True

    df = api.get_asset_data("NQ")

    # Take snapshots at the current last row (will become prev after append)
    old_last_idx = df.index[-1]
    prev_vals_before = df.filter(regex=r'^(bb_|ema_|rsi_|atr_|tr)').iloc[old_last_idx].copy()

    # Append a NEW bar; indicators should recompute only for the new last row
    new_close = df.loc[old_last_idx, "close"] + 1.0
    new_time = df.loc[old_last_idx, "date"] + pd.Timedelta(minutes=1)
    new_row = {
        "close": new_close,
        "open": new_close,
        "high": new_close + 0.2,
        "low": new_close - 0.2,
        "volume": 1,
        "date": new_time,
        "timestamp": int(new_time.value // 10**9),
    }
    df.loc[old_last_idx + 1] = new_row
    api.current_index = old_last_idx + 1  # advance the engine clock

    # Before compute, last row has no indicator columns yet (or NaNs)
    last_vals_before = df.filter(regex=r'^(bb_|ema_|rsi_|atr_|tr)').iloc[-1].copy()

    env.on_bar_advanced(compute_indicators=True)

    # Previous row should remain the same
    prev_vals_after = df.filter(regex=r'^(bb_|ema_|rsi_|atr_|tr)').iloc[old_last_idx]
    pd.testing.assert_series_equal(prev_vals_before.fillna(pd.NA), prev_vals_after.fillna(pd.NA))

    # New last row should now have updated indicator values (not all identical)
    last_vals_after = df.filter(regex=r'^(bb_|ema_|rsi_|atr_|tr)').iloc[-1]
    assert last_vals_after.notna().any()
    # And at least something changed compared to the pre-compute snapshot
    assert any((last_vals_after != last_vals_before) & last_vals_after.notna())


def test_on_bar_advanced_live_path_does_not_recompute_but_calls_bot(env_with_api):
    env, api = env_with_api
    api.is_running = True

    class DummyBot:
        def __init__(self):
            self.calls = 0
        def on_bar(self, e):
            self.calls += 1

    bot = DummyBot()
    env.set_bot(bot)

    df = api.get_asset_data("NQ")
    last_idx = df.index[-1]

    # Record current indicator snapshot
    snap_before = df.filter(regex=r'^(bb_|ema_|rsi_|atr_|tr)').iloc[last_idx].copy()

    # Simulate bar_updated → env wires it to on_bar_advanced(compute_indicators=False)
    api.bar_updated.emit(df)

    # Indicators unchanged but bot called
    snap_after = df.filter(regex=r'^(bb_|ema_|rsi_|atr_|tr)').iloc[last_idx]
    pd.testing.assert_series_equal(snap_before.fillna(pd.NA), snap_after.fillna(pd.NA))
    assert bot.calls == 1

def test_recompute_closed_bar_indicators_respects_cache(env_with_api, monkeypatch):
    env, api = env_with_api
    df = api.get_asset_data("NQ")

    # Pick a mid index to simulate just-closed bar
    api.current_index = len(df) // 2

    # Spy on the actual alias used by _recompute_closed_bar_indicators
    import classes.Trading_Environment as mod
    calls = {"n": 0}
    orig = mod._router_update_at_index
    def spy(df_arg, idx_arg, indicator_config, registry):
        calls["n"] += 1
        return orig(df_arg, idx_arg, indicator_config, registry)
    monkeypatch.setattr(mod, "_router_update_at_index", spy)

    # First recompute → should call once per symbol (1 symbol here)
    env._recompute_closed_bar_indicators()
    assert calls["n"] == 1

    # Second recompute at the same index → cache suppresses extra work
    env._recompute_closed_bar_indicators()
    assert calls["n"] == 1

def test_reset_indicators_drops_columns_and_clears_cache(env_with_api):
    env, api = env_with_api
    df = api.get_asset_data("NQ")
    assert _has_any_indicators(df)

    env.reset_indicators()
    assert not _has_any_indicators(df)

    # After reset, enable running so on_bar_advanced triggers recompute
    api.is_running = True
    env.on_bar_advanced(compute_indicators=True)

    assert _has_any_indicators(df)


def test_proxies_and_helpers(env_with_api):
    env, api = env_with_api

    # portfolio / positions / pnl
    assert env.get_portfolio()["total_equity"] == 123.45
    assert env.get_positions()["SYM"]["qty"] == 1
    pnl = env.get_total_pnl()
    assert pnl["total"] == 3.0

    # orders / filtering
    oid = env.place_order({"symbol": "NQ", "side": "buy", "qty": 1})
    assert oid == 1
    assert env.get_order_status(oid) == "filled"
    assert len(env.get_orders()) == 1
    assert len(env.get_orders("NQ")) == 1
    env.cancel_order(oid)  # just exercise path
    env.modify_stop_loss("NQ", 100.0)
    env.modify_take_profit("NQ", 110.0)

    # data passthroughs
    assert env.get_asset_list() == ["NQ"]
    assert isinstance(env.get_asset_data("NQ"), pd.DataFrame)
    assert isinstance(env.get_latest_data("NQ"), pd.DataFrame)

    # connect/disconnect passthroughs
    env.connect()
    env.disconnect()

