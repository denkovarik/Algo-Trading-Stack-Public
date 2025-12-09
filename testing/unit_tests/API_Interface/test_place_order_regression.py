# Usage: PYTHONPATH=. pytest testing/unit_tests/API_Interface/test_place_order_regression.py

import os
import yaml
import tempfile
import pandas as pd
import pytest

from classes.API_Interface import round_to_tick, Position
from classes.Backtester_Engine import BacktesterEngine

def _iso_minute(i):
    # compact helper for ISO timestamps (UTC)
    return pd.Timestamp("2024-01-01 09:30", tz="UTC") + pd.Timedelta(minutes=i)

def _write_ts_csv(path, rows):
    pd.DataFrame(rows)[
        ["date", "open", "high", "low", "close", "volume", "timestamp"]
    ].to_csv(path, index=False)

def _make_min_config(csv_path, overrides=None):
    o = overrides or {}
    return {
        "initial_cash": o.get("initial_cash", 100000.0),
        "include_slippage": False,
        "commission_per_contract": o.get("commission_per_contract", 0.0),
        "fee_per_fill": o.get("fee_per_trade", 0.0),
        "assets": [
            {
                "symbol": "NQ",
                "type": "futures",
                "file": csv_path,
                "contract_size": 20,
                "tick_size": 0.25,
                "tick_value": 5.0,
                "initial_margin": o.get("initial_margin", 500.0),
                "maintenance_margin": o.get("maintenance_margin", 400.0),
            }
        ],
    }

@pytest.fixture
def engine(tmp_path):
    # Simple 3-bar dataset
    rows = []
    for i in range(3):
        dt = _iso_minute(i)
        rows.append(
            {
                "date": dt.isoformat(),
                "open": 100.0 + i * 1.0,
                "high": 100.5 + i * 1.0,
                "low":  99.5 + i * 1.0,
                "close": 100.25 + i * 1.0,
                "volume": 10 + i,
                "timestamp": int(dt.value // 10**9),
            }
        )
    csv = tmp_path / "nq.csv"
    _write_ts_csv(csv, rows)
    cfg = tmp_path / "cfg.yaml"
    with open(cfg, "w") as f:
        yaml.safe_dump(_make_min_config(str(csv)), f)
    e = BacktesterEngine(str(cfg))
    e.current_index = 0  # start on first bar
    return e

def test_rejects_invalid_qty(engine):
    e = engine
    assert e.place_order({"symbol": "NQ", "side": "buy", "qty": 0}) is None
    assert e.place_order({"symbol": "NQ", "side": "sell", "qty": -1}) is None

def test_explicit_order_id_and_duplicate_skip(engine):
    e = engine
    # First call with provided order_id fills and records order 42
    oid1 = e.place_order({"order_id": 42, "symbol": "NQ", "side": "buy", "qty": 1})
    assert oid1 == 42
    assert 42 in e.orders
    # Second call with same order_id should be skipped and return same id
    oid2 = e.place_order({"order_id": 42, "symbol": "NQ", "side": "buy", "qty": 1})
    assert oid2 == 42
    # Ensure we didn't add a duplicate trade log entry for the skip
    assert sum(t["order_id"] == 42 for t in e.trade_log) == 1

def test_margin_precheck_rejects_when_not_allowed(engine):
    e = engine
    # Make required initial margin enormous so precheck fails
    e.symbol_params["NQ"]["initial_margin"] = 1e9
    oid = e.place_order({"symbol": "NQ", "side": "buy", "qty": 1})
    assert oid is None
    # No order added
    assert len(e.orders) == 0
    # Reset to normal for other tests
    e.symbol_params["NQ"]["initial_margin"] = 500.0

def test_fee_gate_blocks_entries_but_not_exits(engine, tmp_path):
    e = engine
    # Reconfigure with fees high and cash tiny to force entry rejection
    csv2 = tmp_path / "nq2.csv"
    e.get_asset_data("NQ").to_csv(csv2, index=False)
    cfg2 = tmp_path / "cfg2.yaml"
    with open(e.config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["commission_per_contract"] = 100.0
    cfg["fee_per_trade"] = 50.0
    cfg["assets"][0]["file"] = str(csv2)
    cfg["initial_cash"] = 100.0  # tiny cash
    with open(cfg2, "w") as f:
        yaml.safe_dump(cfg, f)
    e.load_backtest_config(str(cfg2))
    e.current_index = 0

    # Entry should be rejected by fee gate (not exit)
    assert e.place_order({"symbol": "NQ", "side": "buy", "qty": 1}) is None

    # Give a position and set cash to 0; exit with triggered_by_sl should still fill
    e.cash = 0.0

    # First, place a buy with fees and margin relaxed to establish a position
    e.symbol_params["NQ"]["commission_per_contract"] = 0.0
    e.symbol_params["NQ"]["fee_per_trade"] = 0.0
    e.symbol_params["NQ"]["initial_margin"] = 1.0   # allow margin preview to pass
    e.cash = 1_000.0                                 # enough equity for margin
    buy_id = e.place_order({"symbol": "NQ", "side": "buy", "qty": 1})
    assert buy_id is not None and e.positions["NQ"].qty == 1

    # Now restore big fees but place an exit flagged as protective (bypasses fee gate)
    e.cash = 0.0
    e.symbol_params["NQ"]["commission_per_contract"] = 100.0
    e.symbol_params["NQ"]["fee_per_trade"] = 50.0
    sell_id = e.place_order(
        {
            "symbol": "NQ",
            "side": "sell",
            "qty": 1,
            "triggered_by_sl": True,
            "stop_loss": 100.0,
        }
    )
    assert sell_id is not None
    assert e.positions["NQ"].qty == 0  # exited

def test_rejected_when_out_of_range_or_no_price_data(engine):
    e = engine
    # Set index past the last bar; order should be created but status 'rejected'
    e.current_index = len(e.get_asset_data("NQ"))
    oid = e.place_order({"symbol": "NQ", "side": "buy", "qty": 1})
    assert oid is not None
    assert e.orders[oid].status == "rejected"

def test_successful_fill_updates_position_cash_margin_and_stops(engine):
    e = engine
    params = e.symbol_params["NQ"]
    tick = params["tick_size"]
    im = params["initial_margin"]

    e.cash = 10000.0
    e.symbol_params["NQ"]["commission_per_contract"] = 2.0
    e.symbol_params["NQ"]["fee_per_trade"] = 3.0

    cash0 = e.cash  # <-- baseline for cash assertion

    # BUY 2 with protective levels
    oid = e.place_order(
        {
            "symbol": "NQ",
            "side": "buy",
            "qty": 2,
            "stop_loss": 100.12,
            "take_profit": 101.13,
        }
    )
    assert oid is not None
    pos = e.positions["NQ"]
    assert pos.qty == 2
    # Commission+fee deducted (realized on entry is 0)
    expected_cost = 2 * 2.0 + 3.0
    assert e.cash == pytest.approx(cash0 - expected_cost)
    # Margin committed after fill
    assert e.used_margin == 2 * im
    # Protective levels rounded to tick
    assert pos.stop_loss_price == round_to_tick(100.12, tick)
    assert pos.take_profit == round_to_tick(101.13, tick)
    # Trade log entry present with expected fields
    tl = e.trade_log[-1]
    for k in (
        "order_id",
        "symbol",
        "side",
        "qty",
        "fill_price",
        "fill_time",
        "realized_pnl_change",
        "net_realized_from_cash",
        "position_after_fill",
        "commission",
        "fee",
    ):
        assert k in tl

def test_reducing_position_releases_margin_and_clears_stops_when_flat(engine):
    e = engine
    im = e.symbol_params["NQ"]["initial_margin"]

    # Establish long 2 (no fees for cleaner math)
    e.symbol_params["NQ"]["commission_per_contract"] = 0.0
    e.symbol_params["NQ"]["fee_per_trade"] = 0.0
    e.place_order({"symbol": "NQ", "side": "buy", "qty": 2, "stop_loss": 99.9, "take_profit": 110.1})
    assert e.positions["NQ"].qty == 2
    assert e.used_margin == 2 * im

    # SELL 1 → reduce
    e.place_order({"symbol": "NQ", "side": "sell", "qty": 1})
    assert e.positions["NQ"].qty == 1
    assert e.used_margin == 1 * im

    # SELL remaining 1 → flat; stops should clear to None; margin to zero
    e.place_order({"symbol": "NQ", "side": "sell", "qty": 1})
    pos = e.positions["NQ"]
    assert pos.qty == 0
    assert pos.stop_loss_price is None
    assert pos.take_profit is None
    assert e.used_margin == 0

