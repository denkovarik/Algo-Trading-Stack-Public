# Usage: PYTHONPATH=. pytest testing/unit_tests/API_Interface/test_engine_fill_and_protective.py
import pytest, tempfile, shutil, os, pandas as pd, yaml
import pytz
from datetime import datetime, timezone
from classes.API_Interface import round_to_tick
from classes.Backtester_Engine import BacktesterEngine


def _iso(y, m, d, hh, mm):
    dt = datetime(y, m, d, hh, mm, tzinfo=timezone.utc)
    return dt.astimezone(pytz.timezone('America/New_York')).isoformat()


def _write_ts_csv(path, rows):
    pd.DataFrame(rows)[
        ["Open", "High", "Low", "Close", "TotalVolume", "TimeStamp"]
    ].to_csv(path, index=False)


def _write_yaml(path, csv_path, **ovr):
    cfg = f"""
            initial_cash: {ovr.get("initial_cash", 100000)}
            include_slippage: {str(ovr.get("include_slippage", False)).lower()}
            slippage_ticks: {ovr.get("slippage_ticks", 0)}
            slippage_pct: {ovr.get("slippage_pct", 0.0)}
            commission_per_contract: {ovr.get("commission_per_contract", 0.0)}
            fee_per_trade: {ovr.get("fee_per_trade", 0.0)}
            skip_synthetic_open_bars: true
            assets:
              - symbol: NQ
                type: futures
                file: "{csv_path}"
                contract_size: 20
                tick_size: 0.25
                tick_value: 5.0
                initial_margin: 500.0
                maintenance_margin: 400.0
            """
    with open(path, "w") as f:
        f.write(cfg)


@pytest.fixture
def engine_base(tmp_path):
    csv = tmp_path / "ts.csv"
    rows = [
        {
            "Open": 100.0, "High": 102.0, "Low": 99.5, "Close": 101.0,
            "TotalVolume": 10, "TimeStamp": _iso(2024, 1, 1, 14, 30)  # 9:30 AM ET
        },
        {
            "Open": 104.0, "High": 105.0, "Low": 103.5, "Close": 104.5,
            "TotalVolume": 12, "TimeStamp": _iso(2024, 1, 1, 14, 31)
        },
        {
            "Open": 104.5, "High": 105.0, "Low": 104.0, "Close": 104.25,
            "TotalVolume": 11, "TimeStamp": _iso(2024, 1, 1, 14, 32)
        },
    ]
    _write_ts_csv(csv, rows)
    yml = tmp_path / "cfg.yaml"
    _write_yaml(yml, csv, include_slippage=False, commission_per_contract=0.0, fee_per_trade=0.0)
    return BacktesterEngine(config_path=str(yml))


def test_get_fill_price_branches_and_slippage(engine_base):
    e = engine_base
    df = e.get_asset_data("NQ")
    e.current_index = 0

    # Market entry = OPEN
    assert e.get_fill_price({"symbol": "NQ", "side": "buy"}, df, 0) == 100.0

    # SL/TP use provided level then rounding
    p = e.get_fill_price(
        {"symbol": "NQ", "side": "sell", "stop_loss": 100.12, "triggered_by_sl": True}, df, 0
    )
    assert p == round_to_tick(100.12, 0.25)  # 100.0

    p = e.get_fill_price(
        {"symbol": "NQ", "side": "sell", "take_profit": 100.13, "triggered_by_tp": True}, df, 0
    )
    assert p == round_to_tick(100.13, 0.25)  # 100.25

    # Forced liquidation = CLOSE
    p = e.get_fill_price(
        {"symbol": "NQ", "side": "sell", "forced_liquidation": True, "closeout": True}, df, 0
    )
    assert p == 101.0

    # Slippage precedence (ticks over pct), then rounding
    # Re-load with slippage enabled
    tmp_dir = tempfile.mkdtemp()
    try:
        csv2 = os.path.join(tmp_dir, "ts.csv")
        pd.DataFrame(df).to_csv(csv2, index=False)
        yml2 = os.path.join(tmp_dir, "cfg.yaml")
        with open(e.config_path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg["include_slippage"] = True
        cfg["slippage_ticks"] = 2
        cfg["slippage_pct"] = 0.5  # ignored
        cfg["assets"][0]["file"] = csv2
        with open(yml2, "w") as f:
            yaml.safe_dump(cfg, f)

        e.load_backtest_config(yml2)
        e.current_index = 0
        e._build_symbol_params_and_load_assets(e.config)

        # volume=10, avg≈11 → volume_penalty≈2.2
        # slip_amt = 2 * 2.2 * 0.25 ≈ 1.1
        # 100.0 + 1.1 → 101.1 → round_to_tick → 101.0
        assert e.get_fill_price({"symbol": "NQ", "side": "buy"}, e.get_asset_data("NQ"), 0) == 101.0
        assert e.get_fill_price({"symbol": "NQ", "side": "sell"}, e.get_asset_data("NQ"), 0) == 99.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_live_snapshot_and_mark_price(engine_base):
    e = engine_base
    df = e.get_asset_data("NQ")
    e.current_index = 0
    e._in_open_phase = True
    snap = e._live_snapshot_df(df)
    assert len(snap) == 1
    assert float(snap.iloc[-1]["open"]) == 100.0
    assert (
        pd.isna(snap.iloc[-1]["high"])
        and pd.isna(snap.iloc[-1]["low"])
        and pd.isna(snap.iloc[-1]["volume"])
    )
    assert float(snap.iloc[-1]["close"]) == 100.0  # open_only mode
    assert e._get_mark_price("NQ") == 100.0
    e._in_open_phase = False
    assert e._get_mark_price("NQ") == 101.0  # prefers CLOSE when not open phase


def test_commission_fee_and_exit_gating(engine_base):
    # Reconfigure with nonzero fees
    e = engine_base
    tmp_dir = tempfile.mkdtemp()
    try:
        csv = os.path.join(tmp_dir, "ts.csv")
        e.get_asset_data("NQ").to_csv(csv, index=False)
        yml = os.path.join(tmp_dir, "cfg.yaml")
        with open(yml, "w") as f:
            f.write(
                f"""
                initial_cash: 5
                include_slippage: false
                commission_per_contract: 2.0
                fee_per_trade: 4.0
                assets:
                  - symbol: NQ
                    type: futures
                    file: "{csv}"
                    contract_size: 20
                    tick_size: 0.25
                    tick_value: 5.0
                    initial_margin: 500.0
                    maintenance_margin: 400.0
                """
            )
        e.load_backtest_config(yml)
        # Entry should be rejected due to insufficient cash for fees (2*qty + 4)
        oid = e.place_order({"symbol": "NQ", "side": "buy", "qty": 1})
        assert oid is None
        # Put on a position by giving cash, then test exit bypass
        e.cash = 1_000.0
        oid2 = e.place_order({"symbol": "NQ", "side": "buy", "qty": 1})
        assert oid2 is not None
        # Force exit with zero cash → should still succeed (exits bypass fee gate)
        e.cash = 0.0
        oid3 = e.place_order({
            "symbol": "NQ",
            "side": "sell",
            "qty": 1,
            "triggered_by_sl": True,
            "stop_loss": 100.0
        })
        assert oid3 is not None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_maintenance_liquidation_at_close(engine_base):
    e = engine_base
    e.place_order({"symbol": "NQ", "side": "buy", "qty": 1})
    e.current_index = 1
    # Force maintenance call
    e.symbol_params["NQ"]["maintenance_margin"] = 999999.0
    e.enforce_maintenance_margin()
    assert e.positions["NQ"].qty == 0
    # Last fill at CLOSE of bar 1 (104.5)
    assert e.trade_log[-1]["fill_price"] == 104.5


def test_intrabar_protective_orders_gap_open_and_policy(engine_base):
    e = engine_base
    sym = "NQ"
    e.place_order({"symbol": sym, "side": "buy", "qty": 1})
    pos = e.positions[sym]
    # Gap-up to t1 open=104.0; TP at open should fill at OPEN
    pos.take_profit = 104.0
    e.current_index = 1
    bar = e.get_asset_data(sym).iloc[e.current_index]
    closed = e._process_protective_orders(sym, pos, bar, policy="worst_case")
    assert closed and e.positions[sym].qty == 0
    # Both SL and TP within H/L → worst_case = SL first
    # Reload minimal 1-row dataset with wide H/L
    tmp_dir = tempfile.mkdtemp()
    try:
        csv = os.path.join(tmp_dir, "ts.csv")
        rows = [
            {
                "Open": 100.0,
                "High": 105.0,
                "Low": 95.0,
                "Close": 100.0,
                "TotalVolume": 10,
                "TimeStamp": _iso(2024, 1, 2, 9, 30)
            }
        ]
        _write_ts_csv(csv, rows)
        yml = os.path.join(tmp_dir, "cfg.yaml")
        _write_yaml(yml, csv, commission_per_contract=0.0, fee_per_trade=0.0)
        e.load_backtest_config(yml)
        e.place_order({"symbol": sym, "side": "buy", "qty": 1})
        pos = e.positions[sym]
        pos.stop_loss_price = 96.0
        pos.take_profit = 104.0
        e.current_index = 0
        bar = e.get_asset_data(sym).iloc[0]
        e._process_protective_orders(sym, pos, bar, policy="worst_case")
        assert e.positions[sym].qty == 0
        assert e.trade_log[-1]["fill_price"] == round_to_tick(96.0, 0.25)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
