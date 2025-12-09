# Usage: PYTHONPATH=. pytest testing/unit_tests/API_Interface/test_portfolio_and_replay.py

import pytest, os, pandas as pd, tempfile, shutil
from datetime import datetime, timezone
from classes.Backtester_Engine import BacktesterEngine


def _iso(y,m,d,hh,mm):
    return datetime(y,m,d,hh,mm,tzinfo=timezone.utc).isoformat()

def _csv(path):
    rows = [
        {
            "Open":100.0,"High":101.0,"Low": 99.0,"Close":100.5,
            "TotalVolume":10,"TimeStamp":_iso(2024,1,1,9,30)
        },
        {
            "Open":101.0,"High":102.0,"Low":100.0,"Close":101.5,
            "TotalVolume":12,"TimeStamp":_iso(2024,1,1,9,31)
        },
        {
            "Open":102.0,"High":103.0,"Low":101.0,"Close":102.5,
            "TotalVolume":11,"TimeStamp":_iso(2024,1,1,9,32)
        },
        {
            "Open":103.0,"High":104.0,"Low":102.0,"Close":103.5,
            "TotalVolume":10,"TimeStamp":_iso(2024,1,1,9,33)
        },
    ]
    pd.DataFrame(rows)[
        ["Open", "High", "Low", "Close", "TotalVolume", "TimeStamp"]
    ].to_csv(
        path,
        index=False
    )

def _yml(path, csv):
    with open(path,"w") as f:
        f.write(
            f"""
                initial_cash: 100000
                include_slippage: false
                commission_per_contract: 0.0
                fee_per_trade: 0.0
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

@pytest.fixture
def engine(tmp_path):
    csv = tmp_path/"ts.csv"; _csv(csv)
    yml = tmp_path/"cfg.yaml"; _yml(yml, csv)
    return BacktesterEngine(config_path=str(yml))

def _run_to_idx(e, idx):
    while e.current_index < idx:
        e.step()

def test_portfolio_math_and_total_pnl(engine):
    e = engine
    # Enter, then move 2 bars to change unrealized
    e.place_order({"symbol":"NQ","side":"buy","qty":1})
    _run_to_idx(e, 2)
    totals = e.get_total_pnl()
    port = e.get_portfolio()

    # realized == cash - initial_cash; 
    # total == realized + unrealized; available == equity - used_margin
    assert pytest.approx(totals["realized"]) == e.cash - e.initial_cash
    assert pytest.approx(totals["total"]) == totals["realized"] + totals["unrealized"]
    assert pytest.approx(port["total_equity"]) == e.cash + totals["unrealized"]
    assert pytest.approx(port["available_equity"]) == port["total_equity"] - port["used_margin"]

def test_close_all_positions_fills_at_close(engine):
    e = engine
    e.place_order({"symbol":"NQ","side":"buy","qty":1})
    e.current_index = 2
    e.close_all_positions()
    # After close_all_positions, qty zero and last fill uses bar CLOSE
    assert e.positions["NQ"].qty == 0
    assert e.trade_log[-1]["fill_price"] == 102.5

def test_resimulate_to_index_is_deterministic(engine):
    e = engine

    # Place buys deterministically based solely on the emitting bar index.
    def on_bar_updated(_df):
        idx = e.current_index
        if idx == 1:
            e.place_order({"symbol": "NQ", "side": "buy", "qty": 1})
        if idx == 2:
            e.place_order({"symbol": "NQ", "side": "buy", "qty": 1})

    e.bar_updated.connect(on_bar_updated)

    # Drive to index 2 the first time: emits at idx=1 and idx=2 → two buys placed.
    while e.current_index < 2:
        e.step()

    state_a = (
        e.cash,
        e.used_margin,
        e.positions.get("NQ", None).qty if e.positions.get("NQ") else 0,
        len(e.trade_log),
    )

    # Replay deterministically to the same target index; callback fires again.
    e._resimulate_to_index(2)

    state_b = (
        e.cash,
        e.used_margin,
        e.positions.get("NQ", None).qty if e.positions.get("NQ") else 0,
        len(e.trade_log),
    )

    assert state_a == state_b

