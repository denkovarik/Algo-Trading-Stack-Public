# Usage:
#   PYTHONPATH=. pytest testing/unit_tests/API_Interface/test_margin_intrabar_worstcase.py -q

import os
import csv
import tempfile
from datetime import datetime, timezone
import yaml

from classes.Backtester_Engine import BacktesterEngine


def _mk_csv(dirpath, rows):
    fp = os.path.join(dirpath, "x_tradestation.csv")
    with open(fp, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["Open", "High", "Low", "Close", "TotalVolume", "TimeStamp"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return fp


def _mk_cfg(tmpdir, csv_path, initial_cash=420.0, im=400.0, mm=400.0):
    cfg = {
        "initial_cash": float(initial_cash),
        "include_slippage": False,
        "commission_per_contract": 0.0,
        "fee_per_trade": 0.0,
        "assets": [{
            "symbol": "X",
            "type": "futures",
            "file": csv_path,
            "contract_size": 1,
            "tick_size": 1.0,
            "tick_value": 10.0,
            "initial_margin": float(im),
            "maintenance_margin": float(mm),
        }]
    }
    yml = os.path.join(tmpdir, "cfg.yaml")
    with open(yml, "w") as f:
        yaml.safe_dump(cfg, f)
    return yml


def test_margin_intrabar_trigger_long_low_breaches():
    """
    Close is safe (no margin call), but LOW breaches → margin call should trigger.
    Setup:
      - tick_size=1, tick_value=10
      - Buy 1 @ 100
      - Close=100 → PnL=0 → Equity = 420 >= MM=400  (no call on close)
      - Low=95   → PnL=-50 → Equity = 370 < MM=400  (should trigger on intrabar worst-case)
    """
    tmp = tempfile.mkdtemp(prefix="bt_mm_worstcase_long_")
    csv_path = _mk_csv(tmp, [
        dict(Open=100.0, High=101.0, Low=95.0, Close=100.0, TotalVolume=1,
             TimeStamp=datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc).isoformat()),
        dict(Open=100.0, High=101.0, Low=99.0, Close=100.0, TotalVolume=1,
             TimeStamp=datetime(2024, 1, 1, 9, 31, tzinfo=timezone.utc).isoformat()),
    ])
    yml = _mk_cfg(tmp, csv_path, initial_cash=420.0, im=400.0, mm=400.0)

    eng = BacktesterEngine(yml)
    eng.connect()

    # Enter long 1 @ bar 0 OPEN (engine fills entries at OPEN)
    oid = eng.place_order({"symbol": "X", "side": "buy", "qty": 1})
    assert oid is not None

    # Sanity: at close-based valuation, equity should be >= maintenance
    # (so old close-only logic would NOT have called margin)
    port = eng.get_portfolio()
    assert port["total_equity"] >= 400.0

    # New behavior: enforce margin using worst-case intrabar (LOW for long)
    eng.enforce_maintenance_margin()

    # Should have liquidated to flat
    pos = eng.get_positions().get("X", {})
    assert pos.get("qty", 0) == 0, "Expected position to be liquidated due to LOW-based breach"


def test_margin_intrabar_trigger_short_high_breaches():
    """
    Close is safe, but HIGH breaches → margin call should trigger for short.
    Setup:
      - tick_size=1, tick_value=10
      - Sell 1 @ 100
      - Close=100 → PnL=0 → Equity = 420 >= MM=400  (no call on close)
      - High=105 → PnL=-50 → Equity = 370 < MM=400  (should trigger on intrabar worst-case)
    """
    tmp = tempfile.mkdtemp(prefix="bt_mm_worstcase_short_")
    csv_path = _mk_csv(tmp, [
        dict(Open=100.0, High=105.0, Low=99.0, Close=100.0, TotalVolume=1,
             TimeStamp=datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc).isoformat()),
        dict(Open=100.0, High=101.0, Low=99.0, Close=100.0, TotalVolume=1,
             TimeStamp=datetime(2024, 1, 1, 9, 31, tzinfo=timezone.utc).isoformat()),
    ])
    yml = _mk_cfg(tmp, csv_path, initial_cash=420.0, im=400.0, mm=400.0)

    eng = BacktesterEngine(yml)
    eng.connect()

    # Enter short 1 @ bar 0 OPEN
    oid = eng.place_order({"symbol": "X", "side": "sell", "qty": 1})
    assert oid is not None

    # Sanity: at close-based valuation, equity >= MM
    port = eng.get_portfolio()
    assert port["total_equity"] >= 400.0

    # New behavior: enforce margin using worst-case intrabar (HIGH for short)
    eng.enforce_maintenance_margin()

    # Should be flat after liquidation
    pos = eng.get_positions().get("X", {})
    assert pos.get("qty", 0) == 0, "Expected position to be liquidated due to HIGH-based breach"

