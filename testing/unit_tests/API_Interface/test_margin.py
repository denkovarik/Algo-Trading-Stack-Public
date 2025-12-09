# Usage: PYTHONPATH=. pytest testing/unit_tests/API_Interface/test_margin.py

import pytest
import yaml
import tempfile
import os

from classes.API_Interface import Position
from classes.Backtester_Engine import BacktesterEngine


# Minimal config for test (British Pound Futures)
MINIMAL_CONFIG = {
    "initial_cash": 100000,
    "assets": [
        {
            "symbol": "6B=F",
            "type": "futures",
            "file": "",  # not loading data
            "contract_size": 62500,
            "tick_size": 0.0001,
            "tick_value": 6.25,
            "initial_margin": 2000,
            "maintenance_margin": 1800,
            "slippage_pct": 0.0007,
            "currency": "USD",
            "exchange": "CME"
        }
    ]
}

@pytest.fixture
def engine():
    # Write config to a temp file
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix='.yaml') as fp:
        yaml.safe_dump(MINIMAL_CONFIG, fp)
        fp.flush()
        e = BacktesterEngine(fp.name)
        yield e
    os.unlink(fp.name)


def test_margin_increase_and_release(engine):
    symbol = "6B=F"
    initial_margin = engine.symbol_params[symbol]["initial_margin"]

    # Open 2 contracts long
    allowed, inc, rel = engine.check_and_update_margin(symbol, "buy", 2, initial_margin, commit=True)
    assert allowed
    assert inc == 2 * initial_margin
    assert rel == 0
    assert engine.used_margin == 2 * initial_margin

    # Increase by 1 more
    allowed, inc, rel = engine.check_and_update_margin(symbol, "buy", 1, initial_margin, commit=True)
    assert allowed
    assert inc == 1 * initial_margin
    assert rel == 0
    assert engine.used_margin == 3 * initial_margin

    # Prepare a position object so prev_qty is known for the reduction
    pos = engine.positions.get(symbol)
    if not pos:
        pos = Position(symbol, engine.symbol_params[symbol]["contract_size"])
        engine.positions[symbol] = pos
    pos.qty = 3  # simulate current position

    # Reduce by 2 -> margin release of 2 * initial_margin; commit=True applies it
    allowed, inc, rel = engine.check_and_update_margin(symbol, "sell", 2, initial_margin, commit=True)
    assert allowed
    assert inc == 0
    assert rel == 2 * initial_margin
    assert engine.used_margin == 1 * initial_margin   # <-- no manual subtraction

    # Close remaining 1 -> release final margin; commit=True applies it
    pos.qty = 1
    allowed, inc, rel = engine.check_and_update_margin(symbol, "sell", 1, initial_margin, commit=True)
    assert allowed
    assert inc == 0
    assert rel == 1 * initial_margin
    assert engine.used_margin == 0                     # <-- no manual subtraction


def test_margin_rejection(engine):
    symbol = "6B=F"
    initial_margin = engine.symbol_params[symbol]["initial_margin"]
    # Simulate no cash for margin
    engine.cash = 0
    allowed, margin_inc, margin_rel = engine.check_and_update_margin(
                                                                        symbol, "buy", 1,
                                                                        initial_margin, 
                                                                        commit=True
                                                                    )
    assert not allowed
    assert margin_inc == 1 * initial_margin
    assert margin_rel == 0

def test_margin_preview(engine):
    symbol = "6B=F"
    initial_margin = engine.symbol_params[symbol]["initial_margin"]
    # Test dry-run (commit=False)
    allowed, margin_inc, margin_rel = engine.check_and_update_margin(
                                                                        symbol, "buy", 1, 
                                                                        initial_margin, 
                                                                        commit=False
                                                                    )
    assert allowed
    assert margin_inc == 1 * initial_margin
    assert margin_rel == 0
    # Should NOT change used_margin
    assert engine.used_margin == 0

