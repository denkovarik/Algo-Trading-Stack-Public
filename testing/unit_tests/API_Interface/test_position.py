# Usage: PYTHONPATH=. pytest testing/unit_tests/API_Interface/test_position.py


import pytest
from classes.API_Interface import Position

def test_position_overflip_long_to_short():
    # Example contract parameters
    symbol = 'TEST'
    contract_size = 1
    tick_size = 1.0
    tick_value = 1.0
    
    # 1. Open a long position: Buy 1 @ 100
    pos = Position(symbol, contract_size)
    pos.update_on_fill(fill_price=100.0, fill_qty=1, side='buy', tick_size=tick_size, tick_value=tick_value)
    assert pos.qty == 1
    assert pos.avg_entry_price == 100.0
    assert pos.realized_pnl == 0.0

    # 2. Over-flip: Sell 2 @ 110 (closes long, opens new short)
    pos.update_on_fill(fill_price=110.0, fill_qty=2, side='sell', tick_size=tick_size, tick_value=tick_value)
    # Should be short 1 contract now
    assert pos.qty == -1

    # Realized PnL for the closed leg: (110-100)/tick_size * tick_value * 1 = 10.0
    assert pos.realized_pnl == pytest.approx(10.0)

    # The avg_entry_price for the new short leg should be 110.0 (the entry price of the over-flip leg)
    assert pos.avg_entry_price == 110.0

    # 3. Now flip back: Buy 2 @ 90 (closes short, opens new long)
    pos.update_on_fill(fill_price=90.0, fill_qty=2, side='buy', tick_size=tick_size, tick_value=tick_value)
    # Should be long 1 contract now
    assert pos.qty == 1

    # Realized PnL for the closed short: (110-90)/tick_size * tick_value * 1 = 20.0
    # So, cumulative realized PnL should be 10.0 + 20.0 = 30.0
    assert pos.realized_pnl == pytest.approx(30.0)

    # The avg_entry_price for the new long leg should be 90.0
    assert pos.avg_entry_price == 90.0

if __name__ == "__main__":
    test_position_overflip_long_to_short()
    print("Test passed: Position over-flip logic works as expected.")

