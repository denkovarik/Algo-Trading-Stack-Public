# Usage: PYTHONPATH=. pytest testing/functional_tests/risk_tests.py

import pytest
import random
import numpy as np
from classes.Backtester_Engine import BacktesterEngine
from classes.Trading_Environment import TradingEnvironment


def compute_safe_position_size(
    side, entry_price, atr, atr_multiple, equity, risk_percent,
    contract_size, tick_size, tick_value, slippage_pct=0.001, initial_margin=None,
    commission_per_contract=0.0, fee_per_trade=0.0
):
    risk_amount = equity * risk_percent

    # Calculate stop loss distance in price terms
    stop_distance_price = atr * atr_multiple

    # Adjust entry fill price by slippage (percentage-based)
    slippage_price = entry_price * slippage_pct
    entry_fill_price = entry_price + slippage_price if side == "long" else entry_price - slippage_price

    # Determine stop-loss price
    stop_loss_price = entry_price - stop_distance_price if side == "long" else entry_price + stop_distance_price

    # Calculate ticks at risk based on adjusted entry price and stop-loss
    ticks_at_risk = abs(entry_fill_price - stop_loss_price) / tick_size

    # Dollar risk per contract (tick_value usually includes contract size)
    risk_per_contract = ticks_at_risk * tick_value

    # Entry and exit commission plus trade fee
    total_commission_and_fee = (2 * commission_per_contract) + fee_per_trade

    # Total risk per contract
    total_risk_per_contract = risk_per_contract + total_commission_and_fee

    # Prevent invalid trades
    if total_risk_per_contract <= 0 or total_risk_per_contract > risk_amount:
        return 0, stop_loss_price, total_risk_per_contract

    # Compute safe quantity based on risk
    qty = int(risk_amount // total_risk_per_contract)

    # Consider margin limit
    if initial_margin:
        qty_by_margin = int(equity // initial_margin)
        qty = min(qty, qty_by_margin)

    return qty, stop_loss_price, total_risk_per_contract



@pytest.mark.parametrize("seed", range(10))
def test_manual_long_order_stop_loss(seed):
    random.seed(seed)
    config_path = "testing/functional_tests/configs/backtest_config_2.yaml"
    engine = BacktesterEngine(config_path=config_path)
    env = TradingEnvironment()
    env.set_api(engine)

    symbol = "CL=F"
    df = env.get_asset_data(symbol)
    assert df is not None and not df.empty

    min_index = 21
    max_index = len(df) - 25
    assert max_index > min_index, "Not enough bars in dataset."
    test_index = random.randint(min_index, max_index)
    engine.current_index = test_index

    initial_equity = engine.initial_cash
    current_bar = df.iloc[engine.current_index]
    current_price = float(current_bar["open"])

    atr_col = [col for col in df.columns if col.startswith("atr_")]
    assert atr_col, "ATR column missing"
    atr = float(current_bar[atr_col[0]])
    assert atr > 0, "ATR must be positive"

    atr_multiple = 3

    asset_cfg = next(a for a in engine.assets if a.get('symbol') == symbol)

    contract_size = asset_cfg['contract_size']
    tick_size = asset_cfg['tick_size']
    tick_value = asset_cfg['tick_value']
    initial_margin = asset_cfg['initial_margin']
    commission_per_contract = asset_cfg['commission_per_contract']
    fee_per_trade = asset_cfg['fee_per_trade']
    slippage_pct = asset_cfg.get('slippage_pct', engine.config.get('slippage_pct', 0.001))

    qty, stop_loss_price, risk_per_contract = compute_safe_position_size(
        side="long",
        entry_price=current_price,
        atr=atr,
        atr_multiple=atr_multiple,
        equity=initial_equity,
        risk_percent=0.01,
        contract_size=contract_size,
        tick_size=tick_size,
        tick_value=tick_value,
        slippage_pct=slippage_pct,
        initial_margin=initial_margin,
        commission_per_contract=commission_per_contract,
        fee_per_trade=fee_per_trade
    )

    if qty < 1:
        pytest.skip("Qty less than 1; trade not viable.")

    order_id = engine.place_order({
        "symbol": symbol,
        "side": "buy",
        "qty": qty,
        "order_type": "market",
        "stop_loss": stop_loss_price
    })

    assert order_id is not None
    pos = engine.positions[symbol]
    assert pos.qty == qty

    max_steps = 5000
    for _ in range(max_steps):
        engine.step()
        pos = engine.positions.get(symbol)
        if pos is None or pos.qty == 0:
            break

    final_equity = engine.get_portfolio()["total_equity"]
    loss = initial_equity - final_equity
    expected_max_loss = initial_equity * 0.01

    print(f"Seed: {seed}")
    print(f"Start Index: {test_index}")
    print(f"End Index: {engine.current_index}")
    print(f"Initial Equity: {initial_equity:.2f}")
    print(f"Final Equity: {final_equity:.2f}")
    print(f"Equity Loss: {loss:.2f}")
    print(f"Allowed max: {expected_max_loss:.2f}")

    assert loss <= expected_max_loss, (
        f"[Seed {seed}] Equity loss {loss:.2f} exceeds allowed max {expected_max_loss:.2f}"
    )



@pytest.mark.parametrize("seed", range(10))
def test_manual_short_order_stop_loss(seed):
    random.seed(seed)
    config_path = "testing/functional_tests/configs/backtest_config_2.yaml"
    engine = BacktesterEngine(config_path=config_path)
    env = TradingEnvironment()
    env.set_api(engine)

    symbol = "CL=F"
    df = env.get_asset_data(symbol)
    assert df is not None and not df.empty

    min_index = 21
    max_index = len(df) - 25
    assert max_index > min_index, "Not enough bars in dataset."
    test_index = random.randint(min_index, max_index)
    engine.current_index = test_index

    initial_equity = engine.initial_cash
    current_bar = df.iloc[engine.current_index]
    current_price = float(current_bar["open"])
    
    print(f'Current Bar Open: {float(current_bar["open"])}')
    print(f'Current Bar Close: {float(current_bar["close"])}')

    atr_col = [col for col in df.columns if col.startswith("atr_")]
    assert atr_col, "ATR column missing"
    atr = float(current_bar[atr_col[0]])
    assert atr > 0, "ATR must be positive"

    atr_multiple = 3

    asset_cfg = next(a for a in engine.assets if a.get('symbol') == symbol)

    contract_size = asset_cfg['contract_size']
    tick_size = asset_cfg['tick_size']
    tick_value = asset_cfg['tick_value']
    initial_margin = asset_cfg['initial_margin']
    commission_per_contract = asset_cfg['commission_per_contract']
    fee_per_trade = asset_cfg['fee_per_trade']
    slippage_pct = asset_cfg.get('slippage_pct', engine.config.get('slippage_pct', 0.001))

    qty, stop_loss_price, risk_per_contract = compute_safe_position_size(
        side="short",
        entry_price=current_price,
        atr=atr,
        atr_multiple=atr_multiple,
        equity=initial_equity,
        risk_percent=0.01,
        contract_size=contract_size,
        tick_size=tick_size,
        tick_value=tick_value,
        slippage_pct=slippage_pct,
        initial_margin=initial_margin,
        commission_per_contract=commission_per_contract,
        fee_per_trade=fee_per_trade
    )
    
    print(f"Set Stop Loss: {stop_loss_price}")
    print(f"Quantity: {qty}")

    if qty < 1:
        pytest.skip("Qty less than 1; trade not viable.")

    order_id = engine.place_order({
        "symbol": symbol,
        "side": "sell",
        "qty": qty,
        "order_type": "market",
        "stop_loss": stop_loss_price
    })

    assert order_id is not None
    pos = engine.positions[symbol]
    assert pos.qty == -qty

    max_steps = 5000
    for _ in range(max_steps):
        engine.step()
        pos = engine.positions.get(symbol)
        if pos is None or pos.qty == 0:
            break

    final_equity = engine.get_portfolio()["total_equity"]
    loss = initial_equity - final_equity
    expected_max_loss = initial_equity * 0.01

    print(f"Seed: {seed}")
    print(f"Start Index: {test_index}")
    print(f"End Index: {engine.current_index}")
    print(f"Initial Equity: {initial_equity:.2f}")
    print(f"Final Equity: {final_equity:.2f}")
    print(f"Equity Loss: {loss:.2f}")
    print(f"Allowed max: {expected_max_loss:.2f}")

    assert loss <= expected_max_loss, (
        f"[Seed {seed}] Equity loss {loss:.2f} exceeds allowed max {expected_max_loss:.2f}"
    )

