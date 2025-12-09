# Usage: PYTHONPATH=. pytest testing/functional_tests/test_backtest_consistency.py -v

import os
# Must be set before importing PyQt to make headless/offscreen work in CI
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import pytest
from PyQt5 import QtWidgets
from tqdm import tqdm

from classes.Backtester_Engine import BacktesterEngine
from classes.Trading_Environment import TradingEnvironment
from bots.trend_following_bot.trend_following_bot import TrendFollowingBot
from bots.exit_strategies import TrailingATRExit
from classes.Backtest_Runner import BacktestApp
from classes.ui_main_window import EquityCurvePlotWindow

# ---- Consistent engine knobs for both variants ----
ENGINE_DEFAULTS = dict(
    live_price_mode="open_only",
    skip_synthetic_open_bars=True,
    include_slippage=False,
    intrabar_tp_sl_policy="worst_case",
)

def _apply_engine_defaults(engine: BacktesterEngine):
    # Reset to a clean starting point; set flags identically in both runs
    engine.reset_backtest()
    engine.is_running = True  # required so env.on_bar_advanced() actually runs the bot
    engine.live_price_mode = ENGINE_DEFAULTS["live_price_mode"]
    engine.skip_synthetic_open_bars = ENGINE_DEFAULTS["skip_synthetic_open_bars"]
    engine.include_slippage = ENGINE_DEFAULTS["include_slippage"]
    engine.config["intrabar_tp_sl_policy"] = ENGINE_DEFAULTS["intrabar_tp_sl_policy"]

# ---- Bot factories: use fresh instances per run (avoid state leakage) ----
def make_rl_bot():
    exit_strategy = TrailingATRExit(atr_multiple=3.0)
    return TrendFollowingBot(
        exit_strategy=exit_strategy,
        base_risk_percent=0.01,
        enforce_sessions=False,
        flatten_before_maintenance=True,
    )

def make_deterministic_bot():
    # Handy if you want to pin this to a fully deterministic exit
    exit_strategy = TrailingATRExit(atr_multiple=3.0)
    return TrendFollowingBot(
        exit_strategy=exit_strategy,
        base_risk_percent=0.01,
        enforce_sessions=False,
        flatten_before_maintenance=True,
    )

# ---- Helpers to run both versions in offscreen mode ----
def run_gui_backtest(config_path, bot_factory):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # Build engine/env/bot just like the GUI variant, but DO NOT add extra connections.
    engine = BacktesterEngine(config_path=config_path)
    env = TradingEnvironment()
    env.set_api(engine)
    bot = bot_factory()
    env.set_bot(bot)

    _apply_engine_defaults(engine)

    # Equity curve window (offscreen)
    eq_win = EquityCurvePlotWindow(engine.equity_history, engine.equity_time_history)
    eq_win.show()

    total_bars = len(engine.df) if getattr(engine, "df", None) is not None else 0
    with tqdm(total=total_bars, desc="GUI Backtest", unit="bar", dynamic_ncols=True,
              leave=True, miniters=1, file=sys.stdout) as pbar:
        for _ in range(total_bars):
            engine.step()
            eq_win.update_curve(engine.equity_history, engine.equity_time_history)
            QtWidgets.QApplication.processEvents()
            pbar.update(1)

    # Finalize
    engine.close_all_positions()
    eq_win.close()
    return list(engine.equity_history), list(engine.equity_time_history), engine.get_portfolio()

def run_headless_backtest(config_path, bot_factory):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # Use the headless app, but force its bar_updated connection to match the GUI wiring.
    # BacktestApp sets bar_updated -> on_bar_advanced(compute_indicators=True) by default
    # we override to False to keep "bot @ OPEN, indicators @ CLOSE" consistent with env wiring
    backtest_app = BacktestApp(config_path, bot_factory())
    engine = backtest_app.engine
    env = backtest_app.env

    _apply_engine_defaults(engine)

    try:
        engine.bar_updated.disconnect()
    except Exception:
        pass
    engine.bar_updated.connect(lambda *_: env.on_bar_advanced(compute_indicators=False)) 

    # Equity curve window (offscreen) provided by BacktestApp
    eq_win = backtest_app.eq_win

    total_bars = len(engine.df) if getattr(engine, "df", None) is not None else 0
    with tqdm(total=total_bars, desc="Headless Backtest", unit="bar", dynamic_ncols=True,
              leave=True, miniters=1, file=sys.stdout) as pbar:
        for _ in range(total_bars):
            engine.step()
            eq_win.update_curve(engine.equity_history, engine.equity_time_history)
            QtWidgets.QApplication.processEvents()
            pbar.update(1)

    engine.close_all_positions()
    eq_win.close()
    return list(engine.equity_history), list(engine.equity_time_history), engine.get_portfolio()

# ---- Regression Test ----
def test_backtest_consistency(capsys):
    """
    Run both GUI and headless variants and assert identical results.
    """
    config_path = "backtest_configs/backtest_config_10_yrs.yaml"

    # Choose which bot to validate. RL is fine (falls back to deterministic if SB3 isn't available),
    # but for maximum determinism you can swap in make_deterministic_bot().
    bot_factory = make_rl_bot

    # Show progress bars while the loops run
    with capsys.disabled():
        eq_gui, times_gui, port_gui = run_gui_backtest(config_path, bot_factory)
        eq_head, times_head, port_head = run_headless_backtest(config_path, bot_factory)

    # --- Assertions ---
    assert len(eq_gui) == len(eq_head), "Equity history length mismatch"

    # Values should match to floating-point precision 
    assert eq_gui == pytest.approx(eq_head, rel=0, abs=1e-9), "Equity values differ"

    # Timestamps should align exactly (same bars appended in the same order)
    assert len(times_gui) == len(times_head), "Equity timestamp length mismatch"
    # Compare as strings to be robust to tz-awareness printing
    assert [str(t) for t in times_gui] == [str(t) for t in times_head], "Equity timestamps differ"

    # Final equity must match too
    assert port_gui["total_equity"] == pytest.approx(port_head["total_equity"], rel=0, abs=1e-9)

