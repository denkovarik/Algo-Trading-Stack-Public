# run_backtest_headless.py 
import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from tqdm import tqdm

from classes.Backtester_Engine import BacktesterEngine
from classes.Trading_Environment import TradingEnvironment
from classes.ui_main_window import EquityCurvePlotWindow
from classes.ui_asset_viewer import AssetSelectorWindow
from classes.Backtest_Runner import BacktestApp, BacktestRunner

from bots.coin_flip_bot.coin_flip_bot import CoinFlipBot
from bots.trend_following_bot.trend_following_bot import TrendFollowingBot
from bots.exit_strategies import RLTrailingATRExit, TrailingATRExit

if __name__ == "__main__":
    config_path = "backtest_configs/backtest_config_10_yrs.yaml"

    # Choose exit strategy
    
    # Fixed ATR Trailing Stop Loss
    exit_strategy = TrailingATRExit(atr_multiple=3.0)
    
    # ML Trailing Stop Loss Exits
    PPO_Models_Dir = "bots/models/PPO_Trailing_Stop_Loss"
    
    # ML Trailing Stop Loss using PPO or LSTM models
    #exit_strategy = RLTrailingATRExit(
    #    model_dir=PPO_Models_Dir,
    #    fallback_multiple=3.0,   # used if a symbol has no model or SB3 isn't available
    #    ema_span=21,             # use 21 by default; you can sync this to your bot's EMA below
    #    debug=False,             # set True to print load/inference fallbacks
    #)

    # Pick your Bot
    bot = CoinFlipBot(
        exit_strategy=exit_strategy,
        base_risk_percent=0.01,
        enforce_sessions=False,
    )
    #bot = TrendFollowingBot(
    #    exit_strategy=exit_strategy,
    #    base_risk_percent=0.01,
    #    enforce_sessions=False,
    #    flatten_before_maintenance=True,
    #    enable_online_learning=False
    #)

    # Run backtest as usual — bots enable online learning internally
    app = BacktestApp(config_path, bot)
    runner = BacktestRunner(app)
    runner.run()

    # Optional: if you added a stats() method on the exit
    if hasattr(exit_strategy, "bandit_stats"):
        print(exit_strategy.bandit_stats())

