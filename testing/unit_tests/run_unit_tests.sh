#!/usr/bin/env bash
set -euo pipefail

# Where this script lives (now in .../testing/unit_tests)
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Project root is TWO levels up from here
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Use the project root for imports and paths
export PYTHONPATH="$PROJECT_ROOT"
cd "$PROJECT_ROOT"

# ---- API_Interface tests ----
python3 -m unittest testing.unit_tests.API_Interface.test_api_interface
pytest "$PROJECT_ROOT/testing/unit_tests/API_Interface/test_api_interface.py"
pytest "$PROJECT_ROOT/testing/unit_tests/API_Interface/test_margin.py"
pytest "$PROJECT_ROOT/testing/unit_tests/API_Interface/test_engine_fill_and_protective.py"
pytest "$PROJECT_ROOT/testing/unit_tests/API_Interface/test_portfolio_and_replay.py"
pytest "$PROJECT_ROOT/testing/unit_tests/API_Interface/test_place_order_regression.py"
pytest "$PROJECT_ROOT/testing/unit_tests/API_Interface/test_margin_intrabar_worstcase.py"

# ---- TradeStationLiveAPI tests ----
python3 -m unittest testing/unit_tests/Tradestation_Live_API/test_live_api_and_bot.py -v

# ---- Trading Environment tests ----
python3 -m unittest testing.unit_tests.Trading_Environment.test_trading_environment
pytest "$PROJECT_ROOT/testing/unit_tests/Trading_Environment/test_env_core.py"
pytest "$PROJECT_ROOT/testing/unit_tests/bots/test_base_strategy_bot_session.py"

# ---- Bot tests ----
pytest "$PROJECT_ROOT/testing/unit_tests/Trading_Environment/test_indicator_helpers.py"

# ---- Indicators / ATR tests ----
pytest "$PROJECT_ROOT/testing/unit_tests/indicators/atr/test_atr_core.py"
pytest "$PROJECT_ROOT/testing/unit_tests/indicators/atr/test_atr_ratios.py"

# ---- Indicators / EMA tests ----
pytest "$PROJECT_ROOT/testing/unit_tests/indicators/ema/test_ema_core.py"
pytest "$PROJECT_ROOT/testing/unit_tests/indicators/ema/test_ema_edge_cases.py"

# ---- Indicators / Bollinger tests ----
pytest "$PROJECT_ROOT/testing/unit_tests/indicators/bollinger"

# ---- Indicators / RSI tests ----
pytest "$PROJECT_ROOT/testing/unit_tests/indicators/rsi"

# ---- Indicators / Donchian tests ----
pytest "$PROJECT_ROOT/testing/unit_tests/indicators/donchian"

# ---- Indicators / MACD tests ----
pytest "$PROJECT_ROOT/testing/unit_tests/indicators/macd"

# ---- Indicators / ROC tests ----
pytest "$PROJECT_ROOT/testing/unit_tests/indicators/roc"

# ---- Indicators / VWAP tests ----
pytest "$PROJECT_ROOT/testing/unit_tests/indicators/vwap"

