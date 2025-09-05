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

# ---- Trading Environment tests ----
python3 -m unittest testing.unit_tests.Trading_Environment.test_trading_environment
pytest "$PROJECT_ROOT/testing/unit_tests/Trading_Environment/test_env_core.py"
pytest "$PROJECT_ROOT/testing/unit_tests/Trading_Environment/test_indicator_helpers.py"

