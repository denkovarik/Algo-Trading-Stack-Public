#!/bin/bash

# Run all functional test scripts from project root

set -e  # Exit on any failure

echo "Running risk_tests.py..."
PYTHONPATH=. pytest testing/functional_tests/risk_tests.py

echo ""
echo "Running test_func_indicators.py..."
PYTHONPATH=. pytest testing/functional_tests/test_func_indicators.py

echo ""
echo "Running test_backtest_consistency..."
PYTHONPATH=. pytest testing/functional_tests/test_backtest_consistency.py -v

echo ""
echo "All functional tests completed."

