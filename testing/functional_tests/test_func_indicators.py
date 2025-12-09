# Usage: PYTHONPATH=. pytest testing/functional_tests/test_func_indicators.py


import pandas as pd
import tempfile
import yaml
import os

from classes.Trading_Environment import TradingEnvironment
from classes.Backtester_Engine import BacktesterEngine

def make_test_csv(path, num_rows=20):
    """Create a simple OHLCV CSV for testing."""
    df = pd.DataFrame({
        'date': pd.date_range(start="2020-01-01", periods=num_rows, freq="min"),
        'open': [100 + i for i in range(num_rows)],
        'high': [101 + i for i in range(num_rows)],
        'low': [99 + i for i in range(num_rows)],
        'close': [100 + i for i in range(num_rows)],
        'volume': [1000 for _ in range(num_rows)],
    })
    df['timestamp'] = df['date'].astype('int64') // 10**9
    df.to_csv(path, index=False)
    return df

def test_indicator_recomputation_with_backtester():
    with tempfile.TemporaryDirectory() as tmpdir:
        symbol = "TEST"
        csv_path = os.path.join(tmpdir, f"{symbol}.csv")

        # Create test CSV file
        df = make_test_csv(csv_path, num_rows=20)

        # Create a minimal config YAML for the backtester
        config = {
            "initial_cash": 1_000_000,
            "commission_per_contract": 0.0,
            "fee_per_trade": 0.0,
            "assets": [{
                "symbol": symbol,
                "type": "futures",
                "contract_size": 1,
                "tick_size": 0.01,
                "tick_value": 10,
                "initial_margin": 1000,
                "maintenance_margin": 500,
                "file": csv_path
            }]
        }
        yaml_path = os.path.join(tmpdir, "config.yaml")
        with open(yaml_path, "w") as f:
            yaml.safe_dump(config, f)

        # Set up Backtester and Trading Environment
        backtester = BacktesterEngine(yaml_path)
        env = TradingEnvironment()
        env.set_api(backtester)

        # Indicator columns should exist immediately after setup
        data = backtester.get_asset_data(symbol)
        indicator_cols = [
            col for col in data.columns
            if col.startswith("bb_")
            or col.startswith("ema_")
            or col.startswith("rsi_")
            or col.startswith("atr_")
            or col.startswith("tr")
        ]
        assert indicator_cols, "Indicators should exist immediately after setup."

        # Advance the bar (simulate running a backtest step)
        backtester.current_index = 5  # move to bar 5
        backtester.is_running = True  # <-- ensure env recomputes on on_bar_advanced
        env.on_bar_advanced()  # This should compute indicators

        data = backtester.get_asset_data(symbol)
        indicator_cols = [
            col for col in data.columns
            if col.startswith("bb_")
            or col.startswith("ema_")
            or col.startswith("rsi_")
            or col.startswith("atr_")
            or col.startswith("tr")
        ]
        assert indicator_cols, "Indicators should be present after bar advanced!"

        # Store which columns exist now
        cols_after_advance = set(data.columns)

        # Call reset_indicators (should drop indicator columns)
        env.reset_indicators()
        data = backtester.get_asset_data(symbol)
        cols_after_reset = set(data.columns)
        assert not any(
            col.startswith("bb_")
            or col.startswith("ema_")
            or col.startswith("rsi_")
            or col.startswith("atr_")
            or col.startswith("tr")
            for col in data.columns
        ), (
            "All indicator columns should be dropped after reset_indicators!"
        )

        # Advance another bar, should recompute indicators
        backtester.current_index += 1
        backtester.is_running = True  # <-- required again before recompute
        env.on_bar_advanced()
        data = backtester.get_asset_data(symbol)
        cols_after_recompute = set(data.columns)
        assert any(col.startswith(("bb_", "ema_", "rsi_", "atr_", "tr")) for col in data.columns), \
            "Indicators should be recomputed after new bar advance following reset!"

        # (Optional) Print for debugging
        print("Columns after indicators computed:", cols_after_advance)
        print("Columns after reset:", cols_after_reset)
        print("Columns after recomputation:", cols_after_recompute)

if __name__ == "__main__":
    test_indicator_recomputation_with_backtester()
    print("Test passed.")

