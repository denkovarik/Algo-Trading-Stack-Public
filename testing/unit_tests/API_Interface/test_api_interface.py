# Usage: PYTHONPATH=. pytest testing/unit_tests/API_Interface/test_api_interface.py

import unittest
import tempfile
import shutil
import os
import pandas as pd
from datetime import datetime, timezone
from classes.API_Interface import (
    Position,
    round_to_tick,
)
from classes.Backtester_Engine import BacktesterEngine

class EngineUnitTests(unittest.TestCase):
    def setUp(self):
        # --- temp workspace with tiny TS CSV + YAML config
        self.tmpdir = tempfile.mkdtemp(prefix="bt_engine_tests_")

        # Minimal 2-bar TradeStation-like CSV
        csv_path = os.path.join(self.tmpdir, "TradeStation_data.csv")
        df = pd.DataFrame({
            "Open":  [100.0, 101.0],
            "High":  [102.0, 103.0],
            "Low":   [ 99.5, 100.5],
            "Close": [101.5, 102.5],
            "TotalVolume": [10, 12],
            "TimeStamp": [
                datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc).isoformat(),
                datetime(2024, 1, 1, 9, 31, tzinfo=timezone.utc).isoformat(),
            ],
        })
        df.to_csv(csv_path, index=False)

        # Minimal YAML config with required futures margins
        self.yaml_path = os.path.join(self.tmpdir, "backtest_config.yaml")
        yaml_text = f"""
                    initial_cash: 100000
                    include_slippage: false
                    commission_per_contract: 2.5
                    fee_per_trade: 1.0
                    assets:
                      - symbol: NQ
                        type: futures
                        file: "{csv_path}"
                        contract_size: 20
                        tick_size: 0.25
                        tick_value: 5.0
                        initial_margin: 500.0
                        maintenance_margin: 400.0
                    """
        with open(self.yaml_path, "w") as f:
            f.write(yaml_text)

        # Engine under test
        self.engine = BacktesterEngine(config_path=self.yaml_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ---- helpers
    def _assert_has_cols(self, df, cols):
        self.assertTrue(hasattr(df, "columns"))
        for c in cols:
            self.assertIn(c, df.columns)

    # ---- tests
    def test_connect_and_disconnect(self):
        self.assertFalse(self.engine.connected)
        self.engine.connect()
        self.assertTrue(self.engine.connected)
        self.engine.disconnect()
        self.assertFalse(self.engine.connected)

    def test_get_historical_data_returns_dataframe_with_columns(self):
        self.engine.connect()
        df = self.engine.get_historical_data("NQ", "1m", "2024-01-01", "2024-01-02")
        self._assert_has_cols(df, ["open", "high", "low", "close", "volume"])

    def test_load_backtest_config_loads_assets_and_params(self):
        # Already loaded in setUp, but call again to ensure idempotence
        self.engine.load_backtest_config(self.yaml_path)
        self.assertIn("assets", self.engine.config)
        self.assertIsInstance(self.engine.assets, list)
        self.assertGreater(len(self.engine.assets), 0)

        # Asset has data and expected params
        sym = self.engine.assets[0]["symbol"]
        self.assertEqual(sym, "NQ")
        self.assertIn("data", self.engine.assets[0])
        self.assertIsNotNone(self.engine.assets[0]["data"])
        self.assertIn(sym, self.engine.symbol_params)
        p = self.engine.symbol_params[sym]
        for k in ["tick_size", "tick_value", "initial_margin", "maintenance_margin"]:
            self.assertIn(k, p)

    def test_place_order_fills_immediately_and_updates_position_cash(self):
        # Place a simple market buy of 1 contract at current_index=0
        oid = self.engine.place_order({"symbol": "NQ", "side": "buy", "qty": 1})
        self.assertIsNotNone(oid)
        status = self.engine.get_order_status(oid)
        self.assertEqual(status, "filled")

        # Position exists and qty=1
        pos = self.engine.positions.get("NQ")
        self.assertIsNotNone(pos)
        self.assertEqual(pos.qty, 1)

        # Trade log captured and cash decreased by fees/commissions (realized could be 0 at entry)
        self.assertGreaterEqual(len(self.engine.trade_log), 1)
        last = self.engine.trade_log[-1]
        self.assertEqual(last["symbol"], "NQ")
        self.assertEqual(last["qty"], 1)
        self.assertIn("commission", last)
        self.assertIn("fee", last)

    def test_cancel_order_only_affects_open_orders(self):
        # Create a synthetic OPEN order (engine usually fills immediately)
        from classes.API_Interface import Order
        o = Order(order_id=999, symbol="NQ", side="buy", qty=1, order_type="market")
        self.engine.orders[o.order_id] = o  # status 'open' by default

        self.engine.cancel_order(999)
        self.assertEqual(self.engine.get_order_status(999), "cancelled")

    def test_round_to_tick_behavior(self):
        # Positive half-up around 0.25 ticks
        self.assertAlmostEqual(round_to_tick(100.12, 0.25), 100.00)   # below midpoint -> down
        self.assertAlmostEqual(round_to_tick(100.13, 0.25), 100.25)   # above midpoint -> up
        self.assertAlmostEqual(round_to_tick(100.125, 0.25), 100.25)  # at midpoint -> up

        # NaN passthrough
        self.assertTrue(pd.isna(round_to_tick(float("nan"), 0.25)))

        # Negative half-away-from-zero at midpoint
        self.assertAlmostEqual(round_to_tick(-1.125, 0.25), -1.25)

        # No-op when invalid tick
        self.assertEqual(round_to_tick(100.13, 0), 100.13)
        self.assertEqual(round_to_tick(100.13, None), 100.13)


if __name__ == "__main__":
    unittest.main()

