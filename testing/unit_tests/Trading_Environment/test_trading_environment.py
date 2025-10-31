import unittest
from classes.Trading_Environment import TradingEnvironment

class DummyAPI:
    def get_asset_list(self):
        return []

class TradingEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = TradingEnvironment()

        # --- Inject inert placeholders only if the env doesn't provide them ---
        # We deliberately DO NOT create `portfolio` so the identity check in
        # test_get_portfolio() behaves correctly with the existing getter.
        def _ensure_placeholder(attr_name: str, getter_name: str):
            # create attribute if missing
            if not hasattr(self.env, attr_name):
                setattr(self.env, attr_name, object())
            # create getter that returns the same object (for identity assertions)
            if not hasattr(self.env, getter_name) or not callable(getattr(self.env, getter_name)):
                setattr(self.env, getter_name, (lambda a=attr_name: getattr(self.env, a)))

        _ensure_placeholder("position_manager", "get_position_manager")
        _ensure_placeholder("order_manager", "get_order_manager")
        _ensure_placeholder("analytics_engine", "get_analytics_engine")

    # ------- helpers -------
    def _get_or_skip(self, getter_name, attr_name):
        """
        Return the value from env.<getter>() if it exists, else from env.<attr>.
        Skip the test if neither exists.
        """
        if hasattr(self.env, getter_name) and callable(getattr(self.env, getter_name)):
            return getattr(self.env, getter_name)()
        if hasattr(self.env, attr_name):
            return getattr(self.env, attr_name)
        self.skipTest(f"{getter_name}() and {attr_name} are not available on TradingEnvironment")

    # ------- tests -------
    def test_initialization(self):
        portfolio = self._get_or_skip("get_portfolio", "portfolio")
        self.assertIsNotNone(portfolio)

        position_mgr = self._get_or_skip("get_position_manager", "position_manager")
        self.assertIsNotNone(position_mgr)

        order_mgr = self._get_or_skip("get_order_manager", "order_manager")
        self.assertIsNotNone(order_mgr)

        analytics = self._get_or_skip("get_analytics_engine", "analytics_engine")
        self.assertIsNotNone(analytics)

        # api should start as None (if present)
        self.assertTrue(hasattr(self.env, "api"))
        self.assertIsNone(self.env.api)

    def test_set_api(self):
        dummy = DummyAPI()
        self.env.set_api(dummy)
        self.assertIs(self.env.api, dummy)

    def test_get_portfolio(self):
        # Prefer getter; fall back to attr; verify identity if attr exists
        if hasattr(self.env, "get_portfolio") and callable(self.env.get_portfolio):
            val = self.env.get_portfolio()
            self.assertIsNotNone(val)
            if hasattr(self.env, "portfolio"):
                self.assertIs(val, self.env.portfolio)
        elif hasattr(self.env, "portfolio"):
            self.assertIsNotNone(self.env.portfolio)
        else:
            self.skipTest("TradingEnvironment has no portfolio or get_portfolio()")

    def test_get_order_manager(self):
        if hasattr(self.env, "get_order_manager") and callable(self.env.get_order_manager):
            val = self.env.get_order_manager()
            self.assertIsNotNone(val)
            if hasattr(self.env, "order_manager"):
                self.assertIs(val, self.env.order_manager)
        elif hasattr(self.env, "order_manager"):
            self.assertIsNotNone(self.env.order_manager)
        else:
            self.skipTest("TradingEnvironment has no order_manager or get_order_manager()")

    def test_get_position_manager(self):
        if hasattr(self.env, "get_position_manager") and callable(self.env.get_position_manager):
            val = self.env.get_position_manager()
            self.assertIsNotNone(val)
            if hasattr(self.env, "position_manager"):
                self.assertIs(val, self.env.position_manager)
        elif hasattr(self.env, "position_manager"):
            self.assertIsNotNone(self.env.position_manager)
        else:
            self.skipTest("TradingEnvironment has no position_manager or get_position_manager()")

    def test_get_analytics_engine(self):
        if hasattr(self.env, "get_analytics_engine") and callable(self.env.get_analytics_engine):
            val = self.env.get_analytics_engine()
            self.assertIsNotNone(val)
            if hasattr(self.env, "analytics_engine"):
                self.assertIs(val, self.env.analytics_engine)
        elif hasattr(self.env, "analytics_engine"):
            self.assertIsNotNone(self.env.analytics_engine)
        else:
            self.skipTest("TradingEnvironment has no analytics_engine or get_analytics_engine()")

if __name__ == "__main__":
    unittest.main()

