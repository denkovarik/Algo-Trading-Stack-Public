# bots/coin_flip_bot.py

from __future__ import annotations
from bots.base_strategy_bot import BaseStrategyBot

class CoinFlipBot(BaseStrategyBot):
    """
    Pure coin‑flip entries at the bar OPEN.

    Everything else (risk-based position sizing, session/holiday checks,
    exit management via the chosen ExitStrategy, maintenance flattening, etc.)
    is handled by BaseStrategyBot.

    Params:
      - p_buy: Probability of choosing 'buy' on any eligible bar (default 0.5).
      - loose_eligibility: If True, accept entries on any bar with minimal data
        sanity (prev close + current open + ATR present), ignoring session gates.
        If False, defer to BaseStrategyBot's stricter wants_entry_at(...) checks.
      - **kwargs: Forwarded to BaseStrategyBot (e.g., exit_strategy, base_risk_percent, etc.)
    """

    def __init__(self, p_buy: float = 0.5, loose_eligibility: bool = False, **kwargs):
        super().__init__(**kwargs)
        if p_buy < 0.0 or p_buy > 1.0:
            raise ValueError("p_buy must be between 0.0 and 1.0")
        self.p_buy = float(p_buy)
        self.loose_eligibility = bool(loose_eligibility)

    # Let the generator (and on_bar) ask us if this bar is acceptable for a new entry.
    def wants_entry_at(self, env, symbol, df, idx_last, idx_prev) -> bool:
        if not self.loose_eligibility:
            # Use the base gates: bar validity, (optional) session/holiday, prev ATR
            return super().wants_entry_at(env, symbol, df, idx_last, idx_prev)

        # Loose mode: minimal, data-only guard so coin-flip can place trades "anywhere"
        if df is None or idx_prev < 0 or idx_last >= len(df):
            return False
        if 'open' not in df.columns or 'close' not in df.columns:
            return False
        if not (df['open'].notna().iloc[idx_last] and df['close'].notna().iloc[idx_prev]):
            return False
        atr = self._get_prev_atr(df, idx_prev)  # reuse base helper
        return (atr is not None) and (atr >= self.min_atr_threshold)

    # --- override just the entry decision (unchanged) ---
    def decide_side(self, env, symbol, df, idx_last, idx_prev) -> str | None:
        """
        Called during the engine's OPEN phase with H/L hidden and prior indicators carried forward.
        Returns 'buy' or 'sell'. Return None to abstain (we never abstain here).
        """
        return 'buy' if self._rng.random() < self.p_buy else 'sell'

