# bots/trend_following_bot.py

from __future__ import annotations

import numpy as np
import pandas as pd

# Inherit all lifecycle / sizing / session / exit handling from the base
from bots.base_strategy_bot import BaseStrategyBot


class TrendFollowingBot(BaseStrategyBot):
    """
    EMA-trend + crossback / ATR-breakout entries.
    Exit management (stop-loss & take-profit) is handled by the BaseStrategyBot
    via the provided ExitStrategy (e.g., TrailingATRExit, FixedRatioExit).

    Customize with:
      - trend_ema_span: which EMA column to use, e.g. ema_50
      - breakout_atr_mult: offset around EMA for breakout entries (in ATRs)
    """

    def __init__(
        self,
        trend_ema_span: int = 50,
        breakout_atr_mult: float = 1.5,
        # Pass-through kwargs go to BaseStrategyBot (risk %, sessions, exit_strategy, etc.)
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.trend_ema_span = int(trend_ema_span)
        self.breakout_atr_mult = float(breakout_atr_mult)

    # --- override only the entry decision ---

    def decide_side(self, env, symbol, df, idx_last, idx_prev) -> str | None:
        """
        Use previous CLOSED bar for signal; place order at current bar OPEN.
        Signal:
          1) Crossback: prev_close crosses EMA, confirmed by current open on trend side
          2) Breakout: current open beyond EMA +/- breakout_atr_mult * ATR
        """
        ema_col = f"ema_{self.trend_ema_span}"
        if ema_col not in df.columns:
            return None

        ema_prev = df[ema_col].iloc[idx_prev]
        if pd.isna(ema_prev):
            return None
        ema_prev = float(ema_prev)

        # ATR from the previous CLOSED bar (helper on the base)
        atr = self._get_prev_atr(df, idx_prev)
        if atr is None or atr < 1e-6:
            return None

        entry_open = df['open'].iloc[idx_last]
        prev_close = df['close'].iloc[idx_prev]
        if pd.isna(entry_open) or pd.isna(prev_close):
            return None
        entry_open = float(entry_open)
        prev_close = float(prev_close)

        # 1) Crossback
        if entry_open > ema_prev and prev_close <= ema_prev:
            return 'buy'
        if entry_open < ema_prev and prev_close >= ema_prev:
            return 'sell'

        # 2) Breakout around EMA by k*ATR
        threshold = self.breakout_atr_mult * atr
        if entry_open > ema_prev + threshold:
            return 'buy'
        if entry_open < ema_prev - threshold:
            return 'sell'

        return None

