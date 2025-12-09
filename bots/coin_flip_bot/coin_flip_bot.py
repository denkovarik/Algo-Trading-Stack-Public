# bots/coin_flip_bot.py

from __future__ import annotations
from typing import Optional
from bots.base_strategy_bot import BaseStrategyBot


class CoinFlipBot(BaseStrategyBot):
    """
    Pure coin-flip entries at the bar OPEN, with an optional daily-loss kill switch.

    What’s new
    ----------
    - daily_loss_limit_pct (inherited from BaseStrategyBot): when > 0, we track the
      day's starting equity (ET). If current equity falls below:
          start_equity * (1 - daily_loss_limit_pct)
      we DISABLE new entries for the remainder of that RTH day.
      (Existing positions are managed by the exit strategy as usual; we do not
      force-flatten here.)

    Params
    ------
    p_buy : float
        Probability of choosing 'buy' on an eligible bar. Default 0.5.
    loose_eligibility : bool
        If True, use minimal data gates (prev close + current open + ATR present),
        skipping session gates; otherwise defer to BaseStrategyBot session/holiday
        checks. Daily loss limit still applies in both modes.
    **kwargs :
        Forwarded to BaseStrategyBot (e.g., exit_strategy, base_risk_percent,
        daily_loss_limit_pct, etc.). See BaseStrategyBot for details.

    Notes
    -----
    - Uses engine portfolio equity via env.get_portfolio()['total_equity']:contentReference[oaicite:2]{index=2}.
    - Defines the "trading day" by converting current timestamp to ET using the
      base helpers and keys by local calendar date:contentReference[oaicite:3]{index=3}.
    """

    def __init__(self, p_buy: float = 0.5, loose_eligibility: bool = False, **kwargs):
        super().__init__(**kwargs)
        if p_buy < 0.0 or p_buy > 1.0:
            raise ValueError("p_buy must be between 0.0 and 1.0")
        self.p_buy = float(p_buy)
        self.loose_eligibility = bool(loose_eligibility)

        # --- Daily loss tracking state (reset each ET trading day) ---
        self._day_key_et: Optional[str] = None           # "YYYY-MM-DD" in ET
        self._day_start_equity: Optional[float] = None
        self._disabled_for_day: bool = False

    # ---------- Daily loss limiter utilities ----------

    def _update_daily_state_and_check_limit(self, env) -> bool:
        """
        Ensure the per-day state is current and update the disabled flag if
        the daily loss limit is breached.

        Returns True if the bot is currently disabled for the day.
        """
        # Feature off if nonpositive
        limit = float(getattr(self, "daily_loss_limit_pct", 0.0) or 0.0)
        if limit <= 0.0:
            return False

        # Establish current ET day key from engine/base timestamp helpers.
        # BaseStrategyBot provides _current_timestamp_utc(...) and _to_eastern(...).
        ts_utc = self._current_timestamp_utc(env)          # pd.Timestamp (UTC) or None
        if ts_utc is None:
            # No clock yet — do not disable preemptively.
            return self._disabled_for_day

        ts_et = self._to_eastern(ts_utc)                   # convert to America/New_York
        day_key = ts_et.date().isoformat()

        # New ET day? Reset day state.
        if day_key != self._day_key_et:
            self._day_key_et = day_key
            self._disabled_for_day = False
            # Capture day start equity at first bar we see for the day
            port = env.get_portfolio()
            self._day_start_equity = float(port.get("total_equity", 0.0))

        # If already disabled, keep it disabled.
        if self._disabled_for_day:
            return True

        # If we don't have a start equity yet, try to initialize it now.
        if self._day_start_equity is None or self._day_start_equity <= 0.0:
            port = env.get_portfolio()
            self._day_start_equity = float(port.get("total_equity", 0.0))

        # Compute current drawdown vs the day start.
        start_eq = float(self._day_start_equity or 0.0)
        if start_eq <= 0.0:
            return self._disabled_for_day  # can't evaluate

        cur_eq = float(env.get_portfolio().get("total_equity", start_eq))  # engine supplies this
        dd_pct = (cur_eq - start_eq) / start_eq

        # If drawdown exceeds limit, disable entries for the rest of the day.
        if dd_pct <= -limit:
            self._disabled_for_day = True

        return self._disabled_for_day

    # Let the generator (and on_bar) ask us if this bar is acceptable for a new entry.
    def wants_entry_at(self, env, symbol, df, idx_last, idx_prev) -> bool:
        # First, update/check the daily-loss limiter
        if self._update_daily_state_and_check_limit(env):
            return False

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

