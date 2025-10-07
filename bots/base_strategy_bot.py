# bots/base_strategy_bot.py

import numpy as np
import pandas as pd
from datetime import date as _date, time as _time, timedelta as _td

from classes.API_Interface import round_to_tick  # tick rounding on all price levels
from bots.exit_strategies import (
    ExitStrategy,
    TrailingATRExit,
    FixedRatioExit,
)


class BaseStrategyBot:
    """
    Base class: coin-flip entries + robust helper toolkit (sessions/holidays, safe sizing,
    and exit management via an ExitStrategy that controls BOTH stop-loss and take-profit).

    Subclass this and override `decide_side(...)` and/or risk logic as needed.
    """

    # Small constants to avoid repeating string literals in hot paths
    _COL_OPEN = "open"
    _COL_CLOSE = "close"
    _COL_DATE = "date"

    def __init__(
        self,
        # NEW preferred API
        exit_strategy: ExitStrategy | None = None,

        # risk & sizing
        base_risk_percent: float = 0.01,
        max_qty: int = 100,

        # ---- Preferred ATR column names (optional; preserves older behavior) ----
        # e.g., ("atr_90", "atr_60"). If None, we fall back to the first 'atr_' column.
        atr_preference: tuple[str, ...] | None = None,

        # Session / holiday rules (ET RTH by default)
        session_tz: str = "America/New_York",
        rth_start: _time = _time(9, 30),
        rth_end: _time = _time(16, 0),
        # optional, simple early-close support
        rth_early_end: _time = _time(13, 0),
        early_close_dates: list[str] | None = None,
        skip_us_holidays: bool = True,
        enforce_sessions: bool = True,

        # Maintenance flattening (cut risk before 5pm ET)
        flatten_before_maintenance: bool = True,
        maintenance_cutoff_et: _time = _time(17, 0),
        flatten_lead_minutes: int = 10,

        # Random seed for the coin flip (stable backtests if desired)
        seed: int | None = None,

        # ---- Eligibility / gating (new) ----
        use_entry_eligibility_hook: bool = True,
        min_atr_threshold: float = 1e-6,

        # ---- Legacy knobs (backward-compat) ----
        stop_loss_strategy=None,                # old StopLossStrategy (SL only)
        tp_rr_multiple: float | None = None,    # old fixed-R TP in the bot

        # ---- Legacy risk gates (accept but currently inert unless you wire them up) ----
        daily_loss_limit_pct: float = 0.0,
        intraday_drawdown_limit_pct: float = 0.0,
        daily_trades_max: int = 0,
        cooldown_bars_after_entry: int = 0,
        ban_open_minutes: int = 0,

        # NEW: opt-in online learning wrap (default off for speed)
        enable_online_learning: bool = False,

        # Future-proof: swallow unknown kwargs without crashing
        **_unused,
    ):
        self.base_risk_percent = float(base_risk_percent)
        self.max_qty = int(max_qty)

        # Store ATR preference (tuple or None)
        self.atr_preference = tuple(atr_preference) if atr_preference else None

        # Store legacy risk-gate knobs (not enforced unless you add logic)
        self.daily_loss_limit_pct = float(daily_loss_limit_pct)
        self.intraday_drawdown_limit_pct = float(intraday_drawdown_limit_pct)
        self.daily_trades_max = int(daily_trades_max)
        self.cooldown_bars_after_entry = int(cooldown_bars_after_entry)
        self.ban_open_minutes = int(ban_open_minutes)

        # Prefer new exit_strategy; otherwise adapt legacy stop_loss_strategy.
        if exit_strategy is not None:
            self.exit_strategy = exit_strategy
        elif stop_loss_strategy is not None:
            # Only used if you pass a legacy SL strategy.
            # from bots.exit_strategies import StopLossToExitAdapter  # add import if you need it
            self.exit_strategy = TrailingATRExit(atr_multiple=3.0)  # fallback if adapter not present
        else:
            # Default: Tom Basso style — trailing ATR with NO TP
            self.exit_strategy = TrailingATRExit(atr_multiple=3.0)

        # Legacy TP (only used if the chosen exit strategy does not set a TP)
        self._legacy_tp_rr_multiple = tp_rr_multiple

        # session/holiday rules
        self.session_tz = session_tz
        self.rth_start = rth_start
        self.rth_end = rth_end
        self.rth_early_end = rth_early_end
        self.early_close_dates = (
            set(pd.to_datetime(d).date() for d in (early_close_dates or []))
        )
        self.skip_us_holidays = skip_us_holidays
        self.enforce_sessions = enforce_sessions
        self._holiday_cache: dict[int, set[_date]] = {}

        # maintenance flattening
        self.flatten_before_maintenance = flatten_before_maintenance
        self.maintenance_cutoff_et = maintenance_cutoff_et
        self.flatten_lead = _td(minutes=int(flatten_lead_minutes))
        self._last_flatten_date: _date | None = None

        # --- NEW: eligibility hook controls ---
        self.use_entry_eligibility_hook = bool(use_entry_eligibility_hook)
        self.min_atr_threshold = float(min_atr_threshold)

        # runtime state
        self.state: dict = {}
        # NEW: tiny cache of ATR columns per DataFrame columns signature
        self._atr_cols_cache: dict[tuple, list[str]] = {}

        # RNG for coin flip
        self._rng = np.random.default_rng(seed)

        # Optional: warn about truly unknown kwargs (helps future cleanups)
        if _unused:
            print(f"[BaseStrategyBot] Ignoring unused kwargs: {sorted(_unused.keys())}")

    # ----------------- lifecycle -----------------

    def reset(self):
        self.state.clear()
        self._last_flatten_date = None
        self._atr_cols_cache.clear()

    # ----------------- subclass hooks -----------------

    def decide_side(self, env, symbol, df, idx_last, idx_prev) -> str | None:
        """Default strategy: pure coin flip."""
        return 'buy' if self._rng.random() < 0.5 else 'sell'

    # ----------------- NEW: entry eligibility hook -----------------

    def wants_entry_at(self, env, symbol, df, idx_last, idx_prev) -> bool:
        """
        Return True if, at df index `idx_last` (current bar), it's acceptable
        for THIS bot to open a new position (ignoring current open positions).

        Mirrors the same gates used in on_bar:
          - bar validity (current OPEN, previous CLOSE present)
          - (optional) session/holiday gate
          - previous-bar ATR exists and is non-trivial
        """
        # Same bar validity criteria used by on_bar
        if df is None or len(df) < 2:
            return False
        if idx_prev < 0 or idx_last >= len(df):
            return False

        # Cheaper single-cell checks
        cols = df.columns
        try:
            i_open = cols.get_loc(self._COL_OPEN)
            i_close = cols.get_loc(self._COL_CLOSE)
        except KeyError:
            return False

        if pd.isna(df.iat[idx_last, i_open]):
            return False
        if pd.isna(df.iat[idx_prev, i_close]):
            return False

        # Session/holiday rules
        if self.enforce_sessions and not self._session_allows_entry(df, idx_last):
            return False

        # Need a previous-bar ATR
        atr = self._get_prev_atr(df, idx_prev)
        if atr is None or atr < self.min_atr_threshold:
            return False

        return True

    # ----------------- main event -----------------

    def on_bar(self, env):
        """
        Runs during the engine's OPEN phase with a live snapshot
        (H/L hidden; previous indicators carried forward safely).
        """
        portfolio = env.get_portfolio()
        equity = portfolio.get('total_equity', 100_000)
        available_equity = portfolio.get('available_equity', equity)

        # NEW: ensure any RL/ML exit models are loaded ONCE per bar (not per symbol)
        if hasattr(self.exit_strategy, "ensure_models_loaded"):
            try:
                self.exit_strategy.ensure_models_loaded(env)
            except Exception:
                # non-fatal; exit strategy will fallback internally
                pass

        # 1) Maintenance/flatten window
        if self._should_flatten_now(env):
            self._flatten_all_positions(env)
            return

        # 2) Per-symbol loop
        for symbol in env.get_asset_list():
            pos = portfolio['positions'].get(symbol)
            df = env.get_latest_data(symbol)
            if not self._validate_bar_data(df):
                continue

            idx_last, idx_prev = len(df) - 1, len(df) - 2

            # NEW: use the eligibility hook (encapsulates session + ATR gates)
            if pos is None or pos.get('qty', 0) == 0:
                if not self.wants_entry_at(env, symbol, df, idx_last, idx_prev):
                    continue
            # else: manage exits below

            # (Keep existing ATR, entry price, params, etc.)
            atr = self._get_prev_atr(df, idx_prev)
            if atr is None or atr < self.min_atr_threshold:
                continue

            # Single-cell open read (cheaper than Series)
            try:
                i_open = df.columns.get_loc(self._COL_OPEN)
                entry_price = float(df.iat[idx_last, i_open])
            except Exception:
                entry_price = float(df[self._COL_OPEN].iloc[idx_last])

            params = self._get_asset_params(env, symbol)
            if params['initial_margin'] is None or params['initial_margin'] <= 0:
                continue  # prevent over-leverage due to missing config

            # Lightweight context (kept for compatibility with existing handlers)
            ctx = {
                'env': env, 'symbol': symbol, 'df': df,
                'idx_last': idx_last, 'idx_prev': idx_prev,
                'pos': pos, 'atr': atr, 'entry_price': entry_price,
                'equity': equity, 'available_equity': available_equity, 'p': params
            }

            if pos is None or pos.get('qty', 0) == 0:
                pass
                self._handle_entry(ctx)
            else:
                self._handle_exit(ctx)

    # ----------------- sizing & params -----------------

    def compute_safe_position_size(
        self,
        side: str,
        entry_price: float,
        atr: float,
        equity: float,
        risk_percent: float,
        tick_size: float,
        tick_value: float,
        initial_margin: float | None = None,
        exit_strategy: ExitStrategy | None = None,
        df=None,
        symbol=None,
        slippage_ticks: int = 0,
        slippage_pct: float = 0.0,
        commission_per_contract: float = 0.0,
        fee_per_trade: float = 0.0,
        precomputed_stop_loss_price: float | None = None,
    ):
        """
        Size by %risk with realistic buffers:
        - Uses the provided (exit_)strategy to determine the stop price,
          or a precomputed stop if given (to keep sizing consistent with placed orders).
        - Risk/contract inflated by expected slippage (entry+stop) and costs.
        - Flat per-trade fee apportioned per contract via a two-pass estimate.
        """
        risk_amount = equity * risk_percent
        strategy = exit_strategy or self.exit_strategy

        # Stop: use precomputed SL if provided; else ask the strategy.
        if precomputed_stop_loss_price is not None:
            stop_loss_price = round_to_tick(precomputed_stop_loss_price, tick_size)
        else:
            sl_raw = strategy.initial_stop(
                side, entry_price, atr, df=df, symbol=symbol, tick_size=tick_size
            )
            stop_loss_price = round_to_tick(sl_raw, tick_size)

        ticks_at_risk = abs(entry_price - stop_loss_price) / max(tick_size, 1e-12)
        risk_per_contract = ticks_at_risk * tick_value  # currency risk per contract

        # Conservative buffer to mirror engine slippage policy (ticks preferred; else pct)
        buffer_wo_fee = 0.0
        if slippage_ticks and slippage_ticks > 0:
            buffer_wo_fee += (slippage_ticks * 2) * tick_value  # entry + stop
        elif slippage_pct and slippage_pct > 0.0:
            dollars_per_point = tick_value / max(tick_size, 1e-12)
            buffer_wo_fee += (abs(entry_price) * slippage_pct * dollars_per_point) * 2

        # Costs: commission per contract now; flat fee apportioned later
        buffer_wo_fee += float(commission_per_contract)

        effective_rpc_wo_fee = risk_per_contract + buffer_wo_fee
        if effective_rpc_wo_fee <= 0 or effective_rpc_wo_fee > risk_amount:
            return 0, stop_loss_price, risk_per_contract

        # First pass qty (ignoring flat per-trade fee)
        qty_est = max(1, int(risk_amount // max(effective_rpc_wo_fee, 1e-12)))

        # Apportion flat fee across estimated contracts
        per_contract_fee = (float(fee_per_trade) / qty_est) if fee_per_trade else 0.0
        effective_rpc = effective_rpc_wo_fee + per_contract_fee

        qty = int(risk_amount // max(effective_rpc, 1e-12))
        return qty, stop_loss_price, risk_per_contract

    def _get_asset_params(self, env, symbol):
        cfg = getattr(env.api, 'config', {})
        asset = next((a for a in getattr(env.api, 'assets', []) if a.get('symbol') == symbol), {})

        # Robust initial_margin fallback to cfg when asset doesn't define it
        im = asset.get('initial_margin')
        if im is None:
            im = cfg.get('initial_margin')

        return {
            'tick_size': float(
                asset.get('tick_size', 0.25)
            ),
            'tick_value': float(
                asset.get('tick_value', 5.0)
            ),
            'initial_margin': float(im) if im is not None else None,
            'slippage_ticks': int(
                asset.get('slippage_ticks', cfg.get('slippage_ticks', 0))
            ),
            'slippage_pct': float(
                asset.get('slippage_pct', cfg.get('slippage_pct', 0.0))
            ),
            'commission_per_contract': float(
                asset.get('commission_per_contract', cfg.get('commission_per_contract', 0.0))
            ),
            'fee_per_trade': float(
                asset.get('fee_per_trade', cfg.get('fee_per_trade', 0.0))
            ),
        }

    # ----------------- session helpers (ported) -----------------

    def _to_eastern(self, ts):
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t.tz_convert(self.session_tz)

    def _is_early_close(self, d):
        try:
            d = d if isinstance(d, _date) else d.date()
        except Exception:
            return False
        return d in self.early_close_dates

    def _rth_end_for(self, d):
        return self.rth_early_end if self._is_early_close(d) else self.rth_end

    def _is_trading_time_et(self, ts_et: pd.Timestamp) -> bool:
        tt = ts_et.time()
        return (tt >= self.rth_start) and (tt < self._rth_end_for(ts_et.date()))

    def _is_trading_day_et(self, ts_et: pd.Timestamp) -> bool:
        d = ts_et.date()
        if ts_et.weekday() >= 5:
            return False
        if not self.skip_us_holidays:
            return True
        return not self._is_us_holiday(d)

    def _is_us_holiday(self, d: _date) -> bool:
        yr = d.year
        if yr not in self._holiday_cache:
            self._holiday_cache[yr] = self._build_us_holiday_set(yr)
        return d in self._holiday_cache[yr]

    def _build_us_holiday_set(self, year: int):
        def _observed(dt: _date):
            if dt.weekday() == 5:  # Sat -> Fri
                return dt - _td(days=1)
            if dt.weekday() == 6:  # Sun -> Mon
                return dt + _td(days=1)
            return dt

        def _nth_weekday(month, weekday, n):
            d0 = _date(year, month, 1)
            add = (weekday - d0.weekday()) % 7
            return d0 + _td(days=add + 7*(n-1))

        def _last_weekday(month, weekday):
            if month < 12:
                d = _date(year, month+1, 1) - _td(days=1)
            else:
                d = _date(year, 12, 31)
            while d.weekday() != weekday:
                d -= _td(days=1)
            return d

        def _easter_sunday(y):
            a = y % 19
            b = y // 100
            c = y % 100
            d = b // 4
            e = b % 4
            f = (b + 8) // 25
            g = (b - f + 1) // 3
            h = (19*a + b - d - g + 15) % 30
            i = c // 4
            k = c % 4
            l = (32 + 2*e + 2*i - h - k) % 7
            m = (a + 11*h + 22*l) // 451
            month = (h + l - 7*m + 114) // 31
            day = ((h + l - 7*m + 114) % 31) + 1
            return _date(y, month, day)

        H = set()
        H.add(_observed(_date(year, 1, 1)))                  # New Year's Day
        H.add(_nth_weekday(1, 0, 3))                         # MLK Day
        H.add(_nth_weekday(2, 0, 3))                         # Presidents' Day
        H.add(_easter_sunday(year) - _td(days=2))            # Good Friday
        H.add(_last_weekday(5, 0))                           # Memorial Day
        if year >= 2021:
            H.add(_observed(_date(year, 6, 19)))             # Juneteenth (observed)
        H.add(_observed(_date(year, 7, 4)))                  # Independence Day
        H.add(_nth_weekday(9, 0, 1))                         # Labor Day
        H.add(_nth_weekday(11, 3, 4))                        # Thanksgiving
        H.add(_observed(_date(year, 12, 25)))                # Christmas
        return H

    def _within_flatten_window(self, ts_et: pd.Timestamp) -> bool:
        if ts_et.weekday() >= 5:
            return False
        cutoff_dt = ts_et.replace(hour=self.maintenance_cutoff_et.hour,
                                  minute=self.maintenance_cutoff_et.minute,
                                  second=0, microsecond=0)
        start_dt = cutoff_dt - self.flatten_lead
        return (ts_et >= start_dt) and (ts_et < cutoff_dt)

    def _flatten_all_positions(self, env):
        portfolio = env.get_portfolio()
        for symbol, pos in portfolio.get('positions', {}).items():
            qty = pos.get('qty', 0)
            if qty == 0:
                continue
            side = 'sell' if qty > 0 else 'buy'
            env.place_order({
                'symbol': symbol,
                'side': side,
                'qty': abs(qty),
                'order_type': 'market'
            })

    def _should_flatten_now(self, env) -> bool:
        ts = self._current_timestamp_utc(env)
        if ts is None or not self.flatten_before_maintenance:
            return False
        ts_et = self._to_eastern(ts)
        if not self._within_flatten_window(ts_et):
            return False
        if self._last_flatten_date == ts_et.date():
            return False
        self._last_flatten_date = ts_et.date()
        return True

    def _validate_bar_data(self, df) -> bool:
        if df is None or len(df) < 2:
            return False
        idx_last = len(df) - 1
        idx_prev = idx_last - 1
        # Cheaper single-cell checks for NaN
        try:
            i_open = df.columns.get_loc(self._COL_OPEN)
            i_close = df.columns.get_loc(self._COL_CLOSE)
        except KeyError:
            return False
        if pd.isna(df.iat[idx_last, i_open]):
            return False
        if pd.isna(df.iat[idx_prev, i_close]):
            return False
        return True

    def _session_allows_entry(self, df, idx_last) -> bool:
        # Use .iat when possible to avoid Series allocation
        try:
            i_date = df.columns.get_loc(self._COL_DATE)
            ts = df.iat[idx_last, i_date]
        except Exception:
            ts = df[self._COL_DATE].iloc[idx_last]
        ts_et = self._to_eastern(ts)
        return self._is_trading_day_et(ts_et) and self._is_trading_time_et(ts_et)

    def _prepare_context(self, **kw):
        """
        Normalize all values we reuse across entry/exit paths.
        """
        return {
            'env': kw['env'],
            'symbol': kw['symbol'],
            'df': kw['df'],
            'idx_last': kw['idx_last'],
            'idx_prev': kw['idx_prev'],
            'pos': kw['pos'],
            'atr': float(kw['atr']),
            'entry_price': float(kw['entry_price']),
            'equity': float(kw['equity']),
            'available_equity': float(kw['available_equity']),
            'p': kw['p'],  # tick_size/tick_value/margin/slippage/fees
        }

    def _handle_entry(self, ctx):
        env, symbol, df = ctx['env'], ctx['symbol'], ctx['df']
        idx_last, idx_prev = ctx['idx_last'], ctx['idx_prev']
        atr, entry_price, equity, avail_eq = ctx['atr'], ctx['entry_price'], ctx['equity'], ctx['available_equity']
        p = ctx['p']

        side = self.decide_side(env, symbol, df, idx_last, idx_prev)
        if side not in ('buy', 'sell'):
            return

        # Ask exit strategy for initial SL/TP (SL/TP are already tick-rounded)
        stop_loss_price, take_profit = self.exit_strategy.initial_levels(
            side=side,
            entry_price=entry_price,
            atr=atr,
            tick_size=p['tick_size'],
            df=df,
            symbol=symbol,
        )

        qty_risk, _, risk_per_contract = self.compute_safe_position_size(
            side=side,
            entry_price=entry_price,
            atr=atr,
            equity=equity,
            risk_percent=self.base_risk_percent,
            tick_size=p['tick_size'],
            tick_value=p['tick_value'],
            initial_margin=p['initial_margin'],
            exit_strategy=self.exit_strategy,
            df=df,
            symbol=symbol,
            slippage_ticks=p['slippage_ticks'],
            slippage_pct=p['slippage_pct'],
            commission_per_contract=p['commission_per_contract'],
            fee_per_trade=p['fee_per_trade'],
            precomputed_stop_loss_price=stop_loss_price,  # ensure sizing matches placed SL
        )

        # Margin cap
        max_qty_margin = int(avail_eq // p['initial_margin']) if p['initial_margin'] else self.max_qty
        qty = int(min(qty_risk, max_qty_margin, self.max_qty))
        if qty < 1:
            return

        # Legacy TP fallback if strategy didn't set one
        if take_profit is None and self._legacy_tp_rr_multiple and risk_per_contract > 0:
            price_move = (risk_per_contract / max(p['tick_value'], 1e-12)) * p['tick_size']
            if side == 'buy':
                take_profit = round_to_tick(entry_price + self._legacy_tp_rr_multiple * price_move, p['tick_size'])
            else:
                take_profit = round_to_tick(entry_price - self._legacy_tp_rr_multiple * price_move, p['tick_size'])

        env.place_order({
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'order_type': 'market',
            'stop_loss': stop_loss_price,
            'take_profit': take_profit,
        })

    def _handle_exit(self, ctx):
        env, symbol, df = ctx['env'], ctx['symbol'], ctx['df']
        idx_prev = ctx['idx_prev']
        pos, atr, p = ctx['pos'], ctx['atr'], ctx['p']

        qty = pos.get('qty', 0)
        side = 'buy' if qty > 0 else 'sell'
        stop_loss = pos.get('stop_loss_price')
        entry_px = pos.get('avg_entry_price', 0.0)
        if stop_loss is None or entry_px == 0:
            return

        # Trail using the previous CLOSED bar (fast single-cell read)
        try:
            i_close = df.columns.get_loc(self._COL_CLOSE)
            current_price_for_trailing = float(df.iat[idx_prev, i_close])
        except Exception:
            current_price_for_trailing = float(df[self._COL_CLOSE].iloc[idx_prev])

        new_stop = self.exit_strategy.update_stop(
            side=side,
            stop_loss=stop_loss,
            entry_price=entry_px,
            current_price=current_price_for_trailing,
            atr=atr,
            tick_size=p['tick_size'],
            df=df,
            symbol=symbol,
        )
        if new_stop is not None:
            env.modify_stop_loss(symbol, round_to_tick(new_stop, p['tick_size']))

    # ----------------- small utilities -----------------

    def _get_prev_atr(self, df: pd.DataFrame, idx_prev: int) -> float | None:
        if df is None or idx_prev < 0:
            return None

        # Preferred names first
        def _probe_at(i: int) -> float | None:
            if self.atr_preference:
                for name in self.atr_preference:
                    if name in df.columns:
                        v = df[name].iloc[i]
                        if pd.notna(v):
                            return float(v)
            # cached candidates
            cols_sig = tuple(df.columns)
            candidates = self._atr_cols_cache.get(cols_sig)
            if candidates is None:
                candidates = [c for c in df.columns if isinstance(c, str) and c.startswith('atr_')]
                self._atr_cols_cache[cols_sig] = candidates
            for c in candidates:
                v = df[c].iloc[i]
                if pd.notna(v):
                    return float(v)
            return None

        # 1) try the previous CLOSED bar (original behavior)
        v = _probe_at(idx_prev)
        if v is not None:
            return v

        # 2) NEW: fall back to the live row (carried-forward indicators, still no look-ahead)
        idx_last = min(idx_prev + 1, len(df) - 1)
        return _probe_at(idx_last)

    def _current_timestamp_utc(self, env) -> pd.Timestamp | None:
        # Prefer engine’s primary clock; fall back to any asset’s latest 'date'
        ts = getattr(env.api, "current_timestamp", None)
        if ts is not None:
            try:
                return pd.to_datetime(int(ts), unit='s', utc=True)
            except Exception:
                pass
        for s in env.get_asset_list():
            df0 = env.get_latest_data(s)
            if df0 is not None and not df0.empty and 'date' in df0.columns:
                return pd.to_datetime(df0['date'].iloc[-1], utc=True)
        return None

