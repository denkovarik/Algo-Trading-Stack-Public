# bots/base_strategy_bot.py
import numpy as np
import pandas as pd
from datetime import date as _date, time as _time, timedelta as _td, datetime as _datetime
from classes.API_Interface import round_to_tick
from bots.exit_strategies import ExitStrategy, TrailingATRExit
from TradeStation.closures import load_closures  # <-- Loads closures CSV


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
        exit_strategy: ExitStrategy | None = None,
        base_risk_percent: float = 0.01,
        max_qty: int = 100,
        max_margin_pct: float = 0.50,
        atr_preference: tuple[str, ...] | None = None,
        session_tz: str = "America/New_York",
        rth_start: _time = _time(9, 30),
        rth_end: _time = _time(16, 0),
        enforce_sessions: bool = False,
        day_trade_only: bool = False,
        boundary_minutes: int = 5,
        closures_csv: str | None = None,
        seed: int | None = None,
        min_atr_threshold: float = 1e-6,
        tp_rr_multiple: float | None = None,
        **_unused,
    ):
        self.base_risk_percent = float(base_risk_percent)
        self.max_qty = int(max_qty)
        self.max_margin_pct = float(max_margin_pct)
        self.atr_preference = tuple(atr_preference) if atr_preference else None

        # Exit strategy
        if exit_strategy is not None:
            self.exit_strategy = exit_strategy
        else:
            self.exit_strategy = TrailingATRExit(atr_multiple=3.0)
        self._legacy_tp_rr_multiple = tp_rr_multiple

        # Session config
        self.session_tz = session_tz
        self.rth_start = rth_start
        self.rth_end = rth_end
        self.day_trade_only = bool(day_trade_only)
        self.enforce_sessions = bool(enforce_sessions)

        # Boundary windows
        self.boundary_minutes = int(boundary_minutes)
        self.boundary_window = _td(minutes=self.boundary_minutes)

        # Load closures CSV
        self.closures_df: pd.DataFrame | None = None
        if closures_csv:
            try:
                self.closures_df = load_closures(closures_csv)
                #print(f"[BaseStrategyBot] Loaded closures from {closures_csv} ({len(self.closures_df)} rows)")
            except Exception as e:
                #print(f"[BaseStrategyBot] Failed to load closures CSV: {e}")
                self.closures_df = None

        # Eligibility
        self.min_atr_threshold = float(min_atr_threshold)

        # Runtime
        self.state: dict = {}
        self._atr_cols_cache: dict[tuple, list[str]] = {}
        self._rng = np.random.default_rng(seed)

        if _unused:
            pass
            #print(f"[BaseStrategyBot] Ignoring unused kwargs: {sorted(_unused.keys())}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self):
        self.state.clear()
        self._atr_cols_cache.clear()

    # ------------------------------------------------------------------
    # Subclass Hook
    # ------------------------------------------------------------------
    def decide_side(self, env, symbol, df, idx_last, idx_prev) -> str | None:
        return 'buy' if self._rng.random() < 0.5 else 'sell'

    # ------------------------------------------------------------------
    # NEW: Entry Eligibility (Base Checks)
    # ------------------------------------------------------------------
    def wants_entry_at(self, env, symbol, df, idx_last, idx_prev) -> bool:
        if df is None or len(df) < 2:
            return False
        if idx_prev < 0 or idx_last >= len(df):
            return False
        if "open" not in df.columns or "close" not in df.columns:
            return False
        if pd.isna(df["open"].iloc[idx_last]) or pd.isna(df["close"].iloc[idx_prev]):
            return False
        if not self._session_allows_entry(df, idx_last):
            return False
        if not self.enforce_sessions and self.day_trade_only:
            ts_et = self._to_eastern(df.iloc[idx_last]["date"])
            t = ts_et.time()
            if not (_time(9, 30) <= t < _time(15, 55)):
                return False
        atr = self._get_prev_atr(df, idx_prev)
        if atr is None or atr < self.min_atr_threshold:
            return False
        return True

    def _get_session_times_from_closures(self, day: _date) -> tuple[_time | None, _time | None]:
        if self.closures_df is None:
            return None, None

        day_str = day.strftime('%Y-%m-%d')
        row = self.closures_df[self.closures_df['Date'] == day_str]

        if row.empty:
            return None, None  # ← NORMAL DAY

        r = row.iloc[0]
        open_t = self._parse_time_from_closure(r.get('open_time_et', ''))
        close_t = self._parse_time_from_closure(r.get('close_time_et', ''))

        # If no open time → holiday → no session
        if open_t is None:
            return None, None

        return open_t, close_t

    def _parse_time_from_closure(self, s: str) -> _time | None:
        if not s or pd.isna(s):
            return None
        s = str(s).strip()
        if ' ' in s:
            s = s.split()[0]
        try:
            h, m = map(int, s.split(':'))
            return _time(h, m)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Session Helpers
    # ------------------------------------------------------------------
    def _to_eastern(self, ts):
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t.tz_convert(self.session_tz)

    def _is_trading_time_et(self, ts_et: pd.Timestamp) -> bool:
        """
        True if current time is within RTH (after open, before close).
        Uses closures CSV for session times, falls back to default.
        """
        tt = ts_et.time()
        day = ts_et.date()

        # Get session close from CSV, or use default 16:00
        _, close_time = self._get_session_times_from_closures(day)
        close_time = close_time or self.rth_end

        return self.rth_start <= tt < close_time

    def _is_trading_day_et(self, ts_et: pd.Timestamp) -> bool:
        d = ts_et.date()
        if ts_et.weekday() >= 5:  # weekend
            return False
        return not self._is_us_holiday(d)  # ← CSV decides

    # ------------------------------------------------------------------
    # on_bar – main event loop
    # ------------------------------------------------------------------
    def on_bar(self, env):
        portfolio = env.get_portfolio()
        equity = portfolio.get('total_equity', 100_000)
        available_equity = portfolio.get('available_equity', equity)

        if hasattr(self.exit_strategy, "ensure_models_loaded"):
            try:
                self.exit_strategy.ensure_models_loaded(env)
            except Exception:
                pass

        for symbol in env.get_asset_list():
            df = env.get_latest_data(symbol)
            if not self._validate_bar_data(df):
                continue

            #print("[DEBUG] In BaseStrategyBot.on_bar from call df = env.get_latest_data(symbol)")
            #print(df)
            idx_last = len(df) - 1
            idx_prev = idx_last - 1

            try:
                ts_et = self._to_eastern(df.iloc[idx_last]['date'])
            except Exception:
                #print("Exception converting date")
                continue
            
            if self.enforce_sessions:
                # ------------------------------------------------------------------
                # 1. PRE-CLOSE: 3:55 PM → Flatten + Block
                # ------------------------------------------------------------------
                if self._is_pre_close_boundary(df, idx_last):
                    #print(f"[FLATTEN] {symbol} @ {ts_et.strftime('%Y-%m-%d %H:%M:%S')} ET — PRE-CLOSE")
                    pos = env.get_positions().get(symbol, {})
                    if pos.get('qty', 0) != 0:
                        self._flatten_all_positions(env)
                    #print(f"[SKIP] {symbol} @ {ts_et.strftime('%Y-%m-%d %H:%M:%S')} ET — PRE-CLOSE")
                    continue

                # ------------------------------------------------------------------
                # 2. POST-RTH BLACKOUT: 4:00 PM → 4:05 PM
                # ------------------------------------------------------------------
                if self._in_post_rth_blackout(df, idx_last):
                    #print(f"[SKIP] {symbol} @ {ts_et.strftime('%Y-%m-%d %H:%M:%S')} ET — POST-RTH BLACKOUT")
                    continue

            pos = env.get_positions().get(symbol, {})
            if pos.get('qty', 0) == 0:
                if self.wants_entry_at(env, symbol, df, idx_last, idx_prev):
                    self._handle_entry(env, symbol, df, idx_last, idx_prev, equity, available_equity)
            else:
                self._handle_exit(env, symbol, df, idx_prev, pos)

    # ----------------------------------------------------------------------
    # POST-RTH BLACKOUT: 4:00 PM → 4:05 PM
    # ----------------------------------------------------------------------
    def _in_post_rth_blackout(self, df, idx_last) -> bool:
        """True from 4:00 PM to 4:05 PM (5 min post-close buffer)."""
        if df is None or idx_last >= len(df):
            return False
        try:
            ts = df.iloc[idx_last]['date']
        except Exception:
            return False

        ts_et = self._to_eastern(ts)
        day = ts_et.date()

        # Use CSV close time or default 16:00
        _, close_time = self._get_session_times_from_closures(day)
        close_time = close_time or _time(16, 0)

        close_dt = pd.Timestamp(_datetime.combine(day, close_time)).tz_localize("America/New_York")
        blackout_start = close_dt
        blackout_end = close_dt + _td(minutes=5)  # 4:05 PM

        return blackout_start <= ts_et < blackout_end


    # ----------------------------------------------------------------------
    # PRE-CLOSE BOUNDARY: 3:55 PM → 4:00 PM
    # ----------------------------------------------------------------------
    def _is_pre_close_boundary(self, df, idx_last) -> bool:
        if df is None or idx_last >= len(df):
            return False
        try:
            ts = df.iloc[idx_last]['date']
        except:
            return False

        ts_et = self._to_eastern(ts)
        day = ts_et.date()

        # CSV has no row for normal days → use default
        _, close_time = self._get_session_times_from_closures(day)
        close_time = close_time or _time(16, 0)  # 4:00 PM

        close_dt = pd.Timestamp(_datetime.combine(day, close_time)).tz_localize("America/New_York")
        pre_close_start = close_dt - _td(minutes=5)

        return pre_close_start <= ts_et < close_dt 

    # ----------------------------------------------------------------------
    # HOLIDAY CHECK: Safe fallback for missing dates
    # ----------------------------------------------------------------------
    def _is_us_holiday(self, d: _date) -> bool:
        """True only if CSV explicitly marks a full holiday."""
        if self.closures_df is None:
            raise RuntimeError("closures_df not loaded")

        day_str = d.strftime('%Y-%m-%d')
        row = self.closures_df[self.closures_df['Date'] == day_str]

        if row.empty:
            return False  # Not in CSV → regular trading day

        r = row.iloc[0]
        open_time = r.get('open_time_et')
        return pd.isna(open_time) or not str(open_time).strip()
                
    def _flatten_all_positions(self, env):
        for symbol in env.get_asset_list():
            pos = env.get_positions().get(symbol, {})
            qty = pos.get('qty', 0)
            if qty != 0:
                env.place_order({
                    'symbol': symbol,
                    'side': 'sell' if qty > 0 else 'buy',
                    'qty': abs(qty),
                    'order_type': 'market'
                })
                
    # ------------------------------------------------------------------
    # Entry / Exit
    # ------------------------------------------------------------------
    def _handle_entry(self, env, symbol, df, idx_last, idx_prev, equity, available_equity):
        atr = self._get_prev_atr(df, idx_prev)
        if atr is None or atr < self.min_atr_threshold:
            return
        try:
            entry_price = float(df[self._COL_OPEN].iloc[idx_last])
        except Exception:
            return
        p = self._get_asset_params(env, symbol)

        if p['initial_margin'] is None or p['initial_margin'] <= 0:
            return

        side = self.decide_side(env, symbol, df, idx_last, idx_prev)
        if side not in ('buy', 'sell'):
            return

        stop_loss_price, take_profit = self.exit_strategy.initial_levels(
            side=side, entry_price=entry_price, atr=atr,
            tick_size=p['tick_size'], df=df, symbol=symbol
        )

        qty_risk, _, _ = self.compute_safe_position_size(
            side=side, entry_price=entry_price, atr=atr, equity=equity,
            risk_percent=self.base_risk_percent, tick_size=p['tick_size'],
            tick_value=p['tick_value'], initial_margin=p['initial_margin'],
            exit_strategy=self.exit_strategy, df=df, symbol=symbol,
            slippage_ticks=p['slippage_ticks'], slippage_pct=p['slippage_pct'],
            commission_per_contract=p['commission_per_contract'],
            fee_per_trade=p['fee_per_trade'],
            precomputed_stop_loss_price=stop_loss_price,
        )
        
        original_stop_offset = entry_price - stop_loss_price
        if original_stop_offset < 0:
            original_stop_offset *= -1

        current_qty = env.get_positions().get(symbol, {}).get('qty', 0)
        if p['initial_margin'] and p['initial_margin'] > 0:
            usable_margin = equity * self.max_margin_pct
            max_allowed_total = int(usable_margin // p['initial_margin'])
            max_allowed_new = max(0, max_allowed_total - abs(current_qty))
        else:
            max_allowed_new = self.max_qty

        qty = int(min(qty_risk, max_allowed_new, self.max_qty))
        
        if qty < 1:
            return

        ts_et = self._to_eastern(df.iloc[idx_last]['date'])
        env.place_order({
            'symbol': symbol, 'side': side, 'qty': qty,
            'order_type': 'market', 'stop_loss': stop_loss_price, 
            'take_profit': take_profit, 'original_stop_offset' : original_stop_offset
        })

    def _handle_exit(self, env, symbol, df, idx_prev, pos):
        atr = self._get_prev_atr(df, idx_prev)
        if atr is None:
            return
        p = self._get_asset_params(env, symbol)
        qty = pos.get('qty', 0)
        side = 'buy' if qty > 0 else 'sell'
        stop_loss = pos.get('stop_loss_price')
        entry_px = pos.get('avg_entry_price', 0.0)
        if stop_loss is None or entry_px == 0:
            return
        try:
            current_price = float(df[self._COL_CLOSE].iloc[idx_prev])
        except Exception:
            return
        new_stop = self.exit_strategy.update_stop(
            side=side, stop_loss=stop_loss, entry_price=entry_px,
            current_price=current_price, atr=atr, tick_size=p['tick_size'],
            df=df, symbol=symbol
        )
        if new_stop is not None:
            env.modify_stop_loss(symbol, round_to_tick(new_stop, p['tick_size']))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _validate_bar_data(self, df) -> bool:
        if df is None or len(df) < 2:
            return False
        idx_last = len(df) - 1
        idx_prev = idx_last - 1
        if "open" not in df.columns or "close" not in df.columns:
            return False
        if pd.isna(df["open"].iloc[idx_last]) or pd.isna(df["close"].iloc[idx_prev]):
            return False
        return True

    def _get_prev_atr(self, df: pd.DataFrame, idx_prev: int) -> float | None:
        if df is None or idx_prev < 0:
            return None
        def _probe(i: int):
            if self.atr_preference:
                for name in self.atr_preference:
                    if name in df.columns:
                        v = df[name].iloc[i]
                        if pd.notna(v):
                            return float(v)
            cols_sig = tuple(df.columns)
            candidates = self._atr_cols_cache.get(cols_sig)
            if candidates is None:
                candidates = [c for c in df.columns if c.startswith('atr_')]
                self._atr_cols_cache[cols_sig] = candidates
            for c in candidates:
                v = df[c].iloc[i]
                if pd.notna(v):
                    return float(v)
            return None
        v = _probe(idx_prev)
        return v if v is not None else _probe(min(idx_prev + 1, len(df) - 1))

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
        risk_amount = equity * risk_percent
        strategy = exit_strategy or self.exit_strategy
        if precomputed_stop_loss_price is not None:
            stop_loss_price = round_to_tick(precomputed_stop_loss_price, tick_size)
        else:
            sl_raw = strategy.initial_stop(
                side, entry_price, atr, df=df, symbol=symbol, tick_size=tick_size
            )
            stop_loss_price = round_to_tick(sl_raw, tick_size)
        ticks_at_risk = abs(entry_price - stop_loss_price) / max(tick_size, 1e-12)
        risk_per_contract = ticks_at_risk * tick_value
        buffer_wo_fee = 0.0
        if slippage_ticks and slippage_ticks > 0:
            buffer_wo_fee += (slippage_ticks * 2) * tick_value
        elif slippage_pct and slippage_pct > 0.0:
            dollars_per_point = tick_value / max(tick_size, 1e-12)
            buffer_wo_fee += (abs(entry_price) * slippage_pct * dollars_per_point) * 2
        buffer_wo_fee += float(commission_per_contract)
        effective_rpc_wo_fee = risk_per_contract + buffer_wo_fee
        if effective_rpc_wo_fee <= 0 or effective_rpc_wo_fee > risk_amount:
            return 0, stop_loss_price, risk_per_contract
        qty_est = max(1, int(risk_amount // max(effective_rpc_wo_fee, 1e-12)))
        per_contract_fee = (float(fee_per_trade) / qty_est) if fee_per_trade else 0.0
        effective_rpc = effective_rpc_wo_fee + per_contract_fee
        qty = int(risk_amount // max(effective_rpc, 1e-12))
        return qty, stop_loss_price, risk_per_contract

    def _get_asset_params(self, env, symbol):
        cfg = getattr(env.api, 'config', {})
        asset = next((a for a in getattr(env.api, 'assets', []) if a.get('symbol') == symbol), {})
        im = asset.get('initial_margin', cfg.get('initial_margin'))
        return {
            'tick_size': float(asset.get('tick_size', 0.25)),
            'tick_value': float(asset.get('tick_value', 5.0)),
            'initial_margin': float(im) if im is not None else None,
            'slippage_ticks': int(asset.get('slippage_ticks', cfg.get('slippage_ticks', 0))),
            'slippage_pct': float(asset.get('slippage_pct', cfg.get('slippage_pct', 0.0))),
            'commission_per_contract': float(asset.get('commission_per_contract', cfg.get('commission_per_contract', 0.0))),
            'fee_per_trade': float(asset.get('fee_per_trade', cfg.get('fee_per_trade', 0.0))),
        }

    def _session_allows_entry(self, df, idx_last) -> bool:
        if not self.enforce_sessions:
            return True
        try:
            ts = df.iloc[idx_last]['date']
        except Exception:
            ts = df[self._COL_DATE].iloc[idx_last]
        ts_et = self._to_eastern(ts)
        return self._is_trading_day_et(ts_et) and self._is_trading_time_et(ts_et)
        
