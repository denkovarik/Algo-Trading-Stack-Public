from abc import ABC, abstractmethod, ABCMeta
import pandas as pd
import yaml
import os
import numpy as np
from PyQt5 import QtCore
import time
import math
from collections import defaultdict, deque

from classes.API_Interface import APIInterface
from classes.API_Interface import round_to_tick, Position, Order, MetaQObjectABC, ensure_timestamp


class BacktesterEngine(QtCore.QObject, APIInterface, metaclass=MetaQObjectABC):
    """
    A simulated broker interface for backtesting purposes.
    Drives bar-by-bar execution and fulfills the APIInterface contract.

    NEW: live stats tracked as the backtest progresses.
    - Per-asset equity series (realized+unrealized) and max drawdown online
    - Per-asset trade campaigns (win rate, counts, gross wins/losses, expectancy)
    - Portfolio drawdown online
    """
    bar_updated = QtCore.pyqtSignal(pd.DataFrame)
    bar_advanced = QtCore.pyqtSignal()
    backtest_finished = QtCore.pyqtSignal()
    bar_closed = QtCore.pyqtSignal(pd.DataFrame)

    def __init__(self, config_path):
        super().__init__()
        self._in_open_phase = False                  # open-phase valuation flag
        self.live_price_mode = "open_only"
        self._rng = np.random.default_rng(0)
        self.config_path = config_path
        self.connected = False
        self.orders = {}
        self.order_log = []
        self.next_order_id = 1
        self.positions = {}
        self.trade_log = []                          # per-fill audit trail 
        self.config = {}
        self.current_index = 0
        self.is_running = False
        self.timer = None
        self.initial_cash = 0.0
        self.cash = self.initial_cash
        self.equity_history = []                     # portfolio equity by bar 
        self.equity_time_history = []
        self.include_slippage = False
        self.used_margin = 0.0
        self.df = pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume", "timestamp"]
        )

        # ==== NEW: live stats containers ====
        self._per_asset_realized_cum = defaultdict(float)   # realized P&L by symbol
        self._equity_by_symbol_history = {} # symbol -> [eq0, eq1, ...]
        self._equity_time_history = []  # mirrors portfolio times
        self._per_asset_dd_state = {}   # symbol -> {'peak': float, 'max_dd': float}
        self._portfolio_dd_state = {'peak': 0.0, 'max_dd': 0.0}

        # Campaign tracking per symbol:
        # Tracks a "trade" from first open to flat (win-rate, expectancy, etc.)
        # symbol -> dict(active, start_time, baseline_realized, side, open_qty_sum)
        self._campaign = {}  
        self._stats_per_asset = {}  # symbol -> dict of stats (wins, losses, etc.)

        self.load_backtest_config(self.config_path)

        self._init_stats_containers()

    # ----------------------- STATS: initialization & helpers -----------------------

    def _init_stats_containers(self):
        symbols = self.get_asset_list()
        self._per_asset_realized_cum = defaultdict(float)
        self._equity_by_symbol_history = {s: [] for s in symbols}
        self._equity_time_history = []

        # seed t0 similar to portfolio equity_time_history[0]
        t0 = None
        primary_df = self.assets[0]['data'] if self.assets else None
        if primary_df is not None and len(primary_df) > 0:
            t0 = primary_df.iloc[0]['date']

        self._equity_time_history.append(t0)
        for s in symbols:
            self._equity_by_symbol_history[s].append(0.0)

        self._per_asset_dd_state = {s: {'peak': 0.0, 'max_dd': 0.0} for s in symbols}
        self._portfolio_dd_state = {'peak': 0.0, 'max_dd': 0.0}
        self._campaign = {
            s: {
                'active': False, 'start_time': None, 'baseline_realized': 0.0, 
                'side': None, 'open_qty_sum': 0
               }
            for s in symbols
        }
        self._stats_per_asset = {
            s: {
                'trades': 0, 'wins': 0, 'losses': 0, 'long_trades': 0, 'short_trades': 0,
                'gross_wins': 0.0, 'gross_losses': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
                'expectancy': 0.0, 'commission_total': 0.0, 'fee_total': 0.0, 'max_dd': 0.0
            } for s in symbols
        }
            
    def _ensure_symbol_stats(self, symbol: str):
        """
        Make sure all per-asset stats containers exist for `symbol`.
        Keeps list lengths aligned with the internal _equity_time_history.
        Safe to call many times.
        """
        # Create realized bucket (defaultdict guarantees a key, but touch it anyway)
        _ = self._per_asset_realized_cum[symbol]

        # Stats row
        if symbol not in self._stats_per_asset:
            self._stats_per_asset[symbol] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'long_trades': 0,
                'short_trades': 0,
                'gross_wins': 0.0,
                'gross_losses': 0.0,   # negative
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'expectancy': 0.0,
                'commission_total': 0.0,
                'fee_total': 0.0,
                'max_dd': 0.0,
            }

        # Campaign row
        if symbol not in self._campaign:
            self._campaign[symbol] = {
                'active': False,
                'start_time': None,
                'baseline_realized': self._per_asset_realized_cum[symbol],
                'side': None,
                'open_qty_sum': 0,
            }

        # Drawdown state
        if symbol not in self._per_asset_dd_state:
            self._per_asset_dd_state[symbol] = {'peak': 0.0, 'max_dd': 0.0}

        # Equity-by-symbol series (keep length aligned to the time axis)
        need_len = len(self._equity_time_history) if self._equity_time_history else 1
        if symbol not in self._equity_by_symbol_history:
            self._equity_by_symbol_history[symbol] = [0.0] * need_len
        else:
            # If we created the time axis first (or later), pad/truncate to match
            cur_len = len(self._equity_by_symbol_history[symbol])
            if cur_len < need_len:
                self._equity_by_symbol_history[symbol].extend([0.0] * (need_len - cur_len))
            elif cur_len > need_len and need_len > 0:
                self._equity_by_symbol_history[symbol] \
                    = self._equity_by_symbol_history[symbol][:need_len]

    def _update_commission_fee_totals(self, symbol, commission, fee):
        st = self._stats_per_asset.get(symbol)
        if st is not None:
            st['commission_total'] += float(commission or 0.0)
            st['fee_total'] += float(fee or 0.0)

    def _maybe_open_campaign(self, symbol, prev_qty, new_qty, side, fill_time):
        camp = self._campaign[symbol]
        # Opening transition: flat -> nonzero
        if (prev_qty == 0) and (new_qty != 0) and not camp['active']:
            camp['active'] = True
            camp['start_time'] = fill_time
            camp['baseline_realized'] = self._per_asset_realized_cum[symbol]
            camp['side'] = 'long' if new_qty > 0 else 'short'
            camp['open_qty_sum'] = abs(new_qty)
            # Count a long/short trade now; it will "complete" when flat
            if camp['side'] == 'long':
                self._stats_per_asset[symbol]['long_trades'] += 1
            else:
                self._stats_per_asset[symbol]['short_trades'] += 1

    def _maybe_close_campaign(self, symbol, prev_qty, new_qty):
        # Closing transition: nonzero -> flat
        if (prev_qty != 0) and (new_qty == 0):
            camp = self._campaign[symbol]
            if camp['active']:
                pnl = self._per_asset_realized_cum[symbol] - camp['baseline_realized']
                st = self._stats_per_asset[symbol]
                st['trades'] += 1
                if pnl > 0:
                    st['wins'] += 1
                    st['gross_wins'] += pnl
                elif pnl < 0:
                    st['losses'] += 1
                    st['gross_losses'] += pnl  # negative
                # Recompute avg win/loss & expectancy
                win_count = max(st['wins'], 0)
                loss_count = max(st['losses'], 0)
                st['avg_win'] = (st['gross_wins'] / win_count) if win_count else 0.0
                st['avg_loss'] = (st['gross_losses'] / loss_count) if loss_count else 0.0  
                total = max(st['trades'], 1)
                win_rate = st['wins'] / total
                loss_rate = st['losses'] / total
                st['expectancy'] = win_rate * st['avg_win'] + loss_rate * st['avg_loss']

                # Reset campaign
                camp['active'] = False
                camp['start_time'] = None
                camp['baseline_realized'] = self._per_asset_realized_cum[symbol]
                camp['side'] = None
                camp['open_qty_sum'] = 0

    def _append_per_asset_equity_and_dd(self):
        """
        Called each time portfolio equity is appended (i.e., once per bar in step()).
        Updates per-asset equity series and running max drawdown.
        """
        # mark time (align to portfolio time)
        t = self.equity_time_history[-1] if self.equity_time_history else None
        self._equity_time_history.append(t)

        # Compute current unrealized per symbol and realized cum (already tracked)
        pos = self.get_positions()  # safe in open/closed phase :contentReference[oaicite:4]{index=4}
        for s in self.get_asset_list():
            unreal = pos.get(s, {}).get('unrealized_pnl', 0.0)
            eq = self._per_asset_realized_cum[s] + float(unreal or 0.0)
            self._equity_by_symbol_history[s].append(eq)

            # Update running drawdown
            dd = self._per_asset_dd_state[s]
            if eq > dd['peak']:
                dd['peak'] = eq
            peak = dd['peak'] if dd['peak'] != 0 else 1e-12  # avoid 0-div
            drawdown = (dd['peak'] - eq) / peak
            if drawdown > dd['max_dd']:
                dd['max_dd'] = drawdown
                # mirror to outward-facing stats
                self._stats_per_asset[s]['max_dd'] = dd['max_dd']

        # Update portfolio running drawdown off portfolio equity series
        port_eq = self.equity_history[-1] if self.equity_history else 0.0
        pdd = self._portfolio_dd_state
        if port_eq > pdd['peak']:
            pdd['peak'] = port_eq
        p_peak = pdd['peak'] if pdd['peak'] != 0 else 1e-12
        p_drawdown = (pdd['peak'] - port_eq) / p_peak
        if p_drawdown > pdd['max_dd']:
            pdd['max_dd'] = p_drawdown

    # ---- Backtest Simulation Control ----

    def start_backtest(self, interval_ms=10):
        self.reset_backtest()
        self.is_running = True

        if self.timer is None:
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.step)

        self.timer.start(interval_ms)

    def pause_backtest(self):
        if self.timer and self.timer.isActive():
            self.timer.stop()
        self.is_running = False

    def resume_backtest(self, interval_ms=10):
        if not self.timer:
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.step)
        self.timer.start(interval_ms)
        self.is_running = True

    def rewind_backtest(self, steps=10):
        """
        Rewind deterministically by re-simulating from the start to the target index.
        Ensures positions/cash/margin/equity/logs are consistent.
        """
        if self.is_running:
            return
        target = max(0, self.current_index - steps)
        self._resimulate_to_index(target)

    def fast_forward_backtest(self, steps=10):
        """
        Fast-forward deterministically by re-simulating from the start to the target index.
        Ensures positions/cash/margin/equity/logs are consistent.
        """
        if self.is_running:
            return
        primary_df = self.assets[0]['data'] if self.assets else None
        last_index = (len(primary_df) - 1) if primary_df is not None else -1
        target = min(last_index, self.current_index + steps)
        self._resimulate_to_index(target)

    def reset_backtest(self):
        self.current_index = 0
        self.is_running = False
        if self.timer:
            self.timer.stop()
        self.orders.clear()
        self.positions.clear()
        self.trade_log.clear()
        self.order_log.clear()
        self.used_margin = 0.0
        self.cash = self.initial_cash

        self.equity_history = []
        self.equity_time_history = []

        # Find the primary asset data to get the date of the first bar
        primary_df = (
            self.assets[0]['data']
            if self.assets and self.assets[0]['data'] is not None
            else None
        )

        # Get initial portfolio equity
        initial_equity = self.cash
        if hasattr(self, "get_portfolio"):
            initial_equity = self.get_portfolio().get("total_equity", self.cash)

        if primary_df is not None and len(primary_df) > 0:
            bar_time = primary_df.iloc[0]['date']
        else:
            bar_time = None

        self.equity_time_history.append(bar_time)
        self.equity_history.append(initial_equity)
        assert (
            len(self.equity_time_history) == len(self.equity_history)
        ), "Equity/time history misaligned in reset_backtest!"

        # NEW: reset live stats/campaigns (per-asset series start at 0 P&L)
        self._init_stats_containers()

    def _process_protective_orders(self, symbol, pos, bar, policy=None):
        """
        Resolve SL/TP on the same bar without bias, including gap-through at the open.
        policy: 'worst_case' (default), 'best_case', 'sl_first', or 'tp_first'
        Returns True if we closed the position on this bar.
        """
        import pandas as pd
        policy = (policy or self.config.get('intrabar_tp_sl_policy', 'worst_case')).lower()
        if pos is None or pos.qty == 0:
            return False
        sl = pos.stop_loss_price
        tp = getattr(pos, 'take_profit', None)
        if sl is None and tp is None:
            return False

        o = bar.get('open', pd.NA)
        h = bar.get('high', pd.NA)
        l = bar.get('low', pd.NA)

        # If this is a synthetic bar (NaN O/H/L), don't attempt intrabar triggers.
        if pd.isna(o) or (pd.isna(h) and pd.isna(l)):
            return False

        is_long = pos.qty > 0
        side_to_close = 'sell' if is_long else 'buy'

        # ---- Gap-through at OPEN: that side wins at the OPEN price ----
        if is_long:
            if sl is not None and not pd.isna(o) and o <= sl:
                return (
                    self._liquidate_at(
                        symbol, side_to_close, abs(pos.qty), float(o), by='sl'
                    )
                    is not None
                )
            if tp is not None and not pd.isna(o) and o >= tp:
                return (
                    self._liquidate_at(
                        symbol, side_to_close, abs(pos.qty), float(o), by='tp'
                    )
                    is not None
                )
            hit_sl = (sl is not None) and (not pd.isna(l)) and (l <= sl)
            hit_tp = (tp is not None) and (not pd.isna(h)) and (h >= tp)
        else:
            if sl is not None and not pd.isna(o) and o >= sl:
                return (
                    self._liquidate_at(
                        symbol, side_to_close, abs(pos.qty), float(o), by='sl'
                    )
                    is not None
                )
            if tp is not None and not pd.isna(o) and o <= tp:
                return (
                    self._liquidate_at(
                        symbol, side_to_close, abs(pos.qty), float(o), by='tp'
                    )
                    is not None
                )
            hit_sl = (sl is not None) and (not pd.isna(h)) and (h >= sl)
            hit_tp = (tp is not None) and (not pd.isna(l)) and (l <= tp)

        if not (hit_sl or hit_tp):
            return False

        # ---- Both touched intrabar: choose policy ----
        if hit_sl and hit_tp:
            if policy == 'best_case':
                prefer = 'tp'
            elif policy == 'sl_first':
                prefer = 'sl'
            elif policy == 'tp_first':
                prefer = 'tp'
            else:  # 'worst_case' default
                prefer = 'sl'   # worst-case = stop hit first for both long and short
        else:
            prefer = 'sl' if hit_sl else 'tp'

        fill_price = float(sl if prefer == 'sl' else tp)
        tag = 'sl' if prefer == 'sl' else 'tp'
        return (
            self._liquidate_at(
                symbol,
                side_to_close,
                abs(pos.qty),
                fill_price,
                by=tag
            )
            is not None
        )

    def step(self):
        primary_df = self.assets[0]['data'] if self.assets else None
        last_index = (len(primary_df) - 1) if primary_df is not None else -1

        if self.current_index < last_index:
            # ---- advance to next bar ----
            self.current_index += 1

            # ============ OPEN PHASE (no look-ahead) ============
            self._in_open_phase = True

            # Emit a masked snapshot (open only) so the bot trades without seeing H/L/C
            if primary_df is not None:
                live_snapshot = self._live_snapshot_df(primary_df)

                # NEW: skip synthetic bars at open phase (no live OPEN)
                should_emit = True
                if getattr(self, "skip_synthetic_open_bars", True):
                    try:
                        open_val = live_snapshot.iloc[-1].get('open')
                    except Exception:
                        open_val = None

                    import pandas as pd
                    should_emit = (
                        open_val is not None
                        and not pd.isna(open_val)
                    )

                if should_emit:
                    self.bar_updated.emit(live_snapshot)  # UI/bot hook: bot runs here

            # ============ INTRABAR: process stops/TP using true OHLC ============
            for asset in self.assets:
                symbol = asset['symbol']
                df = asset['data']
                if df is None or self.current_index >= len(df):
                    continue
                pos = self.positions.get(symbol)
                if pos and pos.qty != 0:
                    bar = df.iloc[self.current_index]
                    self._process_protective_orders(symbol, pos, bar)

            # leave open phase
            self._in_open_phase = False

            # Enforce margin, then record equity for this (now-closed) bar
            self.enforce_maintenance_margin()
            portfolio = self.get_portfolio()
            total_equity = portfolio.get("total_equity", self.cash)

            # time/equity tracking
            if primary_df is not None and self.current_index < len(primary_df):
                bar_time = primary_df.iloc[self.current_index]['date']
            else:
                bar_time = (
                    self.equity_time_history[-1]
                    if self.equity_time_history
                    else None
                )

            self.equity_time_history.append(bar_time)
            self.equity_history.append(total_equity)
            assert (
                len(self.equity_time_history) == len(self.equity_history)
            ), "Equity/time history misaligned!"

            # ---- NEW: per-asset equity & drawdown updates (live) ----
            self._append_per_asset_equity_and_dd()

            # closed-bar indicator recompute
            if primary_df is not None:
                self.bar_closed.emit(primary_df.iloc[: self.current_index + 1])

            self.bar_advanced.emit()

        else:
            # ---- Final bar cleanup ----
            self.close_all_positions()
            self.enforce_maintenance_margin()

            portfolio = self.get_portfolio()
            total_equity = portfolio.get("total_equity", self.cash)

            if primary_df is not None and self.current_index < len(primary_df):
                bar_time = primary_df.iloc[self.current_index]['date']
            else:
                bar_time = (
                    self.equity_time_history[-1]
                    if self.equity_time_history
                    else None
                )

            self.equity_time_history.append(bar_time)
            self.equity_history.append(total_equity)
            assert (
                len(self.equity_time_history) == len(self.equity_history)
            ), "Equity/time history misaligned!"

            # ---- NEW: per-asset equity & drawdown updates (final) ----
            self._append_per_asset_equity_and_dd()

            self.is_running = False
            if self.timer:
                self.timer.stop()
            self.backtest_finished.emit()

    # ---- API Methods ----

    def connect(self): 
        self.connected = True
    
    def disconnect(self): 
        self.connected = False

    def get_historical_data(self, asset, timeframe, start, end):
        return pd.DataFrame({
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        })
        
    def get_fill_price(self, order_dict, df, fill_idx, slippage_pct): 
        """
        Determine fill price and apply slippage.
        - Market & entries: use bar OPEN.
        - SL/TP: use provided trigger price (gap-aware logic already chose it), still apply slippage.
        - Forced closeout: fill at bar CLOSE (end-of-run accuracy).
        Slippage priority: slippage_ticks (if >0) else slippage_pct (argument).
        All fills are rounded to the instrument's tick grid.
        """
        side = order_dict['side']
        fill_bar = df.iloc[fill_idx]

        if order_dict.get('forced_liquidation') or order_dict.get('closeout'):
            if not pd.isna(fill_bar['close']):
                base_fill = float(fill_bar['close'])
            else:
                base_fill = float(fill_bar['open'])
        elif order_dict.get('triggered_by_sl', False) and 'stop_loss' in order_dict:
            base_fill = float(order_dict['stop_loss'])
        elif order_dict.get('triggered_by_tp', False) and 'take_profit' in order_dict:
            base_fill = float(order_dict['take_profit'])
        else:
            base_fill = float(fill_bar['open'])

        # Apply slippage (ticks preferred)
        symbol = order_dict['symbol']
        params = self.symbol_params.get(symbol, {})
        tick_size = params.get('tick_size', 0.25)
        slippage_ticks = int(params.get('slippage_ticks', self.config.get('slippage_ticks', 0)))
        pct = float(slippage_pct)
        if self.include_slippage:
            if slippage_ticks and tick_size:
                slip_amt = slippage_ticks * tick_size
                base_fill = base_fill + slip_amt if side == 'buy' else base_fill - slip_amt
            elif pct:
                base_fill = base_fill * (1 + pct) if side == 'buy' else base_fill * (1 - pct)

        # Final rounding to the nearest tick
        base_fill = round_to_tick(base_fill, tick_size)
        return base_fill
        
    def check_and_update_margin(self, symbol, side, qty, initial_margin, commit=True):
        """
        Checks if there is sufficient margin to open/increase a position.
        If commit=True, reserves/releases margin accordingly.
        Returns (allowed: bool, margin_increase: float, margin_release: float)
        """
        if initial_margin is None or initial_margin <= 0:
            print(
                "[WARNING] Cannot check margin for {}: "
                "missing or invalid initial_margin ({}). "
                "Order skipped.".format(
                    symbol,
                    initial_margin
                )
            )
            return False, 0, 0
            
        pos = self.positions.get(symbol)
        prev_qty = pos.qty if pos else 0
        new_qty = prev_qty + (qty if side == 'buy' else -qty)
        prev_margin = abs(prev_qty) * initial_margin
        new_margin = abs(new_qty) * initial_margin
        margin_increase = max(new_margin - prev_margin, 0)
        margin_release = max(prev_margin - new_margin, 0)
        available_equity = self.get_portfolio().get("total_equity", self.cash) - self.used_margin

        if margin_increase > 0 and available_equity < margin_increase:
            return False, margin_increase, margin_release

        if commit:
            self.used_margin += margin_increase
            self.used_margin -= margin_release
        return True, margin_increase, margin_release    
        
    def enforce_maintenance_margin(self):
        # Total maintenance requirement
        total_maintenance_margin = 0.0
        for symbol, pos in self.positions.items():
            if pos.qty == 0:
                continue
            mm = self.symbol_params[symbol].get('maintenance_margin', 0)
            total_maintenance_margin += abs(pos.qty) * mm

        # === NEW: worst-case intrabar equity (LOW for longs, HIGH for shorts) ===
        import pandas as pd
        total_equity = self.cash
        for symbol, pos in self.positions.items():
            if pos.qty == 0:
                continue
            df = self.get_asset_data(symbol)
            if df is None or self.current_index >= len(df):
                mark_price = self._get_mark_price(symbol, pos)
            else:
                bar = df.iloc[self.current_index]
                if pos.qty > 0:
                    mark_price = bar.get('low')
                else:
                    mark_price = bar.get('high')
                if pd.isna(mark_price):
                    mark_price = self._get_mark_price(symbol, pos)

            tick_size = self.symbol_params[symbol]['tick_size']
            tick_value = self.symbol_params[symbol]['tick_value']
            total_equity += pos.get_unrealized_pnl(float(mark_price), tick_size, tick_value)
        # ========================================================================

        if total_equity < total_maintenance_margin:
            print("[MARGIN CALL] Equity below maintenance margin! Liquidating positions...")

            # Sort by worst open PnL first (signed, tick-scaled), then liquidate until satisfied
            def open_pnl(symbol, pos):
                tick_size = self.symbol_params[symbol]['tick_size']
                tick_value = self.symbol_params[symbol]['tick_value']
                mark = self._get_mark_price(symbol, pos)  # respect open/closed phase
                return ((mark - pos.avg_entry_price) / tick_size) * tick_value * pos.qty

            liquidation_list = sorted(
                [(s, p) for s, p in self.positions.items() if p.qty != 0],
                key=lambda item: open_pnl(item[0], item[1])
            )  # ascending => most negative first

            for symbol, pos in liquidation_list:
                if pos.qty == 0:
                    continue
                side = 'sell' if pos.qty > 0 else 'buy'
                # IMPORTANT: fill liquidation at the bar CLOSE to match valuation that triggered it
                self.place_order({
                    'symbol': symbol,
                    'side': side,
                    'qty': abs(pos.qty),
                    'order_type': 'market',
                    'triggered_by_liquidation': True,
                    'forced_liquidation': True,
                    'closeout': True,
                })
                # Re-evaluate after each liquidation
                new_portfolio = self.get_portfolio()
                new_equity = new_portfolio.get("total_equity", 0.0)
                new_total_mm = 0.0
                for s, p in self.positions.items():
                    if p.qty == 0:
                        continue
                    mm = self.symbol_params[s].get('maintenance_margin', 0)
                    new_total_mm += abs(p.qty) * mm
                if new_equity >= new_total_mm:
                    break
                    
    def apply_commission_and_fee(self, symbol, qty):
        """
        Deduct commission and fees from cash for a given order.
        Returns (total_commission, total_fee)
        """
        params = self.symbol_params[symbol]
        commission_per_contract = params.get(
            'commission_per_contract',
            self.config.get('commission_per_contract', 0.0)
        )
        fee_per_trade = params.get(
            'fee_per_trade',
            self.config.get('fee_per_trade', 0.0)
        )
        total_commission = commission_per_contract * qty
        total_fee = fee_per_trade
        total_cost = total_commission + total_fee

        self.cash -= total_cost

        if self.cash < 0:
            print(f"[WARNING] Account negative after commission/fee deduction: ${self.cash:.2f}")

        return total_commission, total_fee

    def _validate_qty_and_get_params(self, order_dict):
        symbol = order_dict['symbol']
        side = order_dict['side']
        qty = order_dict['qty']
        if qty <= 0:
            print(f"[REJECTED] Invalid qty={qty} for {symbol}")
            return None, None, None, None
        params = self.symbol_params[symbol]
        return symbol, side, qty, params

    def _resolve_order_id(self, provided_id):
        if provided_id is not None:
            existing = self.orders.get(provided_id)
            if existing and existing.status in ['open', 'filled']:
                print(f"[SKIP] Order {provided_id} already exists (status={existing.status}).")
                return provided_id, True  # id, is_skip
            return provided_id, False
        oid = self.next_order_id
        self.next_order_id += 1
        return oid, False

    def _pretrade_margin_preview(self, symbol, side, qty, initial_margin):
        allowed, margin_increase, margin_release_pending = self.check_and_update_margin(
            symbol, side, qty, initial_margin, commit=False
        )
        if not allowed:
            print(f"[REJECTED] Not enough margin to change position in {symbol} by {qty} contracts")
            return None
        return margin_increase, margin_release_pending

    def _compute_fee_costs(self, params, qty):
        commission_per_contract = params.get('commission_per_contract',
                                             self.config.get('commission_per_contract', 0.0))
        fee_per_trade = params.get('fee_per_trade',
                                   self.config.get('fee_per_trade', 0.0))
        total_commission = commission_per_contract * qty
        total_fee = fee_per_trade
        total_cost = total_commission + total_fee
        return total_commission, total_fee, total_cost

    def _should_bypass_fee_gate(self, order_dict):
        return any(order_dict.get(k) for k in (
            'forced_liquidation', 'triggered_by_sl', 'triggered_by_tp', 'closeout'
        ))

    def _create_and_log_order(self, order_id, symbol, side, qty, order_type):
        order = Order(order_id, symbol, side, qty, order_type=order_type)
        self.orders[order_id] = order
        self.order_log.append(order)
        return order

    def _obtain_fill_context(self, symbol):
        df = self.get_asset_data(symbol)
        if df is None or self.current_index >= len(df):
            return None, None
        return df, self.current_index

    def _fill_price_for_order(self, order_dict, df, fill_idx, params):
        slippage_pct = params.get('slippage_pct', self.config.get('slippage_pct', 0.001))
        return self.get_fill_price(order_dict, df, fill_idx, slippage_pct)

    def _mark_filled_and_apply_costs(self, order, fill_price, df, fill_idx, symbol, qty):
        order.status = 'filled'
        order.fill_price = fill_price
        order.fill_time = df.iloc[fill_idx]['date'] if 'date' in df.columns else fill_idx
        total_commission, total_fee = self.apply_commission_and_fee(symbol, qty)
        # NEW: accumulate commission/fee per asset for stats
        self._update_commission_fee_totals(symbol, total_commission, total_fee)
        return total_commission, total_fee

    def _update_position_on_fill(
            self, symbol, contract_size, fill_price, qty, side, 
            tick_size, tick_value
        ):
        self._ensure_symbol_stats(symbol)
        pos = self.positions.get(symbol)
        if not pos:
            pos = Position(symbol, contract_size)
            self.positions[symbol] = pos

        # Track prev qty BEFORE update for campaign edge detection
        prev_qty = pos.qty

        realized = pos.update_on_fill(fill_price, qty, side, tick_size, tick_value)
        self.cash += realized

        # NEW: update realized cum by symbol, and campaign edges (open/close)
        self._per_asset_realized_cum[symbol] += float(realized or 0.0)
        # After update (pos.qty changed)
        new_qty = self.positions[symbol].qty
        # Campaign start (flat->nonzero)
        self._maybe_open_campaign(symbol, prev_qty, new_qty, side, None)  # fill_time added later
        # Campaign finish (nonzero->flat) handled in _maybe_close_campaign caller where we know fill_time

        return pos, realized, prev_qty

    def _commit_margin_after_fill(self, margin_increase, margin_release_pending):
        if margin_increase and margin_increase > 0:
            self.used_margin += margin_increase
        if margin_release_pending and margin_release_pending > 0:
            self.used_margin -= margin_release_pending

    def _apply_protective_levels(self, pos, order_dict, tick_size):
        if pos.qty == 0:
            pos.stop_loss_price = None
            pos.take_profit = None
            return
        if 'stop_loss' in order_dict:
            sl_raw = order_dict.get('stop_loss')
            pos.stop_loss_price = round_to_tick(sl_raw, tick_size) if sl_raw is not None else None
        if 'take_profit' in order_dict:
            tp_raw = order_dict.get('take_profit')
            pos.take_profit = round_to_tick(tp_raw, tick_size) if tp_raw is not None else None

    def _append_trade_log(self, order_id, symbol, side, qty, fill_price, fill_time,
                          realized, pos, commission, fee, triggered_by=None):
        self.trade_log.append({
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'fill_price': fill_price,
            'fill_time': fill_time,
            'realized_pnl_change': realized,
            'net_realized_from_cash': self.cash - self.initial_cash,
            'position_after_fill': pos.qty,
            'commission': commission,
            'fee': fee,
            'exit_tag': triggered_by,  # 'sl', 'tp', 'liquidation', None
        })

    def place_order(self, order_dict):
        # 1) Basic validation & params
        res = self._validate_qty_and_get_params(order_dict)
        if res[0] is None:
            return None
        symbol, side, qty, params = res
        order_type = order_dict.get('order_type', 'market')
        contract_size = params['contract_size']
        tick_size = params['tick_size']
        tick_value = params['tick_value']
        initial_margin = params.get('initial_margin', 0)
        self._ensure_symbol_stats(symbol)

        # 2) Order id resolution (support provided id + skip)
        order_id, is_skip = self._resolve_order_id(order_dict.get("order_id"))
        if is_skip:
            return order_id

        # 3) Margin preview (no commit yet)
        margin_preview = self._pretrade_margin_preview(symbol, side, qty, initial_margin)
        if margin_preview is None:
            return None
        margin_increase, margin_release_pending = margin_preview

        # 4) Fee gate (unless exit/forced close)
        commission, fee, total_cost = self._compute_fee_costs(params, qty)
        if not self._should_bypass_fee_gate(order_dict) and self.cash < total_cost:
            print(f"[REJECTED] Not enough cash to cover commission/fee for {symbol} order: "
                  f"Required ${total_cost:.2f}, Available ${self.cash:.2f}")
            return None

        # 5) Create/log order
        order = self._create_and_log_order(order_id, symbol, side, qty, order_type)

        # 6) Find fill context; reject if no data
        df, fill_idx = self._obtain_fill_context(symbol)
        if df is None:
            print(
                "Order Rejected\n"
                f"  current_index: {self.current_index}\n"
                f"  df length: {0 if df is None else len(df)}"
            )
            order.status = 'rejected'
            return order_id

        # 7) Fill price & mark filled + apply costs
        fill_price = self._fill_price_for_order(order_dict, df, fill_idx, params)
        commission, fee = self._mark_filled_and_apply_costs(
            order, fill_price, df, fill_idx, symbol, qty
        )

        # 8) Position update, then margin commit
        pos, realized, prev_qty = self._update_position_on_fill(
            symbol, contract_size, fill_price, qty, side, tick_size, tick_value
        )
        self._commit_margin_after_fill(margin_increase, margin_release_pending)

        # 9) Protective levels
        self._apply_protective_levels(pos, order_dict, tick_size)

        # Determine trigger tag for logs (for exit attribution later)
        triggered_by = None
        if order_dict.get('forced_liquidation') or order_dict.get('closeout'):
            triggered_by = 'liquidation'
        elif order_dict.get('triggered_by_sl'):
            triggered_by = 'sl'
        elif order_dict.get('triggered_by_tp'):
            triggered_by = 'tp'

        # 10) Trade log append (kept compatible with your existing schema) :contentReference[oaicite:5]{index=5}
        self._append_trade_log(
            order_id, symbol, side, qty, fill_price, order.fill_time,
            realized, pos, commission, fee, triggered_by=triggered_by
        )

        # 11) Check campaign close transition using AFTER state (pos.qty)
        new_qty = pos.qty
        self._maybe_close_campaign(symbol, prev_qty, new_qty)

        return order_id
        
    def _liquidate_at(self, symbol, side, qty, price, by=None):
        """
        Close 'qty' at an explicit 'price', tagging whether SL or TP triggered.
        We pass the price via stop_loss/take_profit so get_fill_price uses it,
        and then the engine still applies configured slippage. 
        """
        import math
        if qty is None or qty <= 0 or price is None or (isinstance(price, float) and math.isnan(price)):
            return None

        order = {
            'symbol': symbol,
            'side': side,
            'qty': int(abs(qty)),
            'order_type': 'market',
        }
        if by == 'sl':
            order['stop_loss'] = float(price)
            order['triggered_by_sl'] = True
        elif by == 'tp':
            order['take_profit'] = float(price)
            order['triggered_by_tp'] = True
        return self.place_order(order)

    def get_order_status(self, order_id):
        order = self.orders.get(order_id, None)
        if order is None:
            return 'unknown'
        return order.status

    def cancel_order(self, order_id):
        order = self.orders.get(order_id, None)
        if order and order.status == 'open':
            order.status = 'cancelled'

    def modify_stop_loss(self, symbol, new_value):
        if symbol in self.positions:
            tick_size = self.symbol_params[symbol]['tick_size']
            self.positions[symbol].stop_loss_price = round_to_tick(new_value, tick_size)

    def modify_take_profit(self, symbol, new_value):
        if symbol in self.positions:
            tick_size = self.symbol_params[symbol]['tick_size']
            self.positions[symbol].take_profit = round_to_tick(new_value, tick_size)

    def close_all_positions(self):
        for symbol, pos in list(self.positions.items()):
            if pos.qty == 0:
                continue
            df = self.get_asset_data(symbol)
            if df is None or len(df) == 0:
                continue
            fill_idx = min(self.current_index, len(df) - 1)
            last_bar = df.iloc[fill_idx]
            if pos.qty > 0:
                close_qty = abs(pos.qty)
                side = "sell"
            elif pos.qty < 0:
                close_qty = abs(pos.qty)
                side = "buy"
            else:
                continue
            print(f"[FORCED LIQUIDATION] {symbol}: current qty={pos.qty}, submitting side={side}, qty={close_qty}")
            order = {
                "symbol": symbol,
                "side": side,
                "qty": close_qty,
                "order_type": "market",
                "forced_liquidation": True,
                "closeout": True
            }
            self.place_order(order)

        # Clear remaining protective orders
        for pos in self.positions.values():
            pos.stop_loss_price = None
            pos.take_profit = None

    # ===== Open-phase mark-to-market helper =====
    def _get_mark_price(self, symbol, pos=None):
        """
        During the open phase, mark at current bar OPEN if available (no look-ahead),
        else previous CLOSE. After bar closes, use current bar CLOSE.
        """
        df = self.get_asset_data(symbol)
        if df is None or len(df) == 0:
            return pos.avg_entry_price if pos else 0.0

        idx = min(self.current_index, len(df) - 1)

        if self._in_open_phase:
            op = df.iloc[idx].get('open')
            if pd.notna(op):
                return float(op)
            if idx > 0:
                prev_close = df.iloc[idx - 1].get('close')
                if pd.notna(prev_close):
                    return float(prev_close)
            return pos.avg_entry_price if pos else 0.0

        cl = df.iloc[idx].get('close')
        if pd.notna(cl):
            return float(cl)

        op = df.iloc[idx].get('open')
        if pd.notna(op):
            return float(op)
        if idx > 0:
            prev_close = df.iloc[idx - 1].get('close')
            if pd.notna(prev_close):
                return float(prev_close)
        return pos.avg_entry_price if pos else 0.0
    # ============================================

    def get_positions(self):
        positions_info = {}
        for symbol, pos in self.positions.items():
            tick_size = self.symbol_params[symbol]['tick_size']
            tick_value = self.symbol_params[symbol]['tick_value']
            current_price = self._get_mark_price(symbol, pos)  # open-phase safe
            positions_info[symbol] = {
                'qty': pos.qty,
                'avg_entry_price': pos.avg_entry_price,
                'realized_pnl': pos.realized_pnl,
                'unrealized_pnl': pos.get_unrealized_pnl(current_price, tick_size, tick_value),
                'contract_size': pos.contract_size,
                'stop_loss_price': getattr(pos, "stop_loss_price", None),
                'take_profit': getattr(pos, "take_profit", None),
            }
        return positions_info

    def get_total_pnl(self):
        # Realized P&L = change in cash since start (commissions/fees already debited)
        realized = self.cash - self.initial_cash

        unrealized = 0.0
        for symbol, pos in self.positions.items():
            tick_size = self.symbol_params[symbol]['tick_size']
            tick_value = self.symbol_params[symbol]['tick_value']
            current_price = self._get_mark_price(symbol, pos)  # open-phase safe
            unrealized += pos.get_unrealized_pnl(current_price, tick_size, tick_value)

        return {'realized': realized, 'unrealized': unrealized, 'total': realized + unrealized}

    def get_portfolio(self):
        positions = self.get_positions()
        open_orders = [vars(order) for order in self.orders.values() if order.status == 'open']
        equity = self.cash
        for symbol, pos in self.positions.items():
            tick_size = self.symbol_params[symbol]['tick_size']
            tick_value = self.symbol_params[symbol]['tick_value']
            current_price = self._get_mark_price(symbol, pos)  # open-phase safe
            equity += pos.get_unrealized_pnl(current_price, tick_size, tick_value)

        available_equity = equity - self.used_margin
        return {
            "cash": self.cash,
            "positions": positions,
            "open_orders": open_orders,
            "total_equity": equity,
            "used_margin": self.used_margin,
            "available_equity": available_equity,
        }

    # ---- Asset Data ----

    def load_backtest_config(self, config_path):
        """
        Load YAML config, apply engine/global defaults, load asset data, align timelines,
        and initialize the primary DataFrame/index pointers.
        """
        config = self._read_config_file(config_path)
        # sets cash, fees, slippage, live mode, skip_synthetic_open_bars
        self._apply_global_defaults(config)
        # validates futures margins and loads per-asset DataFrames
        self._build_symbol_params_and_load_assets(config)
        # reindex all assets to a single master timeline
        self._align_all_assets_to_master_timeline()
        # set self.df and current_index
        self._initialize_primary_df()

    # ---------------------- helpers ----------------------
    
    def _worst_case_intrabar_equity(self) -> float:
        """
        Conservative equity using current bar LOW for longs and HIGH for shorts.
        Falls back to the engine's mark price if intrabar values are NaN
        (e.g., synthetic bars).
        """
        import pandas as pd
        eq = self.cash
        for symbol, pos in self.positions.items():
            if pos.qty == 0:
                continue
            df = self.get_asset_data(symbol)
            if df is None or self.current_index >= len(df):
                # no data → default to current mark
                price = self._get_mark_price(symbol, pos)
            else:
                row = df.iloc[self.current_index]
                if pos.qty > 0:
                    # long: worst is LOW
                    price = row.get('low')
                else:
                    # short: worst is HIGH
                    price = row.get('high')
                if pd.isna(price):
                    price = self._get_mark_price(symbol, pos)

            tick_size  = self.symbol_params[symbol]['tick_size']
            tick_value = self.symbol_params[symbol]['tick_value']
            eq += pos.get_unrealized_pnl(float(price), tick_size, tick_value)
        return eq


    def _read_config_file(self, config_path):
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # Keep a copy around
        self.config = config or {}
        return self.config

    def _apply_global_defaults(self, config):
        """
        Apply engine-level defaults and global flags.
        Adds: skip_synthetic_open_bars (default True).
        """
        import numpy as np

        initial_cash = float(config.get("initial_cash", 0.0))
        self.cash = initial_cash
        self.initial_cash = initial_cash

        # Live bar exposure mode & RNG
        self.live_price_mode = config.get("live_price_mode", "open_only")
        seed = config.get("random_seed", None)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

        # Commissions/fees (global defaults; assets can override)
        self.commission_per_contract = float(config.get("commission_per_contract", 0.0))
        self.fee_per_trade = float(config.get("fee_per_trade", 0.0))

        # Slippage policy (ticks take priority over pct)
        self.include_slippage = bool(config.get("include_slippage", False))
        self.slippage_ticks = int(config.get("slippage_ticks", 0))
        self.slippage_pct = float(config.get("slippage_pct", 0.0))

        # NEW: skip synthetic bars during open phase (primary asset OPEN is NaN)
        self.skip_synthetic_open_bars = bool(config.get("skip_synthetic_open_bars", True))

        # Clear/prepare per-run containers
        self.symbol_params = {}
        self.assets = config.get("assets", [])

    def _build_symbol_params_and_load_assets(self, config):
        """
        Validate assets (futures margin) and load raw CSVs into normalized DataFrames.
        Keeps your existing TradeStation/Yahoo detection and per-asset overrides.
        """
        import os
        import pandas as pd

        assets_cfg = config.get("assets", [])

        # Basic validation for futures margin (unchanged)
        for asset in assets_cfg:
            if asset.get("type") == "futures":
                im = asset.get("initial_margin", None)
                if im is None or float(im) <= 0:
                    raise ValueError(
                        f"[CONFIG ERROR] Asset {asset.get('symbol')} has invalid initial_margin: {im}"
                    )

        # Load each asset, applying global defaults with per-asset overrides
        for asset in assets_cfg:
            symbol = asset["symbol"]

            params = {
                "symbol": symbol,
                "type": asset.get("type", "futures"),
                "file": asset.get("file"),
                "contract_size": asset.get("contract_size", None),
                "tick_size": asset.get("tick_size", 0.25),
                "tick_value": asset.get("tick_value", 0.5),
                "initial_margin": (
                    float(asset.get("initial_margin"))
                    if asset.get("initial_margin") is not None
                    else None
                ),
                "maintenance_margin": (
                    float(asset.get(
                        "maintenance_margin",
                        asset.get("initial_margin", 0.0)
                    ))
                ),
                "commission_per_contract": (
                    float(asset.get(
                        "commission_per_contract",
                        self.commission_per_contract
                    ))
                ),
                "fee_per_trade": float(asset.get("fee_per_trade", self.fee_per_trade)),
                "slippage_ticks": int(asset.get("slippage_ticks", self.slippage_ticks)),
                "slippage_pct": float(asset.get("slippage_pct", self.slippage_pct)),
                "currency": asset.get("currency", config.get("currency", "USD")),
                "exchange": asset.get("exchange", None),
            }

            # Load data using your existing heuristics
            if params["file"] and os.path.exists(params["file"]):
                filename = params["file"].lower()
                if 'yahoo_finance' in filename:
                    df = load_yahoo_csv(params["file"])
                    df = ensure_timestamp(df)
                elif 'tradestation' in filename or 'nq_1m' in filename or 'mnq_1m' in filename:
                    df = load_tradestation_csv(params["file"])
                else:
                    raw = pd.read_csv(params["file"])
                    if "TotalVolume" in raw.columns and "TimeStamp" in raw.columns:
                        df = load_tradestation_csv(params["file"])
                    elif {"Price", "Open"} <= set(raw.columns):
                        df = load_yahoo_csv(params["file"])
                        df = ensure_timestamp(df)
                    else:
                        df = raw
            else:
                print(f"[WARN] Data file not found for {symbol}: {params.get('file')}")
                df = None

            params["data"] = df
            self.symbol_params[symbol] = params
            asset.update(params)

        self.assets = assets_cfg

    def _align_all_assets_to_master_timeline(self):
        """
        Build a master index from all timestamps and reindex each asset:
        - Keep NaN for open/high/low on synthetic bars (gap markers)
        - Forward-fill close
        - Zero-fill volume
        """
        import pandas as pd

        all_timestamps = set()
        for asset in self.assets:
            df = asset.get("data")
            if df is not None and not df.empty:
                if "date" in df.columns:
                    all_timestamps.update(df["date"])
                elif "timestamp" in df.columns:
                    all_timestamps.update(df["timestamp"])

        if not all_timestamps:
            return

        master_index = pd.Index(sorted(all_timestamps))
        for asset in self.assets:
            df = asset.get("data")
            if df is None or df.empty:
                continue

            index_col = "date" if "date" in df.columns else "timestamp"
            df = df.set_index(index_col).reindex(master_index)

            # Do NOT forward-fill O/H/L across gaps; keep NaN to mark synthetic bars
            for col in ["open", "high", "low"]:
                if col in df.columns:
                    df[col] = df[col]  # leave NaNs

            # Optionally ffill close to keep equity curves continuous
            if "close" in df.columns:
                df["close"] = df["close"].ffill()

            if "volume" in df.columns:
                df["volume"] = df["volume"].fillna(0)

            df = df.reset_index().rename(columns={"index": index_col})
            asset["data"] = df

    def _initialize_primary_df(self):
        """
        Set current_index to 0 and choose the first non-empty asset df as primary (self.df).
        """
        self.current_index = 0
        self.df = None
        for asset in self.assets:
            if asset.get("data") is not None and not asset["data"].empty:
                self.df = asset["data"]
                break
        if self.df is None:
            print("[ERROR] No asset data loaded; self.df remains empty!")

    def get_asset_list(self):
        return [asset.get('symbol', f"Asset_{i}") for i, asset in enumerate(self.assets)]

    def get_asset_data(self, symbol):
        for asset in self.assets:
            if asset.get('symbol') == symbol:
                return asset.get('data')
        return None
            
    def _live_snapshot_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a masked snapshot for the current bar:
        - Real OPEN is kept
        - H/L/V hidden on live bar
        - 'close' set to a live estimate per mode
        Null-safe for df=None/empty.
        """
        # --- Null/empty safe guards ---
        if df is None:
            return pd.DataFrame()
        if df.empty:
            return df.copy()

        # Clamp slice end safely
        end = min(self.current_index + 1, len(df))
        if end <= 0:
            return df.iloc[:0].copy()

        # Build view up to and including the current index
        idx = self.current_index
        view = df.iloc[:idx + 1].copy()

        # live open is real; no H/L/C yet
        o = (
            float(view.at[idx, 'open'])
            if 'open' in view.columns and not pd.isna(view.at[idx, 'open'])
            else np.nan
        )
        live_close = o
        if self.live_price_mode == "random_in_hilo":
            # NOTE: references final H/L (turn OFF for strict backtests)
            lo = (
                float(view.at[idx, 'low'])
                if 'low' in view.columns and not pd.isna(view.at[idx, 'low'])
                else np.nan
            )

            hi = (
                float(view.at[idx, 'high'])
                if 'high' in view.columns and not pd.isna(view.at[idx, 'high'])
                else np.nan
            )
            if not np.isnan(lo) and not np.isnan(hi) and hi >= lo:
                live_close = float(self._rng.uniform(lo, hi))

        # hide high/low/volume; set a live "close"
        if 'close' in view.columns: view.at[idx, 'close'] = live_close
        if 'high'  in view.columns: view.at[idx, 'high']  = np.nan
        if 'low'   in view.columns: view.at[idx, 'low']   = np.nan
        if 'volume' in view.columns: view.at[idx, 'volume'] = np.nan

        # carry previous indicator values forward (bot usage at open), only if have a previous row
        if idx > 0:
            prev = view.iloc[idx - 1]
            for col in view.columns:
                if col.startswith(('bb_', 'ema_', 'rsi_', 'atr_', 'tr')):
                    view.at[idx, col] = prev.get(col, np.nan)

        return view

    def get_latest_data(self, symbol):
        for asset in self.assets:
            if asset.get('symbol') == symbol:
                df = asset.get('data')
                if df is None or df.empty:
                    return pd.DataFrame()
                if not self.is_running:
                    return df
                safe_idx = min(self.current_index, len(df) - 1)
                if self._in_open_phase:
                    return self._live_snapshot_df(df)
                return df.iloc[:safe_idx + 1]
        return pd.DataFrame()

    def notify_update(self):
        pass
        
    @property
    def current_timestamp(self):
        """Returns the current bar's timestamp for the PRIMARY asset only (stable clock)."""
        if not self.assets:
            return None
        primary = self.assets[0].get("data")
        if primary is not None and self.current_index < len(primary):
            return primary.iloc[self.current_index].get("timestamp")
        return None

    def is_finished(self):
        """Returns True if the current index is at or beyond the final bar."""
        max_len = max((len(a["data"]) for a in self.assets if a.get("data") is not None), default=0)
        return self.current_index >= max_len - 1

    # ---- Deterministic re-simulation helper ----
    def _resimulate_to_index(self, target_idx: int):
        """
        Rebuild full engine state by replaying bars from the beginning up to target_idx.
        Emits the normal signals so bots and UI stay consistent.
        """
        primary_df = self.assets[0]['data'] if self.assets else None
        last_index = (len(primary_df) - 1) if primary_df is not None else -1
        target_idx = max(0, min(target_idx, last_index))

        # Reset to genesis
        self.reset_backtest()

        # Temporarily mark running so env/bot callbacks execute during replay
        was_running = self.is_running
        self.is_running = True

        # Replay until target index
        while self.current_index < target_idx:
            self.step()

        # Stop running after reaching target (no finalization). Restore original state.
        self.is_running = was_running

    # ----------------------- STATS: public getters -----------------------

    def get_equity_series(self, symbol=None):
        """
        If symbol is None -> return portfolio equity (times, equity).
        Else -> return per-asset cumulative P&L series (times, equity_by_symbol).
        """
        if symbol is None:
            return list(self.equity_time_history), list(self.equity_history)
        if symbol not in self._equity_by_symbol_history:
            return list(self._equity_time_history), []
        return list(self._equity_time_history), list(self._equity_by_symbol_history[symbol])

    def get_win_rate(self, symbol):
        """Return current win rate for the given symbol (0..1)."""
        st = self._stats_per_asset.get(symbol)
        if not st or st['trades'] == 0:
            return 0.0
        return st['wins'] / max(st['trades'], 1)

    def get_max_drawdown(self, symbol=None):
        """
        If symbol is None -> portfolio max DD (0..1).
        Else -> per-asset max DD (0..1) over cumulative P&L.
        """
        if symbol is None:
            return self._portfolio_dd_state.get('max_dd', 0.0)
        st = self._stats_per_asset.get(symbol)
        if not st:
            return 0.0
        return st.get('max_dd', 0.0)

    def get_stats_snapshot(self):
        """
        Return a deep-ish snapshot of per-asset and portfolio stats suitable for UI.
        """
        per_asset = {}
        for s, st in self._stats_per_asset.items():
            total = max(st['trades'], 1)
            win_rate = (st['wins'] / total) if total else 0.0
            loss_rate = (st['losses'] / total) if total else 0.0
            profit_factor = None
            if st['gross_losses'] != 0:
                if abs(st['gross_losses']) > 0:
                    profit_factor = (st['gross_wins'] / abs(st['gross_losses']))
                else:
                    profit_factor = None

            per_asset[s] = {
                'trades': st['trades'],
                'wins': st['wins'],
                'losses': st['losses'],
                'long_trades': st['long_trades'],
                'short_trades': st['short_trades'],
                'win_rate': win_rate,
                'avg_win': st['avg_win'],
                'avg_loss': st['avg_loss'],
                'profit_factor': profit_factor,
                'expectancy': st['expectancy'],
                'commission_total': st['commission_total'],
                'fee_total': st['fee_total'],
                'max_drawdown': st['max_dd'],
            }

        portfolio_stats = {
            'max_drawdown': self._portfolio_dd_state.get('max_dd', 0.0),
            'total_equity': self.equity_history[-1] if self.equity_history else self.initial_cash,
            'initial_cash': self.initial_cash,
            'used_margin': self.used_margin,
        }

        return {
            'per_asset': per_asset,
            'portfolio': portfolio_stats
        }


# ---- Helpers ----

def load_yahoo_csv(filepath):
    # Normalized to UTC + timestamp so it aligns with TradeStation uniformly 
    df = pd.read_csv(filepath, header=0, skiprows=[1,2])
    df = df.rename(columns={
        'Price': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['timestamp'] = df['date'].astype('int64') // 10**9
    df = df.sort_values('date')
    wanted = ["date", "open", "high", "low", "close", "volume", "timestamp"]
    df = df[wanted]
    return df
    
def load_tradestation_csv(filepath):
    """
    Load and normalize TradeStation CSV price data for backtesting.
    Returns a DataFrame with: ['date','open','high','low','close','volume','timestamp']
    """
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "TotalVolume": "volume",
        "TimeStamp": "date"
    })
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["timestamp"] = df["date"].astype("int64") // 10**9
    wanted = ["date", "open", "high", "low", "close", "volume", "timestamp"]
    df = df[wanted]
    return df

