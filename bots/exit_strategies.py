# bots/exit_strategies.py

from __future__ import annotations
from collections import deque, defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from classes.API_Interface import round_to_tick
import csv, time
import os, json 
import joblib, math
from typing import Dict, List, Tuple, Optional


class ExitStrategy:
    """
    Unified interface for exit logic. Responsible for BOTH stop-loss and take-profit.

    Methods:
      - initial_levels: returns (stop_loss_price, take_profit_price_or_None)
      - update_stop: returns new stop or None to keep unchanged

    Backward-compat helpers:
      - initial_stop: used by older sizing code; returns only the stop.
      - ensure_models_loaded: optional no-op hook for RL/model-based strategies.
    """

    def initial_levels(
        self,
        side: str,
        entry_price: float,
        atr: float,
        tick_size: float,
        df=None,
        symbol: Optional[str] = None,
        **kwargs,
    ) -> Tuple[float, Optional[float]]:
        raise NotImplementedError

    def update_stop(
        self,
        side: str,
        stop_loss: float,
        entry_price: float,
        current_price: float,
        atr: float,
        tick_size: float,
        df=None,
        symbol: Optional[str] = None,
        **kwargs,
    ):
        raise NotImplementedError

    # ------- compatibility helpers (older code may call this) -------

    def initial_stop(
        self,
        side: str,
        entry_price: float,
        atr: float,
        df=None,
        symbol: Optional[str] = None,
        tick_size: float = 0.01,
        **kwargs,
    ) -> float:
        sl, _ = self.initial_levels(
            side=side,
            entry_price=entry_price,
            atr=atr,
            tick_size=tick_size,
            df=df,
            symbol=symbol,
            **kwargs,
        )
        return sl

    def ensure_models_loaded(self, env):  # optional
        return


@dataclass
class TrailingATRExit(ExitStrategy):
    """
    Trailing stop with ATR multiple; NO take profit.
    - initial stop: entry ± K*ATR (depending on side)
    - trailing: *only begins after price moves past entry in the trade's favor*,
                then follows K*ATR behind price, never loosening.
    """
    atr_multiple: float = 3.0

    def initial_levels(
        self,
        side: str,
        entry_price: float,
        atr: float,
        tick_size: float,
        **kwargs,
    ):
        k = float(self.atr_multiple)
        if side in ("buy", "long"):
            sl = entry_price - k * atr
        else:
            sl = entry_price + k * atr
        return round_to_tick(sl, tick_size), None  # no TP

    def update_stop(
        self,
        side: str,
        stop_loss: float,
        entry_price: float,
        current_price: float,
        atr: float,
        tick_size: float,
        **kwargs,
    ):
        k = float(self.atr_multiple)

        # Gate trailing until price has moved favorably past the entry
        if side in ("buy", "long"):
            # If price hasn't risen above entry, keep the original stop
            if current_price <= entry_price:
                return None
            candidate = current_price - k * atr
            if candidate > stop_loss:
                return round_to_tick(candidate, tick_size)
        else:
            # side == "sell" / "short": wait until price is below entry
            if current_price >= entry_price:
                return None
            candidate = current_price + k * atr
            if candidate < stop_loss:
                return round_to_tick(candidate, tick_size)

        return None


@dataclass
class FixedRatioExit(ExitStrategy):
    """
    Static stop using ATR multiple + fixed take-profit at R multiple of risk.
    - initial stop: entry ± K*ATR
    - take-profit: entry ± R * |entry - stop|
    - no trailing (update_stop returns None)
    """
    atr_multiple: float = 3.0
    rr_multiple: float = 3.0  # TP = rr_multiple * (price risk)

    def initial_levels(
        self,
        side: str,
        entry_price: float,
        atr: float,
        tick_size: float,
        **kwargs,
    ):
        k = float(self.atr_multiple)
        r = float(self.rr_multiple)

        if side in ("buy", "long"):
            sl = entry_price - k * atr
            move = r * abs(entry_price - sl)
            tp = entry_price + move
        else:
            sl = entry_price + k * atr
            move = r * abs(entry_price - sl)
            tp = entry_price - move

        sl = round_to_tick(sl, tick_size)
        tp = round_to_tick(tp, tick_size)
        return sl, tp

    def update_stop(
        self,
        side: str,
        stop_loss: float,
        entry_price: float,
        current_price: float,
        atr: float,
        tick_size: float,
        **kwargs,
    ):
        # Static stop; no trailing.
        return None


class ExitStrategy:
    """
    Unified interface for exit logic. Responsible for BOTH stop-loss and take-profit.

    Methods:
      - initial_levels: returns (stop_loss_price, take_profit_price_or_None)
      - update_stop: returns new stop or None to keep unchanged

    Backward-compat helpers:
      - initial_stop: used by older sizing code; returns only the stop.
      - ensure_models_loaded: optional no-op hook for RL/model-based strategies.
    """

    def initial_levels(
        self,
        side: str,
        entry_price: float,
        atr: float,
        tick_size: float,
        df=None,
        symbol: Optional[str] = None,
        **kwargs,
    ) -> Tuple[float, Optional[float]]:
        raise NotImplementedError

    def update_stop(
        self,
        side: str,
        stop_loss: float,
        entry_price: float,
        current_price: float,
        atr: float,
        tick_size: float,
        df=None,
        symbol: Optional[str] = None,
        **kwargs,
    ):
        raise NotImplementedError

    # ------- compatibility helpers (older code may call this) -------

    def initial_stop(
        self,
        side: str,
        entry_price: float,
        atr: float,
        df=None,
        symbol: Optional[str] = None,
        tick_size: float = 0.01,
        **kwargs,
    ) -> float:
        sl, _ = self.initial_levels(
            side=side,
            entry_price=entry_price,
            atr=atr,
            tick_size=tick_size,
            df=df,
            symbol=symbol,
            **kwargs,
        )
        return sl

    def ensure_models_loaded(self, env):  # optional
        return


@dataclass
class TrailingATRExit(ExitStrategy):
    """
    Trailing stop with ATR multiple; NO take profit.
    - initial stop: entry ± K*ATR (depending on side)
    - trailing: *only begins after price moves past entry in the trade's favor*,
                then follows K*ATR behind price, never loosening.
    """
    atr_multiple: float = 3.0

    def initial_levels(
        self,
        side: str,
        entry_price: float,
        atr: float,
        tick_size: float,
        **kwargs,
    ):
        k = float(self.atr_multiple)
        if side in ("buy", "long"):
            sl = entry_price - k * atr
        else:
            sl = entry_price + k * atr
        return round_to_tick(sl, tick_size), None  # no TP

    def update_stop(
        self,
        side: str,
        stop_loss: float,
        entry_price: float,
        current_price: float,
        atr: float,
        tick_size: float,
        **kwargs,
    ):
        k = float(self.atr_multiple)

        # Gate trailing until price has moved favorably past the entry
        if side in ("buy", "long"):
            # If price hasn't risen above entry, keep the original stop
            if current_price <= entry_price:
                return None
            candidate = current_price - k * atr
            if candidate > stop_loss:
                return round_to_tick(candidate, tick_size)
        else:
            # side == "sell" / "short": wait until price is below entry
            if current_price >= entry_price:
                return None
            candidate = current_price + k * atr
            if candidate < stop_loss:
                return round_to_tick(candidate, tick_size)

        return None


@dataclass
class FixedRatioExit(ExitStrategy):
    """
    Static stop using ATR multiple + fixed take-profit at R multiple of risk.
    - initial stop: entry ± K*ATR
    - take-profit: entry ± R * |entry - stop|
    - no trailing (update_stop returns None)
    """
    atr_multiple: float = 3.0
    rr_multiple: float = 3.0  # TP = rr_multiple * (price risk)

    def initial_levels(
        self,
        side: str,
        entry_price: float,
        atr: float,
        tick_size: float,
        **kwargs,
    ):
        k = float(self.atr_multiple)
        r = float(self.rr_multiple)

        if side in ("buy", "long"):
            sl = entry_price - k * atr
            move = r * abs(entry_price - sl)
            tp = entry_price + move
        else:
            sl = entry_price + k * atr
            move = r * abs(entry_price - sl)
            tp = entry_price - move

        sl = round_to_tick(sl, tick_size)
        tp = round_to_tick(tp, tick_size)
        return sl, tp

    def update_stop(
        self,
        side: str,
        stop_loss: float,
        entry_price: float,
        current_price: float,
        atr: float,
        tick_size: float,
        **kwargs,
    ):
        # Static stop; no trailing.
        return None


class RLTrailingATRExit(ExitStrategy):
    """
    Trailing stop (no TP) where the ATR multiple is chosen by a PPO policy.
    Uses ONLY previous CLOSED-bar features (no look-ahead).
    Falls back gracefully if models or SB3 aren't available, and logs
    clear warnings/errors on load failures.

    Long-window minute defaults:
      - ema_span default -> 240  (uses 'ema_240' by default)
      - rsi preference   -> ('rsi_120', 'rsi_60', 'rsi_14', then any 'rsi_*')
    """

    SL_MULTIPLIERS = [1.0, 2.0, 3.0, 4.0]

    def __init__(
        self,
        model_dir: str,
        fallback_multiple: float = 3.0,
        ema_span: int = 240,  # long, minute-friendly default to align with generator/trainer
        debug: bool = False,
        force_k: float | None = None,
        debug_first_n: int = 10,
        rsi_preference: tuple[str, ...] = ("rsi_120", "rsi_60", "rsi_14"),
        ema_fallback_order: tuple[str, ...] = ("ema_240", "ema_390", "ema_120", "ema_60"),
    ):
        self.model_dir = model_dir.rstrip("/ ")
        self.fallback_multiple = float(fallback_multiple)
        self.ema_col = f"ema_{int(ema_span)}"
        self.ema_fallback_order = tuple(ema_fallback_order or ())
        self.rsi_preference = tuple(rsi_preference or ())

        self.rl_models: dict[str, object] = {}
        self._loaded_models = False
        self._sb3_unavailable = False
        self.debug = bool(debug)
        self.force_k = None if force_k is None else float(force_k)
        self._dbg_counts = defaultdict(int)
        self._dbg_limit = int(debug_first_n)

    # --- Allow dynamic EMA span change from the bot ---
    def set_ema_span(self, ema_span: int):
        """Update the EMA column to match the bot's trend_ema_span (e.g., 240)."""
        self.ema_col = f"ema_{int(ema_span)}"
        if self.debug:
            print(f"[RL-EXIT] EMA column set to: {self.ema_col}")

    # Optional: allow runtime change of RSI preference
    def set_rsi_preference(self, *cols: str):
        self.rsi_preference = tuple(cols)
        if self.debug:
            print(f"[RL-EXIT] RSI preference set to: {self.rsi_preference}")

    # ---------- lifecycle ----------

    def _model_path(self, symbol: str) -> str:
        # Matches trainer's save path
        return f"{self.model_dir}/ppo_stop_loss_selector_rl_stop_loss_training_{symbol}.zip"

    def load_rl_models(self, symbols: list[str]):
        """
        Load PPO models for each symbol. Always prints an explicit message for:
          - missing/unsupported SB3 import (ERROR)
          - failed per-symbol load (WARNING)
          - successful load (info)
        """
        if self._sb3_unavailable:
            for symbol in symbols:
                self.rl_models[symbol] = None
            self._loaded_models = True
            return

        try:
            from stable_baselines3 import PPO  # type: ignore
        except Exception as e:
            print(f"[RL-EXIT][ERROR] stable_baselines3 not available: {e}. Falling back to fixed ATR multiples.")
            self._sb3_unavailable = True
            for symbol in symbols:
                self.rl_models[symbol] = None
            self._loaded_models = True
            return

        for symbol in symbols:
            path = self._model_path(symbol)
            try:
                self.rl_models[symbol] = PPO.load(path, device="cpu")
                print(f"[RL-EXIT] ✅ Loaded model for {symbol}: {path}")
            except Exception as e:
                print(
                    f"[RL-EXIT][WARNING] Failed to load model for {symbol} at {path}: {e}. "
                    f"Using fallback multiple={self.fallback_multiple}."
                )
                self.rl_models[symbol] = None

        self._loaded_models = True

    def ensure_models_loaded(self, env):
        if not self._loaded_models:
            syms = env.get_asset_list()
            if syms and not isinstance(syms[0], str):
                syms = [a.get("symbol") for a in syms]
            self.load_rl_models(syms)

    # ---------- ExitStrategy interface ----------

    def initial_levels(
        self,
        side: str,
        entry_price: float,
        atr: float,
        tick_size: float,
        df: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        **kwargs,
    ) -> Tuple[float, Optional[float]]:
        k = self._choose_multiple(df, symbol, side, atr)
        sl = entry_price - k * atr if side in ("buy", "long") else entry_price + k * atr
        return round_to_tick(sl, tick_size), None

    def update_stop(
        self,
        side: str,
        stop_loss: float,
        entry_price: float,
        current_price: float,
        atr: float,
        tick_size: float,
        df: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        **kwargs,
    ):
        k = self._choose_multiple(df, symbol, side, atr)

        # Gate trailing until price moves favorably past entry
        if side in ("buy", "long") and current_price <= entry_price:
            return None
        if side in ("sell", "short") and current_price >= entry_price:
            return None

        if side in ("buy", "long"):
            candidate = current_price - k * atr
            if candidate > stop_loss:
                return round_to_tick(candidate, tick_size)
        else:
            candidate = current_price + k * atr
            if candidate < stop_loss:
                return round_to_tick(candidate, tick_size)
        return None

    # ---------- internals ----------

    def _pick_rsi(self, prev_row: pd.Series) -> float:
        # Try preferred RSI columns in order
        for name in self.rsi_preference:
            v = prev_row.get(name, np.nan)
            if np.isfinite(v):
                return float(v)
        # Fallback: any 'rsi_' column
        for c in prev_row.index:
            if isinstance(c, str) and c.startswith("rsi_"):
                v = prev_row.get(c, np.nan)
                if np.isfinite(v):
                    if self.debug:
                        print(f"[RL-EXIT] Using fallback RSI column: {c}")
                    return float(v)
        return np.nan

    def _pick_ema(self, prev_row: pd.Series) -> float:
        # 1) Requested EMA column (e.g., 'ema_240')
        v = prev_row.get(self.ema_col, np.nan)
        if np.isfinite(v):
            return float(v)
        # 2) Fallback order (ema_240, ema_390, ema_120, ema_60 by default)
        for name in self.ema_fallback_order:
            v = prev_row.get(name, np.nan)
            if np.isfinite(v):
                if self.debug:
                    print(f"[RL-EXIT] {self.ema_col} missing; using fallback {name}")
                return float(v)
        # 3) Any 'ema_' column
        for c in prev_row.index:
            if isinstance(c, str) and c.startswith("ema_"):
                v = prev_row.get(c, np.nan)
                if np.isfinite(v):
                    if self.debug:
                        print(f"[RL-EXIT] {self.ema_col} missing; using generic {c}")
                    return float(v)
        return np.nan

    def _choose_multiple(
        self,
        df: Optional[pd.DataFrame],
        symbol: Optional[str],
        side: str,
        atr: float,
    ) -> float:
        # Force constant K for debugging
        if self.force_k is not None:
            return float(self.force_k)

        if df is None or symbol is None or len(df) < 2:
            return self.fallback_multiple

        model = self.rl_models.get(symbol)
        if model is None:
            return self.fallback_multiple

        prev = df.iloc[-2]  # previous CLOSED bar (no look-ahead)
        close_val = prev.get("close", np.nan)
        rsi_val = self._pick_rsi(prev)       # prefer rsi_120 → rsi_60 → rsi_14 → any rsi_*
        ema_val = self._pick_ema(prev)       # prefer ema_{ema_span} (default 240)

        pt_numeric = 1 if side in ("long", "buy") else -1
        obs = np.array([atr, rsi_val, close_val, ema_val, pt_numeric], dtype=np.float32)

        if not np.all(np.isfinite(obs)):
            key = (symbol, "nan_obs")
            if self.debug and self._dbg_counts[key] < self._dbg_limit:
                self._dbg_counts[key] += 1
                print(f"[RL-EXIT] {symbol} NaN/Inf in obs {obs}")
            return self.fallback_multiple

        try:
            action, _ = model.predict(obs, deterministic=True)
            idx = int(np.asarray(action).flatten()[0])
            idx = max(0, min(idx, len(self.SL_MULTIPLIERS) - 1))
            return float(self.SL_MULTIPLIERS[idx])
        except Exception as e:
            key = (symbol, "infer_err")
            if self.debug and self._dbg_counts[key] < self._dbg_limit:
                self._dbg_counts[key] += 1
                print(f"[RL-EXIT] {symbol} inference error: {e}")
            return self.fallback_multiple
            

# --- NEW: Fail-Safe Hybrid Exit (MFE Ratchet + Donchian + Two-Speed Keltner) ---

@dataclass
class FailSafeHybridExit(ExitStrategy):
    """
    Fail-safe hybrid exit with real-time, *readable* logging.

    New:
      - log_mode: 'all' | 'tighten_only' | 'milestones'
      - log_columns: 'full' | 'compact' | custom list
      - throttle_every_n_bars: write a heartbeat row every N bars (unless a milestone fires)
      - summarize_* helpers for quick diagnosis
    """

    # --- activation & initial ---
    k_init: float = 3.0
    start_trailing_R: float = 1.0
    breakeven_R: Optional[float] = 1.0

    # --- MFE ratchet ---
    giveback: float = 0.40
    backoff_atr: float = 0.60

    # --- Donchian structure ---
    donchian_N: int = 30
    buffer_atr_mult: float = 0.80

    # --- Two-speed Keltner ---
    k_fast: float = 1.5
    k_slow: float = 2.3
    atr_slow_window: int = 60

    # --- Impulse / breakout gate ---
    impulse_sigma: float = 1.0
    breakout_N: int = 20

    # --- Indicator column preferences (ordered fallbacks) ---
    atr_fast_preference: tuple[str, ...] = ("atr_30", "atr_60", "atr_90")
    atr_slow_preference: Optional[tuple[str, ...]] = None

    # --- Misc ---
    min_step_ticks: int = 1
    debug: bool = False
    strict_contract: bool = True

    # --- Logging controls ---
    logging_enabled: bool = False
    log_path: Optional[str] = None
    log_flush: bool = True
    log_max_events_per_symbol: int = 200_000

    # >>> Readability controls <<<
    log_mode: str = "milestones"            # 'all' | 'tighten_only' | 'milestones'
    log_columns: str | list[str] = "compact" # 'full' | 'compact' | explicit list
    log_columns_custom: Optional[list[str]] = None
    throttle_every_n_bars: int = 0          # 0 = off (only milestones)

    # state + logs
    _state: Dict[str, Dict] = field(default_factory=lambda: defaultdict(dict))
    _logs: Dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    _csv_header_written: bool = field(default=False, init=False)
    _bar_counter: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # ----------------- helpers -----------------

    def _prev_val(self, df: pd.DataFrame, col: str) -> float:
        if col not in df.columns:
            if self.strict_contract: raise RuntimeError(f"[FailSafeHybridExit] Missing required column: {col}")
            return np.nan
        if len(df) < 2:
            if self.strict_contract: raise RuntimeError("[FailSafeHybridExit] Not enough rows to access previous bar.")
            return np.nan
        v = df[col].iloc[-2]
        if not np.isfinite(v):
            if self.strict_contract: raise RuntimeError(f"[FailSafeHybridExit] Previous-bar NaN/inf: {col}")
            return np.nan
        return float(v)

    def _opt_prev(self, df: pd.DataFrame, *candidates: str) -> float | np.nan:
        for c in candidates:
            if c in df.columns:
                try:
                    v = df[c].iloc[-2]
                    return float(v) if np.isfinite(v) else np.nan
                except Exception:
                    continue
        return np.nan

    def _require_prev_indicator(self, df: pd.DataFrame, candidates: Iterable[str], label: str) -> float:
        for c in candidates:
            try:
                return self._prev_val(df, c)
            except RuntimeError:
                continue
        raise RuntimeError(f"[FailSafeHybridExit] Unable to obtain required {label}. Tried: {list(candidates)}")

    def _atr_fast(self, df: pd.DataFrame) -> float:
        return self._require_prev_indicator(df, self.atr_fast_preference, "ATR_fast")

    def _atr_slow(self, df: pd.DataFrame) -> float:
        prefs = self.atr_slow_preference or (f"atr_{int(self.atr_slow_window)}", "atr_120","atr_150","atr_180","atr_210","atr_240","atr_390")
        return self._require_prev_indicator(df, prefs, "ATR_slow")

    def _previous_hlc(self, df: pd.DataFrame):
        prev_close = self._prev_val(df, "close")
        prev_high  = self._prev_val(df, "high") if "high" in df.columns else prev_close
        prev_low   = self._prev_val(df, "low")  if "low"  in df.columns else prev_close
        return prev_close, prev_high, prev_low

    # ---------- state ----------
    def _ensure_state(self, symbol, entry_price, stop_loss):
        st = self._state[symbol]
        if not st:
            R = abs(float(entry_price) - float(stop_loss))
            st.update(dict(entry=float(entry_price), initial_stop=float(stop_loss),
                           R=float(R), activated=False, peak=float(entry_price), last_close=np.nan))
        return st

    # ---------- logging core ----------
    def _ensure_log(self, symbol: str):
        if symbol not in self._logs or not isinstance(self._logs[symbol], deque):
            self._logs[symbol] = deque(maxlen=int(self.log_max_events_per_symbol))
        if self._logs[symbol].maxlen != int(self.log_max_events_per_symbol):
            self._logs[symbol] = deque(self._logs[symbol], maxlen=int(self.log_max_events_per_symbol))

    def _select_columns(self, event: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(self.log_columns, list):
            cols = self.log_columns
        elif self.log_columns == "compact":
            cols = [
                "timestamp","symbol","side","phase","driver",
                "activated","just_activated","breakeven_applied",
                "prev_stop","new_stop","delta_ticks","mfe","R",
                "entry","current_price_prev_close",
                "atr_fast","atr_ratio_sm","mins_since_open","tod_bucket"
            ]
        else:  # 'full'
            return event
        if self.log_columns_custom:
            cols = self.log_columns_custom
        return {k: event.get(k, np.nan) for k in cols}

    def _should_log(self, symbol: str, event: Dict[str, Any]) -> bool:
        phase = event.get("phase")
        tightened = np.isfinite(event.get("new_stop", np.nan)) and event.get("new_stop") != event.get("prev_stop")
        if self.log_mode == "tighten_only":
            if phase == "init": return True
            return bool(tightened)
        if self.log_mode == "milestones":
            if phase == "init": return True
            if event.get("just_activated"): return True
            if event.get("breakeven_applied"): return True
            if tightened: return True
            # throttle heartbeat?
            if self.throttle_every_n_bars > 0:
                self._bar_counter[symbol] += 1
                if self._bar_counter[symbol] >= self.throttle_every_n_bars:
                    self._bar_counter[symbol] = 0
                    return True
            return False
        # 'all'
        if self.throttle_every_n_bars > 0 and phase == "update" and not tightened and not event.get("just_activated") and not event.get("breakeven_applied"):
            self._bar_counter[symbol] += 1
            if self._bar_counter[symbol] < self.throttle_every_n_bars:
                return False
            self._bar_counter[symbol] = 0
        return True

    def _write_event_csv(self, event: Dict[str, Any]):
        if not (self.logging_enabled and self.log_path):
            return
        path = self.log_path
        try:
            d = os.path.dirname(path)
            if d: os.makedirs(d, exist_ok=True)
        except Exception:
            pass
        filtered = self._select_columns(event)
        write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0) or (not self._csv_header_written)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(filtered.keys()))
            if write_header:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(filtered)
            if self.log_flush:
                f.flush()

    def _append_event(self, symbol: str, event: Dict[str, Any]):
        if not self.logging_enabled:
            return
        if not self._should_log(symbol, event):
            return
        self._ensure_log(symbol)
        filtered = self._select_columns(event)
        self._logs[symbol].append(filtered)
        if self.debug:
            print("[ExitLog]", filtered)
        self._write_event_csv(event)  # write full/compact as configured

    # ----------------- ExitStrategy interface -----------------

    def initial_levels(self, side: str, entry_price: float, atr: float, tick_size: float,
                       df: Optional[pd.DataFrame] = None, symbol: Optional[str] = None, **kwargs):
        if df is None or len(df) < 2:
            raise RuntimeError("[FailSafeHybridExit] initial_levels requires df with ≥2 rows.")
        atr_fast = self._atr_fast(df)
        sl = round_to_tick(entry_price - self.k_init * atr_fast if side in ("buy","long")
                           else entry_price + self.k_init * atr_fast, tick_size)
        st = self._state[symbol]; st.clear()
        R = abs(entry_price - sl)
        st.update(dict(entry=float(entry_price), initial_stop=float(sl), R=float(R),
                       activated=False, peak=float(entry_price), last_close=np.nan))
        prev_close = self._prev_val(df, "close")
        st["last_close"] = prev_close

        ts = df['date'].iloc[-2] if 'date' in df.columns else None
        self._append_event(symbol, {
            "phase":"init","timestamp":ts,"symbol":symbol,"side":side,
            "entry_price":float(entry_price),"initial_stop":float(sl),"R":float(R),
            "atr_fast":float(atr_fast),
            "atr_slow": self._opt_prev(df, f"atr_{self.atr_slow_window}","atr_120","atr_150","atr_180","atr_210","atr_240","atr_390"),
            "atr_ratio": self._opt_prev(df, "atrR_90_390","atr_ratio_90_390","atr_ratio"),
            "atr_ratio_sm": self._opt_prev(df, "atrRsm_14_60","atr_ratio_sm_14_60","atr_ratio_sm"),
            "mins_since_open": self._opt_prev(df, "mins_since_open"),
            "tod_bucket": self._opt_prev(df, "tod_bucket","tod_phase","tod_bin"),
        })
        return sl, None

    def update_stop(self, side: str, stop_loss: float, entry_price: float, current_price: float,
                    atr: float, tick_size: float, df: Optional[pd.DataFrame] = None,
                    symbol: Optional[str] = None, **kwargs):
        if symbol is None: raise RuntimeError("[FailSafeHybridExit] update_stop requires symbol.")
        if df is None or len(df) < 2: raise RuntimeError("[FailSafeHybridExit] update_stop requires df with ≥2 rows.")
        st = self._ensure_state(symbol, entry_price, stop_loss)

        atr_fast = self._atr_fast(df); atr_slow = self._atr_slow(df)
        prev_close, prev_high, prev_low = self._previous_hlc(df)
        if side in ("buy","long"):
            st["peak"] = max(float(st.get("peak", entry_price)), prev_high, prev_close); mfe = st["peak"] - st["entry"]
        else:
            trough = min(float(st.get("peak", entry_price)), prev_low, prev_close); st["peak"] = trough; mfe = st["entry"] - st["peak"]

        R = max(float(st["R"]), 1e-12)
        candidate = None

        breakeven_applied = False
        if self.breakeven_R is not None and mfe >= float(self.breakeven_R) * R:
            breakeven_applied = True
            be = round_to_tick(st["entry"] + tick_size if side in ("buy","long") else st["entry"] - tick_size, tick_size)
            candidate = max(candidate or -np.inf, be) if side in ("buy","long") else min(candidate or np.inf, be)

        just_activated = False
        if not st.get("activated", False) and mfe >= float(self.start_trailing_R) * R:
            st["activated"] = True; just_activated = True

        ratchet = donch = kelt = None; driver = "none"
        if st.get("activated", False):
            ratchet = self._ratchet_candidate(side, st["peak"], mfe, current_price, atr_fast)
            donch   = self._donchian_candidate(side, df, current_price, atr_fast)
            kelt    = self._keltner_candidate(side, df, st, current_price, atr_fast, atr_slow)
            if side in ("buy","long"):
                tightest = max([x for x in [ratchet,donch,kelt] if x is not None] or [-np.inf])
            else:
                tightest = min([x for x in [ratchet,donch,kelt] if x is not None] or [np.inf])
            if np.isfinite(tightest):
                candidate = self._respect_backoff(side, tightest, current_price, atr_fast)
                driver = "ratchet" if (ratchet is not None and math.isclose(tightest,ratchet,rel_tol=0,abs_tol=1e-12)) else \
                         "donch"   if (donch   is not None and math.isclose(tightest,donch,  rel_tol=0,abs_tol=1e-12)) else \
                         "keltner" if (kelt    is not None and math.isclose(tightest,kelt,   rel_tol=0,abs_tol=1e-12)) else "unknown"

        new_stop = self._commit(side, stop_loss, candidate, current_price, atr_fast, tick_size)

        ts = df['date'].iloc[-2] if 'date' in df.columns else None
        atrR   = self._opt_prev(df, "atrR_90_390","atr_ratio_90_390","atr_ratio")
        atrRsm = self._opt_prev(df, "atrRsm_14_60","atr_ratio_sm_14_60","atr_ratio_sm")
        mins_open = self._opt_prev(df, "mins_since_open")
        tod_bkt   = self._opt_prev(df, "tod_bucket","tod_phase","tod_bin")
        bbw_60    = self._opt_prev(df, "bb60_bandwidth","bb60_bw","bb_bw_60")
        dc_pos60  = self._opt_prev(df, "dc60_pos","donchian_pos_60","donchian_pos")

        event = {
            "phase":"update","timestamp":ts,"symbol":symbol,"side":side,
            "activated":bool(st.get("activated", False)),
            "just_activated":bool(just_activated),
            "breakeven_applied":bool(breakeven_applied),
            "driver":driver,
            "ratchet": float(ratchet) if ratchet is not None else np.nan,
            "donch":   float(donch)   if donch   is not None else np.nan,
            "keltner": float(kelt)    if kelt    is not None else np.nan,
            "candidate_after_backoff": float(candidate) if (candidate is not None and np.isfinite(candidate)) else np.nan,
            "prev_stop": float(stop_loss) if np.isfinite(stop_loss) else np.nan,
            "new_stop": float(new_stop) if new_stop is not None else np.nan,
            "delta_ticks": (float((new_stop - stop_loss)/max(tick_size,1e-12)) if (new_stop is not None and np.isfinite(stop_loss)) else 0.0),
            "entry": float(st.get("entry", entry_price)),
            "current_price_prev_close": float(prev_close),
            "current_price_live": float(current_price),
            "peak_or_trough": float(st.get("peak", np.nan)),
            "mfe": float(mfe), "R": float(R),
            "k_init": self.k_init, "start_trailing_R": self.start_trailing_R,
            "breakeven_R": self.breakeven_R if self.breakeven_R is not None else np.nan,
            "giveback": self.giveback, "backoff_atr": self.backoff_atr,
            "donchian_N": self.donchian_N, "buffer_atr_mult": self.buffer_atr_mult,
            "k_fast": self.k_fast, "k_slow": self.k_slow, "atr_slow_window": self.atr_slow_window,
            "impulse_sigma": self.impulse_sigma, "breakout_N": self.breakout_N,
            "min_step_ticks": self.min_step_ticks,
            "atr_fast": float(atr_fast), "atr_slow": float(atr_slow),
            "atr_ratio": float(atrR) if np.isfinite(atrR) else np.nan,
            "atr_ratio_sm": float(atrRsm) if np.isfinite(atrRsm) else np.nan,
            "bb_bw_60": float(bbw_60) if np.isfinite(bbw_60) else np.nan,
            "donch_pos_60": float(dc_pos60) if np.isfinite(dc_pos60) else np.nan,
            "mins_since_open": float(mins_open) if np.isfinite(mins_open) else np.nan,
            "tod_bucket": float(tod_bkt) if np.isfinite(tod_bkt) else np.nan,
        }
        self._append_event(symbol, event)
        return new_stop

    def update_trailing(self, *args, **kwargs): return self.update_stop(*args, **kwargs)

    # -------- candidates --------
    def _ratchet_candidate(self, side, peak, mfe, current_price, atr_fast):
        lock_dist = max(self.giveback * float(mfe), self.backoff_atr * atr_fast)
        cand = float(peak) - lock_dist if side in ("buy","long") else float(peak) + lock_dist
        return self._respect_backoff(side, cand, current_price, atr_fast)

    def _donchian_candidate(self, side, df, current_price, atr_fast):
        N = int(max(2, self.donchian_N))
        lows  = df["low"].iloc[-(N+1):-1]  if "low"  in df.columns else pd.Series([])
        highs = df["high"].iloc[-(N+1):-1] if "high" in df.columns else pd.Series([])
        if lows.empty and highs.empty:
            prev_close = self._prev_val(df, "close")
            lows, highs = pd.Series([prev_close]), pd.Series([prev_close])
        buff = float(self.buffer_atr_mult) * atr_fast
        if side in ("buy","long"):
            base = float(np.nanmin(lows.values)) if len(lows) else self._prev_val(df, "low")
            cand = base + buff
        else:
            base = float(np.nanmax(highs.values)) if len(highs) else self._prev_val(df, "high")
            cand = base - buff
        return self._respect_backoff(side, cand, current_price, atr_fast)

    def _keltner_candidate(self, side, df, st, current_price, atr_fast, atr_slow):
        last_close = st.get("last_close", np.nan)
        prev_close = self._prev_val(df, "close")
        st["last_close"] = prev_close
        impulse = False
        if np.isfinite(last_close) and np.isfinite(prev_close):
            jump = (prev_close - last_close) if side in ("buy","long") else (last_close - prev_close)
            impulse = (jump >= self.impulse_sigma * atr_fast)
        N = int(max(2, self.breakout_N))
        highs = df["high"].iloc[-(N+1):-1] if "high" in df.columns else pd.Series([])
        lows  = df["low"].iloc[-(N+1):-1]  if "low"  in df.columns else pd.Series([])
        breakout = False
        if len(highs) and len(lows):
            breakout = bool(prev_close >= np.nanmax(highs.values)) if side in ("buy","long") else bool(prev_close <= np.nanmin(lows.values))
        use_fast = impulse or breakout
        if side in ("buy","long"):
            cand_fast = st["peak"] - self.k_fast * atr_fast
            cand_slow = st["peak"] - self.k_slow * atr_slow
            cand = cand_fast if use_fast else cand_slow
        else:
            cand_fast = st["peak"] + self.k_fast * atr_fast
            cand_slow = st["peak"] + self.k_slow * atr_slow
            cand = cand_fast if use_fast else cand_slow
        return self._respect_backoff(side, cand, current_price, atr_fast)

    # -------- backoff & commit --------
    def _respect_backoff(self, side, cand, current_price, atr_fast):
        gap = float(self.backoff_atr) * float(atr_fast)
        return min(float(cand), float(current_price) - gap) if side in ("buy","long") \
               else max(float(cand), float(current_price) + gap)

    def _commit(self, side, prev_stop, candidate, current_price, atr, tick_size):
        if candidate is None or not np.isfinite(candidate): return None
        step = float(self.min_step_ticks) * float(tick_size)
        cand = round_to_tick(float(candidate), tick_size)
        if side in ("buy","long"):
            cand = min(cand, current_price - tick_size)
            if cand <= prev_stop + step/2: return None
            return cand if cand > prev_stop else None
        else:
            cand = max(cand, current_price + tick_size)
            if cand >= prev_stop - step/2: return None
            return cand if cand < prev_stop else None

    # ----------------- Public utilities -----------------

    def set_logging(self, enabled: bool, path: Optional[str] = None, flush: Optional[bool] = None,
                    mode: Optional[str] = None, columns: Optional[str | list[str]] = None,
                    throttle_every_n_bars: Optional[int] = None):
        self.logging_enabled = bool(enabled)
        if path is not None:
            self.log_path = path
            self._csv_header_written = False
        if flush is not None: self.log_flush = bool(flush)
        if mode is not None: self.log_mode = str(mode)
        if columns is not None: self.log_columns = columns
        if throttle_every_n_bars is not None: self.throttle_every_n_bars = int(throttle_every_n_bars)

    def get_log_df(self, symbol: Optional[str] = None) -> pd.DataFrame:
        if not self._logs: return pd.DataFrame()
        rows = [e for deq in self._logs.values() for e in deq] if symbol is None else list(self._logs.get(symbol, []))
        return pd.json_normalize(rows, sep="__")

    def get_compact_df(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Convenience: milestones/tightens + compact columns regardless of instance settings."""
        df = self.get_log_df(symbol)
        if df.empty: return df
        tightened = df["new_stop"].notna() & df["prev_stop"].notna() & (df["new_stop"] != df["prev_stop"])
        mask = (df["phase"].eq("init")) | (df.get("just_activated", False) == True) \
               | (df.get("breakeven_applied", False) == True) | tightened
        cols = [
            "timestamp","symbol","side","phase","driver","activated","just_activated",
            "breakeven_applied","prev_stop","new_stop","delta_ticks","mfe","R",
            "entry","current_price_prev_close","atr_fast","atr_ratio_sm","mins_since_open","tod_bucket"
        ]
        cols = [c for c in cols if c in df.columns]
        out = df.loc[mask, cols].copy()
        out.sort_values(["timestamp","symbol"], inplace=True)
        return out

    def summarize_by_driver(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Small table per driver: counts and median tightening in ticks."""
        df = self.get_log_df(symbol)
        if df.empty: return df
        df = df[df["new_stop"].notna() & df["prev_stop"].notna() & (df["new_stop"] != df["prev_stop"])]
        if df.empty: return pd.DataFrame()
        g = df.groupby("driver")["delta_ticks"]
        return pd.DataFrame({"count": g.size(), "median_delta_ticks": g.median(), "mean_delta_ticks": g.mean()}).reset_index()

    def summarize_by_regime(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Bucket by ATR_ratio_sm (trend/chop proxy) and ToD to see where pain concentrates."""
        df = self.get_log_df(symbol)
        if df.empty: return df
        df = df.copy()
        # bins for atr_ratio_sm
        df["regime"] = pd.cut(df["atr_ratio_sm"], [-np.inf, 0.9, 1.1, np.inf], labels=["low","neutral","high"])
        # ToD coarse buckets
        tod = df["mins_since_open"]
        df["tod_bucket_coarse"] = pd.cut(tod, [-np.inf, 60, 180, 360, np.inf], labels=["open_1h","mid_morning","midday","late"])
        # focus on tighten rows
        mask = df["new_stop"].notna() & df["prev_stop"].notna() & (df["new_stop"] != df["prev_stop"])
        g = df[mask].groupby(["regime","tod_bucket_coarse","driver"])["delta_ticks"]
        return g.agg(["count","median","mean"]).reset_index().sort_values(["regime","tod_bucket_coarse","driver"])

