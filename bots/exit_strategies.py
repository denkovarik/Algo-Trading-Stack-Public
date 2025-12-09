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

    Required methods:
      - initial_levels: returns (stop_loss_price, take_profit_price_or_None)
      - update_stop: returns new stop or None to keep unchanged

    Backward-compat helpers:
      - initial_stop: used by older sizing code; returns only the stop.
      - ensure_models_loaded: optional no-op hook for RL/model-based strategies.

    New (optional) contract helpers for plug-and-play exits:
      - required_indicator_config(): list of indicator configs your env/registry can compute if missing.
      - required_indicator_columns(): concrete df columns that must be present before calling this exit.
      - warmup_bars(): number of historical bars required before bar i may be labeled/traded.
    """

    # --------- main interface ---------

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

    # --------- optional plug-and-play hooks (new) ---------

    def required_indicator_config(self) -> List[dict]:
        """
        Return a list of indicator configs (same schema your TradingEnvironment uses):
          {"name": <indicator_name>, "params": {...}}
        The orchestrator/labeler will ensure these are computed if missing.
        Default: []  (no additional compute requested)
        """
        return []

    def required_indicator_columns(self) -> List[str]:
        """
        Return a list of df column names the exit expects to exist (e.g., ["atr_90","dc60_high"]).
        The orchestrator/labeler will verify these after computing configs, before running the exit.
        Default: []  (no explicit requirement)
        """
        return []

    def warmup_bars(self) -> int:
        """
        Minimum number of historical bars the exit needs before bar i is eligible.
        Default: 1 (need at least a previous closed bar).
        """
        return 1

    # --------- compatibility helpers ---------

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

    Contract:
      • Uses previous CLOSED bar for trailing decisions (provide that price via `current_price`).
      • Requires ATR_90 by default; declare it below so the orchestrator/env can ensure it exists.
    """
    atr_multiple: float = 3.0
    atr_window: int = 90            # allow easy reuse; still defaults to 90
    atr_col: Optional[str] = None   # if None, will use f"atr_{atr_window}"

    # ---- Plug-and-play declarations (new; optional but explicit here) ----
    def required_indicator_config(self) -> List[dict]:
        # Ask the env registry to compute ATR(<window>) with prefix 'atr'
        return [{"name": "atr", "params": {"window": int(self.atr_window), "prefix": "atr"}}]

    def required_indicator_columns(self) -> List[str]:
        # Ensure the canonical column exists
        col = self.atr_col or f"atr_{int(self.atr_window)}"
        return [col]

    def warmup_bars(self) -> int:
        # Need at least one previous CLOSED bar (to gate trailing and read previous ATR)
        return 1

    # ---- Core behavior (unchanged logic) ----
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
            if current_price <= entry_price:
                return None
            candidate = current_price - k * atr
            if candidate > stop_loss:
                return round_to_tick(candidate, tick_size)
        else:
            # side == "sell" / "short"
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
            

