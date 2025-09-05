# bots/exit_strategies.py

from __future__ import annotations
from collections import defaultdict
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass


from classes.API_Interface import round_to_tick


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
    Falls back gracefully if models or SB3 aren't available.

    Works for both CoinFlipBot (ema_span=21) and TrendFollowingBot.
    """

    SL_MULTIPLIERS = [1.0, 2.0, 3.0, 4.0]

    def __init__(
        self,
        model_dir: str,
        fallback_multiple: float = 3.0,
        ema_span: int = 21,
        debug: bool = False,
        force_k: float | None = None,
        debug_first_n: int = 10,
    ):
        self.model_dir = model_dir.rstrip("/ ")
        self.fallback_multiple = float(fallback_multiple)
        self.ema_col = f"ema_{int(ema_span)}"
        self.rl_models: dict[str, object] = {}
        self._loaded_models = False
        self._sb3_unavailable = False
        self.debug = bool(debug)
        self.force_k = None if force_k is None else float(force_k)
        self._dbg_counts = defaultdict(int)
        self._dbg_limit = int(debug_first_n)

    # --- Allow dynamic EMA span change from the bot ---
    def set_ema_span(self, ema_span: int):
        """Update the EMA column to match the bot's trend_ema_span."""
        self.ema_col = f"ema_{int(ema_span)}"
        if self.debug:
            print(f"[RL-EXIT] EMA column set to: {self.ema_col}")

    # ---------- lifecycle ----------

    def _model_path(self, symbol: str) -> str:
        # Matches trainer's save path
        return f"{self.model_dir}/ppo_stop_loss_selector_rl_stop_loss_training_{symbol}.zip"

    def load_rl_models(self, symbols: list[str]):
        if self._sb3_unavailable:
            for symbol in symbols:
                self.rl_models[symbol] = None
            self._loaded_models = True
            return
        try:
            from stable_baselines3 import PPO  # type: ignore
        except Exception as e:
            if self.debug:
                print(f"[RL-EXIT] SB3 missing: {e}")
            self._sb3_unavailable = True
            for symbol in symbols:
                self.rl_models[symbol] = None
            self._loaded_models = True
            return
        for symbol in symbols:
            path = self._model_path(symbol)
            try:
                self.rl_models[symbol] = PPO.load(path, device="cpu")
                if self.debug:
                    print(f"[RL-EXIT] Loaded model for {symbol}: {path}")
            except Exception as e:
                if self.debug:
                    print(f"[RL-EXIT] FAILED to load for {symbol} @ {path}: {e} -> fallback {self.fallback_multiple}")
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

        prev = df.iloc[-2]
        close_val = prev.get("close", np.nan)
        rsi_val = prev.get("rsi_14", np.nan)
        ema_val = prev.get(self.ema_col, np.nan)

        # Fallback to any available EMA if the configured one is missing
        if not np.isfinite(ema_val):
            ema_candidates = [c for c in prev.index if isinstance(c, str) and c.startswith("ema_")]
            for c in ema_candidates:
                v = prev.get(c, np.nan)
                if np.isfinite(v):
                    ema_val = v
                    if self.debug:
                        print(f"[RL-EXIT] {symbol}: {self.ema_col} missing/NaN; using {c} instead")
                    break

        pt_numeric = 1 if side in ("long", "buy") else -1
        obs = np.array([atr, rsi_val, close_val, ema_val, pt_numeric], dtype=np.float32)

        if not np.all(np.isfinite(obs)):
            if self.debug:
                print(f"[RL-EXIT] {symbol} NaN/Inf in obs {obs}")
            return self.fallback_multiple

        try:
            action, _ = model.predict(obs, deterministic=True)
            idx = int(np.asarray(action).flatten()[0])
            idx = max(0, min(idx, len(self.SL_MULTIPLIERS) - 1))
            return float(self.SL_MULTIPLIERS[idx])
        except Exception as e:
            if self.debug:
                print(f"[RL-EXIT] {symbol} inference error: {e}")
            return self.fallback_multiple
            
            
class StopLossToExitAdapter(ExitStrategy):
    """Adapter to wrap an old StopLossStrategy so it can be used as an ExitStrategy."""
    def __init__(self, sl_strategy):
        self.sl_strategy = sl_strategy

    def initial_levels(self, side, entry_price, atr, tick_size, **kwargs):
        stop_loss_price = self.sl_strategy.initial_stop(side, entry_price, atr, **kwargs)
        return round_to_tick(stop_loss_price, tick_size), None

    def update_stop(self, side, stop_loss, entry_price, current_price, atr, tick_size, **kwargs):
        return self.sl_strategy.update_stop(side, stop_loss, entry_price, current_price, atr, **kwargs)


