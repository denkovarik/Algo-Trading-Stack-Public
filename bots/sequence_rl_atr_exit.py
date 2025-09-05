# sequence_rl_atr_exit.py
# Drop-in ExitStrategy that uses a TorchScript LSTM classifier on a rolling window
# to choose the ATR multiple each bar. Mirrors RLTrailingATRExit behavior:
# - uses ONLY previous CLOSED-bar features (no look-ahead)
# - trails only after price moves favorably past entry
# - never loosens the stop
#
# Train models with train_sequence_sl_model.py (outputs TorchScript .pt + .meta.json).
# At runtime this class loads per-symbol models from `model_dir`.

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple, Dict

import json
import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # torch not available → inference falls back to fixed multiple
    torch = None  # type: ignore

from classes.API_Interface import round_to_tick
from bots.exit_strategies import ExitStrategy


@dataclass
class SequenceRLATRExit(ExitStrategy):
    """
    Sequence (LSTM) ATR-multiple selector with TorchScript inference.

    Inputs per bar (previous CLOSED bar):
      [ atr, rsi_14, close, ema_<ema_span>, position_flag(+1 long / -1 short) ]

    The model consumes a rolling window of length "lookback" for the above
    feature vector, standardized per window (z-score), and outputs class logits
    over a fixed list of ATR multiples (default [1,2,3,4]).

    Fallback behavior:
      - If a symbol's model or buffer isn't ready → use `fallback_multiple`.
      - If torch is unavailable → always use `fallback_multiple`.
    """
    model_dir: str
    fallback_multiple: float = 3.0
    ema_span: int = 21
    debug: bool = False

    # will be overwritten by meta["sl_multipliers"] if present
    SL_MULTIPLIERS = [1.0, 2.0, 3.0, 4.0]

    # ---- runtime state (lazy-initialized) ----
    def __post_init__(self):
        self.model_dir = self.model_dir.rstrip("/ ")
        self.ema_col = f"ema_{int(self.ema_span)}"
        # Per symbol: model, meta (lookback/features), rolling buffer
        self.models: Dict[str, object] = {}
        self.meta: Dict[str, dict] = {}
        self.buffers: Dict[str, deque] = {}
        self._loaded = False

    # --- Allow dynamic EMA span change from the bot (keeps interface parity) ---
    def set_ema_span(self, ema_span: int):
        self.ema_col = f"ema_{int(ema_span)}"
        if self.debug:
            print(f"[SEQ-EXIT] EMA column set to: {self.ema_col}")

    # ---------------- ExitStrategy interface ----------------

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
        return round_to_tick(sl, tick_size), None  # no TP

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

        # Gate trailing until price has moved favorably past the entry
        if side in ("buy", "long"):
            if current_price <= entry_price:
                return None
            candidate = current_price - k * atr
            if candidate > stop_loss:
                return round_to_tick(candidate, tick_size)
        else:
            if current_price >= entry_price:
                return None
            candidate = current_price + k * atr
            if candidate < stop_loss:
                return round_to_tick(candidate, tick_size)

        return None

    # ---------------- lifecycle ----------------

    def ensure_models_loaded(self, env):
        """
        Lazily load per-symbol models on first use. Called by the bot right
        before trading (so the symbol list is stable).
        """
        if self._loaded:
            return
        syms = env.get_asset_list()
        if syms and not isinstance(syms[0], str):
            syms = [a.get("symbol") for a in syms]
        for s in syms:
            self._load_one(str(s))
        self._loaded = True

    # ---------------- internals ----------------

    def _paths(self, symbol: str):
        base = f"{self.model_dir}/lstm_stop_loss_selector_{symbol}"
        return base + ".pt", base + ".meta.json"

    def _load_one(self, symbol: str):
        """Load TorchScript model and meta for a symbol; set up rolling buffer."""
        mpath, jpath = self._paths(symbol)
        lookback = 32  # default if meta missing
        if torch is None:
            if self.debug:
                print(f"[SEQ-EXIT] torch unavailable; {symbol} will use fallback {self.fallback_multiple}")
            self.models[symbol] = None
            self.meta[symbol] = {}
            self.buffers[symbol] = deque(maxlen=lookback)
            return

        try:
            model = torch.jit.load(mpath, map_location="cpu")
            with open(jpath, "r") as f:
                meta = json.load(f)
            lookback = int(meta.get("lookback", lookback))
            self.models[symbol] = model
            self.meta[symbol] = meta
            self.buffers[symbol] = deque(maxlen=lookback)
            # allow per-symbol override of K-list
            sls = meta.get("sl_multipliers")
            if isinstance(sls, list) and len(sls) >= 2:
                self.SL_MULTIPLIERS = [float(x) for x in sls]
            if self.debug:
                print(f"[SEQ-EXIT] Loaded {symbol}: {mpath} (lookback={lookback})")
        except Exception as e:
            self.models[symbol] = None
            self.meta[symbol] = {}
            self.buffers[symbol] = deque(maxlen=lookback)
            if self.debug:
                print(f"[SEQ-EXIT] Missing/failed model for {symbol}: {e} → fallback {self.fallback_multiple}")

    def _push_obs(self, symbol: str, atr: float, prev_row: pd.Series, side: str):
        """Build per-bar observation and append to buffer if finite."""
        ema_val = prev_row.get(self.ema_col, np.nan)
        if not np.isfinite(ema_val):
            # Fallback: any EMA_* column that's finite
            for c in prev_row.index:
                if isinstance(c, str) and c.startswith("ema_"):
                    v = prev_row.get(c, np.nan)
                    if np.isfinite(v):
                        ema_val = v
                        if self.debug:
                            print(f"[SEQ-EXIT] {symbol}: {self.ema_col} missing/NaN; using {c}")
                        break

        pt_numeric = 1.0 if side in ("long", "buy") else -1.0
        obs = np.array([
            float(atr),
            float(prev_row.get("rsi_14", np.nan)),
            float(prev_row.get("close", np.nan)),
            float(ema_val),
            float(pt_numeric),
        ], dtype=np.float32)

        if np.all(np.isfinite(obs)):
            self.buffers[symbol].append(obs)

    def _choose_multiple(
        self,
        df: Optional[pd.DataFrame],
        symbol: Optional[str],
        side: str,
        atr: float,
    ) -> float:
        """Return the ATR multiple to use right now."""
        if symbol is None or df is None or len(df) < 2:
            return float(self.fallback_multiple)

        # ensure buffer exists
        if symbol not in self.buffers:
            self.buffers[symbol] = deque(maxlen=32)

        # append previous CLOSED bar features
        prev = df.iloc[-2]
        self._push_obs(symbol, atr, prev, side)

        model = self.models.get(symbol)
        meta = self.meta.get(symbol, {})
        lookback = int(meta.get("lookback", self.buffers[symbol].maxlen or 32))

        if (model is None) or (torch is None) or (len(self.buffers[symbol]) < lookback):
            return float(self.fallback_multiple)

        # Build standardized window (z-score) as during training
        window = np.stack(list(self.buffers[symbol])[-lookback:], axis=0)  # (T, F=5)
        mu = window.mean(axis=0, keepdims=True)
        sd = window.std(axis=0, keepdims=True) + 1e-8
        X = (window - mu) / sd

        try:
            with torch.no_grad():
                xb = torch.from_numpy(X).unsqueeze(0)  # (1, T, F)
                logits = model(xb)  # shape (1, n_classes)
                idx = int(torch.argmax(logits, dim=1).item())
        except Exception as e:
            if self.debug:
                print(f"[SEQ-EXIT] {symbol} inference error: {e} → fallback {self.fallback_multiple}")
            return float(self.fallback_multiple)

        idx = max(0, min(idx, len(self.SL_MULTIPLIERS) - 1))
        return float(self.SL_MULTIPLIERS[idx])

