# bots/ppo_bandit_atr_exit.py
# PPO prior (SB3) + online LinUCB contextual bandit for ATR-multiple selection.

from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple, Dict, Any, Set, List
import os
import numpy as np
import pandas as pd

from classes.API_Interface import round_to_tick
from bots.exit_strategies import ExitStrategy
from bots.bandit_overlay import LinUCB

try:
    # SB3 is optional: if missing, we fall back to the configured fallback_multiple
    from stable_baselines3 import PPO
    _SB3_AVAILABLE = True
except Exception:
    PPO = None  # type: ignore
    _SB3_AVAILABLE = False

try:
    import torch  # only used to pull probs cleanly; optional
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


@dataclass
class PPOLinUCBATRExit(ExitStrategy):
    """
    ExitStrategy that blends a PPO prior (discrete class over ATR multiples) with an online
    LinUCB contextual bandit. At each bar:
      1) Build obs = [atr, rsi_14, close, ema_<span>, position_flag(+1 long / -1 short)]
      2) Get prior from PPO (action distribution or deterministic action as one-hot fallback)
      3) Context = concat(prior_probs, obs)
      4) Bandit chooses K via UCB (warmup supports epsilon exploration)
      5) Use K to set/advance a trailing ATR stop (no TP)

    Online learning:
      - Provide ingest_trade_log(env) to update the bandit on *closed trades* with rewards
        (prefers r-multiple if available; otherwise realized PnL delta).
      - Saves/loads bandit state from model_dir/bandit_linucb_<SYMBOL>.npz.

    Determinism:
      - With bandit enabled, epsilon exploration and UCB can introduce stochasticity.
        For fully deterministic inference with PPO only, use RLTrailingATRExit instead
        (or set epsilon_warm=0, bandit_alpha=0 to neuter exploration/bonus).
    """
    model_dir: str
    fallback_multiple: float = 3.0
    ema_span: int = 21
    debug: bool = False

    # bandit knobs
    bandit_alpha: float = 0.6
    bandit_l2: float = 1.0
    warmup_updates: int = 50
    epsilon_warm: float = 0.10
    save_every: int = 50
    reward_clip: float = 5.0
    use_r_multiple: bool = True

    # shared K grid (align with your trainers)
    SL_MULTIPLIERS: List[float] = None  # default below in __post_init__

    # ----- runtime state -----
    def __post_init__(self):
        self.model_dir = self.model_dir.rstrip("/ ")
        if self.SL_MULTIPLIERS is None:
            self.SL_MULTIPLIERS = [1.0, 2.0, 3.0, 4.0]

        self.ema_col = f"ema_{int(self.ema_span)}"
        self.models: Dict[str, Any] = {}   # PPO policies per symbol (or None if unavailable)
        self.bandits: Dict[str, LinUCB] = {}
        self.pending: Dict[str, Dict[str, Any]] = {}  # symbol -> {context, action_index}
        self._since_last_save: int = 0

        # de-dupe processed trade logs
        self._seen_trade_keys: Set[str] = set()

        # basic stats
        self._nz_updates: Dict[str, int] = {}
        self._last_reward: Dict[str, float] = {}

        # tiny rolling buffer reserved for parity with LSTM APIs if desired
        self.buffers: Dict[str, deque] = {}

    # ---------- utilities ----------
    def set_ema_span(self, ema_span: int):
        self.ema_col = f"ema_{int(ema_span)}"

    def _ppo_model_path(self, symbol: str) -> str:
        # Matches your PPO saver naming
        return os.path.join(
            self.model_dir,
            f"ppo_stop_loss_selector_rl_stop_loss_training_{symbol}.zip",
        )

    def _bandit_path(self, symbol: str) -> str:
        return os.path.join(self.model_dir, f"bandit_linucb_{symbol}.npz")

    def ensure_models_loaded(self, env):
        """Lazy-load PPO models and bandits for all symbols at first use."""
        syms = env.get_asset_list()
        if syms and not isinstance(syms[0], str):
            syms = [a.get("symbol") for a in syms]
        for s in syms:
            self._load_symbol(str(s))

    def _load_symbol(self, symbol: str):
        # PPO model (optional)
        if symbol not in self.models:
            if _SB3_AVAILABLE:
                try:
                    path = self._ppo_model_path(symbol)
                    self.models[symbol] = PPO.load(path, device="cpu")
                    if self.debug:
                        print(f"[PPO-BANDIT] Loaded PPO for {symbol}: {path}")
                except Exception as e:
                    self.models[symbol] = None
                    if self.debug:
                        print(f"[PPO-BANDIT] PPO unavailable for {symbol}: {e}")
            else:
                self.models[symbol] = None
                if self.debug:
                    print(f"[PPO-BANDIT] SB3 not available; using fallback for {symbol}")

        # LinUCB (load existing or init fresh)
        if symbol not in self.bandits:
            n_actions = len(self.SL_MULTIPLIERS)
            ctx_dim = n_actions + 5  # prior_probs (K) + obs (5 features)
            try:
                self.bandits[symbol] = LinUCB.load(self._bandit_path(symbol))
                if self.debug:
                    print(f"[PPO-BANDIT] Loaded bandit for {symbol}")
            except Exception:
                self.bandits[symbol] = LinUCB(d=ctx_dim, n_actions=n_actions,
                                              alpha=self.bandit_alpha, l2=self.bandit_l2)
                if self.debug:
                    print(f"[PPO-BANDIT] Init bandit for {symbol} (d={ctx_dim})")
        if symbol not in self.buffers:
            self.buffers[symbol] = deque(maxlen=1)  # reserved

    def _one_hot(self, n: int, i: int) -> np.ndarray:
        v = np.zeros(int(n), dtype=np.float64)
        v[int(np.clip(i, 0, n - 1))] = 1.0
        return v

    def _pack_context(self, priors: np.ndarray, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([priors.reshape(-1), obs.reshape(-1)], axis=0).astype(np.float64)

    def _nearest_k_index(self, k: float) -> int:
        ks = np.asarray(self.SL_MULTIPLIERS, dtype=np.float64)
        return int(np.argmin(np.abs(ks - float(k))))

    def _obs_from_prev(self, atr: float, prev: pd.Series, side: str) -> np.ndarray:
        ema_val = prev.get(self.ema_col, np.nan)
        if not np.isfinite(ema_val):
            # fallback to any available EMA
            for c in prev.index:
                if isinstance(c, str) and c.startswith("ema_"):
                    v = prev.get(c, np.nan)
                    if np.isfinite(v):
                        ema_val = v
                        break
        pt = 1.0 if side in ("long", "buy") else -1.0
        obs = np.array([
            float(atr),
            float(prev.get("rsi_14", np.nan)),
            float(prev.get("close", np.nan)),
            float(ema_val),
            float(pt),
        ], dtype=np.float32)
        return np.nan_to_num(obs)

    def _prob_prior_from_ppo(self, model, obs: np.ndarray) -> Optional[np.ndarray]:
        """Try to get action probabilities (soft prior). One-hot fallback if unavailable."""
        if model is None:
            return None
        try:
            if _TORCH_OK:
                # SB3: get categorical distribution
                obs_tensor = model.policy.obs_to_tensor(obs)[0]
                dist = model.policy.get_distribution(obs_tensor)
                # Categorical → .distribution.probs (shape [n_actions])
                probs = dist.distribution.probs.detach().cpu().numpy().astype(np.float64).reshape(-1)
                # safety
                if probs.size > 0 and np.all(np.isfinite(probs)):
                    probs = probs / max(probs.sum(), 1e-12)
                    return probs
            # Fallback: deterministic action → one-hot
            act, _ = model.predict(obs, deterministic=True)
            idx = int(np.asarray(act).flatten()[0])
            return self._one_hot(len(self.SL_MULTIPLIERS), idx)
        except Exception:
            # Last fallback: deterministic one-hot via predict
            try:
                act, _ = model.predict(obs, deterministic=True)
                idx = int(np.asarray(act).flatten()[0])
                return self._one_hot(len(self.SL_MULTIPLIERS), idx)
            except Exception:
                return None

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
        k, ctx, idx = self._choose_k_and_context(df, symbol, side, atr)
        if symbol:
            self.pending[symbol] = {"context": ctx, "action": int(idx)}
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
        # trailing only after price moves favorably beyond entry
        if side in ("buy", "long"):
            if current_price <= entry_price:
                return None
        else:
            if current_price >= entry_price:
                return None

        k, ctx, idx = self._choose_k_and_context(df, symbol, side, atr)

        # ensure we have a pending decision, in case we joined mid-trade
        if symbol and (symbol not in self.pending):
            self.pending[symbol] = {"context": ctx, "action": int(idx)}

        if side in ("buy", "long"):
            candidate = current_price - k * atr
            if candidate > stop_loss:
                return round_to_tick(candidate, tick_size)
        else:
            candidate = current_price + k * atr
            if candidate < stop_loss:
                return round_to_tick(candidate, tick_size)
        return None

    # ---------- core selection ----------
    def _choose_k_and_context(self, df: Optional[pd.DataFrame], symbol: Optional[str], side: str, atr: float):
        if symbol is None:
            k = float(self.fallback_multiple)
            idx = self._nearest_k_index(k)
            obs = np.array([atr, 0.0, 0.0, 0.0, 1.0 if side in ("long", "buy") else -1.0], dtype=np.float32)
            ctx = self._pack_context(self._one_hot(len(self.SL_MULTIPLIERS), idx), obs)
            return k, ctx, idx

        # make sure symbol state exists
        self._load_symbol(symbol)

        # get prev-closed bar features
        if df is None or len(df) < 2:
            k = float(self.fallback_multiple)
            idx = self._nearest_k_index(k)
            obs = np.array([atr, 0.0, 0.0, 0.0, 1.0 if side in ("long", "buy") else -1.0], dtype=np.float32)
            ctx = self._pack_context(self._one_hot(len(self.SL_MULTIPLIERS), idx), obs)
            return k, ctx, idx

        prev = df.iloc[-2]
        obs = self._obs_from_prev(atr, prev, side)

        # PPO prior (soft probs if available)
        priors = None
        model = self.models.get(symbol)
        if model is not None:
            priors = self._prob_prior_from_ppo(model, obs)

        # If PPO unavailable, build a neutral one-hot on fallback K
        if priors is None:
            k = float(self.fallback_multiple)
            idx = self._nearest_k_index(k)
            priors = self._one_hot(len(self.SL_MULTIPLIERS), idx)

        ctx = self._pack_context(np.asarray(priors, dtype=np.float64), obs.astype(np.float64))

        # choose via LinUCB
        b = self.bandits.get(symbol)
        if b is None:
            self.bandits[symbol] = LinUCB(d=len(ctx), n_actions=len(self.SL_MULTIPLIERS),
                                          alpha=self.bandit_alpha, l2=self.bandit_l2)
            b = self.bandits[symbol]

        # Warmup: follow prior with epsilon exploration
        if b.total_updates() < int(self.warmup_updates):
            if np.random.rand() < float(self.epsilon_warm):
                idx = int(np.random.randint(0, len(self.SL_MULTIPLIERS)))
            else:
                idx = int(np.argmax(priors))
        else:
            idx = b.choose_ucb(ctx)

        idx = int(np.clip(idx, 0, len(self.SL_MULTIPLIERS) - 1))
        return float(self.SL_MULTIPLIERS[idx]), ctx, idx

    # ---------- online learning (called by BaseStrategyBot wrapper) ----------
    def ingest_trade_log(self, env):
        """Update bandits from closed-trade fills."""
        try:
            trades = env.get_orders() or []
        except Exception as e:
            if self.debug:
                print(f"[PPO-BANDIT] ingest_trade_log: env.get_orders() failed: {e}")
            return

        if not trades:
            return

        def _trade_key(t: dict) -> str:
            # Prefer a native id if present
            for k in ("id", "order_id", "trade_id", "fill_id"):
                v = t.get(k)
                if v is not None:
                    return f"{k}:{v}"
            return "sym={}|ts={}|qty={}|px={}|tag={}".format(
                t.get("symbol") or t.get("asset") or t.get("ticker"),
                t.get("timestamp") or t.get("time") or t.get("date"),
                t.get("qty") or t.get("filled_qty") or t.get("quantity"),
                t.get("price") or t.get("fill_price") or t.get("avg_price"),
                t.get("exit_tag"),
            )

        def _get_symbol(t: dict) -> Optional[str]:
            for k in ("symbol", "asset", "ticker"):
                v = t.get(k)
                if isinstance(v, str) and v:
                    return v
            return None

        def _is_close(t: dict) -> bool:
            tag = str(t.get("exit_tag", "")).lower()
            if tag in {"sl", "tp", "liquidation", "exit", "close", "stop", "target"}:
                return True
            # other schema hints
            if str(t.get("order_role", "")).lower() in {"exit", "close"}:
                return True
            if str(t.get("fill_type", "")).lower() in {"close", "exit"}:
                return True
            for k in ("position_after_fill", "pos_after", "qty_after", "position"):
                if k in t:
                    try:
                        return float(t[k]) == 0.0
                    except Exception:
                        pass
            return False

        def _extract_reward(t: dict) -> float:
            if self.use_r_multiple:
                for k in ("r_multiple", "trade_r_multiple", "R"):
                    if k in t:
                        try:
                            r = float(t[k])
                            if np.isfinite(r):
                                return float(np.clip(r, -self.reward_clip, self.reward_clip))
                        except Exception:
                            pass
            for k in ("realized_pnl_change", "realized_pnl", "pnl_change", "pnl"):
                if k in t:
                    try:
                        r = float(t[k])
                        if np.isfinite(r):
                            return float(np.clip(r, -self.reward_clip, self.reward_clip))
                    except Exception:
                        pass
            for k in ("net_realized_from_cash",):
                if k in t:
                    try:
                        v = float(t[k]); r = 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
                        return float(np.clip(r, -self.reward_clip, self.reward_clip))
                    except Exception:
                        pass
            return 0.0

        for t in trades:
            if not isinstance(t, dict):
                continue
            key = _trade_key(t)
            if key in self._seen_trade_keys:
                continue
            self._seen_trade_keys.add(key)

            sym = _get_symbol(t)
            if not sym:
                continue
            if not _is_close(t):
                continue

            pend = self.pending.get(sym)
            if not pend:
                continue  # no context/action captured at entry/update

            ctx = pend.get("context"); act = pend.get("action")
            # clear pending regardless
            self.pending.pop(sym, None)
            if ctx is None or act is None:
                continue

            r = _extract_reward(t)
            if r == 0.0:
                continue

            b = self.bandits.get(sym)
            if b is None:
                n_actions = len(self.SL_MULTIPLIERS)
                self.bandits[sym] = LinUCB(d=len(ctx), n_actions=n_actions,
                                           alpha=self.bandit_alpha, l2=self.bandit_l2)
                b = self.bandits[sym]

            b.update(np.asarray(ctx, dtype=np.float64), int(act), float(r))
            self._since_last_save += 1
            self._nz_updates[sym] = self._nz_updates.get(sym, 0) + 1
            self._last_reward[sym] = float(r)

            if self._since_last_save >= int(self.save_every):
                try:
                    b.save(self._bandit_path(sym))
                    self._since_last_save = 0
                except Exception as e:
                    if self.debug:
                        print(f"[PPO-BANDIT] Warning: save failed for {sym}: {e}")

    # convenience
    def bandit_stats(self) -> Dict[str, Any]:
        out = {}
        for sym, b in self.bandits.items():
            out[sym] = {
                "updates": int(self._nz_updates.get(sym, 0)),
                "last_reward": float(self._last_reward.get(sym, float("nan"))),
                "actions": len(self.SL_MULTIPLIERS),
                "alpha": float(getattr(b, "alpha", self.bandit_alpha)),
                "l2": float(getattr(b, "l2", self.bandit_l2)),
            }
        return out

