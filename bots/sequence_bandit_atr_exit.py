# bots/sequence_bandit_atr_exit.py
# ExitStrategy that blends the LSTM (TorchScript) prior with an online LinUCB contextual bandit.
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple, Dict, Any, Set
import os, json
import numpy as np
import pandas as pd

try:
    import torch
except Exception:
    torch = None  # type: ignore

from classes.API_Interface import round_to_tick
from bots.exit_strategies import ExitStrategy
from bots.bandit_overlay import LinUCB


@dataclass
class SequenceBanditATRExit(ExitStrategy):
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

    # shared K grid (overridden by meta if present)
    SL_MULTIPLIERS = [1.0, 2.0, 3.0, 4.0]

    # runtime state
    def __post_init__(self):
        self.model_dir = self.model_dir.rstrip('/ ')
        self.ema_col = f"ema_{int(self.ema_span)}"
        self.models: Dict[str, object] = {}
        self.meta: Dict[str, dict] = {}
        self.buffers: Dict[str, deque] = {}
        self.bandits: Dict[str, LinUCB] = {}
        self.pending: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        self._since_last_save = 0

        # learning stats
        self._nz_updates: Dict[str, int] = {}
        self._last_reward: Dict[str, float] = {}

        # de-dupe for processed fills
        self._seen_trade_keys: Set[str] = set()

    def set_ema_span(self, ema_span: int):
        self.ema_col = f"ema_{int(ema_span)}"

    # --------- ExitStrategy interface ---------
    def initial_levels(
        self,
        side: str,
        entry_price: float,
        atr: float,
        tick_size: float,
        df: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        **kwargs
    ) -> Tuple[float, Optional[float]]:
        k, ctx, idx = self._choose_k_and_context(df, symbol, side, atr)
        # Always stash a pending decision when we place an entry
        if symbol:
            self.pending[symbol] = {"context": ctx, "action": int(idx)}
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
        **kwargs
    ):
        k, ctx, idx = self._choose_k_and_context(df, symbol, side, atr)

        # If we didn’t see an entry (e.g., restarted mid-trade), ensure we still have a pending context.
        if symbol and (symbol not in self.pending):
            self.pending[symbol] = {"context": ctx, "action": int(idx)}

        # trailing gate
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

    # --------- lifecycle ---------
    def ensure_models_loaded(self, env):
        if self._loaded:
            return
        syms = env.get_asset_list()
        if syms and not isinstance(syms[0], str):
            syms = [a.get("symbol") for a in syms]
        for s in syms:
            self._load_one(str(s))
        self._loaded = True

    def _paths(self, symbol: str):
        base = f"{self.model_dir}/lstm_stop_loss_selector_{symbol}"
        return base + ".pt", base + ".meta.json"

    def _bandit_path(self, symbol: str) -> str:
        return os.path.join(self.model_dir, f"bandit_linucb_{symbol}.npz")

    def _load_one(self, symbol: str):
        # model + meta + buffer
        mpath, jpath = self._paths(symbol)
        lookback = 32
        if torch is None:
            self.models[symbol] = None
            self.meta[symbol] = {}
            self.buffers[symbol] = deque(maxlen=lookback)
        else:
            try:
                model = torch.jit.load(mpath, map_location="cpu")
                with open(jpath, "r") as f:
                    meta = json.load(f)
                lookback = int(meta.get("lookback", lookback))
                self.models[symbol] = model
                self.meta[symbol] = meta
                self.buffers[symbol] = deque(maxlen=lookback)
                sls = meta.get("sl_multipliers")
                if isinstance(sls, list) and len(sls) >= 2:
                    self.SL_MULTIPLIERS = [float(x) for x in sls]
                if self.debug:
                    print(f"[SEQ-BANDIT] Loaded LSTM for {symbol} (lookback={lookback})")
            except Exception as e:
                self.models[symbol] = None
                self.meta[symbol] = {}
                self.buffers[symbol] = deque(maxlen=lookback)
                if self.debug:
                    print(f"[SEQ-BANDIT] Using fallback for {symbol}: {e}")

        # bandit (dimension = n_actions + 5 raw obs)
        n_actions = len(self.SL_MULTIPLIERS)
        ctx_dim = n_actions + 5
        try:
            self.bandits[symbol] = LinUCB.load(self._bandit_path(symbol))
            if self.debug:
                print(f"[SEQ-BANDIT] Loaded bandit for {symbol}")
        except Exception:
            self.bandits[symbol] = LinUCB(d=ctx_dim, n_actions=n_actions,
                                          alpha=self.bandit_alpha, l2=self.bandit_l2)
            if self.debug:
                print(f"[SEQ-BANDIT] Initialized bandit for {symbol} (d={ctx_dim})")

    # --------- core helpers ---------
    def _push_obs(self, symbol: str, atr: float, prev_row: pd.Series, side: str):
        ema_val = prev_row.get(self.ema_col, np.nan)
        if not np.isfinite(ema_val):
            for c in prev_row.index:
                if isinstance(c, str) and c.startswith("ema_"):
                    v = prev_row.get(c, np.nan)
                    if np.isfinite(v):
                        ema_val = v
                        break
        pt = 1.0 if side in ("long", "buy") else -1.0
        obs = np.array([
            atr,
            prev_row.get("rsi_14", np.nan),
            prev_row.get("close", np.nan),
            ema_val,
            pt
        ], dtype=np.float32)
        obs = np.nan_to_num(obs, copy=False)
        if np.all(np.isfinite(obs)):
            self.buffers[symbol].append(obs)

    def _one_hot(self, n: int, i: int) -> np.ndarray:
        v = np.zeros(int(n), dtype=np.float64)
        v[int(np.clip(i, 0, n - 1))] = 1.0
        return v

    def _nearest_k_index(self, k: float) -> int:
        ks = np.asarray(self.SL_MULTIPLIERS, dtype=np.float64)
        return int(np.argmin(np.abs(ks - float(k))))

    def _pack_context(self, priors: np.ndarray, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([priors.reshape(-1), obs.reshape(-1)], axis=0).astype(np.float64)

    def _choose_k_and_context(self, df: Optional[pd.DataFrame], symbol: Optional[str], side: str, atr: float):
        # If no data, still return a fallback context so we can learn later
        if symbol is None or df is None or len(df) < 2:
            k = float(self.fallback_multiple)
            idx = self._nearest_k_index(k)
            pt = 1.0 if side in ("long", "buy") else -1.0
            close_val = float(df.iloc[-1].get("close", 0.0)) if (df is not None and len(df) > 0) else 0.0
            obs = np.array([atr, 0.0, close_val, 0.0, pt], dtype=np.float32)
            ctx = self._pack_context(self._one_hot(len(self.SL_MULTIPLIERS), idx), obs)
            return k, ctx, idx

        if symbol not in self.buffers:
            self.buffers[symbol] = deque(maxlen=32)
        prev = df.iloc[-2]
        self._push_obs(symbol, atr, prev, side)

        model = self.models.get(symbol)
        meta = self.meta.get(symbol, {})
        lookback = int(meta.get("lookback", self.buffers[symbol].maxlen or 32))

        # --- Fallback (no model / not enough history): still build priors+ctx ---
        if (model is None) or (torch is None) or (len(self.buffers[symbol]) < lookback):
            k = float(self.fallback_multiple)
            idx = self._nearest_k_index(k)
            obs = self.buffers[symbol][-1] if len(self.buffers[symbol]) > 0 else np.array(
                [atr, prev.get("rsi_14", 0.0), prev.get("close", 0.0), prev.get(self.ema_col, 0.0),
                 1.0 if side in ("long", "buy") else -1.0],
                dtype=np.float32
            )
            obs = np.nan_to_num(obs, copy=False)
            priors = self._one_hot(len(self.SL_MULTIPLIERS), idx)  # confident one-hot on chosen K
            ctx = self._pack_context(priors, obs)
            return k, ctx, idx

        # --- Model path ---
        win = np.stack(list(self.buffers[symbol])[-lookback:], axis=0)
        mu = win.mean(axis=0, keepdims=True)
        sd = win.std(axis=0, keepdims=True) + 1e-8
        X = (win - mu) / sd

        with torch.no_grad():
            xb = torch.from_numpy(X).unsqueeze(0)
            logits = model(xb).squeeze(0).detach().cpu().numpy()

        # softmax prior
        exps = np.exp(logits - np.max(logits))
        priors = exps / max(exps.sum(), 1e-12)       # shape (n_actions,)
        obs = self.buffers[symbol][-1]               # latest raw obs (5,)
        ctx = self._pack_context(priors.astype(np.float64), obs.astype(np.float64))

        # bandit decision
        b = self.bandits.get(symbol)
        if b is None:
            self.bandits[symbol] = LinUCB(d=ctx.shape[0], n_actions=len(self.SL_MULTIPLIERS),
                                          alpha=self.bandit_alpha, l2=self.bandit_l2)
            b = self.bandits[symbol]

        if b.total_updates() < int(self.warmup_updates):
            # follow prior with epsilon exploration
            if np.random.rand() < float(self.epsilon_warm):
                idx = int(np.random.randint(0, len(self.SL_MULTIPLIERS)))
            else:
                idx = int(np.argmax(priors))
        else:
            idx = b.choose_ucb(ctx)

        idx = int(np.clip(idx, 0, len(self.SL_MULTIPLIERS) - 1))
        return float(self.SL_MULTIPLIERS[idx]), ctx, idx

    # --------- online learning hook ---------
    def ingest_trade_log(self, env):
        """
        Update bandits from closed/exit fills emitted by the engine.

        We tolerate multiple schema variants:
          symbol:  "symbol" | "asset" | "ticker"
          close:   exit_tag in {"sl","tp","liquidation","exit","close"}
                   OR any of:
                       - position_after_fill == 0
                       - pos_after == 0
                       - qty_after == 0
                       - position == 0 (post)
          reward:  prefer "r_multiple" | "trade_r_multiple" | "R" (if use_r_multiple)
                   else "realized_pnl_change" | "realized_pnl" | "pnl_change" | "pnl"
                   else "net_realized_from_cash" (sign)
                   (always clipped to [-reward_clip, +reward_clip])
        """
        # 1) Fetch fills from env
        try:
            trades = env.get_orders() or []
        except Exception as e:
            if self.debug:
                print(f"[SEQ-BANDIT] ingest_trade_log: env.get_orders() failed: {e}")
            return

        if not trades:
            if self.debug:
                print("[SEQ-BANDIT] ingest_trade_log: no trades found on env")
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
            # common engine flags
            if str(t.get("order_role", "")).lower() in {"exit", "close"}:
                return True
            if str(t.get("fill_type", "")).lower() in {"close", "exit"}:
                return True
            # flat position checks (support strings/None)
            for k in ("position_after_fill", "pos_after", "qty_after", "position"):
                if k in t:
                    try:
                        return float(t[k]) == 0.0
                    except Exception:
                        pass
            return False

        def _extract_reward(t: dict) -> float:
            # 1) R-multiple (preferred if present and enabled)
            if self.use_r_multiple:
                for k in ("r_multiple", "trade_r_multiple", "R"):
                    if k in t:
                        try:
                            r = float(t[k])
                            if np.isfinite(r):
                                return float(np.clip(r, -self.reward_clip, self.reward_clip))
                        except Exception:
                            pass
            # 2) Realized P&L deltas
            for k in ("realized_pnl_change", "realized_pnl", "pnl_change", "pnl"):
                if k in t:
                    try:
                        r = float(t[k])
                        if np.isfinite(r):
                            return float(np.clip(r, -self.reward_clip, self.reward_clip))
                    except Exception:
                        pass
            # 3) Signed cumulative fallback
            for k in ("net_realized_from_cash",):
                if k in t:
                    try:
                        v = float(t[k])
                        r = 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
                        return float(np.clip(r, -self.reward_clip, self.reward_clip))
                    except Exception:
                        pass
            return 0.0

        updated_any = False
        skipped_no_ctx = 0
        skipped_not_close = 0
        skipped_zero_r = 0
        skipped_dup = 0

        for t in trades:
            if not isinstance(t, dict):
                continue

            # De-dupe
            key = _trade_key(t)
            if key in self._seen_trade_keys:
                skipped_dup += 1
                continue
            self._seen_trade_keys.add(key)

            sym = _get_symbol(t)
            if not sym:
                continue

            if not _is_close(t):
                skipped_not_close += 1
                continue

            pend = self.pending.get(sym)
            if not pend:
                skipped_no_ctx += 1
                continue

            ctx = pend.get("context")
            act = pend.get("action")
            # Clear pending regardless (avoid double use)
            self.pending.pop(sym, None)

            if ctx is None or act is None:
                continue

            r = _extract_reward(t)
            if r == 0.0:
                skipped_zero_r += 1
                continue

            # 3) Update bandit
            b = self.bandits.get(sym)
            if b is None:
                self.bandits[sym] = LinUCB(d=len(ctx), n_actions=len(self.SL_MULTIPLIERS),
                                           alpha=self.bandit_alpha, l2=self.bandit_l2)
                b = self.bandits[sym]

            b.update(np.asarray(ctx, dtype=np.float64), int(act), float(r))
            self._since_last_save += 1
            updated_any = True

            # stats
            self._nz_updates[sym] = self._nz_updates.get(sym, 0) + 1
            self._last_reward[sym] = float(r)

            # periodic persistence
            if self._since_last_save >= self.save_every:
                try:
                    b.save(self._bandit_path(sym))
                    self._since_last_save = 0
                except Exception as e:
                    if self.debug:
                        print(f"[SEQ-BANDIT] Warning: save failed for {sym}: {e}")

            if self.debug:
                tag = t.get("exit_tag", None)
                pos_after = t.get("position_after_fill", t.get("pos_after", None))
                print(f"[SEQ-BANDIT] Learn {sym}: a={act} r={r:.3f} tag={tag} pos_after={pos_after} nz_updates={self._nz_updates[sym]}")

        if self.debug and not updated_any:
            print("[SEQ-BANDIT] ingest_trade_log: scanned trades but nothing updated.",
                  f"no_ctx={skipped_no_ctx} not_close={skipped_not_close} zero_r={skipped_zero_r} dup={skipped_dup}")

    # --------- convenience ---------
    def bandit_stats(self) -> Dict[str, Any]:
        """Small helper for printing at the end of a run."""
        out = {}
        for sym, b in self.bandits.items():
            out[sym] = {
                "updates": int(self._nz_updates.get(sym, 0)),
                "last_reward": float(self._last_reward.get(sym, float('nan'))),
                "actions": len(self.SL_MULTIPLIERS),
                "alpha": float(getattr(b, "alpha", self.bandit_alpha)),
                "l2": float(getattr(b, "l2", self.bandit_l2)),
            }
        return out

