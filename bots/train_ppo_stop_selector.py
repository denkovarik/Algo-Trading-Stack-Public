"""
Train PPO to select the ATR multiple (discrete action) from offline training CSVs.

Usage examples:
  # Train PPO models for all symbols found in input_dir (default SL list 1,2,3,4)
  PYTHONPATH=. python3 bots/train_ppo_stop_selector.py \
      --input_dir bots/data/yahoo_finance/training_data \
      --output_dir bots/models/PPO_Trailing_Stop_Loss \
      --total_timesteps 300000

  # Different ATR-multiple set (must match RLTrailingATRExit.SL_MULTIPLIERS order if you change it)
  PYTHONPATH=. python3 bots/train_ppo_stop_selector.py \
      --input_dir bots/data/yahoo_finance/training_data \
      --output_dir bots/models/PPO_Trailing_Stop_Loss \
      --sl 1.0,2.0,3.0,4.0 \
      --total_timesteps 500000

  # GPU/CPU selection (defaults to auto)
  PYTHONPATH=. python3 bots/train_ppo_stop_selector.py --device cpu
"""

from __future__ import annotations
import os, re, argparse
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

# -------- Data mapping --------
REQ_FEATURES = ["atr", "rsi_14", "close", "ema_21", "position_type"]
LABEL_COL = "best_sl_multiplier"
ALIASES = {
    "atr": ["atr", "atr_14", "ATR", "ATR_14"],
    "rsi_14": ["rsi_14", "RSI_14", "rsi", "RSI"],
    "close": ["close", "Close", "CLOSE"],
    "ema_21": ["ema_21", "EMA_21", "ema", "EMA"],
    "position_type": ["position_type", "pos_type", "position_flag", "posflag"],
    LABEL_COL: [LABEL_COL, "label", "y", "best_k", "best_multiple"],
}

def _first_present(df: pd.DataFrame, opts: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for k in opts:
        if k in df.columns: return k
        if k.lower() in lower: return lower[k.lower()]
    return None

def load_symbol_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists(): return None
    df = pd.read_csv(csv_path)
    rename_map: Dict[str, str] = {}
    for canon, opts in ALIASES.items():
        found = _first_present(df, opts)
        if found: rename_map[found] = canon
    if rename_map: df = df.rename(columns=rename_map)
    need = REQ_FEATURES + [LABEL_COL]
    if any(c not in df.columns for c in need):
        print(f"[SKIP] {csv_path.name}: missing columns")
        return None
    df = df[need].dropna().reset_index(drop=True)
    if len(df) < 100:
        print(f"[SKIP] {csv_path.name}: too few rows ({len(df)})")
        return None
    return df

def discover_symbol(csv_file: Path) -> str:
    m = re.match(r"^rl_stop_loss_training_(.+)\.csv$", csv_file.name)
    return m.group(1) if m else csv_file.stem.replace("rl_stop_loss_training_", "")

# -------- Env --------
class StopLossDatasetEnv(gym.Env):
    """Offline bandit env over dataset rows (shuffles each reset)."""
    metadata = {"render_modes": []}
    def __init__(self, data: pd.DataFrame, sl_values: List[float], seed: int = 42):
        super().__init__()
        self.sl_values = list(sl_values)
        inv = {float(k): i for i, k in enumerate(self.sl_values)}
        X = data[REQ_FEATURES].to_numpy(dtype=np.float32)
        y_raw = data[LABEL_COL].astype(float).to_numpy()
        y = np.array([inv[val] for val in y_raw], dtype=np.int64)

        self._X = X
        self._y = y
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(X.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.sl_values))

        self.rng = np.random.default_rng(seed)
        self._order = np.arange(len(self._X))
        self._cursor = 0

    def _reshuffle(self):
        self.rng.shuffle(self._order)
        self._cursor = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reshuffle()
        idx = self._order[self._cursor]
        self._cursor += 1
        return self._X[idx].copy(), {}

    def step(self, action: int):
        idx = self._order[self._cursor - 1]
        reward = 1.0 if int(action) == int(self._y[idx]) else 0.0
        terminated = False
        truncated = False
        if self._cursor >= len(self._X):
            terminated = True
            obs = np.zeros_like(self._X[0], dtype=np.float32)
        else:
            obs = self._X[self._order[self._cursor]].copy()
            self._cursor += 1
        return obs, float(reward), terminated, truncated, {}

# -------- Training --------
def make_vec_envs(df: pd.DataFrame, sl_values: List[float], num_envs: int, seed: int, use_subproc: bool):
    def factory(rank: int):
        return lambda: StopLossDatasetEnv(df, sl_values, seed=seed + rank)
    env_fns = [factory(i) for i in range(num_envs)]
    Vec = SubprocVecEnv if use_subproc and num_envs > 1 else DummyVecEnv
    return Vec(env_fns)

def train_one_symbol(
    csv_path: Path,
    out_dir: Path,
    sl_values: List[float],
    total_timesteps: int,
    lr: float,
    batch_size: int,
    n_steps: int,
    n_epochs: int,
    num_envs: int,
    seed: int,
    device: str,
    net_arch: List[int],
    eval_freq: int,
    use_subproc: bool,
) -> bool:
    df = load_symbol_csv(csv_path)
    if df is None: return False
    symbol = discover_symbol(csv_path)
    print(f"\n[TRAIN] {symbol} — rows: {len(df)}, actions: {sl_values}, envs: {num_envs}")

    vec = make_vec_envs(df, sl_values, num_envs=num_envs, seed=seed, use_subproc=use_subproc)
    vec = VecMonitor(vec)

    policy_kwargs = dict(net_arch=net_arch)
    model = PPO(
        policy="MlpPolicy",
        env=vec,
        learning_rate=lr,
        n_steps=n_steps,              # per-env rollout length → total batch = n_steps * num_envs
        batch_size=batch_size,        # SGD minibatch size
        n_epochs=n_epochs,            # fewer epochs when batch is big
        gamma=0.0,                    # bandit
        gae_lambda=0.0,               # bandit
        clip_range=0.2,
        verbose=1,
        seed=seed,
        device=device,
        policy_kwargs=policy_kwargs,
    )

    cb = None
    if eval_freq > 0:
        cb = EvalCallback(vec, eval_freq=eval_freq, n_eval_episodes=1, deterministic=True, verbose=0)

    model.learn(total_timesteps=total_timesteps, callback=cb)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ppo_stop_loss_selector_rl_stop_loss_training_{symbol}.zip"
    model.save(str(out_path))
    print(f"[SAVED] {out_path}")
    return True

def parse_args():
    p = argparse.ArgumentParser(description="Fast PPO trainer for SL multiple selection.")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--sl", default="1.0,2.0,3.0,4.0")
    p.add_argument("--total_timesteps", type=int, default=200_000)      # ↓ default for speed
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=4096)              # ↑ big batch
    p.add_argument("--n_steps", type=int, default=1024)                 # ↑ longer rollout per env
    p.add_argument("--n_epochs", type=int, default=6)                   # ↓ fewer epochs
    p.add_argument("--num_envs", type=int, default=8)                   # ↑ parallel envs
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")                          # cpu | cuda | auto
    p.add_argument("--net_arch", default="64", help="Comma sizes, e.g. 64 or 128,64")
    p.add_argument("--eval_freq", type=int, default=0)                  # 0 disables eval
    p.add_argument("--use_subproc", action="store_true", help="Use SubprocVecEnv for CPU throughput")
    return p.parse_args()

def main():
    args = parse_args()
    set_random_seed(args.seed)

    sl_values = [float(x.strip()) for x in args.sl.split(",") if x.strip()]
    if len(sl_values) < 2:
        raise SystemExit(f"--sl must have >=2 values, got {sl_values}")

    net_arch = [int(x.strip()) for x in args.net_arch.split(",") if x.strip()]

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    csvs = sorted(in_dir.glob("rl_stop_loss_training_*.csv"))
    if not csvs:
        raise SystemExit(f"No training CSVs in {in_dir}")

    ok = 0
    for csv in csvs:
        try:
            if train_one_symbol(
                csv, out_dir, sl_values,
                args.total_timesteps, args.learning_rate,
                args.batch_size, args.n_steps, args.n_epochs,
                args.num_envs, args.seed, args.device,
                net_arch, args.eval_freq, args.use_subproc
            ):
                ok += 1
        except Exception as e:
            print(f"[ERROR] {csv.name}: {e}")

    print(f"\n✅ Done: {ok}/{len(csvs)} models saved to {out_dir}")

if __name__ == "__main__":
    main()

