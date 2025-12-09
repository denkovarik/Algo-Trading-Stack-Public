# Usage:
#   PYTHONPATH=. python3 bots/generate_ML_SL_Training_data.py [flags]
#
# Default behavior (same as before, but explicit)
# PYTHONPATH=. python3 bots/generate_ML_SL_Training_data.py
#
# Custom config + output directory
# PYTHONPATH=. python3 bots/generate_ML_SL_Training_data.py \
#  --config bots/configs/ml_sl_config.yaml \
#  --output-dir bots/data/yahoo_finance/training_data

# Different stop-loss multiples
# PYTHONPATH=. python3 bots/generate_ML_SL_Training_data.py --sl 1.0,1.5,2.0,3.0,4.0
#
# Only certain symbols
# PYTHONPATH=. python3 bots/generate_ML_SL_Training_data.py --symbols GC=F,CL=F,6B=F
#
# Scan a tighter range
# PYTHONPATH=. python3 bots/generate_ML_SL_Training_data.py --warmup 200 --tail-buffer 50

# Only long side, stricter eligibility, different coin-flip probability
# PYTHONPATH=. python3 bots/generate_ML_SL_Training_data.py | 
#   --long-only --p-buy 0.6 --strict-eligibility


from __future__ import annotations
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from classes.Backtester_Engine import BacktesterEngine
from classes.Trading_Environment import TradingEnvironment

from bots.coin_flip_bot.coin_flip_bot import CoinFlipBot


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate ML training data for stop-loss selection (ATR-multiple search)."
    )
    ap.add_argument(
        "--config",
        default="bots/configs/ml_sl_config.yaml",
        help="Backtester config path (default: bots/configs/ml_sl_config.yaml)",
    )
    ap.add_argument(
        "--output-dir",
        default="bots/data/yahoo_finance/training_data/",
        help="Directory to write CSVs (default: bots/data/yahoo_finance/training_data/)",
    )
    ap.add_argument(
        "--sl",
        default="1.0,2.0,3.0,4.0",
        help="Comma-separated ATR SL multiples to test (default: 1.0,2.0,3.0,4.0)",
    )
    ap.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbol list to include (e.g., GC=F,CL=F). "
             "If omitted, uses all futures symbols from the config.",
    )
    ap.add_argument(
        "--min-bars",
        type=int,
        default=200,
        help="Minimum bars required to process a symbol (default: 200)",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Bars to skip from the start (indicator warmup) (default: 100)",
    )
    ap.add_argument(
        "--tail-buffer",
        type=int,
        default=100,
        help="Bars to skip at the end (default: 100)",
    )
    side = ap.add_mutually_exclusive_group()
    side.add_argument("--long-only", action="store_true", help="Only evaluate long side")
    side.add_argument("--short-only", action="store_true", help="Only evaluate short side")

    # CoinFlip bot knobs
    ap.add_argument("--p-buy", type=float, default=0.5, help="CoinFlipBot p_buy (default: 0.5)")
    ap.add_argument(
        "--strict-eligibility",
        action="store_true",
        help="Use strict eligibility (by default we use loose_eligibility=True)",
    )

    return ap.parse_args(argv)


def extract_features(df: pd.DataFrame, idx: int, position_type: str) -> Optional[Dict[str, Any]]:
    """Pull the same features you had before; skip row if any missing."""
    bar = df.iloc[idx]
    try:
        return {
            "atr": bar["atr_14"],
            "rsi": bar["rsi_14"],
            "close": bar["close"],
            "ema_21": bar["ema_21"],
            "position_type": 1 if position_type == "long" else -1,
        }
    except KeyError as e:
        print(f"[SKIP ROW] Missing column: {e} at index {idx}")
        return None


def simulate_all_multipliers(
    df: pd.DataFrame, entry_idx: int, position_type: str, sl_multipliers: List[float]
) -> Optional[Dict[float, float]]:
    """
    Simulate forward once from entry_idx for a given side,
    checking all SL multiples in parallel.
    Returns: dict {sl_mult: pnl} for each multiple.
    """
    if entry_idx >= len(df) or entry_idx <= 0:
        return None

    entry_price = df["open"].iloc[entry_idx]
    atr = df["atr_14"].iloc[entry_idx]
    if not np.isfinite(entry_price) or not np.isfinite(atr):
        return None

    # Set stop levels for each multiplier
    stops = {}
    for m in sl_multipliers:
        if position_type == "long":
            stops[m] = entry_price - m * atr
        else:
            stops[m] = entry_price + m * atr

    active = set(sl_multipliers)
    results = {m: None for m in sl_multipliers}
    exit_prices: Dict[float, float] = {}

    # Walk forward until all stops are hit or data ends
    last_i = entry_idx  # for safety on empty loop
    for i in range(entry_idx + 1, len(df)):
        last_i = i
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]

        remove_list = []
        for m in active:
            sl_price = stops[m]
            if position_type == "long":
                if low <= sl_price:  # SL hit
                    exit_prices[m] = sl_price
                    remove_list.append(m)
            else:
                if high >= sl_price:  # SL hit
                    exit_prices[m] = sl_price
                    remove_list.append(m)
        for m in remove_list:
            active.remove(m)
        if not active:
            break

    # Any still active exit at last close
    last_close = df["close"].iloc[min(last_i, len(df) - 1)]
    for m in active:
        exit_prices[m] = last_close

    # Compute PnL per contract for each multiplier (price difference only; same as your original)
    for m in sl_multipliers:
        if position_type == "long":
            results[m] = exit_prices[m] - entry_price
        else:
            results[m] = entry_price - exit_prices[m]

    return results


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Parse SL multiples
    try:
        sl_multipliers = [float(x.strip()) for x in args.sl.split(",") if x.strip()]
        if not sl_multipliers:
            raise ValueError
    except Exception:
        print(f"[ERROR] Invalid --sl value: {args.sl}")
        return 2

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize engine/env
    engine = BacktesterEngine(config_path=args.config)
    env = TradingEnvironment()
    env.set_api(engine)

    # Bot (respect user choice of eligibility strictness)
    bot = CoinFlipBot(
        p_buy=args.p_buy, 
        loose_eligibility=(not args.strict_eligibility),
        enable_online_learning=False
    )

    # Symbol selection
    include_symbols = set()
    if args.symbols:
        include_symbols = {s.strip() for s in args.symbols.split(",") if s.strip()}

    sides: List[str]
    if args.long_only:
        sides = ["long"]
    elif args.short_only:
        sides = ["short"]
    else:
        sides = ["long", "short"]

    # Process each asset in config
    for asset in engine.assets:
        symbol = asset.get("symbol")
        asset_type = asset.get("type", "unknown")
        filename = asset.get("file", "")

        if asset_type != "futures":
            print(f"[SKIP] {symbol}: not futures")
            continue
        if include_symbols and (symbol not in include_symbols):
            print(f"[SKIP] {symbol}: not in --symbols")
            continue

        df = engine.get_asset_data(symbol)
        if df is None or len(df) < args.min_bars:
            print(f"[SKIP] {symbol}: not enough data or missing file: {filename}")
            continue

        required_cols = ["atr_14", "rsi_14", "close", "ema_21", "high", "low", "open"]
        if not all(col in df.columns for col in required_cols):
            print(f"[SKIP] {symbol}: missing required cols")
            continue

        data_rows = []
        start_idx = max(args.warmup, 1)
        end_idx = max(0, len(df) - args.tail_buffer)

        print(f"Processing {symbol} ({filename})... range [{start_idx}, {end_idx})")

        for idx in tqdm(range(start_idx, end_idx), ascii=True, desc=f"{symbol}"):
            idx_prev = idx - 1
            if not bot.wants_entry_at(env, symbol, df, idx, idx_prev):
                continue

            for position_type in sides:
                pnl_map = simulate_all_multipliers(df, idx, position_type, sl_multipliers)
                if not pnl_map:
                    continue

                # Pick the multiplier with highest PnL
                best_mult = max(pnl_map, key=lambda m: pnl_map[m])
                best_pnl = pnl_map[best_mult]

                features = extract_features(df, idx, position_type)
                if features is None:
                    continue
                features["best_sl_multiplier"] = best_mult
                features["best_pnl"] = best_pnl
                data_rows.append(features)

        training_df = pd.DataFrame(data_rows)
        output_path = output_dir / f"rl_stop_loss_training_{symbol}.csv"
        training_df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

