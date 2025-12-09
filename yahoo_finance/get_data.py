#!/usr/bin/env python3
"""
Yahoo Finance Futures Downloader.
Keeps DataFrame format exactly as yfinance.download() returns it.
"""

import os
import time
import argparse
import pandas as pd
import yfinance as yf

FUTURES_CONTRACTS = {
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'Crude_Oil': 'CL=F',
    'Soybeans': 'ZS=F',
    'Sugar': 'SB=F',
    'US_Treasury_Bonds': 'ZB=F',
    'Euro': '6E=F',
    'British_Pound': '6B=F',
    'Live_Cattle': 'LE=F',
}

INTERVAL_TO_DIR = {
    "1d":  "1Day_timeframe",
    "1h":  "1Hour_timeframe",
    "1wk": "1Week_timeframe",
    "1mo": "1Month_timeframe",
}

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _compute_window(years=None, date_from=None, date_to=None):
    if years is not None:
        end = pd.Timestamp.today()
        start = end - pd.DateOffset(years=years)
        label = f"{years}yr"
        return start, end, label
    if date_from and date_to:
        start = pd.Timestamp(date_from)
        end = pd.Timestamp(date_to)
        if start >= end:
            raise ValueError("--from must be before --to")
        return start, end, f"{start.year}-{end.year}"
    raise ValueError("Must pass either --years or both --from and --to")

def download_range(ticker, start, end, interval="1d", auto_adjust=True):
    # Download single ticker to avoid MultiIndex
    return yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, help="Number of years back from today.")
    ap.add_argument("--from", dest="date_from", help="Start date YYYY-MM-DD")
    ap.add_argument("--to", dest="date_to", help="End date YYYY-MM-DD")
    ap.add_argument("--interval", default="1d", choices=INTERVAL_TO_DIR.keys())
    ap.add_argument("--data-root", default="yahoo_finance/data")
    ap.add_argument("--category", default="Futures")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-adjust", action="store_true")
    args = ap.parse_args()

    start, end, label = _compute_window(args.years, args.date_from, args.date_to)
    interval_dir = INTERVAL_TO_DIR[args.interval]

    for name, ticker in FUTURES_CONTRACTS.items():
        safe_name = name.replace(" ", "_")
        out_dir = os.path.join(args.data_root, args.category, name, interval_dir)
        _ensure_dir(out_dir)

        out_fp = os.path.join(out_dir, f"{safe_name.lower()}_{label}.csv")
        if os.path.exists(out_fp) and not args.overwrite:
            print(f"⏭️  {name}: already exists, skipping.")
            continue

        print(f"↓  {name} [{ticker}] {args.interval} {start.date()} → {end.date()} ... ", end="", flush=True)
        try:
            df = download_range(ticker, start, end, args.interval, auto_adjust=(not args.no_adjust))
            tmp = out_fp + ".tmp"
            df.to_csv(tmp)
            os.replace(tmp, out_fp)
            print(f"saved {len(df):,} rows → {out_fp}")
        except Exception as e:
            print(f"FAILED: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()

