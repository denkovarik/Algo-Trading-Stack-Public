> ⚠️ **Disclaimer**  
> This repository is provided **for educational and research purposes only**.  
> It is **not financial advice**, and nothing in this project should be interpreted as a recommendation to trade, invest, or risk capital in any way.  
> The codebase is still in **testing and development** — functionality may be incomplete, unstable, or subject to change without notice.  
> Use entirely at your own risk: the authors and contributors assume **no responsibility or liability** for any losses, damages, or adverse outcomes arising from use of this software.


# Algo-Trading-Stack

*A modular research and backtesting framework for algorithmic trading with reinforcement learning (RL) and contextual bandit exits.*

---

## 🌐 Website

For full documentation, guides, and an overview of the project, visit the website:  
👉 [https://algo-trading-stack.vercel.app](https://algo-trading-stack.vercel.app)

---

## 🚀 Overview

**Algo-Trading-Stack** is a full‑featured trading research platform built for:

* **Backtesting**: realistic fills (tick rounding, slippage, commissions/fees), leverage, and margin/liquidation checks.
* **Trading environment**: pluggable indicator pipeline (EMA, RSI, ATR, Bollinger Bands), safe no‑lookahead execution.
* **Bots**:

  * *Coin Flip Bot* (baseline / experimental)
  * *Trend Following Bot* (EMA trend + ATR breakout)
* **Exit strategies**:

  * Traditional (*Trailing ATR*, *Fixed Ratio*)
  * ML‑driven (*PPO Trailing ATR*, *Sequence LSTM Exit*)
  * Hybrid (*PPO/LSTM + LinUCB Bandit*)
* **UI layer** (PyQt + PyQtGraph):

  * Interactive candlestick charting with indicators
  * Order entry & position panels
  * Equity curve and statistics dialogs
* **Research tooling**:

  * Training data generation for stop‑loss selection
  * PPO/LSTM model trainers
  * Online learning integration (LinUCB overlays)

---

## 🏗 Architecture

```
+===========================================================================================+
|                                      RESEARCH PLATFORM                                    |
|                                (extensible, componentized core)                           |
+===========================================================================================+

 Data & Brokers                        Core Runtime                   Strategies / Exits
 --------------                        ----------------               -------------------
 • Yahoo Finance CSVs       -->        • BacktesterEngine      <---   • Bots (CoinFlip, TrendFollowing)
 • TradeStation (live) <-- Adapter(s)  • TradingEnvironment           • ExitStrategy plug-ins
                                       • Indicator Registry           • Trailing ATR / Fixed R
                                       • Event-driven phases          • PPO/LSTM RL exits
                                                                      • Hybrid LinUCB exits

 Runners & UI
 ------------
 • GUI backtest (PyQt)
 • Headless backtest (tqdm)
 • Live trading (TradeStation)*

 Outputs
 -------
 • Equity curves & stats • Orders & trade logs • Charts & indicators
```

---

## ⚡ Features

* **Phase-aware simulation**: OPEN snapshot (no lookahead), INTRABAR protective orders, CLOSE equity/indicators.
* **Margin enforcement**: intrabar worst‑case valuation; automatic liquidation if below maintenance margin.
* **Per‑asset stats**: win/loss counts, expectancy, profit factor, commissions/fees, max drawdown.
* **Online learners**: LinUCB bandit overlays blended with PPO/LSTM priors.
* **UI**:

  * Candlestick chart with indicators (BB, EMA, RSI, ATR)
  * Order markers + SL/TP lines
  * Live/scroll modes, zoom/pan
  * Position and order entry panels
  * Statistics dialog and equity curve plots

---

## 🖥 Setup

### Clone Repo
```bash
git clone https://github.com/denkovarik/Algo-Trading-Stack-Public.git
```

### Install Dependencies
```bash
cd Algo-Trading-Stack-Public
```
```bash
./setup/install.sh
```

### Activate Python Virtual Environment
```bash
source venv/bin/activate
```

### Download Data From Yahoo Finance (for Demos)
```bash
./setup/fetch_sample_portfolio_futures_data.sh
```

---

## 🖥 Usage

### Generate ML training data

```bash
PYTHONPATH=. python3 bots/generate_ML_SL_Training_data.py \
  --config bots/configs/ml_sl_config.yaml \
  --output-dir bots/data/yahoo_finance/training_data
```

### Train PPO stop‑loss selector

```bash
PYTHONPATH=. python3 bots/train_ppo_stop_selector.py \
  --input_dir bots/data/yahoo_finance/training_data \
  --output_dir bots/models/PPO_Trailing_Stop_Loss \
  --total_timesteps 300000
```

### Run the GUI backtester

```bash
PYTHONPATH=. python3 run_backtest.py
```

### Run the headless backtester

```bash
PYTHONPATH=. python3 run_backtest_headless.py
```

### Run live with TradeStation (data‑only)

```bash
PYTHONPATH=. python3 run_live.py
```

---

## Know Issues
* Live data feed throws errors
* Zooming on charts can erase price data on UI
* Probably alot more bugs

---

## ⚠️ Disclaimer

This project is **for research and educational purposes only**.
It is **not financial advice** and should not be used for live trading with real capital without significant adaptation, testing, and risk controls.

