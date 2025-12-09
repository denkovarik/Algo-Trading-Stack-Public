# Trading System High-Level Design (2025 Comprehensive Blueprint)

This document provides a **full high-level picture** of the trading research platform. It is designed to be read by **engineers, quants, and LLMs** to understand how all components fit together, how to extend them, and how to use the system effectively.

---

## 1. Overview

The project is a **modular research and backtesting framework** for algorithmic trading.  

It supports:

- **Backtesting**: bar-driven, phase-aware, realistic fills (tick rounding, slippage, commissions/fees), leverage and maintenance margin enforcement.
- **Live trading**: via TradeStation live data (with other broker adapters possible).
- **Strategies (Bots)**: modular bots that implement entry signals, plug into shared risk sizing, and delegate exits to ExitStrategies.
- **ExitStrategies**: traditional ATR-based stops, ML-trained RL exits, or hybrid bandit overlays.
- **Research tooling**: offline dataset generation, PPO/LSTM model training, and online LinUCB contextual bandits.
- **User Interfaces**:  
  - GUI backtester (PyQt + PyQtGraph)  
  - Headless backtester (CLI + tqdm)  
  - Live TradeStation viewer  

**Goal:** provide a **reproducible research environment** where strategies and exits can be developed, tested, and compared consistently in simulation, then applied in live data contexts.

---

## 2. Architecture at a Glance

```
+===========================================================================================+
|                                ALGO-TRADING-STACK (2025)                                  |
|                        Modular Backtesting & Research Platform                            |
+===========================================================================================+

 Data & Brokers                        Core Runtime                   Strategies / Exits
 --------------                        ----------------               -------------------
 • Yahoo Finance CSVs       -->        • BacktesterEngine      <---   • Bots (CoinFlip, TrendFollowing)
 • TradeStation (live) <-- Adapter(s)  • TradingEnvironment           • ExitStrategy plug-ins
                                       • Indicator Registry           • Trailing ATR / Fixed Ratio
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

## 3. Core Components

### 3.1 API Interface

Defines the **execution & data contract** (`APIInterface`):

- `connect` / `disconnect`
- `place_order`, `cancel_order`, `modify_stop_loss`, `modify_take_profit`
- `get_positions`, `get_portfolio`, `get_total_pnl`
- `get_asset_data`, `get_latest_data`, `get_asset_list`

**Implementations:**
- **BacktesterEngine** (simulation):
  - Phase-aware (OPEN / INTRABAR / CLOSE).
  - Orders, fills, slippage, commissions, margin.
  - Equity curves, campaign stats, per-asset DD and expectancy.
- **TradeStationLiveAPI** (live data):
  - Fetches minute bars + quotes.
  - Emits `bar_updated` to UI.
  - Order methods stubbed (data-only).
- *(Alpaca adapter possible but not included).*

---

### 3.2 BacktesterEngine (simulation)

- **Phases:**
  - **OPEN**: bot runs on open snapshot (H/L hidden).
  - **INTRABAR**: stops/TP triggered with gap-aware rules and tie-break policy.
  - **CLOSE**: equity marked, maintenance margin enforced, indicators updated.
- **Margin enforcement:**
  - Intrabar worst-case valuation (LOW for longs, HIGH for shorts).
  - Forced liquidation at bar CLOSE.
- **Accounting:**
  - Futures tick_size/tick_value/contract_size respected.
  - Realized/unrealized PnL, commissions, fees deducted.
- **Statistics:**
  - Per-asset equity series, drawdowns, expectancy, win-rate.
  - Campaign grouping (flat→nonzero→flat).
- **Signals:**
  - `bar_updated(df)`, `bar_closed(df)`, `bar_advanced()`, `backtest_finished()`.

---

### 3.3 Trading Environment

Acts as the **hub** between adapter, bots, and UI:

- **Indicator pipeline**: EMA, RSI, ATR, Bollinger Bands (via registry).
- **Signal wiring**:
  - At OPEN → compute snapshot, call bot `on_bar(env)`.
  - At CLOSE → recompute indicators only for last bar.
- **Proxy methods**: `get_portfolio`, `get_positions`, `place_order`, `cancel_order`, `get_orders`, etc.
- **Safe no-lookahead**: indicators carried forward so bots never see future data.

---

### 3.4 Strategies (Bots)

All bots inherit from **BaseStrategyBot**, which provides:

- Risk-aware position sizing:
  - % equity at risk, slippage buffers, commission/fee accounting.
  - Margin caps: `qty ≤ available_equity / initial_margin`.
- Eligibility gates:
  - Session hours & holidays.
  - ATR ≥ threshold.
- Maintenance flattening:
  - Auto-closes positions before 5pm ET.
- Lifecycle:
  - If flat → call `decide_side`, compute SL/TP, size, place order.
  - If in position → trail stop with `exit_strategy.update_stop`.
- Online learning hook:
  - After each bar, call `exit.ingest_trade_log(env)` if supported.

**Built-in bots:**
- **CoinFlipBot** — random entries (baseline).
- **TrendFollowingBot** — EMA crossback or ATR breakout entries.

---

### 3.5 Exit Strategies

All implement:

- `initial_levels(...) -> (stop_loss, take_profit|None)`
- `update_stop(...) -> new_stop|None`

**Variants:**
- **TrailingATRExit** — ATR trailing stop, no TP.
- **FixedRatioExit** — static ATR stop + fixed R-multiple TP.
- **RLTrailingATRExit (PPO)** — ATR multiple chosen by PPO policy.
- **SequenceRLATRExit (LSTM)** — TorchScript LSTM rolling classifier.
- **Hybrid exits (PPO/LSTM + LinUCB Bandit)** — blend prior with online bandit updates.

---

### 3.6 User Interface

- **GUI Backtester**:
  - Control panel (start/pause/resume/restart, speed slider, rewind/FF).
  - Equity curve plots.
  - Statistics dialog (trades, win/loss, expectancy, DD, fees).
  - Asset viewers with candlestick chart, indicators, order plotting, SL/TP lines, order entry & position panels.
- **Headless Backtester**:
  - CLI with tqdm progress bar.
  - Optionally prints bandit stats after run.
- **Live Viewer**:
  - Minimal PyQt chart for TradeStation live data.

---

### 3.7 Research & Training

- **Dataset generation** (`generate_ML_SL_Training_data.py`):
  - Scans historical bars, simulates all SL multiples, labels best.
  - Exports training CSVs with features [ATR, RSI, EMA, Close, PositionType].
- **Model training**:
  - PPO trainer (`train_ppo_stop_selector.py`) → `.zip` policies.
  - LSTM trainer (`train_lstm_sl_model.py`) → TorchScript `.pt` + `.meta.json`.
- **Online learning**:
  - LinUCB contextual bandits (`bandit_overlay.py`).
  - Updated from closed trades in real time.
  - Saves/loads per-symbol bandit state.

---

## 4. Data & Metadata Management

- **Sources**:
  - TradeStation CSVs: `TimeStamp, Open, High, Low, Close, TotalVolume`.
  - Yahoo Finance CSVs: mapped to `[date, open, high, low, close, volume, timestamp]`.
- **Master timeline**:
  - Union of all asset timestamps.
  - O/H/L left NaN on synthetic bars (gap markers).
  - CLOSE forward-filled, volume zero-filled.
- **Config-driven params** per symbol:
  - `tick_size`, `tick_value`, `contract_size`
  - `initial_margin`, `maintenance_margin`
  - `commission_per_contract`, `fee_per_trade`
  - `slippage_ticks` or `slippage_pct`

---

## 5. Entry Points

- **GUI backtest**:
  ```bash
  PYTHONPATH=. python3 run_backtest.py
  ```
- **Headless backtest**:
  ```bash
  PYTHONPATH=. python3 run_backtest_headless.py
  ```
- **Live (TradeStation)**:
  ```bash
  PYTHONPATH=. python3 run_live.py
  ```
- **Generate training data**:
  ```bash
  PYTHONPATH=. python3 bots/generate_ML_SL_Training_data.py
  ```
- **Train PPO exits**:
  ```bash
  PYTHONPATH=. python3 bots/train_ppo_stop_selector.py
  ```
- **Train LSTM exits**:
  ```bash
  PYTHONPATH=. python3 bots/train_lstm_sl_model.py
  ```

---

## 6. Advantages of the Design

- **Realism**: phase-aware sim with intrabar liquidation & gap-aware SL/TP.
- **Extensibility**: bots, exits, and adapters all plug into stable contracts.
- **Unified workflow**: same bot/exit code runs in sim and live modes.
- **Research loop**: backtests produce labeled data → PPO/LSTM training → hybrid exits with online learning.
- **Reproducibility**: random seeds, deterministic backtests (except during bandit exploration).
- **Modular UI**: can run GUI, headless, or live charting.

---

This document serves as the **comprehensive blueprint** for the project. It is intended to give both **engineers and LLMs** the full high-level picture of the system: its architecture, extension points, workflows, and usage.
