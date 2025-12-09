from abc import ABC, abstractmethod, ABCMeta
import pandas as pd
import yaml
import os
import numpy as np
from PyQt5 import QtCore
import time
import math

# ----------------- helpers -----------------

def round_to_tick(px, tick):
    """Round any price to the nearest exchange tick (deterministic, no banker's rounding)."""
    if tick is None or tick <= 0 or pd.isna(px):
        return px
    scaled = px / tick
    if scaled >= 0:
        return math.floor(scaled + 0.5) * tick  # half-up
    else:
        return -math.floor(-scaled + 0.5) * tick  # half-away-from-zero

# ----- Abstract API Interface -----

class APIInterface(ABC):
    @abstractmethod
    def connect(self): pass
    @abstractmethod
    def disconnect(self): pass
    @abstractmethod
    def get_historical_data(self, asset, timeframe, start, end): pass
    @abstractmethod
    def place_order(self, order): pass
    @abstractmethod
    def get_order_status(self, order_id): pass
    @abstractmethod
    def cancel_order(self, order_id): pass
    @abstractmethod
    def get_positions(self): pass
    @abstractmethod
    def get_total_pnl(self): pass
    @abstractmethod
    def get_portfolio(self): pass

# ----- Data Structures -----

class Order:
    def __init__(self, order_id, symbol, side, qty, price=None, order_type='market'):
        self.order_id = order_id
        self.symbol = symbol
        self.side = side  # 'buy' or 'sell'
        self.qty = qty
        self.price = price  # Limit/market price; market orders use None
        self.order_type = order_type
        self.status = 'open'
        self.fill_price = None
        self.fill_time = None
        
    def __str__(self):
        return (f"Order(order_id={self.order_id}, symbol={self.symbol}, side={self.side}, "
                f"qty={self.qty}, price={self.price}, order_type={self.order_type}, "
                f"status={self.status}, "
                f"fill_price={self.fill_price}, fill_time={self.fill_time})")

class Position:
    def __init__(self, symbol, contract_size):
        self.symbol = symbol
        self.contract_size = contract_size 
        self.qty = 0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.stop_loss_price = None
        self.take_profit = None 
        self.stop_order_id = None
        
    def __str__(self):
        return (
            f"Position(\n"
            f"  symbol={self.symbol},\n"
            f"  qty={self.qty},\n"
            f"  avg_entry_price={self.avg_entry_price},\n"
            f"  realized_pnl={self.realized_pnl},\n"
            f"  stop_loss_price={self.stop_loss_price},\n"
            f"  take_profit={self.take_profit}\n"
            f")"
        )
        
    def get_unrealized_pnl(self, current_price, tick_size, tick_value):
        if self.qty == 0:
            return 0.0
        ticks_moved = (current_price - self.avg_entry_price) / tick_size
        return ticks_moved * tick_value * self.qty

    def update_on_fill(self, fill_price, fill_qty, side, tick_size, tick_value):
        previous_qty = self.qty
        realized = 0.0

        # +ve for buy, -ve for sell
        fill_signed_qty = fill_qty if side == 'buy' else -fill_qty
        new_qty = previous_qty + fill_signed_qty

        # Closing or reversing a position
        if previous_qty * fill_signed_qty < 0:
            closing_qty = min(abs(previous_qty), abs(fill_signed_qty))
            if previous_qty > 0:
                pnl_per_tick = (fill_price - self.avg_entry_price) / tick_size
            else:
                pnl_per_tick = (self.avg_entry_price - fill_price) / tick_size
            realized += pnl_per_tick * tick_value * closing_qty
            self.realized_pnl += realized

            # Flipping position (reversal): any leftover qty opens new position at new price
            leftover_qty = abs(fill_signed_qty) - closing_qty
            if leftover_qty > 0:
                self.avg_entry_price = fill_price
            elif abs(fill_signed_qty) == abs(previous_qty):
                # Fully closed
                self.avg_entry_price = 0.0

        else:  # Adding to position (same side)
            if new_qty != 0:
                total_position_cost = self.avg_entry_price * abs(previous_qty) \
                    + fill_price * abs(fill_signed_qty)
                self.avg_entry_price = total_position_cost / (abs(new_qty) + 1e-10)
            else:
                self.avg_entry_price = 0.0

        self.qty = new_qty

        if self.qty == 0:
            self.avg_entry_price = 0.0
            self.stop_loss_price = None
            self.take_profit = None

        # --- Failsafe check: If trying to close and still nonzero, warn and zero ---
        if (previous_qty != 0 and abs(fill_signed_qty) == abs(previous_qty)
            and previous_qty * fill_signed_qty < 0 and self.qty != 0):
            print(
                f"[WARNING] {self.symbol}: Forced closure failed! Manually zeroing qty. "
                f"(prev_qty={previous_qty}, fill_signed_qty={fill_signed_qty}, side={side})"
            )
            self.qty = 0
            self.avg_entry_price = 0.0
            self.stop_loss_price = None
            self.take_profit = None

        return realized

# ----- Meta Class -----

class MetaQObjectABC(type(QtCore.QObject), ABCMeta):
    pass


def ensure_timestamp(df):
    if 'timestamp' not in df.columns and 'date' in df.columns:
        df['timestamp'] = df['date'].astype('int64') // 10**9
    return df

