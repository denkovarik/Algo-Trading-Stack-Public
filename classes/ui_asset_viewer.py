# ui_asset_viewer.py

import math
import time
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from pyqtgraph import DateAxisItem
import numpy as np
import pandas as pd
from functools import partial
from pyqtgraph import ScatterPlotItem



class HistoricalCandlestickItem(pg.GraphicsObject):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.picture = QtGui.QPicture()
        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)

        df = self.df
        if df.empty or 'timestamp' not in df.columns:
            painter.end()
            return

        timestamps = df['timestamp'].values

        # Directly use diffs, no sorting, just stable incremental timestamps
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            positive_intervals = intervals[intervals > 0]
            w_median = np.median(positive_intervals) if positive_intervals.size else 86400
        else:
            w_median = 86400

        # Enforce explicit minimum candle width
        MIN_CANDLE_WIDTH = 1.0
        w_median = max(w_median, MIN_CANDLE_WIDTH)

        for _, row in df.iterrows():
            t, open_, high = row['timestamp'], row['open'], row['high']
            low, close = row['low'], row['close']
            painter.setPen(pg.mkPen('w', width=0.8))
            painter.drawLine(QtCore.QPointF(t, low), QtCore.QPointF(t, high))
            color = pg.mkBrush('g') if close >= open_ else pg.mkBrush('r')
            painter.setBrush(color)
            body_top, body_bottom = max(open_, close), min(open_, close)
            height = max(body_top - body_bottom, 1e-8)
            painter.drawRect(QtCore.QRectF(t - w_median/3, body_bottom, 2*w_median/3, height))

        painter.end()

    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        df = self.df
        if df.empty or 'timestamp' not in df.columns:
            return QtCore.QRectF(0, 0, 1, 1)
        min_x, max_x = float(df['timestamp'].min()), float(df['timestamp'].max())
        min_price, max_price = float(df['low'].min()), float(df['high'].max())
        price_buffer = (max_price - min_price) * 0.05 if (max_price - min_price) else 1
        return QtCore.QRectF(min_x, min_price - price_buffer, 
                             max_x - min_x, (max_price - min_price) + 2 * price_buffer)

    def update_data(self, df):
        self.df = df.reset_index(drop=True)
        self.generatePicture()
        self.update()

    def append_bar(self, bar):
        # Direct append without sorting
        self.df.loc[len(self.df)] = bar
        self.generatePicture()
        self.update()


class LiveCandlestickItem(pg.GraphicsObject):
    def __init__(self, last_bar, df_reference=None):
        super().__init__()
        self.bar = last_bar
        self.df_reference = df_reference  # Reference DataFrame for median interval
        self.w_median = self.calculate_w_median()
        self.picture = QtGui.QPicture()
        self.generatePicture()

    def calculate_w_median(self, rolling_window=90):
        if self.df_reference is not None and len(self.df_reference) > 1:
            timestamps = self.df_reference['timestamp'].values[-rolling_window:]
            if len(timestamps) > 1:
                intervals = np.diff(timestamps)
                positive_intervals = intervals[intervals > 0]
                if positive_intervals.size > 0:
                    return np.median(positive_intervals)
        return 86400  # default 1 day (fallback)

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        bar = self.bar
        if bar is None or bar.empty or 'timestamp' not in bar:
            p.end()
            return

        t = bar['timestamp']
        open_, high, low, close = bar['open'], bar['high'], bar['low'], bar['close']
        p.setPen(pg.mkPen('y', width=1.2))
        p.drawLine(QtCore.QPointF(t, low), QtCore.QPointF(t, high))
        color = pg.mkBrush('g') if close >= open_ else pg.mkBrush('r')
        p.setBrush(color)
        p.setPen(pg.mkPen('y', width=1.0))
        body_top = max(open_, close)
        body_bottom = min(open_, close)
        height = max(body_top - body_bottom, 1e-8)
        body_width = 2 * self.w_median / 3
        p.drawRect(QtCore.QRectF(t - body_width / 2, body_bottom, body_width, height))
        p.end()

    def update_bar(self, new_bar, df_reference=None, rolling_window=90):
        self.bar = new_bar
        if df_reference is not None:
            self.df_reference = df_reference
            self.w_median = self.calculate_w_median()
        self.generatePicture()
        self.update()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        bar = self.bar
        if bar is None or bar.empty:
            return QtCore.QRectF(0, 0, 1, 1)
        price_buffer = (bar['high'] - bar['low']) * 0.05 if (bar['high'] - bar['low']) > 0 else 1
        body_width = 2 * self.w_median / 3
        return QtCore.QRectF(
            bar['timestamp'] - body_width / 2,
            bar['low'] - price_buffer,
            body_width,
            (bar['high'] - bar['low']) + 2 * price_buffer if (bar['high'] - bar['low']) > 0 else 1
        )


class CandlestickChartWidget(QtWidgets.QWidget):
    chart_updated = QtCore.pyqtSignal()

    INDICATOR_PLOT_STYLES = {
        'bb_ma':    {'pen': pg.mkPen('y', width=1.5), 'name': 'BB MA'},
        'bb_upper': {'pen': pg.mkPen('c', style=QtCore.Qt.DashLine), 'name': 'BB Upper'},
        'bb_lower': {'pen': pg.mkPen('m', style=QtCore.Qt.DashLine), 'name': 'BB Lower'},
        'ema_21':   {'pen': pg.mkPen('w', width=1), 'name': 'EMA 21'},
        'rsi_14':   {'pen': pg.mkPen('b', width=1), 'name': 'RSI 14'},
        'atr_14':   {'pen': pg.mkPen('orange', width=1.5), 'name': 'ATR (14)'},
    }
    GROUPED_INDICATORS = {
        'Bollinger Bands': ['bb_ma', 'bb_upper', 'bb_lower'],
        'EMA': ['ema_21'],
        'RSI': ['rsi_14'],
        'ATR': ['atr_14'],
    }

    # New constants
    MAX_INITIAL_BARS = 500      # Default number of bars to load when opening chart
    MAX_BARS_ON_SCREEN = 2000   # Maximum bars allowed in view at once

    def __init__(self, symbol, trading_env, parent=None):
        super().__init__(parent)
        self._xrange_initialized = False
        self.symbol = symbol
        self.trading_env = trading_env

        self.rolling_window_default = 90
        self.rolling_window = self.rolling_window_default
        self.following_latest = True

        layout = QtWidgets.QVBoxLayout(self)
        self.live_button = QtWidgets.QPushButton("Return to Live")
        self.live_button.setMaximumWidth(160)
        self.live_button.clicked.connect(self.return_to_live)
        self.live_button.hide()

        self.indicator_btn = QtWidgets.QPushButton("Indicators…")
        self.indicator_btn.setMaximumWidth(160)
        self.indicator_btn.clicked.connect(self.show_indicator_selector)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(self.indicator_btn)
        top_bar.addStretch(1)
        top_bar.addWidget(self.live_button)
        layout.addLayout(top_bar)

        self.plotWidget = pg.PlotWidget(axisItems={'bottom': DateAxisItem()})
        layout.addWidget(self.plotWidget, stretch=1)
        self.setLayout(layout)
        self.setMinimumWidth(800)

        # --- Candlestick Items ---
        df_full = self.trading_env.get_latest_data(self.symbol, self.rolling_window)
        if len(df_full) > self.MAX_INITIAL_BARS:
            df_init = df_full.iloc[-self.MAX_INITIAL_BARS:]
        else:
            df_init = df_full

        self.hist_item = HistoricalCandlestickItem(df_init)
        last_bar = df_full.iloc[-1] if not df_full.empty else pd.Series()
        self.live_item = LiveCandlestickItem(last_bar, df_full)
        self.plotWidget.addItem(self.hist_item)
        self.plotWidget.addItem(self.live_item)

        self.indicator_lines = {}
        self.order_items = []

        self.atrPlot = pg.PlotWidget(axisItems={'bottom': DateAxisItem()})
        self.atrPlot.setFixedHeight(120)
        self.atrPlot.setTitle('ATR (14)')
        self.atrPlot.hide()
        layout.addWidget(self.atrPlot)

        self.plotWidget.setMouseEnabled(x=True, y=True)

        vb = self.plotWidget.getViewBox()
        vb.sigXRangeChanged.connect(self._on_xrange_changed)
        vb.sigRangeChangedManually.connect(self._on_user_interaction)

        self.plotWidget.showGrid(x=True, y=True)
        self.plotWidget.setTitle(f"{symbol} Candlestick Chart")
        self.plotWidget.setLabel('bottom', 'Date')
        self.plotWidget.setLabel('left', 'Price')

        self.active_groups = set(['Bollinger Bands'])
        self.indicator_lines = {}

        self.update_chart()

    def enable_panning_zoom(self):
        self.plotWidget.setMouseEnabled(x=True, y=True)

    def on_bar_updated(self, df):
        if df is None or df.empty or 'timestamp' not in df.columns:
            return

        if len(df) >= 2:
            new_bar = df.iloc[-2].to_dict()
            self.hist_item.append_bar(new_bar)
            self.live_item.update_bar(df.iloc[-1], df, self.rolling_window)

            if self.following_latest:
                self.auto_scroll_to_latest()

            self.plotWidget.repaint()

        self.update_chart()
            
    def auto_scroll_to_latest(self):
        df = self.trading_env.get_latest_data(self.symbol, self.rolling_window)
        if df.empty or 'timestamp' not in df.columns:
            return
        ts_max = float(df['timestamp'].iloc[-1])
        if len(df) > self.rolling_window:
            ts_min = float(df['timestamp'].iloc[-self.rolling_window])
        else:
            ts_min = float(df['timestamp'].iloc[0])
        if not any(math.isnan(v) for v in [ts_min, ts_max]):
            self.plotWidget.setXRange(ts_min, ts_max + 60, padding=0.05)

    def update_chart(self):
        df = self.trading_env.get_latest_data(self.symbol, self.rolling_window)
        if df is None or df.empty or 'timestamp' not in df.columns:
            print("[ERROR] No valid data available.")
            return

        total_bars = len(df)
        bars_to_show = min(max(5, self.rolling_window), total_bars)

        # Slice the data to plot
        if self.following_latest:
            df_plot = df[-bars_to_show:]
        else:
            x_min, x_max = self.plotWidget.getViewBox().viewRange()[0]
            mask = (df['timestamp'] >= x_min) & (df['timestamp'] <= x_max)
            df_plot = df[mask]
            if df_plot.empty:
                print("[WARNING] df_plot empty in selected range; reverting to latest bars.")
                df_plot = df[-bars_to_show:]

        # Enforce max bars on screen
        if len(df_plot) > self.MAX_BARS_ON_SCREEN:
            step = max(1, len(df_plot) // self.MAX_BARS_ON_SCREEN)
            df_plot = df_plot.iloc[::step]

        # --- Split history vs. live (last row) ---
        if len(df_plot) == 0:
            print("[ERROR] No bars to plot.")
            return

        hist = df_plot.iloc[:-1].copy()      # fully closed bars
        live = df_plot.iloc[-1].copy()       # current live bar (may have NaN H/L by design)

        # Drop NaNs ONLY on history; keep live even if H/L are NaN
        if not hist.empty:
            hist = hist.dropna(subset=['timestamp', 'open', 'high', 'low', 'close'])

        # Update candlestick items
        self.hist_item.update_data(hist)

        # Candle width heuristic: prefer history, else fall back to whole slice
        timestamps = hist['timestamp'].to_numpy() if len(hist) > 1 else df_plot['timestamp'].to_numpy()
        w_median = np.median(np.diff(timestamps)) if len(timestamps) > 1 else 86400
        self.hist_item.w_median = w_median
        self.live_item.w_median = w_median
        self.live_item.update_bar(live)

        # Update indicators and orders (use df_plot so indicators can include the live point)
        self.plot_indicators(df_plot)
        orders = self.trading_env.get_orders(self.symbol)
        self.plot_orders(orders, df)
        self.plot_active_orders()

        # Autoscale Y-axis
        viewbox = self.plotWidget.getViewBox()
        self._autoscale_y_axis(viewbox, viewbox.viewRange()[0])

        # Visual refresh
        self.plotWidget.getPlotItem().update()
        self.plotWidget.repaint()
        QtWidgets.QApplication.processEvents()
        self.chart_updated.emit()
        
    def plot_orders(self, orders, df):
        # Remove previous order markers
        if hasattr(self, 'order_scatter'):
            self.plotWidget.removeItem(self.order_scatter)
        # Prepare scatter data
        spots = []
        for order in orders:
            # Only plot filled orders for this symbol
            if order['symbol'] != self.symbol or order['fill_price'] is None:
                continue
            # Get x = timestamp, y = price
            if isinstance(order['fill_time'], str):
                ts = pd.to_datetime(order['fill_time'])
            else:
                ts = order['fill_time']
            # If your chart x-axis is integer timestamp, convert accordingly
            if 'timestamp' in df.columns and isinstance(ts, pd.Timestamp):
                x_val = ts.timestamp()
            elif 'timestamp' in self.df.columns:
                x_val = order['fill_time']
            else:
                x_val = ts
            y_val = order['fill_price']
            # Style: green up for buy, red down for sell
            brush = 'g' if order['side'] == 'buy' else 'r'
            symbol = 't1' if order['side'] == 'buy' else 't'
            spots.append({'pos': (x_val, y_val), 'brush': brush, 'symbol': symbol, 'size': 12})
        self.order_scatter = ScatterPlotItem(spots=spots)
        self.plotWidget.addItem(self.order_scatter)

    def plot_indicators(self, df):
        # Remove existing indicator lines
        for line in self.indicator_lines.values():
            self.plotWidget.removeItem(line)
        self.indicator_lines.clear()

        # Clear ATR subplot
        self.atrPlot.clear()
        show_atr = False

        cols_to_plot = []
        for group in self.active_groups:
            cols_to_plot.extend(self.GROUPED_INDICATORS.get(group, []))

        for col in cols_to_plot:
            if col.startswith("atr_"):
                show_atr = True
                continue  # Plot ATR below
            if col not in df.columns:
                continue
            mask = df[col].notnull()
            if not mask.any():
                continue
            x = df['timestamp'][mask].to_numpy()
            y = df[col][mask].to_numpy()
            style = self.INDICATOR_PLOT_STYLES.get(col, {'pen': pg.mkPen('w', width=1)})
            self.indicator_lines[col] = self.plotWidget.plot(x, y, **style)

        # ATR subplot logic
        if show_atr:
            self.atrPlot.show()
            for col in cols_to_plot:
                if col.startswith('atr_') and col in df.columns:
                    mask = df[col].notnull()
                    if not mask.any():
                        continue
                    x = df['timestamp'][mask].to_numpy()
                    y = df[col][mask].to_numpy()
                    self.atrPlot.plot(x, y, pen=pg.mkPen('orange', width=1.5))
                    self.atrPlot.setLabel('left', 'ATR')
                    self.atrPlot.setTitle(col.upper())
                    if y.size:
                        y_min, y_max = y.min(), y.max()
                        buffer = (y_max - y_min) * 0.1 if (y_max - y_min) else 1
                        self.atrPlot.setYRange(y_min - buffer, y_max + buffer, padding=0)
        else:
            self.atrPlot.hide()

    def plot_active_orders(self):
        # Remove previous
        for item in self.order_items:
            self.plotWidget.removeItem(item)
        self.order_items = []
        positions = self.trading_env.get_positions()
        pos = positions.get(self.symbol)
        if pos and pos['qty'] != 0:
            color = 'g' if pos['qty'] > 0 else 'r'
            entry_line = pg.InfiniteLine(pos=pos['avg_entry_price'], angle=0, 
                pen=pg.mkPen(color, width=2, style=QtCore.Qt.DashLine))
            self.plotWidget.addItem(entry_line)
            self.order_items.append(entry_line)
            # SL
            stop_loss = pos.get('stop_loss_price')
            if stop_loss is not None:
                sl_line = pg.InfiniteLine(pos=stop_loss, angle=0, pen=pg.mkPen('r', width=1))
                self.plotWidget.addItem(sl_line)
                self.order_items.append(sl_line)
            # TP
            take_profit = pos.get('take_profit')
            if take_profit is not None:
                tp_line = pg.InfiniteLine(pos=take_profit, angle=0, pen=pg.mkPen('g', width=1))
                self.plotWidget.addItem(tp_line)
                self.order_items.append(tp_line)
                
    def show_indicator_selector(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Indicators")
        layout = QtWidgets.QVBoxLayout(dialog)
        checks = {}
        for group_name in self.GROUPED_INDICATORS:
            cb = QtWidgets.QCheckBox(group_name)
            cb.setChecked(group_name in self.active_groups)
            checks[group_name] = cb
            layout.addWidget(cb)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(btns)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.active_groups = {k for k, cb in checks.items() if cb.isChecked()}
            self.update_chart()

    def _autoscale_y_axis(self, viewbox, x_range):
        df = self.trading_env.get_latest_data(self.symbol, self.rolling_window)
        if df is None or df.empty:
            return
        mask = (df['timestamp'] >= x_range[0]) & (df['timestamp'] <= x_range[1])
        visible = df[mask]
        y_values = []
        if not visible.empty:
            y_values.append(visible['low'].min())
            y_values.append(visible['high'].max())
        positions = self.trading_env.get_positions()
        pos = positions.get(self.symbol)
        if pos:
            if pos.get("stop_loss_price") is not None:
                y_values.append(pos["stop_loss_price"])
            if pos.get("take_profit") is not None:
                y_values.append(pos["take_profit"])
        if y_values:
            y_min, y_max = min(y_values), max(y_values)
            y_buffer = (y_max - y_min) * 0.1 if (y_max - y_min) != 0 else 1
            self.plotWidget.setYRange(y_min - y_buffer, y_max + y_buffer, padding=0)

    def _on_xrange_changed(self, viewbox, xrange):
        df = self.trading_env.get_latest_data(self.symbol, self.rolling_window)
        if df is None or df.empty:
            return

        x_min, x_max = xrange
        mask = (df['timestamp'] >= x_min) & (df['timestamp'] <= x_max)
        visible = df[mask]
        y_values = []
        if not visible.empty:
            y_values.append(visible['low'].min())
            y_values.append(visible['high'].max())
        positions = self.trading_env.get_positions()
        pos = positions.get(self.symbol)
        if pos:
            if pos.get("stop_loss_price") is not None:
                y_values.append(pos["stop_loss_price"])
            if pos.get("take_profit") is not None:
                y_values.append(pos["take_profit"])
        if y_values:
            y_min, y_max = min(y_values), max(y_values)
            y_buffer = (y_max - y_min) * 0.1 if (y_max - y_min) != 0 else 1
            self.plotWidget.setYRange(y_min - y_buffer, y_max + y_buffer, padding=0)

    def _on_user_interaction(self):
        # Get current visible x-range (timestamps)
        x_min, x_max = self.plotWidget.getViewBox().viewRange()[0]
        span = x_max - x_min

        if not hasattr(self, "_last_xrange"):
            self._last_xrange = (x_min, x_max)
            self._last_span = span

        old_x_min, old_x_max = self._last_xrange
        old_span = self._last_span

        # Use a relative threshold for zoom detection
        zoomed = abs(span - old_span) / max(abs(old_span), 1e-12) > 1e-4
        panned = abs(x_min - old_x_min) > 1e-6 and not zoomed

        df = self.trading_env.get_latest_data(self.symbol, self.rolling_window)
        latest_bar_visible = False

        if df is not None and not df.empty:
            latest_ts = float(df['timestamp'].iloc[-1])
            # Allow a buffer (use median bar width if available)
            bar_width = df['timestamp'].diff().dropna().median() if len(df) > 1 else 1
            tolerance = bar_width * 1.2 if bar_width else 1
            latest_bar_visible = x_max >= latest_ts - tolerance

        if latest_bar_visible:
            if not self.following_latest:
                print("Latest bar is in view: following live")
                self.live_button.hide()
                self.following_latest = True
        else:
            if zoomed:
                pass
            elif panned:
                self.following_latest = False
                self.live_button.show()
            else:
                pass

        if df is not None and not df.empty:
            bars_in_view = ((df['timestamp'] >= x_min) & (df['timestamp'] <= x_max)).sum()

            # Clamp zoom-out to max allowed bars
            if bars_in_view > self.MAX_BARS_ON_SCREEN:
                ts_max = float(df['timestamp'].iloc[-1]) if self.following_latest else x_max
                ts_min = ts_max - (self.MAX_BARS_ON_SCREEN * bar_width)
                self.plotWidget.setXRange(ts_min, ts_max, padding=0)
                bars_in_view = self.MAX_BARS_ON_SCREEN

            self.rolling_window = bars_in_view

        self._last_xrange = (x_min, x_max)
        self._last_span = span

        self.update_chart()

    def return_to_live(self):
        self.following_latest = True
        self.live_button.hide()
        df = self.trading_env.get_latest_data(self.symbol, self.rolling_window)
        if df is not None and not df.empty and 'timestamp' in df.columns:
            N = len(df)
            candles_to_show = min(self.rolling_window, N)
            x_max = float(df['timestamp'].iloc[-1])
            if N > candles_to_show:
                x_min = float(df['timestamp'].iloc[-candles_to_show])
            else:
                x_min = float(df['timestamp'].iloc[0])
            self.plotWidget.setXRange(x_min, x_max, padding=0)
        self.update_chart()
            
            
class OrderEntryWidget(QtWidgets.QGroupBox):
    order_placed = QtCore.pyqtSignal(dict)

    def __init__(self, symbol, trading_env, parent=None):
        super().__init__("Place Order", parent)
        self.symbol = symbol
        self.trading_env = trading_env

        # -----------------------------------------------------------------
        #  ATR rolling window (set by bot or default)
        # -----------------------------------------------------------------
        self.atr_rolling_window = 90  # default for 5-min charts

        layout = QtWidgets.QFormLayout(self)

        # ----- Side / Qty -------------------------------------------------
        self.side_box = QtWidgets.QComboBox()
        self.side_box.addItems(["Buy (Long)", "Sell (Short)"])
        self.qty_box = QtWidgets.QSpinBox()
        self.qty_box.setMinimum(1)
        self.qty_box.setMaximum(1_000_000)
        self.qty_box.setValue(1)

        # ----- SL / TP Inputs --------------------------------------------
        self.sl_multiple_box = QtWidgets.QLineEdit()          # ATR multiple (e.g., 2.0)
        self.sl_multiple_box.setPlaceholderText("e.g. 2.0")
        self.tp_box = QtWidgets.QLineEdit()                   # Optional TP price
        self.tp_box.setPlaceholderText("optional")

        self.place_order_btn = QtWidgets.QPushButton("Place Order")
        self.place_order_btn.clicked.connect(self._emit_order)

        layout.addRow("Side:", self.side_box)
        layout.addRow("Quantity:", self.qty_box)
        layout.addRow("Stop Loss (ATR ×):", self.sl_multiple_box)
        layout.addRow("Take Profit (price):", self.tp_box)
        layout.addRow(self.place_order_btn)

    # -----------------------------------------------------------------
    #  Set ATR window from bot (e.g., 14 for daily, 90 for 5-min)
    # -----------------------------------------------------------------
    def set_atr_rolling_window(self, n_bars: int):
        if n_bars > 0:
            self.atr_rolling_window = int(n_bars)
            print(f"[UI] ATR rolling window set to {self.atr_rolling_window} bars")

    # -----------------------------------------------------------------
    #  _emit_order – calculate stop price and emit order
    # -----------------------------------------------------------------
    def _emit_order(self):
        side = "buy" if self.side_box.currentIndex() == 0 else "sell"
        qty = self.qty_box.value()

        order = {
            "symbol": self.symbol,
            "side": side,
            "qty": qty,
            "order_type": "market",
        }

        sl_mult_text = self.sl_multiple_box.text().strip()
        if sl_mult_text:
            try:
                atr_multiple = float(sl_mult_text)
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Error", "ATR multiple must be a number.")
                return

            df = self.trading_env.get_latest_data(self.symbol, self.atr_rolling_window)
            atr = float(df["atr_14"].dropna().iloc[-1]) if not df["atr_14"].dropna().empty else 1.0
            offset = atr_multiple * atr
            order["original_stop_offset"] = round(atr_multiple * atr, 2)
            print(f"[UI] Will attach stop {offset:.2f} ticks from fill price")

        self.order_placed.emit(order)
        

class PositionPanelWidget(QtWidgets.QGroupBox):
    close_position = QtCore.pyqtSignal()
    sl_tp_modified = QtCore.pyqtSignal(str, float)

    def __init__(self, symbol, trading_env, parent=None):
        super().__init__("Open Position", parent)
        self.symbol = symbol
        self.trading_env = trading_env
        layout = QtWidgets.QFormLayout(self)
        self.qty_label = QtWidgets.QLabel("-")
        self.entry_label = QtWidgets.QLabel("-")
        self.unrealized_label = QtWidgets.QLabel("-")
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(lambda: self.close_position.emit())
        layout.addRow("Qty", self.qty_label)
        layout.addRow("Entry Price", self.entry_label)
        layout.addRow("Unrealized PnL", self.unrealized_label)
        layout.addRow("", self.close_btn)
        # SL/TP controls
        self.sl_label = QtWidgets.QLabel("-")
        self.tp_label = QtWidgets.QLabel("-")
        self.raise_sl_btn = QtWidgets.QPushButton("▲")
        self.lower_sl_btn = QtWidgets.QPushButton("▼")
        self.raise_tp_btn = QtWidgets.QPushButton("▲")
        self.lower_tp_btn = QtWidgets.QPushButton("▼")
        self.raise_sl_btn.clicked.connect(lambda: self._modify('sl', +1))
        self.lower_sl_btn.clicked.connect(lambda: self._modify('sl', -1))
        self.raise_tp_btn.clicked.connect(lambda: self._modify('tp', +1))
        self.lower_tp_btn.clicked.connect(lambda: self._modify('tp', -1))
        sl_btn_row = QtWidgets.QHBoxLayout()
        sl_btn_row.addWidget(self.lower_sl_btn)
        sl_btn_row.addWidget(self.raise_sl_btn)
        tp_btn_row = QtWidgets.QHBoxLayout()
        tp_btn_row.addWidget(self.lower_tp_btn)
        tp_btn_row.addWidget(self.raise_tp_btn)
        layout.addRow("Stop Loss", self.sl_label)
        layout.addRow("", sl_btn_row)
        layout.addRow("Take Profit", self.tp_label)
        layout.addRow("", tp_btn_row)
        self.setMinimumWidth(220)
        self.update_panel()

    def update_panel(self):
        positions = self.trading_env.get_positions()
        pos = positions.get(self.symbol)
        if pos and pos.get('qty', 0) != 0:
            self.qty_label.setText(str(pos["qty"]))
            self.entry_label.setText(f'{pos["avg_entry_price"]:.2f}')
            self.unrealized_label.setText(f'{pos["unrealized_pnl"]:.2f}')
            sl_val = pos.get('stop_loss_price')
            tp_val = pos.get('take_profit')
            self.sl_label.setText(f"{sl_val:.2f}" if sl_val is not None else "-")
            self.tp_label.setText(f"{tp_val:.2f}" if tp_val is not None else "-")
            self.close_btn.setEnabled(True)
        else:
            self.qty_label.setText("-")
            self.entry_label.setText("-")
            self.unrealized_label.setText("-")
            self.sl_label.setText("-")
            self.tp_label.setText("-")
            self.close_btn.setEnabled(False)

    def _modify(self, which, direction):
        df = self.trading_env.get_latest_data(self.symbol, self.rolling_window)
        step = 1.0
        if 'atr_14' in df.columns and not df['atr_14'].dropna().empty:
            atr = float(df['atr_14'].dropna().iloc[-1])
            step = 0.5 * atr
        current_val = None
        positions = self.trading_env.get_positions()
        pos = positions.get(self.symbol)
        if which == 'sl':
            current_val = pos.get('stop_loss_price')
        elif which == 'tp':
            current_val = pos.get('take_profit')
        if current_val is not None:
            new_value = current_val + step * direction
            self.sl_tp_modified.emit(which, new_value)


class AssetViewer(QtWidgets.QWidget):
    def __init__(self, symbol, trading_env):
        super().__init__()
        self.symbol = symbol
        self.trading_env = trading_env
        self.df = pd.DataFrame()  # Maintain internal DataFrame explicitly

        # Existing widgets
        self.chart = CandlestickChartWidget(symbol, trading_env)
        self.order_entry = OrderEntryWidget(symbol, trading_env)
        self.position_panel = PositionPanelWidget(symbol, trading_env)

        # Layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.chart, 2)
        right_col = QtWidgets.QVBoxLayout()
        right_col.addWidget(self.position_panel)
        right_col.addWidget(self.order_entry)
        right_col.addStretch(1)
        layout.addLayout(right_col, 1)
        self.setLayout(layout)
        self.setWindowTitle(symbol)
        self.resize(1200, 700)

        # Connections
        self.order_entry.order_placed.connect(self.on_order_placed)
        self.position_panel.close_position.connect(self.on_close_position)
        self.position_panel.sl_tp_modified.connect(self.on_modify_sl_tp)
        self.chart.chart_updated.connect(self.on_chart_updated)

        if hasattr(self.trading_env.api, 'bar_updated'):
            self.trading_env.api.bar_updated.connect(self.chart.on_bar_updated)

        # Initial update
        self.refresh_panels()

    def enable_panning_zoom(self):
        self.chart.enable_panning_zoom()
        self.chart.update_chart()

    def on_order_placed(self, order):
        self.trading_env.place_order(order)
        self.refresh_panels()

    def on_close_position(self):
        positions = self.trading_env.get_positions()
        pos = positions.get(self.symbol)
        if not pos or pos["qty"] == 0:
            QtWidgets.QMessageBox.information(self, "No Position", "No open position to close!")
            return
        qty = abs(pos["qty"])
        side = "sell" if pos["qty"] > 0 else "buy"
        order = {
            "symbol": self.symbol,
            "side": side,
            "qty": qty,
            "order_type": "market"
        }
        self.trading_env.place_order(order)
        self.refresh_panels()

    def on_modify_sl_tp(self, which, new_value):
        if which == 'sl':
            self.trading_env.modify_stop_loss(self.symbol, new_value)
        elif which == 'tp':
            self.trading_env.modify_take_profit(self.symbol, new_value)
        self.refresh_panels()

    def on_chart_updated(self):
        self.position_panel.update_panel()

    def refresh_panels(self):
        self.chart.update_chart()
        self.position_panel.update_panel()

    def return_to_live(self):
        self.chart.return_to_live()
        self.chart.update_chart()


class ChartOnlyWindow(QtWidgets.QWidget):
    def __init__(self, symbol, trading_env):
        super().__init__()
        self.setWindowTitle(f"{symbol} Chart")
        layout = QtWidgets.QVBoxLayout(self)
        self.chart = CandlestickChartWidget(symbol, trading_env)
        layout.addWidget(self.chart)
        self.setLayout(layout)
        self.resize(1200, 700)


class AssetSelectorWindow(QtWidgets.QWidget):
    def __init__(self, symbols, env):
        super().__init__()
        self.setWindowTitle("Select Asset to View")
        self.env = env

        layout = QtWidgets.QVBoxLayout(self)
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(symbols)
        layout.addWidget(QtWidgets.QLabel("Select an asset to view its chart:"))
        layout.addWidget(self.combo)

        self.open_btn = QtWidgets.QPushButton("Open Chart")
        self.open_btn.clicked.connect(self.open_chart)
        layout.addWidget(self.open_btn)

        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        layout.addWidget(self.close_btn)

        self.chart_windows = []

    def open_chart(self):
        symbol = self.combo.currentText()
        if symbol:
            win = ChartOnlyWindow(symbol, self.env)
            win.show()
            self.chart_windows.append(win)

