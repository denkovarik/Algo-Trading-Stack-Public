import sys
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from pyqtgraph import DateAxisItem
from classes.ui_asset_viewer import AssetViewer
from classes.ui_helpers import format_portfolio
from classes.ui_statistics import StatisticsDialog


# --- Main Application Window ---

class BacktestWindow(QtWidgets.QWidget):
    def __init__(self, trading_env, backtester):
        super().__init__()
        self.trading_env = trading_env
        self.backtester = backtester
        self.setWindowTitle("Backtest Control Panel")
        layout = QtWidgets.QVBoxLayout(self)

        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(self.backtester.get_asset_list())
        layout.addWidget(self.combo)

        self.chart_button = QtWidgets.QPushButton("Open Chart")
        self.chart_button.clicked.connect(self.open_chart)
        layout.addWidget(self.chart_button)

        self.backtest_button = QtWidgets.QPushButton("Start Backtest")
        self.backtest_button.clicked.connect(self.toggle_backtest)
        layout.addWidget(self.backtest_button)

        self.stop_button = QtWidgets.QPushButton("Stop Backtest")
        self.stop_button.clicked.connect(self.stop_backtest)
        layout.addWidget(self.stop_button)

        self.rev100_button = QtWidgets.QPushButton("⏮")
        self.rev10_button  = QtWidgets.QPushButton("⏪")
        self.fwd10_button  = QtWidgets.QPushButton("⏩")
        self.fwd100_button = QtWidgets.QPushButton("⏭")
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.rev100_button)
        button_row.addWidget(self.rev10_button)
        button_row.addWidget(self.fwd10_button)
        button_row.addWidget(self.fwd100_button)
        layout.addLayout(button_row)

        # ---- Navigation buttons FIX: call self.rewind_backtest, etc. not direct backtester ----
        self.rev100_button.clicked.connect(lambda: self.rewind_backtest(100))
        self.rev10_button.clicked.connect(lambda: self.rewind_backtest(10))
        self.fwd10_button.clicked.connect(lambda: self.fast_forward_backtest(10))
        self.fwd100_button.clicked.connect(lambda: self.fast_forward_backtest(100))

        self.equity_windows = []
        self.equity_curve_button = QtWidgets.QPushButton("Show Equity Curve")
        self.equity_curve_button.clicked.connect(self.show_equity_curve)
        layout.addWidget(self.equity_curve_button)

        self.portfolio_button = QtWidgets.QPushButton("Show Portfolio")
        self.portfolio_button.clicked.connect(self.show_portfolio)
        layout.addWidget(self.portfolio_button)

        self.chart_windows = []
        self.backtester.bar_updated.connect(self.on_new_bar_updated)
        self.backtester.backtest_finished.connect(self.on_backtest_finished)

        self.backtest_running = False
        self.backtest_paused = False

        speed_row = QtWidgets.QHBoxLayout()
        self.speed_label = QtWidgets.QLabel("Speed:")
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(20)
        self.speed_slider.setValue(5)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.speed_value_label = QtWidgets.QLabel("5")
        self.speed_slider.valueChanged.connect(self.update_backtest_speed)
        speed_row.addWidget(QtWidgets.QLabel("Slow"))
        speed_row.addWidget(self.speed_slider)
        speed_row.addWidget(QtWidgets.QLabel("Fast"))
        layout.addLayout(speed_row)

        self._pending_backtest_interval = 50
        self.setLayout(layout)
        
        self.stats_button = QtWidgets.QPushButton("Show Statistics")
        self.stats_button.clicked.connect(self.show_statistics)
        layout.addWidget(self.stats_button)

    def get_speed_interval_ms(self):
        value = self.speed_slider.value()
        slider_min = self.speed_slider.minimum()
        slider_max = self.speed_slider.maximum()
        return int(10 + (1000 - 10) * (slider_max - value) / (slider_max - slider_min))

    def open_chart(self):
        symbol = self.combo.currentText()
        if symbol:
            chart_win = AssetViewer(symbol, self.trading_env)
            chart_win.resize(1200, 700)
            chart_win.show()
            chart_win.chart.update_chart()
            self.backtester.bar_updated.connect(chart_win.chart.on_bar_updated)
            self.chart_windows.append(chart_win)
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "No data available.")
            
    def show_statistics(self):
        stats = self.backtester.get_stats_snapshot()  # BacktesterEngine snapshot
        dlg = StatisticsDialog(stats, self)           # shared dialog
        dlg.exec_()

    def toggle_backtest(self):
        interval_ms = self.get_speed_interval_ms()
        label = self.backtest_button.text()

        if label == "Restart Backtest":
            # Reset and start a fresh backtest
            self.backtester.reset_backtest()
            self.trading_env.reset_indicators()
            for symbol in self.backtester.get_asset_list():
                df = self.backtester.get_asset_data(symbol)
                if df is not None and not df.empty:
                    self.trading_env._compute_indicators(symbol, df)
            for chart_win in self.chart_windows:
                chart_win.refresh_panels()

            self.backtester.start_backtest(interval_ms=interval_ms)
            self.backtest_button.setText("Pause Backtest")
            self.backtest_running = True
            self.backtest_paused = False
            return

        if not self.backtest_running:
            if self.backtest_paused:
                self.backtester.resume_backtest(interval_ms=interval_ms)
            else:
                self.trading_env.reset_indicators()
                self.backtester.start_backtest(interval_ms=interval_ms)
            self.backtest_button.setText("Pause Backtest")
            self.backtest_running = True
            self.backtest_paused = False
        else:
            self.backtester.pause_backtest()
            self.backtest_button.setText("Resume Backtest")
            self.backtest_running = False
            self.backtest_paused = True

        self.rev100_button.setEnabled(not self.backtest_running)
        self.rev10_button.setEnabled(not self.backtest_running)
        self.fwd10_button.setEnabled(not self.backtest_running)
        self.fwd100_button.setEnabled(not self.backtest_running)

    def stop_backtest(self):
        # Pause the backtest (stop the timer)
        self.backtester.pause_backtest()

        # Reset internal simulation state
        self.backtester.reset_backtest()

        # Clear and recompute indicators from scratch
        self.trading_env.reset_indicators()
        for symbol in self.backtester.get_asset_list():
            df = self.backtester.get_asset_data(symbol)
            if df is not None and not df.empty:
                self.trading_env._compute_indicators(symbol, df)

        # Refresh all open charts
        for chart_win in self.chart_windows:
            chart_win.refresh_panels()

        # Reset button state and flags
        self.backtest_button.setText("Start Backtest")
        self.backtest_running = False
        self.backtest_paused = False

    def rewind_backtest(self, steps=10):
        if not self.backtest_running:
            self.backtester.current_index = max(0, self.backtester.current_index - steps)
            # ---- TRUNCATE equity lists! ----
            self.backtester.equity_history = self.backtester.equity_history[:self.backtester.current_index+1]
            self.backtester.equity_time_history = self.backtester.equity_time_history[:self.backtester.current_index+1]
            self.backtester.bar_advanced.emit()
            primary_df = self.backtester.assets[0]['data']
            if primary_df is not None:
                self.backtester.bar_updated.emit(primary_df.iloc[:self.backtester.current_index + 1])

    def fast_forward_backtest(self, steps=10):
        if not self.backtest_running:
            max_len = max(len(asset['data']) for asset in self.backtester.assets if asset['data'] is not None)
            self.backtester.current_index = min(max_len - 1, self.backtester.current_index + steps)
            # ---- TRUNCATE equity lists! ----
            self.backtester.equity_history = self.backtester.equity_history[:self.backtester.current_index+1]
            self.backtester.equity_time_history = self.backtester.equity_time_history[:self.backtester.current_index+1]
            self.backtester.bar_advanced.emit()
            primary_df = self.backtester.assets[0]['data']
            if primary_df is not None:
                self.backtester.bar_updated.emit(primary_df.iloc[:self.backtester.current_index + 1])

    def on_backtest_finished(self):
        self.backtest_running = False
        self.backtest_paused = False  # finished, not paused
        self.backtest_button.setText("Restart Backtest")
        self.rev100_button.setEnabled(True)
        self.rev10_button.setEnabled(True)
        self.fwd10_button.setEnabled(True)
        self.fwd100_button.setEnabled(True)

    def update_backtest_speed(self, value):
        self.speed_value_label.setText(str(value))
        if self.backtest_running:
            interval_ms = self.get_speed_interval_ms()
            self.backtester.resume_backtest(interval_ms=interval_ms)

    def on_new_bar(self):
        for chart_win in self.chart_windows:
            chart_win.chart.update_chart()
        for win in self.equity_windows:
            win.update_curve(
                self.backtester.equity_history,
                self.backtester.equity_time_history,
                freq='h',
                gap_policy='pad',        # <- fills weekends
                include_partial=False    # <- avoids flicker inside the current hour
            )
            
    def on_backtest_finished(self):
        self.backtest_running = False
        self.backtest_paused = False
        self.backtest_button.setText("Restart Backtest")
        self.rev100_button.setEnabled(True)
        self.rev10_button.setEnabled(True)
        self.fwd10_button.setEnabled(True)
        self.fwd100_button.setEnabled(True)

        # --- ✨ NEW: write results for notebooks to pick up ---
        try:
            import os, json, pandas as pd, time
            export_dir = os.environ.get("ATS_EXPORT_DIR", "exports/last_run")
            os.makedirs(export_dir, exist_ok=True)

            stats = self.backtester.get_stats_snapshot()
            t, eq = self.backtester.get_equity_series()
            df_eq = pd.DataFrame({"time": pd.to_datetime(t), "equity": eq})

            with open(os.path.join(export_dir, "stats.json"), "w") as f:
                json.dump(stats, f, default=float, indent=2)
            df_eq.to_csv(os.path.join(export_dir, "equity.csv"), index=False)

            # optional: breadcrumb
            with open(os.path.join(export_dir, "metadata.txt"), "w") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
        except Exception as e:
            print("[export warning]", e)
            
    def on_new_bar_updated(self, df):
        # update all open chart windows explicitly
        for chart_win in self.chart_windows:
            chart_win.chart.on_bar_updated(df)

        # update equity windows explicitly
        for win in self.equity_windows:
            win.update_curve(
                self.backtester.equity_history,
                self.backtester.equity_time_history,
                freq='h',
                gap_policy='pad',        # <- fills weekends
                include_partial=False    # <- avoids flicker inside the current hour
            )

    def show_equity_curve(self):
        if not self.backtester.equity_history or not self.backtester.equity_time_history \
                or len(self.backtester.equity_history) != len(self.backtester.equity_time_history):
            QtWidgets.QMessageBox.warning(self, "No Data", "No equity curve data to plot.")
            return
        plot_win = EquityCurvePlotWindow(self.backtester.equity_history, 
                                         self.backtester.equity_time_history)
        plot_win.show()
        self.equity_windows.append(plot_win)

    def show_portfolio(self):
        portfolio = self.backtester.get_portfolio()
        msg = self.format_portfolio(portfolio)
        QtWidgets.QMessageBox.information(self, "Portfolio", msg)

    def format_portfolio(self, portfolio):
        lines = []
        lines.append(f"Cash: ${portfolio.get('cash', 0):,.2f}")
        lines.append(f"Total Equity: ${portfolio.get('total_equity', 0):,.2f}\n")
        lines.append("Positions:")
        positions = portfolio.get("positions", {})
        if not positions:
            lines.append("  (none)")
        else:
            for sym, pos in positions.items():
                lines.append(f"  {sym}: qty={pos['qty']} avg_entry={pos['avg_entry_price']:.2f} "
                             f"realized={pos['realized_pnl']:.2f} unrealized={pos['unrealized_pnl']:.2f}")
        lines.append("\nOpen Orders:")
        orders = portfolio.get("open_orders", [])
        if not orders:
            lines.append("  (none)")
        else:
            for o in orders:
                lines.append(f"  #{o['order_id']} {o['side']} {o['qty']} {o['symbol']} "
                             f"@ {o['price']} status={o['status']}")
        return "\n".join(lines)


class EquityCurvePlotWindow(QtWidgets.QWidget):
    def __init__(self, equity_history, time_history):
        super().__init__()
        self.setWindowTitle("Equity Curve")
        layout = QtWidgets.QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': DateAxisItem()})
        layout.addWidget(self.plot_widget)
        self.curve = self.plot_widget.plot([], [], pen=pg.mkPen('g', width=2), symbol=None)
        self.plot_widget.setLabel('left', "Equity ($)")
        self.plot_widget.setLabel('bottom', "Date/Time")
        self.plot_widget.setTitle("Equity Curve")
        self.update_curve(equity_history, time_history)

    def update_curve(self, equity_history, time_history,
                     max_points=1000, freq='h', gap_policy='pad', include_partial=False):
        import numpy as np
        import pandas as pd

        n = min(len(equity_history), len(time_history))
        if n == 0:
            self.curve.setData([], [])
            return

        # Build a time-indexed series
        t_idx = pd.to_datetime(time_history[:n])
        s = pd.Series(equity_history[:n], index=t_idx)

        if freq:
            s = s.resample(freq).last()

            # ---- handle gaps (weekends, holidays) ----
            if gap_policy == 'pad':      # flat line across closures
                s = s.ffill().bfill()
            elif gap_policy == 'connect':  # skip empty hours; compress time
                s = s.dropna()
            elif gap_policy == 'keep':     # current behavior: show a break
                pass
            else:
                raise ValueError("gap_policy must be 'pad', 'connect', or 'keep'.")

            # Drop the last, not-yet-complete bucket if requested
            if not include_partial and len(s):
                end_last = s.index[-1] + pd.tseries.frequencies.to_offset(freq)
                if t_idx[-1] < end_last:
                    s = s.iloc[:-1]

        # Convert to epoch seconds for DateAxisItem
        x = (s.index.astype('int64') // 10**9).tolist()
        y = s.values.astype(float).tolist()

        # Stable decimation, always include last point
        N = len(x)
        if N > max_points:
            step = max(1, N // (max_points - 1))
            x, y = x[::step], y[::step]
            last_ts = int(s.index[-1].value // 10**9)
            if x[-1] != last_ts:
                x.append(last_ts); y.append(float(s.iloc[-1]))

        self.curve.setData(x, y)
        
        
class LiveMainWindow(QtWidgets.QWidget):
    def __init__(self, trading_env, symbols):
        super().__init__()
        self.trading_env = trading_env
        self.setWindowTitle("Live Trading System")
        layout = QtWidgets.QVBoxLayout(self)

        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(symbols)
        layout.addWidget(self.combo)

        self.chart_button = QtWidgets.QPushButton("Open Live Chart")
        self.chart_button.clicked.connect(self.open_chart)
        layout.addWidget(self.chart_button)

        self.chart_windows = [] 

    def open_chart(self):
        symbol = self.combo.currentText()
        if symbol:
            chart_win = AssetViewer(symbol, self.trading_env)
            chart_win.resize(1200, 700)
            chart_win.show()
            chart_win.chart.update_chart()
            self.trading_env.api.bar_updated.connect(chart_win.chart.on_bar_updated)
            self.chart_windows.append(chart_win)
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "No data available.")
  

def launch_gui(trading_env, backtester):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = BacktestWindow(trading_env, backtester)
    win.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
    win.show()

    # Run the event loop only if we're not already in one
    if not getattr(app, "_in_event_loop", False):
        app._in_event_loop = True
        try:
            app.exec_()
        finally:
            app._in_event_loop = False

 
    
def launch_live_gui(trading_env, symbols):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    win = LiveMainWindow(trading_env, symbols)
    
    # Explicitly connect API to start polling immediately
    trading_env.connect()  # explicitly start live data polling
    
    win.show()
    sys.exit(app.exec_())
    



