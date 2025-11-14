# classes/Backtest_Runner.py

import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QTimer, Qt, QEvent
from PyQt5.QtGui import QFont, QFontDatabase, QFontInfo
from PyQt5.QtWidgets import (
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QFormLayout, QLabel, QPushButton
)
from tqdm import tqdm
import numpy as np

from classes.Backtester_Engine import BacktesterEngine
from classes.Trading_Environment import TradingEnvironment
from classes.ui_main_window import EquityCurvePlotWindow
from classes.ui_statistics import StatisticsDialog
from classes.ui_asset_viewer import AssetSelectorWindow

from bots.coin_flip_bot.coin_flip_bot import CoinFlipBot
from bots.trend_following_bot.trend_following_bot import TrendFollowingBot
from bots.exit_strategies import TrailingATRExit, RLTrailingATRExit


class BacktestApp:
    def __init__(self, config_path, bot):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        self.engine = BacktesterEngine(config_path=config_path)
        self.env = TradingEnvironment(precompute_indicators=True)
        self.env.set_api(self.engine)
        self.env.set_bot(bot)

        self._configure_engine()
        self.eq_win = EquityCurvePlotWindow(self.engine.equity_history, self.engine.equity_time_history)
        self.eq_win.show()

    def _configure_engine(self):
        self.engine.reset_backtest()
        self.engine.is_running = True
        self.engine.live_price_mode = "open_only"
        self.engine.skip_synthetic_open_bars = True
        self.engine.include_slippage = False
        self.engine.config['intrabar_tp_sl_policy'] = 'worst_case'


class BacktestRunner:
    def __init__(
        self,
        backtest_app: BacktestApp,
        plot_update_every: int = 500,     # redraw every N bars
        plot_max_points: int = 4000       # downsample to at most this many points
    ):
        self.app = backtest_app
        self.plot_update_every = max(1, int(plot_update_every))
        self.plot_max_points = max(100, int(plot_max_points))
        
    def _decimate_series(self, times, values, max_points: int):
        """
        Cheap stride decimation that preserves the last point.
        times, values are Python lists (as stored by the engine).
        """
        n = len(values)
        if n <= max_points:
            return times, values
        step = n // max_points
        idx = np.arange(0, n-1, step, dtype=int)
        if idx.size == 0 or idx[-1] != (n - 1):
            idx = np.append(idx, n - 1)
        # Keep same types the chart expects (lists)
        t_ds = [times[i] for i in idx]
        v_ds = [values[i] for i in idx]
        return t_ds, v_ds

    def run(self):
        engine = self.app.engine
        total_bars = len(engine.df) if getattr(engine, "df", None) is not None else 0

        with tqdm(total=total_bars, unit="bar", desc="Backtest", dynamic_ncols=True) as pbar:
            for i in range(total_bars):
                engine.step()
                # Throttle redraws
                if (i % self.plot_update_every == 0) or (i == total_bars - 1):
                    eq_win = getattr(self.app, "eq_win", None)
                    if eq_win is not None:
                        # Downsample before drawing
                        t, v = engine.equity_time_history, engine.equity_history
                        t_ds, v_ds = self._decimate_series(t, v, self.plot_max_points)
                        # NOTE: your window’s API is (values, times) in current code
                        eq_win.update_curve(v_ds, t_ds)  # was every bar
                        QtWidgets.QApplication.processEvents()  # also throttled
                pbar.update(1)

        engine.close_all_positions()
        QTimer.singleShot(500, self.show_results)
        sys.exit(self.app.app.exec())

    def show_results(self):
        symbols = self.app.engine.get_asset_list()
        selector = AssetSelectorWindow(symbols, self.app.env)
        selector.show()

        self.post_run_win = PostRunWindow(self.app.engine)
        self.post_run_win.show()

        original_open_chart = selector.open_chart

        def open_chart_and_scroll():
            original_open_chart()
            chart_window = selector.chart_windows[-1]
            chart_window.chart.return_to_live()

        try:
            selector.open_btn.clicked.disconnect()
        except Exception:
            pass
        selector.open_btn.clicked.connect(open_chart_and_scroll)

        print("Select an asset to open chart. Close all windows to exit.")


# ---------------------- Stats UI ----------------------

def _system_fixed_font() -> QtGui.QFont:
    """Return a guaranteed fixed-pitch font (for labels if needed)."""
    f = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    f.setStyleHint(QFont.Monospace)
    f.setFixedPitch(True)
    if not QFontInfo(f).fixedPitch():
        for family in ("DejaVu Sans Mono", "Liberation Mono", "Consolas", "Menlo", "Courier"):
            f = QtGui.QFont(family)
            f.setStyleHint(QFont.Monospace)
            f.setFixedPitch(True)
            if QFontInfo(f).fixedPitch():
                break
    return f


class TitleBar(QtWidgets.QWidget):
    """
    Custom (frameless) title bar that lets you drag the dialog *without* the WM re-centering jump.
    """
    def __init__(self, parent_dialog: QtWidgets.QDialog, title: str = "Backtest Statistics"):
        super().__init__(parent_dialog)
        self._dlg = parent_dialog
        self._press_offset = None

        self.setFixedHeight(34)
        self.setAutoFillBackground(True)
        self.setCursor(Qt.ArrowCursor)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(8)

        self.title_lbl = QLabel(title)
        f = self.title_lbl.font()
        f.setPointSize(f.pointSize() + 1)
        f.setBold(True)
        self.title_lbl.setFont(f)

        lay.addWidget(self.title_lbl, 1)

        self.min_btn = QPushButton("—")
        self.min_btn.setFixedSize(26, 22)
        self.min_btn.clicked.connect(self._dlg.showMinimized)
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(26, 22)
        self.close_btn.clicked.connect(self._dlg.reject)

        for b in (self.min_btn, self.close_btn):
            b.setCursor(Qt.PointingHandCursor)
            b.setFlat(True)

        lay.addWidget(self.min_btn)
        lay.addWidget(self.close_btn)

        self.setStyleSheet("""
            TitleBar { background: palette(window); }
            QLabel { padding-left: 2px; }
            QPushButton { border: none; }
            QPushButton:hover { background: rgba(0,0,0,0.08); }
            QPushButton:pressed { background: rgba(0,0,0,0.16); }
        """)

    # --- Mouse drag to move window (no jump) ---
    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == Qt.LeftButton:
            # store the offset between the cursor and the window's top-left corner
            self._press_offset = e.globalPos() - self._dlg.frameGeometry().topLeft()
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()
        else:
            e.ignore()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self._press_offset is not None and (e.buttons() & Qt.LeftButton):
            self._dlg.move(e.globalPos() - self._press_offset)
            e.accept()
        else:
            e.ignore()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        self._press_offset = None
        self.setCursor(Qt.ArrowCursor)
        e.accept()


class StatisticsDialog(QtWidgets.QDialog):
    """
    Frameless dialog with a custom TitleBar (prevents jump-to-center) and a table for stats.
    """
    HEADERS = [
        "Symbol", "Trades", "Wins", "Losses", "Win%",
        "AvgWin", "AvgLoss", "PF", "Expectancy", "Comm", "Fees", "MaxDD"
    ]

    def __init__(self, stats: dict, parent=None):
        super().__init__(parent)
        # Frameless -> we control dragging and there is no WM jump
        self.setWindowFlags(
            Qt.Window | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint
        )
        self.resize(1200, 650)
        self.setSizeGripEnabled(True)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- Custom title bar (drag here to move) ----
        self.title_bar = TitleBar(self, "Backtest Statistics")
        outer.addWidget(self.title_bar)

        # content area
        content = QtWidgets.QWidget(self)
        outer.addWidget(content, 1)
        main = QtWidgets.QVBoxLayout(content)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(8)

        # --- Portfolio summary (top box) ---
        port_box = QGroupBox("Portfolio", content)
        form = QFormLayout(port_box)
        port = stats.get("portfolio", {}) if isinstance(stats, dict) else {}
        form.addRow(QLabel("Initial Cash:"), QLabel(self._money(port.get("initial_cash", 0.0))))
        form.addRow(QLabel("Final Equity:"), QLabel(self._money(port.get("total_equity", 0.0))))
        form.addRow(QLabel("Used Margin:"), QLabel(self._money(port.get("used_margin", 0.0))))
        form.addRow(QLabel("Max Drawdown:"), QLabel(self._pct(port.get("max_drawdown", 0.0))))
        main.addWidget(port_box)

        # --- Per-asset table ---
        self.table = QTableWidget(0, len(self.HEADERS), content)
        self.table.setHorizontalHeaderLabels(self.HEADERS)

        # Passive table so dragging the window is never intercepted
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setAutoScroll(False)
        self.table.setFocusPolicy(Qt.StrongFocus)

        # Cosmetics: spacing between columns
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableView::item { padding: 6px 16px; }
            QHeaderView::section { padding: 8px 18px; font-weight: 600; }
        """)
        main.addWidget(self.table, 1)

        # Fill rows (disable sorting while populating)
        self.table.setSortingEnabled(False)
        per_asset = stats.get("per_asset", {}) if isinstance(stats, dict) else {}
        for sym in sorted(per_asset.keys()):
            self._append_row(sym, per_asset.get(sym, {}))

        # Auto-size, add breathing room, then allow interactive resize/sort
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self._add_breathing_room(range(1, len(self.HEADERS)), extra=24)
        header.setSectionResizeMode(QHeaderView.Interactive)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, Qt.AscendingOrder)

        # Buttons
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close, parent=content)
        btns.rejected.connect(self.reject)
        main.addWidget(btns)

        # subtle border so frameless looks like a window
        self.setStyleSheet("""
            QDialog { border: 1px solid rgba(0,0,0,0.25); border-radius: 6px; }
        """)

    # ---------- table helpers ----------

    def _append_row(self, symbol: str, st: dict):
        row = self.table.rowCount()
        self.table.insertRow(row)

        def intval(v):  return 0 if v is None else int(v)
        def money(v):   return self._money(v)
        def pct(v):     return self._pct(v)
        def pf(v):
            if v is None or (isinstance(v, float) and (v != v)):  # NaN
                return "n/a"
            try:
                return f"{float(v):.2f}"
            except Exception:
                return "n/a"

        data = [
            symbol,
            f"{intval(st.get('trades'))}",
            f"{intval(st.get('wins'))}",
            f"{intval(st.get('losses'))}",
            pct(st.get('win_rate')),
            money(st.get('avg_win')),
            money(st.get('avg_loss')),
            pf(st.get('profit_factor')),
            money(st.get('expectancy')),
            money(st.get('commission_total')),
            money(st.get('fee_total')),
            pct(st.get('max_drawdown')),
        ]

        for col, text in enumerate(data):
            item = QTableWidgetItem(text)
            if col == 0:
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                f = item.font(); f.setBold(True); item.setFont(f)
            else:
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
            self.table.setItem(row, col, item)

    def _add_breathing_room(self, cols, extra=18):
        for c in cols:
            self.table.setColumnWidth(c, self.table.columnWidth(c) + int(extra))

    # ---------- formatting ----------

    def _money(self, v) -> str:
        try:
            return f"{float(v):,.2f}"
        except Exception:
            return "0.00"

    def _pct(self, v) -> str:
        try:
            return f"{float(v):.2%}"
        except Exception:
            return "0.00%"


class PostRunWindow(QtWidgets.QWidget):
    """
    Tiny utility window that appears after the headless backtest finishes.
    Contains a single 'Show Statistics' button that pops a StatisticsDialog.
    """
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.setWindowTitle("Backtest Complete")
        self.setMinimumWidth(260)
        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel("Backtest finished.")
        layout.addWidget(label)

        self.stats_btn = QtWidgets.QPushButton("Show Statistics")
        layout.addWidget(self.stats_btn)

        close_btn = QtWidgets.QPushButton("Close")
        layout.addWidget(close_btn)

        self.stats_btn.clicked.connect(self.on_show_stats)
        close_btn.clicked.connect(self.close)

    def on_show_stats(self):
        stats = self.engine.get_stats_snapshot()
        dlg = StatisticsDialog(stats, self)
        dlg.exec_()

