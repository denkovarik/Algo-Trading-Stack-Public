# classes/ui_statistics.py
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QFormLayout, QLabel, QPushButton
)
from PyQt5.QtGui import QFont, QFontDatabase, QFontInfo

def _system_fixed_font() -> QtGui.QFont:
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

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == Qt.LeftButton:
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
    HEADERS = [
        "Symbol", "Trades", "Wins", "Losses", "Win%",
        "AvgWin", "AvgLoss", "PF", "Expectancy", "Comm", "Fees", "MaxDD"
    ]

    def __init__(self, stats: dict, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint)
        self.resize(1200, 650)
        self.setSizeGripEnabled(True)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)

        self.title_bar = TitleBar(self, "Backtest Statistics")
        outer.addWidget(self.title_bar)

        content = QtWidgets.QWidget(self); outer.addWidget(content, 1)
        main = QtWidgets.QVBoxLayout(content); main.setContentsMargins(8, 8, 8, 8); main.setSpacing(8)

        port_box = QGroupBox("Portfolio", content)
        form = QFormLayout(port_box)
        port = stats.get("portfolio", {}) if isinstance(stats, dict) else {}
        form.addRow(QLabel("Initial Cash:"), QLabel(self._money(port.get("initial_cash", 0.0))))
        form.addRow(QLabel("Final Equity:"), QLabel(self._money(port.get("total_equity", 0.0))))
        form.addRow(QLabel("Used Margin:"), QLabel(self._money(port.get("used_margin", 0.0))))
        form.addRow(QLabel("Max Drawdown:"), QLabel(self._pct(port.get("max_drawdown", 0.0))))
        main.addWidget(port_box)

        self.table = QTableWidget(0, len(self.HEADERS), content)
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setAutoScroll(False)
        self.table.setFocusPolicy(Qt.StrongFocus)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableView::item { padding: 6px 16px; }
            QHeaderView::section { padding: 8px 18px; font-weight: 600; }
        """)
        main.addWidget(self.table, 1)

        self.table.setSortingEnabled(False)
        per_asset = stats.get("per_asset", {}) if isinstance(stats, dict) else {}
        for sym, st in sorted(per_asset.items()):
            self._append_row(sym, st)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self._add_breathing_room(range(1, len(self.HEADERS)), extra=24)
        header.setSectionResizeMode(QHeaderView.Interactive)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, Qt.AscendingOrder)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close, parent=content)
        btns.rejected.connect(self.reject)
        main.addWidget(btns)

        self.setStyleSheet("QDialog { border: 1px solid rgba(0,0,0,0.25); border-radius: 6px; }")

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

    def _money(self, v) -> str:
        try: return f"{float(v):,.2f}"
        except Exception: return "0.00"

    def _pct(self, v) -> str:
        try: return f"{float(v):.2%}"
        except Exception: return "0.00%"

