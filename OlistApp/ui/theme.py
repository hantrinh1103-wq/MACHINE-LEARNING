"""Bảng màu, stylesheet và các widget nhỏ dùng chung cho giao diện mới (xanh đậm + cam)."""
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QApplication, QMenu, QFrame, QHBoxLayout, QHeaderView, QLabel, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

NAVY = "#0B2545"
NAVY_2 = "#13315C"
ORANGE = "#F26B21"
ORANGE_DARK = "#D9570F"
ORANGE_SOFT = "#FDEBDD"
BG = "#F3F5F9"
CARD = "#FFFFFF"
TEXT = "#1B2A41"
MUTED = "#66758C"
BORDER = "#DFE5EE"
GREEN = "#1E8E5A"
GREEN_SOFT = "#E1F4EA"
GREY_SOFT = "#ECEFF4"
CLUSTER_COLORS = ["#0B2545", "#F26B21", "#2A9D8F", "#8E6BBF", "#E0A100", "#5B8DEF",
                  "#C0392B", "#7A8794", "#3FA34D", "#B56576"]

# Mã ký tự trong font "Segoe MDL2 Assets" (có sẵn trên Windows 10/11)
ICON_HOME, ICON_CLOCK, ICON_STAR, ICON_PEOPLE = "", "", "", ""
ICON_USER, ICON_LOCK, ICON_POWER, ICON_SEARCH = "", "", "", ""

QSS = f"""
* {{ font-family: 'Segoe UI'; font-size: 13px; color: {TEXT}; }}
QMainWindow, QWidget#content, QWidget#page {{ background: {BG}; }}

/* ---------- sidebar ---------- */
QFrame#sidebar {{ background: {NAVY}; }}
QFrame#sidebar QLabel {{ color: #FFFFFF; background: transparent; }}
QLabel#brand {{ font-size: 17px; font-weight: 800; }}
QLabel#brandSub {{ color: #AEBBD0; font-size: 12px; }}
QLabel#navCaption {{ color: #7F90AC; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
QPushButton#nav {{ background: transparent; color: #C9D3E3; border: none; border-left: 4px solid transparent;
    text-align: left; padding: 12px 16px; font-size: 14px; }}
QPushButton#nav:hover {{ background: {NAVY_2}; color: #FFFFFF; }}
QPushButton#nav:checked {{ background: {NAVY_2}; color: #FFFFFF; border-left: 4px solid {ORANGE};
    font-weight: 700; }}
QLabel#userName {{ font-weight: 700; }}
QLabel#userRole {{ color: #AEBBD0; font-size: 12px; }}
QPushButton#logout {{ background: transparent; color: #FFFFFF; border: 1px solid #3A5078; border-radius: 6px;
    padding: 8px 12px; text-align: left; }}
QPushButton#logout:hover {{ background: {ORANGE}; border-color: {ORANGE}; }}

/* ---------- thanh tiêu đề ---------- */
QFrame#topbar {{ background: {CARD}; border-bottom: 1px solid {BORDER}; }}
QLabel#pageTitle {{ font-size: 20px; font-weight: 700; color: {NAVY}; }}
QLabel#pageSub {{ color: {MUTED}; }}
QLabel#chip {{ background: {GREEN_SOFT}; color: {GREEN}; border-radius: 12px; padding: 4px 12px;
    font-weight: 600; }}

/* ---------- thẻ ---------- */
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; }}
QFrame#card QLabel {{ background: transparent; }}
QLabel#cardTitle {{ font-size: 14px; font-weight: 700; color: {NAVY}; }}
QLabel#empty {{ color: {MUTED}; font-size: 14px; }}
QLabel#cardHint, QLabel#muted {{ color: {MUTED}; font-size: 12px; }}
QFrame#kpi {{ background: {CARD}; border: 1px solid {BORDER}; border-left: 4px solid {NAVY};
    border-radius: 10px; }}
QFrame#kpi[accent="orange"] {{ border-left: 4px solid {ORANGE}; }}
QFrame#kpi QLabel {{ background: transparent; }}
QLabel#kpiLabel {{ color: {MUTED}; font-size: 12px; }}
QLabel#kpiValue {{ font-size: 24px; font-weight: 700; color: {NAVY}; }}
QLabel#kpiValueSmall {{ font-size: 17px; font-weight: 700; color: {NAVY}; }}
QLabel#scoreValue {{ font-size: 34px; font-weight: 700; color: {NAVY}; }}
QLabel#badge {{ border-radius: 12px; padding: 4px 12px; font-weight: 700; }}

/* ---------- nhập liệu ---------- */
QLineEdit, QSpinBox, QComboBox {{ background: #FFFFFF; border: 1px solid #C9D2E0; border-radius: 6px;
    padding: 7px 10px; min-height: 20px; selection-background-color: {ORANGE}; }}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border: 2px solid {ORANGE}; padding: 6px 9px; }}
QComboBox QLineEdit {{ border: none; padding: 0; background: transparent; min-height: 0; }}
QComboBox QLineEdit:focus {{ border: none; padding: 0; }}
QComboBox QAbstractItemView {{ background: #FFFFFF; selection-background-color: {ORANGE_SOFT};
    selection-color: {TEXT}; }}
QPushButton#primary {{ background: {ORANGE}; color: #FFFFFF; border: none; border-radius: 6px;
    padding: 9px 18px; font-weight: 700; }}
QPushButton#primary:hover {{ background: {ORANGE_DARK}; }}
QPushButton#primary:disabled {{ background: #F5B892; }}
QPushButton#secondary {{ background: #FFFFFF; color: {NAVY}; border: 1px solid {NAVY}; border-radius: 6px;
    padding: 8px 16px; font-weight: 600; }}
QPushButton#secondary:hover {{ background: {NAVY}; color: #FFFFFF; }}
QPushButton:focus {{ outline: none; }}
QPushButton#primary:focus, QPushButton#secondary:focus {{ border: 2px solid {NAVY_2}; }}

/* ---------- bảng, tab ---------- */
QTableWidget {{ background: #FFFFFF; alternate-background-color: #F7F9FC; border: 1px solid {BORDER};
    border-radius: 6px; gridline-color: transparent; selection-background-color: {ORANGE_SOFT};
    selection-color: {TEXT}; }}
QTableWidget::item {{ padding: 4px 8px; }}
QHeaderView::section {{ background: #EEF2F8; color: {NAVY}; font-weight: 700; border: none;
    border-bottom: 2px solid {NAVY}; padding: 7px 8px; }}
QTabWidget::pane {{ border: none; border-top: 1px solid {BORDER}; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 9px 16px; border: none;
    border-bottom: 3px solid transparent; font-weight: 600; }}
QTabBar::tab:hover {{ color: {NAVY}; }}
QTabBar::tab:selected {{ color: {NAVY}; border-bottom: 3px solid {ORANGE}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: #C3CCDA; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: #C3CCDA; border-radius: 5px; min-width: 30px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---------- đăng nhập ---------- */
QFrame#loginBrand {{ background: {NAVY}; }}
QFrame#loginBrand QLabel {{ color: #FFFFFF; background: transparent; }}
QLabel#loginLogo {{ font-size: 40px; font-weight: 800; }}
QLabel#loginTagline {{ font-size: 19px; font-weight: 600; }}
QLabel#loginDesc, QLabel#loginFoot {{ color: #AEBBD0; font-size: 12px; }}
QLabel#loginBullet {{ font-size: 13px; }}
QFrame#loginForm {{ background: #FFFFFF; }}
QFrame#loginForm QLabel {{ background: transparent; }}
QLabel#loginTitle {{ font-size: 24px; font-weight: 800; color: {NAVY}; letter-spacing: 2px; }}
QLabel#fieldLabel {{ font-weight: 600; color: {NAVY}; }}
QLabel#error {{ color: #C0392B; font-weight: 600; }}
QPushButton#link {{ background: transparent; border: none; color: {ORANGE_DARK}; font-weight: 700;
    text-decoration: underline; padding: 0; }}
QPushButton#link:hover {{ color: {NAVY}; }}
QCheckBox {{ color: {MUTED}; }}
"""


def glyph_icon(code, color="#FFFFFF", size=20):
    """Tạo QIcon từ một ký tự của font biểu tượng Segoe MDL2 Assets."""
    pm = QPixmap(size * 2, size * 2)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    f = QFont("Segoe MDL2 Assets")
    f.setPixelSize(int(size * 1.5))
    p.setFont(f)
    p.setPen(QColor(color))
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, code)
    p.end()
    return QIcon(pm)


def card(title=None, hint=None):
    """Trả về (khung thẻ, layout dọc bên trong)."""
    frame = QFrame()
    frame.setObjectName("card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(16, 14, 16, 16)
    lay.setSpacing(10)
    if title:
        t = QLabel(title)
        t.setObjectName("cardTitle")
        lay.addWidget(t)
    if hint:
        h = QLabel(hint)
        h.setObjectName("cardHint")
        h.setWordWrap(True)
        lay.addWidget(h)
    return frame, lay


class Kpi(QFrame):
    def __init__(self, label, value="–", note="", accent="navy", small=False):
        super().__init__()
        self.setObjectName("kpi")
        self.setProperty("accent", accent)
        self.setMinimumHeight(64 if small else 88)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        self.label = QLabel(label)
        self.label.setObjectName("kpiLabel")
        self.value = QLabel(value)
        self.value.setObjectName("kpiValueSmall" if small else "kpiValue")
        self.value.setWordWrap(True)
        self.note = QLabel(note)
        self.note.setObjectName("muted")
        self.note.setWordWrap(True)
        lay.addWidget(self.label)
        lay.addWidget(self.value)
        if not small:
            lay.addWidget(self.note)

    def set(self, value, note=None):
        self.value.setText(str(value))
        if note is not None:
            self.note.setText(note)


class RiskBar(QWidget):
    """Thanh điểm rủi ro 0–1 có vạch đánh dấu ngưỡng cảnh báo."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(48)
        self.score, self.threshold = None, None

    def setValue(self, score, threshold):
        self.score, self.threshold = score, threshold
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width() - 2
        track = QRectF(1, 24, w, 12)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(GREY_SOFT))
        p.drawRoundedRect(track, 6, 6)
        if self.score is not None:
            warn = self.threshold is not None and self.score >= self.threshold
            p.setBrush(QColor(ORANGE if warn else NAVY))
            p.drawRoundedRect(QRectF(1, 24, max(w * self.score, 12), 12), 6, 6)
        if self.threshold is not None:
            x = 1 + w * self.threshold
            p.setPen(QPen(QColor(TEXT), 2))
            p.drawLine(int(x), 19, int(x), 41)
            f = p.font()
            f.setPixelSize(11)
            p.setFont(f)
            p.setPen(QColor(MUTED))
            text = f"Ngưỡng cảnh báo {self.threshold * 100:.1f}/100"
            tw = p.fontMetrics().horizontalAdvance(text)
            p.drawText(int(min(max(x - tw / 2, 0), w - tw)), 13, text)
        p.end()


class RiskResult(QWidget):
    """Khối kết quả dự đoán: điểm rủi ro lớn, nhãn kết luận, thanh điểm, dòng giải thích."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        row = QHBoxLayout()
        self.score = QLabel("–")
        self.score.setObjectName("scoreValue")
        self.badge = QLabel("Chưa có đơn")
        self.badge.setObjectName("badge")
        row.addWidget(self.score)
        row.addStretch(1)
        row.addWidget(self.badge, alignment=Qt.AlignmentFlag.AlignVCenter)
        lay.addLayout(row)
        self.bar = RiskBar()
        lay.addWidget(self.bar)
        self.detail = QLabel("Chọn một đơn hàng để xem điểm rủi ro.")
        self.detail.setObjectName("muted")
        self.detail.setWordWrap(True)
        lay.addWidget(self.detail)
        self._badge(GREY_SOFT, MUTED)

    def _badge(self, bg, fg):
        self.badge.setStyleSheet(f"background: {bg}; color: {fg};")

    def setResult(self, result, positive_text, negative_text, actual_text):
        if result is None:
            self.score.setText("–")
            self.badge.setText("Chưa đủ dữ liệu")
            self._badge(GREY_SOFT, MUTED)
            self.bar.setValue(None, None)
            self.detail.setText("Đơn này không thuộc tập dữ liệu đã tiền xử lý của mô hình nên không tính được điểm.")
            return
        self.score.setText(f"{result['score'] * 100:.1f}/100")
        self.bar.setValue(result["score"], result["threshold"])
        if result["label"] == 1:
            self.badge.setText(positive_text)
            self._badge(ORANGE_SOFT, ORANGE_DARK)
        else:
            self.badge.setText(negative_text)
            self._badge(GREEN_SOFT, GREEN)
        self.detail.setText("Điểm càng cao càng cần chú ý.")


def make_table():
    table = QTableWidget()
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(30)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.horizontalHeader().setHighlightSections(False)
    table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    table.customContextMenuRequested.connect(lambda pos: table_copy_menu(table, pos))
    return table


def table_copy_menu(table, pos):
    item = table.itemAt(pos)
    if item is None:
        return
    menu = QMenu(table)
    copy_value = menu.addAction("Sao chép giá trị")
    raw_id = item.data(Qt.ItemDataRole.UserRole)
    copy_original = menu.addAction("Sao chép mã gốc") if raw_id else None
    action = menu.exec(table.viewport().mapToGlobal(pos))
    if action == copy_value:
        QApplication.clipboard().setText(item.text())
    elif copy_original is not None and action == copy_original:
        QApplication.clipboard().setText(raw_id)


def set_original_id(item, raw_id):
    item.setData(Qt.ItemDataRole.UserRole, str(raw_id))
    item.setToolTip(f"Mã gốc: {raw_id}\nNhấp chuột phải để sao chép mã gốc.")


def fmt(value, decimals=4):
    if isinstance(value, bool):
        return "✓" if value else ""
    if isinstance(value, float):
        if value != value:
            return ""
        return f"{value:,.0f}" if value == int(value) and abs(value) >= 1000 else f"{value:.{decimals}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def fill_table(table, df, decimals=4, highlight_rows=(), stretch_last=True):
    table.setSortingEnabled(False)
    table.setRowCount(len(df))
    table.setColumnCount(len(df.columns))
    table.setHorizontalHeaderLabels([str(c) for c in df.columns])
    for r, row in enumerate(df.itertuples(index=False)):
        for c, value in enumerate(row):
            if hasattr(value, "item"):
                value = value.item()
            item = QTableWidgetItem(fmt(value, decimals))
            originals = df.attrs.get("raw_ids", {}).get(df.columns[c])
            if originals is not None:
                set_original_id(item, originals[r])
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if r in highlight_rows:
                item.setBackground(QColor(ORANGE_SOFT))
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            table.setItem(r, c, item)
    table.resizeColumnsToContents()
    table.horizontalHeader().setStretchLastSection(False)      # tắt rồi bật để Qt tính lại độ rộng cột cuối
    table.horizontalHeader().setStretchLastSection(stretch_last)


def fill_key_value(table, info, raw_ids=None):
    table.setRowCount(len(info))
    table.setColumnCount(2)
    table.setHorizontalHeaderLabels(["Thuộc tính", "Giá trị"])
    for r, (k, v) in enumerate(info.items()):
        key = QTableWidgetItem(k)
        key.setForeground(QColor(MUTED))
        table.setItem(r, 0, key)
        value = QTableWidgetItem(str(v))
        if raw_ids and k in raw_ids:
            set_original_id(value, raw_ids[k])
        table.setItem(r, 1, value)
    table.resizeColumnToContents(0)
    table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
