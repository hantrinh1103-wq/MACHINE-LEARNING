"""Màn hình đăng nhập: nửa trái xanh đậm giới thiệu hệ thống, nửa phải là biểu mẫu."""
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (QApplication, QCheckBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                             QPushButton, QVBoxLayout, QWidget)

from ui import theme as th

# Tài khoản demo của đồ án (chưa nối cơ sở dữ liệu người dùng)
ACCOUNTS = {"admin": ("admin123", "Quản trị viên")}


class Avatar(QWidget):
    def __init__(self, size=84):
        super().__init__()
        self.setFixedSize(size, size)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = self.width()
        p.setPen(QPen(QColor(th.ORANGE), 3))
        p.setBrush(QColor(th.NAVY))
        p.drawEllipse(QRectF(2, 2, s - 4, s - 4))
        f = QFont("Segoe MDL2 Assets")
        f.setPixelSize(int(s * 0.45))
        p.setFont(f)
        p.setPen(QColor("#FFFFFF"))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, th.ICON_USER)
        p.end()


class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.main = None
        self.setWindowTitle("Smart E-commerce Order Management – Đăng nhập")
        self.setFixedSize(920, 580)
        self.setStyleSheet(th.QSS)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._brand_panel())
        root.addWidget(self._form_panel(), 1)

    # ------------------------------------------------------------ nửa trái
    def _brand_panel(self):
        panel = QFrame()
        panel.setObjectName("loginBrand")
        panel.setFixedWidth(400)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(44, 40, 40, 40)
        lay.setSpacing(0)
        lay.addStretch(1)                      # chỉ có tên đề tài, canh giữa theo chiều dọc
        bar = QFrame()
        bar.setFixedSize(56, 5)
        bar.setStyleSheet(f"background: {th.ORANGE}; border-radius: 2px;")
        lay.addWidget(bar)
        lay.addSpacing(22)
        logo = QLabel("Smart\nE-commerce\nOrder\nManagement")
        logo.setObjectName("loginLogo")
        lay.addWidget(logo)
        lay.addStretch(1)
        return panel

    # ------------------------------------------------------------ nửa phải
    def _form_panel(self):
        panel = QFrame()
        panel.setObjectName("loginForm")
        outer = QHBoxLayout(panel)
        outer.addStretch(1)
        box = QWidget()
        box.setFixedWidth(330)
        outer.addWidget(box)
        outer.addStretch(1)
        lay = QVBoxLayout(box)
        lay.setSpacing(6)
        lay.addStretch(1)
        lay.addWidget(Avatar(), alignment=Qt.AlignmentFlag.AlignHCenter)
        lay.addSpacing(6)
        title = QLabel("ĐĂNG NHẬP")
        title.setObjectName("loginTitle")
        lay.addWidget(title, alignment=Qt.AlignmentFlag.AlignHCenter)
        sub = QLabel("Đăng nhập để vào hệ thống quản lý đơn hàng")
        sub.setObjectName("muted")
        lay.addWidget(sub, alignment=Qt.AlignmentFlag.AlignHCenter)
        lay.addSpacing(14)

        lay.addWidget(self._label("Tên đăng nhập"))
        self.lineEditUser = QLineEdit()
        self.lineEditUser.addAction(th.glyph_icon(th.ICON_USER, th.NAVY), QLineEdit.ActionPosition.LeadingPosition)
        lay.addWidget(self.lineEditUser)
        lay.addSpacing(6)
        lay.addWidget(self._label("Mật khẩu"))
        self.lineEditPassword = QLineEdit()
        self.lineEditPassword.setEchoMode(QLineEdit.EchoMode.Password)
        self.lineEditPassword.addAction(th.glyph_icon(th.ICON_LOCK, th.NAVY),
                                        QLineEdit.ActionPosition.LeadingPosition)
        lay.addWidget(self.lineEditPassword)
        self.checkBoxShow = QCheckBox("Hiện mật khẩu")
        lay.addWidget(self.checkBoxShow)

        self.labelError = QLabel("")
        self.labelError.setObjectName("error")
        self.labelError.setWordWrap(True)
        self.labelError.setMinimumHeight(20)
        lay.addWidget(self.labelError)

        self.pushButtonLogin = QPushButton("Đăng nhập")
        self.pushButtonLogin.setObjectName("primary")
        self.pushButtonLogin.setMinimumHeight(42)
        self.pushButtonLogin.setCursor(Qt.CursorShape.PointingHandCursor)
        lay.addWidget(self.pushButtonLogin)
        lay.addStretch(1)

        self.pushButtonLogin.clicked.connect(self.login)
        self.lineEditUser.returnPressed.connect(self.lineEditPassword.setFocus)
        self.lineEditPassword.returnPressed.connect(self.login)
        self.lineEditUser.textEdited.connect(lambda: self.labelError.setText(""))
        self.lineEditPassword.textEdited.connect(lambda: self.labelError.setText(""))
        self.checkBoxShow.toggled.connect(lambda on: self.lineEditPassword.setEchoMode(
            QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        return panel

    def _label(self, text):
        lab = QLabel(text)
        lab.setObjectName("fieldLabel")
        return lab

    # ------------------------------------------------------------ xử lý
    def login(self):
        user, password = self.lineEditUser.text().strip(), self.lineEditPassword.text()
        if not user or not password:
            self.labelError.setText("Nhập đủ tên đăng nhập và mật khẩu.")
            return
        if user not in ACCOUNTS or ACCOUNTS[user][0] != password:
            self.labelError.setText("Tên đăng nhập hoặc mật khẩu chưa đúng. Kiểm tra lại rồi thử lần nữa.")
            self.lineEditPassword.selectAll()
            self.lineEditPassword.setFocus()
            return
        self.pushButtonLogin.setEnabled(False)
        self.pushButtonLogin.setText("Đang nạp mô hình và dữ liệu…")
        QApplication.processEvents()
        try:
            from ui.AppWindow import AppWindow
            self.main = AppWindow(user, ACCOUNTS[user][1])
        except Exception as error:
            QMessageBox.critical(self, "Không nạp được mô hình/dữ liệu", str(error))
            return
        finally:
            self.pushButtonLogin.setEnabled(True)
            self.pushButtonLogin.setText("Đăng nhập")
        self.main.loggedOut.connect(self.show_again)
        self.main.show()
        self.hide()

    def show_again(self):
        self.main = None
        self.labelError.setText("")
        self.lineEditPassword.clear()
        self.show()
        self.lineEditPassword.setFocus()
