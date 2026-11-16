import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtWidgets import QApplication

from ui.LoginWindow import LoginWindow

app = QApplication(sys.argv)
login = LoginWindow()       # đăng nhập xong sẽ mở ui/AppWindow.py (tài khoản demo: admin / admin123)
login.show()
app.exec()
