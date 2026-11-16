"""Chạy giao diện mới ở chế độ offscreen và chụp từng màn hình vào results/screenshots/v2."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "windows")   # offscreen không có font nên chữ thành ô vuông
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from PyQt6.QtWidgets import QApplication

from ui.LoginWindow import LoginWindow

OUT = ROOT / "results" / "screenshots" / "v2"
OUT.mkdir(parents=True, exist_ok=True)
app = QApplication(sys.argv)


def shot(widget, name):
    app.processEvents()
    widget.grab().save(str(OUT / f"{name}.png"))
    print("saved", name)


login = LoginWindow()
login.show()
shot(login, "0_login")
login.lineEditUser.setText("admin")
login.lineEditPassword.setText("sai")
login.login()
shot(login, "0_login_error")
login.lineEditPassword.setText("admin123")
login.login()
win = login.main
win.resize(1400, 860)
shot(win, "1_tong_quan_trong")

for page, name in ((1, "2_delivery_trong"), (2, "3_review_trong"), (3, "4_seller_trong")):
    win.goto(page)
    shot(win, name)
win.goto(0)
pred = win.predictor
late = pred.delivery_store.labels["test"].query("target == 1")["order_id"].iloc[0]
multi = pred.items.groupby("order_id")["seller_id"].nunique()
multi_id = next(o for o in multi[multi > 1].index if o in pred.delivery_store)
win.lookup(late)
shot(win, "1_tong_quan")
for key, page in (("delivery", 1), ("review", 2)):
    win.goto(page)
    tabs = win.ui[key]["tabs"]
    for i, name in enumerate(("cm", "pr", "sosanh")):
        tabs.setCurrentIndex(i)
        shot(win, f"{page + 1}_{key}_{name}")
win.lookup(multi_id)
win.goto(3)
win.runKMeans()
tabs = win.tabsSeller
for i, name in enumerate(("hoso", "chonk", "chatluong", "pca", "danhsach")):
    tabs.setCurrentIndex(i)
    shot(win, f"4_seller_{name}")
tabs.setCurrentIndex(3)
win.comboBoxView.setCurrentIndex(1)
shot(win, "4_seller_boxplot")
win.comboBoxK.setCurrentIndex(win.comboBoxK.findData(4))
win.runKMeans()
tabs.setCurrentIndex(2)
shot(win, "4_seller_k4")
print("OK")
