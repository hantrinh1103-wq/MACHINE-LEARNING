"""Kiểm tra luồng nghiệp vụ bằng Qt, không mở cửa sổ tương tác."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "windows")

from PyQt6.QtCore import QItemSelectionModel
from PyQt6.QtWidgets import QApplication
from ui.AppWindow import AppWindow

app = QApplication([])
w = AppWindow()
w.resize(1400, 900)
assert len(w.rankedSellers) == len(w.predictor.sellers)
assert w.rankedSellers["eligible"].any()
for criterion in range(3):
    w.sellerCriterion.setCurrentIndex(criterion)
    model = w.sellerRanking.selectionModel()
    for row in range(2):
        model.select(w.sellerRanking.model().index(row, 0),
                     QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
    w.compareSelectedSellers()
    assert w.sellerComparison.rowCount() == 2
w.sellerSearch.setText("NO_SUCH_SELLER")
assert w.sellerRanking.rowCount() == 0
w.sellerSearch.clear()
w.sellerState.setCurrentIndex(1)
assert w.rankedSellers["seller_state"].eq(w.sellerState.currentData()).all()
w.sellerState.setCurrentIndex(0)
order_id = w._sample_orders()[0][0]
w.lookup(order_id)
assert w.ui["delivery"]["guidance"].text()
assert "Chưa đủ" in w._guidance("review", None, order_id)
for i, name in enumerate(("overview", "delivery", "review", "seller_selection")):
    w.goto(i)
    app.processEvents()
    w.grab().save(str(Path("results") / f"{name}_preview.png"))
w.close()
print("Business UI smoke checks passed", flush=True)
