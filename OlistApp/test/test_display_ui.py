"""Integration check: python -m unittest discover -s test -p test_display_ui.py"""
import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import QApplication
from ui.AppWindow import AppWindow
from ml.display import state_label


class DisplayUiTests(unittest.TestCase):
    def test_lookup_filter_and_original_ids(self):
        app = QApplication.instance() or QApplication([])
        # The Windows offscreen plugin does not automatically discover system fonts.
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for name in ("segoeui.ttf", "segoeuib.ttf", "segmdl2.ttf"):
            if (fonts / name).exists():
                QFontDatabase.addApplicationFont(str(fonts / name))
        app.setFont(QFont("Segoe UI", 10))
        win = AppWindow()
        try:
            p = win.predictor
            original = win._sample_orders()[0][0]
            short = p.order_codes.display(original)
            win.lookup(original)
            expected = p.predict_delivery(original)
            win.lookup(short.lower())
            self.assertEqual(win.order_id, original)
            self.assertTrue(all(edit.currentText() == short for edit in win.orderEdits))
            self.assertEqual(p.predict_delivery(win.order_id), expected)
            item = win.ui["delivery"]["info"].item(0, 1)
            self.assertEqual(item.text(), short)
            self.assertEqual(item.data(Qt.ItemDataRole.UserRole), original)
            sellers = win.sellersOfOrder(original)
            self.assertTrue(all(value.startswith("SEL-") for value in sellers.iloc[:, 0]))
            raw_seller = win.tableOrderSellers.item(0, 0).data(Qt.ItemDataRole.UserRole)
            self.assertIn(raw_seller, p.sellers.index)
            sid = win.cluster["clusters"]["seller_id"].iloc[0]
            for query in (sid, p.seller_codes.display(sid).lower()):
                win.lineEditSellerFilter.setText(query)
                self.assertEqual(win.tableSellers.rowCount(), 1)
                self.assertEqual(win.tableSellers.item(0, 0).data(Qt.ItemDataRole.UserRole), sid)
            for query in ("SP", "Sao Paulo", "AC", "BA"):
                win.lineEditSellerFilter.setText(query)
                code = "SP" if query == "Sao Paulo" else query
                expected_count = (win.cluster["clusters"]["seller_state"] == code).sum()
                self.assertEqual(win.tableSellers.rowCount(), expected_count)
                self.assertTrue(all(win.tableSellers.item(i, 1).text() == state_label(code)
                                    for i in range(win.tableSellers.rowCount())))
            win.lineEditSellerFilter.setText("no-such-seller")
            self.assertEqual(win.tableSellers.rowCount(), 0)
            win.lineEditSellerFilter.setText("")
            with_items = set(p.items["order_id"])
            empty = next(o for o in p.orders.index if o not in with_items)
            self.assertTrue(win.sellersOfOrder(empty).empty)
            out = Path(__file__).resolve().parents[1] / "results/screenshots/display_ids"
            out.mkdir(parents=True, exist_ok=True)
            win.resize(1500, 950)
            win.show()
            app.processEvents()
            self.assertTrue(win.grab().save(str(out / "orders.png")))
            win.goto(3)
            win.stackSeller.setCurrentIndex(1)
            win.tabsSeller.setCurrentIndex(4)
            app.processEvents()
            self.assertTrue(win.grab().save(str(out / "sellers.png")))
        finally:
            win.close()


if __name__ == "__main__":
    unittest.main()
