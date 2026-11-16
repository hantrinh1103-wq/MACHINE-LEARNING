"""Run with: python -m unittest discover -s test -p test_display_ids.py"""
import tempfile
import unittest
from pathlib import Path

from ml.display import DisplayIds, search_text, state_label


class DisplayIdsTests(unittest.TestCase):
    def test_persist_reorder_append_and_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "orders.csv"
            first = DisplayIds(path, ["ccc", "bbb"], "ORD", 6)
            original_mapping = first.by_original.copy()
            again = DisplayIds(path, ["bbb", "ccc"], "ORD", 6)
            self.assertEqual(again.by_original, original_mapping)
            expanded = DisplayIds(path, ["aaa", "bbb"], "ORD", 6)
            self.assertEqual(expanded.display("bbb"), first.display("bbb"))
            self.assertEqual(expanded.display("aaa"), "ORD-000003")
            restored = DisplayIds(path, ["ccc", "aaa", "bbb"], "ORD", 6)
            self.assertEqual(restored.display("ccc"), first.display("ccc"))
            for original, display in restored.by_original.items():
                self.assertEqual(restored.resolve(display.lower()), original)
                self.assertEqual(restored.resolve(original), original)
            self.assertEqual(len(set(restored.by_original.values())), 3)

    def test_reject_duplicate_display_code(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sellers.csv"
            path.write_text("original_id,display_id\na,SEL-0001\nb,SEL-0001\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                DisplayIds(path, ["a", "b"], "SEL", 4)

    def test_state_names_and_search(self):
        self.assertEqual(state_label("sp"), "São Paulo (SP)")
        self.assertIn(search_text("sao paulo"), search_text(state_label("SP")))
        self.assertEqual(state_label("XX"), "XX")
        self.assertEqual(state_label(float("nan")), "Chưa có thông tin")


if __name__ == "__main__":
    unittest.main()
