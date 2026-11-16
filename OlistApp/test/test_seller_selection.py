import unittest

import pandas as pd

from ml.seller_selection import rank_sellers


class SellerSelectionTests(unittest.TestCase):
    def setUp(self):
        self.history = pd.DataFrame({
            "seller_id": ["small", "steady", "slow", "missing"],
            "completed_order_count": [5, 100, 100, 0],
            "reviewed_order_count": [5, 80, 8, 0],
            "late_order_rate": [0., .05, .3, float("nan")],
            "mean_delivery_days": [1., 5., 10., float("nan")],
            "low_review_rate": [0., .1, 0., float("nan")],
        })

    def test_few_orders_do_not_outrank_supported_seller(self):
        ranked = rank_sellers(self.history)
        self.assertEqual(ranked.iloc[0]["seller_id"], "steady")
        self.assertFalse(ranked.set_index("seller_id").loc["small", "eligible"])

    def test_review_requires_enough_reviews(self):
        ranked = rank_sellers(self.history, "Đánh giá tốt").set_index("seller_id")
        self.assertFalse(ranked.loc["slow", "eligible"])
        self.assertTrue(ranked.loc["steady", "eligible"])

    def test_shrinkage_and_missing_data(self):
        ranked = rank_sellers(self.history).set_index("seller_id")
        self.assertGreater(ranked.loc["small", "adjusted"], 0.)
        self.assertFalse(ranked.loc["missing", "eligible"])

    def test_speed_and_empty_input(self):
        self.assertEqual(rank_sellers(self.history, "Giao nhanh").iloc[0]["seller_id"], "steady")
        self.assertTrue(rank_sellers(self.history.iloc[:0]).empty)


if __name__ == "__main__":
    unittest.main()
