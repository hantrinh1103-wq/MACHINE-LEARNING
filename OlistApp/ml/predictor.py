"""
Lớp OrderPredictor: nạp model đã huấn luyện và dữ liệu, trả kết quả cho giao diện.

Ba mô hình tích hợp theo mục 5.5.2 của báo cáo:
- Giao trễ: XGBoost của mục 5.2 (models/delivery_model.joblib), ngưỡng F1 trên validation, tính khi đơn được xác nhận.
- Đánh giá thấp: Random Forest nhánh sau giao hàng của mục 5.3, 161 cột (ml/import_review_v2.py -> models/review_post_model.joblib).
- Cụm nhà bán: K-Means K = 2 của ml/seller_clustering.py (results/seller_clusters.csv).
Đặc trưng của một đơn được lấy từ ma trận đã tiền xử lý theo order_id: giao trễ từ checkpoint 4.5, đánh giá thấp từ kết quả 5.3.
"""
import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from ml.paths import DATA_DIR, MODEL_DIR, RESULT_DIR, RAW_DIR
from ml.display import DisplayIds, state_label

SPLITS = ("train", "validation", "test")
SPLIT_VI = {"train": "huấn luyện", "validation": "xác thực", "test": "kiểm thử"}


class FeatureStore:
    """Giữ ma trận X của 3 tập và bảng tra order_id -> (tập, dòng)."""

    def __init__(self, task):
        self.X, self.labels, index = {}, {}, []
        for s in SPLITS:
            self.X[s] = sparse.load_npz(DATA_DIR / f"{task}_{s}_X.npz").tocsr()
            lab = pd.read_csv(DATA_DIR / f"{task}_{s}_labels.csv")
            self.labels[s] = lab
            index.append(pd.DataFrame({"order_id": lab["order_id"], "split": s, "row": np.arange(len(lab))}))
        self.index = pd.concat(index).set_index("order_id")

    def __contains__(self, order_id):
        return order_id in self.index.index

    def row(self, order_id):
        loc = self.index.loc[order_id]
        return self.X[loc["split"]][int(loc["row"])], loc["split"], int(self.labels[loc["split"]]["target"].iloc[int(loc["row"])])


class OrderPredictor:
    def __init__(self):
        d = joblib.load(MODEL_DIR / "delivery_model.joblib")
        self.delivery_model, self.delivery_threshold = d["model"], float(d["threshold"])
        self.delivery_name = {"xgboost": "XGBoost", "logistic_regression": "Logistic Regression"}.get(
            d.get("family"), str(d.get("family")))
        r = joblib.load(MODEL_DIR / "review_post_model.joblib")
        self.review_model, self.review_threshold, self.review_name = r["model"], float(r["threshold"]), r["name"]
        self.delivery_store = FeatureStore("delivery")
        self.review_store = FeatureStore("review_post")

        self.orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv").set_index("order_id")
        self.customers = pd.read_csv(RAW_DIR / "olist_customers_dataset.csv").set_index("customer_id")
        self.items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
        self.sellers = pd.read_csv(RAW_DIR / "olist_sellers_dataset.csv").set_index("seller_id")
        self.products = pd.read_csv(RAW_DIR / "olist_products_dataset.csv").set_index("product_id")
        reviews = pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv")
        self.review_score = reviews.groupby("order_id")["review_score"].agg(lambda s: s.iloc[-1])
        self.clusters = pd.read_csv(RESULT_DIR / "seller_clusters.csv").set_index("seller_id")
        self.kmeans = joblib.load(MODEL_DIR / "seller_kmeans.joblib")
        self.order_codes = DisplayIds(DATA_DIR / "display_ids" / "orders.csv", self.orders.index, "ORD", 6)
        self.seller_codes = DisplayIds(DATA_DIR / "display_ids" / "sellers.csv", self.sellers.index, "SEL", 4)

    # ---------- tra cứu ----------
    def order_ids(self):
        return self.delivery_store.index.index.tolist()

    def random_order_id(self, rng=None):
        rng = rng or np.random.default_rng()
        return str(rng.choice(self.order_ids()))

    def order_info(self, order_id):
        if order_id not in self.orders.index:
            raise KeyError(f"Không tìm thấy đơn {order_id} trong olist_orders_dataset.csv")
        o = self.orders.loc[order_id]
        c = self.customers.loc[o["customer_id"]]
        it = self.items[self.items["order_id"] == order_id]
        info = {
            "Mã đơn hàng": self.order_codes.display(order_id),
            "Trạng thái": {"delivered": "Đã giao", "shipped": "Đang vận chuyển", "processing": "Đang xử lý",
                           "approved": "Đã xác nhận", "created": "Mới tạo", "invoiced": "Đã lập hóa đơn",
                           "canceled": "Đã hủy", "unavailable": "Không thể đáp ứng"}.get(o["order_status"], o["order_status"]),
            "Thời điểm đặt": o["order_purchase_timestamp"],
            "Thời điểm xác nhận": o["order_approved_at"],
            "Ngày giao dự kiến": str(o["order_estimated_delivery_date"])[:10],
            "Ngày giao thực tế": str(o["order_delivered_customer_date"])[:10],
            "Bang khách hàng (Brazil)": state_label(c['customer_state']),
            "Thành phố khách hàng": str(c['customer_city']).title(),
            "Số dòng hàng": len(it),
            "Số nhà bán": it['seller_id'].nunique(),
            "Tiền hàng": f"{it['price'].sum():.2f}",
            "Phí vận chuyển": f"{it['freight_value'].sum():.2f}",
            "Điểm sao thực tế": str(self.review_score.get(order_id, "chưa có")),
        }
        return info

    # ---------- dự đoán ----------
    def _predict(self, store, model, threshold, order_id):
        if order_id not in store:
            return None
        x, split, actual = store.row(order_id)
        col = int(np.flatnonzero(model.classes_ == 1)[0])
        score = float(model.predict_proba(x)[:, col][0])
        return {"score": score, "threshold": threshold, "label": int(score >= threshold),
                "split": SPLIT_VI[split], "actual": actual}

    def predict_delivery(self, order_id):
        return self._predict(self.delivery_store, self.delivery_model, self.delivery_threshold, order_id)

    def predict_review(self, order_id):
        return self._predict(self.review_store, self.review_model, self.review_threshold, order_id)

    def sellers_of_order(self, order_id):
        it = self.items[self.items["order_id"] == order_id]
        rows = []
        for sid, g in it.groupby("seller_id"):
            s = self.sellers.loc[sid]
            if sid in self.clusters.index:
                c = self.clusters.loc[sid]
                cluster, name = c["cluster"], c["cluster_name"]
                extra = (f"{int(c['completed_order_count'])} đơn, trễ {c['late_order_rate']:.1%}, "
                         f"giao TB {c['mean_delivery_days']:.1f} ngày, 1–2 sao {c['low_review_rate']:.1%}")
            else:
                cluster, name, extra = "-", "Chưa đủ dữ liệu phân nhóm (dưới 5 đơn hoàn tất)", ""
            rows.append({"seller_id": sid, "Bang": state_label(s["seller_state"]), "Số dòng hàng": len(g),
                         "Tiền hàng": round(g["price"].sum(), 2), "Cụm": cluster, "Tên cụm": name,
                         "Lịch sử seller": extra})
        return pd.DataFrame(rows, columns=["seller_id", "Bang", "Số dòng hàng", "Tiền hàng",
                                           "Cụm", "Tên cụm", "Lịch sử seller"])
