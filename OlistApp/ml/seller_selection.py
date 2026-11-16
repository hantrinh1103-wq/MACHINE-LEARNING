"""Quy tắc so sánh lịch sử nhà bán; không thay đổi mô hình dự đoán."""
import numpy as np
import pandas as pd

PRIOR_ORDERS = 20
MIN_ORDERS = 20
MIN_REVIEWS = 10
CRITERIA = {
    "Đúng hẹn": ("late_order_rate", False),
    "Giao nhanh": ("mean_delivery_days", False),
    "Đánh giá tốt": ("low_review_rate", False),
}


def rank_sellers(history, criterion="Đúng hẹn"):
    """Co số liệu ít mẫu về trung bình chung trước khi xếp theo tiêu chí.

    Chỉ đề xuất khi có >=20 đơn, và >=10 đánh giá cho tiêu chí đánh giá.
    Các ngưỡng là quy tắc nghiệp vụ, không phải độ tin cậy thống kê.
    """
    df = history.copy()
    metric, ascending_good = CRITERIA[criterion]
    n = df["reviewed_order_count"] if metric == "low_review_rate" else df["completed_order_count"]
    valid = df[metric].notna() & n.gt(0)
    baseline = np.average(df.loc[valid, metric], weights=n[valid]) if valid.any() else np.nan
    df["adjusted"] = (df[metric] * n + baseline * PRIOR_ORDERS) / (n + PRIOR_ORDERS)
    df["eligible"] = df["completed_order_count"].ge(MIN_ORDERS) & valid
    if metric == "low_review_rate":
        df["eligible"] &= df["reviewed_order_count"].ge(MIN_REVIEWS)
    return df.sort_values(["eligible", "adjusted", "completed_order_count", "seller_id"],
                          ascending=[False, not ascending_good, False, True], na_position="last")
