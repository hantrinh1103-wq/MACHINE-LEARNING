"""
Phân nhóm nhà bán hàng bằng K-Means, làm lại đúng notebook CODE/5.4.ipynb (mục 5.4 của báo cáo).

- Dùng toàn bộ giai đoạn dữ liệu (kể cả 2018); chỉ lấy đơn đã giao và hợp lệ về thời gian.
- Giữ nhà bán có ít nhất 5 đơn hoàn tất đủ điều kiện (1.766 nhà bán).
- 5 đặc trưng; log1p cho số đơn và giá trị sản phẩm TB/đơn; điền thiếu bằng trung vị; StandardScaler.
- Thử K = 2..10 (k-means++, n_init=50, max_iter=500, random_state=42); chọn K có Silhouette cao nhất.
- Đối chiếu thêm Davies–Bouldin, Calinski–Harabasz, ARI giữa 5 seed và ARI qua 30 lần lấy mẫu 80%.

Chạy:  python ml/seller_clustering.py [--k 2]
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_samples, silhouette_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.paths import MODEL_DIR, RAW_DIR, RESULT_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MIN_ORDERS = 5
RANDOM_STATE = 42
K_VALUES = list(range(2, 11))
SEEDS = [7, 19, 42, 71, 101]
N_REPEATS, SAMPLE_FRACTION = 30, 0.8
# tên cột -> (nhãn hiển thị, mô tả khi cụm cao, mô tả khi cụm thấp)
FEATURES = {
    "completed_order_count": ("Số đơn hoàn tất", "Nhiều đơn", "Ít đơn"),
    "mean_product_sales_per_order": ("Giá trị sản phẩm TB/đơn", "Giá trị đơn cao", "Giá trị đơn thấp"),
    "late_order_rate": ("Tỷ lệ giao trễ", "Giao trễ nhiều", "Giao trễ ít"),
    "mean_delivery_days": ("Thời gian giao TB", "Giao chậm", "Giao nhanh"),
    "low_review_rate": ("Tỷ lệ đánh giá thấp", "Nhiều đánh giá thấp", "Ít đánh giá thấp"),
}
FEATURE_COLS = list(FEATURES)
LOG_FEATURES = ["completed_order_count", "mean_product_sales_per_order"]
PROFILE_COLS = FEATURE_COLS + ["mean_review_score"]
# Tên hai nhóm theo mục 5.4.4 của báo cáo (K = 2)
NAME_GOOD = "Giao nhận và đánh giá thuận lợi hơn"
NAME_REVIEW = "Cần ưu tiên rà soát giao nhận"

_cache = {}


# ------------------------------------------------------------------ dữ liệu
def build_seller_data(include_all=False):
    """Tổng hợp đặc trưng theo nhà bán từ 4 CSV gốc (ô 4 và 6 của notebook)."""
    orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv")
    reviews = pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv")
    items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
    sellers = pd.read_csv(RAW_DIR / "olist_sellers_dataset.csv")
    for col in ["order_purchase_timestamp", "order_approved_at", "order_delivered_customer_date",
                "order_estimated_delivery_date"]:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
    for col in ["review_creation_date", "review_answer_timestamp"]:
        reviews[col] = pd.to_datetime(reviews[col], errors="coerce")

    latest_reviews = (reviews.sort_values(["order_id", "review_answer_timestamp", "review_creation_date",
                                           "review_id"], kind="stable", na_position="first")
                      .drop_duplicates("order_id", keep="last"))
    om = orders.merge(latest_reviews[["order_id", "review_score", "review_answer_timestamp"]],
                      on="order_id", how="left", validate="one_to_one")
    approved, purchase = om["order_approved_at"], om["order_purchase_timestamp"]
    actual, estimated = om["order_delivered_customer_date"], om["order_estimated_delivery_date"]
    valid = (om["order_status"].eq("delivered") & approved.notna() & actual.notna() & estimated.notna()
             & approved.ge(purchase) & estimated.dt.normalize().ge(approved.dt.normalize()))
    ho = om.loc[valid].copy()
    ho["delivery_days"] = (ho["order_delivered_customer_date"] - ho["order_approved_at"]).dt.total_seconds() / 86400
    ho.loc[ho["delivery_days"].lt(0), "delivery_days"] = np.nan
    ho["late_delivery"] = (ho["order_delivered_customer_date"].dt.normalize()
                           > ho["order_estimated_delivery_date"].dt.normalize()).astype(int)
    known = ho["review_score"].isin([1, 2, 3, 4, 5]) & ho["review_answer_timestamp"].notna()
    ho["eligible_review_score"] = ho["review_score"].where(known)
    ho["eligible_low_review"] = np.where(known, ho["review_score"].le(2).astype(int), np.nan)

    pairs = items[["seller_id", "order_id"]].drop_duplicates()
    hist = pairs.merge(ho[["order_id", "late_delivery", "delivery_days", "eligible_review_score",
                           "eligible_low_review"]], on="order_id", how="inner", validate="many_to_one")
    perf = hist.groupby("seller_id", as_index=False).agg(
        completed_order_count=("order_id", "nunique"),
        late_order_rate=("late_delivery", "mean"),
        mean_delivery_days=("delivery_days", "mean"),
        reviewed_order_count=("eligible_review_score", "count"),
        mean_review_score=("eligible_review_score", "mean"),
        low_review_rate=("eligible_low_review", "mean"))
    sales = (items.loc[items["order_id"].isin(ho["order_id"])]
             .groupby("seller_id", as_index=False).agg(total_product_sales=("price", "sum")))
    seller_all = (sellers[["seller_id", "seller_state"]]
                  .merge(perf, on="seller_id", how="left", validate="one_to_one")
                  .merge(sales, on="seller_id", how="left", validate="one_to_one"))
    counts = ["completed_order_count", "reviewed_order_count", "total_product_sales"]
    seller_all[counts] = seller_all[counts].fillna(0)
    denom = seller_all["completed_order_count"].replace(0, np.nan)
    seller_all["mean_product_sales_per_order"] = seller_all["total_product_sales"] / denom
    seller_all["review_coverage_rate"] = seller_all["reviewed_order_count"] / denom
    seller_data = seller_all.loc[seller_all["completed_order_count"].ge(MIN_ORDERS)].copy().reset_index(drop=True)
    coverage = pd.DataFrame({
        "Nhóm": ["Toàn bộ nhà bán", "Không có đơn đủ điều kiện", f"Có từ 1 đến {MIN_ORDERS - 1} đơn",
                 "Đủ điều kiện phân nhóm"],
        "Số nhà bán": [len(seller_all), int(seller_all["completed_order_count"].eq(0).sum()),
                       int(seller_all["completed_order_count"].between(1, MIN_ORDERS - 1).sum()),
                       len(seller_data)]})
    return (seller_all if include_all else seller_data), coverage, len(ho)


def transform_features(frame):
    result = frame[FEATURE_COLS].copy().replace([np.inf, -np.inf], np.nan)
    for col in LOG_FEATURES:
        result[col] = np.log1p(result[col])
    return result


def make_preprocessor():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])


def make_kmeans(k, n_init=50, seed=RANDOM_STATE):
    return KMeans(n_clusters=k, init="k-means++", n_init=n_init, max_iter=500, random_state=seed)


def prepare():
    """Dữ liệu nhà bán + ma trận đã chuẩn hóa (tính một lần, dùng lại cho giao diện)."""
    if not _cache:
        seller_data, coverage, n_orders = build_seller_data()
        preprocessor = make_preprocessor()
        X = preprocessor.fit_transform(transform_features(seller_data))
        assert np.isfinite(X).all()
        _cache.update(seller_data=seller_data, coverage=coverage, n_orders=n_orders,
                      preprocessor=preprocessor, X=X)
    return _cache


# ------------------------------------------------------------------ chọn K
def evaluate_k(X):
    rows, models = [], {}
    for k in K_VALUES:
        model = make_kmeans(k)
        labels = model.fit_predict(X)
        sizes = np.bincount(labels, minlength=k)
        by_seed = [make_kmeans(k, n_init=20, seed=s).fit(X).labels_ for s in SEEDS]
        ari = [adjusted_rand_score(a, b) for a, b in combinations(by_seed, 2)]
        rows.append({"K": k, "Inertia": model.inertia_, "Silhouette": silhouette_score(X, labels),
                     "Davies-Bouldin": davies_bouldin_score(X, labels),
                     "Calinski-Harabasz": calinski_harabasz_score(X, labels),
                     "Cụm nhỏ nhất": int(sizes.min()), "Cụm lớn nhất": int(sizes.max()),
                     "ARI seed trung bình": float(np.mean(ari)), "ARI seed thấp nhất": float(np.min(ari))})
        models[k] = model
    return pd.DataFrame(rows), models


def name_clusters(centroids):
    """K = 2: đặt tên theo mục 5.4.4. K khác: ghép 2 đặc trưng lệch nhiều nhất so với trung bình."""
    if len(centroids) == 2:
        worse = centroids["late_order_rate"].idxmax()
        return {c: (NAME_REVIEW if c == worse else NAME_GOOD) for c in centroids.index}
    names = {}
    for c, row in centroids.iterrows():
        order = row.abs().sort_values(ascending=False).index
        n = 2
        while True:
            name = ", ".join(FEATURES[f][1] if row[f] > 0 else FEATURES[f][2] for f in order[:n])
            if name not in names.values() or n == len(order):
                break
            n += 1
        names[c] = name
    return names


def fit(k, model=None):
    """Huấn luyện K-Means với K cho trước và trả về mọi bảng cần cho báo cáo/giao diện."""
    data = prepare()
    seller_data, X = data["seller_data"], data["X"]
    model = model or make_kmeans(k).fit(X)
    labels = model.labels_
    codes = [f"C{i + 1}" for i in range(k)]
    centroids = pd.DataFrame(model.cluster_centers_, columns=FEATURE_COLS, index=codes)
    names = name_clusters(centroids)
    sil = silhouette_samples(X, labels)

    clusters = seller_data.copy()
    clusters["cluster_id"] = labels
    clusters["cluster"] = [codes[i] for i in labels]
    clusters["cluster_name"] = clusters["cluster"].map(names)
    clusters["silhouette"] = sil

    g = clusters.groupby("cluster")
    sizes = g.size()
    means = g[PROFILE_COLS].mean()
    means.insert(0, "Tỷ trọng (%)", sizes / len(clusters) * 100)
    means.insert(0, "Số nhà bán", sizes)
    means["Tên nhóm"] = pd.Series(names)
    means.index.name = "Cụm"
    medians = g[PROFILE_COLS].median()
    cluster_quality = pd.DataFrame({
        "Cụm": codes, "Số nhà bán": sizes.reindex(codes).to_numpy(),
        "Silhouette trung bình": g["silhouette"].mean().reindex(codes).to_numpy(),
        "Tỷ lệ Silhouette âm": g["silhouette"].apply(lambda v: v.lt(0).mean()).reindex(codes).to_numpy()})
    quality = {"K": k, "Số nhà bán": len(clusters), "Inertia": float(model.inertia_),
               "Silhouette": float(silhouette_score(X, labels)),
               "Davies-Bouldin": float(davies_bouldin_score(X, labels)),
               "Calinski-Harabasz": float(calinski_harabasz_score(X, labels)),
               "Số nhà bán Silhouette âm": int((sil < 0).sum()), "Tỷ lệ Silhouette âm": float((sil < 0).mean())}
    pca = PCA(n_components=2)
    xy = pca.fit_transform(X)
    return {"k": k, "model": model, "codes": codes, "names": names, "clusters": clusters, "means": means,
            "medians": medians, "centroids": centroids, "quality": quality,
            "cluster_quality": cluster_quality, "pca": xy, "pca_var": pca.explained_variance_ratio_}


def stability(labels, k):
    """30 lần lấy 80% nhà bán, học lại tiền xử lý + K-Means, gán cụm cho toàn bộ và so bằng ARI."""
    seller_data = prepare()["seller_data"]
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for rep in range(N_REPEATS):
        idx = rng.choice(len(seller_data), size=int(SAMPLE_FRACTION * len(seller_data)), replace=False)
        pre = make_preprocessor()
        sample_X = pre.fit_transform(transform_features(seller_data.iloc[idx]))
        m = make_kmeans(k, n_init=30, seed=RANDOM_STATE + rep).fit(sample_X)
        reassigned = m.predict(pre.transform(transform_features(seller_data)))
        rows.append({"Lần chạy": rep + 1, "ARI": adjusted_rand_score(labels, reassigned)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ chạy và lưu
def run(k=None):
    data = prepare()
    k_report, models = evaluate_k(data["X"])
    best_k = int(k_report.loc[k_report["Silhouette"].idxmax(), "K"])
    k = k or best_k
    result = fit(k, models[k])
    stab = stability(result["model"].labels_, k)

    k_report.to_csv(RESULT_DIR / "kmeans_k_selection.csv", index=False, encoding="utf-8-sig")
    stab.to_csv(RESULT_DIR / "kmeans_stability.csv", index=False, encoding="utf-8-sig")
    data["coverage"].to_csv(RESULT_DIR / "seller_coverage.csv", index=False, encoding="utf-8-sig")
    result["clusters"].to_csv(RESULT_DIR / "seller_clusters.csv", index=False, encoding="utf-8-sig")
    result["means"].to_csv(RESULT_DIR / "cluster_profile.csv", encoding="utf-8-sig")
    result["medians"].to_csv(RESULT_DIR / "cluster_medians.csv", encoding="utf-8-sig")
    result["centroids"].to_csv(RESULT_DIR / "cluster_centroids_standardized.csv", encoding="utf-8-sig")
    result["cluster_quality"].to_csv(RESULT_DIR / "kmeans_silhouette_by_cluster.csv", index=False,
                                     encoding="utf-8-sig")
    joblib.dump({"model": result["model"], "preprocessor": data["preprocessor"], "features": FEATURE_COLS,
                 "log1p_features": LOG_FEATURES, "k": k, "best_silhouette_k": best_k,
                 "names": result["names"], "quality": result["quality"],
                 "ari_mean": float(stab["ARI"].mean()), "eligible_orders": data["n_orders"]},
                MODEL_DIR / "seller_kmeans.joblib")
    return data, k_report, best_k, result, stab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None, help="ép số cụm thay cho K có Silhouette cao nhất")
    args = parser.parse_args()
    data, k_report, best_k, result, stab = run(args.k)
    pd.set_option("display.width", 220)
    print(f"Số đơn đủ điều kiện: {data['n_orders']:,}")
    print(data["coverage"].to_string(index=False))
    print("\n", k_report.round(4).to_string(index=False))
    print(f"\nK có Silhouette cao nhất: {best_k}; K dùng: {result['k']}")
    print(pd.Series(result["quality"]).round(4).to_string())
    print("\n", result["cluster_quality"].round(4).to_string(index=False))
    print("\nTrung bình giữa các nhà bán:\n", result["means"].round(4).to_string())
    print("\nTrung vị:\n", result["medians"].round(4).to_string())
    print("\nTâm cụm (chuẩn hóa):\n", result["centroids"].round(2).to_string())
    print("\nARI qua lấy mẫu lại:\n", stab["ARI"].describe().round(4).to_string())


if __name__ == "__main__":
    main()
