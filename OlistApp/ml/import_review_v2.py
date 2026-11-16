"""Nhập kết quả mục 5.3 (161 cột, nhánh sau giao hàng) vào app, thay cho bản 131 cột trong train_review_post.py.

Đầu vào: thư mục kết quả của
    python olist_5_3_review_prediction.py train --bundle review_v2_training_bundle.joblib --output-dir <dir>
Chạy:   python ml/import_review_v2.py <dir>
Ghi:    data/review_post_*  ·  models/review_post_model.joblib  ·  results/review_post_*
Cấu hình mô hình, ngưỡng và ma trận lấy nguyên từ <dir>; app chỉ tính lại bảng chỉ số để hiển thị.
"""
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.paths import DATA_DIR, MODEL_DIR, RESULT_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BRANCH = "post_delivery"
MODELS = ["Random Forest", "XGBoost", "Logistic Regression", "Decision Tree"]
FILE_NAMES = {"Random Forest": "random_forest", "XGBoost": "xgboost",
              "Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree"}


def metrics(y, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"Ngưỡng": threshold, "Accuracy": accuracy_score(y, pred),
            "Precision lớp 1": precision_score(y, pred, zero_division=0),
            "Recall lớp 1": recall_score(y, pred, zero_division=0),
            "F1 lớp 1": f1_score(y, pred, zero_division=0), "ROC-AUC": roc_auc_score(y, score),
            "Average Precision": average_precision_score(y, score), "Tỷ lệ đơn bị cảnh báo": pred.mean(),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def main(src):
    src = Path(src)
    meta = json.loads((src / "run_metadata.json").read_text(encoding="utf-8"))
    thresholds = meta["thresholds"][BRANCH]
    selected = meta["selected_models"][BRANCH]

    # ma trận + nhãn (order_id, target) + tên đặc trưng
    for s in ("train", "validation", "test"):
        shutil.copy(src / "data" / f"{BRANCH}_{s}_X.npz", DATA_DIR / f"review_post_{s}_X.npz")
        lab = pd.read_csv(src / "data" / f"{BRANCH}_{s}_labels.csv")[["order_id", "target"]]
        lab.to_csv(DATA_DIR / f"review_post_{s}_labels.csv", index=False)
    names = pd.read_csv(src / "data" / f"{BRANCH}_feature_names.csv")["feature_name"]
    names.rename("Tên đặc trưng").to_csv(DATA_DIR / "review_post_feature_names.csv", index=False, encoding="utf-8-sig")

    # model đã chọn ở 5.3 (không huấn luyện lại)
    joblib.dump({"model": joblib.load(src / "models" / f"{BRANCH}_{FILE_NAMES[selected]}.joblib"),
                 "threshold": float(thresholds[selected]), "name": selected, "branch": BRANCH,
                 "feature_names": names.tolist(), "thresholds": thresholds},
                MODEL_DIR / "review_post_model.joblib", compress=3)

    # bảng so sánh: điểm dự đoán của 5.3 -> chỉ số tại ngưỡng chọn trên validation
    rows = []
    for split in ("validation", "test"):
        p = pd.read_csv(src / f"{BRANCH}_{split}_predictions.csv")
        part = [{"Mô hình": m, "Tập": split, **metrics(p["target"].to_numpy(), p[m].to_numpy(), thresholds[m])}
                for m in MODELS]
        part = pd.DataFrame(part)
        if split == "validation":
            part = part.sort_values("Average Precision", ascending=False)
        else:
            part["Được chọn"] = part["Mô hình"] == selected
            sel = p[["order_id", "target", selected]].rename(columns={selected: "score"})
            sel["prediction"] = (sel["score"] >= thresholds[selected]).astype(int)
            sel.to_csv(RESULT_DIR / "review_post_test_predictions.csv", index=False)
        rows.append(part)
    comparison = pd.concat(rows, ignore_index=True)
    comparison.to_csv(RESULT_DIR / "review_post_comparison.csv", index=False, encoding="utf-8-sig")

    n = {s: len(pd.read_csv(DATA_DIR / f"review_post_{s}_labels.csv")) for s in ("train", "validation", "test")}
    best_ap = comparison[comparison["Tập"] == "validation"].iloc[0]["Mô hình"]
    (RESULT_DIR / "review_post_summary.json").write_text(json.dumps(
        {"integrated": selected, "best_validation_ap": best_ap, "threshold": float(thresholds[selected]),
         "n_features": int(len(names)), "n_train": n["train"], "n_validation": n["validation"],
         "n_test": n["test"], "source": "5.3 (olist_5_3_review_prediction.py)"},
        ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
