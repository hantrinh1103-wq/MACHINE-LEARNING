"""Đường dẫn dùng chung cho toàn bộ ứng dụng (tương đối theo vị trí file)."""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]          # .../CODE/OlistApp
DATA_DIR = APP_DIR / "data"                              # ma trận đã tiền xử lý (giao trễ: checkpoint 4.5; đánh giá thấp: kết quả 5.3)
MODEL_DIR = APP_DIR / "models"                           # model đã huấn luyện
RESULT_DIR = APP_DIR / "results"                         # bảng, hình kết quả
RAW_DIR = DATA_DIR / "raw"                               # 6 CSV Olist gốc (nằm trong app để chạy độc lập)

for _d in (MODEL_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
