

## Tổng quan

Dự án dùng bộ dữ liệu thương mại điện tử Olist (Brazil) để xây dựng ứng dụng hỗ trợ vận hành đơn hàng, gồm ba chức năng học máy:

- **Cảnh báo giao hàng trễ** bằng XGBoost, khi đơn được xác nhận.
- **Cảnh báo đánh giá 1–2 sao** bằng Random Forest, sau khi giao hàng.
- **Phân nhóm nhà bán hàng** bằng K-Means (K = 2), kèm đề xuất nhà bán theo quy tắc nghiệp vụ.

Thư mục `OlistApp/` chứa toàn bộ mã nguồn (`ml/`, `ui/`, `test/`), dữ liệu (`data/`) và các mô hình đã huấn luyện (`models/`), nên clone về là chạy được.

## Yêu cầu

- Python 3.13 (đã chạy thử trên Windows)
- Các thư viện sau (đã liệt kê trong `OlistApp/requirements.txt`):

| Thư viện | Phiên bản | Dùng cho |
| --- | --- | --- |
| PyQt6 | >= 6.6 | Giao diện ứng dụng |
| numpy | mới nhất | Xử lý số liệu |
| pandas | mới nhất | Đọc và xử lý dữ liệu CSV |
| scipy | mới nhất | Ma trận thưa (.npz) |
| scikit-learn | 1.9.1 | Random Forest, K-Means |
| xgboost | mới nhất | Mô hình cảnh báo giao trễ |
| joblib | mới nhất | Nạp mô hình đã huấn luyện |
| matplotlib | mới nhất | Biểu đồ |

## Cài đặt và chạy

```powershell
git clone https://github.com/hantrinh1103-wq/MACHINE-LEARNING
cd MACHINE-LEARNING/OlistApp
python -m pip install -r requirements.txt
python test/run_app.py
```

Nếu báo `pip` không được nhận diện, dùng `python -m pip` như trên (hoặc `py -m pip`).

**Đăng nhập (demo):** `admin` / `admin123`
