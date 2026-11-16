# Ứng dụng PyQt6 – Smart E-commerce Order Management

## Chạy nhanh

```powershell
git clone https://github.com/hantrinh1103-wq/MACHINE-LEARNING
cd MACHINE-LEARNING/OlistApp
python -m pip install -r requirements.txt
python test/run_app.py
```

**Tài khoản đăng nhập (demo):**

| Tên đăng nhập | Mật khẩu |
| --- | --- |
| `admin` | `admin123` |

Thư mục đã có sẵn dữ liệu gốc (`data/raw`), dữ liệu đã tiền xử lý và các model đã huấn luyện, nên clone về là chạy được, không cần tải thêm. Đã chạy thử với Python 3.13 trên Windows.

## Mô hình tích hợp

Ứng dụng tích hợp ba mô hình được chọn ở mục 5.5.2 của báo cáo:

| Chức năng | Mô hình tích hợp | Nguồn |
| --- | --- | --- |
| Cảnh báo giao hàng trễ (khi đơn được xác nhận) | XGBoost, ngưỡng 0,5111 chọn theo F1 trên validation | `Olist_5_2/outputs/models/xgboost/` (mục 5.2) → `models/delivery_model.joblib` |
| Cảnh báo đánh giá 1–2 sao (sau khi giao, trước phản hồi đầu tiên) | Random Forest, nhánh sau giao hàng | `ml/import_review_v2.py` (từ kết quả 5.3) → `models/review_post_model.joblib` |
| Phân nhóm nhà bán hàng | K-Means, K = 2 (Silhouette cao nhất) | `ml/seller_clustering.py`, làm lại đúng `CODE/5.4.ipynb` → `models/seller_kmeans.joblib` |


Màu và widget dùng chung nằm ở `ui/theme.py`. Ảnh chụp từng màn hình: `python test/shot_new_ui.py` → `results/screenshots/v2/` (tự tạo khi chạy).

## Cấu trúc

```text
OlistApp/
  data/        ma trận X (.npz) và nhãn của 3 tập: delivery_* (giao trễ, 128 cột), review_post_* (đánh giá thấp sau giao, 161 cột); raw/ (6 CSV gốc); display_ids/ (bảng mã hiển thị)
  models/      delivery_model.joblib, review_post_model.joblib, seller_kmeans.joblib
  results/     bảng so sánh, dự đoán test, kết quả K-Means
  ml/          paths.py, predictor.py, seller_clustering.py, import_review_v2.py (nhập model 5.3),
               display.py (mã hiển thị)
  ui/          theme.py, LoginWindow.py, AppWindow.py
  test/        run_app.py (chạy app), shot_new_ui.py (chụp màn hình)
```

Dữ liệu gốc được đọc từ `data/raw/*.csv` (6 tệp: customers, orders, order_items, order_reviews, products, sellers), nên có thể chép riêng thư mục `OlistApp` đi chạy.

## Luồng sử dụng nghiệp vụ

- **Việc cần theo dõi**: danh sách cảnh báo trên tập kiểm thử lịch sử, nhấp đúp để tra cứu đơn. Số đếm không phải đơn đang xử lý trực tiếp.
- **Theo dõi giao hàng / Chăm sóc sau giao**: xem điểm cảnh báo, thông tin đối chiếu và gợi ý xử lý. Chỉ số và biểu đồ kỹ thuật nằm trong tab **Đánh giá mô hình**.
- **Hiệu quả nhà bán**: chọn ưu tiên đúng hẹn, giao nhanh hoặc đánh giá tốt; lọc theo bang, tìm mã nhà bán; giữ Ctrl chọn 2–3 dòng rồi bấm so sánh. Phân tích K-Means nằm trong tab chuyên sâu.

Đề xuất nhà bán dùng quy tắc nghiệp vụ trên lịch sử, không huấn luyện mô hình mới. Chỉ đề xuất khi có ít nhất 20 đơn hợp lệ; tiêu chí đánh giá cần thêm ít nhất 10 đánh giá. Chỉ số xếp hạng được điều chỉnh theo công thức `(giá trị × số mẫu + trung bình chung × 20) / (số mẫu + 20)`, rồi ưu tiên giá trị thấp hơn của tỷ lệ trễ, thời gian giao hoặc tỷ lệ đánh giá thấp. Khi bằng nhau ưu tiên nhiều đơn hơn. Bảng hiển thị số liệu gốc, phần đề xuất hiển thị chỉ số sau điều chỉnh. Các ngưỡng là chính sách minh họa, không phải chứng nhận chất lượng.

Lịch sử nhà bán tổng hợp toàn kỳ, không dùng làm đặc trưng mới cho dự đoán đơn. Đề xuất chưa xét sản phẩm, tồn kho, giá bán hiện tại hoặc địa chỉ nhận cụ thể. Điểm cảnh báo /100 không phải xác suất đã hiệu chỉnh. Thông tin đối chiếu không phải giải thích đóng góp đặc trưng hay kết luận nguyên nhân; nếu cần giải thích từng dự đoán phải bổ sung phương pháp phân tích phù hợp với mô hình.

Kiểm tra: `python -m unittest test.test_seller_selection -v`; kiểm tra các luồng Qt trên Windows: `python test/smoke_business_ui.py`.

## Chạy lại mô hình (tùy chọn)

```powershell
python ml/seller_clustering.py        # K-Means theo 5.4, thêm --k 4 để ép số cụm
python ml/import_review_v2.py <thư mục kết quả 5.3>   # nhập Random Forest nhánh sau giao (161 cột) từ olist_5_3_review_prediction.py
```

## Ghi chú

- Giao diện dùng mã đơn `ORD-000001`, mã nhà bán `SEL-0001` và tên bang đầy đủ (Brazil).
  Ô tra cứu đơn nhận cả mã ngắn và mã gốc; bộ lọc nhà bán nhận mã ngắn, mã gốc, tên bang có/không dấu hoặc viết tắt.
  Di chuột lên mã trong bảng để xem mã gốc; nhấp chuột phải để sao chép mã hiển thị hoặc mã gốc.
  Bảng ánh xạ được lưu tự động trong `data/display_ids/orders.csv` và `sellers.csv`.
  Giữ hai tệp này khi sao chép/cập nhật dữ liệu để mã hiển thị ổn định; mã trong CSV gốc và mô hình được giữ nguyên.

- K-Means cho ra đúng số liệu mục 5.4: 1.766 nhà bán, C1 = 1.393, C2 = 373, Silhouette 0,3125, ARI lấy mẫu lại 0,9477. Nhà bán dưới 5 đơn hoàn tất được ghi "Chưa đủ dữ liệu phân nhóm".
- Mô hình đánh giá thấp lấy từ kết quả `olist_5_3_review_prediction.py train --bundle review_v2_training_bundle.joblib` (161 cột, gồm 18 đặc trưng lịch sử nhà bán), nên khớp bảng 5.3: Random Forest nhánh sau giao, ngưỡng 0,3215, AP validation 0,2760. Số dòng 38.963 / 23.729 / 26.958 (train / validation / test). Chạy 5.3 cần scikit-learn 1.6.1 (bản của bundle); app đọc model này bằng scikit-learn mới hơn vẫn cho đúng điểm dự đoán nhưng có cảnh báo phiên bản.
- Đặc trưng của một đơn lấy từ ma trận đã tiền xử lý theo `order_id`; đơn chưa giao hoặc ngoài tập sẽ báo "không có đặc trưng".
