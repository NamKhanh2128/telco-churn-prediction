# 4.1. Tiền xử lý dữ liệu và pipeline chống rò rỉ

> Bản nháp — Nam Khánh. Số liệu tham chiếu `reports/results_snapshot.md` (§ = mục trong snapshot);
> sẽ cập nhật theo tag `v1.0-results`.

## 4.1.1. Nguyên tắc: chia trước, học sau

**Rò rỉ dữ liệu (data leakage)** xảy ra khi thông tin của tập kiểm tra lọt vào quá trình huấn luyện,
khiến kết quả đánh giá đẹp hơn thực tế. Các lỗi điển hình trong bài toán churn là chuẩn hóa, điền
giá trị thiếu hoặc sinh mẫu SMOTE trên **toàn bộ** dữ liệu rồi mới chia train/test.

Nhóm áp dụng hai quy tắc:

1. **Chia dữ liệu trước mọi bước xử lý.**
2. **Mọi bước "học" từ dữ liệu** (điền khuyết, mã hóa, chuẩn hóa, SMOTE, calibration, tuning) đều nằm
   trong một `Pipeline` của scikit-learn và chỉ được `fit` trên tập train. Tập test chỉ dùng **một lần**
   để báo cáo.

## 4.1.2. Chia dữ liệu

| Tập | Tỉ lệ | Vai trò |
|---|---:|---|
| Train | 70% | Huấn luyện; tuning bằng 5-fold cross-validation trên chính tập này |
| Validation | 15% | Chọn ngưỡng quyết định theo chi phí (mục 5.3) |
| Test | 15% | Đánh giá cuối cùng, không dùng cho bất kỳ lựa chọn nào |

Phép chia có phân tầng theo nhãn (stratified) để giữ tỉ lệ churn ở cả ba tập, seed cố định 42.

**Iranian Churn — chia theo nhóm (group-aware).** Audit dữ liệu (`src/iranian_audit.py`) phát hiện
300 dòng trùng hoàn toàn, 314 dòng có vector đặc trưng trùng nhau (162 nhóm), trong đó 14 nhóm trùng
đặc trưng nhưng **khác nhãn**. Nếu chia ngẫu nhiên theo dòng, một bản sao nằm ở train và bản kia ở
test, mô hình chỉ cần "nhớ" là đoán đúng → điểm test bị thổi phồng. Vì vậy mỗi vector đặc trưng được
băm thành một mã nhóm và cả nhóm luôn nằm trọn trong một tập (`StratifiedGroupKFold`, tỉ lệ thực tế
≈ 71/14/14). Cross-validation khi tune trên Iranian cũng chia theo nhóm, để chọn tham số không dựa
trên điểm CV bị thổi phồng.

## 4.1.3. Các bước trong pipeline

```
[IBM] TelcoFeatureEngineer          ← đặc trưng tạo theo từng dòng, không học tham số
   ▼
ColumnTransformer
 ├── số:      SimpleImputer(median) → StandardScaler (chỉ cho LogReg, KNN, SVM)
 └── phân loại: SimpleImputer(most_frequent) → OneHotEncoder(drop="first", handle_unknown="ignore")
   ▼
[chỉ trong ablation A2] SMOTE        ← imblearn Pipeline: chỉ sinh mẫu khi fit, không đụng tập test
   ▼
Mô hình (→ CalibratedClassifierCV khi cần xác suất tin cậy)
```

**Đặc trưng tạo thêm cho IBM** (`src/preprocessing.py::TelcoFeatureEngineer`):

| Đặc trưng | Cách tính | Ý nghĩa |
|---|---|---|
| `charges_per_tenure` | MonthlyCharges / (tenure + 1) | Chi phí cao so với thời gian gắn bó |
| `total_additional_services` | Số dịch vụ cộng thêm đang dùng (0–6) | Mức độ "gắn" với hệ sinh thái |
| `streaming_services` | StreamingTV + StreamingMovies (0–2) | Nhu cầu giải trí |
| `security_support_services` | OnlineSecurity + TechSupport (0–2) | Được hỗ trợ kỹ thuật |
| `is_high_risk_contract` | Hợp đồng theo tháng **và** thanh toán Electronic check | Nhóm rủi ro cao từ EDA |
| `has_bundle` | Có cả Internet và điện thoại | Gói kết hợp |

Các đặc trưng này tính độc lập trên từng dòng (bước `fit` không học gì), nên không thể làm rò rỉ
thông tin giữa train và test.

**`TotalCharges` bị loại khỏi giao thức chính.** Cột này gần như bằng `tenure × MonthlyCharges`, tức
là thông tin trùng lặp. Ablation A1 giữ lại cột này và chỉ thấy thay đổi rất nhỏ (snapshot §3, dòng
`ibm_with_total_charges`). IBM có 11 ô `TotalCharges` trống, đều ở khách có `tenure = 0`; khi giữ cột
(A1), bộ điền khuyết trong pipeline xử lý mà không nhìn thấy tập test.

**Chuẩn hóa theo loại mô hình.** Cây quyết định và Random Forest chỉ so sánh giá trị với ngưỡng, nên
không bị ảnh hưởng bởi phép biến đổi đơn điệu như chuẩn hóa; ablation A3 xác nhận điều này (snapshot §6).
LogReg, KNN và SVM dựa trên khoảng cách hoặc độ lớn hệ số nên bắt buộc chuẩn hóa.

**Mất cân bằng lớp** (churn ≈ 15–27%) được xử lý bằng trọng số lớp (`class_weight`) thay vì SMOTE
trong giao thức chính. SMOTE chỉ dùng trong ablation A2 và luôn nằm trong `imblearn.pipeline.Pipeline`,
nên chỉ sinh mẫu trên fold train.

## 4.1.4. Tuning và calibration

- **Tuning:** `RandomizedSearchCV` với tối đa 30 tổ hợp, 5-fold CV trên tập train, tiêu chí chọn là
  **average precision (AP)** — phù hợp dữ liệu mất cân bằng hơn accuracy. Lưới tham số của mọi mô hình
  khai báo tập trung trong `MODEL_PARAM_GRIDS` (`src/compare_datasets.py`).
- **Calibration:** xác suất của Random Forest được hiệu chỉnh bằng Platt scaling (sigmoid) qua
  `CalibratedClassifierCV` với 5-fold CV **trên tập train**. Model deploy dùng `ensemble=False`
  (một rừng + một bộ hiệu chỉnh): chất lượng xác suất tương đương phương án 5 rừng nhưng nhỏ hơn khoảng
  4 lần (snapshot §8). Lý thuyết calibration ở mục 4.2.

## 4.1.5. Tái lập kết quả

| Yếu tố | Cách đảm bảo |
|---|---|
| Ngẫu nhiên | Seed cố định cho chia dữ liệu, tuning, mô hình |
| Dữ liệu | SHA-256 của từng file được ghi vào `metadata.json` |
| Môi trường | Phiên bản Python/scikit-learn ghi trong metadata |
| Quy trình | Một lệnh `python -m src.reproduce`: test → huấn luyện → benchmark 3 dataset → ablation → audit → các phân tích → xuất `reports/results_snapshot.md` (kèm mã commit git) |
| Phiên bản số liệu | Tag git `v1.0-results`; báo cáo chỉ dùng số trong snapshot của tag |

Khi chạy lại toàn bộ từ một commit sạch, mọi chỉ số, cấu hình model và hình đều trùng khớp tuyệt đối
với lần chạy trước — bằng chứng pipeline tái lập được.
