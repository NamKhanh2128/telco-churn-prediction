# Hướng dẫn kỹ thuật cho nhóm

> Tài liệu cho buổi hướng dẫn code (Nam Khánh, trước 28/11). Mục tiêu: sau buổi này **mỗi người chạy
> được pipeline trên máy mình và biết bắt đầu phần code của mình từ đâu**.

## 1. Cài đặt (10 phút)

```bash
git clone https://github.com/NamKhanh2128/telco-churn-prediction.git
cd telco-churn-prediction
python -m venv venv
venv\Scripts\activate                 # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
git checkout <branch-của-bạn>          # xem bảng ở PLAN.md mục 1
```

Dữ liệu **không nằm trong git**. Đặt 3 file CSV đúng đường dẫn trong `PLAN.md` mục 7 (bảng SHA-256),
hoặc chạy `python -m src.download_data` khi Quỳnh Anh hoàn thành script.

Kiểm tra môi trường:

```bash
python -m pytest -q              # tất cả test phải pass
python -m src.reproduce --quick  # chạy nhanh toàn bộ (~10 phút), phải báo "completed"
```

Chế độ `--quick` ghi **mọi** kết quả vào `results/quick/` (bị ignore): không đổi file nào đang được
git theo dõi, nên chạy thử bao nhiêu lần cũng được. Snapshot của lần chạy nhanh nằm ở
`results/quick/results/results_snapshot_quick.md`.

## 2. Code nằm ở đâu, file nào làm gì

| File | Vai trò | Phụ trách |
|---|---|---|
| `src/config.py` | Đường dẫn, seed, tỉ lệ chia 70/15/15, lưới tham số, chi phí `COST_*`, `CALIBRATION_ENSEMBLE` | theo phần (PLAN.md mục 6) |
| `src/compare_datasets.py` | Benchmark 3 dataset: `load_dataset`, `split_data`, `BENCHMARK_MODELS`, `_build_model`, `MODEL_PARAM_GRIDS`, `run_dataset`, `fit_deployed_rf` | Nam Khánh (phần DT/KNN/SVM: Thùy Linh) |
| `models/tuned_params.json` | Tham số tốt nhất của lần tune chính thức (có trong git), mọi phân tích đọc từ đây | sinh tự động bởi `--tune` |
| `src/preprocessing.py` | `TelcoFeatureEngineer`, `build_preprocessor`, `build_model_pipeline` | Nam Khánh |
| `src/train.py` | IBM: 7 mô hình + RF đã calibrate để deploy + SHAP | Nam Khánh |
| `src/evaluate.py` | `compute_metrics`: Accuracy, Precision, Recall, F1, ROC-AUC, AP, Brier, ECE, lift | Nam Khánh |
| `src/optimize_threshold.py` | `find_cost_optimal_threshold`: ngưỡng theo chi phí | Nam Khánh |
| `src/explain.py` | SHAP, `explain_customer` (giải thích từng khách cho API) | Nam Khánh |
| `src/ablation.py`, `src/model_stability.py`, `src/calibration_analysis.py`, `src/rf_analysis.py` | Các phân tích cho mục 5.1–5.3 | Nam Khánh |
| `src/policy_analysis.py` | Phân tích ngưỡng/chi phí/top-k | Nhật Linh |
| `src/download_data.py` | Tải và kiểm tra dữ liệu | Quỳnh Anh |
| `src/reproduce.py`, `src/export_results.py` | Chạy lại toàn bộ → `reports/results_snapshot.md` | Nam Khánh |
| `api/`, `frontend/`, `Dockerfile` | Demo | Hải Anh |
| `tests/` | Mỗi người một file `tests/test_<mảng>.py` | từng người |

## 3. Luồng dữ liệu và quy tắc chống rò rỉ

```
load_dataset ─► _split_data (70/15/15; Iranian chia theo nhóm dòng trùng)
                  │
     train ───────┼─► pipeline.fit:  feature_engineer → preprocessor (imputer, one-hot, scaler) → [SMOTE] → model
     validation ──┼─► chọn ngưỡng (không bao giờ chọn trên test)
     test ────────┴─► đánh giá MỘT lần cuối
```

**Quy tắc vàng:** mọi thứ "học" từ dữ liệu (imputer, scaler, encoder, SMOTE, calibration, tuning)
phải nằm **trong pipeline** và chỉ `fit` trên train. Không bao giờ `fit` trên toàn bộ dữ liệu rồi
mới chia.

## 4. Kết quả nằm ở đâu

| Thư mục | Nội dung | Có trong git? |
|---|---|---|
| `results/` | CSV/JSON trung gian, sinh lại được | Không |
| `figures/` | Hình cho báo cáo (`figures/rf/`, `figures/eda/`…) | Có (thư mục con) |
| `reports/results_snapshot.md` | **Số liệu chính thức** để đưa vào báo cáo | Có — chỉ Nam Khánh commit |
| `models/` | `config.json`, `metadata.json`, `tuned_params.json` (model `.joblib` bị ignore) | Có (trừ `.joblib`) |
| `results/quick/` | Toàn bộ đầu ra của `reproduce --quick` | Không |

Chạy **từng script** (`python -m src.train`, `python -m src.compare_datasets`, …) sẽ ghi vào thư mục
chính thức, làm đổi `models/*.json`, `figures/rf/`, `figures/dataset_runs/`. Hai cách tránh:

```bash
# Cách 1: cho script ghi vào thư mục nháp (PowerShell: $env:CHURN_OUTPUT_ROOT="results/scratch";
# nhớ xóa biến này khi xong: Remove-Item Env:CHURN_OUTPUT_ROOT). Snapshot của sandbox ghi ở
# results/scratch/results/results_snapshot_quick.md, không đụng reports/.
CHURN_OUTPUT_ROOT=results/scratch python -m src.compare_datasets --dataset ibm

# Cách 2: lỡ chạy rồi thì khôi phục trước khi commit
git restore models figures/rf figures/dataset_runs reports/results_snapshot.md
```

## 5. Bắt đầu phần code của bạn

### Thùy Linh — thêm DT/KNN/SVM vào benchmark (`src/compare_datasets.py`)

1. Thêm tên mô hình vào `BENCHMARK_MODELS`.
2. Thêm nhánh trong `_build_model`, ngay chỗ có chú thích *"Comparison models (Thùy Linh)"*
   (lấy `src/train.py` bước 4 làm mẫu; nhớ import lớp mô hình và `DEFAULT_*_PARAMS`):

```python
elif name == 'KNN':
    classifier = KNeighborsClassifier(**DEFAULT_KNN_PARAMS)
    scale_numeric = True                      # KNN/SVM bắt buộc chuẩn hóa
elif name == 'SVM_RBF':
    classifier = _calibrated(SVC(**DEFAULT_SVM_RBF_PARAMS))   # Platt, giống train.py
    scale_numeric = True
```

3. Thêm lưới vào `MODEL_PARAM_GRIDS`, ví dụ `'KNN': KNN_PARAM_GRID`. Khung tự đổi tiền tố
   `classifier__` ↔ `classifier__estimator__` khi mô hình được bọc `CalibratedClassifierCV`.
4. `python -m pytest tests/test_experiments.py -q`: các test "contract" tự kiểm tra **mọi** mô hình
   trong `BENCHMARK_MODELS` (fit được, `predict_proba` trong [0, 1], khóa lưới đúng với pipeline).
5. Chạy `python -m src.compare_datasets --dataset all --tune` → xem
   `results/dataset_runs/model_comparison_long.csv`; tham số tốt nhất được ghi vào
   `models/tuned_params.json`. Chỉ chạy lại một vài mô hình: `--models KNN SVM_RBF` (RF và LogReg luôn
   chạy kèm). Độ ổn định 10 seed (`src.model_stability`) và bảng calibration tự nhận mô hình mới.
6. Test riêng của bạn trong `tests/test_models.py`.

### Nhật Linh — `src/policy_analysis.py`

```python
from src.compare_datasets import fit_deployed_rf
from src.config import COST_FP
from src.optimize_threshold import find_cost_optimal_threshold

run = fit_deployed_rf('ibm', seed=42)          # đúng model deploy: RF đã tune + calibration
p_val = run['model'].predict_proba(run['X_val'])[:, 1]    # chọn ngưỡng trên validation
p_test = run['model'].predict_proba(run['X_test'])[:, 1]  # đánh giá trên test
for ratio in [2, 3, 5, 10, 20]:                # tỉ lệ chi phí FN/FP cần thử
    result = find_cost_optimal_threshold(
        run['y_val'], p_val, cost_fp=COST_FP, cost_fn=COST_FP * ratio, verbose=False,
    )
    print(ratio, result['best_threshold'], result['savings'])
```

Lặp qua các tỉ lệ FN/FP, các giá trị top-k và các seed (`fit_deployed_rf('ibm', seed=s)`); ghi CSV
vào `RESULTS_DIR / 'policy'`, hình vào `FIGURES_DIR / 'policy'` (import từ `src.config`, **không**
viết cứng `'results/...'`: khi chạy `--quick`, hai biến này trỏ vào sandbox). Test trong
`tests/test_policy.py`.

**Nối vào pipeline chung không cần sửa file của Khánh:**
- `src/reproduce.py` tự chạy `python -m src.policy_analysis` khi file tồn tại (danh sách
  `OPTIONAL_STEPS`); nếu script có cờ `--quick` thì chế độ nhanh sẽ truyền cờ này.
- Ghi một đoạn Markdown vào `RESULTS_DIR / 'policy' / 'snapshot.md'` (bắt đầu bằng tiêu đề `### ...`);
  `src.export_results` tự chèn vào snapshot, mục *10. Additional analyses*.

### Quỳnh Anh — `src/download_data.py`

```python
import urllib.request, zipfile
from src.data_loader import get_file_sha256   # hàm tính SHA-256 có sẵn

ORANGE_URL = 'https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/telecom_churn.csv'
# Iranian: lấy link tải chính thức trên trang UCI dataset 563 (file .zip chứa "Customer Churn.csv")

def verify(path, expected_sha256):
    return path.exists() and get_file_sha256(path) == expected_sha256
```

Tải về thư mục tạm, kiểm tra SHA-256 rồi mới chép vào `data/raw/`. Test hàm `verify` trong
`tests/test_download.py` bằng một file nhỏ tạo trong `tmp_path` (không cần mạng).
`src.reproduce` chỉ **ghi lại** SHA-256 của dữ liệu đã dùng (vào `results/reproduce_log.json` và đầu
snapshot); việc **kiểm tra** khớp bảng trong `PLAN.md` là của `download_data.py`.

### Hải Anh — giải thích từng khách trong API

```python
from src.explain import explain_customer
factors = explain_customer(_pipeline, df, top_n=5)
# [{'feature': 'tenure', 'value': '3', 'contribution': 0.052,
#   'direction': 'increases_churn', 'impact': 'high', 'source_features': ['tenure']}, ...]
```

Chạy được với model đã calibrate đang deploy (khoảng 0.5 giây mỗi khách). Trả về đúng các khóa
`feature`, `value`, `impact` mà `ChurnFactor` trong `api/schemas.py` đang dùng, cộng thêm hướng tác động
và `source_features`: với đặc trưng tạo thêm (ví dụ `charges_per_tenure`) đây là các cột gốc khách đã nhập
(`MonthlyCharges`, `tenure`), dùng để hiển thị cho dễ hiểu. `contribution` đo trên điểm của rừng
**trước** calibration nên không cộng ra đúng xác suất API trả về; thứ tự và chiều tác động là của điểm
rừng, thường (không phải luôn luôn) trùng với giải thích trên xác suất đã calibrate. Nếu API vẫn muốn
định dạng ngắn `{feature, value, impact}` chỉ gồm cột gốc, dùng
`get_top_factors_for_customer(_pipeline, df, feature_columns)` (đã chạy được với model calibrate).

## 6. Viết test

- Đặt trong `tests/test_<mảng>.py`; dùng dữ liệu giả nhỏ (xem `tests/test_experiments.py::_toy`).
- Test phải chạy nhanh (< vài giây) và **không cần mạng**.
- Chạy riêng một file: `python -m pytest tests/test_models.py -q`.

## 7. Git và PR

```bash
git fetch origin && git merge origin/main     # cập nhật từ main, ít nhất 2 lần/tuần
python -m pytest -q                           # phải pass
git add <đúng-file-của-mình>                  # không dùng "git add -A"
git commit -m "Add KNN and SVM to the 3-dataset benchmark"
git push
```

Mở PR vào `main` trên GitHub. Checklist trước khi gửi: test pass; không commit dữ liệu, model,
`results/`; không commit `models/*.json`, `figures/rf/`, `figures/dataset_runs/` do pipeline tự sinh
(mục 4); mô tả PR ghi rõ đã chạy lệnh gì.

## 8. Lỗi hay gặp

| Lỗi | Cách xử lý |
|---|---|
| `FileNotFoundError: Dataset not found` | Chép CSV vào đúng đường dẫn (PLAN.md mục 7) |
| SHA-256 không khớp | Tải lại file đúng nguồn |
| `ValueError: Unknown model 'X'` | Thêm nhánh cho mô hình trong `_build_model` |
| `WARNING: no tuned parameters for ...` | Chưa có `models/tuned_params.json` cho run đó: lấy bản mới từ `main`, hoặc chạy `python -m src.compare_datasets --dataset all --tune` |
| `warning: LF will be replaced by CRLF` | Bình thường trên Windows, bỏ qua |
| `ModuleNotFoundError: imblearn` | `pip install -r requirements.txt` |
| Commit lỡ kèm `models/config.json`… | `git restore` như mục 4 trước khi commit |
