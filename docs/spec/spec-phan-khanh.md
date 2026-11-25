# Spec tri thức — Phần của Nguyễn Đức Nam Khánh

**Random Forest (phương pháp chính) và khung thí nghiệm của nhóm**

> Tài liệu tổng hợp *mọi thứ* thuộc phần của Nam Khánh: làm gì, vì sao, code ở đâu, chạy thế nào,
> kết quả nói gì, các bạn khác nối vào ra sao và việc gì còn phải làm đến ngày bảo vệ.
> Số liệu lấy từ `reports/results_snapshot.md` (gọi tắt "snapshot §n"). Khi có tag `v1.0-results`,
> mọi con số trong báo cáo phải lấy từ snapshot của tag đó.
>
> Cập nhật lần cuối: 25/11/2026 (sau đợt sửa lỗi review và chạy lại toàn bộ).

## Mục lục

1. [Vai trò và phạm vi](#1-vai-trò-và-phạm-vi)
2. [Bức tranh tổng thể](#2-bức-tranh-tổng-thể)
3. [Giao thức thực nghiệm và lý do chọn](#3-giao-thức-thực-nghiệm-và-lý-do-chọn)
4. [Từng module: làm gì, vào/ra, lưu ý](#4-từng-module-làm-gì-vàora-lưu-ý)
5. [Artifact và snapshot](#5-artifact-và-snapshot)
6. [Kết quả hiện tại và cách diễn giải](#6-kết-quả-hiện-tại-và-cách-diễn-giải)
7. [Hợp đồng tích hợp với từng thành viên](#7-hợp-đồng-tích-hợp-với-từng-thành-viên)
8. [Kiểm thử và tái lập](#8-kiểm-thử-và-tái-lập)
9. [Hạn chế đã biết](#9-hạn-chế-đã-biết)
10. [Việc còn lại — runbook theo ngày](#10-việc-còn-lại--runbook-theo-ngày)
11. [Kiến thức nền cho phần bảo vệ](#11-kiến-thức-nền-cho-phần-bảo-vệ)
12. [Lệnh hay dùng](#12-lệnh-hay-dùng)
13. [Nhật ký thay đổi 25/11](#13-nhật-ký-thay-đổi-2511)

---

## 1. Vai trò và phạm vi

### 1.1. Mục tiêu (theo `plans/namkhanh.md`)

1. Chứng minh và giải thích vì sao Random Forest (RF) là phương pháp chính của đề tài.
2. Xây khung thí nghiệm chung để các bạn cắm mô hình/phân tích của mình vào; tự làm các thí nghiệm
   lõi: ablation, độ ổn định nhiều seed kèm kiểm định thống kê, calibration, phân tích RF.
3. Mọi con số tái lập được bằng **một lệnh** (`python -m src.reproduce`), gắn với tag
   `v1.0-results`, và báo cáo dùng đúng các số đó.

### 1.2. Vùng file phụ trách

| Nhóm | File |
|---|---|
| Khung thí nghiệm, mô hình chính | `src/train.py`, `src/compare_datasets.py` (trừ các nhánh DT/KNN/SVM trong `_build_model` — của Thùy Linh), `src/preprocessing.py`, `src/data_loader.py`, `src/evaluate.py`, `src/optimize_threshold.py`, `src/explain.py`, `src/iranian_audit.py` |
| Phân tích | `src/ablation.py`, `src/model_stability.py`, `src/calibration_analysis.py`, `src/rf_analysis.py`, `notebooks/02_random_forest_analysis.ipynb`, `figures/rf/` |
| Tái lập | `src/reproduce.py`, `src/export_results.py`, `reports/results_snapshot.md`, `models/tuned_params.json`, `models/config.json`, `models/metadata.json` |
| Cấu hình | Trong `src/config.py`: phần đường dẫn/sandbox, `DEFAULT_RF_PARAMS`, `RF_PARAM_GRID`, `CALIBRATION_*`, giao thức chia dữ liệu |
| Test | `tests/test_experiments.py`, `tests/test_pipeline.py` |
| Tài liệu | `docs/huong-dan-ky-thuat.md`, `docs/report/04-1-tien-xu-ly.md`, `04-2-random-forest.md`, `05-2-phan-tich-rf-ablation.md`, `docs/slides/namkhanh-random-forest.md`, `docs/defense_qa/namkhanh.md`, tài liệu này |

### 1.3. Vùng **không** đụng vào (để các bạn tích hợp dễ)

| Thành viên | Vùng của bạn ấy | Khánh chỉ làm |
|---|---|---|
| Hải Anh | `api/`, `frontend/`, `Dockerfile`, `docker-compose.yml`, `requirements*.txt`, `.github/`, `README.md`, `PLAN.md`, `mustdo.md`; `skipif` cho test cần dữ liệu thật (PR CI của Hải Anh) | Cung cấp model, `config.json`, `metadata.json`, `explain_customer` |
| Quỳnh Anh | `src/download_data.py`, `tests/test_download.py`, `notebooks/01_*`, `figures/eda/`, `docs/data_dictionary.md` | Review PR; `reproduce` chỉ *ghi* SHA-256, việc *kiểm tra* hash là của `download_data.py` |
| Thùy Linh | Nhánh DT/KNN/SVM trong `_build_model`, `DEFAULT_*_PARAMS`/`*_PARAM_GRID` của DT/KNN/SVM, `tests/test_models.py`, `notebooks/03_*`, `figures/models/` | Để sẵn chỗ chèn có chú thích; test "contract" tự kiểm tra mô hình mới |
| Nhật Linh | `src/policy_analysis.py`, `tests/test_policy.py`, `COST_*` trong `config.py`, `notebooks/04_*`, `figures/policy/` | `fit_deployed_rf`, `OPTIONAL_STEPS` trong reproduce, cơ chế fragment snapshot |

---

## 2. Bức tranh tổng thể

```
data/raw/*.csv  (3 file, không có trong git; SHA-256 trong PLAN.md mục 7)
   │
   ▼ load_dataset(name)                      src/compare_datasets.py (IBM còn có src/data_loader.py)
X, y  ─► split_data(seed)  70/15/15, phân tầng; Iranian chia theo nhóm dòng trùng
   │
   ├─ train (70%) ──► Pipeline.fit: [TelcoFeatureEngineer] → ColumnTransformer → mô hình
   │                   └─ --tune: RandomizedSearchCV (30 tổ hợp, 5-fold, tiêu chí AP)
   │                        └─ tham số tốt nhất → models/tuned_params.json  (có trong git)
   │                  RF đã tune ──► make_deployed_model: Platt sigmoid, ensemble=False
   ├─ validation (15%) ─► chọn ngưỡng theo chi phí (find_cost_optimal_threshold)
   └─ test (15%) ──────► báo cáo, không dùng cho lựa chọn nào
                           │
                           ▼
results/ (bị ignore) ──► src/export_results.py ──► reports/results_snapshot.md (có trong git)
models/rf_best.joblib + config.json + metadata.json ──► API của Hải Anh
```

Ba bộ dữ liệu được chạy **riêng**, không gộp: IBM Telco (7043 khách, churn 26.5%, nhiều biến
phân loại), Orange Telecom (3333 khách, churn 14.5%, hành vi cuộc gọi), Iranian Churn (3150 khách,
churn 15.7%, tần suất sử dụng, khiếu nại, gói cước). IBM là dataset chính hiện tại (quyết định D1
trong `PLAN.md`, có thể đổi sang Iranian sau khi hỏi giảng viên).

---

## 3. Giao thức thực nghiệm và lý do chọn

| Quyết định | Cụ thể | Lý do |
|---|---|---|
| Chia dữ liệu | 70/15/15 phân tầng theo nhãn, seed 42 (`TEST_SIZE = VALIDATION_SIZE = 0.15`) | Train để học + tune bằng CV; validation để chọn ngưỡng/calibration; test chỉ để báo cáo |
| Iranian | Chia theo nhóm: mỗi vector đặc trưng giống hệt nhau là một nhóm (`feature_groups`), cả nhóm nằm trọn một tập (`StratifiedGroupKFold`); tỉ lệ thực tế ≈ 71/14/14 | 300 dòng trùng hoàn toàn, 14 nhóm trùng đặc trưng nhưng khác nhãn: tránh để bản sao của một khách nằm cả ở train và test |
| Chống rò rỉ | Mọi bước học từ dữ liệu (imputer, one-hot, scaler, SMOTE, calibration, tuning) nằm trong `Pipeline`, chỉ `fit` trên train | Quy tắc vàng, xem mục 4.1 của báo cáo |
| Đặc trưng IBM | 6 đặc trưng tạo thêm (`TelcoFeatureEngineer`), bỏ `TotalCharges` | `TotalCharges ≈ tenure × MonthlyCharges` (trùng thông tin); ablation A1 kiểm chứng |
| Mất cân bằng | `class_weight="balanced_subsample"` cho RF; không dùng SMOTE trong giao thức chính | Đơn giản, không sinh dữ liệu giả; ablation A2 so sánh |
| Chỉ số chính | **AP (PR-AUC)**; kèm Precision/Recall/F1 lớp churn, ROC-AUC, Brier, ECE, lift@10% | Accuracy gây hiểu lầm khi mất cân bằng (Dummy đạt ~0.73 trên IBM) |
| Tuning | `RandomizedSearchCV`: `n_iter = min(30, số tổ hợp)`, 5-fold (Iranian: chia theo nhóm), `refit="average_precision"`, seed 42 | Lưới RF có 216 tổ hợp; 30 tổ hợp ngẫu nhiên đủ tốt mà rẻ |
| Tune một lần | Tham số tune trên train của seed 42, ghi vào `models/tuned_params.json`, mọi phân tích dùng lại | Tiết kiệm thời gian; hạn chế ghi ở mục 9 |
| Calibration | Platt (sigmoid), 5-fold CV trên train, `ensemble=False` (`CALIBRATION_ENSEMBLE`), RF được `clone` trước khi bọc | Xác suất đúng nghĩa để chọn ngưỡng; một rừng thay vì năm (file nhỏ, chất lượng như nhau — snapshot §8) |
| Ngưỡng | Chọn trên validation, tối thiểu chi phí `FP·COST_FP + FN·COST_FN` (`cost_mode="classification"`) | Chi phí bỏ sót khách (FN) lớn hơn chi phí gửi ưu đãi (FP); `COST_*` hiện là giả định |
| Kiểm định | t có hiệu chỉnh Nadeau–Bengio (chính), Wilcoxon, McNemar; RF vs mô hình mạnh nhất còn lại | 10 lần chia có tập train chồng nhau → t thường quá lạc quan |

---

## 4. Từng module: làm gì, vào/ra, lưu ý

### 4.1. `src/config.py` (phần của Khánh)

- **Đường dẫn và sandbox:** `OUTPUT_ROOT` = giá trị biến môi trường `CHURN_OUTPUT_ROOT` nếu có, ngược
  lại là gốc repo. `MODELS_DIR`, `FIGURES_DIR`, `RESULTS_DIR` nằm dưới `OUTPUT_ROOT`; dữ liệu luôn đọc
  từ `data/raw/`. `SANDBOXED = True` khi đang chạy trong sandbox.
- `OFFICIAL_TUNED_PARAMS_PATH = models/tuned_params.json` (luôn trỏ vào repo, kể cả khi sandbox).
- Giao thức: `TEST_SIZE`, `VALIDATION_SIZE`, `RANDOM_STATE = 42`, `CV_FOLDS = 5`,
  `PRIMARY_SCORING = "average_precision"`, `SEARCH_ITERATIONS = 30`, `SEARCH_N_JOBS = 4`.
- Calibration: `CALIBRATE_PROBABILITIES = True`, `CALIBRATION_METHOD = "sigmoid"`,
  `CALIBRATION_ENSEMBLE = False`.
- RF: `DEFAULT_RF_PARAMS` (200 cây, `balanced_subsample`, seed 42), `RF_PARAM_GRID` (5 tham số, 216 tổ hợp).
- **Không sửa** khối `COST_*` (Nhật Linh, sau quyết định D3) và `DEFAULT_*_PARAMS`/`*_PARAM_GRID`
  của DT/KNN/SVM (Thùy Linh).

### 4.2. `src/data_loader.py`, `src/preprocessing.py`

- `load_raw_data()` (IBM cho `train.py`): kiểm tra schema, đổi `TotalCharges` sang số, map nhãn
  Yes/No → 1/0, bỏ `customerID`, bỏ `TotalCharges` theo giao thức chính.
- `detect_feature_types(X, verbose=True)`: thứ tự cột số cố định theo `NUMERIC_FEATURES` rồi tới 6
  đặc trưng tạo thêm. **Cả `train.py` và `compare_datasets.py` đều dùng hàm này cho IBM**: cùng seed
  nhưng khác thứ tự cột là một khu rừng khác.
- `get_file_sha256(path)`: hash dữ liệu/model.
- `TelcoFeatureEngineer(enabled)`: 6 đặc trưng theo từng dòng, `fit` không học gì (không thể rò rỉ).
  `ENGINEERED_FEATURE_SOURCES`: mỗi đặc trưng tạo thêm được tính từ những cột gốc nào (dùng khi
  giải thích cho người dùng).
- `build_preprocessor(numeric, categorical, scale_numeric)`: số → `SimpleImputer(median)` [→
  `StandardScaler`]; phân loại → `SimpleImputer(most_frequent)` → `OneHotEncoder(drop="first",
  handle_unknown="ignore")`. Cây không cần scale (`scale_numeric=False`).
- `build_model_pipeline(preprocessor, classifier, feature_engineer=None)`.

### 4.3. `src/compare_datasets.py` — khung benchmark 3 dataset

| Thành phần | Ý nghĩa |
|---|---|
| `DATASET_PROFILES` | Đường dẫn và cột nhãn của `ibm`, `orange`, `iranian` |
| `load_dataset(name, keep_total_charges=False)` | Đọc và làm sạch tối thiểu; trả `X, y, profile, audit` (audit có SHA-256) |
| `feature_lists(X, name)` | Cột số/phân loại; IBM qua `detect_feature_types` |
| `BENCHMARK_MODELS` | `['Dummy', 'LogReg', 'RandomForest']` (+ DT/KNN/SVM khi Thùy Linh thêm) |
| `REQUIRED_MODELS` | `('LogReg', 'RandomForest')` — luôn chạy, kể cả với `--models` |
| `MODEL_PARAM_GRIDS` | Lưới khi `--tune`; `_grid_for_pipeline` tự đổi `classifier__` ↔ `classifier__estimator__` khi mô hình bọc `CalibratedClassifierCV` |
| `_build_model(name, numeric, categorical, is_ibm)` | Pipeline chưa fit; có chú thích chỗ chèn DT/KNN/SVM; `_calibrated(...)` bọc Platt cho LogReg/SVM |
| `_fit_model(name, pipeline, X, y, tune, groups)` | Fit, hoặc tune bằng `RandomizedSearchCV` với CV từ `cv_splitter(groups)` |
| `cv_splitter(groups)` | `StratifiedGroupKFold` nếu có nhóm, ngược lại `StratifiedKFold` (5 fold, seed 42) |
| `calibration_cv(X_train, y_train, groups)` | `CV_FOLDS` hoặc danh sách fold chia theo nhóm (Iranian) |
| `make_deployed_model(rf, X_train, y_train, groups)` | Model deploy **chưa fit**: `CalibratedClassifierCV(clone(rf), sigmoid, cv, ensemble=False)` |
| `split_data(...)` (= `_split_data`) | Chia 70/15/15 theo seed; `is_group_aware(name, split_method)` |
| `tuned_params(run, model)` / `load_tuned_params()` / `save_tuned_params(...)` | Đọc/ghi `models/tuned_params.json` (tên tham số rút gọn, ví dụ `max_depth`). Đọc: file chính thức trước, file của sandbox đè lên theo từng (run, mô hình). Ghi: run đầy đủ thay cả mục của run; `--models` chỉ cập nhật các mô hình được tune lại. Thiếu thì cảnh báo một lần và dùng mặc định |
| `build_benchmark_model(model, X, name, run_name=None)` | Pipeline của một mô hình với tham số đã tune |
| `fit_deployed_rf(name, seed, split_method)` | Model deploy đã fit trên một seed + 6 tập dữ liệu — dành cho phân tích chính sách |
| `run_name_for(name, keep_total_charges, split_method)` | `ibm`, `ibm_with_total_charges` (A1), `iranian`, `iranian_row_wise` — biến thể không ghi đè run chính |
| `run_dataset(...)` | Chạy một dataset: fit mọi mô hình, calibrate RF, chọn ngưỡng, ghi `results/dataset_runs/<run>/` và `models/dataset_runs/<run>/rf_calibrated.joblib`; `--tune` ghi thêm `tuned_params.json` |
| `update_long_table(rows, replace)` | Bảng dài `model_comparison_long.csv`; `replace="pair"` khi chỉ chạy lại một số mô hình |

CLI: `python -m src.compare_datasets --dataset {ibm,orange,iranian,both,all} [--tune]
[--keep-total-charges] [--split-method {default,group_aware,row_wise}] [--models M ...]`.

### 4.4. `src/train.py` — mô hình deploy (IBM)

Chạy 7 mô hình trên IBM (Dummy, LogReg, DecisionTree, RandomForest, KNN, SVM_Linear, SVM_RBF), có
`--tune` và `--with-shap`. RF đã tune → `make_deployed_model` → chọn ngưỡng trên validation →
`models/rf_best.joblib`, `models/config.json` (ngưỡng, chi phí), `models/metadata.json` (cột đầu vào,
độ quan trọng, calibration, SHA-256, phiên bản), `results/comparison_table.csv` (thêm hai dòng
`RF_Calibrated@0.5` và `RF_Calibrated@<ngưỡng>`), `results/policy_metrics.json`.

- `metadata.json → global_feature_importance` chỉ gồm cột gốc (API hiển thị giá trị khách đã nhập);
  `global_feature_importance_model_inputs` có cả 6 đặc trưng tạo thêm.
- `--keep-total-charges` bị chặn nếu không đặt `CHURN_OUTPUT_ROOT` (tránh ghi đè model deploy);
  A1 chính thức là `compare_datasets --dataset ibm --keep-total-charges`.
- Vì cùng thứ tự cột, cùng phép chia, cùng cấu hình tìm kiếm, RF của `train.py` và RF của run `ibm`
  trong `compare_datasets` là **cùng một mô hình**.

### 4.5. `src/evaluate.py`, `src/optimize_threshold.py`

- `compute_metrics(y_true, y_pred, y_proba)` → `accuracy`, `precision_churn`, `recall_churn`,
  `f1_churn`, `f1_macro`, `balanced_accuracy`, `roc_auc`, `average_precision`, `brier_score`,
  `log_loss`, `ece` (10 bin đều), `lift_at_10pct`.
- `find_cost_optimal_threshold(y, p, cost_fp, cost_fn, cost_mode, thresholds, verbose=True)`: duyệt
  mọi ngưỡng là các xác suất dự đoán (thêm 0 và 1), trả `best_threshold`, `best_cost`,
  `default_cost` (ngưỡng 0.5), `savings`, `best_f1_threshold`. Gọi trong vòng lặp thì `verbose=False`.
- `save_model_config(...)` ghi `models/config.json` cho API.

### 4.6. `src/explain.py` — SHAP

- `compute_shap_values(pipeline, X)`: TreeSHAP trên rừng, giá trị của lớp churn.
- `signed_contributions(model, customer)`: đóng góp **có dấu** của từng đầu vào (gộp các cột one-hot
  về biến gốc); chạy được với pipeline thường và với `CalibratedClassifierCV` (trung bình trên các rừng).
- `explain_customer(model, customer, top_n)`: danh sách `{feature, value, contribution, direction,
  impact, source_features}` — dùng cho API và báo cáo. **Thang đo:** đóng góp là SHAP của điểm rừng
  *trước* calibration. Calibration là hàm đơn điệu nên giữ nguyên thứ hạng rủi ro giữa các khách,
  nhưng giá trị SHAP không chuyển nguyên vẹn qua hàm phi tuyến: thứ tự/chiều của các yếu tố là của
  điểm rừng, thường (không phải luôn luôn) trùng với giải thích trên xác suất đã calibrate.
- `get_top_factors_for_customer(model, customer, original_columns, top_n)`: định dạng ngắn cũ
  `{feature, value, impact}`, chỉ xếp hạng cột gốc; nay chạy được với model đã calibrate.
- `aggregate_shap_to_original_features` (|SHAP|, cho độ quan trọng toàn cục),
  `plot_shap_summary`, `plot_aggregated_importance` (nay gồm cả đặc trưng tạo thêm).

### 4.7. `src/iranian_audit.py`

Audit Iranian: dòng trùng, nhóm trùng khác nhãn, độ chồng lấn train/test khi chia theo dòng, thông tin
tương hỗ của từng cột với nhãn; sau đó so sánh 10 seed chia theo dòng với 10 seed chia theo nhóm bằng
RF với tham số đã tune của run `iranian` (`rf_params()`). Kết quả: snapshot §4,
`results/dataset_runs/iranian/audit_report.json`, `figures/dataset_runs/iranian_stability_metrics.png`.

### 4.8. `src/ablation.py` — A2, A3 (A1 nằm ở `compare_datasets`)

Mỗi biến thể chỉ đổi **một** yếu tố, mọi thứ khác là RF đã tune của run chính (`tuned_rf_params`):

| Thí nghiệm | Biến thể |
|---|---|
| A2 — mất cân bằng | `class_weight="balanced_subsample"` (mặc định) · không xử lý · SMOTE trong `imblearn` Pipeline |
| A3 — chuẩn hóa | không scale (mặc định) · `StandardScaler` |

5 seed (42–46; `--quick`: 1 seed), 3 dataset, ngưỡng 0.5; ghi chỉ số trên test và `val_*` trên
validation (để ra quyết định). Kết quả: `results/ablation/`, snapshot §6.

### 4.9. `src/model_stability.py` — 10 seed và kiểm định

Mọi mô hình trong `BENCHMARK_MODELS` trừ Dummy × 10 seed (`[13, 42, 73, 101, 2025, 303, 404, 505,
606, 707]`; `--quick`: 3 seed) × 3 dataset, tham số đã tune cố định, ngưỡng 0.5. RF được so với mô
hình có AP trung bình cao nhất còn lại:

- `corrected_resampled_ttest` (Nadeau & Bengio 2003): $t = \bar d / \sqrt{(1/k + n_{test}/n_{train})\,s_d^2}$,
  bậc tự do $k-1$ — **dùng để trích dẫn**;
- `paired_wilcoxon` (không tính đến chồng lấn; 10 cặp thì p nhỏ nhất là 0.002);
- `mcnemar_exact` trên tập test của seed 42.

Kết quả: `results/model_stability/{stability_runs,stability_summary,stat_tests}.csv`, snapshot §7.
Mô hình Thùy Linh thêm vào `BENCHMARK_MODELS` tự được đưa vào.

### 4.10. `src/calibration_analysis.py`

So sánh RF chưa calibrate, sigmoid `ensemble=True` (5 rừng), sigmoid `ensemble=False` (model deploy),
isotonic, và các mô hình khác; ROC-AUC/AP/Brier/log-loss/ECE trên test, AP/Brier/ECE trên validation
(cột `val_*` — **lựa chọn dựa trên cột này**), thời gian fit, kích thước file (thường và nén
`compress=3`). Iranian dùng fold chia theo nhóm. Kết quả: `results/calibration/`,
`figures/rf/reliability_<dataset>.png`, snapshot §8.

### 4.11. `src/rf_analysis.py` + notebook 02

Cho mỗi dataset: validation curve theo `n_estimators` (50→500) và `max_depth` (3→None); OOB vs CV vs
test; độ quan trọng 3 cách (MDI, permutation trên test, mean |SHAP|) và hệ số Spearman giữa chúng;
SHAP cục bộ cho 3 khách TP/FN/FP (cột `rf_score` = điểm rừng chưa calibrate). Kết quả:
`results/rf_analysis/`, `figures/rf/{validation_curve,importance_methods,local_shap}_<dataset>.png`,
snapshot §9. Notebook 02 **chỉ hiển thị** kết quả (không tự chạy lại); thiếu kết quả thì báo lệnh cần chạy.

### 4.12. `src/reproduce.py` và `src/export_results.py`

- `python -m src.reproduce`: xóa các bảng gộp nhiều lần chạy (`MERGED_TABLES`) → pytest → train →
  compare (tune) → A1 → audit → ablation → stability → calibration → rf_analysis → module tùy chọn
  (`OPTIONAL_STEPS`, hiện có `src.policy_analysis`) → export. Dừng ngay khi một bước lỗi. Log
  `results/reproduce_log.json`: commit, cây có sửa đổi chưa commit không, phiên bản thư viện, SHA-256
  của 3 file dữ liệu, từng lệnh, mã thoát và thời gian.
- `--quick`: không tune, ít seed, mọi đầu ra vào `results/quick/` (đặt `CHURN_OUTPUT_ROOT`) → không
  file nào trong git bị đổi; snapshot nhanh ở `results/quick/results/results_snapshot_quick.md`.
- `export_results`: đọc `results/`, ghi `reports/results_snapshot.md` (§1–§9) và nối mọi
  `results/<tên>/snapshot.md` của module khác vào §10. Trong sandbox (hoặc sau lượt quick) snapshot
  luôn ghi vào `<sandbox>/results/results_snapshot_quick.md`, không bao giờ đụng `reports/`.

---

## 5. Artifact và snapshot

| Artifact | Sinh bởi | Trong git? | Ai dùng |
|---|---|---|---|
| `models/rf_best.joblib` | `train.py` | Không | API (Hải Anh) |
| `models/config.json` | `train.py` | Có | API: ngưỡng, chi phí |
| `models/metadata.json` | `train.py` | Có | API: cột đầu vào, độ quan trọng; báo cáo: SHA-256, phiên bản |
| `models/tuned_params.json` | `compare_datasets --tune` | Có | Mọi phân tích |
| `models/dataset_runs/<run>/rf_calibrated.joblib` | `compare_datasets` | Không | Phân tích thêm |
| `results/**` | mọi script | Không | `export_results` |
| `figures/rf/*.png`, `figures/dataset_runs/*.png` | các phân tích | Có | Báo cáo, slide |
| `reports/results_snapshot.md` | `export_results` | Có | **Nguồn số liệu duy nhất** cho báo cáo/slide |

Các mục trong snapshot: §1 IBM 7 mô hình (+2 dòng RF đã calibrate) · §2 ngưỡng theo chi phí ·
§3 benchmark 3 dataset (RF vs LogReg, có A1) · §4 Iranian chia theo dòng vs theo nhóm · §5 mọi mô
hình × dataset · §6 ablation A2/A3 · §7 10 seed + kiểm định · §8 calibration và kích thước ·
§9 phân tích RF · §10 phần bổ sung của module khác.

---

## 6. Kết quả hiện tại và cách diễn giải

<!-- RESULTS-START -->
_(điền sau lượt chạy chính thức)_
<!-- RESULTS-END -->

---

## 7. Hợp đồng tích hợp với từng thành viên

### 7.1. Thùy Linh — DT, KNN, SVM vào benchmark 3 dataset

1. Thêm tên vào `BENCHMARK_MODELS`; thêm nhánh `elif` ở chỗ có chú thích *"Comparison models
   (Thùy Linh)"* trong `_build_model` (import lớp mô hình, dùng `DEFAULT_*_PARAMS`, SVM bọc
   `_calibrated(...)`, `scale_numeric=True` cho KNN/SVM).
2. Thêm lưới vào `MODEL_PARAM_GRIDS` (lưới đặt trong `config.py`).
3. `python -m pytest tests/test_experiments.py -q`: test contract kiểm tra mọi mô hình (fit được,
   xác suất trong [0, 1], khóa lưới khớp pipeline).
4. `python -m src.compare_datasets --dataset all --tune` (chỉ mô hình mới: `--models DecisionTree KNN
   SVM_Linear SVM_RBF`). Tham số tốt nhất tự ghi vào `models/tuned_params.json`; `model_stability`
   và `calibration_analysis` tự nhận mô hình mới.
5. Khánh review: không đổi nhánh RF/LogReg/Dummy; tham số tốt nhất không nằm ở biên lưới (nếu có thì
   mở rộng lưới); `model_comparison_long.csv` đủ 6–7 mô hình × 3 dataset.

### 7.2. Nhật Linh — phân tích ngưỡng và chính sách

- Lấy model: `fit_deployed_rf('ibm', seed=s)` → `model`, `X_val/y_val` (chọn ngưỡng), `X_test/y_test`
  (báo cáo). Không tự dựng `CalibratedClassifierCV` (dễ quên `clone` và fold theo nhóm).
- `find_cost_optimal_threshold(..., verbose=False)` trong vòng lặp; `cost_mode="action"` cho chi phí
  theo số khách được liên hệ.
- Nối vào pipeline: tạo `src/policy_analysis.py` chạy được bằng `python -m src.policy_analysis` (có
  `--quick` nếu muốn bản nhanh) → `reproduce` tự chạy. Thư mục đầu ra lấy từ `src.config`:
  `RESULTS_DIR / 'policy'`, `FIGURES_DIR / 'policy'` (không viết cứng `results/...`, để `--quick` ở
  trong sandbox). Ghi `RESULTS_DIR / 'policy' / 'snapshot.md'` (bắt đầu bằng `### ...`) → snapshot tự
  có §10.
- `COST_*` chỉ đổi sau khi chốt D3; khi đổi, Khánh chạy lại `train.py` để `models/config.json` có ngưỡng mới.

### 7.3. Quỳnh Anh — dữ liệu

- `src/download_data.py` kiểm tra SHA-256 theo bảng `PLAN.md` mục 7; `reproduce` ghi lại hash đã dùng
  vào log và đầu snapshot để đối chiếu.
- Nếu phát hiện vấn đề làm sạch dữ liệu, gửi đề xuất sửa `load_dataset` cho Khánh (không tự sửa để
  tránh đổi số liệu của mọi phân tích).
- Câu hỏi còn mở cho EDA: `Complains`, `Status` của Iranian có được đo **trước** thời điểm churn
  không (nguy cơ rò rỉ theo thời gian).

### 7.4. Hải Anh — API, CI, deploy

- Model: `models/rf_best.joblib` (một rừng + sigmoid); nén thêm khi deploy bằng
  `joblib.dump(model, path, compress=3)`. Ngưỡng ở `models/config.json`; cột đầu vào ở
  `metadata.json → feature_columns`.
- Giải thích từng khách: `explain_customer(model, df, top_n=5)` (khoảng 0.15–0.5 giây/khách); hoặc
  định dạng ngắn `get_top_factors_for_customer(model, df, feature_columns)`.
- CI: các test đọc CSV thật trong `tests/test_pipeline.py` cần `pytest.mark.skipif` (PR CI của Hải
  Anh, báo Khánh review). `tests/test_experiments.py` không cần dữ liệu thật.
- Nếu D1 = Iranian: dùng `models/dataset_runs/iranian/rf_calibrated.joblib` và
  `results/dataset_runs/iranian/metadata.json` (cột đầu vào), hoặc nhờ Khánh xuất model Iranian riêng.

### 7.5. Checklist khi Khánh review PR

- [ ] PR chỉ sửa file trong vùng của người gửi; không kèm `results/`, `.joblib`, dữ liệu.
- [ ] Không kèm `models/*.json`, `figures/rf/`, `figures/dataset_runs/`, `reports/` do chạy thử.
- [ ] `python -m pytest -q` pass trên máy Khánh; `python -m src.reproduce --quick` báo `completed`.
- [ ] Không có ngưỡng/chi phí chọn trên tập test.
- [ ] Code mới đọc tham số qua `config`/`tuned_params`, không gán cứng.

---

## 8. Kiểm thử và tái lập

- `tests/test_experiments.py` (không cần dữ liệu thật): lưới tham số, bảng dài, seed chia dữ liệu,
  nhóm trùng, SMOTE chỉ khi fit, McNemar/Wilcoxon/t có hiệu chỉnh, `explain_customer` (kể cả model đã
  calibrate, `source_features`), kích thước model, chọn ca TP/FN/FP, contract mọi mô hình, cùng thứ tự
  cột IBM, model deploy không giữ bản rừng thừa, fold calibration theo nhóm, đọc/ghi `tuned_params`,
  tên run của biến thể, module tùy chọn trong reproduce, fragment snapshot.
- `tests/test_pipeline.py`: kiểm tra trên dữ liệu thật (chia 70/15/15, Iranian không rò rỉ nhóm,
  profile 3 dataset…).
- Tái lập: seed cố định ở chia dữ liệu, tuning, mô hình, SMOTE; dữ liệu có SHA-256; phiên bản thư viện
  ghi trong log. Kiểm tra định kỳ bằng cách chạy lại từ một commit sạch và so snapshot.
- Quy ước số liệu: báo cáo/slide chỉ trích từ snapshot của tag `v1.0-results`.

---

## 9. Hạn chế đã biết

| Hạn chế | Ảnh hưởng | Cách nói khi bảo vệ |
|---|---|---|
| Tham số tune một lần trên train của seed 42, dùng lại cho 10 seed | Một phần test của seed khác từng nằm trong train lúc tune → hơi lợi cho mô hình đã tune (RF và LogReg như nhau) | Đã ghi rõ; so sánh vẫn công bằng giữa các mô hình |
| Chi phí FP/FN là giả định | Ngưỡng 0.05 rất thấp, khuyến nghị "liên hệ gần hết khách" | Đang làm phân tích độ nhạy + chính sách top-k (Nhật Linh) |
| Rủi ro rò rỉ theo thời gian ở Iranian (`Complains`, `Status`) | AP rất cao có thể một phần do thông tin "tương lai" | Đã ghi là câu hỏi mở; đề xuất kiểm tra |
| Calibration bên trong LogReg/SVM chia fold theo dòng (kể cả Iranian) | Rất nhỏ với mô hình tuyến tính | Model deploy (RF) đã chia theo nhóm |
| Không hiệu chỉnh so sánh nhiều lần | Một trong nhiều p-value nhỏ có thể do ngẫu nhiên | Kết luận dựa trên hiệu ứng lớn, nhất quán qua 10 seed |
| SHAP giải thích điểm rừng trước calibration | Đóng góp không cộng ra đúng xác suất hiển thị; thứ tự/chiều của yếu tố nhỏ có thể khác nếu giải thích trên xác suất đã calibrate | Thứ hạng rủi ro giữa các khách giữ nguyên; giải thích là của điểm rừng |
| So sánh F1 / McNemar ở ngưỡng 0.5 | RF (class_weight, chưa calibrate) và LogReg (đã calibrate) có ngưỡng 0.5 mang ý nghĩa khác nhau | Chất lượng mô hình so bằng AP/ROC-AUC (không phụ thuộc ngưỡng); F1@0.5 chỉ là so điểm vận hành |
| OOB và CV tính trên cùng tập train đã dùng để tune | Cả hai hơi lạc quan (CV nhiều hơn); trên Iranian OOB còn lạc quan thêm vì bản trùng | Chỉ tập test là độc lập với việc tune; báo cáo test và trung bình 10 seed |
| Nguồn gốc bộ Orange (bản mirror) | Khó trích dẫn | Quỳnh Anh đang tìm nguồn gốc/license |

---

## 10. Việc còn lại — runbook theo ngày

| Ngày | Việc | Cách làm / tiêu chí xong |
|---|---|---|
| 25–26/11 | Đẩy nhánh, mở PR `namkhanh/rf-main-model` → `main` | Các bạn cần khung trước 27/11; báo nhóm đọc `docs/huong-dan-ky-thuat.md` và tài liệu này |
| 25–27/11 | Quản trị repo | Mời collaborator; bật branch protection cho `main` (bắt buộc PR, 1 approval); chia sẻ 3 file CSV qua Drive nhóm |
| trước 28/11 | Buổi hướng dẫn kỹ thuật 30–45 phút | Theo `docs/huong-dan-ky-thuat.md`; kết thúc khi cả 5 người chạy `reproduce --quick` báo `completed` |
| 29/11 | Review PR `download_data.py` (Quỳnh Anh) | Checklist 7.5; hash khớp `PLAN.md` |
| 01/12 | Review PR DT/KNN/SVM (Thùy Linh) | Mục 7.1; chạy `compare_datasets --dataset all --tune` |
| 04/12 | Nhận `COST_*` (Nhật Linh) | Chạy lại `python -m src.train --tune --with-shap`; báo Hải Anh ngưỡng mới |
| 05/12 | Review PR `policy_analysis.py` | `reproduce --quick` phải chạy luôn bước này; snapshot có §10 |
| **06/12** | **Đóng băng số liệu** | Trên `main` sạch: `python -m src.reproduce` (~40 phút) → commit `reports/results_snapshot.md`, `models/*.json`, `figures/rf/`, `figures/dataset_runs/` → `git tag v1.0-results && git push origin v1.0-results` → báo nhóm |
| 07–13/12 | Viết 4.1, 4.2, 5.2 bằng số của tag; bàn giao | Thùy Linh: cách đọc `stability_summary.csv`/`stat_tests.csv`; Nhật Linh: reliability diagram, `fit_deployed_rf` |
| trước 18/12 | Đối chiếu số liệu | Mọi số trong `docs/report/` và slide khớp snapshot của tag; gửi danh sách chỗ lệch |
| tuần 4 | Thử tái lập trên máy sạch | Một bạn clone mới → đặt dữ liệu → `python -m src.reproduce` → so snapshot với tag |
| tuần 4 | Slide và bảo vệ | 2–3 slide RF (`docs/slides/namkhanh-random-forest.md`); câu trả lời ở `docs/defense_qa/namkhanh.md`; 2 buổi tổng duyệt |

---

## 11. Kiến thức nền cho phần bảo vệ

**Random Forest.** Nhiều cây quyết định, mỗi cây học trên một mẫu bootstrap và chỉ xét một tập con
ngẫu nhiên đặc trưng ở mỗi nút (`max_features`). Dự đoán = trung bình xác suất các cây. Phương sai
của trung bình $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$: thêm cây làm giảm số hạng thứ hai, chọn
ngẫu nhiên đặc trưng làm giảm tương quan $\rho$. Thêm cây không gây overfit; độ sâu thì có thể.

**OOB.** Mỗi dòng vắng mặt ở khoảng 36.8% mẫu bootstrap; dự đoán dòng đó bằng các cây "chưa thấy" nó
cho một ước lượng gần như cross-validation miễn phí.

**AP vs ROC-AUC.** ROC-AUC đo khả năng xếp hạng trên cả hai lớp, ít nhạy với mất cân bằng; AP (diện
tích dưới đường Precision–Recall) tập trung vào lớp thiểu số, mức "đoán bừa" bằng tỉ lệ churn. Với
churn 15–27%, AP phân biệt mô hình tốt hơn.

**Calibration.** Mô hình đã calibrate thì "xác suất 0.3" nghĩa là khoảng 30% khách như vậy rời đi.
RF của nhóm lệch chủ yếu do `class_weight` (học như thể hai lớp cân bằng, đẩy xác suất churn lên) và
do lấy trung bình nhiều cây. Platt: khớp $1/(1+e^{Af+B})$ trên dự đoán cross-validation (2 tham số,
ổn định); isotonic: hàm bậc thang đơn điệu, linh hoạt hơn nhưng dễ overfit. Đo bằng Brier và ECE.
`ensemble=False`: dùng dự đoán CV để khớp sigmoid, rồi huấn luyện lại một rừng trên toàn bộ train.

**Ngưỡng theo chi phí.** Với FN đắt gấp 10 lần FP, ngưỡng tối ưu lý thuyết cho xác suất đã calibrate
là $c_{FP}/(c_{FP}+c_{FN}) = 50/550 \approx 0.09$; ngưỡng thực nghiệm trên validation nằm gần đó.

**class_weight vs SMOTE.** `class_weight` tăng trọng số lỗi của lớp churn khi chọn phép chia, không
tạo dữ liệu; SMOTE sinh mẫu tổng hợp nằm giữa các khách churn gần nhau. SMOTE phải nằm trong pipeline
(chỉ trên fold train) — làm trước khi chia sẽ khiến mẫu tổng hợp mang thông tin của tập test.

**Ba cách đo độ quan trọng.** MDI (tổng mức giảm Gini, tính trên train, thiên vị biến nhiều giá trị),
permutation (AP trên test giảm bao nhiêu khi xáo một cột; chia điểm giữa biến tương quan), SHAP (giá
trị Shapley của từng biến cho từng dự đoán; có dấu; là tương quan mô hình học, không phải nhân quả).

**Kiểm định.** McNemar: trên cùng một tập test, đếm số khách A đúng/B sai (b) và A sai/B đúng (c);
kiểm định nhị thức trên b và c. Wilcoxon: xếp hạng các hiệu số giữa hai mô hình qua 10 seed. t có
hiệu chỉnh: như t ghép cặp nhưng phương sai nhân thêm $n_{test}/n_{train}$ vì các tập train chồng
nhau. p nhỏ nghĩa là khác biệt khó do ngẫu nhiên chia dữ liệu; **không** cho biết khác biệt lớn hay
có ý nghĩa thực tế, và không tự sửa cho việc so sánh nhiều lần.

**Rò rỉ dữ liệu.** Thông tin của tập test lọt vào lúc huấn luyện: fit scaler/imputer/SMOTE trên toàn
bộ dữ liệu, chọn ngưỡng/tham số trên test, bản sao của cùng khách ở cả train và test, hoặc đặc trưng
đo sau thời điểm dự đoán.

---

## 12. Lệnh hay dùng

```bash
python -m pytest -q                                   # toàn bộ test
python -m src.reproduce --quick                       # kiểm tra nhanh, không đổi file trong git
python -m src.reproduce                               # chạy chính thức (~40 phút)
python -m src.train --tune --with-shap                # chỉ model deploy IBM
python -m src.compare_datasets --dataset all --tune   # benchmark 3 dataset + tuned_params.json
python -m src.compare_datasets --dataset ibm --models RandomForest LogReg
python -m src.model_stability --dataset ibm --quick
CHURN_OUTPUT_ROOT=results/scratch python -m src.rf_analysis --dataset iranian   # chạy thử không đụng file trong git
python -m src.export_results                          # xuất lại snapshot từ results/
```

---

## 13. Nhật ký thay đổi 25/11

| Thay đổi | Lý do |
|---|---|
| `make_deployed_model` clone RF trước khi calibrate | `CalibratedClassifierCV` giữ nguyên estimator được truyền vào; truyền rừng đã fit làm file model chứa thêm một rừng không dùng (gấp đôi dung lượng) |
| IBM dùng chung `detect_feature_types` ở `train.py` và `compare_datasets.py` | Trước đây hai script xếp cột số khác thứ tự → cùng seed nhưng hai khu rừng khác nhau, số §1 và §3/§5 không khớp |
| `models/tuned_params.json` (có trong git) thay cho đọc `results/` | Trước đây chạy `compare` không `--tune` ghi đè tham số và các phân tích lặng lẽ dùng mặc định |
| Sandbox `CHURN_OUTPUT_ROOT`; `reproduce --quick` ghi vào `results/quick/` | Chạy thử từng làm đổi `models/*.json`, `figures/rf/`, `figures/dataset_runs/` trong git |
| `iranian_audit` dùng tham số đã tune và thư mục hình qua `config` | Tham số từng gán cứng; hình bỏ qua sandbox |
| Calibration CV chia theo nhóm cho Iranian | Fold chia theo dòng làm sigmoid khớp trên dự đoán lạc quan |
| Ablation dùng RF đã tune, có thêm `val_*` | Ablation trên tham số mặc định không mô tả model deploy; quyết định đọc trên validation |
| Calibration analysis chọn phương án theo validation | Không dùng tập test để ra lựa chọn |
| Kiểm định t có hiệu chỉnh Nadeau–Bengio | Wilcoxon/t thường bỏ qua việc các tập train chồng nhau |
| `rf_analysis`: sửa vạch "tuned" trên validation curve, cột `rf_score` | Vạch từng vẽ ở `max_depth=None` bất kể giá trị đã tune |
| `explain.py`: `source_features`, `get_top_factors_for_customer` chạy với model calibrate, ghi rõ thang đo | Tích hợp API |
| Biến thể run có tên riêng (`iranian_row_wise`), cờ `--models`, `train --keep-total-charges` cần sandbox | Không biến thể nào ghi đè run chính hay model deploy |
| `reproduce`: module tùy chọn, fragment snapshot, ghi SHA-256 + phiên bản thư viện | Các bạn nối phần của mình mà không sửa file của Khánh |
| `matplotlib.use("Agg")` chỉ khi không chạy trong notebook; `optimize_threshold` in ASCII, có `verbose` | Notebook của các bạn vẫn hiển thị hình; không lỗi mã hóa trên Windows |
