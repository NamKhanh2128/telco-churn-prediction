# Plan — Nguyễn Đức Nam Khánh · Random Forest (mô hình chính) & thí nghiệm AI

- **Branch:** `namkhanh/rf-main-model`
- **Vùng file phụ trách:**
  - Mô hình & thí nghiệm: toàn bộ code trong `src/` — `train.py`, `compare_datasets.py`,
    `preprocessing.py`, `data_loader.py`, `explain.py`, `optimize_threshold.py`, `evaluate.py`,
    `iranian_audit.py`, `ablation.py`, `model_stability.py`, `policy_analysis.py` (mới)
    (trừ `src/download_data.py` của Hải Anh)
  - Tái lập kết quả: `src/reproduce.py`, `src/export_results.py`, `reports/results_snapshot.md`
  - `DEFAULT_RF_PARAMS`, `RF_PARAM_GRID` trong `src/config.py`; `tests/test_pipeline.py`
  - `notebooks/02_*`, `figures/rf/`, `figures/policy/`
  - Báo cáo: `docs/report/04-1-tien-xu-ly.md`, `04-2-random-forest.md`, `05-2-phan-tich-rf-ablation.md`
- **Mục tiêu:**
  1. Chứng minh và giải thích vì sao Random Forest là phương pháp chính.
  2. Viết code toàn bộ thí nghiệm: benchmark các mô hình trên 3 dataset, ablation, độ ổn định nhiều
     seed kèm kiểm định thống kê, phân tích ngưỡng theo chi phí.
  3. Mọi con số tái lập được bằng một lệnh và gắn với tag `v1.0-results`.

## Đã xong (24/11)

- [x] Split 70/15/15; thêm SVM_RBF; thêm các dòng `RF_Calibrated` vào bảng so sánh.
- [x] `src/reproduce.py` (một lệnh chạy lại toàn bộ) và `src/export_results.py`
  (`results/` → `reports/results_snapshot.md`); snapshot gắn với commit sạch.
- [x] `--quick` chỉ ghi snapshot vào `results/` (bị ignore), không ghi đè snapshot chính thức.
- [x] Tạo branch và plan cho từng người.

## Đọc trước

- `PLAN.md`, `mustdo.md` (mục P0), `reports/results_snapshot.md`
- `src/train.py` bước 4–8 (định nghĩa RF, tuning, calibration, chọn ngưỡng)
- `src/compare_datasets.py`: `_build_model`, `_fit_model`, `_split_data`, `_group_split_indices`, `run_dataset`
- `src/optimize_threshold.py` (2 cách tính chi phí `classification` / `action`), `src/evaluate.py`
  (ECE, lift), `src/explain.py` (SHAP gộp về đặc trưng gốc)
- `telco_churn_pipeline.md` mục 10.2 (bảng ablation A1–A4)

## Việc cần làm

### Tuần 0–1 (24–29/11) — mở khóa cho cả nhóm, benchmark

- [ ] **Quản trị repo** (chỉ chủ repo làm được): mời 4 bạn làm collaborator (GitHub → *Settings →
  Collaborators*); bật *Settings → Branches → Branch protection* cho `main` (bắt buộc PR, 1 approval).
- [ ] **Dữ liệu:** chia sẻ 3 file CSV qua Drive riêng của nhóm (**trước 25/11**).
- [ ] **Mở rộng benchmark 3 dataset.** Ưu tiên, PR **trước 01/12** vì Thùy Linh cần để làm bảng so sánh.
  - `_build_model` trong `src/compare_datasets.py`: thêm `DecisionTree`, `KNN`, `SVM_Linear`,
    `SVM_RBF`, dùng lại `DEFAULT_*_PARAMS` trong `config.py`.
    - KNN/SVM: `scale_numeric=True`.
    - SVM bọc `CalibratedClassifierCV` giống `src/train.py`.
  - `_fit_model`: khi có `--tune`, tune **mọi** mô hình có lưới tham số. Lưu ý tiền tố
    `classifier__estimator__` khi mô hình bị bọc bởi `CalibratedClassifierCV`.
  - Ghi thêm bảng dài `results/dataset_runs/model_comparison_long.csv`, mỗi dòng là
    (dataset, model) với các cột `accuracy`, `precision_churn`, `recall_churn`, `f1_churn`,
    `f1_macro`, `roc_auc`, `average_precision`, `train_time_s`, `best_params`.
    Thêm section tương ứng trong `src/export_results.py`.
  - **Không đổi tên/xóa cột** của `dataset_comparison.csv`, vì snapshot §3 đang đọc nó.
  - Test: mỗi mô hình mới fit được trên dữ liệu nhỏ và `predict_proba` nằm trong [0, 1].
- [ ] **`src/ablation.py`** (chạy bằng `python -m src.ablation`, ghi `results/ablation/*.csv`):
  - **A2:** RF + SMOTE (dùng `imblearn.pipeline.Pipeline`, SMOTE chỉ áp trong fold train) vs RF +
    `class_weight` vs RF không xử lý mất cân bằng → F1/Recall/Precision lớp churn, PR-AUC.
  - **A3:** RF có vs không `StandardScaler` (kỳ vọng gần như giống hệt → chứng minh cây không cần scale).
  - A1 đã có (`results/dataset_runs/ibm_with_total_charges/`). Cùng split 70/15/15, cùng seed. Viết 1 test nhỏ.
- [ ] **`src/policy_analysis.py`**, phần chưa phụ thuộc D3 (chạy bằng `python -m src.policy_analysis`,
  ghi `results/policy/*.csv` và `figures/policy/`), trên model RF đã calibrate:
  - Độ nhạy theo tỉ lệ chi phí FN/FP ∈ {2, 3, 5, 10, 20}: ngưỡng tối ưu chọn trên validation;
    precision, recall, chi phí, savings trên test.
  - Chính sách top-k (k = 5%, 10%, 20%, 30%): precision@k, recall@k, lift@k, chi phí.
  - **A4:** RF @ ngưỡng 0.5 vs @ ngưỡng tối ưu.
  - Reliability diagram (`sklearn.calibration.CalibrationDisplay`) cho RF thô, RF đã calibrate,
    LogReg, SVM, kèm bảng Brier/ECE.

### Tuần 2 (30/11–06/12) — phân tích RF, độ ổn định, đóng băng số liệu

- [ ] **Notebook `notebooks/02_random_forest_analysis.ipynb`** trên dataset chính (D1):
  - Validation curve theo `n_estimators` (50 → 500) và `max_depth` (3 → None): giải thích bias–variance
    và vì sao bộ tham số tuned (`best_params` trong `results/test_metrics.json`) hợp lý.
  - OOB score (`oob_score=True`) so với điểm CV.
  - Độ quan trọng đặc trưng theo 3 cách: MDI, permutation importance (trên test), SHAP. Lập bảng
    top-10 và giải thích vì sao các cách cho kết quả khác nhau.
  - Giải thích cục bộ bằng SHAP cho 3 khách hàng: TP rủi ro cao, FN bị bỏ sót, FP báo nhầm
    (`src/explain.get_top_factors_for_customer`). Lưu hình vào `figures/rf/`.
- [ ] **`src/model_stability.py`** (chạy bằng `python -m src.model_stability`, ghi
  `results/model_stability/*.csv`):
  - 10 seed chia dữ liệu × 6 mô hình (LogReg, DT, RF, KNN, SVM_Linear, SVM_RBF) trên dataset chính,
    dùng `_build_model` và tham số đã tune, cố định, không tune lại mỗi seed.
  - Iranian dùng group-aware split cho từng seed (`_group_split_indices`).
  - Xuất mean ± std của Accuracy, Precision, Recall, F1 (lớp churn), ROC-AUC, AP.
- [ ] **Kiểm định thống kê** RF với mô hình mạnh nhất còn lại:
  - Wilcoxon signed-rank trên 10 seed (`scipy.stats.wilcoxon`).
  - McNemar trên tập test, tính từ các cặp bất đồng (`scipy.stats.binomtest`).
  - Ghi `results/model_stability/stat_tests.csv`, có cảnh báo về so sánh nhiều lần.
- [ ] Hoàn thiện `src/policy_analysis.py`:
  - Độ nhạy theo seed (10 seed): mean ± std của ngưỡng và savings.
  - Kết quả tại chi phí đã chốt (sau khi Nhật Linh cập nhật `COST_*`).
- [ ] **Calibration của model deploy:** `models/rf_best.joblib` ≈ 91 MB vì `CalibratedClassifierCV(cv=5)`
  lưu 5 bản RF. So sánh với `ensemble=False` (1 RF + 1 sigmoid) về metric và kích thước; chốt phương án
  và báo Hải Anh để nén và deploy.
- [ ] **Review PR code thí nghiệm** (`src/`) của các bạn; nối các script mới vào `src/reproduce.py`
  và thêm section vào `src/export_results.py` (gồm cả lưới tham số cuối của Thùy Linh, trước 04/12).
- [ ] (Tùy chọn) Viết hàm top-factors theo SHAP từng khách cho API, thay cho feature importance toàn
  cục trong `api/predict.py::_get_top_factors`; bàn giao cho Hải Anh.
- [ ] **06/12 — Đóng băng số liệu:**
  1. Trên `main` sạch, đã có `COST_*` cuối cùng: chạy `python -m src.reproduce`.
  2. Commit `reports/results_snapshot.md`, `models/config.json`, `models/metadata.json`,
     `figures/dataset_runs/`.
  3. Chạy `git tag v1.0-results && git push origin v1.0-results`.
  4. Báo cả nhóm: từ giờ chỉ lấy số từ snapshot của tag này.

### Tuần 3 (07–13/12) — viết

- [ ] **Mục 4.1 – Tiền xử lý & chống rò rỉ:**
  - Thứ tự split → fit pipeline chỉ trên train.
  - `TelcoFeatureEngineer`, imputation, one-hot, scaling theo từng loại mô hình.
  - Group-aware split cho Iranian.
  - Seed, SHA-256 của dữ liệu, `src/reproduce.py` và tag kết quả.
- [ ] **Mục 4.2 – Random Forest:** bagging, chọn ngẫu nhiên đặc trưng, tiêu chí Gini/entropy, OOB,
  feature importance, `class_weight="balanced_subsample"`, vì sao cần calibrate xác suất (Platt).
- [ ] **Mục 5.2 – Phân tích RF & ablation:** kết quả notebook 02, bảng A1–A3, nhận xét. Nói thẳng:
  trên IBM, RF không vượt LogReg về ROC-AUC/AP; lợi thế của RF thể hiện trên Orange/Iranian
  (phối hợp với mục 5.4 của Thùy Linh).
- [ ] **Bàn giao kết quả** và giải thích cách đọc file CSV/hình:
  - Cho Thùy Linh: độ ổn định và kiểm định, dùng ở mục 5.1.
  - Cho Nhật Linh: phân tích ngưỡng và calibration, dùng ở mục 5.3.
- [ ] Làm 2–3 slide phần Random Forest theo khung của Nhật Linh.

### Tuần 4 (14–20/12)

- [ ] **Thử tái lập trên máy sạch:** nhờ 1 bạn clone mới → đặt dữ liệu → `python -m src.reproduce`
  → so snapshot với tag `v1.0-results`.
- [ ] Review chéo mục 4.4, 5.3, 7.2 (Nhật Linh).
- [ ] Viết câu trả lời phần mình trong `docs/defense_qa.md`. Tham gia 2 buổi tổng duyệt.

## Phụ thuộc

- **Cần từ:**
  - Hải Anh: D1 (dataset chính cho phân tích RF).
  - Nhật Linh: `COST_*` sau khi chốt D3 (04/12).
  - Thùy Linh: lưới tham số cuối của DT/KNN/SVM (04/12).
  - Các bạn: PR code thí nghiệm trước 05/12.
- **Cung cấp cho:**
  - Cả nhóm: quyền truy cập và dữ liệu (25/11), tag `v1.0-results` (06/12).
  - Thùy Linh: benchmark đủ mô hình trên 3 dataset (01/12), kết quả độ ổn định và kiểm định.
  - Nhật Linh: kết quả phân tích ngưỡng.
  - Hải Anh: model cuối và phương án calibration để deploy.

## Câu hỏi bảo vệ cần chuẩn bị

1. Tại sao chọn RF làm phương pháp chính khi trên IBM LogReg ngang hoặc tốt hơn về ROC-AUC?
2. RF khác Decision Tree thế nào? Vì sao bagging giảm phương sai? OOB error là gì?
3. Vì sao phải calibrate RF trước khi chọn ngưỡng? Platt scaling khác isotonic thế nào?
4. SHAP khác feature importance có sẵn của RF ở điểm nào? SHAP có cho biết quan hệ nhân quả không?
5. `class_weight` khác SMOTE thế nào? Vì sao SMOTE phải nằm trong pipeline, sau khi chia dữ liệu?
6. Vì sao bỏ `TotalCharges` (A1)? Code đảm bảo không rò rỉ dữ liệu từ test vào train như thế nào?
7. Kiểm định Wilcoxon và McNemar cho biết điều gì? p-value nhỏ có nghĩa là gì, và không có nghĩa là gì?
8. Làm sao tái lập đúng kết quả? (seed, phiên bản thư viện, SHA-256 dữ liệu, tag, `src/reproduce.py`)

## Lệnh hay dùng

```bash
python -m src.train --tune --with-shap                # pipeline IBM đầy đủ
python -m src.compare_datasets --dataset all --tune   # benchmark 3 dataset
python -m src.reproduce                               # chạy final (~15 phút)
python -m src.reproduce --quick                       # kiểm tra nhanh, không ghi đè snapshot
python -m pytest -q
git fetch origin && git merge origin/main
```
