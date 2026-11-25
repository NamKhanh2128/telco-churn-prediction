# Plan — Nguyễn Đức Nam Khánh · Trưởng nhóm · Random Forest (mô hình chính) & thí nghiệm AI

- **Branch:** `namkhanh/rf-main-model`
- **Vùng file phụ trách:**
  - Khung thí nghiệm & mô hình chính trong `src/`: `train.py`, `compare_datasets.py` (gồm cả nhánh
    DT/KNN/SVM trong `_build_model`, theo phân công 25/11), `preprocessing.py`, `data_loader.py`,
    `explain.py`, `optimize_threshold.py`, `evaluate.py`, `iranian_audit.py`, `ablation.py`,
    `model_stability.py`, `calibration_analysis.py`, `rf_analysis.py` (mới); `tests/test_experiments.py`
  - Tài liệu kỹ thuật: `docs/huong-dan-ky-thuat.md`, `docs/spec/spec-phan-khanh.md`
  - Tái lập kết quả: `src/reproduce.py`, `src/export_results.py`, `reports/results_snapshot.md`
  - `DEFAULT_RF_PARAMS`, `RF_PARAM_GRID` trong `src/config.py`; `tests/test_pipeline.py`
  - `notebooks/02_*`, `figures/rf/`
  - Báo cáo: `docs/report/04-1-tien-xu-ly.md`, `04-2-random-forest.md`, `05-2-phan-tich-rf-ablation.md`
- **Mục tiêu:**
  1. Chứng minh và giải thích vì sao Random Forest là phương pháp chính.
  2. Xây khung thí nghiệm chung để các bạn cắm mô hình/phân tích của mình vào; viết các thí nghiệm
     cốt lõi: ablation, độ ổn định nhiều seed kèm kiểm định thống kê, calibration.
  3. Mọi con số tái lập được bằng một lệnh, gắn với tag `v1.0-results`, và báo cáo dùng đúng các số đó.

## Đã xong (24/11)

- [x] Split 70/15/15; thêm SVM_RBF; thêm các dòng `RF_Calibrated` vào bảng so sánh.
- [x] `src/reproduce.py` (một lệnh chạy lại toàn bộ) và `src/export_results.py`
  (`results/` → `reports/results_snapshot.md`); snapshot gắn với commit sạch.
- [x] `--quick` chỉ ghi snapshot vào `results/` (bị ignore), không ghi đè snapshot chính thức.
- [x] Tạo branch và plan cho từng người.

## Đã xong (25/11) — rà soát và làm chắc phần của Khánh

- [x] Spec tri thức toàn bộ phần của Khánh: `docs/spec/spec-phan-khanh.md` (giao thức, từng module,
  kết quả, hợp đồng tích hợp với từng bạn, hạn chế, runbook đến ngày bảo vệ).
- [x] Sửa lỗi sau review: model deploy chỉ còn một rừng (clone trước khi calibrate); `train.py` và
  `compare_datasets.py` dùng chung thứ tự cột IBM (cùng một RF); tham số tune lưu ở
  `models/tuned_params.json` (có trong git); calibration CV chia theo nhóm cho Iranian; ablation dùng
  RF đã tune; lựa chọn calibration đọc trên validation; thêm kiểm định t có hiệu chỉnh Nadeau–Bengio;
  sửa vạch "tuned" trên validation curve.
- [x] Điểm nối cho các bạn: `--quick` chạy trong sandbox `results/quick/` (không đổi file trong git);
  `reproduce` tự chạy `src.policy_analysis` khi có; fragment `results/<tên>/snapshot.md` tự vào
  snapshot §10; `fit_deployed_rf`, `split_data`; test contract tự kiểm tra mọi mô hình trong
  `BENCHMARK_MODELS`; cờ `--models`.
- [x] Chạy lại toàn bộ (`python -m src.reproduce`) và cập nhật snapshot, bản nháp 4.1/4.2/5.2, slide,
  câu trả lời bảo vệ.

## Đọc trước

- `PLAN.md`, `mustdo.md` (mục P0), `reports/results_snapshot.md`
- `src/train.py` bước 4–8 (định nghĩa RF, tuning, calibration, chọn ngưỡng)
- `src/compare_datasets.py`: `_build_model`, `_fit_model`, `_split_data`, `_group_split_indices`, `run_dataset`
- `src/evaluate.py` (ECE, lift), `src/explain.py` (SHAP gộp về đặc trưng gốc)
- `telco_churn_pipeline.md` mục 10.2 (bảng ablation A1–A4)

## Việc cần làm

### Tuần 0–1 (24–29/11) — mở khóa cho cả nhóm, khung thí nghiệm

- [ ] **Quản trị repo** (chỉ chủ repo làm được): mời 4 bạn làm collaborator (GitHub → *Settings →
  Collaborators*); bật *Settings → Branches → Branch protection* cho `main` (bắt buộc PR, 1 approval).
- [ ] **Dữ liệu:** chia sẻ 3 file CSV qua Drive riêng của nhóm (**trước 25/11**).
- [ ] **Buổi hướng dẫn kỹ thuật cho nhóm** (30–45 phút, **trước 28/11**): cấu trúc code, cách chạy
  `python -m src.reproduce --quick`, cách viết test, cách mở PR. Kết thúc buổi, mỗi bạn chạy được
  pipeline trên máy mình.
  - ⏳ Tài liệu cho buổi này đã có: `docs/huong-dan-ky-thuat.md` (cài đặt, cấu trúc code, công thức bắt đầu cho từng người).
- [x] **Khung benchmark 3 dataset** (**trước 27/11**):
  - `_fit_model` trong `src/compare_datasets.py`: khi có `--tune`, tune **mọi** mô hình có lưới tham số
    (lấy lưới theo tên mô hình; lưu ý tiền tố `classifier__estimator__` khi mô hình bị bọc bởi
    `CalibratedClassifierCV`).
  - Ghi thêm bảng dài `results/dataset_runs/model_comparison_long.csv`, mỗi dòng là (dataset, model)
    với các cột `accuracy`, `precision_churn`, `recall_churn`, `f1_churn`, `f1_macro`, `roc_auc`,
    `average_precision`, `train_time_s`, `best_params`; thêm section tương ứng trong `src/export_results.py`.
  - **Không đổi tên/xóa cột** của `dataset_comparison.csv`, vì snapshot §3 đang đọc nó. Viết test cho khung.
  - ✅ Xong (24/11): `BENCHMARK_MODELS`, `MODEL_PARAM_GRIDS`, `model_comparison_long.csv`, snapshot §5; CV khi tune trên Iranian chia theo nhóm.
- [x] **`src/ablation.py`** (chạy bằng `python -m src.ablation`, ghi `results/ablation/*.csv`):
  - **A2:** RF + SMOTE (dùng `imblearn.pipeline.Pipeline`, SMOTE chỉ áp trong fold train) vs RF +
    `class_weight` vs RF không xử lý mất cân bằng → F1/Recall/Precision lớp churn, PR-AUC.
  - **A3:** RF có vs không `StandardScaler` (kỳ vọng gần như giống hệt → chứng minh cây không cần scale).
  - A1 đã có (`results/dataset_runs/ibm_with_total_charges/`). Cùng split 70/15/15, cùng seed. Viết 1 test nhỏ.
  - ✅ Xong (24/11): 5 seed × 3 dataset, snapshot §6.

### Tuần 2 (30/11–06/12) — phân tích RF, độ ổn định, đóng băng số liệu

- [x] **Notebook `notebooks/02_random_forest_analysis.ipynb`** trên dataset chính (D1):
  - Validation curve theo `n_estimators` (50 → 500) và `max_depth` (3 → None): giải thích bias–variance
    và vì sao bộ tham số tuned (`best_params` trong `results/test_metrics.json`) hợp lý.
  - OOB score (`oob_score=True`) so với điểm CV.
  - Độ quan trọng đặc trưng theo 3 cách: MDI, permutation importance (trên test), SHAP. Lập bảng
    top-10 và giải thích vì sao các cách cho kết quả khác nhau.
  - Giải thích cục bộ bằng SHAP cho 3 khách hàng: TP rủi ro cao, FN bị bỏ sót, FP báo nhầm
    (`src/explain.explain_customer`, gọi trong `src/rf_analysis.py`). Lưu hình vào `figures/rf/`.
  - ✅ Xong (24/11): phân tích trong `src/rf_analysis.py` (cả 3 dataset), notebook hiển thị cho IBM — đổi `DATASET` sau khi chốt D1. Snapshot §9.
- [x] **`src/model_stability.py`** (chạy bằng `python -m src.model_stability`, ghi
  `results/model_stability/*.csv`):
  - 10 seed chia dữ liệu × 6 mô hình (LogReg, DT, RF, KNN, SVM_Linear, SVM_RBF) trên dataset chính,
    dùng `_build_model` (đủ 6 mô hình khi DT/KNN/SVM được thêm, ~01/12) và tham số đã tune, cố định.
  - Iranian dùng group-aware split cho từng seed (`_group_split_indices`).
  - Xuất mean ± std của Accuracy, Precision, Recall, F1 (lớp churn), ROC-AUC, AP.
  - ✅ Xong (24/11): hiện chạy LogReg và RF; tự nhận DT/KNN/SVM khi được thêm vào `BENCHMARK_MODELS`. Snapshot §7.
- [ ] **Chạy DT, KNN, SVM trên cả 3 bộ dữ liệu** (25/11–01/12, theo phân công 25/11): thêm nhánh vào
  `_build_model`, tên vào `BENCHMARK_MODELS`, lưới vào `MODEL_PARAM_GRIDS`; chạy
  `python -m src.compare_datasets --dataset all --tune` rồi `src.model_stability`. Thùy Linh duyệt lưới
  tham số (mở rộng nếu tham số tốt nhất nằm ở biên lưới, 04/12).
- [x] **Kiểm định thống kê** RF với mô hình mạnh nhất còn lại:
  - Wilcoxon signed-rank trên 10 seed (`scipy.stats.wilcoxon`).
  - McNemar trên tập test, tính từ các cặp bất đồng (`scipy.stats.binomtest`).
  - Ghi `results/model_stability/stat_tests.csv`, có cảnh báo về so sánh nhiều lần.
  - ✅ Xong (24/11): `results/model_stability/stat_tests.csv`, snapshot §7.
- [x] **Calibration của model deploy:**
  - Reliability diagram (`sklearn.calibration.CalibrationDisplay`) cho RF thô, RF đã calibrate,
    LogReg, SVM, kèm bảng Brier/ECE (`figures/rf/`).
  - `models/rf_best.joblib` ≈ 91 MB vì `CalibratedClassifierCV(cv=5)` lưu 5 bản RF cộng một rừng thừa đã fit: so sánh với
    `ensemble=False` (1 RF + 1 sigmoid) về metric và kích thước; chốt phương án, báo Hải Anh để deploy.
  - ✅ Đã chốt (24/11): `CALIBRATION_ENSEMBLE = False` — chất lượng như 5 rừng; sau khi bỏ rừng thừa (25/11), model IBM 91 MB → 17 MB (snapshot §8). **Cần báo Hải Anh.**
- [ ] **Review PR code thí nghiệm** của các bạn và nối vào `src/reproduce.py` + `src/export_results.py`:
  - Thùy Linh: lưới tham số cuối của DT/KNN/SVM (04/12), `tests/test_models.py`.
  - Nhật Linh: `src/policy_analysis.py` (05/12).
  - Quỳnh Anh: `src/download_data.py` (29/11).
  - ⏳ Các script của Khánh đã nối vào `src/reproduce.py` + snapshot §5–§9; chờ PR của Thùy Linh, Nhật Linh, Quỳnh Anh.
- [x] (Tùy chọn) Viết hàm top-factors theo SHAP từng khách cho API, thay cho feature importance toàn
  cục trong `api/predict.py::_get_top_factors`; bàn giao cho Hải Anh.
  - ✅ Xong (24/11): `src/explain.py::explain_customer` — chạy được với model đã calibrate, ~0.5 giây/khách; hướng dẫn dùng cho Hải Anh ở `docs/huong-dan-ky-thuat.md` mục 5.
- [ ] **06/12 — Đóng băng số liệu:**
  1. Trên `main` sạch, đã có `COST_*` cuối cùng: chạy `python -m src.reproduce`.
  2. Commit `reports/results_snapshot.md`, `models/config.json`, `models/metadata.json`,
     `models/tuned_params.json`, `figures/rf/`, `figures/dataset_runs/`.
  3. Chạy `git tag v1.0-results && git push origin v1.0-results`.
  4. Báo cả nhóm: từ giờ chỉ lấy số từ snapshot của tag này.

### Tuần 3 (07–13/12) — viết

- [x] **Mục 4.1 – Tiền xử lý & chống rò rỉ:**
  - Thứ tự split → fit pipeline chỉ trên train.
  - `TelcoFeatureEngineer`, imputation, one-hot, scaling theo từng loại mô hình.
  - Group-aware split cho Iranian.
  - Seed, SHA-256 của dữ liệu, `src/reproduce.py` và tag kết quả.
  - ✅ Bản nháp: `docs/report/04-1-tien-xu-ly.md`.
- [x] **Mục 4.2 – Random Forest:** bagging, chọn ngẫu nhiên đặc trưng, tiêu chí Gini/entropy, OOB,
  feature importance, `class_weight="balanced_subsample"`, vì sao cần calibrate xác suất (Platt).
  - ✅ Bản nháp: `docs/report/04-2-random-forest.md`.
- [x] **Mục 5.2 – Phân tích RF & ablation:** kết quả notebook 02, bảng A1–A3, nhận xét. Nói thẳng:
  trên IBM, RF không vượt LogReg về ROC-AUC/AP; lợi thế của RF thể hiện trên Orange/Iranian
  (phối hợp với mục 5.4 của Thùy Linh).
  - ✅ Bản nháp với số liệu thật: `docs/report/05-2-phan-tich-rf-ablation.md` (cập nhật số theo tag `v1.0-results`).
- [ ] **Bàn giao và hỗ trợ:**
  - Cho Thùy Linh: kết quả độ ổn định và kiểm định (giải thích cách đọc file CSV/hình), dùng ở mục 5.1.
  - Cho Nhật Linh: reliability diagram cho phần calibration ở mục 5.3; cách dùng model RF đã calibrate
    và các hàm chia dữ liệu/dựng mô hình.
- [ ] Làm 2–3 slide phần Random Forest theo khung của Hải Anh.
  - ⏳ Nội dung nháp (kèm ghi chú thuyết trình): `docs/slides/namkhanh-random-forest.md`; chờ khung slide của Hải Anh.

### Tuần 4 (14–20/12)

- [ ] **Đối chiếu số liệu** (trước 18/12): kiểm tra mọi con số trong `docs/report/` và slide khớp
  `reports/results_snapshot.md` của tag `v1.0-results`; gửi danh sách chỗ lệch cho người viết sửa
  và báo Hải Anh.
- [ ] **Thử tái lập trên máy sạch:** nhờ 1 bạn clone mới → đặt dữ liệu → `python -m src.reproduce`
  → so snapshot với tag `v1.0-results`.
  - ⏳ Đã thử trên một bản clone sạch ở máy Khánh (24/11, chế độ `--quick`): đạt. Làm lại với tag `v1.0-results` ở tuần 4.
- [ ] Review chéo mục 4.4, 5.3, 7.2 (Nhật Linh).
- [ ] Viết câu trả lời phần mình cho `docs/defense_qa.md`. Tham gia 2 buổi tổng duyệt.
  - ⏳ Câu trả lời đã có: `docs/defense_qa/namkhanh.md`; còn 2 buổi tổng duyệt.

## Phụ thuộc

- **Cần từ:**
  - Thùy Linh: ý kiến giảng viên về D1 — dataset chính cho phân tích RF (29/11); lưới tham số cuối của
    DT/KNN/SVM (04/12).
  - Nhật Linh: `COST_*` sau khi chốt D3 (04/12); `src/policy_analysis.py` (05/12).
  - Quỳnh Anh: `src/download_data.py` (29/11).
- **Cung cấp cho:**
  - Cả nhóm: quyền truy cập và dữ liệu (25/11); buổi hướng dẫn kỹ thuật (28/11); tag `v1.0-results`
    (06/12); danh sách chỗ lệch số liệu (18/12).
  - Thùy Linh: kết quả DT/KNN/SVM trên 3 dataset (01/12), kết quả độ ổn định và kiểm định.
  - Nhật Linh: reliability diagram, model RF đã calibrate, hướng dẫn dùng các hàm chia dữ liệu/dựng mô hình.
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
python -m src.reproduce                               # chạy final (~40 phút khi máy rảnh; lượt 25/11 mất 3 giờ 27 phút vì máy chạy việc khác)
python -m src.reproduce --quick                       # kiểm tra nhanh, ghi vào results/quick/ (không đổi file trong git)
python -m pytest -q
git fetch origin && git merge origin/main
```
