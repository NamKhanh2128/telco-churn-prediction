# Plan — Mai Thùy Linh · Các mô hình so sánh: Decision Tree, KNN, SVM

- **Branch:** `thuylinh/model-comparison`
- **Vùng file phụ trách:**
  - Phần DT/KNN/SVM trong `_build_model` (`src/compare_datasets.py`); `DEFAULT_*_PARAMS` và
    `*_PARAM_GRID` của DT, KNN, SVM trong `src/config.py`; `tests/test_models.py` (mới)
  - `notebooks/03_*`, `figures/models/`, `docs/report/references.md`
  - `docs/report/04-3-dt-knn-svm.md`, `05-1-so-sanh-mo-hinh.md`, `05-4-ba-dataset.md`
- **Mục tiêu:** tự cài đặt và chịu trách nhiệm về các mô hình so sánh mà đề tài yêu cầu (**DT, KNN,
  SVM** so với RF, theo Accuracy, Precision, Recall, F1) trên cả 3 dataset: code chạy đúng, siêu tham số
  hợp lý, bảng và hình rõ ràng, lý thuyết và phân tích chặt chẽ.

## Hiện trạng

- `src/train.py` (IBM) đã chạy đủ Dummy, LogReg, DecisionTree, RandomForest, KNN, SVM_Linear,
  SVM_RBF, có tuning khi chạy `--tune` → dùng làm mẫu.
- `src/compare_datasets.py` (3 dataset) mới chạy Dummy, LogReg, RandomForest. Nam Khánh làm khung tune
  mọi mô hình và bảng dài `model_comparison_long.csv` (~27/11); bạn thêm DT/KNN/SVM vào khung đó.
- Nam Khánh viết `src/model_stability.py` (10 seed × 6 mô hình, kèm kiểm định thống kê), sinh
  `results/model_stability/`.

## Đọc trước

- `PLAN.md`, `reports/results_snapshot.md` §1 và §3
- `src/config.py` (các `DEFAULT_*_PARAMS`, `*_PARAM_GRID`), `src/train.py` bước 4–6 (mẫu định nghĩa
  DT/KNN/SVM), `src/compare_datasets.py` (`_build_model`, `_fit_model`), `src/evaluate.py::compute_metrics`

## Việc cần làm

### Tuần 1 (24–29/11) — cài mô hình, lý thuyết

- [ ] Chạy `python -m src.train` và `python -m src.compare_datasets --dataset all`, đọc
  `results/comparison_table.csv` và `results/test_metrics.json` (có `best_params`).
- [ ] **Đề xuất D2** (giao thức đánh giá), bàn cùng Nam Khánh, gửi Hải Anh **trước 27/11**: chia
  70/15/15, tuning 5-fold CV theo AP, số seed cho phân tích ổn định, cách xử lý dòng trùng của Iranian.
- [ ] **Cài DT, KNN, SVM vào benchmark 3 dataset** (sau khi khung của Nam Khánh xong ~27/11; PR
  **trước 01/12**):
  - Thêm `DecisionTree`, `KNN`, `SVM_Linear`, `SVM_RBF` vào `_build_model` trong `src/compare_datasets.py`,
    dùng `DEFAULT_*_PARAMS` trong `config.py`, lấy `src/train.py` làm mẫu:
    - KNN/SVM: `scale_numeric=True`.
    - SVM bọc `CalibratedClassifierCV` để có xác suất.
  - Chạy `python -m src.compare_datasets --dataset all --tune`, kiểm tra đủ 6 mô hình × 3 dataset trong
    `results/dataset_runs/model_comparison_long.csv`.
  - `tests/test_models.py`: mỗi mô hình fit được trên dữ liệu nhỏ và `predict_proba` nằm trong [0, 1].
- [ ] **Viết nháp mục 4.3 – DT, KNN, SVM.** Phần này không phụ thuộc kết quả, nên viết sớm:
  - DT: tiêu chí chia Gini/entropy, độ sâu, tỉa cây (`ccp_alpha`), vì sao dễ overfit.
  - KNN: tham số k, độ đo khoảng cách, `weights`, vì sao bắt buộc chuẩn hóa.
  - SVM: lề cực đại, tham số `C`, kernel RBF và `gamma`, vì sao cần Platt scaling để có xác suất.
- [ ] **Danh mục tài liệu tham khảo chung** `docs/report/references.md`: tạo file, chọn một kiểu
  trích dẫn (APA hoặc IEEE). Các bạn gửi tài liệu của phần mình để bạn gộp (chốt trước 16/12).
  Tài liệu nền cho mục 4.3:
  - Breiman et al. (1984), *Classification and Regression Trees*
  - Cover & Hart (1967), *Nearest neighbor pattern classification*
  - Cortes & Vapnik (1995), *Support-vector networks*
  - Platt (1999), *Probabilistic outputs for support vector machines*

### Tuần 2 (30/11–06/12) — siêu tham số, bảng và hình

- [ ] **Rà siêu tham số:**
  - Đọc `best_params` của DT/KNN/SVM trong `model_comparison_long.csv` và `results/test_metrics.json`.
  - Nếu tham số tốt nhất nằm ở biên lưới thì mở rộng `*_PARAM_GRID` trong `src/config.py`:
    DT thêm `ccp_alpha`; KNN thêm `n_neighbors` 31, 51; SVM RBF mở rộng `C`/`gamma`.
  - Gửi PR nhỏ **trước 04/12** để kịp tag `v1.0-results`.
- [ ] **Notebook `notebooks/03_model_comparison.ipynb`** → `figures/models/`:
  - Bảng chính trên dataset chính: Accuracy, Precision, Recall, F1 (lớp churn) + ROC-AUC, PR-AUC.
  - Bảng siêu tham số tốt nhất của từng mô hình.
  - Heatmap mô hình × dataset theo F1 và ROC-AUC, từ `model_comparison_long.csv`.
  - Biểu đồ cột 4 độ đo chính cho 6 mô hình, error bar = std qua 10 seed (từ `results/model_stability/`
    của Nam Khánh).
- [ ] **So sánh với baseline Chen et al. (2025)** trên IBM: tự tìm bài gốc, xác minh tác giả, nơi
  đăng và con số baseline (đang ghi ở `telco_churn_pipeline.md` mục 2.2). Không truy được nguồn thì
  bỏ phần so sánh này. Nếu xác minh được: lập bảng "nhóm vs bài báo" và giải thích chênh lệch:
  pipeline chống rò rỉ, cách chia dữ liệu, ngưỡng quyết định.

### Tuần 3 (07–13/12) — viết

- [ ] **Mục 5.1 – Thiết lập & so sánh mô hình:** giao thức (split, CV, tuning, seed), bảng chính
  (mean ± std), thảo luận:
  - Accuracy dễ gây hiểu lầm: Dummy vẫn đạt ~0.73 trên IBM.
  - Đánh đổi Precision/Recall ở ngưỡng 0.5: DT/RF recall cao nhờ `class_weight`, SVM/LogReg precision
    cao hơn.
  - Thời gian huấn luyện.
  - Kết quả kiểm định thống kê (Nam Khánh cung cấp).
- [ ] **Mục 5.4 – So sánh trên 3 dataset:** mô hình nào thắng ở dataset nào và vì sao. Dữ liệu nhiều
  biến phân loại (IBM) thì mô hình tuyến tính đủ tốt; dữ liệu nhiều biến định lượng/phi tuyến
  (Orange, Iranian) thì cây/ensemble vượt trội. Phối hợp với Nam Khánh (mục 5.2).
- [ ] Hoàn thiện mục 4.3.
- [ ] Làm 2–3 slide phần so sánh mô hình theo khung của Hải Anh.

### Tuần 4 (14–20/12)

- [ ] Review chéo chương 2–3 (Quỳnh Anh).
- [ ] Sửa các mục của mình theo góp ý của Hải Anh; viết câu trả lời phần mình cho `docs/defense_qa.md`.

## Phụ thuộc

- **Cần từ:**
  - Hải Anh: D1 (dataset chính).
  - Nam Khánh: khung benchmark (27/11); kết quả độ ổn định và kiểm định thống kê.
  - Các bạn: tài liệu tham khảo của phần mình (trước 16/12).
- **Cung cấp cho:**
  - Hải Anh: đề xuất D2 (27/11).
  - Nam Khánh: DT/KNN/SVM trong benchmark (01/12); lưới tham số cuối (04/12); bảng 3 dataset cho lập
    luận chọn RF.
  - Cả nhóm: `docs/report/references.md`.

## Câu hỏi bảo vệ cần chuẩn bị

1. Vì sao KNN và SVM phải chuẩn hóa dữ liệu còn cây quyết định thì không?
2. Kernel RBF hoạt động thế nào? `C` và `gamma` ảnh hưởng ra sao tới overfitting?
3. SVM không cho xác suất; nhóm lấy xác suất bằng cách nào (Platt scaling)?
4. Vì sao Accuracy không phù hợp làm độ đo chính khi dữ liệu mất cân bằng?
5. Vì sao DT có recall cao nhưng precision thấp?
6. Vì sao tuning bằng `RandomizedSearchCV` với scoring là AP? Vì sao không được tune trên tập test?
7. Kết quả khác bài báo Chen et al. vì sao? Chênh lệch giữa các mô hình có ý nghĩa thống kê không?

## Lệnh hay dùng

```bash
python -m src.compare_datasets --dataset all            # nhanh, không tune
python -m src.compare_datasets --dataset all --tune     # đầy đủ
python -m pytest tests/test_models.py -q
jupyter notebook notebooks/
git fetch origin && git merge origin/main
```
