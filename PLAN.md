# PLAN TỔNG — Dự đoán khả năng khách hàng rời bỏ dịch vụ viễn thông

> **Đề tài:** Xây dựng mô hình học máy dự đoán khả năng khách hàng rời bỏ dịch vụ viễn thông
> dựa trên thông tin và hành vi sử dụng dịch vụ (thời gian đăng ký, tần suất sử dụng, mức phí,
> khiếu nại, gói cước). **Random Forest là phương pháp chính**, so sánh với **Decision Tree, KNN
> và SVM**. Đánh giá bằng **Accuracy, Precision, Recall, F1-score**.
>
> Cập nhật: 24/11/2026 · Người điều phối: Lê Hải Anh · Kỹ thuật/tích hợp: Nguyễn Đức Nam Khánh

---

## 1. Thành viên, vai trò và branch

Mỗi người làm **chỉ trên branch của mình**, mở Pull Request (PR) vào `main` khi xong từng phần.
Plan chi tiết của từng người nằm ở file `plans/<tên>.md` **trên branch của người đó**.

| Thành viên | Vai trò | Branch | Plan chi tiết |
|---|---|---|---|
| **Lê Hải Anh** (trưởng nhóm) | Điều phối; Random Forest; ablation, độ ổn định nhiều seed, phân tích ngưỡng; ráp báo cáo | `haianh/rf-main-model` | `plans/haianh.md` |
| **Quách Nguyễn Quỳnh Anh** | Dữ liệu, EDA, nguồn gốc dữ liệu, tổng quan tài liệu | `quynhanh/data-eda` | `plans/quynhanh.md` |
| **Mai Thùy Linh** | Các mô hình so sánh DT/KNN/SVM: siêu tham số, phân tích và trình bày kết quả | `thuylinh/model-comparison` | `plans/thuylinh.md` |
| **Nguyễn Nhật Linh** | Đánh giá, chi phí và chính sách quyết định, góc nhìn kinh doanh; khung slide | `nhatlinh/evaluation-business` | `plans/nhatlinh.md` |
| **Nguyễn Đức Nam Khánh** | Hạ tầng và tái lập; benchmark các mô hình trên 3 dataset; API, giao diện, deploy | `namkhanh/infra-deploy` | `plans/namkhanh.md` |

Xem plan của mình: `git checkout <branch>` rồi mở `plans/<tên>.md`, hoặc chọn branch trên GitHub.

---

## 2. Hiện trạng (đã có trên `main`)

- Pipeline leak-free: feature engineering, imputation, scaling nằm **trong** pipeline (`src/preprocessing.py`).
- Chia dữ liệu **70/15/15** (train/validation/test). Ngưỡng chỉ được chọn trên validation.
- IBM Telco: 7 mô hình (Dummy, LogReg, DecisionTree, RandomForest, KNN, SVM_Linear, SVM_RBF) +
  RF đã calibrate (Platt), có tuning (`python -m src.train --tune --with-shap`).
- Benchmark 3 dataset độc lập: IBM Telco, Orange Telecom, Iranian Churn (`src/compare_datasets.py`) —
  hiện mới so **RF với LogReg**.
- Audit Iranian: 300 dòng trùng hoàn toàn, 14 nhóm trùng đặc trưng nhưng khác nhãn → dùng
  **group-aware split**; độ ổn định 10 seed (`src/iranian_audit.py`).
- Ablation A1 (giữ/bỏ `TotalCharges`), SHAP, FastAPI + frontend, Dockerfile.
- **Một lệnh chạy lại toàn bộ:** `python -m src.reproduce` → sinh
  [`reports/results_snapshot.md`](reports/results_snapshot.md). **Mọi con số trong báo cáo lấy từ file này.**
- 20 unit test (`python -m pytest -q`).

### Phát hiện quan trọng định hướng công việc

1. **Trên IBM, RF không thắng rõ ràng**: LogReg/SVM tuyến tính ngang hoặc nhỉnh hơn về ROC-AUC/AP
   (snapshot §1). RF vượt trội rõ trên **Orange** và **Iranian** (snapshot §3). Báo cáo phải trình
   bày trung thực điều này và lập luận cho RF dựa trên dữ liệu có biến định lượng/phi tuyến.
2. **Iranian là dataset duy nhất có đúng các đặc trưng đề tài nêu** (khiếu nại `Complains`, tần suất
   sử dụng `Frequency of use`, thời gian đăng ký `Subscription Length`, mức phí `Charge Amount`,
   gói cước `Tariff Plan`) → cần hỏi giảng viên có chọn Iranian làm dataset chính không (D1).
3. **Chi phí FP = $50, FN = $500 chỉ là giả định** → ngưỡng tối ưu trên IBM ≈ 0.06, demo gắn nhãn
   "Churn" cả khách hàng hợp đồng 2 năm, xác suất 8.5%. Phải chốt chi phí/chính sách (D3, D4).
4. **Chưa có:** notebook EDA, so sánh DT/KNN/SVM trên cả 3 dataset, ablation A2–A4, phân tích độ
   nhạy ngưỡng, báo cáo, slide, link deploy công khai.

---

## 3. Các quyết định cần chốt (gate của cả dự án)

Hải Anh mang bảng này đi gặp giảng viên **trước 29/11**, rồi điền cột "Kết quả" bằng một PR nhỏ vào `main`.

| # | Quyết định | Phương án | Đề xuất bởi | Hạn | Kết quả |
|---|---|---|---|---|---|
| D1 | Dataset chính | IBM Telco (hiện tại) **hoặc** Iranian (khớp mô tả đề tài); 2 dataset còn lại làm benchmark phụ | Hải Anh + Quỳnh Anh | 29/11 | ☐ |
| D2 | Giao thức đánh giá | 70/15/15 stratified; Iranian dùng group-aware; tuning 5-fold CV theo AP | Thùy Linh | 29/11 | ☐ |
| D3 | Chi phí FP/FN và cách tính | classification cost (FP·c_FP + FN·c_FN) **hoặc** action cost (số KH được liên hệ·c_action + FN·c_FN); 2–3 kịch bản có căn cứ | Nhật Linh | 29/11 | ☐ |
| D4 | Chính sách quyết định cho demo | ngưỡng tối ưu chi phí / top-k theo ngân sách / 3 mức rủi ro | Nhật Linh + Nam Khánh | 04/12 | ☐ |
| D5 | Kênh deploy | Render (Docker) / Hugging Face Spaces (Docker) | Nam Khánh | 04/12 | ☐ |
| D6 | Định dạng báo cáo nộp | Word / LaTeX (Overleaf) — theo yêu cầu của giảng viên | Hải Anh | 29/11 | ☐ |

**Nếu D1 = Iranian**, ảnh hưởng: Nam Khánh đổi schema API/frontend; Hải Anh phân tích RF/SHAP trên
Iranian; Thùy Linh lấy Iranian làm bảng so sánh chính; Quỳnh Anh làm data dictionary cho Iranian.

---

## 4. Lộ trình 4 tuần

> Ngày dưới đây tính từ 24/11/2026. **Nếu hạn nộp khác, Hải Anh dời toàn bộ mốc tương ứng.**

| Tuần | Thời gian | Mốc | Hoàn thành khi |
|---|---|---|---|
| 0 | 24–25/11 | **M0 – Sẵn sàng** | Cả 5 người có quyền push, đủ 3 file dữ liệu (đúng SHA-256), chạy được `python -m src.reproduce --quick`, đã checkout branch của mình |
| 1 | 24/11–29/11 | **M1 – Chốt giao thức** | D1, D2, D3, D6 có kết quả; EDA bản nháp; khung báo cáo (`docs/report/00-outline.md`) |
| 2 | 30/11–06/12 | **M2 – Đóng băng số liệu** | Mọi code thực nghiệm đã merge; Nam Khánh chạy `python -m src.reproduce` trên `main` sạch → tag **`v1.0-results`** |
| 3 | 07/12–13/12 | **M3 – Bản nháp đầy đủ** | Tất cả chương nháp đã merge vào `docs/report/`; có URL demo công khai |
| 4 | 14/12–20/12 | **M4 – Nộp & bảo vệ** | Review chéo xong; báo cáo định dạng cuối; slide; 2 buổi tổng duyệt |

**Quy tắc số liệu:** sau tag `v1.0-results`, báo cáo/slide **chỉ** dùng số trong
`reports/results_snapshot.md` của tag đó. Nếu phải sửa code thí nghiệm sau M2 → chạy lại toàn bộ,
tạo tag `v1.1-results`, cập nhật lại mọi con số.

### Phụ thuộc giữa các thành viên

```
Hải Anh (D1, D6) ─┬─► Quỳnh Anh: EDA/data dictionary cho dataset chính
                  ├─► Thùy Linh: bảng so sánh chính trên dataset chính
                  └─► Nam Khánh: schema API/frontend (nếu đổi dataset)
Nam Khánh (benchmark DT/KNN/SVM × 3 dataset, ~01/12) ─┬─► Hải Anh: độ ổn định 10 seed + kiểm định
                                                      └─► Thùy Linh: bảng/hình so sánh mô hình
Nhật Linh (D3) ─► COST_* ─► Hải Anh: phân tích ngưỡng ─► Nhật Linh: viết mục 5.3
Hải Anh (độ ổn định, kiểm định) ─► Thùy Linh: viết mục 5.1
Code thí nghiệm merge trước 05/12 ─► Nam Khánh: tag v1.0-results ─► tất cả: số liệu vào báo cáo/slide
```

---

## 5. Báo cáo — ai viết phần nào

Viết nháp bằng Markdown trong `docs/report/` (mỗi người **chỉ tạo file của mình** → không xung đột
khi merge). Tuần 4, Hải Anh chuyển sang định dạng nộp (D6).

| Chương | Nội dung | Người viết | File nháp |
|---|---|---|---|
| 0 | Dàn ý, quy ước trình bày, danh sách hình/bảng | Hải Anh | `docs/report/00-outline.md` |
| 1 | Giới thiệu: bài toán, động cơ, mục tiêu, phạm vi | Hải Anh | `docs/report/01-gioi-thieu.md` |
| 2 | Tổng quan nghiên cứu liên quan | Quỳnh Anh | `docs/report/02-tong-quan.md` |
| 3 | Dữ liệu và phân tích khám phá (EDA), audit dữ liệu | Quỳnh Anh | `docs/report/03-du-lieu-eda.md` |
| 4.1 | Tiền xử lý & pipeline chống rò rỉ dữ liệu | Nam Khánh | `docs/report/04-1-tien-xu-ly.md` |
| 4.2 | Random Forest (lý thuyết, siêu tham số, calibration) | Hải Anh | `docs/report/04-2-random-forest.md` |
| 4.3 | Decision Tree, KNN, SVM | Thùy Linh | `docs/report/04-3-dt-knn-svm.md` |
| 4.4 | Độ đo đánh giá, calibration, ngưỡng theo chi phí | Nhật Linh | `docs/report/04-4-danh-gia-nguong.md` |
| 5.1 | Thiết lập thực nghiệm & so sánh các mô hình | Thùy Linh | `docs/report/05-1-so-sanh-mo-hinh.md` |
| 5.2 | Phân tích Random Forest & ablation | Hải Anh | `docs/report/05-2-phan-tich-rf-ablation.md` |
| 5.3 | Ngưỡng quyết định & phân tích kinh doanh | Nhật Linh | `docs/report/05-3-nguong-chi-phi.md` |
| 5.4 | So sánh trên 3 dataset | Thùy Linh | `docs/report/05-4-ba-dataset.md` |
| 6 | Hệ thống demo: API, giao diện, triển khai | Nam Khánh | `docs/report/06-he-thong.md` |
| 7.1 | Kết luận & hướng phát triển | Hải Anh | `docs/report/07-1-ket-luan.md` |
| 7.2 | Hạn chế của nghiên cứu | Nhật Linh | `docs/report/07-2-han-che.md` |

**Review chéo (tuần 4):** Hải Anh ↔ Nhật Linh · Quỳnh Anh → chương 6 (Nam Khánh) ·
Thùy Linh → chương 2–3 (Quỳnh Anh) · Nam Khánh → chương 4.3, 5.1, 5.4 (Thùy Linh).

**Slide:** mỗi người làm 2–3 slide cho phần của mình; Nhật Linh làm khung slide (trước 06/12) và ghép bản cuối.

**Câu hỏi bảo vệ (`docs/defense_qa.md`):** mỗi người viết câu trả lời cho các câu hỏi trong plan
của mình; Thùy Linh tổng hợp thành một file.

---

## 6. Phân vùng file (tránh xung đột khi merge)

| Vùng | Chủ sở hữu | Người khác muốn sửa thì… |
|---|---|---|
| `notebooks/01_*`, `figures/eda/`, `data/raw/**/README.md`, `docs/data_dictionary.md`, `docs/report/references.md` | Quỳnh Anh | nhắn Quỳnh Anh |
| `notebooks/02_*`, `src/ablation.py`, `src/model_stability.py`, `src/policy_analysis.py`, `src/explain.py`, `src/optimize_threshold.py`, `src/evaluate.py`, `DEFAULT_RF_PARAMS`/`RF_PARAM_GRID` trong `src/config.py`, `figures/rf/`, `figures/policy/` | Hải Anh | nhắn Hải Anh |
| `DEFAULT_*_PARAMS`/`*_PARAM_GRID` của DT, KNN, SVM trong `src/config.py`; `notebooks/03_*`, `figures/models/`, `docs/defense_qa.md` | Thùy Linh | nhắn Thùy Linh |
| `COST_*` trong `src/config.py`, `notebooks/04_*`, `docs/slides/` | Nhật Linh | nhắn Nhật Linh |
| `src/train.py`, `src/compare_datasets.py`, `src/preprocessing.py`, `src/data_loader.py`, `src/reproduce.py`, `src/export_results.py`, `api/`, `frontend/`, `Dockerfile`, `docker-compose.yml`, `requirements*.txt`, `tests/`, `.github/`, `README.md`, `.gitignore` | Nam Khánh | nhắn Nam Khánh |
| `PLAN.md`, `mustdo.md` | Hải Anh | PR nhỏ, Hải Anh duyệt |

Script phân tích mới chỉ cần ghi CSV vào `results/<vùng>/`. Nam Khánh sẽ nối nó vào
`src/reproduce.py` và thêm section vào `reports/results_snapshot.md` (qua `src/export_results.py`).

---

## 7. Cách làm việc với Git

### Bước 0 — Chuẩn bị (một lần)

1. Nam Khánh mời cả nhóm làm collaborator: GitHub → *Settings → Collaborators → Add people*.
2. Cài đặt:

   ```bash
   git clone https://github.com/NamKhanh2128/telco-churn-prediction.git
   cd telco-churn-prediction
   python -m venv venv
   venv\Scripts\activate            # Windows  (Linux/Mac: source venv/bin/activate)
   pip install -r requirements.txt
   git checkout <branch-của-bạn>     # ví dụ: git checkout quynhanh/data-eda
   ```

3. **Dữ liệu không nằm trong git** (`*.csv` bị ignore). Lấy 3 file (Nam Khánh chia sẻ qua Drive nhóm),
   đặt đúng đường dẫn và kiểm tra SHA-256:

   | Dataset | Đường dẫn | SHA-256 |
   |---|---|---|
   | IBM Telco | `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |
   | Iranian Churn | `data/raw/iranian-uci/Customer Churn.csv` | `90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1` |
   | Orange Telecom | `data/raw/orange-telecom/telecom_churn.csv` | `5d0aecc74b6feda887e9d2735363d51ff671b373153863562b7bbfb36464e0a4` |

   ```powershell
   Get-FileHash "data\raw\iranian-uci\Customer Churn.csv" -Algorithm SHA256
   ```

4. Kiểm tra môi trường: `python -m src.reproduce --quick` (vài phút; phải báo `completed`).

### Hằng ngày

```bash
git pull                              # cập nhật branch của mình
git fetch origin
git merge origin/main                 # lấy thay đổi mới của main — ít nhất 2 lần/tuần
# ... làm việc ...
python -m pytest -q                   # phải pass trước khi push
git add <đúng-file-của-mình>          # KHÔNG dùng "git add -A"
git commit -m "Add EDA notebook for Iranian dataset"
git push
```

### Pull Request vào `main`

- PR nhỏ, một chủ đề (ví dụ: "Add KNN/SVM to 3-dataset benchmark"). Mô tả: làm gì, chạy lệnh gì để kiểm tra.
- Reviewer: **code → Nam Khánh**, **báo cáo/tài liệu → Hải Anh**. Cần 1 approval mới merge.
- `python -m pytest -q` phải pass.

### Không bao giờ commit

- Dữ liệu (`data/**/*.csv`), model (`*.joblib`), `results/` — đã nằm trong `.gitignore`.
- Secret: `.env`, token, mật khẩu, file credential.
- **Artifact dùng chung do pipeline sinh ra:** `reports/results_snapshot.md`, `models/config.json`,
  `models/metadata.json`, `figures/dataset_runs/*.png` — chỉ Nam Khánh commit sau lần chạy final
  trên `main`. `--quick` không đụng tới snapshot (ghi vào `results/results_snapshot_quick.md`),
  nhưng mọi lần chạy `src.train`/`src.compare_datasets` vẫn ghi đè 3 file còn lại. Trước khi commit,
  khôi phục chúng:

  ```bash
  git restore reports/results_snapshot.md models/config.json models/metadata.json figures/dataset_runs
  ```

---

## 8. Checklist nghiệm thu cuối (Definition of Done)

- [ ] D1–D6 đã chốt và ghi trong mục 3.
- [ ] Bảng so sánh RF vs DT vs KNN vs SVM (Accuracy, Precision, Recall, F1 + ROC-AUC/PR-AUC) trên dataset chính, có mean ± std qua nhiều seed.
- [ ] Ablation A1 (TotalCharges), A2 (SMOTE vs class_weight), A3 (scale), A4 (ngưỡng 0.5 vs tối ưu).
- [ ] Phân tích ngưỡng với chi phí đã duyệt + độ nhạy theo tỉ lệ chi phí và seed.
- [ ] EDA, data dictionary, nguồn gốc/license/citation của cả 3 dataset.
- [ ] Mọi số trong báo cáo/slide khớp `reports/results_snapshot.md` tại tag `v1.0-results` (hoặc tag mới hơn).
- [ ] Demo công khai chạy từ đúng model artifact của tag kết quả; có phương án demo offline dự phòng.
- [ ] README, citation khớp dataset và kết quả thực tế.
- [ ] Mỗi thành viên trả lời được câu hỏi bảo vệ trong plan của mình (`docs/defense_qa.md`).

Chi tiết từng việc: xem plan con trên branch của từng người. Danh sách việc thủ công còn mở
(P0/P1) tiếp tục được theo dõi trong [`mustdo.md`](mustdo.md).
