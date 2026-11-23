# Plan — Quách Nguyễn Quỳnh Anh · Dữ liệu, EDA & Tổng quan tài liệu

- **Branch:** `quynhanh/data-eda`
- **Vùng file phụ trách:** `notebooks/01_*`, `figures/eda/`, `data/raw/**/README.md`,
  `docs/data_dictionary.md`, `docs/report/02-tong-quan.md`, `docs/report/03-du-lieu-eda.md`,
  `docs/report/references.md`
- **Mục tiêu:** hiểu và mô tả dữ liệu tốt nhất nhóm; mọi quyết định tiền xử lý/mô hình phải truy
  ngược được về một phát hiện EDA; nguồn gốc, license và trích dẫn của dữ liệu và tài liệu phải chuẩn.

## Đọc trước

- `PLAN.md`, `reports/results_snapshot.md`
- `src/compare_datasets.py::load_dataset` (cách đọc và làm sạch 3 dataset), `src/data_loader.py`
- `data/raw/README.md`, `data/raw/iranian-uci/README.md`, `data/raw/orange-telecom/README.md`
- `results/dataset_runs/iranian/audit_report.json` và `feature_audit.csv` (sinh ra sau khi chạy
  `python -m src.iranian_audit`)

## Việc cần làm

### Tuần 1 (24–29/11) — EDA

- [ ] **Notebook EDA** — một notebook cho mỗi dataset: `notebooks/01a_eda_ibm.ipynb`,
  `01b_eda_iranian.ipynb`, `01c_eda_orange.ipynb`. Làm Iranian và IBM trước vì D1 chọn giữa hai bộ
  này. Sau khi chốt D1: **làm đầy đủ cho dataset chính**; 2 dataset phụ làm gọn (tổng quan, mất cân
  bằng, 3–4 hình quan trọng nhất). Notebook đầy đủ gồm:
  - Kích thước, kiểu dữ liệu, giá trị thiếu. IBM: `TotalCharges` có 11 ô trống, đều ở khách
    hàng `tenure = 0` → giải thích vì sao imputation nằm trong pipeline.
  - Dòng trùng lặp. Iranian: 300 dòng trùng hoàn toàn, 14 nhóm trùng đặc trưng nhưng khác nhãn
    (lấy từ `audit_report.json`).
  - Phân bố nhãn và mức mất cân bằng (IBM ≈ 26.5%, Iranian ≈ 15.7%, Orange ≈ 14.5% churn).
  - Tỉ lệ churn theo từng biến phân loại, ví dụ:
    - IBM: `Contract`, `InternetService`, `PaymentMethod`, `TechSupport`
    - Iranian: `Complains`, `Tariff Plan`, `Status`, `Age Group`
    - Orange: `International plan`, `Voice mail plan`, `Customer service calls`
  - Phân bố biến số theo nhãn (boxplot/KDE): `tenure`, `MonthlyCharges`; `Subscription Length`,
    `Frequency of use`, `Charge Amount`, `Seconds of Use`; `Total day minutes`, `Customer service calls`.
  - Ma trận tương quan và **biến dư thừa**:
    - Tự tính R² giữa `TotalCharges` và `tenure × MonthlyCharges` (IBM).
    - Orange: `Total * charge` gần như tỉ lệ tuyến tính với `Total * minutes`.
    - Iranian: `Customer Value` tương quan mạnh với các biến sử dụng.
  - Kết thúc bằng danh sách **"Phát hiện → Hệ quả cho mô hình"**, ví dụ: mất cân bằng → dùng
    F1/PR-AUC thay vì chỉ Accuracy; biến trùng thông tin → bỏ hoặc làm ablation.
- [ ] Lưu hình vào `figures/eda/` (PNG, dpi 150, có tiêu đề, nhãn trục, chú thích tiếng Việt).
- [ ] Gửi Hải Anh 5–7 gạch đầu dòng tóm tắt dữ liệu và audit Iranian để đưa vào trang tóm tắt
  cho giảng viên (**trước 27/11**).

### Tuần 2 (30/11–06/12) — nguồn gốc, data dictionary, tài liệu

- [ ] **`docs/data_dictionary.md`** cho dataset chính (D1), kèm 2 dataset phụ nếu kịp: tên cột,
  kiểu, ý nghĩa, đơn vị, miền giá trị, ghi chú. **Đánh dấu cột có rủi ro rò rỉ thời gian**: với
  Iranian là `Complains`, `Status`, `Customer Value` — UCI nói đặc trưng tổng hợp từ tháng 1–9 và
  nhãn churn ở tháng 12, nhưng file không có timestamp từng dòng.
- [ ] **Nguồn gốc & license** — cập nhật `data/raw/**/README.md`:
  - IBM Telco: nguồn gốc (IBM sample data, bản Kaggle), license, ngày tải, SHA-256.
  - Iranian: UCI id 563, DOI `10.24432/C5JW3Z`, CC BY 4.0. Tìm bài báo gốc được UCI ghi ở mục
    *Introductory Paper* để trích dẫn.
  - Orange: hiện là bản mirror của mlcourse.ai → tìm nguồn gốc và license. Nếu không xác định
    được, ghi rõ là hạn chế.
- [ ] **Hỗ trợ P0 trong `mustdo.md`:** soạn email hỏi chủ dữ liệu Iranian về các dòng trùng và
  thời điểm có của `Complains`/`Status`/`Customer Value`. Hải Anh duyệt rồi gửi.
- [ ] **Tổng quan tài liệu:** kiểm tra 5 tài liệu trong `README.md` và trong `telco_churn_pipeline.md`
  mục 2. Với mỗi tài liệu: tìm bản gốc, xác minh tác giả/năm/nơi đăng, phân loại peer-reviewed hay
  preprint. **Tài liệu nào không truy được nguồn thì đề nghị bỏ**, kể cả các con số trích từ nó
  (ví dụ baseline của Chen et al. 2025). Bổ sung tài liệu nền tảng: Breiman (2001) *Random Forests*,
  Machine Learning 45(1). Lập bảng: dataset, phương pháp, độ đo, kết quả, hạn chế.
- [ ] **Quản lý danh mục tài liệu tham khảo chung** (một file `docs/report/references.md`): gộp tài
  liệu của bạn với tài liệu Thùy Linh gửi cho mục 4.3; thống nhất một kiểu trích dẫn.
- [ ] Gửi Thùy Linh con số baseline Chen et al. **đã xác minh** để so sánh ở mục 5.1.

### Tuần 3 (07–13/12) — viết

- [ ] **Chương 2 – Tổng quan:** các hướng tiếp cận dự đoán churn (thống kê, cây/ensemble,
  boosting, deep learning), các lỗi phương pháp thường gặp (rò rỉ dữ liệu do SMOTE/scale trước khi
  chia, biến dư thừa, không calibrate, dùng Accuracy với dữ liệu mất cân bằng), vị trí của đề tài.
- [ ] **Chương 3 – Dữ liệu & EDA:** mô tả 3 dataset (bảng tổng quan), lý do chọn dataset chính,
  kết quả EDA (6–10 hình chọn lọc), audit trùng lặp Iranian và lý do dùng group-aware split.
- [ ] Gửi Nhật Linh các hạn chế về dữ liệu cho mục 7.2: dữ liệu công khai cũ, thiên lệch mẫu,
  không có timestamp.
- [ ] Làm 2–3 slide phần dữ liệu & EDA theo khung của Nhật Linh.

### Tuần 4 (14–20/12)

- [ ] Review chéo chương 6 (Nam Khánh): đọc như người ngoài, tự chạy demo theo hướng dẫn.
- [ ] Sửa chương 2–3 theo góp ý của Thùy Linh và Hải Anh. Chốt danh mục tài liệu tham khảo.
- [ ] Viết câu trả lời phần mình trong `docs/defense_qa.md`.

## Phụ thuộc

- **Cần từ:** Hải Anh — kết quả D1 (để biết dataset nào làm kỹ nhất).
- **Cung cấp cho:** Hải Anh — tóm tắt dữ liệu (trước 27/11); Thùy Linh — baseline Chen et al. đã
  xác minh; Nhật Linh — các hạn chế dữ liệu; Nam Khánh — data dictionary (để làm form nhập liệu
  nếu đổi sang Iranian); cả nhóm — danh mục tài liệu tham khảo chung.

## Câu hỏi bảo vệ cần chuẩn bị

1. Vì sao dùng 3 dataset? Mỗi dataset khác nhau ở điểm nào và đóng vai trò gì?
2. Dữ liệu mất cân bằng ảnh hưởng gì tới Accuracy? Nhóm đã xử lý ra sao?
3. Rò rỉ do dòng trùng (duplicate leakage) là gì? Group-aware split giải quyết thế nào?
4. Vì sao `TotalCharges` là biến dư thừa? Bằng chứng định lượng là gì?
5. `Complains` có phải rò rỉ dữ liệu không? Làm sao biết đặc trưng đã có trước thời điểm dự đoán?
6. License của từng dataset cho phép dùng thế nào? Trích dẫn ra sao?

## Lệnh hay dùng

```bash
python -m src.iranian_audit                  # sinh audit_report.json, feature_audit.csv
jupyter notebook notebooks/                  # mở notebook
python -m pytest -q
git fetch origin && git merge origin/main
```

> Notebook giữ output để người đọc xem được kết quả, nhưng mỗi file nên dưới ~10 MB.
> Hình dùng trong báo cáo phải lưu thành file trong `figures/eda/`.
