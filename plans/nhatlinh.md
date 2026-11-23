# Plan — Nguyễn Nhật Linh · Đánh giá, chi phí & góc nhìn kinh doanh

- **Branch:** `nhatlinh/evaluation-business`
- **Vùng file phụ trách:**
  - `COST_*` trong `src/config.py` (chỉ sửa sau khi chốt D3)
  - `notebooks/04_*`, `docs/slides/`
  - `docs/report/04-4-danh-gia-nguong.md`, `05-3-nguong-chi-phi.md`, `07-2-han-che.md`
- **Mục tiêu:** biến xác suất của mô hình thành **quyết định có ý nghĩa kinh doanh**: đề xuất chi phí
  và chính sách có căn cứ, diễn giải kết quả cho người không chuyên, trình bày trung thực các giới
  hạn; làm khung slide cho cả nhóm.

## Hiện trạng

- `find_cost_optimal_threshold` hỗ trợ 2 cách tính chi phí:
  - `classification`: FP·c_FP + FN·c_FN
  - `action`: (số khách được liên hệ)·c_action + FN·c_FN
- Ngưỡng được chọn trên validation, đánh giá trên test (`results/policy_metrics.json`,
  `figures/cost_threshold.png`).
- Chi phí **giả định** FP = $50, FN = $500 → ngưỡng tối ưu trên IBM ≈ 0.06 (recall ≈ 0.98,
  precision ≈ 0.36). Hệ quả: demo gắn nhãn "Churn" cho cả khách hợp đồng 2 năm, xác suất 8.5%.
  **Đây là vấn đề lớn nhất về mặt thuyết phục của demo.**
- Hải Anh viết `src/policy_analysis.py`: độ nhạy theo tỉ lệ chi phí và theo seed, top-k, A4,
  reliability diagram. Kết quả nằm ở `results/policy/` và `figures/policy/`.

## Đọc trước

- `PLAN.md` (D3, D4), `mustdo.md` (P0 chi phí), `reports/results_snapshot.md` §1–§2
- `src/optimize_threshold.py`, `src/evaluate.py` (ECE, lift), `src/train.py` bước 7
- `api/predict.py` (cách API đổi xác suất thành nhãn và mức rủi ro High/Medium/Low)

## Việc cần làm

### Tuần 1 (24–29/11) — đề xuất chi phí và chính sách

- [ ] **Đề xuất cho D3** (gửi Hải Anh **trước 27/11**). Bảng giả định chi phí có căn cứ:
  - Chi phí một ưu đãi giữ chân cho mỗi khách được liên hệ.
  - Giá trị mất đi khi mất một khách ≈ `MonthlyCharges` × số tháng kỳ vọng còn lại.
  - Tỉ lệ thành công của chiến dịch giữ chân (nếu có).

  Đưa ra 2–3 kịch bản (thận trọng / trung bình / lạc quan), nói rõ dùng `classification` hay
  `action` cost và vì sao.
- [ ] **Đề xuất cho D4** — chính sách cho demo, phối hợp với Nam Khánh. Các phương án:
  - (a) ngưỡng tối ưu theo chi phí thật;
  - (b) **top-k theo ngân sách**: chỉ liên hệ 10–20% khách rủi ro nhất, dựa trên lift@k;
  - (c) 3 mức rủi ro theo quantile xác suất.

  So sánh ưu/nhược và chọn 1 phương án để API dùng.
- [ ] Đọc hiểu code đánh giá; chạy `python -m src.train` và xem các hình trong `figures/`.
- [ ] **Viết nháp mục 4.4 – Độ đo & ngưỡng.** Phần này không phụ thuộc kết quả, nên viết sớm:
  - Định nghĩa và công thức: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier, ECE, lift@k.
  - Vì sao PR-AUC phù hợp hơn ROC-AUC khi dữ liệu mất cân bằng.
  - Công thức chi phí `classification` và `action`.

### Tuần 2 (30/11–06/12) — diễn giải kết quả, khung slide

- [ ] **Sau khi D3 chốt:** gửi PR nhỏ cập nhật `COST_FP`, `COST_FN`, `COST_ACTION` trong
  `src/config.py` (trước 04/12), rồi báo Hải Anh (phân tích ngưỡng) và Nam Khánh (chạy final).
- [ ] **Notebook `notebooks/04_threshold_business.ipynb`**, dùng kết quả của `src/policy_analysis.py`
  và model đã lưu:
  - Chọn bảng/hình chính cho mục 5.3 từ `results/policy/` và `figures/policy/`.
  - Confusion matrix tại chính sách đã chọn, diễn giải theo kinh doanh: liên hệ bao nhiêu khách, giữ
    được bao nhiêu khách sắp rời, tốn bao nhiêu, bỏ sót bao nhiêu.
  - **Kiểm tra công bằng theo nhóm khách hàng:** recall/precision theo `SeniorCitizen`, `gender` (IBM)
    hoặc `Age Group` (Iranian), tính từ dự đoán trên tập test.
- [ ] **Khung slide** (mẫu + bố cục 15–20 slide) gửi cả nhóm **trước 06/12**:
  bài toán → dữ liệu → phương pháp → kết quả so sánh → RF & giải thích → ngưỡng/kinh doanh → demo →
  hạn chế → kết luận. Ghi rõ ai làm slide nào (mỗi người 2–3 slide phần của mình).

### Tuần 3 (07–13/12) — viết

- [ ] **Mục 5.3 – Ngưỡng & kinh doanh:** kết quả phân tích ngưỡng, A4, calibration, công bằng; nói rõ
  savings là **mô phỏng dựa trên giả định**, không phải lợi nhuận thực tế.
- [ ] **Mục 7.2 – Hạn chế:**
  - Không có dữ liệu can thiệp nên không biết ưu đãi có thực sự giữ chân được khách hay không.
  - Tương quan không phải nhân quả (kể cả SHAP).
  - Dữ liệu trôi theo thời gian (concept drift).
  - Thiên lệch giữa các nhóm khách hàng.
  - Chi phí là giả định.
  - Dữ liệu công khai, cũ, không có timestamp.

  Quỳnh Anh gửi phần hạn chế về dữ liệu.
- [ ] Hoàn thiện mục 4.4. Làm 2–3 slide phần ngưỡng/kinh doanh và hạn chế.

### Tuần 4 (14–20/12)

- [ ] Ghép slide của cả nhóm, cập nhật số theo snapshot tag `v1.0-results`, thống nhất định dạng.
- [ ] Review chéo chương 1, 4.2, 5.2, 7.1 (Hải Anh).
- [ ] Viết câu trả lời phần mình trong `docs/defense_qa.md`. Tham gia 2 buổi tổng duyệt.

## Phụ thuộc

- **Cần từ:**
  - Hải Anh: kết quả D1, D3; kết quả `src/policy_analysis.py`.
  - Nam Khánh: cách API dùng ngưỡng/mức rủi ro.
  - Quỳnh Anh: các hạn chế về dữ liệu.
- **Cung cấp cho:**
  - Hải Anh: đề xuất chi phí D3 (27/11).
  - Nam Khánh: chính sách D4 và `COST_*` (04/12).
  - Cả nhóm: khung slide (06/12) và bản slide cuối.

## Câu hỏi bảo vệ cần chuẩn bị

1. Vì sao không dùng Accuracy làm độ đo chính?
2. Khi dữ liệu mất cân bằng, PR-AUC khác ROC-AUC thế nào?
3. Vì sao chọn ngưỡng trên validation mà không chọn trên test?
4. Ngưỡng ≈ 0.06 có hợp lý không? Nó phụ thuộc vào giả định nào?
5. Calibration là gì? ECE đo cái gì? Vì sao phải calibrate trước khi tối ưu ngưỡng?
6. `classification` cost khác `action` cost thế nào? Nhóm chọn cách nào và vì sao?
7. Con số "tiết kiệm" có phải lợi nhuận thật không? Muốn kiểm chứng thì phải làm thí nghiệm gì
   (A/B test chiến dịch giữ chân)?

## Lệnh hay dùng

```bash
python -m src.train                       # sinh policy_metrics.json, cost_threshold.png
python -m src.reproduce --quick
jupyter notebook notebooks/
python -m pytest -q
git fetch origin && git merge origin/main
```
