# Câu hỏi bảo vệ — Nam Khánh (Random Forest & thí nghiệm)

> Câu trả lời ngắn (2–4 câu) để Hải Anh gộp vào `docs/defense_qa.md`. Số liệu: xem
> `reports/results_snapshot.md` của tag `v1.0-results` (§ = mục trong snapshot).

**1. Tại sao chọn RF làm phương pháp chính khi trên IBM LogReg ngang hoặc tốt hơn về ROC-AUC?**
Trên IBM, phần lớn đặc trưng là biến phân loại; sau one-hot, quan hệ với churn gần tuyến tính nên
LogReg đủ tốt, và nhóm báo cáo trung thực điều đó (§1, §5). Trên Orange và Iranian — dữ liệu hành vi
định lượng có ngưỡng phi tuyến (số cuộc gọi CSKH, khiếu nại, tần suất dùng) — RF vượt rõ LogReg về AP
và ROC-AUC, và vượt ổn định qua 10 seed (§3, §7). RF còn ít giả định, không cần chuẩn hóa, và giải
thích được bằng SHAP, nên phù hợp làm phương pháp chính cho bài toán churn tổng quát.

**2. RF khác Decision Tree thế nào? Vì sao bagging giảm phương sai? OOB error là gì?**
RF là trung bình của nhiều cây, mỗi cây học trên một mẫu bootstrap và chỉ xét một tập con ngẫu nhiên
đặc trưng ở mỗi nút. Phương sai của trung bình là ρσ² + (1−ρ)σ²/B: nhiều cây và cây ít tương quan thì
phương sai giảm. Mỗi dòng không nằm trong khoảng 36.8% mẫu bootstrap; dự đoán nó bằng các cây "chưa
thấy" nó cho ước lượng OOB — gần như cross-validation miễn phí (so sánh OOB/CV/test ở §9).

**3. Vì sao phải calibrate RF trước khi chọn ngưỡng? Platt scaling khác isotonic thế nào?**
Ngưỡng theo chi phí giả định xác suất phản ánh đúng tần suất thật, mà xác suất RF bị kéo về giữa do
lấy trung bình nhiều cây. Platt scaling khớp một hàm sigmoid 2 tham số nên ổn định; isotonic là hàm
đơn điệu tùy ý, linh hoạt hơn nhưng dễ overfit khi ít dữ liệu. Trên IBM, calibration giảm ECE rõ rệt
mà không đổi thứ hạng (§8); nhóm deploy sigmoid với `ensemble=False` vì chất lượng như nhau mà model
nhỏ hơn khoảng 4 lần.

**4. SHAP khác feature importance có sẵn của RF ở điểm nào? SHAP có cho biết quan hệ nhân quả không?**
MDI có sẵn chỉ cho độ quan trọng toàn cục, tính trên dữ liệu train và thiên vị biến nhiều giá trị.
SHAP cho đóng góp có dấu của từng đặc trưng vào **từng dự đoán**, nên giải thích được từng khách hàng.
SHAP mô tả điều **mô hình** dựa vào, không phải nguyên nhân: muốn biết ưu đãi có giữ chân được khách
hay không cần thí nghiệm can thiệp như A/B test.

**5. `class_weight` khác SMOTE thế nào? Vì sao SMOTE phải nằm trong pipeline, sau khi chia dữ liệu?**
`class_weight` tăng trọng số lớp thiểu số trong hàm mất mát, không tạo dữ liệu mới; SMOTE sinh mẫu
thiểu số nhân tạo bằng nội suy giữa các láng giềng. Nếu SMOTE chạy trước khi chia, mẫu nhân tạo sinh
từ dòng test lọt vào train → điểm test bị thổi phồng. Vì vậy SMOTE nằm trong `imblearn` Pipeline, chỉ
chạy khi `fit` trên fold train; ablation A2 so sánh hai cách (§6).

**6. Vì sao bỏ `TotalCharges` (A1)? Code đảm bảo không rò rỉ dữ liệu như thế nào?**
`TotalCharges` ≈ `tenure × MonthlyCharges` nên gần như trùng thông tin; A1 giữ lại cột này và chỉ thấy
thay đổi không đáng kể (§3). Chống rò rỉ: chia dữ liệu trước; mọi bước học từ dữ liệu (điền khuyết,
mã hóa, chuẩn hóa, SMOTE, calibration, tuning) nằm trong Pipeline và chỉ fit trên train; ngưỡng chọn
trên validation; Iranian chia theo nhóm để các dòng trùng không nằm ở cả train và test.

**7. Kiểm định Wilcoxon và McNemar cho biết điều gì? p-value nhỏ nghĩa là gì, và không có nghĩa là gì?**
Wilcoxon so sánh điểm của RF và đối thủ mạnh nhất trên cùng 10 lần chia (cặp theo seed); McNemar so
sánh hai mô hình trên cùng tập test qua số mẫu một mô hình đúng còn mô hình kia sai (§7). p-value nhỏ
nghĩa là chênh lệch khó xảy ra nếu hai mô hình thực sự ngang nhau. Nó **không** cho biết chênh lệch
lớn hay có ý nghĩa thực tế, và khi so nhiều cặp phải cẩn trọng (chưa hiệu chỉnh so sánh bội).

**8. Làm sao tái lập đúng kết quả?**
Seed cố định; SHA-256 của dữ liệu và phiên bản thư viện ghi trong metadata; một lệnh
`python -m src.reproduce` chạy toàn bộ và xuất snapshot kèm mã commit; số liệu báo cáo gắn với tag
`v1.0-results`. Chạy lại từ một commit sạch cho kết quả trùng khớp tuyệt đối.
