# Câu hỏi bảo vệ — Nam Khánh (Random Forest & thí nghiệm)

> Câu trả lời ngắn (2–4 câu) để Hải Anh gộp vào `docs/defense_qa.md`. Số liệu hiện lấy từ
> `reports/results_snapshot.md` của lượt chạy tại commit `89fa839` (§ = mục trong snapshot); sẽ cập nhật theo
> tag `v1.0-results`.

**1. Tại sao chọn RF làm phương pháp chính khi trên IBM LogReg ngang hoặc tốt hơn về ROC-AUC?**
Trên IBM, phần lớn đặc trưng là biến phân loại; sau one-hot, quan hệ với churn gần tuyến tính nên
LogReg đã làm khá tốt, và nhóm báo cáo trung thực điều đó (AP 0.666 so với 0.668, p = 0.60; §7). Trên
Orange và Iranian — dữ liệu hành vi định lượng có ngưỡng phi tuyến (số cuộc gọi CSKH, khiếu nại, tần suất
dùng) — RF vượt rõ LogReg về AP và ROC-AUC ở cả 10 lần chia (§3, §7). RF còn ít giả định, không cần chuẩn
hóa, và giải thích được bằng SHAP, nên phù hợp làm phương pháp chính cho bài toán churn tổng quát.

**2. RF khác Decision Tree thế nào? Vì sao bagging giảm phương sai? OOB error là gì?**
RF là trung bình của nhiều cây, mỗi cây học trên một mẫu bootstrap và chỉ xét một tập con ngẫu nhiên
đặc trưng ở mỗi nút. Phương sai của trung bình là ρσ² + (1−ρ)σ²/B: nhiều cây và cây ít tương quan thì
phương sai giảm. Mỗi dòng không nằm trong khoảng 36.8% mẫu bootstrap; dự đoán nó bằng các cây "chưa
thấy" nó cho ước lượng OOB — gần như cross-validation miễn phí (so sánh OOB/CV/test ở §9).

**3. Vì sao phải calibrate RF trước khi chọn ngưỡng? Platt scaling khác isotonic thế nào?**
Ngưỡng theo chi phí giả định xác suất phản ánh đúng tần suất thật. RF của nhóm dùng `class_weight` nên xác
suất bị đẩy lên: trung bình dự đoán 0.386 trong khi tỉ lệ churn thật là 0.265, ECE 0.121 (§8). Platt scaling
khớp một hàm sigmoid 2 tham số nên ổn định; isotonic là hàm bậc thang đơn điệu, linh hoạt hơn nhưng dễ
overfit khi ít dữ liệu. Calibrate xong ECE còn 0.029 mà thứ hạng không đổi; nhóm deploy sigmoid với
`ensemble=False` (một rừng) vì trên validation chất lượng như 5 rừng mà file chỉ 17 MB thay vì 74 MB.

**4. SHAP khác feature importance có sẵn của RF ở điểm nào? SHAP có cho biết quan hệ nhân quả không?**
MDI có sẵn chỉ cho độ quan trọng toàn cục, tính trên dữ liệu train và thiên vị biến nhiều giá trị.
SHAP cho đóng góp có dấu của từng đặc trưng vào **từng dự đoán**, nên giải thích được từng khách hàng.
SHAP mô tả điều **mô hình** dựa vào, không phải nguyên nhân: muốn biết ưu đãi có giữ chân được khách
hay không cần thí nghiệm can thiệp như A/B test.

**5. `class_weight` khác SMOTE thế nào? Vì sao SMOTE phải nằm trong pipeline, sau khi chia dữ liệu?**
`class_weight` tăng trọng số lớp thiểu số khi cây chọn phép chia, không tạo dữ liệu mới; SMOTE sinh mẫu
thiểu số nhân tạo bằng nội suy giữa các láng giềng. Nếu SMOTE chạy trước khi chia, mẫu nhân tạo sinh từ dòng
test lọt vào train → điểm test bị thổi phồng; vì vậy SMOTE nằm trong `imblearn` Pipeline, chỉ chạy khi `fit`
trên fold train. Ablation A2 (§6) cho thấy với rừng đã tune, cả hai cách đều không làm AP tốt hơn — chúng chủ
yếu dời điểm vận hành ở ngưỡng 0.5, mà ngưỡng cuối cùng nhóm chọn theo chi phí.

**6. Vì sao bỏ `TotalCharges` (A1)? Code đảm bảo không rò rỉ dữ liệu như thế nào?**
`TotalCharges` ≈ `tenure × MonthlyCharges` nên gần như trùng thông tin; A1 giữ lại cột này và AP chỉ đổi từ
0.665 lên 0.668 (§3). Chống rò rỉ: chia dữ liệu trước; mọi bước học từ dữ liệu (điền khuyết, mã hóa, chuẩn
hóa, SMOTE, calibration, tuning) nằm trong Pipeline và chỉ fit trên train; ngưỡng và phương án calibration
chọn trên validation; Iranian chia theo nhóm để các dòng trùng không nằm ở cả train và test.

**7. Các kiểm định thống kê cho biết điều gì? p-value nhỏ nghĩa là gì, và không có nghĩa là gì?**
Nhóm so RF với mô hình mạnh nhất còn lại trên cùng 10 lần chia (§7). Kiểm định chính là t có hiệu chỉnh
của Nadeau và Bengio: 10 tập train chồng lên nhau nên các lần chia không độc lập, hệ số của phương sai được
cộng thêm n_test/n_train (thành 1/k + n_test/n_train, phương sai lớn gấp khoảng 3 lần với 10 lần chia). Wilcoxon (xếp hạng hiệu số) và McNemar (số khách một mô hình đúng còn mô hình kia sai, trên
cùng tập test) dùng để đối chiếu. p-value nhỏ nghĩa là chênh lệch khó xảy ra nếu hai mô hình thực sự ngang
nhau; nó **không** cho biết chênh lệch lớn hay có ý nghĩa thực tế, và các p-value chưa hiệu chỉnh cho việc
so sánh nhiều lần.

**8. Làm sao tái lập đúng kết quả?**
Seed cố định ở mọi bước ngẫu nhiên; SHA-256 của dữ liệu và phiên bản thư viện ghi trong log và đầu snapshot;
tham số đã tune lưu trong `models/tuned_params.json`; một lệnh `python -m src.reproduce` chạy toàn bộ và xuất
snapshot kèm mã commit; số liệu báo cáo gắn với tag `v1.0-results`. Chạy lại cùng commit trên cùng môi trường
cho số liệu giống hệt (đã thử với benchmark Iranian: hai lần chạy vào hai thư mục riêng cho mọi chỉ số trùng nhau).

**9. Trên IBM, F1 của RF ở ngưỡng 0.5 cao hơn LogReg nhiều (0.628 so với 0.578, p rất nhỏ), sao vẫn nói
hai mô hình ngang nhau?**
F1 phụ thuộc ngưỡng. RF trong so sánh dùng `class_weight` và chưa calibrate nên xác suất bị đẩy lên, ngưỡng
0.5 với nó tương đương một ngưỡng thấp hơn với LogReg (đã calibrate) → recall cao hơn. Trên tập test seed 42 (§1),
RF chưa calibrate có F1 0.633; calibrate xong, F1 ở ngưỡng 0.5 chỉ còn 0.595, ngang LogReg (0.594). Hai số
0.628 và 0.578 trong câu hỏi là trung bình 10 seed (§7). So sánh chất lượng mô hình phải dùng chỉ số không phụ thuộc
ngưỡng (AP, ROC-AUC), và ở đó hai mô hình ngang nhau.

**10. Tham số tune một lần rồi dùng cho cả 10 seed — có thiên vị không?**
Có một chút: tham số được chọn trên tập train của seed 42, mà tập test của các seed khác có thể chứa một
phần dữ liệu đó, nên kết quả 10 seed hơi lạc quan cho các mô hình đã tune. Thiên vị này như nhau cho RF và
LogReg nên không làm lệch so sánh giữa hai mô hình; nhóm ghi rõ hạn chế này trong snapshot.
