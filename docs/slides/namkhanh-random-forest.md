# Slide phần Random Forest — Nam Khánh (nội dung nháp)

> Nội dung để ráp vào khung slide của Hải Anh. Số liệu theo snapshot commit `89fa839`; cập nhật theo
> tag `v1.0-results`. Mỗi slide kèm ghi chú thuyết trình (~1 phút).

---

## Slide 1 — Random Forest: vì sao là phương pháp chính

**Nội dung trên slide**
- Rừng gồm nhiều cây; mỗi cây học trên một mẫu bootstrap và chỉ xét một phần đặc trưng ở mỗi nút
- Lấy trung bình → giảm phương sai, học được quan hệ phi tuyến và tương tác
- Không cần chuẩn hóa, ít giả định; giải thích được bằng SHAP
- Tune: IBM 500 cây, độ sâu 10; Iranian, Orange: cây không giới hạn độ sâu

**Hình:** `figures/rf/validation_curve_ibm.png`

**Ghi chú thuyết trình:** Thêm cây không làm overfit — đường cross-validation đi ngang từ khoảng 200 cây.
Về độ sâu, trên IBM từ độ sâu 8 trở lên điểm cross-validation gần như không đổi (chênh dưới 0.004) trong
khi điểm trên train tiếp tục tăng: cây sâu hơn chỉ học thuộc dữ liệu train. Độ sâu 10 nằm trong vùng phẳng
đó. Trên Iranian và Orange thì ngược lại, cây càng sâu càng tốt.

---

## Slide 2 — RF so với mô hình khác (10 lần chia dữ liệu)

**Nội dung trên slide**

| | AP — RF | AP — LogReg | Kết luận |
|---|---:|---:|---|
| IBM | 0.666 | 0.668 | Ngang nhau (p = 0.60) |
| Orange | **0.835** | 0.465 | RF vượt trội (p < 0.001) |
| Iranian | **0.912** | 0.768 | RF vượt trội (p < 0.001) |

- RF mạnh trên dữ liệu hành vi định lượng (khiếu nại, cuộc gọi CSKH, tần suất dùng)
- IBM nhiều biến phân loại → mô hình tuyến tính đã làm khá tốt

**Ghi chú thuyết trình:** Mỗi con số là trung bình của 10 lần chia dữ liệu khác nhau, không phải một lần
may mắn. Trên Orange và Iranian, RF thắng ở cả 10 lần chia. p-value tính bằng kiểm định t có hiệu chỉnh
(Nadeau và Bengio), vì 10 tập train chồng lên nhau nên kiểm định t thông thường quá lạc quan. Trên IBM,
nhóm ghi nhận trung thực là RF không hơn LogReg về khả năng xếp hạng.

---

## Slide 3 — Mô hình dựa vào đâu để dự đoán

**Nội dung trên slide**
- IBM: thời gian gắn bó, loại hợp đồng, cáp quang, mức cước
- Iranian: khiếu nại, trạng thái thuê bao, tần suất sử dụng
- Orange: số cuộc gọi chăm sóc khách hàng, cước ban ngày, gói quốc tế
- Giải thích từng khách: vì sao bị đánh giá rủi ro cao/thấp
- ⚠ Đây là tương quan mô hình học được, không phải nhân quả

**Hình:** `figures/rf/local_shap_ibm.png` (hoặc `importance_methods_ibm.png`)

**Ghi chú thuyết trình:** Khách bị dự đoán đúng là churn mới dùng 1 tháng, cáp quang, hợp đồng theo
tháng. Khách bị bỏ sót đã gắn bó 48 tháng — lý do rời đi không nằm trong dữ liệu. Ba cách đo độ quan
trọng có thể lệch nhau, ví dụ `State` trên Orange được MDI đánh giá cao nhưng thực tế gần như không ảnh
hưởng hiệu năng.

---

## Slide dự phòng — Calibration và ablation (dùng khi hội đồng hỏi)

- Calibration: ECE giảm từ 0.121 xuống 0.029; RF chưa calibrate dự đoán trung bình 0.386 trong khi tỉ lệ
  churn thật là 0.265 (do `class_weight`)
- Deploy 1 rừng + sigmoid: chất lượng như 5 rừng, model 91 MB → 17 MB
- A1: bỏ `TotalCharges` gần như không mất gì (thông tin trùng `tenure × MonthlyCharges`)
- A2: `class_weight` và SMOTE đều không làm AP tốt hơn, chỉ dời điểm vận hành ở ngưỡng 0.5; ngưỡng được
  chọn theo chi phí sau khi calibrate
- A3: chuẩn hóa không ảnh hưởng tới cây (F1 và AP khác nhau ≤ 0.0012)

**Hình:** `figures/rf/reliability_ibm.png`
