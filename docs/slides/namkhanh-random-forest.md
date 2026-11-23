# Slide phần Random Forest — Nam Khánh (nội dung nháp)

> Nội dung để ráp vào khung slide của Hải Anh. Số liệu theo snapshot commit `341ea0a`; cập nhật theo
> tag `v1.0-results`. Mỗi slide kèm ghi chú thuyết trình (~1 phút).

---

## Slide 1 — Random Forest: vì sao là phương pháp chính

**Nội dung trên slide**
- Rừng gồm nhiều cây; mỗi cây học trên một mẫu bootstrap và chỉ xét một phần đặc trưng ở mỗi nút
- Lấy trung bình → giảm phương sai, học được quan hệ phi tuyến và tương tác
- Không cần chuẩn hóa, ít giả định; giải thích được bằng SHAP
- Tune: IBM 500 cây, độ sâu 10; Iranian, Orange: cây không giới hạn độ sâu

**Hình:** `figures/rf/validation_curve_ibm.png`

**Ghi chú thuyết trình:** Thêm cây không làm overfit — đường cross-validation đi ngang sau 300 cây.
Độ sâu thì khác: trên IBM, cây quá sâu học thuộc dữ liệu train (đường đỏ tăng, đường xanh giảm), nên
giới hạn ở độ sâu khoảng 10 là tốt nhất.

---

## Slide 2 — RF so với mô hình khác (10 lần chia dữ liệu)

**Nội dung trên slide**

| | AP — RF | AP — LogReg | Kết luận |
|---|---:|---:|---|
| IBM | 0.665 | 0.668 | Ngang nhau (p = 0.19) |
| Orange | **0.835** | 0.465 | RF vượt trội (p = 0.002) |
| Iranian | **0.912** | 0.769 | RF vượt trội (p = 0.002) |

- RF mạnh trên dữ liệu hành vi định lượng (khiếu nại, cuộc gọi CSKH, tần suất dùng)
- IBM nhiều biến phân loại → mô hình tuyến tính đã đủ tốt; khác biệt nằm ở ngưỡng quyết định

**Ghi chú thuyết trình:** Mỗi con số là trung bình của 10 lần chia dữ liệu khác nhau, không phải một lần
may mắn. p = 0.002 nghĩa là RF thắng cả 10 lần. Trên IBM, nhóm trung thực ghi nhận RF không hơn LogReg
về xếp hạng.

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

- Calibration: ECE giảm từ 0.121 xuống 0.031; deploy 1 rừng + sigmoid, model 91 MB → 34 MB
- A1: bỏ `TotalCharges` gần như không mất gì (thông tin trùng `tenure × MonthlyCharges`)
- A2: SMOTE tăng recall ở ngưỡng 0.5 nhưng giảm AP; ngưỡng được chọn theo chi phí nên không dùng SMOTE
- A3: chuẩn hóa không ảnh hưởng tới cây (khác biệt < 0.002)

**Hình:** `figures/rf/reliability_ibm.png`
