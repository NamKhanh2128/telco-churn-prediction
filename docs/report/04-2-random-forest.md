# 4.2. Random Forest — phương pháp chính

> Bản nháp — Nam Khánh. Phần lý thuyết; kết quả thực nghiệm ở mục 5.2.

## 4.2.1. Từ cây quyết định đến rừng ngẫu nhiên

Một **cây quyết định** chia không gian đặc trưng bằng các câu hỏi dạng "đặc trưng ≤ ngưỡng?", mỗi lần
chọn phép chia làm giảm nhiều nhất độ "lẫn" của nút, đo bằng chỉ số Gini

$$G(t) = 1 - \sum_{k} p_k(t)^2,$$

với $p_k(t)$ là tỉ lệ lớp $k$ trong nút $t$. Cây sâu khớp rất tốt dữ liệu huấn luyện nhưng
**phương sai cao**: đổi một ít dữ liệu là cấu trúc cây thay đổi mạnh (mục 4.3).

**Random Forest** (Breiman, 2001) giảm phương sai bằng cách kết hợp nhiều cây khác nhau:

1. **Bagging** (Breiman, 1996): mỗi cây học trên một mẫu bootstrap — rút ngẫu nhiên có hoàn lại $n$
   dòng từ $n$ dòng train.
2. **Chọn ngẫu nhiên đặc trưng** (random subspace; Ho, 1998): tại mỗi nút chỉ xét một tập con ngẫu
   nhiên các đặc trưng (`max_features`, ví dụ $\sqrt{d}$), để các cây ít giống nhau hơn.
3. **Dự đoán:** xác suất churn là trung bình xác suất của các cây; nhãn là churn nếu xác suất ≥ ngưỡng.

## 4.2.2. Vì sao lấy trung bình lại giảm sai số

Với $B$ cây, mỗi cây có phương sai $\sigma^2$ và hệ số tương quan giữa hai cây là $\rho$, phương sai
của trung bình là

$$\operatorname{Var}\Big(\frac{1}{B}\sum_{b=1}^{B} T_b\Big) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2 .$$

Tăng số cây $B$ làm số hạng thứ hai về 0; giảm tương quan $\rho$ (nhờ chọn ngẫu nhiên đặc trưng) làm
giảm số hạng thứ nhất. Vì vậy thêm cây không làm mô hình overfit hơn — chỉ tốn thời gian, và hiệu
năng đi ngang sau một số lượng cây nhất định (validation curve ở mục 5.2).

## 4.2.3. Ước lượng out-of-bag (OOB)

Xác suất một dòng **không** được chọn vào một mẫu bootstrap là $(1 - 1/n)^n \to e^{-1} \approx 36.8\%$.
Mỗi dòng vì thế là "dữ liệu chưa thấy" của khoảng một phần ba số cây. Dự đoán dòng đó chỉ bằng các
cây này cho **ước lượng OOB** — gần như một lần cross-validation "miễn phí". Nhóm so sánh OOB với CV
và test để kiểm tra độ tin cậy của đánh giá (mục 5.2).

## 4.2.4. Đo độ quan trọng của đặc trưng

| Cách đo | Ý tưởng | Ưu điểm | Hạn chế |
|---|---|---|---|
| **MDI** (mean decrease in impurity) | Tổng mức giảm Gini nhờ đặc trưng, trung bình trên các cây | Có sẵn, nhanh | Tính trên dữ liệu train; thiên vị đặc trưng nhiều giá trị (Strobl et al., 2007) |
| **Permutation** | Xáo trộn một cột trên tập test, đo hiệu năng giảm bao nhiêu | Gắn trực tiếp với hiệu năng thật | Chia điểm giữa các đặc trưng tương quan, có thể âm |
| **SHAP** (Lundberg & Lee, 2017) | Chia "phần đóng góp" vào dự đoán theo giá trị Shapley; TreeSHAP tính chính xác cho cây (Lundberg et al., 2020) | Vừa toàn cục vừa cục bộ (từng khách), có dấu | Là tương quan mô hình học được, **không phải nhân quả** |

Sau one-hot encoding, một biến như `Contract` thành nhiều cột; nhóm cộng đóng góp các cột này về
biến gốc để kết quả dễ đọc (`src/explain.py`).

## 4.2.5. Siêu tham số và không gian tìm kiếm

| Tham số | Ý nghĩa | Giá trị thử |
|---|---|---|
| `n_estimators` | Số cây | 100, 200, 500 |
| `max_depth` | Độ sâu tối đa của cây | 5, 10, 15, không giới hạn |
| `min_samples_split` | Số mẫu tối thiểu để chia nút | 2, 5, 10 |
| `min_samples_leaf` | Số mẫu tối thiểu ở lá | 1, 2, 5 |
| `max_features` | Số đặc trưng xét tại mỗi nút | `sqrt`, `log2` |
| `class_weight` | Trọng số lớp | `balanced_subsample` (cố định) |

`class_weight="balanced_subsample"` tính lại trọng số lớp trên **từng mẫu bootstrap**, để lớp churn
(thiểu số) không bị lấn át khi cây chọn phép chia. Tuning dùng `RandomizedSearchCV` (30 tổ hợp, 5-fold
CV, tiêu chí AP — mục 4.1.4).

## 4.2.6. Hiệu chỉnh xác suất (calibration)

Xác suất của Random Forest thường **không phản ánh đúng tần suất thật**: việc lấy trung bình nhiều cây
kéo dự đoán ra xa 0 và 1 (Niculescu-Mizil & Caruana, 2005). Điều này không ảnh hưởng tới thứ hạng
(ROC-AUC, AP) nhưng ảnh hưởng tới **chọn ngưỡng theo chi phí**, vốn giả định "xác suất 0.3 nghĩa là
30% khách như vậy rời đi".

- **Platt scaling** (sigmoid; Platt, 1999): khớp $\hat p = 1/(1 + e^{A f + B})$ trên dự đoán
  cross-validation; chỉ 2 tham số nên ổn định với dữ liệu vừa và nhỏ.
- **Isotonic regression:** hàm đơn điệu tùy ý, linh hoạt hơn nhưng dễ overfit khi ít dữ liệu.

Nhóm dùng Platt scaling với 5-fold CV trên tập train. Độ tin cậy của xác suất đo bằng **Brier score**
và **ECE** (expected calibration error), minh họa bằng reliability diagram (`figures/rf/`). So sánh
sigmoid/isotonic và `ensemble=True/False` ở snapshot §8.

## 4.2.7. Vì sao chọn Random Forest làm phương pháp chính

- Học được **quan hệ phi tuyến và tương tác** (ví dụ "hợp đồng theo tháng *và* mới dùng vài tháng")
  mà không cần tự tạo đặc trưng.
- **Ít giả định:** không cần chuẩn hóa, ít nhạy với ngoại lai, xử lý tốt hỗn hợp biến số và biến phân loại.
- **Ổn định** hơn một cây đơn nhờ bagging; có sẵn ước lượng OOB.
- **Giải thích được** ở cả mức toàn cục (độ quan trọng) và từng khách (SHAP), cần cho người dùng nghiệp vụ.

Hạn chế: xác suất cần hiệu chỉnh; model lớn hơn mô hình tuyến tính; khó diễn giải bằng một quy tắc
đơn giản như một cây. Trên dữ liệu chủ yếu là biến phân loại (IBM), mô hình tuyến tính có thể ngang
Random Forest — kết quả thực tế được phân tích trung thực ở mục 5.1, 5.2 và 5.4.

## Tài liệu tham khảo của mục

- Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123–140.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Ho, T. K. (1998). The random subspace method for constructing decision forests. *IEEE TPAMI*, 20(8), 832–844.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*.
- Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*.
- Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). Bias in random forest variable importance measures. *BMC Bioinformatics*, 8, 25.
