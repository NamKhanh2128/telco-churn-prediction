# 4.3 Các mô hình so sánh: Decision Tree, KNN và SVM

> Bản nháp lý thuyết cho mục 4.3 · Phụ trách: Mai Thùy Linh (`thuylinh/model-comparison`).

Để đánh giá khách quan hiệu quả của phương pháp chính **Random Forest** (mục 4.2), nghiên cứu đối chiếu với ba họ mô hình kinh điển được đề cập trong đề tài: **Decision Tree (DT)**, **K-Nearest Neighbors (KNN)** và **Support Vector Machine (SVM)** (bao gồm biến thể tuyến tính `SVM_Linear` và phi tuyến `SVM_RBF`), cùng hai mô hình cơ sở (*Dummy* và *Logistic Regression*).

---

## 4.3.1 Decision Tree (Cây quyết định)

**Nguyên lý hoạt động:**  
Decision Tree phân chia không gian đặc trưng theo cấu trúc cây nhị phân từ nút gốc đến các nút lá bằng thuật toán tham lam đệ quy (CART — Breiman et al., 1984). Tại mỗi nút, thuật toán chọn cặp *(đặc trưng $j$, ngưỡng chia $t$)* sao cho mức độ hỗn tạp (*impurity*) của hai nút con giảm nhiều nhất. Hai tiêu chí đo độ bất thuần phổ biến cho bài toán phân loại nhị phân là:

- **Chỉ số Gini:** $G = 1 - \sum_{k \in \{0,1\}} p_k^2$
- **Entropy thông tin:** $H = -\sum_{k \in \{0,1\}} p_k \log_2(p_k)$

trong đó $p_k$ là tỉ lệ mẫu thuộc lớp $k$ tại nút đang xét.

**Kiểm soát quá khớp (overfitting) và tỉa cây:**  
Cây quyết định đơn lẻ có phương sai (*variance*) cao: nếu không giới hạn, cây sẽ phát triển sâu cho đến khi mỗi nút lá chỉ chứa một vài mẫu huấn luyện, dẫn đến học thuộc nhiễu (*overfitting*) và giảm khả năng tổng quát hóa trên tập kiểm thử. Do đó, nghiên cứu tinh chỉnh các siêu tham số tiền tỉa (*pre-pruning*) gồm `max_depth`, `min_samples_split`, `min_samples_leaf`, đồng thời xem xét tham số tỉa hậu kỳ `ccp_alpha` (*Minimal Cost-Complexity Pruning*).

**Xử lý mất cân bằng và chuẩn hóa dữ liệu:**  
- Để tránh cây thiên vị lớp đa số (*No Churn*), mô hình sử dụng `class_weight="balanced"`, tự động gán trọng số tỉ lệ nghịch với tần suất lớp trong hàm mất mát. Điều này giúp tăng mạnh khả năng phát hiện khách hàng rời bỏ (*Recall*) nhưng thường đi kèm đánh đổi giảm *Precision* ở ngưỡng mặc định `0.5`.
- Vì cây chỉ dựa trên thứ tự giá trị để tìm ngưỡng chia đơn biến ($x_j \le t$), phép biến đổi đơn điệu như `StandardScaler` không làm thay đổi cấu trúc cây; do đó **không cần chuẩn hóa** đặc trưng số đối với Decision Tree.

---

## 4.3.2 K-Nearest Neighbors (KNN)

**Nguyên lý hoạt động:**  
KNN (Cover & Hart, 1967) là thuật toán học lười (*lazy learning*) và phi tham số (*non-parametric*): mô hình không xây dựng hàm phân tách tường minh trong giai đoạn huấn luyện mà lưu trữ toàn bộ tập huấn luyện. Khi dự đoán cho một khách hàng mới $x$, mô hình tìm $k$ điểm gần nhất trong tập huấn luyện theo độ đo khoảng cách $d(x, x_i)$ và bỏ phiếu theo nhãn của $k$ láng giềng này.

Các siêu tham số cốt lõi được tinh chỉnh gồm:
- **Số láng giềng `n_neighbors` ($k$):** $k$ nhỏ (ví dụ $k=3$) khiến ranh giới quyết định phức tạp, nhạy với nhiễu; $k$ lớn làm mượt ranh giới nhưng có thể làm mờ đặc trưng của lớp thiểu số (*Churn*).
- **Độ đo khoảng cách `metric`:** Khoảng cách Euclid ($L_2$) hoặc Manhattan ($L_1$).
- **Trọng số `weights`:** `uniform` (mọi láng giềng có trọng số bằng nhau) hoặc `distance` (láng giềng gần hơn có trọng số $1 / d(x, x_i)$ lớn hơn).

**Vì sao KNN bắt buộc phải chuẩn hóa dữ liệu?**  
Khoảng cách Euclid và Manhattan phụ thuộc trực tiếp vào thang đo của từng chiều đặc trưng. Trong bài toán viễn thông, các biến như `MonthlyCharges` (20–120 USD) hoặc `Seconds of Use` (hàng nghìn giây) có biên độ lớn hơn rất nhiều so với các biến nhị phân sau mã hóa One-Hot (`0` hoặc `1`) hay `SeniorCitizen`. Nếu không chuẩn hóa bằng `StandardScaler` (đưa về trung bình `0`, độ lệch chuẩn `1` **chỉ fit trên tập train**), khoảng cách sẽ bị chi phối hoàn toàn bởi một vài biến có giá trị lớn. Ngoài ra, trên tập dữ liệu có nhiều biến phân loại sau One-Hot encoding (như IBM Telco), số chiều tăng lên khiến khoảng cách giữa các điểm trở nên kém phân biệt (*curse of dimensionality*), làm KNN dễ yếu thế hơn mô hình tuyến tính và mô hình cây.

---

## 4.3.3 Support Vector Machine (SVM)

**Nguyên lý tối đa hóa lề và tham số `C`:**  
SVM (Cortes & Vapnik, 1995) tìm siêu phẳng phân tách hai lớp sao cho **lề (*margin*)** — khoảng cách từ siêu phẳng tới các điểm dữ liệu gần nhất (*support vectors*) — là lớn nhất. Với dữ liệu không tách biệt tuyến tính hoàn toàn, bài toán tối ưu lề mềm (*soft-margin*) cho phép một số điểm vi phạm lề thông qua biến bù $\xi_i \ge 0$:

$$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{n} \xi_i$$

Tham số **`C`** điều khiển sự đánh đổi giữa độ rộng lề và lỗi phân loại trên tập huấn luyện: `C` nhỏ ưu tiên lề rộng (giảm variance, dễ underfit), trong khi `C` lớn phạt nặng các điểm sai (giảm bias, dễ overfit).

**Kernel RBF và tham số `gamma`:**  
Bên cạnh biến thể tuyến tính (`LinearSVC`), nghiên cứu sử dụng SVM phi tuyến với **hàm nhân RBF** (*Radial Basis Function*):

$$K(x, x') = \exp(-\gamma \|x - x'\|^2)$$

Thủ thuật kernel (*kernel trick*) ánh xạ ngầm dữ liệu sang không gian đặc trưng nhiều chiều hơn để tìm ranh giới phi tuyến mà không cần tính tọa độ tường minh. Trong đó, **`gamma`** quyết định phạm vi ảnh hưởng của từng điểm dữ liệu: `gamma` lớn khiến vùng ảnh hưởng hẹp, ranh giới uốn lượn sát từng điểm huấn luyện (nguy cơ overfit cao); `gamma` nhỏ tạo ranh giới mượt hơn. Giống KNN, cả khoảng cách trong kernel RBF lẫn chuẩn hóa vector trọng số $\|w\|^2$ đều nhạy với thang đo đặc trưng, nên **SVM bắt buộc phải đi kèm `StandardScaler`**.

**Hiệu chuẩn xác suất bằng Platt scaling:**  
SVM vốn chỉ xuất điểm quyết định (*decision function* — khoảng cách có dấu tới siêu phẳng), **không** trực tiếp cho ra xác suất hậu nghiệm $P(y=1 \mid x)$ hợp lệ để tính các độ đo hiệu chuẩn (Brier score, ECE) hay tối ưu ngưỡng theo chi phí kinh doanh. Do đó, cả `SVM_Linear` và `SVM_RBF` trong pipeline đều được bọc trong `CalibratedClassifierCV(method='sigmoid', cv=5)` áp dụng **Platt scaling** (Platt, 1999): khớp một hàm hồi quy logistic (sigmoid) trên đầu ra `decision_function` qua 5-fold cross-validation nội bộ trên tập train để thu được xác suất đã hiệu chuẩn.

---

## Tóm tắt cấu hình và không gian siêu tham số

| Mô hình | Lớp `scikit-learn` | Chuẩn hóa số | Xử lý mất cân bằng | Không gian tìm kiếm (`src/config.py`) |
|---|---|---|---|---|
| **DecisionTree** | `DecisionTreeClassifier` | Không | `class_weight="balanced"` | `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion` |
| **KNN** | `KNeighborsClassifier` | Có (`StandardScaler`) | Không có tham số `class_weight` | `n_neighbors`, `weights`, `metric` |
| **SVM_Linear** | `LinearSVC` + Platt (`sigmoid`) | Có (`StandardScaler`) | `class_weight="balanced"` | `C`, `loss` (`hinge`, `squared_hinge`) |
| **SVM_RBF** | `SVC(kernel="rbf")` + Platt (`sigmoid`) | Có (`StandardScaler`) | `class_weight="balanced"` | `estimator__C`, `estimator__gamma` |
