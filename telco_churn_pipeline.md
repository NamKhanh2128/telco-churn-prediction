# 🔬 Telco Customer Churn Prediction — Master Pipeline

> **Đề tài:** Dự đoán khả năng khách hàng rời bỏ dịch vụ viễn thông
> **Phương pháp chính:** Random Forest · So sánh: Decision Tree, KNN, SVM
> **Metrics:** Accuracy, Precision, Recall, F1-score + Cost-based threshold
> **Ngày tạo:** 2026-10-08

---

## Tổng quan Pipeline

```text
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        MASTER PIPELINE                             │
 │                                                                    │
 │  1. Problem Definition ──► 2. Literature Review ──► 3. Dataset     │
 │           │                                             │          │
 │           ▼                                             ▼          │
 │  4. EDA ──────────► 5. Preprocessing ──────────► 6. Baseline       │
 │                                                       │            │
 │                                                       ▼            │
 │  7. Proposed Model ──► 8. Training ──► 9. Evaluation               │
 │                                             │                      │
 │                                             ▼                      │
 │  10. Comparison / Ablation ──────────► 11. Conclusion              │
 │                                             │                      │
 │                                             ▼                      │
 │  ╔═══════════════════════════════════════════════════╗              │
 │  ║  12. API (FastAPI) → 13. Frontend → 14. Deploy   ║              │
 │  ╚═══════════════════════════════════════════════════╝              │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## Milestone Overview

| Milestone | Mục tiêu | Deadline gợi ý |
|-----------|----------|-----------------|
| **M1 — Research & Data** | Bước 1–4 xong, hiểu rõ data | Tuần 1 |
| **M2 — ML Pipeline** | Bước 5–10 xong, có bảng so sánh | Tuần 2–3 |
| **M3 — Product** | API + Frontend chạy local | Tuần 4 |
| **M4 — Deploy & Report** | Docker + báo cáo hoàn chỉnh | Tuần 5 |

---

## Bước 1 — Problem Definition

### 1.1 Phát biểu bài toán

> Cho tập dữ liệu gồm N khách hàng viễn thông, mỗi khách hàng có d đặc trưng mô tả hành vi sử dụng dịch vụ. Xây dựng mô hình phân loại nhị phân để dự đoán nhãn `Churn ∈ {Yes, No}`, tối ưu theo **Recall lớp Churn** (giảm thiểu bỏ sót khách hàng sắp rời bỏ).

### 1.2 Lý do dùng ML thay vì rule-based

| Tiêu chí | Rule-based | ML |
|----------|------------|----|
| Khả năng phát hiện pattern phi tuyến | ✗ | ✓ |
| Mở rộng khi thêm feature | Khó | Tự động |
| Giải thích được | ✓ | ✓ (SHAP, feature importance) |

### 1.3 Business framing

```text
Chi phí giữ chân 1 khách hàng (retention offer):  ~$50–$100
Chi phí mất 1 khách hàng (CLV lost):              ~$500–$2,000
→ Ratio ≈ 1:10 đến 1:20
→ Ưu tiên Recall > Precision (chấp nhận False Positive để không bỏ sót)
```

> [!IMPORTANT]
> Ghi rõ business context này vào báo cáo. Đây là điểm khác biệt giữa bài tập "chạy model" và bài tập "có tư duy".

### 1.4 Deliverable bước 1

- [ ] Viết 1 đoạn Problem Statement (~200 từ)
- [ ] Xác định target variable: `Churn`
- [ ] Xác định metric chính: **F1-score (macro)** + **Recall lớp Churn**
- [ ] Ghi rõ: Accuracy là metric phụ, không dùng làm tiêu chí chọn model

---

## Bước 2 — Literature Review

### 2.1 Bộ paper tham chiếu chính

| # | Vai trò | Paper | Trích dẫn chính |
|---|---------|-------|-----------------|
| 1 | **Methodology / Pitfalls** | *The Hidden Costs of 99% Accuracy: A Trustworthiness Audit of the Telco Customer Churn Benchmark* (2026) | SMOTE leakage, feature redundancy (`TotalCharges ≈ tenure × MonthlyCharges`, R² = 0.9991), calibration, cost-based threshold |
| 2 | **Baseline trực tiếp** | *A Comprehensive Analysis of Churn Prediction in Telecommunications Using ML* — Chen et al. (2025) | LogReg: Acc 0.7318 / F1 0.6129; RF: Acc 0.7270 / F1 0.6011 trên IBM Telco |
| 3 | **Ensemble mở rộng** | *Enhancing Customer Churn Prediction: An Adaptive Ensemble Learning Approach* (2024) | Stacking/Boosting cải tiến RF |
| 4 | **Business metric** | *e-Profits: A Business-Aligned Evaluation Metric* (2025) | Tại sao Accuracy/F1 chưa đủ khi quy ra chi phí |
| 5 | **Tổng quan** | *A Survey on Churn Analysis* (2020) | Taxonomy các phương pháp churn |

### 2.2 Baseline từ Chen et al. (2025)

| Model | Accuracy | F1-score |
|-------|----------|----------|
| Logistic Regression | 0.7318 | 0.6129 |
| SGD | 0.7194 | 0.6074 |
| Decision Tree | 0.7441 | 0.5994 |
| Random Forest | 0.7270 | 0.6011 |
| MLP | 0.8226 | 0.8209 |

> [!WARNING]
> Nhiều paper báo RF đạt F1 > 0.85 trên Telco — hầu hết do **SMOTE leakage** (áp SMOTE trước khi chia train/test). Pipeline sạch cho RF chỉ đạt ~0.57–0.60 F1 lớp churn. **Đừng hoảng khi số thấp — hãy giải thích đúng.**

### 2.3 Deliverable bước 2

- [ ] Viết Related Work (~500–800 từ) dựa trên 5 paper trên
- [ ] Ghi rõ gap: "Phần lớn nghiên cứu không kiểm tra leakage pipeline"
- [ ] Bảng so sánh baseline trích từ Chen et al.

---

## Bước 3 — Dataset

### 3.1 Lựa chọn bộ dữ liệu

| Tiêu chí | IBM Telco (WA_Fn-UseC_-Telco-Customer-Churn) | Iranian Telecom Churn |
|-----------|----------------------------------------------|----------------------|
| Số dòng | 7,043 | 3,150 |
| Số cột | 21 | 14 (13 feature + 1 target) |
| Tỷ lệ Churn | 26.54% (1,869 / 5,174) | 15.71% |
| Có cột khiếu nại? | ✗ | ✓ (Complains) |
| Có tần suất sử dụng? | ✗ (có tenure, charges) | ✓ (Frequency of use, Seconds of use) |
| Có gói cước? | ✓ (InternetService, Contract, etc.) | ✗ |

> [!IMPORTANT]
> **Quyết định quan trọng:** Đề tài ghi "thời gian đăng ký, tần suất sử dụng, mức phí, khiếu nại, gói cước" — IBM Telco **không có** cột khiếu nại và tần suất sử dụng. Hai lựa chọn:
> 1. **Dùng IBM Telco** → sửa lại mô tả đặc trưng trong đề cương cho khớp
> 2. **Dùng Iranian Telecom** → khớp mô tả nhưng ít hơn 7K dòng
>
> **Khuyến nghị:** Dùng **IBM Telco** (phổ biến hơn, nhiều benchmark hơn, dễ so sánh kết quả) và sửa đề cương.

### 3.2 Mô tả IBM Telco Dataset

```text
Nguồn:  Kaggle — IBM Sample Data Sets
File:   WA_Fn-UseC_-Telco-Customer-Churn.csv
Shape:  (7043, 21)
Target: Churn (Yes/No)
```

#### Danh sách đặc trưng

| # | Feature | Type | Mô tả |
|---|---------|------|--------|
| 1 | `customerID` | ID | Mã KH — **DROP** |
| 2 | `gender` | Categorical | Male / Female |
| 3 | `SeniorCitizen` | Binary (0/1) | Khách hàng cao tuổi |
| 4 | `Partner` | Binary | Có đối tác |
| 5 | `Dependents` | Binary | Có người phụ thuộc |
| 6 | `tenure` | Numeric | Số tháng sử dụng DV |
| 7 | `PhoneService` | Binary | Dịch vụ điện thoại |
| 8 | `MultipleLines` | Categorical | Nhiều đường dây |
| 9 | `InternetService` | Categorical | DSL / Fiber optic / No |
| 10 | `OnlineSecurity` | Categorical | Bảo mật online |
| 11 | `OnlineBackup` | Categorical | Sao lưu online |
| 12 | `DeviceProtection` | Categorical | Bảo vệ thiết bị |
| 13 | `TechSupport` | Categorical | Hỗ trợ kỹ thuật |
| 14 | `StreamingTV` | Categorical | Streaming TV |
| 15 | `StreamingMovies` | Categorical | Streaming phim |
| 16 | `Contract` | Categorical | Month-to-month / 1yr / 2yr |
| 17 | `PaperlessBilling` | Binary | Hóa đơn điện tử |
| 18 | `PaymentMethod` | Categorical | Phương thức thanh toán |
| 19 | `MonthlyCharges` | Numeric | Phí hàng tháng |
| 20 | `TotalCharges` | Numeric* | Tổng phí — **XEM MỤC 3.3** |
| 21 | `Churn` | Binary | **TARGET** |

### 3.3 Cảnh báo: TotalCharges — đặc trưng thừa

```text
TotalCharges ≈ tenure × MonthlyCharges
R² = 0.9991 (theo bài audit)
```

> [!CAUTION]
> `TotalCharges` là quasi-redundant feature. Giữ nó trong model:
> - **Không** cải thiện accuracy đáng kể
> - **Làm sai lệch** SHAP values / feature importance (RF sẽ chia importance giữa 3 feature tương quan)
>
> **Khuyến nghị:** Chạy 2 phiên bản: có và không có `TotalCharges`. So sánh trong phần Ablation Study.

### 3.4 Deliverable bước 3

- [ ] Download dataset, đặt vào `data/raw/`
- [ ] Viết mô tả dataset (~300 từ)
- [ ] Ghi rõ: "TotalCharges sẽ được kiểm tra redundancy trong bước EDA"
- [ ] Kiểm tra `TotalCharges` có 11 giá trị rỗng (string `" "`) — cần xử lý

---

## Bước 4 — Exploratory Data Analysis (EDA)

### 4.1 Checklist EDA

```text
EDA Checklist
│
├── 4.1.1  Shape, dtypes, null counts
│          → df.info(), df.describe()
│          → Phát hiện TotalCharges là object (có " " thay vì NaN)
│
├── 4.1.2  Class Distribution
│          → df['Churn'].value_counts(normalize=True)
│          → Kỳ vọng: ~73.46% No, ~26.54% Yes
│          → VẼ: Bar chart + ghi % lên bar
│
├── 4.1.3  Univariate Analysis
│          ├── Numeric: histogram + boxplot (tenure, MonthlyCharges, TotalCharges)
│          └── Categorical: bar chart cho từng feature
│
├── 4.1.4  Bivariate Analysis (feature vs Churn)
│          ├── Numeric: violin plot / KDE plot tách theo Churn
│          └── Categorical: stacked bar chart (% churn trong mỗi category)
│
├── 4.1.5  Correlation Analysis
│          ├── Numeric: heatmap Pearson correlation
│          ├── Kiểm tra: corr(tenure, TotalCharges) → kỳ vọng > 0.8
│          └── Kiểm tra: TotalCharges ≈ tenure × MonthlyCharges
│
├── 4.1.6  Feature Redundancy Test
│          ├── Tính: computed = tenure * MonthlyCharges
│          ├── So sánh: R² giữa computed và TotalCharges
│          └── Kỳ vọng R² ≈ 0.999
│
└── 4.1.7  Key Insights (ghi lại 5–7 findings chính)
```

### 4.2 Code mẫu — Redundancy Test

```python
import numpy as np
from sklearn.metrics import r2_score

# Sau khi convert TotalCharges sang numeric
computed = df['tenure'] * df['MonthlyCharges']
r2 = r2_score(df['TotalCharges'].dropna(), computed.loc[df['TotalCharges'].dropna().index])
print(f"R² (TotalCharges vs tenure×MonthlyCharges) = {r2:.4f}")
# Kỳ vọng: ≈ 0.9991
```

### 4.3 Deliverable bước 4

- [ ] 7–10 biểu đồ EDA (lưu vào `figures/eda/`)
- [ ] Viết EDA findings (~400 từ)
- [ ] Kết luận: feature nào quan trọng nhất (tenure, Contract, MonthlyCharges)
- [ ] Ghi rõ: TotalCharges redundant → sẽ test bỏ trong ablation

---

## Bước 5 — Data Preprocessing

### 5.1 Pipeline ĐÚNG (Leak-free)

> [!CAUTION]
> **ĐÂY LÀ BƯỚC QUAN TRỌNG NHẤT.** Sai ở đây → toàn bộ kết quả sai.
>
> **Quy tắc vàng:** Mọi transform (scale, encode, SMOTE) phải **fit trên train** và **transform trên test**. KHÔNG BAO GIỜ fit trên toàn bộ data rồi mới chia.

```text
RAW DATA
    │
    ├── 5.1  Drop customerID
    ├── 5.2  Fix TotalCharges (11 giá trị " " → NaN → impute bằng median)
    ├── 5.3  Encode target: Churn → 1/0
    │
    ▼
TRAIN / TEST SPLIT  ◄── CHIA TRƯỚC, XỬ LÝ SAU
    │
    ├── train_raw (80%)
    └── test_raw  (20%)
         │
         ▼
    ┌─────────────────────────────────────┐
    │  FIT trên train_raw:                │
    │  ├── OrdinalEncoder / OneHotEncoder │
    │  ├── StandardScaler (cho KNN, SVM)  │
    │  └── SMOTE (chỉ trên train!)        │
    │                                     │
    │  TRANSFORM trên test_raw:           │
    │  ├── encoder.transform(test)        │
    │  └── scaler.transform(test)         │
    │  ⚠️  KHÔNG SMOTE trên test!         │
    └─────────────────────────────────────┘
```

### 5.2 Code mẫu — Pipeline đúng

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline  # imblearn, không phải sklearn
from imblearn.over_sampling import SMOTE

# ─── 5.2.1  Load & basic clean ─────────────────────────────
df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop(columns=["customerID"], inplace=True)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# 11 NaN → impute TRƯỚC khi chia (vì chỉ là convert type, không phải statistical fit)
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ─── 5.2.2  Feature / Target ────────────────────────────────
X = df.drop(columns=["Churn"])
y = df["Churn"]

# ─── 5.2.3  Train / Test split ──────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── 5.2.4  Column Transformer ──────────────────────────────
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical_features = [c for c in X.columns if c not in numeric_features]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
])

# ─── 5.2.5  Pipeline với SMOTE (chỉ dùng cho KNN/SVM) ──────
# Dùng imblearn Pipeline để SMOTE chỉ áp dụng trong fit()
pipeline_with_smote = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    # ("classifier", ...)  ← thêm sau
])

# Pipeline KHÔNG SMOTE (cho RF, Decision Tree — tự xử lý class_weight)
pipeline_no_smote = Pipeline([
    ("preprocessor", preprocessor),
    # ("classifier", ...)  ← thêm sau
])
```

### 5.3 SMOTE Leakage — Tại sao quan trọng

```text
Pipeline SAI (leaky):                Pipeline ĐÚNG (clean):
─────────────────────                ────────────────────
1. Load all data                     1. Load all data
2. SMOTE(all data)     ◄── LEAK!     2. Split train/test
3. Split train/test                  3. SMOTE(train only)
4. Train model                       4. Train model
5. Evaluate on test                  5. Evaluate on test

F1 churn (leaky):  ~0.72            F1 churn (clean):  ~0.57
Chênh lệch:       +13.11 pp (ảo!)
```

> [!WARNING]
> **Bài audit ghi rõ:** SMOTE leakage đẩy F1 lớp churn lên **+13.11 điểm phần trăm** so với pipeline sạch. Đây là lỗi phổ biến nhất trong các bài tập Telco Churn.

### 5.4 Scale hay không Scale?

| Model | Cần Scale? | Lý do |
|-------|-----------|-------|
| Decision Tree | ✗ | Split dựa trên threshold, không ảnh hưởng bởi magnitude |
| Random Forest | ✗ | Ensemble of trees, cùng lý do |
| KNN | **✓ BẮT BUỘC** | Distance-based → feature có range lớn sẽ dominate |
| SVM | **✓ BẮT BUỘC** | Kernel computation nhạy với scale |

> [!TIP]
> Để code gọn, dùng 1 pipeline có scale cho tất cả. Scale không làm hại tree-based models — chỉ cần ghi rõ trong phần thiết lập thí nghiệm: *"StandardScaler được áp dụng cho tất cả models để đảm bảo công bằng khi so sánh, mặc dù tree-based models không yêu cầu."*

### 5.5 Deliverable bước 5

- [ ] Code preprocessing pipeline (leak-free) → `src/preprocessing.py`
- [ ] Viết phần Preprocessing trong báo cáo (~300 từ)
- [ ] Ghi rõ thứ tự: split → encode → scale → SMOTE (train only)
- [ ] Giải thích tại sao chọn/không chọn SMOTE

---

## Bước 6 — Baseline Model

### 6.1 Baseline cần thiết

| Baseline | Mục đích |
|----------|----------|
| **Dummy (majority class)** | Đoán tất cả "No Churn" → Acc ~73.46%. Mọi model phải **tốt hơn** baseline này |
| **Logistic Regression** | Baseline ML đơn giản nhất. So sánh: nếu RF không thắng LogReg thì RF chưa đủ tốt |

### 6.2 Code mẫu

```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

# ─── Dummy baseline ─────────────────────────────────────
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)
print("=== DUMMY BASELINE ===")
print(classification_report(y_test, y_pred_dummy, target_names=["No Churn", "Churn"]))

# ─── Logistic Regression baseline ───────────────────────
lr_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
])
lr_pipe.fit(X_train, y_train)
y_pred_lr = lr_pipe.predict(X_test)
print("=== LOGISTIC REGRESSION ===")
print(classification_report(y_test, y_pred_lr, target_names=["No Churn", "Churn"]))
```

### 6.3 Kỳ vọng kết quả baseline (pipeline sạch)

| Model | Accuracy | F1 (Churn) | Recall (Churn) |
|-------|----------|-----------|----------------|
| Dummy (majority) | ~0.735 | 0.000 | 0.000 |
| Logistic Regression | ~0.73–0.75 | ~0.55–0.62 | ~0.50–0.60 |

### 6.4 Deliverable bước 6

- [ ] Chạy Dummy + LogReg baseline
- [ ] Ghi kết quả vào bảng
- [ ] Nhận xét: "Dummy đạt 73% accuracy nhưng F1 Churn = 0 → Accuracy một mình không đáng tin"

---

## Bước 7 — Proposed Models

### 7.1 Danh sách 4 models + config

#### 7.1.1 Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

dt_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced",
        max_depth=None,       # sẽ tune
        min_samples_split=2,  # sẽ tune
    )),
])
```

**Hyperparameter search space:**

```python
dt_params = {
    "classifier__max_depth": [3, 5, 7, 10, 15, None],
    "classifier__min_samples_split": [2, 5, 10, 20],
    "classifier__min_samples_leaf": [1, 2, 5, 10],
    "classifier__criterion": ["gini", "entropy"],
}
```

#### 7.1.2 Random Forest (Phương pháp chính)

```python
from sklearn.ensemble import RandomForestClassifier

rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced_subsample",  # xử lý imbalance nội tại
        n_jobs=-1,
    )),
])
```

**Hyperparameter search space:**

```python
rf_params = {
    "classifier__n_estimators": [100, 200, 500],
    "classifier__max_depth": [5, 10, 15, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 5],
    "classifier__max_features": ["sqrt", "log2", 0.3],
}
```

> [!TIP]
> Dùng `class_weight="balanced_subsample"` thay vì SMOTE cho RF. Đây là cách xử lý imbalance **không có risk leakage**.

#### 7.1.3 KNN

```python
from sklearn.neighbors import KNeighborsClassifier

knn_pipe = Pipeline([
    ("preprocessor", preprocessor),  # StandardScaler BẮT BUỘC
    ("classifier", KNeighborsClassifier()),
])
```

**Hyperparameter search space:**

```python
knn_params = {
    "classifier__n_neighbors": [3, 5, 7, 11, 15, 21],
    "classifier__weights": ["uniform", "distance"],
    "classifier__metric": ["euclidean", "manhattan"],
}
```

> [!WARNING]
> **KNN sau one-hot encoding:** Data sau one-hot khá thưa (nhiều chiều), KNN sẽ chậm và kém hiệu quả. Cân nhắc:
> - Giảm chiều bằng PCA trước KNN
> - Hoặc chỉ dùng numeric features cho KNN

#### 7.1.4 SVM

```python
from sklearn.svm import LinearSVC  # LinearSVC thay vì SVC để chạy nhanh

svm_pipe = Pipeline([
    ("preprocessor", preprocessor),  # StandardScaler BẮT BUỘC
    ("classifier", LinearSVC(
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )),
])
```

**Hyperparameter search space:**

```python
svm_params = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__loss": ["hinge", "squared_hinge"],
}
```

> [!WARNING]
> **SVC (kernel RBF) trên Telco cực chậm** — bài audit phải loại SVC khỏi benchmark. Dùng **LinearSVC** hoặc `SVC(kernel="linear")`. Nếu muốn RBF, lấy mẫu con ~2000 dòng để train.

### 7.2 Tuning Strategy

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Ví dụ cho RF
grid_rf = GridSearchCV(
    rf_pipe,
    rf_params,
    cv=cv,
    scoring="f1",           # ← F1 cho lớp positive (Churn)
    n_jobs=-1,
    verbose=1,
    refit=True,
)
grid_rf.fit(X_train, y_train)
print(f"Best params: {grid_rf.best_params_}")
print(f"Best CV F1:  {grid_rf.best_score_:.4f}")
```

> [!TIP]
> Nếu search space lớn, dùng `RandomizedSearchCV(n_iter=50)` thay vì `GridSearchCV` để tiết kiệm thời gian.

### 7.3 Deliverable bước 7

- [ ] Code cho 4 models với pipeline
- [ ] Hyperparameter search space cho mỗi model
- [ ] Giải thích tại sao chọn `LinearSVC` thay vì `SVC`
- [ ] Giải thích `class_weight="balanced"` vs SMOTE

---

## Bước 8 — Training

### 8.1 Training Protocol

```text
Cross-validation Setup:
├── Method:     StratifiedKFold (k=5)
├── Scoring:    F1-score (lớp Churn)
├── Tuning:     GridSearchCV / RandomizedSearchCV
├── Refit:      True (refit trên toàn bộ train set với best params)
└── Random state: 42 (reproducibility)
```

### 8.2 Logging kết quả

```python
import json
from datetime import datetime

results = {}
for name, grid in [("DT", grid_dt), ("RF", grid_rf), ("KNN", grid_knn), ("SVM", grid_svm)]:
    results[name] = {
        "best_params": grid.best_params_,
        "best_cv_f1": round(grid.best_score_, 4),
        "fit_time": round(grid.cv_results_["mean_fit_time"].mean(), 2),
    }

# Save
with open("results/training_log.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
```

### 8.3 Deliverable bước 8

- [ ] Chạy GridSearchCV cho 4 models
- [ ] Lưu best params vào `results/training_log.json`
- [ ] Ghi lại thời gian training cho mỗi model

---

## Bước 9 — Evaluation

### 9.1 Bộ metric đầy đủ

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt

def evaluate_model(name, model, X_test, y_test):
    """Evaluate model và trả về dict metrics."""
    y_pred = model.predict(X_test)

    # Nếu model có predict_proba
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, y_scores)
    else:
        roc_auc = None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (Churn)": precision_score(y_test, y_pred, pos_label=1),
        "Recall (Churn)": recall_score(y_test, y_pred, pos_label=1),
        "F1 (Churn)": f1_score(y_test, y_pred, pos_label=1),
        "F1 (Macro)": f1_score(y_test, y_pred, average="macro"),
        "ROC-AUC": roc_auc,
    }

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC: {roc_auc:.4f}" if roc_auc else "ROC-AUC: N/A")

    return metrics
```

### 9.2 Confusion Matrix Visualization

```python
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, (name, model) in zip(axes, models.items()):
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["No Churn", "Churn"],
        cmap="Blues", ax=ax
    )
    ax.set_title(name)
plt.tight_layout()
plt.savefig("figures/confusion_matrices.png", dpi=150)
plt.show()
```

### 9.3 Cost-based Threshold Optimization (Điểm nhấn học thuật)

```python
def find_cost_optimal_threshold(y_test, y_proba, cost_fp=50, cost_fn=500):
    """
    Tìm ngưỡng tối ưu theo chi phí business.

    cost_fp: Chi phí gửi retention offer cho KH không rời bỏ ($50)
    cost_fn: Chi phí mất KH do không phát hiện ($500)
    """
    thresholds = np.linspace(0.05, 0.95, 100)
    costs = []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        fp = ((y_pred_t == 1) & (y_test == 0)).sum()
        fn = ((y_pred_t == 0) & (y_test == 1)).sum()
        total_cost = fp * cost_fp + fn * cost_fn
        costs.append(total_cost)

    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    best_cost = costs[best_idx]

    # So sánh với ngưỡng mặc định 0.5
    y_pred_default = (y_proba >= 0.5).astype(int)
    fp_default = ((y_pred_default == 1) & (y_test == 0)).sum()
    fn_default = ((y_pred_default == 0) & (y_test == 1)).sum()
    default_cost = fp_default * cost_fp + fn_default * cost_fn

    print(f"Ngưỡng tối ưu chi phí:  {best_threshold:.3f}")
    print(f"Chi phí tại ngưỡng opt: ${best_cost:,.0f}")
    print(f"Chi phí tại ngưỡng 0.5: ${default_cost:,.0f}")
    print(f"Tiết kiệm:             ${default_cost - best_cost:,.0f}")

    return best_threshold, thresholds, costs

# Áp dụng cho RF (best model)
best_rf = grid_rf.best_estimator_
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]
opt_threshold, thresholds, costs = find_cost_optimal_threshold(y_test, y_proba_rf)
```

> [!TIP]
> Bài audit chỉ ra ngưỡng tối ưu chi phí **thấp hơn** ngưỡng tối ưu F1, tiết kiệm ~$77,646 trên mỗi 1,000 khách hàng. Đây là điểm **rất ấn tượng** trong báo cáo.

### 9.4 Deliverable bước 9

- [ ] Bảng metrics cho tất cả models
- [ ] 4 confusion matrix
- [ ] ROC curve (chồng 4 models trên 1 plot)
- [ ] Precision-Recall curve
- [ ] Cost-based threshold analysis cho RF
- [ ] Viết phần Evaluation (~500 từ)

---

## Bước 10 — Comparison & Ablation Study

### 10.1 Bảng so sánh tổng hợp

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) | F1 (Macro) | ROC-AUC | Training Time |
|-------|----------|-------------------|----------------|-----------|-----------|---------|---------------|
| Dummy | 0.735 | 0.000 | 0.000 | 0.000 | — | — | — |
| Logistic Reg. | — | — | — | — | — | — | — |
| Decision Tree | — | — | — | — | — | — | — |
| **Random Forest** | — | — | — | — | — | — | — |
| KNN | — | — | — | — | — | — | — |
| SVM (Linear) | — | — | — | — | — | — | — |

### 10.2 Ablation Study — 4 thí nghiệm

| # | Thí nghiệm | Biến đổi | So sánh |
|---|------------|----------|---------|
| A1 | **TotalCharges redundancy** | RF có vs không có TotalCharges | Acc, F1, Feature Importance |
| A2 | **SMOTE effect** | RF + SMOTE vs RF + class_weight | F1 Churn, Recall |
| A3 | **Scale effect** | RF có scale vs không scale | Acc, F1 (kỳ vọng: không thay đổi) |
| A4 | **Threshold effect** | RF @ threshold=0.5 vs threshold tối ưu | Cost, Recall, Precision |

### 10.3 Feature Importance (RF)

```python
import shap

# SHAP values cho RF
explainer = shap.TreeExplainer(best_rf.named_steps["classifier"])
X_test_transformed = best_rf.named_steps["preprocessor"].transform(X_test)
shap_values = explainer.shap_values(X_test_transformed)

# Summary plot
shap.summary_plot(shap_values[1], X_test_transformed,
                  feature_names=best_rf.named_steps["preprocessor"].get_feature_names_out(),
                  show=False)
plt.tight_layout()
plt.savefig("figures/shap_summary.png", dpi=150)
```

### 10.4 So sánh với Chen et al. Baseline

```text
Kết quả của bạn (pipeline sạch) vs Chen et al. (2025):

                 Bạn        Chen et al.    Chênh lệch
RF Accuracy:     ___         0.7270         ___
RF F1:           ___         0.6011         ___
```

### 10.5 Deliverable bước 10

- [ ] Bảng so sánh 6 models (Dummy + LR + 4 proposed)
- [ ] 4 ablation experiments
- [ ] SHAP summary plot
- [ ] Feature importance bar chart (RF built-in)
- [ ] So sánh kết quả với Chen et al.
- [ ] Viết phần Comparison (~400 từ)

---

## Bước 11 — Conclusion

### 11.1 Template kết luận

```text
1. Tóm tắt kết quả:
   - RF đạt F1 = ___ trên tập test, cao hơn DT/KNN/SVM __ điểm
   - Cost-based threshold giảm chi phí ___ USD so với ngưỡng mặc định

2. Phát hiện quan trọng:
   - TotalCharges là feature thừa (R² = 0.999)
   - SMOTE trước khi split gây inflation F1 +13 pp
   - Accuracy 73% = đoán tất cả "không rời bỏ"

3. Hạn chế:
   - Dataset chỉ 7K dòng, 1 snapshot
   - Không có temporal features (thời điểm đăng ký, xu hướng)
   - Chưa test trên real-time data

4. Hướng phát triển:
   - Ensemble (Stacking, XGBoost, LightGBM)
   - Feature engineering từ domain knowledge
   - Deploy thành API + dashboard
```

### 11.2 Deliverable bước 11

- [ ] Viết Conclusion (~300 từ)
- [ ] Viết Limitations (~200 từ)
- [ ] Viết Future Work (~200 từ)

---

## Bước 12–14 — Product (API + Frontend + Deploy)

### 12. API — FastAPI

```text
project/
├── api/
│   ├── main.py              # FastAPI app
│   ├── schemas.py            # Pydantic models (input/output)
│   ├── predict.py            # Load model + predict
│   └── model/
│       └── rf_best.joblib    # Trained model
```

```python
# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Telco Churn Prediction API")

model = joblib.load("api/model/rf_best.joblib")

class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

class PredictionOutput(BaseModel):
    churn_probability: float
    churn_prediction: str
    risk_level: str  # Low / Medium / High

@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    df = pd.DataFrame([customer.dict()])
    proba = model.predict_proba(df)[:, 1][0]

    # Cost-optimal threshold (từ bước 9)
    THRESHOLD = 0.35  # ← thay bằng giá trị tìm được

    prediction = "Churn" if proba >= THRESHOLD else "No Churn"
    risk = "High" if proba >= 0.7 else "Medium" if proba >= THRESHOLD else "Low"

    return PredictionOutput(
        churn_probability=round(proba, 4),
        churn_prediction=prediction,
        risk_level=risk,
    )
```

### 13. Frontend

```text
frontend/
├── index.html
├── style.css
└── app.js
```

**Tính năng chính:**
- Form nhập thông tin khách hàng
- Hiển thị kết quả: xác suất churn + risk level + gauge chart
- Giải thích (top 5 features ảnh hưởng) — SHAP
- Lịch sử dự đoán (localStorage)

### 14. Docker + Deploy

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY frontend/ frontend/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: "3.9"
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=api/model/rf_best.joblib
```

---

## Cấu trúc thư mục hoàn chỉnh

```text
telco-churn-prediction/
│
├── data/
│   ├── raw/                          # Dataset gốc
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/                    # Data sau preprocessing
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb        # Data cleaning & feature engineering
│   ├── 03_modeling.ipynb             # Train & evaluate models
│   └── 04_analysis.ipynb             # Ablation study & cost analysis
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Load & validate data
│   ├── preprocessing.py              # Pipeline preprocessing
│   ├── models.py                     # Model definitions & hyperparams
│   ├── evaluation.py                 # Metrics & visualization
│   └── utils.py                      # Helper functions
│
├── api/
│   ├── main.py                       # FastAPI application
│   ├── schemas.py                    # Request/Response models
│   ├── predict.py                    # Prediction logic
│   └── model/
│       └── rf_best.joblib            # Serialized best model
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── figures/
│   ├── eda/                          # EDA plots
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── precision_recall_curves.png
│   ├── shap_summary.png
│   ├── cost_threshold.png
│   └── feature_importance.png
│
├── results/
│   ├── training_log.json             # Best params & CV scores
│   ├── test_metrics.json             # Final test set metrics
│   └── comparison_table.csv          # Model comparison
│
├── report/
│   └── main.tex / main.docx          # Báo cáo
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Reporting Checklist (từ bài Audit)

> [!IMPORTANT]
> Áp dụng 4-component checklist từ bài Trustworthiness Audit vào báo cáo:

| # | Thành phần | Câu hỏi phải trả lời | Vị trí trong báo cáo |
|---|-----------|----------------------|----------------------|
| 1 | **Pipeline Order** | SMOTE/scale được áp TRƯỚC hay SAU khi chia train/test? | §5 Preprocessing |
| 2 | **Feature Redundancy** | TotalCharges có phải feature thừa không? R² = ? | §4 EDA + §10 Ablation |
| 3 | **Calibration** | Xác suất predict_proba có phản ánh đúng xác suất thực không? | §9 Evaluation |
| 4 | **Cost Threshold** | Ngưỡng quyết định tối ưu chi phí là bao nhiêu? Tiết kiệm bao nhiêu? | §9 Evaluation |

---

## Checklist tổng — Tracking tiến độ

```text
MILESTONE 1 — RESEARCH & DATA
├── [ ] Problem Statement
├── [ ] Literature Review (5 papers)
├── [ ] Dataset downloaded & described
├── [ ] EDA completed (7–10 plots)
└── [ ] Key findings documented

MILESTONE 2 — ML PIPELINE
├── [ ] Preprocessing pipeline (leak-free)
├── [ ] Baseline (Dummy + LogReg)
├── [ ] 4 models trained (DT, RF, KNN, SVM)
├── [ ] Hyperparameter tuning (GridSearchCV)
├── [ ] Evaluation metrics table
├── [ ] Confusion matrices
├── [ ] ROC & PR curves
├── [ ] Cost-based threshold analysis
├── [ ] Ablation study (4 experiments)
├── [ ] SHAP feature importance
└── [ ] So sánh với Chen et al. baseline

MILESTONE 3 — PRODUCT
├── [ ] FastAPI backend
├── [ ] Frontend (form + result display)
├── [ ] Model serialized (joblib)
└── [ ] End-to-end test local

MILESTONE 4 — DEPLOY & REPORT
├── [ ] Dockerfile + docker-compose
├── [ ] Deploy (cloud hoặc local)
├── [ ] Báo cáo hoàn chỉnh
├── [ ] README.md
└── [ ] Demo video / presentation
```

---

> [!NOTE]
> **Bạn hiện đang ở bước nào?** Gửi cho mình, mình sẽ hỗ trợ chi tiết từng bước — viết code, review kết quả, sửa pipeline.
