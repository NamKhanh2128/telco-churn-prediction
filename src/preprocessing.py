"""
Leak-free preprocessing pipeline.

KEY DESIGN DECISIONS (từ feedback + bài Audit):

1. Imputation (SimpleImputer) nằm BÊN TRONG ColumnTransformer
   → fit trên train, transform trên test. Không fit trên toàn bộ data.

2. SMOTE chỉ áp dụng trong imblearn Pipeline (chỉ chạy trong .fit())
   → test set KHÔNG BỊ SMOTE. Tránh leakage +13 pp F1 ảo.

3. StandardScaler cần cho KNN/SVM, không hại tree-based models.
   → Áp dụng cho tất cả để công bằng khi so sánh.

4. class_weight="balanced" cho tree-based models thay vì SMOTE
   → Đơn giản hơn, không risk leakage, hiệu quả tương đương.

Pipeline Structure:
─────────────────
ColumnTransformer
│
├── Numeric
│    ├── SimpleImputer(strategy="median")
│    └── StandardScaler()
│
└── Categorical
     ├── SimpleImputer(strategy="most_frequent")
     └── OneHotEncoder(drop="first", handle_unknown="ignore")
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(numeric_features, categorical_features, scale_numeric=True):
    """
    Build a ColumnTransformer that handles imputation + encoding + scaling.

    Imputation is INSIDE the pipeline → leak-free.

    Parameters
    ----------
    numeric_features : list[str]
    categorical_features : list[str]

    Returns
    -------
    sklearn.compose.ColumnTransformer
    """
    numeric_steps = [
        ('imputer', SimpleImputer(strategy='median')),
    ]
    if scale_numeric:
        numeric_steps.append(('scaler', StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # drop any columns not in numeric/categorical lists
    )

    return preprocessor


def build_model_pipeline(preprocessor, classifier):
    """
    Build a full Pipeline: preprocessor → classifier.

    For tree-based models (DT, RF): use class_weight="balanced" instead of SMOTE.
    For distance-based models (KNN, SVM): SMOTE can be added via imblearn Pipeline
    if needed, but class_weight is preferred.

    Parameters
    ----------
    preprocessor : ColumnTransformer
    classifier : sklearn estimator

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    return pipe


def get_feature_names_from_preprocessor(preprocessor, numeric_features, categorical_features):
    """
    Extract feature names after preprocessing.

    Useful for SHAP / feature importance interpretation.

    Parameters
    ----------
    preprocessor : fitted ColumnTransformer

    Returns
    -------
    list[str]
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback: manually construct
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
        return list(numeric_features) + cat_names
