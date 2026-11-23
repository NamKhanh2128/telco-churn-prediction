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

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Leak-free feature engineering transformer for Telco customer churn.

    Creates domain-specific non-linear interaction and ratio features:
    1. charges_per_tenure: MonthlyCharges / (tenure + 1)
    2. total_additional_services: count of active add-on services (0-6)
    3. streaming_services: count of streaming packages (0-2)
    4. security_support_services: count of security/tech packages (0-2)
    5. is_high_risk_contract: binary indicator for Month-to-month + Electronic check
    6. has_bundle: binary indicator for Internet + Phone bundle
    """

    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.enabled:
            return X
        X = X.copy()
        if 'MonthlyCharges' in X.columns and 'tenure' in X.columns:
            tenure_series = pd.to_numeric(X['tenure'], errors='coerce').fillna(0)
            X['charges_per_tenure'] = X['MonthlyCharges'] / (tenure_series + 1)

        services = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
        ]
        present_services = [s for s in services if s in X.columns]
        if present_services:
            X['total_additional_services'] = (X[present_services] == 'Yes').sum(axis=1)
            streaming = [s for s in ['StreamingTV', 'StreamingMovies'] if s in X.columns]
            if streaming:
                X['streaming_services'] = (X[streaming] == 'Yes').sum(axis=1)
            security = [s for s in ['OnlineSecurity', 'TechSupport'] if s in X.columns]
            if security:
                X['security_support_services'] = (X[security] == 'Yes').sum(axis=1)

        if 'Contract' in X.columns and 'PaymentMethod' in X.columns:
            X['is_high_risk_contract'] = (
                (X['Contract'] == 'Month-to-month') &
                (X['PaymentMethod'] == 'Electronic check')
            ).astype(int)

        if 'InternetService' in X.columns and 'PhoneService' in X.columns:
            X['has_bundle'] = (
                (X['InternetService'] != 'No') &
                (X['PhoneService'] == 'Yes')
            ).astype(int)

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        output_features = list(input_features)
        if self.enabled:
            for feat in [
                'charges_per_tenure', 'total_additional_services',
                'streaming_services', 'security_support_services',
                'is_high_risk_contract', 'has_bundle',
            ]:
                if feat not in output_features:
                    output_features.append(feat)
        return output_features


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


def build_model_pipeline(preprocessor, classifier, feature_engineer=None):
    """
    Build a full Pipeline: (optional feature_engineer) → preprocessor → classifier.

    For tree-based models (DT, RF): use class_weight="balanced" instead of SMOTE.
    For distance-based models (KNN, SVM): SMOTE can be added via imblearn Pipeline
    if needed, but class_weight is preferred.

    Parameters
    ----------
    preprocessor : ColumnTransformer
    classifier : sklearn estimator
    feature_engineer : TelcoFeatureEngineer, optional

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    steps = []
    if feature_engineer is not None:
        steps.append(("feature_engineer", feature_engineer))
    steps.append(("preprocessor", preprocessor))
    steps.append(("classifier", classifier))

    return Pipeline(steps=steps)


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
