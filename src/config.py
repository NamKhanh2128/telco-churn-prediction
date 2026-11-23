"""
Project-wide configuration.

Tập trung toàn bộ paths, hyperparameters, cost ratios ở đây
để không hard-code rải rác trong code.
"""

from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATASET_PATH = DATA_RAW_DIR / DATASET_FILENAME
ORANGE_DATASET_FILENAME = "telecom_churn.csv"
ORANGE_DATASET_PATH = DATA_RAW_DIR / "orange-telecom" / ORANGE_DATASET_FILENAME
IRANIAN_DATASET_PATH = DATA_RAW_DIR / "iranian-uci" / "Customer Churn.csv"

# ─── Dataset ─────────────────────────────────────────────────
TARGET_COL = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]

# Feature engineering for IBM Telco (interaction and ratio features)
USE_FEATURE_ENGINEERING = True
ENGINEERED_NUMERIC_FEATURES = [
    "charges_per_tenure",
    "total_additional_services",
    "streaming_services",
    "security_support_services",
    "is_high_risk_contract",
    "has_bundle",
]

# Will be auto-detected from non-numeric, non-target, non-ID columns
CATEGORICAL_FEATURES = None  # Set to None = auto-detect

# ─── Train / Test ────────────────────────────────────────────
# Final protocol: 70% train / 15% validation / 15% test (fractions of the full dataset).
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42
CV_FOLDS = 5

# Primary research protocol. Use --keep-total-charges for the A1 ablation.
DROP_TOTAL_CHARGES = True
CALIBRATE_PROBABILITIES = True
CALIBRATION_METHOD = 'sigmoid'
# Deployed RF calibration: False = one forest + one sigmoid calibrator fitted on
# cross-validated predictions. Same quality as ensemble=True (5 forests) but
# ~4x smaller on disk — see `python -m src.calibration_analysis`.
CALIBRATION_ENSEMBLE = False
PRIMARY_SCORING = 'average_precision'
SEARCH_ITERATIONS = 30
SEARCH_N_JOBS = 4

# ─── Business Cost Ratios ───────────────────────────────────
# FP: gửi retention offer cho KH không churn → lãng phí $50
# FN: không phát hiện KH sắp churn → mất CLV ~$500
COST_FP = 50    # False Positive cost (USD)
COST_FN = 500   # False Negative cost (USD)
COST_ACTION = COST_FP  # Cost of contacting/flagging any customer

# ─── Model defaults ─────────────────────────────────────────
DEFAULT_RF_PARAMS = {
    "n_estimators": 200,
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": SEARCH_N_JOBS,
}

DEFAULT_DT_PARAMS = {
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

DEFAULT_LR_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

DEFAULT_KNN_PARAMS = {
    "n_neighbors": 5,
    "weights": "distance",
}

DEFAULT_SVM_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 5000,
    "random_state": RANDOM_STATE,
}

# Kernel SVM (RBF) — the non-linear "SVM" named in the project brief.
DEFAULT_SVM_RBF_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# ─── Hyperparameter search spaces ───────────────────────────
RF_PARAM_GRID = {
    "classifier__n_estimators": [100, 200, 500],
    "classifier__max_depth": [5, 10, 15, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 5],
    "classifier__max_features": ["sqrt", "log2"],
}

DT_PARAM_GRID = {
    "classifier__max_depth": [3, 5, 7, 10, 15, None],
    "classifier__min_samples_split": [2, 5, 10, 20],
    "classifier__min_samples_leaf": [1, 2, 5, 10],
    "classifier__criterion": ["gini", "entropy"],
}

KNN_PARAM_GRID = {
    "classifier__n_neighbors": [3, 5, 7, 11, 15, 21],
    "classifier__weights": ["uniform", "distance"],
    "classifier__metric": ["euclidean", "manhattan"],
}

SVM_PARAM_GRID = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__loss": ["hinge", "squared_hinge"],
}

# Search space for the calibrated RBF SVM (CalibratedClassifierCV wraps SVC).
SVM_RBF_PARAM_GRID = {
    "classifier__estimator__C": [0.1, 0.5, 1, 5, 10],
    "classifier__estimator__gamma": ["scale", 0.01, 0.05, 0.1],
}
