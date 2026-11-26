"""
Unit tests for comparison models (Decision Tree, KNN, SVM_Linear, SVM_RBF).

Run: pytest tests/test_models.py -q
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CALIBRATION_METHOD,
    DEFAULT_DT_PARAMS,
    DEFAULT_KNN_PARAMS,
    DEFAULT_SVM_PARAMS,
    DEFAULT_SVM_RBF_PARAMS,
    DT_PARAM_GRID,
    KNN_PARAM_GRID,
    SVM_PARAM_GRID,
    SVM_RBF_PARAM_GRID,
)
from src.preprocessing import build_model_pipeline, build_preprocessor


@pytest.fixture
def toy_dataset():
    """Small synthetic dataset with numeric and categorical columns (no disk I/O)."""
    rng = np.random.default_rng(42)
    n = 80
    X = pd.DataFrame({
        "tenure": rng.integers(1, 72, size=n),
        "MonthlyCharges": rng.uniform(20.0, 110.0, size=n),
        "SeniorCitizen": rng.choice([0, 1], size=n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
    })
    # Signal correlated with tenure & Contract so models can learn cleanly
    prob = np.where(X["Contract"] == "Month-to-month", 0.55, 0.15)
    prob = np.where(X["tenure"] < 12, np.minimum(prob + 0.2, 0.9), prob)
    y = pd.Series(rng.binomial(1, prob), name="Churn")
    numeric_features = ["tenure", "MonthlyCharges", "SeniorCitizen"]
    categorical_features = ["Contract", "InternetService"]
    return X, y, numeric_features, categorical_features


def _build_comparison_pipeline(model_name, numeric_features, categorical_features):
    """Build comparison model pipeline matching src/train.py conventions."""
    if model_name == "DecisionTree":
        classifier = DecisionTreeClassifier(**DEFAULT_DT_PARAMS)
        scale_numeric = False
    elif model_name == "KNN":
        classifier = KNeighborsClassifier(**DEFAULT_KNN_PARAMS)
        scale_numeric = True
    elif model_name == "SVM_Linear":
        classifier = CalibratedClassifierCV(
            LinearSVC(**DEFAULT_SVM_PARAMS),
            method=CALIBRATION_METHOD,
            cv=3,
        )
        scale_numeric = True
    elif model_name == "SVM_RBF":
        classifier = CalibratedClassifierCV(
            SVC(**DEFAULT_SVM_RBF_PARAMS),
            method=CALIBRATION_METHOD,
            cv=3,
        )
        scale_numeric = True
    else:
        raise ValueError(f"Unknown model: {model_name}")

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=scale_numeric,
    )
    return build_model_pipeline(preprocessor, classifier)


@pytest.mark.parametrize("model_name", ["DecisionTree", "KNN", "SVM_Linear", "SVM_RBF"])
def test_comparison_model_fit_and_predict_proba(toy_dataset, model_name):
    """Each comparison model fits on small data and outputs valid probabilities in [0, 1]."""
    X, y, num_cols, cat_cols = toy_dataset
    pipeline = _build_comparison_pipeline(model_name, num_cols, cat_cols)

    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    proba = pipeline.predict_proba(X)

    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def _numeric_steps(pipeline):
    """Return the step mapping of the numeric sub-pipeline before or after fit."""
    transformers = {
        name: trans
        for name, trans, _ in pipeline.named_steps["preprocessor"].transformers
    }
    return transformers["num"].named_steps


def test_scaling_policy_for_comparison_models(toy_dataset):
    """KNN and SVM must scale numeric features; DecisionTree must not."""
    _, _, num_cols, cat_cols = toy_dataset

    for scaled_model in ["KNN", "SVM_Linear", "SVM_RBF"]:
        pipe = _build_comparison_pipeline(scaled_model, num_cols, cat_cols)
        assert isinstance(_numeric_steps(pipe).get("scaler"), StandardScaler), (
            f"{scaled_model} should use StandardScaler on numeric features"
        )

    dt_pipe = _build_comparison_pipeline("DecisionTree", num_cols, cat_cols)
    assert "scaler" not in _numeric_steps(dt_pipe), (
        "DecisionTree should not scale numeric features"
    )


def test_param_grids_match_pipeline_params(toy_dataset):
    """Config hyperparameter grids use valid parameter keys for their pipelines."""
    _, _, num_cols, cat_cols = toy_dataset

    # DecisionTree & KNN: direct classifier__ keys
    for model_name, grid in [("DecisionTree", DT_PARAM_GRID), ("KNN", KNN_PARAM_GRID)]:
        pipe = _build_comparison_pipeline(model_name, num_cols, cat_cols)
        valid_keys = set(pipe.get_params().keys())
        assert set(grid.keys()).issubset(valid_keys), (
            f"Invalid keys in {model_name} grid: {set(grid.keys()) - valid_keys}"
        )

    # SVM_RBF: wrapped in CalibratedClassifierCV (classifier__estimator__*)
    svm_rbf_pipe = _build_comparison_pipeline("SVM_RBF", num_cols, cat_cols)
    assert set(SVM_RBF_PARAM_GRID.keys()).issubset(set(svm_rbf_pipe.get_params().keys()))

    # SVM_Linear trong benchmark thực tế bọc CalibratedClassifierCV → grid dùng
    # classifier__estimator__C (xem SVM_RBF_PARAM_GRID). SVM_PARAM_GRID với key
    # classifier__* dùng cho LinearSVC trần trong src/train.py. Test dưới đây
    # kiểm tra grid keys hợp lệ với pipeline LinearSVC trần (không bọc calibration).
    linear_pipe = build_model_pipeline(
        build_preprocessor(num_cols, cat_cols, scale_numeric=True),
        LinearSVC(**DEFAULT_SVM_PARAMS),
    )
    assert set(SVM_PARAM_GRID.keys()).issubset(set(linear_pipe.get_params().keys()))
