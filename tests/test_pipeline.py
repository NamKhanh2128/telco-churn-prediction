"""
Tests for the ML pipeline — sanity checks before training.

Run: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Adjust path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASET_PATH, NUMERIC_FEATURES, TARGET_COL
from src.preprocessing import build_preprocessor, build_model_pipeline
from src.evaluate import compute_metrics
from src.optimize_threshold import find_cost_optimal_threshold
from src.compare_datasets import load_dataset


class TestDataLoading:
    """Test data loading and validation."""

    @pytest.fixture
    def sample_data(self):
        """Create a small sample dataset for testing."""
        np.random.seed(42)
        n = 100
        data = {
            "gender": np.random.choice(["Male", "Female"], n),
            "SeniorCitizen": np.random.choice([0, 1], n),
            "Partner": np.random.choice(["Yes", "No"], n),
            "Dependents": np.random.choice(["Yes", "No"], n),
            "tenure": np.random.randint(0, 72, n),
            "PhoneService": np.random.choice(["Yes", "No"], n),
            "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
            "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": np.random.choice(["Yes", "No"], n),
            "PaymentMethod": np.random.choice([
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ], n),
            "MonthlyCharges": np.random.uniform(20, 100, n),
            "TotalCharges": np.random.uniform(100, 5000, n),
            "Churn": np.random.choice([0, 1], n, p=[0.73, 0.27]),
        }
        return pd.DataFrame(data)

    def test_sample_data_shape(self, sample_data):
        assert sample_data.shape == (100, 20)

    def test_target_is_binary(self, sample_data):
        assert set(sample_data[TARGET_COL].unique()).issubset({0, 1})


class TestPreprocessing:
    """Test preprocessing pipeline."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        data = {
            "tenure": np.random.randint(0, 72, n),
            "MonthlyCharges": np.random.uniform(20, 100, n),
            "TotalCharges": np.random.uniform(100, 5000, n),
            "SeniorCitizen": np.random.choice([0, 1], n),
            "gender": np.random.choice(["Male", "Female"], n),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        }
        return pd.DataFrame(data)

    def test_preprocessor_builds(self, sample_data):
        numeric = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        categorical = ["gender", "Contract"]
        preprocessor = build_preprocessor(numeric, categorical)
        assert preprocessor is not None

    def test_preprocessor_transforms(self, sample_data):
        numeric = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        categorical = ["gender", "Contract"]
        preprocessor = build_preprocessor(numeric, categorical)

        X_transformed = preprocessor.fit_transform(sample_data)
        assert X_transformed.shape[0] == 100
        # 4 numeric + (2-1) gender + (3-1) contract = 4 + 1 + 2 = 7
        assert X_transformed.shape[1] == 7

    def test_no_nan_after_transform(self, sample_data):
        # Inject NaN
        sample_data.loc[0, "TotalCharges"] = np.nan
        sample_data.loc[5, "MonthlyCharges"] = np.nan

        numeric = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        categorical = ["gender", "Contract"]
        preprocessor = build_preprocessor(numeric, categorical)

        X_transformed = preprocessor.fit_transform(sample_data)
        assert not np.isnan(X_transformed).any(), "NaN found after preprocessing"

    def test_pipeline_with_classifier(self, sample_data):
        from sklearn.ensemble import RandomForestClassifier

        numeric = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        categorical = ["gender", "Contract"]
        preprocessor = build_preprocessor(numeric, categorical)

        y = np.random.choice([0, 1], len(sample_data))
        pipe = build_model_pipeline(preprocessor, RandomForestClassifier(n_estimators=10, random_state=42))
        pipe.fit(sample_data, y)

        predictions = pipe.predict(sample_data)
        assert len(predictions) == len(sample_data)
        assert set(predictions).issubset({0, 1})


class TestLeakagePrevention:
    """Verify that preprocessing is leak-free."""

    def test_imputer_not_fit_on_test(self):
        """
        Ensure that imputer statistics come from train only.
        """
        from sklearn.model_selection import train_test_split

        np.random.seed(42)
        n = 200

        # Create data where train and test have very different distributions
        train_charges = np.full(160, 50.0)  # train median = 50
        test_charges = np.full(40, 500.0)   # test median = 500

        # Inject NaN in test
        test_charges[0] = np.nan

        all_charges = np.concatenate([train_charges, test_charges])
        df = pd.DataFrame({
            "tenure": np.random.randint(0, 72, n),
            "MonthlyCharges": np.random.uniform(20, 100, n),
            "TotalCharges": all_charges,
            "SeniorCitizen": np.random.choice([0, 1], n),
        })

        X_train = df.iloc[:160]
        X_test = df.iloc[160:]

        preprocessor = build_preprocessor(
            ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"], []
        )
        preprocessor.fit(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # The imputed value for test NaN should be ~50 (train median),
        # NOT ~500 (test median)
        imputed_value = X_test_transformed[0, 2]  # TotalCharges is 3rd numeric
        # After scaling, the imputed value should be close to 0 (since train mean ≈ 50)
        # If leakage occurred, it would be far from 0
        assert abs(imputed_value) < 1.0, (
            f"Possible leakage: imputed value {imputed_value} suggests test data influenced imputation"
        )


def test_tree_preprocessor_can_skip_numeric_scaling():
    preprocessor = build_preprocessor(['tenure'], ['Contract'], scale_numeric=False)
    assert 'scaler' not in preprocessor.transformers[0][1].named_steps


def test_metrics_include_ranking_and_calibration_fields():
    metrics = compute_metrics(
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0.1, 0.8, 0.2, 0.7],
    )
    assert metrics['average_precision'] == 1.0
    assert metrics['brier_score'] is not None
    assert metrics['ece'] is not None
    assert metrics['lift_at_10pct'] is not None


def test_threshold_optimization_accepts_lists_and_action_cost_mode():
    result = find_cost_optimal_threshold(
        [0, 1, 0, 1],
        [0.1, 0.8, 0.9, 0.2],
        cost_fp=1,
        cost_fn=5,
        cost_action=2,
        cost_mode='action',
    )
    assert result['cost_mode'] == 'action'
    assert result['cost_action'] == 2
    assert 0.0 <= result['best_threshold'] <= 1.0


def test_iranian_dataset_profile_loads_without_concatenation():
    X, y, profile, audit = load_dataset('iranian')
    assert profile['label'] == 'Iranian Churn'
    assert X.shape == (3150, 13)
    assert set(y.unique()).issubset({0, 1})
    assert audit['missing_values'] == 0


def test_ibm_and_iranian_profiles_are_independent():
    ibm_X, _, _, _ = load_dataset('ibm')
    iranian_X, _, _, _ = load_dataset('iranian')
    assert 'TotalCharges' not in ibm_X.columns
    assert 'TotalCharges' not in iranian_X.columns
    assert set(ibm_X.columns) != set(iranian_X.columns)
