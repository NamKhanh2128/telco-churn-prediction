"""
Prediction logic — loads model + config, runs inference, generates explanation.

Separated from main.py so it can be unit-tested independently.
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "rf_best.joblib"
CONFIG_PATH = MODELS_DIR / "config.json"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Global state — loaded once at startup
_pipeline = None
_config = None
_metadata = None
_feature_importance = {}
_original_columns = None


def load_model():
    """Load the trained pipeline and config. Called once at app startup."""
    global _pipeline, _config, _metadata, _feature_importance, _original_columns

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Run 'python -m src.train' first."
        )

    _pipeline = joblib.load(MODEL_PATH)
    print(f"[predict] Loaded model from {MODEL_PATH}")

    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
        print(f'[predict] Loaded metadata from {METADATA_PATH}')
    else:
        _metadata = {}
    _feature_importance = _metadata.get('global_feature_importance', {})

    # Load threshold config
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            _config = json.load(f)
        print(f"[predict] Threshold: {_config['threshold']}")
    else:
        _config = {"threshold": 0.5, "cost_fp": 50, "cost_fn": 500}
        print(f"[predict] WARNING: config.json not found, using default threshold 0.5")

    # Store original column order for SHAP
    # These must match the columns the model was trained on
    _original_columns = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ]


    if _metadata.get('feature_columns'):
        _original_columns = list(_metadata['feature_columns'])


def predict_churn(customer_dict: dict) -> dict:
    """
    Run prediction for a single customer.

    Parameters
    ----------
    customer_dict : dict
        Customer features (from Pydantic model)

    Returns
    -------
    dict
        churn_probability, prediction, risk_level, threshold, top_factors, recommendation
    """
    if _pipeline is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Build DataFrame (model expects DataFrame input)
    df = pd.DataFrame([customer_dict])

    # Ensure column order matches training
    df = df.reindex(columns=_original_columns)

    # Predict probability
    proba = _pipeline.predict_proba(df)[:, 1][0]
    threshold = _config.get('threshold', 0.5)

    # Decision
    prediction = "Churn" if proba >= threshold else "No Churn"

    # Risk level
    if proba >= 0.7:
        risk_level = "High"
    elif proba >= threshold:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Top factors (simplified — using feature importance instead of per-instance SHAP for speed)
    top_factors = _get_top_factors(df, proba)

    # Recommendation
    recommendation = _generate_recommendation(risk_level, top_factors)

    return {
        "churn_probability": round(float(proba), 4),
        "prediction": prediction,
        "risk_level": risk_level,
        "threshold": threshold,
        "top_factors": top_factors,
        "recommendation": recommendation,
    }


def _get_top_factors(customer_df, proba, top_n=5):
    """
    Get the top factors driving the prediction.

    Uses a simplified approach for API speed:
    - Get RF feature importances
    - Map back to original features
    - Return top N with the customer's actual values

    For detailed SHAP per-instance, use src/explain.py.
    """
    try:
        if _feature_importance:
            sorted_features = sorted(
                _feature_importance.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            max_imp = sorted_features[0][1] if sorted_features else 1
            return [
                {
                    'feature': feature,
                    'value': str(customer_df[feature].iloc[0]) if feature in customer_df else 'N/A',
                    'impact': (
                        'high' if importance / max_imp > 0.6
                        else 'medium' if importance / max_imp > 0.3
                        else 'low'
                    ),
                }
                for feature, importance in sorted_features[:top_n]
            ]

        base_pipeline = _pipeline
        calibrated = getattr(_pipeline, 'calibrated_classifiers_', None)
        if calibrated:
            base_pipeline = getattr(calibrated[0], 'estimator', base_pipeline)
        classifier = base_pipeline.named_steps['classifier']
        preprocessor = base_pipeline.named_steps['preprocessor']

        # Get feature importances
        importances = classifier.feature_importances_
        feature_names = list(preprocessor.get_feature_names_out())

        # Aggregate to original features
        aggregated = {}
        for orig_col in _original_columns:
            matching_indices = []
            for i, enc_name in enumerate(feature_names):
                clean_name = enc_name
                for prefix in ["num__", "cat__"]:
                    if clean_name.startswith(prefix):
                        clean_name = clean_name[len(prefix):]
                if clean_name == orig_col or clean_name.startswith(f"{orig_col}_"):
                    matching_indices.append(i)

            if matching_indices:
                aggregated[orig_col] = sum(importances[i] for i in matching_indices)

        # Sort by importance
        sorted_features = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
        max_imp = sorted_features[0][1] if sorted_features else 1

        factors = []
        for feat, imp in sorted_features[:top_n]:
            ratio = imp / max_imp if max_imp > 0 else 0
            impact = "high" if ratio > 0.6 else "medium" if ratio > 0.3 else "low"
            value = str(customer_df[feat].iloc[0]) if feat in customer_df.columns else "N/A"
            factors.append({
                "feature": feat,
                "value": value,
                "impact": impact,
            })

        return factors

    except Exception:
        return []


def _generate_recommendation(risk_level, top_factors):
    """Generate actionable recommendation based on risk and factors."""
    if risk_level == "Low":
        return "Low churn risk. Continue standard service."

    factor_names = [f["feature"] for f in top_factors[:3]]

    suggestions = []

    if "Contract" in factor_names:
        suggestions.append("offer a longer-term contract with discount")
    if "tenure" in factor_names:
        suggestions.append("provide loyalty rewards for early-stage customers")
    if "MonthlyCharges" in factor_names:
        suggestions.append("review pricing or offer a competitive plan")
    if "TechSupport" in factor_names:
        suggestions.append("offer complimentary tech support")
    if "InternetService" in factor_names:
        suggestions.append("consider service quality improvements")

    if not suggestions:
        suggestions.append("proactive retention outreach recommended")

    if risk_level == "High":
        prefix = "HIGH RISK — Immediate action needed: "
    else:
        prefix = "Medium risk — Consider: "

    return prefix + "; ".join(suggestions) + "."


def get_threshold():
    """Get current threshold from config."""
    if _config is None:
        return 0.5
    return _config.get("threshold", 0.5)


def is_model_loaded():
    """Check if model is loaded."""
    return _pipeline is not None
