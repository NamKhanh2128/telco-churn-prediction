"""
SHAP explainability with feature aggregation.

KEY DESIGN DECISION (từ feedback):
    Sau OneHotEncoder, SHAP sẽ cho feature dạng:
        Contract_Month-to-month, Contract_One year, Contract_Two year

    Người dùng sản phẩm cần thấy:
        Contract ↑

    → Aggregate SHAP values từ encoded features về original features.

Pipeline:
    SHAP internal → aggregate encoded features → original feature name → UI
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR
from src.preprocessing import ENGINEERED_FEATURE_SOURCES


def compute_shap_values(pipeline, X_test, feature_names=None):
    """
    Compute SHAP values for the best model.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
        Must have "preprocessor" and "classifier" steps
    X_test : pd.DataFrame
        Raw test features (before preprocessing)
    feature_names : list[str], optional
        Feature names after preprocessing

    Returns
    -------
    shap_values : np.ndarray
    expected_value : float
    feature_names_out : list[str]
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform test data (passing through feature_engineer if present)
    if "feature_engineer" in pipeline.named_steps:
        X_test = pipeline.named_steps["feature_engineer"].transform(X_test)
    X_test_transformed = preprocessor.transform(X_test)

    # Get feature names
    if feature_names is None:
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X_test_transformed.shape[1])]

    # Compute SHAP
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test_transformed)

    # For binary classification, take values for class 1 (Churn)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
        expected_value = expected_value[1]

    return shap_values, expected_value, feature_names


def aggregate_shap_to_original_features(shap_values, encoded_feature_names, original_columns):
    """
    Aggregate SHAP values from one-hot encoded features back to original features.

    Example:
        Contract_Month-to-month, Contract_One year, Contract_Two year
        → Contract (sum of absolute SHAP values)

    Parameters
    ----------
    shap_values : np.ndarray
        Shape (n_samples, n_encoded_features)
    encoded_feature_names : list[str]
        Feature names after preprocessing (e.g., "cat__Contract_One year")
    original_columns : list[str]
        Original feature column names (e.g., "Contract")

    Returns
    -------
    pd.DataFrame
        Aggregated SHAP values with original feature names as columns
    """
    aggregated = {}

    for orig_col in original_columns:
        # Find all encoded features that map to this original column
        # ColumnTransformer prefixes with "num__" or "cat__"
        matching_indices = []
        for i, enc_name in enumerate(encoded_feature_names):
            # Strip prefix (num__, cat__)
            clean_name = enc_name
            for prefix in ["num__", "cat__"]:
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix):]

            # Match: exact match for numeric, or starts with "OrigCol_" for categorical
            if clean_name == orig_col or clean_name.startswith(f"{orig_col}_"):
                matching_indices.append(i)

        if matching_indices:
            # Sum absolute values for aggregation
            aggregated[orig_col] = np.abs(shap_values[:, matching_indices]).sum(axis=1)

    return pd.DataFrame(aggregated)


def input_features(pipeline):
    """Columns the pipeline's preprocessor consumes (raw + engineered features)."""
    preprocessor = pipeline.named_steps["preprocessor"]
    features = []
    for name, _, columns in preprocessor.transformers_:
        if name == "remainder" or isinstance(columns, str):
            continue
        features.extend(list(columns))
    return features


def aggregate_signed_shap(shap_values, encoded_feature_names, features):
    """Sum SIGNED SHAP values of encoded columns back to each input feature.

    Unlike aggregate_shap_to_original_features (absolute values, for global
    importance), the sign is kept: positive pushes towards churn, negative
    away from it. Used for per-customer explanations.
    """
    aggregated = {}
    for feature in features:
        indices = []
        for index, encoded_name in enumerate(encoded_feature_names):
            clean_name = encoded_name
            for prefix in ["num__", "cat__"]:
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix):]
            if clean_name == feature or clean_name.startswith(f"{feature}_"):
                indices.append(index)
        if indices:
            aggregated[feature] = np.asarray(shap_values)[:, indices].sum(axis=1)
    return pd.DataFrame(aggregated)


def _base_pipelines(model):
    """Fitted pipelines inside a model (one per CalibratedClassifierCV fold)."""
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        return [calibrated_clf.estimator for calibrated_clf in calibrated]
    return [model]


def signed_contributions(model, customer_df):
    """Signed SHAP contribution of every model input for one customer.

    ``model`` is a fitted tree pipeline or the deployed
    CalibratedClassifierCV(pipeline); for the latter the SHAP values of each
    calibrated fold's forest are averaged (one forest when ensemble=False).

    Returns ``(contributions, view)``: a Series indexed by model input (raw and
    engineered features) and a one-row frame with the customer's raw and
    engineered values.
    """
    pipelines = _base_pipelines(model)
    total = None
    view = customer_df
    for pipeline in pipelines:
        shap_values, _, encoded_names = compute_shap_values(pipeline, customer_df)
        signed = aggregate_signed_shap(
            shap_values, encoded_names, input_features(pipeline),
        ).iloc[0]
        total = signed if total is None else total.add(signed, fill_value=0.0)
        if "feature_engineer" in pipeline.named_steps:
            view = pipeline.named_steps["feature_engineer"].transform(customer_df)
    if total is None:
        return pd.Series(dtype=float), view
    return total / len(pipelines), view


def _impact(ratio):
    return "high" if ratio > 0.6 else "medium" if ratio > 0.3 else "low"


def explain_customer(model, customer_df, top_n=5):
    """Per-customer SHAP explanation for the API and the report.

    Works with a plain tree pipeline or with the deployed
    CalibratedClassifierCV(pipeline) (see signed_contributions).

    Scale: a contribution is a SHAP value of the forest's churn score (the
    average vote of the trees), i.e. BEFORE Platt calibration. The
    contributions add up to "forest score - average forest score", not to the
    calibrated probability the API shows. Calibration is monotonic, so it keeps
    the customers' risk ranking; but SHAP values do not carry over exactly
    through a non-linear map, so the factor order and direction are those of
    the forest score and usually, not necessarily, match an explanation of the
    calibrated probability.

    Returns a list (largest |contribution| first) of
    {"feature", "value", "contribution", "direction", "impact",
    "source_features"} where direction is "increases_churn" /
    "decreases_churn", impact is "high" / "medium" / "low" relative to the
    strongest factor, and source_features lists the raw columns a feature is
    computed from (the feature itself for raw columns, e.g. MonthlyCharges and
    tenure for the engineered charges_per_tenure).
    """
    contributions, view = signed_contributions(model, customer_df)
    if contributions.empty:
        return []
    ordered = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    strongest = float(ordered.abs().iloc[0]) or 1.0

    factors = []
    for feature, contribution in ordered.head(top_n).items():
        factors.append({
            "feature": feature,
            "value": str(view[feature].iloc[0]) if feature in view.columns else "N/A",
            "contribution": round(float(contribution), 4),
            "direction": "increases_churn" if contribution > 0 else "decreases_churn",
            "impact": _impact(abs(float(contribution)) / strongest),
            "source_features": list(ENGINEERED_FEATURE_SOURCES.get(feature, [feature])),
        })
    return factors


def get_top_factors_for_customer(pipeline, customer_df, original_columns, top_n=5):
    """
    Top factors of one customer in the short format {feature, value, impact}.

    Kept for callers of the older API. Only the columns in
    ``original_columns`` are ranked (by |signed SHAP contribution|), so every
    factor has a value the customer actually entered. Works with a plain tree
    pipeline and with the deployed calibrated model.

    Parameters
    ----------
    pipeline : fitted tree pipeline or CalibratedClassifierCV(pipeline)
    customer_df : pd.DataFrame
        Single row DataFrame with customer features
    original_columns : list[str]
        Raw feature names to rank
    top_n : int
        Number of top factors to return

    Returns
    -------
    list[dict]
        [{"feature": "Contract", "value": "Month-to-month", "impact": "high"}, ...]
    """
    contributions, _ = signed_contributions(pipeline, customer_df)
    contributions = contributions.reindex(
        [column for column in original_columns if column in contributions.index]
    )
    if contributions.empty:
        return []

    importance = contributions.abs().sort_values(ascending=False)
    max_val = float(importance.iloc[0])
    factors = []
    for feat, val in importance.head(top_n).items():
        ratio = float(val) / max_val if max_val > 0 else 0.0
        feat_value = customer_df[feat].iloc[0] if feat in customer_df.columns else "N/A"
        factors.append({
            "feature": feat,
            "value": str(feat_value),
            "impact": _impact(ratio),
        })
    return factors


def plot_shap_summary(pipeline, X_test, feature_names=None, save=True):
    """
    Generate SHAP summary plot.
    """
    shap_values, _, feature_names = compute_shap_values(pipeline, X_test, feature_names)

    X_test_proc = X_test
    if "feature_engineer" in pipeline.named_steps:
        X_test_proc = pipeline.named_steps["feature_engineer"].transform(X_test_proc)
    X_test_transformed = pipeline.named_steps["preprocessor"].transform(X_test_proc)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test_transformed,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.title("SHAP Feature Importance (Top 15)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        print(f"[explain] Saved shap_summary.png")
    plt.close()


def plot_aggregated_importance(pipeline, X_test, original_columns, save=True):
    """
    Plot feature importance aggregated to original feature names.

    This is what goes in the report — clean, interpretable.
    """
    shap_values, _, feature_names = compute_shap_values(pipeline, X_test)
    agg_df = aggregate_shap_to_original_features(shap_values, feature_names, original_columns)

    # Mean absolute SHAP per feature
    mean_importance = agg_df.mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    mean_importance.plot(kind="barh", ax=ax, color="#3498db", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("Feature Importance (Aggregated)", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "feature_importance_aggregated.png", dpi=150, bbox_inches="tight")
        print(f"[explain] Saved feature_importance_aggregated.png")
    plt.close(fig)
