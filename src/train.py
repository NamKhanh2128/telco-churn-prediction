"""
Main training script — the single entry point to train all models.

Usage:
    python -m src.train

Output:
    models/rf_best.joblib       ← Best Random Forest pipeline
    models/config.json          ← Threshold + cost config for API
    results/test_metrics.json   ← All metrics
    results/comparison_table.csv
    figures/confusion_matrices.png
    figures/roc_curves.png
    figures/precision_recall_curves.png
    figures/cost_threshold.png
    figures/shap_summary.png
    figures/feature_importance_aggregated.png

Pipeline order:
    1. Load data
    2. Split train/validation/test (BEFORE any transform)
    3. Build preprocessor (imputation INSIDE pipeline)
    4. Train Dummy + LogReg baselines
    5. Train DT + RF (main) + optional KNN/SVM
    6. Evaluate all on test set
    7. Calibrate RF and optimize threshold on validation only
    8. SHAP explainability (RF)
    9. Save everything
"""

import argparse
import json
import platform
import sys
import time
import joblib
import numpy as np
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    ParameterGrid,
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

from src.config import (
    TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE, CV_FOLDS,
    MODELS_DIR, RESULTS_DIR, DATASET_PATH,
    DROP_TOTAL_CHARGES, CALIBRATE_PROBABILITIES, CALIBRATION_METHOD,
    CALIBRATION_ENSEMBLE,
    PRIMARY_SCORING, SEARCH_ITERATIONS, SEARCH_N_JOBS,
    DEFAULT_RF_PARAMS, DEFAULT_DT_PARAMS, DEFAULT_LR_PARAMS,
    DEFAULT_KNN_PARAMS, DEFAULT_SVM_PARAMS, DEFAULT_SVM_RBF_PARAMS,
    RF_PARAM_GRID, DT_PARAM_GRID, KNN_PARAM_GRID, SVM_PARAM_GRID,
    SVM_RBF_PARAM_GRID,
    USE_FEATURE_ENGINEERING, ENGINEERED_NUMERIC_FEATURES,
)
from src.data_loader import (
    load_raw_data,
    get_file_sha256,
    get_feature_target,
    detect_feature_types,
)
from src.preprocessing import (
    build_preprocessor,
    build_model_pipeline,
    TelcoFeatureEngineer,
)
from src.evaluate import (
    print_evaluation, compute_metrics,
    plot_confusion_matrices, plot_roc_curves, plot_precision_recall_curves,
    plot_metric_summary, build_comparison_table, save_results,
)
from src.optimize_threshold import (
    find_cost_optimal_threshold, plot_threshold_analysis, save_model_config,
)
from src.explain import plot_shap_summary, plot_aggregated_importance


# ═══════════════════════════════════════════════════════════
#  Configuration — adjust these flags before running
# ═══════════════════════════════════════════════════════════

# Set to True to run hyperparameter tuning (slower but better)
DO_TUNING = False  # Start with False for quick baseline

# Set to True to include KNN and SVM (slower, optional per feedback)
INCLUDE_KNN_SVM = True

# Set to True to run SHAP analysis (requires shap package)
DO_SHAP = False

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def train_and_evaluate_model(name, pipeline, X_train, y_train, X_test, y_test,
                              param_grid=None, do_tuning=False):
    """
    Train a single model, optionally with RandomizedSearchCV, and evaluate.

    Returns
    -------
    dict with keys: pipeline, metrics, best_params, train_time
    """
    print(f"\n{'─' * 55}")
    print(f"  Training: {name}")
    print(f"{'─' * 55}")

    start = time.time()

    if do_tuning and param_grid:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        all_params = list(ParameterGrid(param_grid))
        n_iter = min(SEARCH_ITERATIONS, len(all_params))
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring={
                'average_precision': 'average_precision',
                'f1_macro': 'f1_macro',
                'f1_churn': 'f1',
            },
            refit=PRIMARY_SCORING,
            n_jobs=SEARCH_N_JOBS,
            verbose=0,
            random_state=RANDOM_STATE,
            return_train_score=True,
        )
        search.fit(X_train, y_train)
        fitted_pipeline = search.best_estimator_
        best_params = search.best_params_
        cv_score = round(search.best_score_, 4)
        print(f'  Best CV {PRIMARY_SCORING}: {cv_score}')
        print(f'  Best params: {best_params}')
    else:
        pipeline.fit(X_train, y_train)
        fitted_pipeline = pipeline
        best_params = None

    train_time = time.time() - start

    # Evaluate on test set
    y_pred = fitted_pipeline.predict(X_test)

    # Get probabilities if available
    y_proba = None
    if hasattr(fitted_pipeline, "predict_proba"):
        y_proba = fitted_pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(fitted_pipeline, "decision_function"):
        y_proba = fitted_pipeline.decision_function(X_test)

    metrics = print_evaluation(name, y_test, y_pred, y_proba)

    result = {
        "pipeline": fitted_pipeline,
        "metrics": metrics,
        "train_time": train_time,
    }
    if best_params:
        result["best_params"] = best_params

    return result


def main(drop_total_charges=None):
    """Main training pipeline."""

    print("\n" + "═" * 60)
    print("  TELCO CUSTOMER CHURN PREDICTION — TRAINING PIPELINE")
    print("═" * 60)

    # ─── 1. Load data ────────────────────────────────────
    print("\n[1/9] Loading data...")
    if drop_total_charges is None:
        drop_total_charges = DROP_TOTAL_CHARGES
    df = load_raw_data(drop_total_charges=drop_total_charges)
    X, y = get_feature_target(df)
    numeric_features, categorical_features = detect_feature_types(X)

    # ─── 2. Split ────────────────────────────────────────
    print("\n[2/9] Splitting train/test...")
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    validation_fraction = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_dev,
        y_train_dev,
        test_size=validation_fraction,
        random_state=RANDOM_STATE,
        stratify=y_train_dev,
    )
    print(f'  Validation: {X_val.shape[0]:,} samples')
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Test:  {X_test.shape[0]:,} samples")
    print(f"  Train churn rate: {y_train.mean():.2%}")
    print(f"  Test churn rate:  {y_test.mean():.2%}")

    # ─── 3. Build preprocessor ───────────────────────────
    print("\n[3/9] Building preprocessor...")
    fe_transformer = TelcoFeatureEngineer(enabled=USE_FEATURE_ENGINEERING)
    def make_pipeline(model, scale_numeric):
        return build_model_pipeline(
            build_preprocessor(
                numeric_features,
                categorical_features,
                scale_numeric=scale_numeric,
            ),
            model,
            feature_engineer=fe_transformer,
        )

    # ─── 4. Define models ────────────────────────────────
    print("\n[4/9] Defining models...")

    models = {}

    # Baseline: Dummy
    models["Dummy"] = {
        "pipeline": DummyClassifier(strategy="most_frequent"),
        "param_grid": None,
    }

    # Baseline: Logistic Regression (Calibrated with Platt scaling for reliable risk scoring)
    if CALIBRATE_PROBABILITIES:
        models["LogReg"] = {
            "pipeline": make_pipeline(
                CalibratedClassifierCV(
                    LogisticRegression(**DEFAULT_LR_PARAMS),
                    method=CALIBRATION_METHOD,
                    cv=CV_FOLDS,
                ),
                True,
            ),
            "param_grid": {
                "classifier__estimator__C": [0.01, 0.1, 1.0, 5.0, 10.0],
            },
        }
    else:
        models["LogReg"] = {
            "pipeline": make_pipeline(LogisticRegression(**DEFAULT_LR_PARAMS), True),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1.0, 5.0, 10.0],
            },
        }

    # Decision Tree
    models["DecisionTree"] = {
        "pipeline": make_pipeline(DecisionTreeClassifier(**DEFAULT_DT_PARAMS), False),
        "param_grid": DT_PARAM_GRID,
    }

    # Random Forest ⭐ (main method)
    models["RandomForest"] = {
        "pipeline": make_pipeline(RandomForestClassifier(**DEFAULT_RF_PARAMS), False),
        "param_grid": RF_PARAM_GRID,
    }

    # Optional: KNN and SVM
    if INCLUDE_KNN_SVM:
        models["KNN"] = {
            "pipeline": make_pipeline(KNeighborsClassifier(**DEFAULT_KNN_PARAMS), True),
            "param_grid": KNN_PARAM_GRID,
        }

        # Linear SVM (Calibrated with Platt scaling so predict_proba, Brier, and ECE are valid)
        if CALIBRATE_PROBABILITIES:
            models["SVM_Linear"] = {
                "pipeline": make_pipeline(
                    CalibratedClassifierCV(
                        LinearSVC(**DEFAULT_SVM_PARAMS),
                        method=CALIBRATION_METHOD,
                        cv=CV_FOLDS,
                    ),
                    True,
                ),
                "param_grid": {
                    "classifier__estimator__C": [0.01, 0.1, 1, 10],
                },
            }
        else:
            models["SVM_Linear"] = {
                "pipeline": make_pipeline(LinearSVC(**DEFAULT_SVM_PARAMS), True),
                "param_grid": SVM_PARAM_GRID,
            }

        # Kernel SVM (RBF). SVC has no native probabilities, so it is wrapped in
        # Platt scaling (same as SVM_Linear) to get valid predict_proba/Brier/ECE.
        models["SVM_RBF"] = {
            "pipeline": make_pipeline(
                CalibratedClassifierCV(
                    SVC(**DEFAULT_SVM_RBF_PARAMS),
                    method=CALIBRATION_METHOD,
                    cv=CV_FOLDS,
                ),
                True,
            ),
            "param_grid": SVM_RBF_PARAM_GRID,
        }

    # ─── 5. Train & Evaluate ─────────────────────────────
    print("\n[5/9] Training & evaluating models...")

    results = {}
    for name, model_def in models.items():
        is_dummy = name == "Dummy"
        results[name] = train_and_evaluate_model(
            name=name,
            pipeline=model_def["pipeline"],
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            param_grid=model_def["param_grid"],
            do_tuning=(DO_TUNING and not is_dummy),
        )

    # ─── 6. Comparison table ─────────────────────────────
    # (rebuilt in step 7 with the calibrated RF rows appended)
    print("\n[6/9] Building comparison table...")
    comparison_df = build_comparison_table(results)
    print("\n" + comparison_df.to_string())

    # ─── 7. Cost-based threshold (RF) ────────────────────
    print("\n[7/9] Cost-based threshold optimization (RF)...")
    rf_pipeline = results["RandomForest"]["pipeline"]

    rf_deployed = rf_pipeline
    if CALIBRATE_PROBABILITIES:
        rf_deployed = CalibratedClassifierCV(
            estimator=rf_pipeline,
            method=CALIBRATION_METHOD,
            cv=CV_FOLDS,
            ensemble=CALIBRATION_ENSEMBLE,
            n_jobs=SEARCH_N_JOBS,
        )
        rf_deployed.fit(X_train, y_train)
        print(f'  Calibrated RF probabilities with method={CALIBRATION_METHOD}')

    y_proba_val = rf_deployed.predict_proba(X_val)[:, 1]
    threshold_result = find_cost_optimal_threshold(
        y_val,
        y_proba_val,
        cost_mode='classification',
    )
    plot_threshold_analysis(threshold_result, save=True)
    model_config = save_model_config(threshold_result, model_name='random_forest')

    y_proba_test = rf_deployed.predict_proba(X_test)[:, 1]
    calibrated_default_metrics = compute_metrics(
        y_test,
        (y_proba_test >= 0.5).astype(int),
        y_proba_test,
    )
    y_pred_threshold = (y_proba_test >= model_config['threshold']).astype(int)
    threshold_metrics = compute_metrics(y_test, y_pred_threshold, y_proba_test)
    results['RandomForest']['threshold_metrics'] = threshold_metrics
    results['RandomForest']['calibrated_default_metrics'] = calibrated_default_metrics
    results['RandomForest']['policy_metrics'] = {
        'validation': threshold_result,
        'test': threshold_metrics,
    }

    # The "RandomForest" row is the uncalibrated tuned forest. Add explicit rows
    # for the deployed (calibrated) model at 0.5 and at the cost-optimal
    # threshold so the table never mixes the two silently.
    comparison_extra = {
        'RF_Calibrated@0.5': {'metrics': calibrated_default_metrics},
        f"RF_Calibrated@{model_config['threshold']:.4f}": {'metrics': threshold_metrics},
    }
    comparison_df = build_comparison_table({**results, **comparison_extra})

    # ─── 8. Save best model ──────────────────────────────
    print("\n[8/9] Saving best model...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "rf_best.joblib"
    joblib.dump(rf_deployed, model_path)
    print(f"  Saved: {model_path}")

    # ─── 9. Plots & SHAP ────────────────────────────────
    print("\n[9/9] Generating plots...")

    # Filter out Dummy for plot (it doesn't have the same interface)
    plot_results = {k: v for k, v in results.items() if k != "Dummy"}

    plot_confusion_matrices(plot_results, X_test, y_test)
    plot_roc_curves(plot_results, X_test, y_test)
    plot_precision_recall_curves(plot_results, X_test, y_test)
    plot_metric_summary(plot_results)

    if DO_SHAP:
        print("\n  Running SHAP analysis...")
        try:
            original_columns = list(X.columns)
            plot_shap_summary(rf_pipeline, X_test)
            plot_aggregated_importance(rf_pipeline, X_test, original_columns)
        except Exception as e:
            print(f"  WARNING: SHAP failed: {e}")
            print("  Skipping SHAP plots. Install shap: pip install shap")

    # ─── Save results ────────────────────────────────────
    save_results(results, comparison_df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_models_policy = {}
    for eval_name in ["RandomForest", "LogReg", "SVM_Linear", "SVM_RBF"]:
        if eval_name in results:
            m_pipe = rf_deployed if eval_name == "RandomForest" else results[eval_name]["pipeline"]
            if hasattr(m_pipe, "predict_proba"):
                m_val_p = m_pipe.predict_proba(X_val)[:, 1]
                m_opt = find_cost_optimal_threshold(y_val, m_val_p, cost_mode='classification')
                m_test_p = m_pipe.predict_proba(X_test)[:, 1]
                m_test_pol = compute_metrics(y_test, (m_test_p >= m_opt['best_threshold']).astype(int), m_test_p)
                all_models_policy[eval_name] = {
                    'best_threshold': m_opt['best_threshold'],
                    'validation_cost': m_opt['best_cost'],
                    'validation_savings': m_opt['savings'],
                    'test_policy_f1': m_test_pol['f1_churn'],
                    'test_policy_recall': m_test_pol['recall_churn'],
                    'test_policy_precision': m_test_pol['precision_churn'],
                }

    policy_payload = {
        'cost_mode': threshold_result['cost_mode'],
        'threshold_source': 'validation',
        'cost_fp': threshold_result['cost_fp'],
        'cost_fn': threshold_result['cost_fn'],
        'best_threshold': threshold_result['best_threshold'],
        'default_cost_on_validation': threshold_result['default_cost'],
        'optimal_cost_on_validation': threshold_result['best_cost'],
        'savings_on_validation': threshold_result['savings'],
        'test_metrics_at_default_threshold': calibrated_default_metrics,
        'test_metrics_at_selected_threshold': threshold_metrics,
        'all_models_policy': all_models_policy,
    }
    with open(RESULTS_DIR / 'policy_metrics.json', 'w') as f:
        json.dump(policy_payload, f, indent=2, default=str)

    global_feature_importance = {}
    try:
        fitted_preprocessor = rf_pipeline.named_steps['preprocessor']
        fitted_classifier = rf_pipeline.named_steps['classifier']
        transformed_names = list(fitted_preprocessor.get_feature_names_out())
        for original_feature in X.columns:
            importance = 0.0
            for index, transformed_name in enumerate(transformed_names):
                clean_name = transformed_name
                if clean_name.startswith('num__'):
                    clean_name = clean_name[5:]
                elif clean_name.startswith('cat__'):
                    clean_name = clean_name[5:]
                if clean_name == original_feature or clean_name.startswith(f'{original_feature}_'):
                    importance += float(fitted_classifier.feature_importances_[index])
            if importance > 0:
                global_feature_importance[original_feature] = round(importance, 8)
    except Exception as exc:
        print(f'  WARNING: could not save global feature importance: {exc}')

    metadata = {
        'model_version': 'rf-calibrated-v1',
        'model_file': model_path.name,
        'feature_columns': list(X.columns),
        'numeric_features': list(numeric_features),
        'categorical_features': list(categorical_features),
        'feature_engineering': USE_FEATURE_ENGINEERING,
        'global_feature_importance': global_feature_importance,
        'total_charges_policy': 'drop' if drop_total_charges else 'keep_ablation',
        'calibration': {
            'enabled': CALIBRATE_PROBABILITIES,
            'method': CALIBRATION_METHOD if CALIBRATE_PROBABILITIES else None,
            'cv_folds': CV_FOLDS if CALIBRATE_PROBABILITIES else None,
            'ensemble': CALIBRATION_ENSEMBLE if CALIBRATE_PROBABILITIES else None,
        },
        'split': {
            'train': int(len(X_train)),
            'validation': int(len(X_val)),
            'test': int(len(X_test)),
            'random_state': RANDOM_STATE,
        },
        'threshold': model_config,
        'dataset_sha256': get_file_sha256(DATASET_PATH),
        'python_version': platform.python_version(),
        'sklearn_version': sklearn.__version__,
    }
    with open(MODELS_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    with open(RESULTS_DIR / 'training_log.json', 'w') as f:
        json.dump(
            {
                'status': 'completed',
                'model_version': metadata['model_version'],
                'dataset_sha256': metadata['dataset_sha256'],
                'total_charges_policy': metadata['total_charges_policy'],
                'calibration': metadata['calibration'],
                'primary_scoring': PRIMARY_SCORING,
                'search_iterations': SEARCH_ITERATIONS if DO_TUNING else 0,
                'model_path': str(model_path),
            },
            f,
            indent=2,
            default=str,
        )

    # ─── Done ────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ✅ TRAINING COMPLETE")
    print("═" * 60)
    print(f"  Best model:     RandomForest")
    print(f"  Model file:     {model_path}")
    print(f"  Threshold:      {model_config.get('threshold', 0.5)}")
    print(f"  Results:        results/test_metrics.json")
    print(f"  Comparison:     results/comparison_table.csv")
    print(f"  Figures:        figures/")
    print("═" * 60)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the telco churn model.')
    parser.add_argument(
        '--keep-total-charges',
        action='store_true',
        help='Run the A1 ablation that keeps TotalCharges.',
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Run RandomizedSearchCV instead of the quick baseline.',
    )
    parser.add_argument(
        '--with-shap',
        action='store_true',
        help='Generate optional SHAP figures after training.',
    )
    args = parser.parse_args()
    if args.tune:
        DO_TUNING = True
    if args.with_shap:
        DO_SHAP = True
    main(drop_total_charges=False if args.keep_total_charges else None)
