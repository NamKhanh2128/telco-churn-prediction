'''Run the approved churn datasets independently and compare them.

This module deliberately keeps IBM Telco, Orange Telecom, and Iranian Churn in
separate artifacts. It uses the same train/validation/test protocol and the
same Random Forest deployment policy for all datasets, so the comparison is
about dataset behaviour across real-world paradigms rather than concatenated data.
'''

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from src.config import (
    CALIBRATION_METHOD,
    CALIBRATE_PROBABILITIES,
    CV_FOLDS,
    COST_FN,
    COST_FP,
    DATA_RAW_DIR,
    ENGINEERED_NUMERIC_FEATURES,
    FIGURES_DIR,
    MODELS_DIR,
    PRIMARY_SCORING,
    RANDOM_STATE,
    RESULTS_DIR,
    SEARCH_ITERATIONS,
    SEARCH_N_JOBS,
    TEST_SIZE,
    USE_FEATURE_ENGINEERING,
    VALIDATION_SIZE,
    DEFAULT_LR_PARAMS,
    DEFAULT_RF_PARAMS,
    RF_PARAM_GRID,
)
from src.data_loader import get_file_sha256
from src.evaluate import compute_metrics
from src.optimize_threshold import find_cost_optimal_threshold
from src.preprocessing import (
    build_model_pipeline,
    build_preprocessor,
    TelcoFeatureEngineer,
)


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


DATASET_PROFILES = {
    'ibm': {
        'label': 'IBM Telco',
        'path': DATA_RAW_DIR / 'WA_Fn-UseC_-Telco-Customer-Churn.csv',
        'target': 'Churn',
    },
    'orange': {
        'label': 'Orange Telecom',
        'path': DATA_RAW_DIR / 'orange-telecom' / 'telecom_churn.csv',
        'target': 'Churn',
    },
    'iranian': {
        'label': 'Iranian Churn',
        'path': DATA_RAW_DIR / 'iranian-uci' / 'Customer Churn.csv',
        'target': 'Churn',
    },
}


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file_handle:
        json.dump(payload, file_handle, indent=2, default=str)


def _normalise_columns(df):
    result = df.copy()
    result.columns = (
        result.columns.astype(str)
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
    )
    return result


def load_dataset(dataset_name, keep_total_charges=False):
    '''Load one dataset and return X, y, profile and a small audit.'''
    profile = DATASET_PROFILES[dataset_name]
    path = Path(profile['path'])
    if not path.exists():
        raise FileNotFoundError(
            f'Dataset not found at {path}. '
            f'Download the official file before running this comparison.'
        )

    raw = pd.read_csv(path)
    raw = _normalise_columns(raw)

    target = profile['target']
    if target not in raw.columns:
        raise ValueError(f'Missing target column {target!r} in {path}')

    if dataset_name == 'ibm':
        if 'customerID' not in raw.columns:
            raise ValueError('IBM dataset is missing customerID')
        raw['TotalCharges'] = pd.to_numeric(raw['TotalCharges'], errors='coerce')
        raw[target] = (
            raw[target]
            .astype(str)
            .str.strip()
            .map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
        )
        raw = raw.drop(columns=['customerID'])
        if not keep_total_charges:
            raw = raw.drop(columns=['TotalCharges'])

    elif dataset_name == 'orange':
        raw[target] = (
            raw[target]
            .astype(str)
            .str.strip()
            .map({
                'True': 1, 'False': 0,
                '1': 1, '0': 0,
                'Yes': 1, 'No': 0,
                'True.': 1, 'False.': 0,
            })
        )
        cat_columns = ['State', 'Area code', 'International plan', 'Voice mail plan']
        for col in cat_columns:
            if col in raw.columns:
                raw[col] = raw[col].astype(str).str.strip()
        num_columns = [col for col in raw.columns if col not in cat_columns and col != target]
        for col in num_columns:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')

    else:
        # Iranian
        raw[target] = pd.to_numeric(raw[target], errors='coerce')
        feature_columns = [column for column in raw.columns if column != target]
        for column in feature_columns:
            raw[column] = pd.to_numeric(raw[column], errors='coerce')

    if raw[target].isna().any():
        raise ValueError(f'Target contains missing or unmapped values in {path}')

    X = raw.drop(columns=[target])
    y = raw[target].astype(int)
    missing = int(X.isna().sum().sum())
    audit = {
        'rows': int(len(raw)),
        'features': int(X.shape[1]),
        'missing_values': missing,
        'duplicate_rows': int(raw.duplicated().sum()),
        'churn_rate': float(y.mean()),
        'sha256': get_file_sha256(path),
    }
    return X, y, profile, audit


def _scores(estimator, X):
    if hasattr(estimator, 'predict_proba'):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, 'decision_function'):
        return estimator.decision_function(X)
    return None


def _build_model(name, numeric_features, categorical_features, is_ibm=False):
    if name == 'RandomForest':
        classifier = RandomForestClassifier(**DEFAULT_RF_PARAMS)
        scale_numeric = False
    elif name == 'LogReg':
        base_lr = LogisticRegression(**DEFAULT_LR_PARAMS)
        if CALIBRATE_PROBABILITIES:
            classifier = CalibratedClassifierCV(
                base_lr,
                method=CALIBRATION_METHOD,
                cv=CV_FOLDS,
            )
        else:
            classifier = base_lr
        scale_numeric = True
    else:
        classifier = DummyClassifier(strategy='most_frequent')
        scale_numeric = False

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=scale_numeric,
    )
    feature_engineer = TelcoFeatureEngineer(enabled=True) if is_ibm else None
    return build_model_pipeline(
        preprocessor,
        classifier,
        feature_engineer=feature_engineer,
    )


def _fit_model(name, pipeline, X_train, y_train, tune=False):
    start = time.time()
    best_params = None

    if tune and name == 'RandomForest':
        cv = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        n_iter = min(SEARCH_ITERATIONS, len(list(ParameterGrid(RF_PARAM_GRID))))
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=RF_PARAM_GRID,
            n_iter=n_iter,
            scoring={
                'average_precision': 'average_precision',
                'f1_macro': 'f1_macro',
                'f1_churn': 'f1',
            },
            refit=PRIMARY_SCORING,
            cv=cv,
            n_jobs=SEARCH_N_JOBS,
            random_state=RANDOM_STATE,
            return_train_score=True,
        )
        search.fit(X_train, y_train)
        fitted = search.best_estimator_
        best_params = search.best_params_
    else:
        pipeline.fit(X_train, y_train)
        fitted = pipeline

    return fitted, best_params, time.time() - start


def _choose_group_fold(X, y, groups, n_splits, target_fraction, seed):
    '''Choose the stratified-group fold closest to target size and prevalence.'''
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    dummy = np.zeros((len(y), 1))
    overall_rate = float(np.mean(y))
    candidates = []
    for train_idx, heldout_idx in splitter.split(dummy, y, groups):
        heldout_rate = float(np.mean(y.iloc[heldout_idx]))
        fraction_error = abs(len(heldout_idx) / len(y) - target_fraction)
        rate_error = abs(heldout_rate - overall_rate)
        candidates.append(
            (fraction_error + 0.25 * rate_error, train_idx, heldout_idx)
        )
    _, train_idx, heldout_idx = min(candidates, key=lambda item: item[0])
    return np.asarray(train_idx), np.asarray(heldout_idx)


def _group_split_indices(X, y, groups, seed=RANDOM_STATE):
    '''Create an approximately 70/15/15 split without duplicate groups crossing.'''
    train_dev, test = _choose_group_fold(
        X,
        y,
        groups,
        n_splits=max(2, round(1 / TEST_SIZE)),
        target_fraction=TEST_SIZE,
        seed=seed,
    )
    train_dev_groups = groups[train_dev]
    train_dev_y = y.iloc[train_dev].reset_index(drop=True)
    train_rel, validation_rel = _choose_group_fold(
        X.iloc[train_dev].reset_index(drop=True),
        train_dev_y,
        train_dev_groups,
        n_splits=max(2, round((1 - TEST_SIZE) / VALIDATION_SIZE)),
        target_fraction=VALIDATION_SIZE / (1 - TEST_SIZE),
        seed=seed + 1000,
    )
    train = train_dev[train_rel]
    validation = train_dev[validation_rel]
    return np.asarray(train), np.asarray(validation), np.asarray(test)


def _split_data(X, y, dataset_name='ibm', split_method='default'):
    '''Split dataset into 70/15/15 train/val/test splits.'''
    if dataset_name == 'iranian' and split_method != 'row_wise':
        # Group-aware split to prevent identical feature vectors crossing boundaries
        groups = pd.util.hash_pandas_object(X, index=False).astype(str).to_numpy()
        train_idx, val_idx, test_idx = _group_split_indices(X, y, groups, seed=RANDOM_STATE)
        return (
            X.iloc[train_idx].reset_index(drop=True),
            X.iloc[val_idx].reset_index(drop=True),
            X.iloc[test_idx].reset_index(drop=True),
            y.iloc[train_idx].reset_index(drop=True),
            y.iloc[val_idx].reset_index(drop=True),
            y.iloc[test_idx].reset_index(drop=True),
        )

    X_train_dev, X_test, y_train_dev, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    validation_fraction = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_dev,
        y_train_dev,
        test_size=validation_fraction,
        random_state=RANDOM_STATE,
        stratify=y_train_dev,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_dataset(dataset_name, tune=False, keep_total_charges=False, split_method='default'):
    '''Train one dataset and write artifacts under a dataset-specific folder.'''
    run_name = dataset_name
    if dataset_name == 'ibm' and keep_total_charges:
        run_name = 'ibm_with_total_charges'

    X, y, profile, audit = load_dataset(
        dataset_name,
        keep_total_charges=keep_total_charges,
    )

    numeric_features = list(X.select_dtypes(include=np.number).columns)
    categorical_features = [
        column for column in X.columns if column not in numeric_features
    ]

    is_ibm = (dataset_name == 'ibm')
    if is_ibm and USE_FEATURE_ENGINEERING:
        for feat in ENGINEERED_NUMERIC_FEATURES:
            if feat not in numeric_features:
                numeric_features.append(feat)

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        X, y, dataset_name=dataset_name, split_method=split_method,
    )

    actual_split = (
        'group_aware'
        if (dataset_name == 'iranian' and split_method != 'row_wise')
        else 'stratified_row_wise'
    )

    dataset_label = profile['label']
    print(f'\n=== {dataset_label} ({run_name}) ===')
    print(f'rows={len(X):,}, features={X.shape[1]}, churn={y.mean():.2%}, split={actual_split}')
    print(f'train={len(X_train):,}, validation={len(X_val):,}, test={len(X_test):,}')

    model_results = {}
    fitted_models = {}
    for model_name in ['Dummy', 'LogReg', 'RandomForest']:
        pipeline = _build_model(
            model_name,
            numeric_features,
            categorical_features,
            is_ibm=is_ibm,
        )
        fitted, best_params, train_time = _fit_model(
            model_name,
            pipeline,
            X_train,
            y_train,
            tune=tune,
        )
        y_pred = fitted.predict(X_test)
        y_score = _scores(fitted, X_test)
        metrics = compute_metrics(y_test, y_pred, y_score)
        model_results[model_name] = {
            'metrics': metrics,
            'best_params': best_params,
            'train_time_s': round(train_time, 3),
        }
        fitted_models[model_name] = fitted

    rf_base = fitted_models['RandomForest']
    rf_deployed = rf_base
    if CALIBRATE_PROBABILITIES:
        rf_deployed = CalibratedClassifierCV(
            estimator=rf_base,
            method=CALIBRATION_METHOD,
            cv=CV_FOLDS,
            n_jobs=SEARCH_N_JOBS,
        )
        rf_deployed.fit(X_train, y_train)

    y_val_proba = rf_deployed.predict_proba(X_val)[:, 1]
    threshold_result = find_cost_optimal_threshold(
        y_val,
        y_val_proba,
        cost_fp=COST_FP,
        cost_fn=COST_FN,
        cost_mode='classification',
    )
    threshold = threshold_result['best_threshold']
    y_test_proba = rf_deployed.predict_proba(X_test)[:, 1]
    y_test_default = (y_test_proba >= 0.5).astype(int)
    y_test_policy = (y_test_proba >= threshold).astype(int)
    calibrated_default = compute_metrics(y_test, y_test_default, y_test_proba)
    policy_metrics = compute_metrics(y_test, y_test_policy, y_test_proba)

    result_dir = RESULTS_DIR / 'dataset_runs' / run_name
    model_dir = MODELS_DIR / 'dataset_runs' / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'rf_calibrated.joblib'
    joblib.dump(rf_deployed, model_path)

    metadata = {
        'dataset_key': dataset_name,
        'dataset_label': profile['label'],
        'run_name': run_name,
        'source_path': str(profile['path']),
        'dataset_sha256': audit['sha256'],
        'rows': audit['rows'],
        'features': audit['features'],
        'churn_rate': audit['churn_rate'],
        'missing_values': audit['missing_values'],
        'feature_columns': list(X.columns),
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'split_method': actual_split,
        'feature_engineering': is_ibm and USE_FEATURE_ENGINEERING,
        'total_charges_policy': (
            'keep_ablation' if keep_total_charges else 'drop'
        ) if dataset_name == 'ibm' else 'not_applicable',
        'split': {
            'train': len(X_train),
            'validation': len(X_val),
            'test': len(X_test),
            'random_state': RANDOM_STATE,
        },
        'calibration': {
            'enabled': CALIBRATE_PROBABILITIES,
            'method': CALIBRATION_METHOD if CALIBRATE_PROBABILITIES else None,
            'cv_folds': CV_FOLDS if CALIBRATE_PROBABILITIES else None,
        },
        'python_version': platform.python_version(),
        'sklearn_version': sklearn.__version__,
    }

    threshold_summary = {
        key: threshold_result[key]
        for key in [
            'cost_mode',
            'cost_fp',
            'cost_fn',
            'best_threshold',
            'best_cost',
            'default_cost',
            'savings',
        ]
    }
    comparison = {
        'dataset': run_name,
        'dataset_label': profile['label'],
        'rows': audit['rows'],
        'features': audit['features'],
        'churn_rate': round(audit['churn_rate'], 6),
        'rf_average_precision': model_results['RandomForest']['metrics']['average_precision'],
        'rf_roc_auc': model_results['RandomForest']['metrics']['roc_auc'],
        'rf_brier_score': calibrated_default['brier_score'],
        'rf_ece': calibrated_default['ece'],
        'rf_f1_default': calibrated_default['f1_churn'],
        'rf_f1_policy': policy_metrics['f1_churn'],
        'rf_recall_policy': policy_metrics['recall_churn'],
        'rf_precision_policy': policy_metrics['precision_churn'],
        'rf_lift_at_10pct': policy_metrics['lift_at_10pct'],
        'selected_threshold': threshold,
        'validation_cost': threshold_result['best_cost'],
        'validation_savings_vs_05': threshold_result['savings'],
        'logreg_average_precision': model_results['LogReg']['metrics']['average_precision'],
        'logreg_roc_auc': model_results['LogReg']['metrics']['roc_auc'],
    }

    _write_json(result_dir / 'model_metrics.json', model_results)
    _write_json(result_dir / 'policy_metrics.json', {
        'validation_threshold': threshold_summary,
        'test_default_threshold': calibrated_default,
        'test_selected_threshold': policy_metrics,
    })
    _write_json(result_dir / 'metadata.json', metadata)
    _write_json(result_dir / 'training_log.json', {
        'status': 'completed',
        'dataset': run_name,
        'tuned': tune,
        'split_method': actual_split,
        'search_iterations': SEARCH_ITERATIONS if tune else 0,
        'model_path': str(model_path),
    })
    pd.DataFrame([comparison]).to_csv(
        result_dir / 'comparison_table.csv',
        index=False,
    )

    rf_ap = comparison['rf_average_precision']
    rf_auc = comparison['rf_roc_auc']
    rf_policy_f1 = comparison['rf_f1_policy']
    print(
        f'RF AP={rf_ap:.4f}, '
        f'ROC-AUC={rf_auc:.4f}, '
        f'policy F1={rf_policy_f1:.4f}, '
        f'threshold={threshold:.4f}'
    )
    return comparison


def save_dataset_comparison_plot(table):
    '''Save a compact multi-dataset metric comparison with exact value labels.'''
    if table.empty:
        return None

    metric_specs = [
        ('rf_average_precision', 'Average Precision', 'higher is better'),
        ('rf_roc_auc', 'ROC-AUC', 'higher is better'),
        ('rf_brier_score', 'Brier Score', 'lower is better'),
        ('rf_ece', 'Calibration ECE', 'lower is better'),
        ('rf_f1_policy', 'Policy F1', 'higher is better'),
        ('rf_lift_at_10pct', 'Lift @ top 10%', 'higher is better'),
    ]
    labels = table['dataset_label'].astype(str).tolist()
    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#B07AA1']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    footer_lines = []
    header = 'Dataset'.ljust(20)
    for short_name, _, _ in metric_specs:
        metric_name = short_name.removeprefix('rf_')[:9]
        header += f' {metric_name:>10}'
    footer_lines.append(header)

    for ax, (column, title, direction) in zip(axes.ravel(), metric_specs):
        values = pd.to_numeric(table[column], errors='coerce').to_numpy()
        plot_values = np.nan_to_num(values, nan=0.0)
        bars = ax.bar(
            range(len(labels)),
            plot_values,
            color=colors[:len(labels)],
            alpha=0.9,
        )
        for bar, value in zip(bars, values):
            label = 'n/a' if not np.isfinite(value) else f'{value:.3f}'
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=9,
            )
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.text(
            0.02,
            0.95,
            direction,
            transform=ax.transAxes,
            fontsize=8,
            color='#555555',
            va='top',
        )
        ax.set_ylabel('Score')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
        upper_bound = max(float(np.nanmax(plot_values)) * 1.28, 0.05)
        ax.set_ylim(0, upper_bound)
        ax.grid(axis='y', alpha=0.25)

    for _, row in table.iterrows():
        line = str(row['dataset_label'])[:20].ljust(20)
        for column, _, _ in metric_specs:
            value = row[column]
            text_value = 'n/a' if pd.isna(value) else f'{float(value):.3f}'
            line += f' {text_value:>10}'
        footer_lines.append(line)

    fig.suptitle(
        'Multi-Dataset Benchmark — Random Forest Evaluation Across Paradigms',
        fontsize=15,
        fontweight='bold',
        y=0.99,
    )
    fig.text(
        0.5,
        0.018,
        '\n'.join(footer_lines)
        + '\nBar labels are exact test metrics. Iranian uses leak-free group-aware split.'
        + '\nThreshold-dependent Policy F1 uses validation-selected business threshold (Cost FP=$50, FN=$500).',
        ha='center',
        va='bottom',
        fontsize=8.5,
        family='monospace',
        color='#333333',
        bbox={
            'boxstyle': 'round,pad=0.45',
            'facecolor': '#f7f7f7',
            'edgecolor': '#cccccc',
        },
    )
    fig.tight_layout(rect=[0.02, 0.16, 0.98, 0.96])

    figure_dir = FIGURES_DIR / 'dataset_runs'
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / 'dataset_comparison_metrics.png'
    fig.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[compare] Saved {figure_path}')
    return figure_path


def update_comparison_table(rows):
    '''Merge current runs into one comparison table without overwriting rows.'''
    comparison_dir = RESULTS_DIR / 'dataset_runs'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = comparison_dir / 'dataset_comparison.csv'
    current = pd.DataFrame(rows)
    if comparison_path.exists():
        previous = pd.read_csv(comparison_path)
        current = pd.concat([previous, current], ignore_index=True)
    current = current.drop_duplicates(subset=['dataset'], keep='last')
    current = current.sort_values('dataset').reset_index(drop=True)
    current.to_csv(comparison_path, index=False)
    _write_json(
        comparison_dir / 'dataset_comparison.json',
        current.to_dict(orient='records'),
    )
    return current


def main():
    parser = argparse.ArgumentParser(
        description='Run IBM, Orange, and Iranian churn datasets independently.'
    )
    parser.add_argument(
        '--dataset',
        choices=['ibm', 'orange', 'iranian', 'both', 'all'],
        default='all',
        help='Dataset to run: "all" runs all three; "both" runs IBM and Iranian.',
    )
    parser.add_argument(
        '--split-method',
        choices=['default', 'group_aware', 'row_wise'],
        default='default',
        help='Split method for Iranian: group_aware (default) or row_wise.',
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Tune the Random Forest with the shared search protocol.',
    )
    parser.add_argument(
        '--keep-total-charges',
        action='store_true',
        help='IBM-only ablation; keep TotalCharges in a separate run folder.',
    )
    args = parser.parse_args()

    if args.dataset == 'all':
        dataset_names = ['ibm', 'orange', 'iranian']
    elif args.dataset == 'both':
        dataset_names = ['ibm', 'iranian']
    else:
        dataset_names = [args.dataset]

    rows = []
    for dataset_name in dataset_names:
        rows.append(
            run_dataset(
                dataset_name,
                tune=args.tune,
                keep_total_charges=(
                    args.keep_total_charges and dataset_name == 'ibm'
                ),
                split_method=args.split_method,
            )
        )

    table = update_comparison_table(rows)
    comparison_figure = save_dataset_comparison_plot(table)
    comparison_dir = RESULTS_DIR / 'dataset_runs'
    print('\n=== DATASET COMPARISON ===')
    print(table.to_string(index=False))
    print(f'\nSaved comparison to {comparison_dir}')
    print(f'Saved comparison figure to {comparison_figure}')


if __name__ == '__main__':
    main()
