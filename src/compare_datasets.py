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
if 'ipykernel' not in sys.modules:  # scripts: no display; notebooks keep inline plots
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from src.config import (
    CALIBRATION_ENSEMBLE,
    CALIBRATION_METHOD,
    CALIBRATE_PROBABILITIES,
    CV_FOLDS,
    COST_FN,
    COST_FP,
    DATA_RAW_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    OFFICIAL_TUNED_PARAMS_PATH,
    PRIMARY_SCORING,
    RANDOM_STATE,
    RESULTS_DIR,
    SEARCH_ITERATIONS,
    SEARCH_N_JOBS,
    TEST_SIZE,
    TUNED_PARAMS_FILENAME,
    USE_FEATURE_ENGINEERING,
    VALIDATION_SIZE,
    DEFAULT_DT_PARAMS,
    DEFAULT_KNN_PARAMS,
    DEFAULT_LR_PARAMS,
    DEFAULT_RF_PARAMS,
    DEFAULT_SVM_PARAMS,
    DEFAULT_SVM_RBF_PARAMS,
    DT_PARAM_GRID,
    KNN_PARAM_GRID,
    RF_PARAM_GRID,
    SVM_PARAM_GRID,
    SVM_RBF_PARAM_GRID,
)
from src.data_loader import detect_feature_types, get_file_sha256
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


# ─── Benchmark framework ─────────────────────────────────────
# Models evaluated on every dataset, in this order. To add a model (e.g. the
# DT/KNN/SVM comparison models):
#   1. add its name here;
#   2. add a branch in _build_model (classifier + whether numeric columns are scaled);
#   3. for --tune, add its grid to MODEL_PARAM_GRIDS (grid itself lives in src.config).
# Everything else picks the new model up automatically: the per-run and long
# comparison tables, src.model_stability (10 seeds + tests vs RF),
# src.calibration_analysis and the snapshot. tests/test_experiments.py checks
# every model listed here (fits, valid probabilities, grid keys match).
BENCHMARK_MODELS = [
    'Dummy',
    'LogReg',
    'DecisionTree',
    'RandomForest',
    'KNN',
    'SVM_Linear',
    'SVM_RBF',
]
# RandomForest (deployed model) and LogReg (comparison columns of
# dataset_comparison.csv) are always run, even with --models.
REQUIRED_MODELS = ('LogReg', 'RandomForest')

# Search spaces used with --tune. Keys may be written for a bare classifier
# ("classifier__C") or for one wrapped in CalibratedClassifierCV
# ("classifier__estimator__C"); _grid_for_pipeline adapts them to the pipeline.
MODEL_PARAM_GRIDS = {
    'LogReg': {'classifier__C': [0.01, 0.1, 1.0, 5.0, 10.0]},
    'DecisionTree': DT_PARAM_GRID,
    'RandomForest': RF_PARAM_GRID,
    'KNN': KNN_PARAM_GRID,
    'SVM_Linear': SVM_PARAM_GRID,
    'SVM_RBF': SVM_RBF_PARAM_GRID,
}

LONG_TABLE_METRICS = [
    'accuracy',
    'precision_churn',
    'recall_churn',
    'f1_churn',
    'f1_macro',
    'balanced_accuracy',
    'roc_auc',
    'average_precision',
    'brier_score',
    'ece',
]


def _grid_for_pipeline(model_name, pipeline):
    '''Return the --tune grid for a model, with keys matching the pipeline.

    CalibratedClassifierCV hides the real estimator one level deeper, so a
    grid written as "classifier__C" must become "classifier__estimator__C"
    (and vice versa). Returns None when the model has no grid.
    '''
    grid = MODEL_PARAM_GRIDS.get(model_name)
    if not grid:
        return None
    classifier = pipeline.named_steps.get('classifier')
    wrapped = isinstance(classifier, CalibratedClassifierCV)
    adapted = {}
    for key, values in grid.items():
        if not key.startswith('classifier__'):
            raise ValueError(f'Grid key {key!r} must start with "classifier__"')
        param = key[len('classifier__'):]
        if param.startswith('estimator__'):
            param = param[len('estimator__'):]
        new_key = (
            f'classifier__estimator__{param}' if wrapped else f'classifier__{param}'
        )
        adapted[new_key] = list(values)
    return adapted


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


def feature_lists(X, dataset_name):
    '''Numeric/categorical columns the pipeline expects for one dataset.

    IBM uses src.data_loader.detect_feature_types — the same order as
    src.train — plus the engineered numeric features that TelcoFeatureEngineer
    adds inside the pipeline. Column order changes which features a seeded
    forest samples, so both scripts must agree to train the same model.
    '''
    if dataset_name == 'ibm':
        return detect_feature_types(X, verbose=False)
    numeric_features = list(X.select_dtypes(include=np.number).columns)
    categorical_features = [
        column for column in X.columns if column not in numeric_features
    ]
    return numeric_features, categorical_features


# ─── Tuned hyper-parameters ──────────────────────────────────
# `--tune` writes the best parameters of each run to models/tuned_params.json
# (tracked in git), as short names: {"ibm": {"RandomForest": {"max_depth": 10}}}.
# ablation / model_stability / calibration_analysis / rf_analysis read them
# through tuned_params(), so every analysis uses the configuration of the main
# benchmark, also on a fresh clone. Untuned runs never overwrite the file.
_MISSING_TUNED_WARNED = set()


def _tuned_params_paths():
    '''Where tuned parameters are looked up: the current outputs, then the official file.'''
    paths = [MODELS_DIR / TUNED_PARAMS_FILENAME]
    if OFFICIAL_TUNED_PARAMS_PATH not in paths:
        paths.append(OFFICIAL_TUNED_PARAMS_PATH)
    return paths


def load_tuned_params():
    '''All tuned parameters: {run_name: {model_name: {param: value}}} ({} if none).

    The official file is read first and the current outputs (a sandbox) are
    laid on top per (run, model), so a sandbox that tuned only one dataset or
    a few models does not hide the official parameters of the others.
    '''
    merged = {}
    for path in reversed(_tuned_params_paths()):
        if path.exists():
            with open(path, encoding='utf-8') as handle:
                for run_name, models in json.load(handle).get('runs', {}).items():
                    merged.setdefault(run_name, {}).update(models)
    return merged


def tuned_params(run_name, model_name):
    '''Best hyper-parameters of the last tuned run (short names), or {}.

    Falls back to the defaults of src.config with a one-time warning when the
    model has a grid but was never tuned on this run.
    '''
    params = (load_tuned_params().get(run_name) or {}).get(model_name)
    if params is None:
        if model_name in MODEL_PARAM_GRIDS and (run_name, model_name) not in _MISSING_TUNED_WARNED:
            _MISSING_TUNED_WARNED.add((run_name, model_name))
            print(
                f'[compare] WARNING: no tuned parameters for {model_name} on {run_name!r} '
                f'in {TUNED_PARAMS_FILENAME}; using the defaults of src.config. '
                f'Run `python -m src.compare_datasets --tune` to tune.'
            )
        return {}
    return dict(params)


def save_tuned_params(run_name, model_results, provenance, partial=False):
    '''Record the best parameters of one tuned run, keeping the other runs.

    partial=False (a run over every benchmark model) replaces the run's entry,
    so a model removed from BENCHMARK_MODELS disappears; partial=True (--models
    subset) only updates the models that were re-tuned.
    '''
    path = MODELS_DIR / TUNED_PARAMS_FILENAME
    payload = {}
    if path.exists():
        with open(path, encoding='utf-8') as handle:
            payload = json.load(handle)
    payload['_note'] = (
        'Written by `python -m src.compare_datasets --tune`; read by the analysis '
        'scripts through src.compare_datasets.tuned_params(). Do not edit by hand.'
    )
    entries = {
        model_name: {
            key.split('__')[-1]: value for key, value in result['best_params'].items()
        }
        for model_name, result in model_results.items()
        if result.get('best_params')
    }
    runs = payload.setdefault('runs', {})
    if partial:
        runs.setdefault(run_name, {}).update(entries)
    else:
        runs[run_name] = entries
    payload.setdefault('provenance', {})[run_name] = provenance
    payload['runs'] = dict(sorted(payload['runs'].items()))
    payload['provenance'] = dict(sorted(payload['provenance'].items()))
    _write_json(path, payload)
    return path


def build_benchmark_model(model_name, X, dataset_name, run_name=None):
    '''Benchmark pipeline for one model, with tuned parameters when available.'''
    numeric_features, categorical_features = feature_lists(X, dataset_name)
    pipeline = _build_model(
        model_name,
        numeric_features,
        categorical_features,
        is_ibm=(dataset_name == 'ibm'),
    )
    params = tuned_params(run_name or dataset_name, model_name)
    grid = _grid_for_pipeline(model_name, pipeline) or {}
    usable = {}
    for key, value in params.items():
        param = key.split('__')[-1]
        for grid_key in grid:
            if grid_key.split('__')[-1] == param:
                usable[grid_key] = value
    if usable:
        pipeline.set_params(**usable)
    return pipeline


def _scores(estimator, X):
    if hasattr(estimator, 'predict_proba'):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, 'decision_function'):
        return estimator.decision_function(X)
    return None


def _calibrated(estimator):
    '''Platt-scale a model whose scores are not probabilities (LogReg, SVM).

    With CALIBRATE_PROBABILITIES the model is wrapped in CalibratedClassifierCV,
    so its grid keys become "classifier__estimator__<param>" (handled by
    _grid_for_pipeline).
    '''
    if not CALIBRATE_PROBABILITIES:
        return estimator
    return CalibratedClassifierCV(estimator, method=CALIBRATION_METHOD, cv=CV_FOLDS)


def _build_model(name, numeric_features, categorical_features, is_ibm=False):
    '''Unfitted pipeline [feature_engineer (IBM)] -> preprocessor -> classifier.

    Each branch sets the classifier and whether numeric columns are scaled
    (True for distance-based and linear models, False for trees).
    '''
    if name == 'RandomForest':
        classifier = RandomForestClassifier(**DEFAULT_RF_PARAMS)
        scale_numeric = False
    elif name == 'LogReg':
        classifier = _calibrated(LogisticRegression(**DEFAULT_LR_PARAMS))
        scale_numeric = True
    elif name == 'DecisionTree':
        classifier = DecisionTreeClassifier(**DEFAULT_DT_PARAMS)
        scale_numeric = False
    elif name == 'KNN':
        classifier = KNeighborsClassifier(**DEFAULT_KNN_PARAMS)
        scale_numeric = True
    elif name == 'SVM_Linear':
        classifier = _calibrated(LinearSVC(**DEFAULT_SVM_PARAMS))
        scale_numeric = True
    elif name == 'SVM_RBF':
        classifier = _calibrated(SVC(**DEFAULT_SVM_RBF_PARAMS))
        scale_numeric = True
    elif name == 'Dummy':
        classifier = DummyClassifier(strategy='most_frequent')
        scale_numeric = False
    else:
        raise ValueError(
            f'Unknown model {name!r}: add a branch for it in _build_model '
            f'(and a grid in MODEL_PARAM_GRIDS if it should be tuned).'
        )

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=scale_numeric,
    )
    feature_engineer = (
        TelcoFeatureEngineer(enabled=USE_FEATURE_ENGINEERING) if is_ibm else None
    )
    return build_model_pipeline(
        preprocessor,
        classifier,
        feature_engineer=feature_engineer,
    )


def feature_groups(X):
    '''Group id per row: identical feature vectors share a group.'''
    return pd.util.hash_pandas_object(X, index=False).astype(str).to_numpy()


def cv_splitter(groups=None):
    '''CV used for tuning and analysis.

    With groups (Iranian, group-aware protocol) duplicate feature vectors stay
    in the same fold, so cross-validation scores are not inflated by copies.
    '''
    if groups is not None:
        return StratifiedGroupKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE,
        )
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def calibration_cv(X_train, y_train, groups=None):
    '''CV for CalibratedClassifierCV: group-aware folds when groups are given.

    On Iranian a row-wise CV would put copies of one customer on both sides of
    a fold, so the sigmoid would be fitted on optimistic predictions.
    '''
    if groups is None:
        return CV_FOLDS
    return list(cv_splitter(groups).split(X_train, y_train, groups))


def make_deployed_model(rf_pipeline, X_train, y_train, groups=None):
    '''Unfitted deployed model: the RF pipeline + the calibration of src.config.

    The RF is cloned first. CalibratedClassifierCV keeps the estimator it is
    given in `.estimator`; passing an already fitted forest would store a
    second, unused copy in the saved model (twice the file size).
    '''
    if not CALIBRATE_PROBABILITIES:
        return clone(rf_pipeline)
    return CalibratedClassifierCV(
        estimator=clone(rf_pipeline),
        method=CALIBRATION_METHOD,
        cv=calibration_cv(X_train, y_train, groups),
        ensemble=CALIBRATION_ENSEMBLE,
        n_jobs=SEARCH_N_JOBS,
    )


def _fit_model(name, pipeline, X_train, y_train, tune=False, groups=None):
    '''Fit one benchmark model; with tune=True search its grid (if it has one).'''
    start = time.time()
    best_params = None
    grid = _grid_for_pipeline(name, pipeline) if tune else None

    if grid:
        cv = cv_splitter(groups)
        n_iter = min(SEARCH_ITERATIONS, len(list(ParameterGrid(grid))))
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=grid,
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
        search.fit(X_train, y_train, groups=groups)
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


def is_group_aware(dataset_name, split_method='default'):
    '''Iranian is split by duplicate groups unless row_wise is asked for.'''
    return dataset_name == 'iranian' and split_method != 'row_wise'


def _split_data(X, y, dataset_name='ibm', split_method='default', seed=RANDOM_STATE):
    '''Split dataset into 70/15/15 train/val/test splits (seed-controlled).'''
    if is_group_aware(dataset_name, split_method):
        # Group-aware split to prevent identical feature vectors crossing boundaries
        groups = feature_groups(X)
        train_idx, val_idx, test_idx = _group_split_indices(X, y, groups, seed=seed)
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
        random_state=seed,
        stratify=y,
    )
    validation_fraction = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_dev,
        y_train_dev,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y_train_dev,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_name_for(dataset_name, keep_total_charges=False, split_method='default'):
    '''Folder name of one run; variants never overwrite the main run.'''
    if dataset_name == 'ibm' and keep_total_charges:
        return 'ibm_with_total_charges'
    if dataset_name == 'iranian' and split_method == 'row_wise':
        return 'iranian_row_wise'
    return dataset_name


def _run_label(dataset_name, run_name):
    label = DATASET_PROFILES[dataset_name]['label']
    suffix = {
        'ibm_with_total_charges': ' (A1: +TotalCharges)',
        'iranian_row_wise': ' (row-wise split)',
    }.get(run_name, '')
    return label + suffix


def run_dataset(dataset_name, tune=False, keep_total_charges=False, split_method='default',
                models=None):
    '''Train one dataset and write artifacts under a dataset-specific folder.

    models: optional subset of BENCHMARK_MODELS; RandomForest and LogReg are
    always included (deployed model and comparison columns).
    '''
    run_name = run_name_for(dataset_name, keep_total_charges, split_method)
    model_names = [
        name for name in BENCHMARK_MODELS
        if models is None or name in models or name in REQUIRED_MODELS
    ]

    X, y, profile, audit = load_dataset(
        dataset_name,
        keep_total_charges=keep_total_charges,
    )

    numeric_features, categorical_features = feature_lists(X, dataset_name)
    is_ibm = (dataset_name == 'ibm')

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        X, y, dataset_name=dataset_name, split_method=split_method,
    )

    group_aware = is_group_aware(dataset_name, split_method)
    actual_split = 'group_aware' if group_aware else 'stratified_row_wise'
    # Group-aware protocol: tuning and calibration CV keep duplicates in one fold.
    train_groups = feature_groups(X_train) if group_aware else None

    dataset_label = _run_label(dataset_name, run_name)
    print(f'\n=== {dataset_label} ({run_name}) ===')
    print(f'rows={len(X):,}, features={X.shape[1]}, churn={y.mean():.2%}, split={actual_split}')
    print(f'train={len(X_train):,}, validation={len(X_val):,}, test={len(X_test):,}')

    model_results = {}
    fitted_models = {}
    for model_name in model_names:
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
            groups=train_groups,
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
    rf_deployed = make_deployed_model(rf_base, X_train, y_train, groups=train_groups)
    rf_deployed.fit(X_train, y_train)

    if tune:
        tuned_path = save_tuned_params(run_name, model_results, {
            'dataset_sha256': audit['sha256'],
            'split': actual_split,
            'split_seed': RANDOM_STATE,
            'train_rows': int(len(X_train)),
            'cv': f'{type(cv_splitter(train_groups)).__name__}(n_splits={CV_FOLDS})',
            'scoring': PRIMARY_SCORING,
            'search_iterations': SEARCH_ITERATIONS,
            'sklearn_version': sklearn.__version__,
        }, partial=models is not None)
        print(f'[compare] Saved tuned parameters of {run_name!r} to {tuned_path}')

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
        'dataset_label': dataset_label,
        'run_name': run_name,
        'tuned': tune,
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
            'cv': (
                ('group_aware' if group_aware else 'stratified')
                if CALIBRATE_PROBABILITIES else None
            ),
            'ensemble': CALIBRATION_ENSEMBLE if CALIBRATE_PROBABILITIES else None,
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
        'dataset_label': dataset_label,
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
    pd.DataFrame(_model_rows(run_name, dataset_label, model_results)).to_csv(
        result_dir / 'model_comparison.csv',
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
        + '\nThreshold-dependent Policy F1 uses the validation-selected cost threshold '
        # "$" starts matplotlib mathtext, so it must be escaped.
        + f'(placeholder costs FP=\\${COST_FP}, FN=\\${COST_FN}).',
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


def _model_rows(run_name, dataset_label, model_results):
    '''One row per benchmark model with shared test metrics (threshold 0.5).'''
    rows = []
    for model_name, result in model_results.items():
        metrics = result['metrics']
        row = {
            'dataset': run_name,
            'dataset_label': dataset_label,
            'model': model_name,
        }
        row.update({key: metrics.get(key) for key in LONG_TABLE_METRICS})
        row['train_time_s'] = result.get('train_time_s')
        best_params = result.get('best_params')
        row['best_params'] = (
            json.dumps(best_params, sort_keys=True, default=str) if best_params else ''
        )
        rows.append(row)
    return rows


def update_long_table(new_rows, replace='dataset'):
    '''Merge per-model rows into results/dataset_runs/model_comparison_long.csv.

    replace='dataset' (default): rows of datasets that were just re-run replace
    their previous rows entirely, so a model removed from BENCHMARK_MODELS also
    disappears. replace='pair': only the re-run (dataset, model) pairs are
    replaced — used with --models, which re-runs a subset of the models.
    '''
    path = RESULTS_DIR / 'dataset_runs' / 'model_comparison_long.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    current = pd.DataFrame(new_rows)
    if path.exists():
        previous = pd.read_csv(path)
        if replace == 'pair':
            rerun = set(zip(current['dataset'], current['model']))
            keep = [pair not in rerun for pair in zip(previous['dataset'], previous['model'])]
            previous = previous[keep]
        else:
            previous = previous[~previous['dataset'].isin(current['dataset'].unique())]
        current = pd.concat([previous, current], ignore_index=True)
    order = {name: index for index, name in enumerate(BENCHMARK_MODELS)}
    current['_order'] = current['model'].map(order).fillna(len(order))
    current = (
        current.sort_values(['dataset', '_order', 'model'])
        .drop(columns='_order')
        .reset_index(drop=True)
    )
    current.to_csv(path, index=False)
    return current


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


# ─── Public helpers for the other analyses ───────────────────
# Stable names for notebooks and the other members' scripts (e.g. the policy
# analysis): import these instead of the underscored functions above.
split_data = _split_data
build_model = _build_model
fit_model = _fit_model


def fit_deployed_rf(dataset_name='ibm', seed=RANDOM_STATE, split_method='default'):
    '''Fit the deployed model (tuned RF + configured calibration) on one split.

    For analyses that need the production model on one or several seeds, such
    as the cost/threshold policy study. Choose thresholds on the validation
    split and report on the test split. Returns a dict with the fitted
    ``model``, the six frames ``X_train, X_val, X_test, y_train, y_val,
    y_test`` and the ``run_name`` whose tuned parameters were used.
    '''
    X, y, _, _ = load_dataset(dataset_name)
    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        X, y, dataset_name=dataset_name, split_method=split_method, seed=seed,
    )
    groups = feature_groups(X_train) if is_group_aware(dataset_name, split_method) else None
    run_name = run_name_for(dataset_name, split_method=split_method)
    rf_pipeline = build_benchmark_model('RandomForest', X, dataset_name, run_name=run_name)
    model = make_deployed_model(rf_pipeline, X_train, y_train, groups=groups)
    model.fit(X_train, y_train)
    return {
        'model': model,
        'run_name': run_name,
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
    }


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
        help='Tune every model that has a grid in MODEL_PARAM_GRIDS.',
    )
    parser.add_argument(
        '--keep-total-charges',
        action='store_true',
        help='IBM-only ablation; keep TotalCharges in a separate run folder.',
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=BENCHMARK_MODELS,
        help='Run only these models (RandomForest and LogReg are always run).',
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
                models=args.models,
            )
        )

    table = update_comparison_table(rows)
    comparison_figure = save_dataset_comparison_plot(table)
    comparison_dir = RESULTS_DIR / 'dataset_runs'

    long_rows = []
    for row in rows:
        per_run = comparison_dir / row['dataset'] / 'model_comparison.csv'
        long_rows.extend(pd.read_csv(per_run).to_dict(orient='records'))
    long_table = update_long_table(long_rows, replace='pair' if args.models else 'dataset')

    print('\n=== DATASET COMPARISON ===')
    print(table.to_string(index=False))
    print('\n=== ALL MODELS (test set, threshold 0.5) ===')
    print(
        long_table[
            ['dataset', 'model', 'accuracy', 'precision_churn', 'recall_churn',
             'f1_churn', 'roc_auc', 'average_precision']
        ].to_string(index=False)
    )
    print(f'\nSaved comparison to {comparison_dir}')
    print(f'Saved comparison figure to {comparison_figure}')


if __name__ == '__main__':
    main()
