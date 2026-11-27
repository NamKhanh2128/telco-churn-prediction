"""Calibration of the deployed Random Forest: quality vs model size.

Usage:
    python -m src.calibration_analysis               # IBM (the deployed model)
    python -m src.calibration_analysis --dataset all

Compares, on the main 70/15/15 split:
    - RF uncalibrated (tuned benchmark parameters)
    - RF + sigmoid (Platt), cv=5, ensemble=True   ← 5 forests (the earlier deployment)
    - RF + sigmoid (Platt), cv=5, ensemble=False  ← one forest + one calibrator
      (deployed since this analysis; see CALIBRATION_ENSEMBLE in src.config)
    - RF + isotonic, cv=5, ensemble=True          (for the Platt-vs-isotonic question)
    - every other benchmark model except Dummy (already calibrated where needed)

Reports ROC-AUC, AP, Brier, log-loss and ECE on the test split, the mean
predicted probability next to the test churn rate, AP / Brier / ECE on the
validation split (val_* columns: the deployment choice is read from these,
never from test), fit time, and the size of the serialized model with and
without joblib compression. The calibration CV of the RF variants is
group-aware on Iranian (duplicates stay in one fold); LogReg keeps its built-in
row-wise cv=5. Saves a reliability diagram per dataset (test split).

Outputs:
    results/calibration/calibration_comparison.csv
    figures/rf/reliability_<dataset>.png
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib
if 'ipykernel' not in sys.modules:  # scripts: no display; notebooks keep inline plots
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

from src.compare_datasets import (
    BENCHMARK_MODELS,
    _split_data,
    build_benchmark_model,
    calibration_cv,
    feature_groups,
    is_group_aware,
    load_dataset,
)
from src.config import (
    CALIBRATION_ENSEMBLE, FIGURES_DIR, RESULTS_DIR, SEARCH_N_JOBS,
)
from src.evaluate import compute_metrics

CALIBRATION_DIR = RESULTS_DIR / 'calibration'
RF_FIGURES_DIR = FIGURES_DIR / 'rf'
DATASETS = ['ibm', 'orange', 'iranian']
METRICS = ['roc_auc', 'average_precision', 'brier_score', 'log_loss', 'ece']
# Validation metrics that drive the deployment choice (prefixed "val_").
DECISION_METRICS = ['average_precision', 'brier_score', 'ece']
ENSEMBLE_VARIANT = 'RF + sigmoid, cv=5, ensemble=True'
COMPACT_VARIANT = 'RF + sigmoid, cv=5, ensemble=False'
DEPLOYED_VARIANT = COMPACT_VARIANT if not CALIBRATION_ENSEMBLE else ENSEMBLE_VARIANT


def model_size_mb(model):
    """Serialized size in MB without and with joblib compression (level 3)."""
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / 'raw.joblib'
        compressed_path = Path(tmp) / 'compressed.joblib'
        joblib.dump(model, raw_path)
        joblib.dump(model, compressed_path, compress=3)
        return (
            round(raw_path.stat().st_size / 1e6, 2),
            round(compressed_path.stat().st_size / 1e6, 2),
        )


def candidate_models(X, dataset_name, cv):
    """(label, unfitted estimator) pairs to compare.

    cv: calibration folds, i.e. CV_FOLDS or a list of group-aware splits on
    Iranian (see src.compare_datasets.calibration_cv).
    """
    rf = build_benchmark_model('RandomForest', X, dataset_name)
    candidates = [
        ('RF uncalibrated', rf),
        (ENSEMBLE_VARIANT, CalibratedClassifierCV(
            estimator=clone(rf), method='sigmoid', cv=cv,
            ensemble=True, n_jobs=SEARCH_N_JOBS,
        )),
        (COMPACT_VARIANT, CalibratedClassifierCV(
            estimator=clone(rf), method='sigmoid', cv=cv,
            ensemble=False, n_jobs=SEARCH_N_JOBS,
        )),
        ('RF + isotonic, cv=5, ensemble=True', CalibratedClassifierCV(
            estimator=clone(rf), method='isotonic', cv=cv,
            ensemble=True, n_jobs=SEARCH_N_JOBS,
        )),
    ]
    for model_name in BENCHMARK_MODELS:
        if model_name in ('Dummy', 'RandomForest'):
            continue
        candidates.append((model_name, build_benchmark_model(model_name, X, dataset_name)))
    return candidates


def analyse_dataset(dataset_name):
    """Fit every candidate on one dataset; return rows and test probabilities."""
    X, y, _, _ = load_dataset(dataset_name)
    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        X, y, dataset_name=dataset_name,
    )
    groups = feature_groups(X_train) if is_group_aware(dataset_name) else None
    cv = calibration_cv(X_train, y_train, groups)
    rows, probabilities = [], {}
    for label, estimator in candidate_models(X, dataset_name, cv):
        start = time.time()
        estimator.fit(X_train, y_train)
        fit_seconds = time.time() - start
        proba = estimator.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, (proba >= 0.5).astype(int), proba)
        val_proba = estimator.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, (val_proba >= 0.5).astype(int), val_proba)
        size_raw, size_compressed = model_size_mb(estimator)
        row = {'dataset': dataset_name, 'model': label, 'deployed': label == DEPLOYED_VARIANT}
        row.update({key: metrics[key] for key in METRICS})
        # Mean predicted churn probability vs the real test churn rate: shows the
        # direction of the miscalibration (class_weight pushes it upwards).
        row['mean_proba'] = round(float(proba.mean()), 4)
        row['test_churn_rate'] = round(float(y_test.mean()), 4)
        row.update({f'val_{key}': val_metrics[key] for key in DECISION_METRICS})
        row.update({
            'fit_seconds': round(fit_seconds, 2),
            'size_mb': size_raw,
            'size_mb_compressed': size_compressed,
        })
        rows.append(row)
        probabilities[label] = proba
        print(f'[calibration] {dataset_name}: {label} done')
    save_reliability_diagram(dataset_name, y_test, probabilities)
    return rows


def save_reliability_diagram(dataset_name, y_test, probabilities):
    """Reliability diagram of every candidate on one dataset."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for label, proba in probabilities.items():
        CalibrationDisplay.from_predictions(
            y_test, proba, n_bins=10, strategy='uniform', name=label, ax=ax,
        )
    ax.set_title(f'Reliability diagram — {dataset_name} (test set)', fontweight='bold')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of churners')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(alpha=0.25)
    RF_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = RF_FIGURES_DIR / f'reliability_{dataset_name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[calibration] Saved {path}')


def recommendation(table):
    """Plain-language verdict on ensemble=False vs ensemble=True.

    Read from the validation metrics (val_*), so the choice of the deployed
    calibration never looks at the test split.
    """
    prefix = 'val_' if 'val_average_precision' in table else ''
    lines = []
    for dataset_name, group in table.groupby('dataset', sort=False):
        indexed = group.set_index('model')
        if ENSEMBLE_VARIANT not in indexed.index or COMPACT_VARIANT not in indexed.index:
            continue
        deployed, compact = indexed.loc[ENSEMBLE_VARIANT], indexed.loc[COMPACT_VARIANT]
        d_ap = compact[f'{prefix}average_precision'] - deployed[f'{prefix}average_precision']
        d_brier = compact[f'{prefix}brier_score'] - deployed[f'{prefix}brier_score']
        verdict = (
            'ensemble=False is as good and much smaller'
            if abs(d_ap) <= 0.01 and d_brier <= 0.005
            else 'keep ensemble=True (quality drops with ensemble=False)'
        )
        lines.append(
            f'{dataset_name} (validation): dAP={d_ap:+.4f}, dBrier={d_brier:+.4f}, '
            f'size {deployed["size_mb"]} MB → {compact["size_mb"]} MB '
            f'({compact["size_mb_compressed"]} MB compressed) → {verdict}'
        )
    return lines


def main():
    parser = argparse.ArgumentParser(description='Calibration quality vs model size.')
    parser.add_argument('--dataset', choices=DATASETS + ['all'], default='ibm')
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    rows = []
    for dataset_name in datasets:
        rows.extend(analyse_dataset(dataset_name))

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(CALIBRATION_DIR / 'calibration_comparison.csv', index=False)
    print('\n=== CALIBRATION (test set) ===')
    print(table.to_string(index=False))
    print('\n=== RECOMMENDATION ===')
    for line in recommendation(table):
        print(line)
    print(f'\nSaved {CALIBRATION_DIR}')


if __name__ == '__main__':
    main()
