"""Calibration of the deployed Random Forest: quality vs model size.

Usage:
    python -m src.calibration_analysis               # IBM (the deployed model)
    python -m src.calibration_analysis --dataset all

Compares, on the main 70/15/15 split (test set):
    - RF uncalibrated (tuned benchmark parameters)
    - RF + sigmoid (Platt), cv=5, ensemble=True   ← 5 forests (the earlier deployment)
    - RF + sigmoid (Platt), cv=5, ensemble=False  ← one forest + one calibrator
      (deployed since this analysis; see CALIBRATION_ENSEMBLE in src.config)
    - RF + isotonic, cv=5, ensemble=True          (for the Platt-vs-isotonic question)
    - every other benchmark model except Dummy (already calibrated where needed)

Reports ROC-AUC, AP, Brier, log-loss and ECE, fit time, and the size of the
serialized model with and without joblib compression. Saves a reliability
diagram per dataset.

Outputs:
    results/calibration/calibration_comparison.csv
    figures/rf/reliability_<dataset>.png
"""

import argparse
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

from src.compare_datasets import (
    BENCHMARK_MODELS,
    _split_data,
    build_benchmark_model,
    load_dataset,
)
from src.config import (
    CALIBRATION_ENSEMBLE, CV_FOLDS, FIGURES_DIR, RESULTS_DIR, SEARCH_N_JOBS,
)
from src.evaluate import compute_metrics

CALIBRATION_DIR = RESULTS_DIR / 'calibration'
RF_FIGURES_DIR = FIGURES_DIR / 'rf'
DATASETS = ['ibm', 'orange', 'iranian']
METRICS = ['roc_auc', 'average_precision', 'brier_score', 'log_loss', 'ece']
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


def candidate_models(X, dataset_name):
    """(label, unfitted estimator) pairs to compare."""
    rf = build_benchmark_model('RandomForest', X, dataset_name)
    candidates = [
        ('RF uncalibrated', rf),
        (ENSEMBLE_VARIANT, CalibratedClassifierCV(
            estimator=clone(rf), method='sigmoid', cv=CV_FOLDS,
            ensemble=True, n_jobs=SEARCH_N_JOBS,
        )),
        (COMPACT_VARIANT, CalibratedClassifierCV(
            estimator=clone(rf), method='sigmoid', cv=CV_FOLDS,
            ensemble=False, n_jobs=SEARCH_N_JOBS,
        )),
        ('RF + isotonic, cv=5, ensemble=True', CalibratedClassifierCV(
            estimator=clone(rf), method='isotonic', cv=CV_FOLDS,
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
    X_train, _, X_test, y_train, _, y_test = _split_data(X, y, dataset_name=dataset_name)
    rows, probabilities = [], {}
    for label, estimator in candidate_models(X, dataset_name):
        start = time.time()
        estimator.fit(X_train, y_train)
        fit_seconds = time.time() - start
        proba = estimator.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, (proba >= 0.5).astype(int), proba)
        size_raw, size_compressed = model_size_mb(estimator)
        row = {'dataset': dataset_name, 'model': label, 'deployed': label == DEPLOYED_VARIANT}
        row.update({key: metrics[key] for key in METRICS})
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
    """Plain-language verdict on ensemble=False vs the deployed calibration."""
    lines = []
    for dataset_name, group in table.groupby('dataset', sort=False):
        indexed = group.set_index('model')
        if ENSEMBLE_VARIANT not in indexed.index or COMPACT_VARIANT not in indexed.index:
            continue
        deployed, compact = indexed.loc[ENSEMBLE_VARIANT], indexed.loc[COMPACT_VARIANT]
        d_ap = compact['average_precision'] - deployed['average_precision']
        d_brier = compact['brier_score'] - deployed['brier_score']
        verdict = (
            'ensemble=False is as good and much smaller'
            if abs(d_ap) <= 0.01 and d_brier <= 0.005
            else 'keep ensemble=True (quality drops with ensemble=False)'
        )
        lines.append(
            f'{dataset_name}: ΔAP={d_ap:+.4f}, ΔBrier={d_brier:+.4f}, '
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
