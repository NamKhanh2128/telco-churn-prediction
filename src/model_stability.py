"""Repeated-split stability of every benchmark model + tests against RF.

Usage:
    python -m src.model_stability                  # all datasets, 10 seeds
    python -m src.model_stability --dataset ibm
    python -m src.model_stability --quick          # 3 seeds (smoke run)

For each dataset and seed the data is split 70/15/15 with that seed
(group-aware for Iranian). Every model in BENCHMARK_MODELS except Dummy is
fitted on the training split with the hyper-parameters tuned in the main
benchmark (results/dataset_runs/<dataset>/model_metrics.json; defaults when
the benchmark was not tuned) and scored on the test split at threshold 0.5.
Models added to BENCHMARK_MODELS later (DT, KNN, SVM) are picked up
automatically.

Statistical tests compare RandomForest with the strongest other model
(highest mean average precision):
    - Wilcoxon signed-rank test on the paired per-seed scores (AP and F1);
    - McNemar exact test on the test predictions of the main seed (42).
Both are two-sided and NOT corrected for multiple comparisons.

Outputs (results/model_stability/):
    stability_runs.csv     one row per dataset/model/seed
    stability_summary.csv  mean, std, min, max per dataset/model/metric
    stat_tests.csv         RF vs strongest other model per dataset
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from src.compare_datasets import (
    BENCHMARK_MODELS,
    _split_data,
    build_benchmark_model,
    load_dataset,
)
from src.config import RANDOM_STATE, RESULTS_DIR
from src.evaluate import compute_metrics

STABILITY_DIR = RESULTS_DIR / 'model_stability'
SEEDS = [13, 42, 73, 101, 2025, 303, 404, 505, 606, 707]  # same as src.iranian_audit
QUICK_SEEDS = [13, 42, 73]
DATASETS = ['ibm', 'orange', 'iranian']
METRICS = [
    'accuracy', 'precision_churn', 'recall_churn', 'f1_churn',
    'roc_auc', 'average_precision',
]
TEST_METRICS = ['average_precision', 'f1_churn']


def run_stability(dataset_name, seeds, models):
    """Fit every model on every seed; return metric rows and main-seed predictions."""
    X, y, _, _ = load_dataset(dataset_name)
    rows = []
    main_predictions = {}
    for seed in seeds:
        X_train, _, X_test, y_train, _, y_test = _split_data(
            X, y, dataset_name=dataset_name, seed=seed,
        )
        for model_name in models:
            pipeline = build_benchmark_model(model_name, X, dataset_name)
            pipeline.fit(X_train, y_train)
            proba = pipeline.predict_proba(X_test)[:, 1]
            prediction = (proba >= 0.5).astype(int)
            metrics = compute_metrics(y_test, prediction, proba)
            row = {'dataset': dataset_name, 'model': model_name, 'seed': seed}
            row.update({key: metrics[key] for key in METRICS})
            rows.append(row)
            if seed == RANDOM_STATE:
                main_predictions[model_name] = (np.asarray(y_test), prediction)
        print(f'[stability] {dataset_name}: seed {seed} done')
    return rows, main_predictions


def summarise(runs):
    """Mean, std, min, max over seeds for every dataset/model/metric."""
    long = runs.melt(
        id_vars=['dataset', 'model', 'seed'], value_vars=METRICS, var_name='metric',
    )
    summary = (
        long.groupby(['dataset', 'model', 'metric'], sort=False)['value']
        .agg(['mean', 'std', 'min', 'max', 'count'])
        .reset_index()
    )
    summary['std'] = summary['std'].fillna(0.0)
    return summary.round(4)


def mcnemar_exact(y_true, pred_a, pred_b):
    """Exact McNemar test on paired predictions; returns (b, c, p_value).

    b = A correct & B wrong, c = A wrong & B correct.
    """
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p_value = 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)
    return b, c, p_value


def paired_wilcoxon(scores_a, scores_b):
    """Two-sided Wilcoxon signed-rank p-value; 1.0 when all pairs are equal."""
    differences = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    if np.allclose(differences, 0.0):
        return None, 1.0
    result = wilcoxon(scores_a, scores_b, zero_method='wilcox', alternative='two-sided')
    return float(result.statistic), float(result.pvalue)


def rf_vs_strongest(runs, main_predictions, dataset_name):
    """Test RandomForest against the other model with the highest mean AP."""
    data = runs[runs['dataset'] == dataset_name]
    others = data[data['model'] != 'RandomForest']
    if others.empty or 'RandomForest' not in set(data['model']):
        return []
    rival = others.groupby('model')['average_precision'].mean().idxmax()
    rf = data[data['model'] == 'RandomForest'].sort_values('seed')
    other = data[data['model'] == rival].sort_values('seed')
    rows = []
    for metric in TEST_METRICS:
        statistic, p_value = paired_wilcoxon(rf[metric].values, other[metric].values)
        rows.append({
            'dataset': dataset_name,
            'test': f'wilcoxon_{metric}',
            'rf_mean': round(float(rf[metric].mean()), 4),
            'rival': rival,
            'rival_mean': round(float(other[metric].mean()), 4),
            'statistic': None if statistic is None else round(statistic, 4),
            'p_value': _sig(p_value),
            'n_pairs': int(len(rf)),
        })
    if 'RandomForest' in main_predictions and rival in main_predictions:
        y_true, rf_pred = main_predictions['RandomForest']
        _, rival_pred = main_predictions[rival]
        b, c, p_value = mcnemar_exact(y_true, rf_pred, rival_pred)
        rows.append({
            'dataset': dataset_name,
            'test': f'mcnemar_accuracy_seed{RANDOM_STATE}',
            'rf_mean': round(float(np.mean(rf_pred == y_true)), 4),
            'rival': rival,
            'rival_mean': round(float(np.mean(rival_pred == y_true)), 4),
            'statistic': f'b={b}, c={c}',
            'p_value': _sig(p_value),
            'n_pairs': int(len(y_true)),
        })
    return rows


def _sig(value):
    """Keep three significant digits so tiny p-values do not round to 0."""
    return float(f'{value:.3g}')


def main():
    parser = argparse.ArgumentParser(description='Repeated-split stability + tests vs RF.')
    parser.add_argument('--dataset', choices=DATASETS + ['all'], default='all')
    parser.add_argument('--quick', action='store_true', help='Use 3 seeds.')
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    seeds = QUICK_SEEDS if args.quick else SEEDS
    models = [name for name in BENCHMARK_MODELS if name != 'Dummy']

    all_rows, test_rows = [], []
    for dataset_name in datasets:
        rows, main_predictions = run_stability(dataset_name, seeds, models)
        all_rows.extend(rows)
        test_rows.extend(
            rf_vs_strongest(pd.DataFrame(rows), main_predictions, dataset_name)
        )

    STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(all_rows)
    runs.to_csv(STABILITY_DIR / 'stability_runs.csv', index=False)
    summary = summarise(runs)
    summary.to_csv(STABILITY_DIR / 'stability_summary.csv', index=False)
    tests = pd.DataFrame(test_rows)
    tests.to_csv(STABILITY_DIR / 'stat_tests.csv', index=False)

    print('\n=== STABILITY (mean ± std over seeds, test set, threshold 0.5) ===')
    view = summary[summary['metric'].isin(['f1_churn', 'roc_auc', 'average_precision'])]
    print(view.to_string(index=False))
    print('\n=== RF vs strongest other model ===')
    print(tests.to_string(index=False))
    print(f'\nSaved {STABILITY_DIR}')


if __name__ == '__main__':
    main()
