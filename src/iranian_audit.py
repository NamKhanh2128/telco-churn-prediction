'''Audit Iranian Churn leakage risk and measure split stability.

The ordinary comparison uses a row-wise stratified split.  This module adds
two checks that are important for the Iranian dataset:

1. Audit exact and feature-level duplicates, target associations, and timing
   assumptions.
2. Compare repeated row-wise stratified splits with group-aware splits where
   identical feature vectors cannot cross train/validation/test boundaries.

The stability run uses the Random Forest parameters tuned on the Iranian run
(models/tuned_params.json, read through src.compare_datasets.tuned_params;
FIXED_RF_PARAMS fill anything not tuned) but does not re-tune or calibrate on
every seed.  It is a split-stability and leakage-sensitivity check, not the
final deployment benchmark.
'''

import argparse
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.compare_datasets import tuned_params
from src.config import (
    COST_FN,
    COST_FP,
    DATA_RAW_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    SEARCH_N_JOBS,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from src.evaluate import compute_metrics
from src.optimize_threshold import find_cost_optimal_threshold


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


DATA_PATH = DATA_RAW_DIR / 'iranian-uci' / 'Customer Churn.csv'
AUDIT_DIR = RESULTS_DIR / 'dataset_runs' / 'iranian'
# Through src.config, so CHURN_OUTPUT_ROOT (quick runs) also redirects this figure.
FIGURE_DIR = FIGURES_DIR / 'dataset_runs'
SOURCE_URL = (
    'https://archive.ics.uci.edu/dataset/563/'
    'iranian%2Bchurn%2Bdataset'
)

DEFAULT_SEEDS = [13, 42, 73, 101, 2025, 303, 404, 505, 606, 707]


def load_iranian():
    '''Load the approved Iranian file and return feature matrix and target.'''
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Missing Iranian dataset: {DATA_PATH}')
    frame = pd.read_csv(DATA_PATH)
    if 'Churn' not in frame.columns:
        raise ValueError('Iranian dataset must contain a Churn column')
    frame.columns = (
        frame.columns.astype(str)
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
    )
    X = frame.drop(columns=['Churn'])
    y = frame['Churn'].astype(int)
    return frame, X, y


def feature_groups(X):
    '''Create a stable group id for identical feature vectors.'''
    return pd.util.hash_pandas_object(X, index=False).astype(str).to_numpy()


def _random_split_indices(X, y, seed):
    '''Match the project row-wise 70/15/15 split protocol.'''
    indices = np.arange(len(X))
    train_dev, test = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y,
    )
    validation_fraction = VALIDATION_SIZE / (1 - TEST_SIZE)
    train, validation = train_test_split(
        train_dev,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y.iloc[train_dev],
    )
    return np.asarray(train), np.asarray(validation), np.asarray(test)


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


def _group_split_indices(X, y, groups, seed):
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


def _split_overlap(groups, train, validation, test):
    '''Count feature-vector groups shared by nominally separate splits.'''
    train_groups = set(groups[train])
    validation_groups = set(groups[validation])
    test_groups = set(groups[test])
    return {
        'test_rows_matching_train': int(
            np.isin(groups[test], list(train_groups)).sum()
        ),
        'test_rows_matching_validation': int(
            np.isin(groups[test], list(validation_groups)).sum()
        ),
        'validation_rows_matching_train': int(
            np.isin(groups[validation], list(train_groups)).sum()
        ),
        'test_groups_matching_train': int(len(test_groups & train_groups)),
        'test_groups_matching_validation': int(
            len(test_groups & validation_groups)
        ),
    }


def audit_dataset(frame, X, y, groups):
    '''Run deterministic duplicate, target, and feature-association checks.'''
    exact_duplicate_rows = int(frame.duplicated().sum())
    grouped = (
        pd.DataFrame({'group': groups, 'target': y.to_numpy()})
        .groupby('group')['target']
        .agg(['size', 'nunique'])
    )
    feature_duplicate_rows = int(len(frame) - len(grouped))
    duplicate_groups = int((grouped['size'] > 1).sum())
    conflicting_groups = grouped[grouped['nunique'] > 1]
    conflicting_rows = int(conflicting_groups['size'].sum())

    direct_equal = [
        column for column in X.columns
        if np.array_equal(X[column].to_numpy(), y.to_numpy())
    ]
    direct_complement = [
        column for column in X.columns
        if np.array_equal(X[column].to_numpy(), 1 - y.to_numpy())
    ]
    constant_features = [
        column for column in X.columns if X[column].nunique(dropna=False) <= 1
    ]

    discrete_mask = np.array(
        [X[column].nunique() <= 12 for column in X.columns],
        dtype=bool,
    )
    mutual_information = mutual_info_classif(
        X,
        y,
        discrete_features=discrete_mask,
        random_state=42,
    )

    feature_rows = []
    review_names = {'Complains', 'Status', 'Customer Value'}
    for index, column in enumerate(X.columns):
        correlation = float(X[column].corr(y))
        grouped_rates = (
            pd.DataFrame({'value': X[column], 'target': y})
            .groupby('value')['target']
            .agg(['mean', 'count'])
        )
        if len(grouped_rates) <= 20:
            rate_min = float(grouped_rates['mean'].min())
            rate_max = float(grouped_rates['mean'].max())
        else:
            rate_min = None
            rate_max = None
        flags = []
        if column in review_names:
            flags.append('semantic_review')
        if abs(correlation) >= 0.45:
            flags.append('strong_univariate_association')
        feature_rows.append(
            {
                'feature': column,
                'n_unique': int(X[column].nunique()),
                'correlation_with_churn': round(correlation, 4),
                'absolute_correlation': round(abs(correlation), 4),
                'mutual_information': round(float(mutual_information[index]), 4),
                'target_rate_min_by_value': rate_min,
                'target_rate_max_by_value': rate_max,
                'review_flags': '|'.join(flags),
            }
        )

    train, validation, test = _random_split_indices(X, y, seed=42)
    overlap = _split_overlap(groups, train, validation, test)

    report = {
        'dataset': 'iranian',
        'source_url': SOURCE_URL,
        'rows': int(len(frame)),
        'features': int(X.shape[1]),
        'missing_values': int(frame.isna().sum().sum()),
        'churn_rate': float(y.mean()),
        'exact_duplicate_rows': exact_duplicate_rows,
        'feature_duplicate_rows': feature_duplicate_rows,
        'duplicate_feature_groups': duplicate_groups,
        'conflicting_duplicate_feature_groups': int(len(conflicting_groups)),
        'rows_in_conflicting_duplicate_groups': conflicting_rows,
        'largest_duplicate_feature_group': int(grouped['size'].max()),
        'direct_target_equal_features': direct_equal,
        'direct_target_complement_features': direct_complement,
        'constant_features': constant_features,
        'rowwise_seed_42_feature_overlap': overlap,
        'temporal_assumption': (
            'UCI documentation states that features are aggregates from '
            'months 1-9 and Churn is observed at month 12, with a 3-month '
            'planning gap. The CSV has no row-level timestamps, so this '
            'timing cannot be independently verified per customer.'
        ),
        'manual_review_required': [
            'Confirm Status and Customer Value were available at the scoring date.',
            'Confirm Complains is an input known before the intervention window.',
            'Resolve or group-split duplicate feature vectors before final benchmarking.',
            'Confirm that no post-month-9 data was merged into the downloaded file.',
        ],
        'feature_associations': feature_rows,
        'conclusion': (
            'No direct target copy/complement was found. Feature-level '
            'duplicates cross the row-wise split, so the original high score '
            'is potentially optimistic until group-aware evaluation is run.'
        ),
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_DIR / 'audit_report.json', 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, default=str)
    pd.DataFrame(feature_rows).sort_values(
        'absolute_correlation',
        ascending=False,
    ).to_csv(AUDIT_DIR / 'feature_audit.csv', index=False)
    return report


# Used only for parameters the Iranian benchmark has not tuned (e.g. on a fresh
# clone before the first `--tune` run).
FIXED_RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'log2',
}


def rf_params():
    '''RF parameters of the stability check: tuned Iranian values over FIXED_RF_PARAMS.'''
    params = dict(FIXED_RF_PARAMS)
    params.update(tuned_params('iranian', 'RandomForest'))
    return params


def _build_model(seed):
    '''Build the tuned RF used for the stability sensitivity check.'''
    classifier = RandomForestClassifier(**{
        'class_weight': 'balanced_subsample',
        **rf_params(),
        'random_state': seed,
        'n_jobs': SEARCH_N_JOBS,
    })
    return Pipeline(
        [
            ('imputer', SimpleImputer(strategy='median')),
            ('classifier', classifier),
        ]
    )


def _evaluate_split(X, y, train, validation, test, protocol, seed, groups):
    '''Fit one fixed model and evaluate default and validation policy scores.'''
    model = _build_model(seed)
    model.fit(X.iloc[train], y.iloc[train])
    validation_score = model.predict_proba(X.iloc[validation])[:, 1]
    test_score = model.predict_proba(X.iloc[test])[:, 1]

    threshold_result = find_cost_optimal_threshold(
        y.iloc[validation],
        validation_score,
        cost_fp=COST_FP,
        cost_fn=COST_FN,
        cost_mode='classification',
    )
    threshold = threshold_result['best_threshold']
    default_metrics = compute_metrics(
        y.iloc[test],
        (test_score >= 0.5).astype(int),
        test_score,
    )
    policy_metrics = compute_metrics(
        y.iloc[test],
        (test_score >= threshold).astype(int),
        test_score,
    )
    overlap = _split_overlap(groups, train, validation, test)
    row = {
        'protocol': protocol,
        'seed': seed,
        'train_rows': len(train),
        'validation_rows': len(validation),
        'test_rows': len(test),
        'churn_rate_test': round(float(y.iloc[test].mean()), 6),
        'duplicate_test_rows_matching_train': (
            overlap['test_rows_matching_train']
        ),
        'rf_average_precision': default_metrics['average_precision'],
        'rf_roc_auc': default_metrics['roc_auc'],
        'rf_brier_score': default_metrics['brier_score'],
        'rf_ece': default_metrics['ece'],
        'rf_f1_default': default_metrics['f1_churn'],
        'rf_f1_policy': policy_metrics['f1_churn'],
        'rf_recall_policy': policy_metrics['recall_churn'],
        'rf_precision_policy': policy_metrics['precision_churn'],
        'selected_threshold': threshold,
        'validation_cost': threshold_result['best_cost'],
        'validation_savings_vs_05': threshold_result['savings'],
    }
    return row


def run_stability(X, y, seeds):
    '''Run row-wise and duplicate-group-aware repeated evaluations.'''
    groups = feature_groups(X)
    rows = []
    for seed in seeds:
        train, validation, test = _random_split_indices(X, y, seed)
        rows.append(
            _evaluate_split(
                X, y, train, validation, test, 'rowwise', seed, groups
            )
        )
        train, validation, test = _group_split_indices(X, y, groups, seed)
        rows.append(
            _evaluate_split(
                X, y, train, validation, test, 'group_aware', seed, groups
            )
        )

    run_frame = pd.DataFrame(rows)
    metric_columns = [
        'rf_average_precision',
        'rf_roc_auc',
        'rf_brier_score',
        'rf_ece',
        'rf_f1_default',
        'rf_f1_policy',
        'rf_recall_policy',
        'rf_precision_policy',
        'selected_threshold',
        'validation_cost',
        'validation_savings_vs_05',
    ]
    summary_rows = []
    for protocol, group in run_frame.groupby('protocol'):
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors='coerce')
            summary_rows.append(
                {
                    'protocol': protocol,
                    'metric': metric,
                    'mean': round(float(values.mean()), 4),
                    'std': round(float(values.std(ddof=1)), 4),
                    'min': round(float(values.min()), 4),
                    'max': round(float(values.max()), 4),
                }
            )
    summary_frame = pd.DataFrame(summary_rows)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    run_frame.to_csv(AUDIT_DIR / 'stability_runs.csv', index=False)
    summary_frame.to_csv(AUDIT_DIR / 'stability_summary.csv', index=False)
    payload = {
        'seeds': list(seeds),
        'model_protocol': (
            'Fixed tuned Random Forest configuration; no re-tuning and no '
            'calibration inside repeated runs. Threshold is selected on the '
            'validation split only.'
        ),
        'cost_assumptions': {
            'cost_fp': COST_FP,
            'cost_fn': COST_FN,
            'cost_mode': 'classification',
        },
        'runs': run_frame.to_dict(orient='records'),
        'summary': summary_frame.to_dict(orient='records'),
    }
    with open(
        AUDIT_DIR / 'stability_report.json',
        'w',
        encoding='utf-8',
    ) as handle:
        json.dump(payload, handle, indent=2, default=str)
    return run_frame, summary_frame


def save_stability_plot(run_frame):
    '''Plot seed stability with readable mean +/- standard deviation labels.'''
    metric_specs = [
        ('rf_average_precision', 'Average Precision', 'higher is better', 'score'),
        ('rf_roc_auc', 'ROC-AUC', 'higher is better', 'score'),
        ('rf_f1_policy', 'Policy F1', 'higher is better', 'score'),
        ('rf_recall_policy', 'Policy Recall', 'higher is better', 'score'),
        ('rf_precision_policy', 'Policy Precision', 'higher is better', 'score'),
        ('selected_threshold', 'Selected Threshold', 'validation-selected', 'score'),
        ('validation_cost', 'Validation Cost', 'placeholder-cost scale', 'money'),
        ('validation_savings_vs_05', 'Savings vs 0.5', 'placeholder-cost scale', 'money'),
    ]
    protocols = ['rowwise', 'group_aware']
    labels = ['Row-wise', 'Group-aware']
    colors = ['#4C78A8', '#F58518']
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))

    for ax, (column, title, direction, value_type) in zip(
        axes.ravel(), metric_specs
    ):
        grouped = (
            run_frame.groupby('protocol')[column]
            .agg(['mean', 'std', 'min', 'max'])
            .reindex(protocols)
        )
        means = grouped['mean'].to_numpy(dtype=float)
        stds = grouped['std'].fillna(0).to_numpy(dtype=float)
        bars = ax.bar(
            range(len(labels)),
            means,
            yerr=stds,
            capsize=5,
            color=colors,
            alpha=0.9,
        )
        for bar, mean, std in zip(bars, means, stds):
            if value_type == 'money':
                text = f'${mean:,.0f}\n+/- ${std:,.0f}'
            else:
                text = f'{mean:.3f}\n+/- {std:.3f}'
            ax.annotate(
                text,
                xy=(bar.get_x() + bar.get_width() / 2, mean + std),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=8,
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
        ax.set_ylabel('Value')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
        upper = max(float(np.nanmax(means + stds)) * 1.35, 0.05)
        ax.set_ylim(0, upper)
        ax.grid(axis='y', alpha=0.25)

    fig.suptitle(
        'Iranian Churn — Repeated Split Stability',
        fontsize=15,
        fontweight='bold',
        y=0.99,
    )
    fig.text(
        0.5,
        0.015,
        'Bars are means across 10 seeds; error bars and labels show +/- 1 SD. '
        'Costs and savings use placeholder COST_FP/COST_FN and are not final business claims.',
        ha='center',
        va='bottom',
        fontsize=8.5,
        color='#333333',
        bbox={
            'boxstyle': 'round,pad=0.45',
            'facecolor': '#f7f7f7',
            'edgecolor': '#cccccc',
        },
    )
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.95])
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURE_DIR / 'iranian_stability_metrics.png'
    fig.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return figure_path


def main():
    parser = argparse.ArgumentParser(
        description='Audit and stress-test the Iranian Churn dataset.'
    )
    parser.add_argument(
        '--audit-only',
        action='store_true',
        help='Only write the leakage and feature audit.',
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=DEFAULT_SEEDS,
        help='Seeds used for repeated row-wise and group-aware splits.',
    )
    args = parser.parse_args()

    frame, X, y = load_iranian()
    groups = feature_groups(X)
    report = audit_dataset(frame, X, y, groups)
    overlap = report['rowwise_seed_42_feature_overlap']
    row_count = report['rows']
    feature_count = report['features']
    exact_duplicate_count = report['exact_duplicate_rows']
    feature_duplicate_count = report['feature_duplicate_rows']
    conflicting_group_count = report['conflicting_duplicate_feature_groups']
    overlap_count = overlap['test_rows_matching_train']
    audit_path = AUDIT_DIR / 'audit_report.json'
    stability_path = AUDIT_DIR / 'stability_report.json'
    print('=== IRANIAN LEAKAGE AUDIT ===')
    print(f'rows={row_count:,}, features={feature_count}')
    print(f'exact_duplicate_rows={exact_duplicate_count:,}')
    print(f'feature_duplicate_rows={feature_duplicate_count:,}')
    print(
        'conflicting_duplicate_groups='
        f'{conflicting_group_count}'
    )
    print(
        'seed42 test rows matching train by feature='
        f'{overlap_count}'
    )
    print(f'saved={audit_path}')

    if args.audit_only:
        return

    runs, summary = run_stability(X, y, args.seeds)
    figure_path = save_stability_plot(runs)
    print('\n=== STABILITY SUMMARY ===')
    print(
        summary[
            summary['metric'].isin(
                [
                    'rf_average_precision',
                    'rf_roc_auc',
                    'rf_f1_policy',
                    'rf_recall_policy',
                    'rf_precision_policy',
                    'selected_threshold',
                ]
            )
        ].to_string(index=False)
    )
    print(f'\nsaved={stability_path}')
    print(f'saved={figure_path}')
    print(f'runs={len(runs)}')


if __name__ == '__main__':
    main()
