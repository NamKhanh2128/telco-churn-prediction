"""Random Forest ablation studies A2 and A3.

Usage:
    python -m src.ablation                    # all three datasets, 5 seeds
    python -m src.ablation --dataset ibm
    python -m src.ablation --quick            # 1 seed (smoke run)

Experiments (each changes exactly one factor; everything else fixed):
    A2  class-imbalance handling:
        - class_weight="balanced_subsample"   (default of the main protocol)
        - no imbalance handling               (class_weight=None)
        - SMOTE inside the pipeline           (class_weight=None; SMOTE is
          applied to the training folds only, after preprocessing)
    A3  numeric scaling:
        - no scaling                          (default for trees)
        - StandardScaler on numeric features

A1 (TotalCharges) is `python -m src.compare_datasets --dataset ibm
--keep-total-charges`; A4 (threshold 0.5 vs cost-optimal) lives in
`src.policy_analysis`.

Every variant uses the main 70/15/15 split (group-aware for Iranian), the
default RF hyper-parameters (DEFAULT_RF_PARAMS) and is scored on the test
split at threshold 0.5. Outputs:
    results/ablation/ablation_runs.csv     one row per dataset/variant/seed
    results/ablation/ablation_summary.csv  mean and std over seeds
"""

import argparse
import time

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier

from src.compare_datasets import _split_data, feature_lists, load_dataset
from src.config import DEFAULT_RF_PARAMS, RESULTS_DIR
from src.evaluate import compute_metrics
from src.preprocessing import TelcoFeatureEngineer, build_preprocessor

ABLATION_DIR = RESULTS_DIR / 'ablation'
SEEDS = [42, 43, 44, 45, 46]
DATASETS = ['ibm', 'orange', 'iranian']
METRICS = [
    'accuracy', 'precision_churn', 'recall_churn', 'f1_churn',
    'roc_auc', 'average_precision',
]

# (experiment, variant label, class_weight, use SMOTE, scale numeric)
VARIANTS = [
    ('A2', 'class_weight=balanced_subsample (default)', 'balanced_subsample', False, False),
    ('A2', 'no imbalance handling', None, False, False),
    ('A2', 'SMOTE in pipeline (no class_weight)', None, True, False),
    ('A3', 'no scaling (default)', 'balanced_subsample', False, False),
    ('A3', 'StandardScaler', 'balanced_subsample', False, True),
]


def build_variant(numeric_features, categorical_features, *, is_ibm,
                  class_weight, use_smote, scale_numeric, seed):
    """RF pipeline for one ablation variant.

    imblearn's Pipeline only resamples inside fit(); predict/predict_proba
    never see SMOTE, so the test split is untouched.
    """
    params = dict(DEFAULT_RF_PARAMS)
    params.update(class_weight=class_weight, random_state=seed)
    steps = []
    if is_ibm:
        steps.append(('feature_engineer', TelcoFeatureEngineer(enabled=True)))
    steps.append((
        'preprocessor',
        build_preprocessor(numeric_features, categorical_features, scale_numeric=scale_numeric),
    ))
    if use_smote:
        steps.append(('smote', SMOTE(random_state=seed)))
    steps.append(('classifier', RandomForestClassifier(**params)))
    return ImbPipeline(steps=steps)


def run_ablation(dataset_name, seeds):
    """Run every variant on one dataset for each seed; return result rows."""
    X, y, _, _ = load_dataset(dataset_name)
    numeric_features, categorical_features = feature_lists(X, dataset_name)
    rows = []
    for seed in seeds:
        X_train, _, X_test, y_train, _, y_test = _split_data(
            X, y, dataset_name=dataset_name, seed=seed,
        )
        for experiment, label, class_weight, use_smote, scale_numeric in VARIANTS:
            pipeline = build_variant(
                numeric_features,
                categorical_features,
                is_ibm=(dataset_name == 'ibm'),
                class_weight=class_weight,
                use_smote=use_smote,
                scale_numeric=scale_numeric,
                seed=seed,
            )
            start = time.time()
            pipeline.fit(X_train, y_train)
            fit_seconds = time.time() - start
            proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, (proba >= 0.5).astype(int), proba)
            row = {
                'dataset': dataset_name,
                'experiment': experiment,
                'variant': label,
                'seed': seed,
                'fit_seconds': round(fit_seconds, 2),
            }
            row.update({key: metrics[key] for key in METRICS})
            rows.append(row)
        print(f'[ablation] {dataset_name}: seed {seed} done')
    return rows


def summarise(runs):
    """Mean and std over seeds for every dataset/experiment/variant."""
    grouped = runs.groupby(['dataset', 'experiment', 'variant'], sort=False)[METRICS]
    mean = grouped.mean().add_suffix('_mean')
    std = grouped.std(ddof=1).fillna(0.0).add_suffix('_std')
    summary = pd.concat([mean, std], axis=1).reset_index()
    summary['n_seeds'] = grouped.size().values
    return summary.round(4)


def main():
    parser = argparse.ArgumentParser(description='Random Forest ablations A2 and A3.')
    parser.add_argument('--dataset', choices=DATASETS + ['all'], default='all')
    parser.add_argument('--quick', action='store_true', help='Run a single seed.')
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    seeds = SEEDS[:1] if args.quick else SEEDS
    rows = []
    for dataset_name in datasets:
        rows.extend(run_ablation(dataset_name, seeds))

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(ABLATION_DIR / 'ablation_runs.csv', index=False)
    summary = summarise(runs)
    summary.to_csv(ABLATION_DIR / 'ablation_summary.csv', index=False)

    columns = ['dataset', 'experiment', 'variant', 'f1_churn_mean', 'f1_churn_std',
               'recall_churn_mean', 'precision_churn_mean', 'average_precision_mean']
    print('\n=== ABLATION SUMMARY (test set, threshold 0.5, mean over seeds) ===')
    print(summary[columns].to_string(index=False))
    print(f'\nSaved {ABLATION_DIR}')


if __name__ == '__main__':
    main()
