"""Random Forest deep-dive: hyper-parameters, OOB, feature importance, local SHAP.

Usage:
    python -m src.rf_analysis                    # all three datasets
    python -m src.rf_analysis --dataset ibm
    python -m src.rf_analysis --quick            # fewer curve points / repeats

For each dataset (main 70/15/15 split; RF with the tuned benchmark parameters
from models/tuned_params.json):

1. Validation curves — CV average precision on the training split while
   varying n_estimators and max_depth (other parameters fixed at the tuned
   values): shows where the chosen setting sits on the bias/variance curve.
   Read differences against the fold spread (cv_std): steps smaller than
   that are noise, not evidence for one setting.
2. OOB score vs cross-validation vs test score. The CV score uses the same
   folds that picked the parameters, so it is the (optimistic) tuning score.
   OOB is computed on the same training rows that chose the parameters, so it
   shares part of that optimism (less than CV); only the test split is
   independent of the tuning.
3. Feature importance three ways — MDI (impurity, on model inputs incl.
   engineered features), permutation importance (test split, raw inputs) and
   mean |SHAP| (model inputs) — plus Spearman rank agreement between them.
4. Local SHAP explanations of three test customers: the most confident true
   positive (TP), the most clearly missed churner (FN) and the strongest false
   alarm (FP), at threshold 0.5 of the uncalibrated forest's score. SHAP
   explains that score (column rf_score). The deployed model applies a
   monotonic Platt calibration on top: it keeps the risk ranking of the
   customers, but SHAP values do not carry over exactly through a non-linear
   map, so on the calibrated-probability scale the size and order of the
   contributions can differ (and, rarely, the sign of a small one).

Iranian uses group-aware CV folds (duplicate rows stay in one fold). Its OOB
score can still be optimistic: a bootstrap sample may hold one copy of a
duplicate while the other copy is out-of-bag.

Outputs:
    results/rf_analysis/<dataset>_validation_curve.csv
    results/rf_analysis/<dataset>_importance.csv
    results/rf_analysis/<dataset>_local_shap.csv
    results/rf_analysis/rf_summary.csv              one row per dataset
    figures/rf/validation_curve_<dataset>.png
    figures/rf/importance_methods_<dataset>.png
    figures/rf/local_shap_<dataset>.png
"""

import argparse
import sys

import matplotlib
if 'ipykernel' not in sys.modules:  # scripts: no display; notebooks keep inline plots
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score, validation_curve

from src.compare_datasets import (
    _split_data,
    build_benchmark_model,
    cv_splitter,
    feature_groups,
    load_dataset,
)
from src.config import FIGURES_DIR, RANDOM_STATE, RESULTS_DIR, SEARCH_N_JOBS
from src.explain import (
    aggregate_shap_to_original_features,
    aggregate_signed_shap,
    compute_shap_values,
    explain_customer,
    input_features,
)

OUT_DIR = RESULTS_DIR / 'rf_analysis'
FIG_DIR = FIGURES_DIR / 'rf'
DATASETS = ['ibm', 'orange', 'iranian']
# Curve points cover every value of RF_PARAM_GRID (n_estimators 100/200/500,
# max_depth 5/10/15/None) plus a few in between; _with_tuned also adds the
# tuned value, so the chosen setting is always scored and marked.
CURVES = {
    'classifier__n_estimators': [50, 100, 200, 300, 500],
    'classifier__max_depth': [3, 5, 8, 10, 12, 15, None],
}
QUICK_CURVES = {
    'classifier__n_estimators': [50, 200],
    'classifier__max_depth': [5, None],
}


def _label(value):
    return 'None' if value is None else str(value)


def _with_tuned(curves, tuned):
    """Curve values plus the tuned value of each parameter (numbers sorted, None last)."""
    result = {}
    for param, values in curves.items():
        values = list(values)
        if param in tuned and tuned[param] not in values:
            values.append(tuned[param])
        numbers = sorted(value for value in values if value is not None)
        result[param] = numbers + ([None] if None in values else [])
    return result


def validation_curves(pipeline, X_train, y_train, groups, curves):
    """CV average precision for each value of each curve parameter."""
    rows = []
    for param, values in curves.items():
        estimator = clone(pipeline).set_params(classifier__n_jobs=1)
        train_scores, cv_scores = validation_curve(
            estimator, X_train, y_train,
            param_name=param, param_range=values, groups=groups,
            cv=cv_splitter(groups), scoring='average_precision', n_jobs=SEARCH_N_JOBS,
        )
        for value, train, cv in zip(values, train_scores, cv_scores):
            rows.append({
                'param': param.split('__')[-1],
                'value': _label(value),
                'train_mean': round(float(train.mean()), 4),
                'train_std': round(float(train.std()), 4),
                'cv_mean': round(float(cv.mean()), 4),
                'cv_std': round(float(cv.std()), 4),
            })
    return pd.DataFrame(rows)


def plot_validation_curves(curve_table, tuned, dataset_name):
    params = list(dict.fromkeys(curve_table['param']))
    fig, axes = plt.subplots(1, len(params), figsize=(6 * len(params), 4.6))
    axes = np.atleast_1d(axes)
    for ax, param in zip(axes, params):
        data = curve_table[curve_table['param'] == param]
        x = np.arange(len(data))
        for prefix, label, color in [('train', 'Train', '#E45756'), ('cv', 'Cross-validation', '#4C78A8')]:
            mean = data[f'{prefix}_mean'].to_numpy()
            std = data[f'{prefix}_std'].to_numpy()
            ax.plot(x, mean, marker='o', color=color, label=label)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        labels = data['value'].tolist()
        tuned_label = _label(tuned.get(f'classifier__{param}'))
        if tuned_label in labels:
            ax.axvline(labels.index(tuned_label), color='#555555', linestyle='--',
                       label=f'tuned = {tuned_label}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(param)
        ax.set_ylabel('Average precision')
        ax.set_title(f'{param} — {dataset_name}', fontweight='bold')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / f'validation_curve_{dataset_name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def importance_table(fitted, X_test, y_test, n_repeats, shap_rows):
    """MDI, permutation and mean |SHAP| importance side by side."""
    features = input_features(fitted)
    encoded = list(fitted.named_steps['preprocessor'].get_feature_names_out())
    mdi = aggregate_signed_shap(
        fitted.named_steps['classifier'].feature_importances_[None, :], encoded, features,
    ).iloc[0]

    permutation = permutation_importance(
        fitted, X_test, y_test, scoring='average_precision',
        n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=1,
    )
    permutation = pd.Series(permutation.importances_mean, index=X_test.columns)

    sample = X_test.sample(min(len(X_test), shap_rows), random_state=RANDOM_STATE)
    shap_values, _, shap_names = compute_shap_values(fitted, sample)
    shap_mean = aggregate_shap_to_original_features(shap_values, shap_names, features).mean()

    table = pd.DataFrame({'mdi': mdi, 'permutation': permutation, 'shap': shap_mean})
    for column in ['mdi', 'permutation', 'shap']:
        table[f'{column}_rank'] = table[column].rank(ascending=False, method='min')
    table.index.name = 'feature'
    return table.sort_values('shap', ascending=False).round(5)


def rank_agreement(table, first, second):
    both = table[[first, second]].dropna()
    if len(both) < 3:
        return None
    return round(float(spearmanr(both[first], both[second]).statistic), 3)


def plot_importance(table, dataset_name, top=10):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    titles = {
        'mdi': 'MDI (impurity) — model inputs',
        'permutation': 'Permutation (test AP drop) — raw inputs',
        'shap': 'Mean |SHAP| — model inputs',
    }
    for ax, (column, title) in zip(axes, titles.items()):
        data = table[column].dropna().sort_values(ascending=False).head(top)[::-1]
        ax.barh(data.index, data.values, color='#4C78A8')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='x', alpha=0.25)
    fig.suptitle(f'Feature importance, three methods — {dataset_name}', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / f'importance_methods_{dataset_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def pick_cases(y_true, proba):
    """Indices of the TP / FN / FP customers to explain (threshold 0.5)."""
    y_true = np.asarray(y_true)
    predicted = proba >= 0.5
    cases = {}
    choices = [
        ('TP (most confident churner caught)', (y_true == 1) & predicted, np.argmax),
        ('FN (churner most clearly missed)', (y_true == 1) & ~predicted, np.argmin),
        ('FP (strongest false alarm)', (y_true == 0) & predicted, np.argmax),
    ]
    for name, mask, chooser in choices:
        candidates = np.flatnonzero(mask)
        if candidates.size:
            cases[name] = int(candidates[chooser(proba[candidates])])
    return cases


def local_explanations(fitted, X_test, y_test, proba):
    """Top SHAP factors of the TP / FN / FP customers.

    rf_score is the uncalibrated forest's churn score that SHAP decomposes,
    not the calibrated probability of the deployed model.
    """
    rows = []
    for case, index in pick_cases(y_test, proba).items():
        for rank, factor in enumerate(explain_customer(fitted, X_test.iloc[[index]], top_n=6), 1):
            factor = dict(factor)
            factor['source_features'] = ', '.join(factor.get('source_features', []))
            rows.append({
                'case': case,
                'test_row': index,
                'rf_score': round(float(proba[index]), 4),
                'actual_churn': int(np.asarray(y_test)[index]),
                'rank': rank,
                **factor,
            })
    return pd.DataFrame(rows)


def plot_local(local, dataset_name):
    cases = list(dict.fromkeys(local['case']))
    if not cases:
        return
    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 4.4))
    axes = np.atleast_1d(axes)
    for ax, case in zip(axes, cases):
        data = local[local['case'] == case].iloc[::-1]
        labels = [f'{f} = {v}'[:38] for f, v in zip(data['feature'], data['value'])]
        colors = ['#E45756' if c > 0 else '#4C78A8' for c in data['contribution']]
        ax.barh(labels, data['contribution'], color=colors)
        ax.axvline(0, color='#333333', linewidth=0.8)
        score = data['rf_score'].iloc[0]
        ax.set_title(f'{case}\nRF score (uncalibrated) = {score:.2f}', fontsize=9.5, fontweight='bold')
        ax.set_xlabel('SHAP contribution (red = towards churn)')
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='x', alpha=0.25)
    fig.suptitle(f'Local SHAP explanations — {dataset_name}', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / f'local_shap_{dataset_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def analyse_dataset(dataset_name, quick=False):
    X, y, _, _ = load_dataset(dataset_name)
    X_train, _, X_test, y_train, _, y_test = _split_data(X, y, dataset_name=dataset_name)
    groups = feature_groups(X_train) if dataset_name == 'iranian' else None
    pipeline = build_benchmark_model('RandomForest', X, dataset_name)
    tuned = {
        key: value for key, value in pipeline.get_params().items()
        if key in ('classifier__n_estimators', 'classifier__max_depth')
    }

    curves = validation_curves(
        pipeline, X_train, y_train, groups,
        _with_tuned(QUICK_CURVES if quick else CURVES, tuned),
    )
    curves.to_csv(OUT_DIR / f'{dataset_name}_validation_curve.csv', index=False)
    plot_validation_curves(curves, tuned, dataset_name)
    print(f'[rf_analysis] {dataset_name}: validation curves done')

    fitted = clone(pipeline).set_params(
        classifier__oob_score=True, classifier__n_jobs=SEARCH_N_JOBS,
    ).fit(X_train, y_train)
    oob = fitted.named_steps['classifier'].oob_decision_function_[:, 1]
    mask = np.isfinite(oob)
    y_train_array = np.asarray(y_train)
    cv_ap = cross_val_score(
        clone(pipeline).set_params(classifier__n_jobs=1), X_train, y_train,
        groups=groups, cv=cv_splitter(groups), scoring='average_precision', n_jobs=SEARCH_N_JOBS,
    )
    proba = fitted.predict_proba(X_test)[:, 1]
    print(f'[rf_analysis] {dataset_name}: OOB/CV/test done')

    importance = importance_table(
        fitted, X_test, y_test, n_repeats=3 if quick else 10, shap_rows=100 if quick else 300,
    )
    importance.to_csv(OUT_DIR / f'{dataset_name}_importance.csv')
    plot_importance(importance, dataset_name)
    print(f'[rf_analysis] {dataset_name}: importance done')

    local = local_explanations(fitted, X_test, y_test, proba)
    local.to_csv(OUT_DIR / f'{dataset_name}_local_shap.csv', index=False)
    plot_local(local, dataset_name)
    print(f'[rf_analysis] {dataset_name}: local SHAP done')

    def top(column):
        return '; '.join(importance[column].dropna().sort_values(ascending=False).head(5).index)

    return {
        'dataset': dataset_name,
        'n_estimators': _label(tuned.get('classifier__n_estimators')),
        'max_depth': _label(tuned.get('classifier__max_depth')),
        'oob_ap': round(float(average_precision_score(y_train_array[mask], oob[mask])), 4),
        'oob_roc_auc': round(float(roc_auc_score(y_train_array[mask], oob[mask])), 4),
        'cv_ap_mean': round(float(cv_ap.mean()), 4),
        'cv_ap_std': round(float(cv_ap.std()), 4),
        'test_ap': round(float(average_precision_score(y_test, proba)), 4),
        'test_roc_auc': round(float(roc_auc_score(y_test, proba)), 4),
        'spearman_mdi_permutation': rank_agreement(importance, 'mdi', 'permutation'),
        'spearman_mdi_shap': rank_agreement(importance, 'mdi', 'shap'),
        'spearman_permutation_shap': rank_agreement(importance, 'permutation', 'shap'),
        'top5_mdi': top('mdi'),
        'top5_permutation': top('permutation'),
        'top5_shap': top('shap'),
    }


def main():
    parser = argparse.ArgumentParser(description='Random Forest deep-dive analysis.')
    parser.add_argument('--dataset', choices=DATASETS + ['all'], default='all')
    parser.add_argument('--quick', action='store_true', help='Fewer curve points and repeats.')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    summaries = [analyse_dataset(name, quick=args.quick) for name in datasets]

    summary = pd.DataFrame(summaries)
    path = OUT_DIR / 'rf_summary.csv'
    if path.exists():
        previous = pd.read_csv(path)
        previous = previous[~previous['dataset'].isin(summary['dataset'])]
        summary = pd.concat([previous, summary], ignore_index=True)
    summary = summary.sort_values('dataset').reset_index(drop=True)
    summary.to_csv(path, index=False)
    print('\n=== RF ANALYSIS SUMMARY ===')
    print(summary.drop(columns=['top5_mdi', 'top5_permutation', 'top5_shap']).to_string(index=False))
    print(f'\nSaved {OUT_DIR} and {FIG_DIR}')


if __name__ == '__main__':
    main()
