"""Export the numbers the report cites into one tracked Markdown file.

Usage:
    python -m src.export_results

Reads only generated artifacts under results/ (which are git-ignored) and writes
reports/results_snapshot.md (which is tracked). README, mustdo and the written
report should quote numbers from this snapshot, never type them by hand.

If the last `src.reproduce` run was `--quick`, the snapshot goes to the
git-ignored results/results_snapshot_quick.md instead, so smoke runs on a
member's machine never overwrite the official numbers.
"""

import json

import pandas as pd

from src.config import COST_FN, COST_FP, PROJECT_ROOT, RESULTS_DIR

OUT_PATH = PROJECT_ROOT / 'reports' / 'results_snapshot.md'
QUICK_OUT_PATH = RESULTS_DIR / 'results_snapshot_quick.md'
RUNS_DIR = RESULTS_DIR / 'dataset_runs'


def _load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def _fmt(value, digits=4):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return '—'
    if isinstance(value, (int, float)):
        if float(value).is_integer() and abs(value) >= 1:
            return f'{int(value):,}'
        return f'{value:.{digits}f}'
    return str(value)


def _md_table(frame, digits=4):
    headers = list(frame.columns)
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '|' + '|'.join(['---'] + ['---:'] * (len(headers) - 1)) + '|',
    ]
    for _, row in frame.iterrows():
        lines.append('| ' + ' | '.join(_fmt(row[h], digits) for h in headers) + ' |')
    return '\n'.join(lines)


def _ibm_model_table():
    table = pd.read_csv(RESULTS_DIR / 'comparison_table.csv')
    columns = {
        'Model': 'Model',
        'accuracy': 'Accuracy',
        'precision_churn': 'Precision (churn)',
        'recall_churn': 'Recall (churn)',
        'f1_churn': 'F1 (churn)',
        'f1_macro': 'F1 macro',
        'roc_auc': 'ROC-AUC',
        'average_precision': 'PR-AUC (AP)',
        'brier_score': 'Brier',
        'ece': 'ECE',
    }
    return table[list(columns)].rename(columns=columns)


def _policy_section():
    policy = _load_json(RESULTS_DIR / 'policy_metrics.json')
    rows = [
        {
            'Model': name,
            'Threshold (val)': item['best_threshold'],
            'Val cost': item['validation_cost'],
            'Val savings vs 0.5': item['validation_savings'],
            'Test precision': item['test_policy_precision'],
            'Test recall': item['test_policy_recall'],
            'Test F1': item['test_policy_f1'],
        }
        for name, item in policy['all_models_policy'].items()
    ]
    return pd.DataFrame(rows)


def _dataset_table():
    table = pd.read_csv(RUNS_DIR / 'dataset_comparison.csv')
    columns = {
        'dataset': 'Run',
        'rows': 'Rows',
        'features': 'Features',
        'churn_rate': 'Churn rate',
        'rf_roc_auc': 'RF ROC-AUC',
        'rf_average_precision': 'RF AP',
        'logreg_roc_auc': 'LogReg ROC-AUC',
        'logreg_average_precision': 'LogReg AP',
        'rf_brier_score': 'RF Brier (cal.)',
        'rf_ece': 'RF ECE (cal.)',
        'selected_threshold': 'Threshold',
        'rf_f1_policy': 'RF F1 @thr',
        'rf_recall_policy': 'RF Recall @thr',
    }
    return table[list(columns)].rename(columns=columns)


def _stability_table():
    path = RUNS_DIR / 'iranian' / 'stability_summary.csv'
    if not path.exists():
        return None
    table = pd.read_csv(path)
    keep = ['rf_average_precision', 'rf_roc_auc', 'rf_f1_policy', 'selected_threshold']
    return table[table['metric'].isin(keep)].reset_index(drop=True)


def _read_optional(path):
    return pd.read_csv(path) if path.exists() else None


def _mean_std(mean, std):
    return f'{mean:.4f} ± {std:.4f}'


def _all_models_section():
    table = _read_optional(RUNS_DIR / 'model_comparison_long.csv')
    if table is None:
        return []
    columns = {
        'dataset': 'Run',
        'model': 'Model',
        'accuracy': 'Accuracy',
        'precision_churn': 'Precision (churn)',
        'recall_churn': 'Recall (churn)',
        'f1_churn': 'F1 (churn)',
        'roc_auc': 'ROC-AUC',
        'average_precision': 'PR-AUC (AP)',
    }
    return [
        '## 5. All benchmark models × datasets (test set, threshold 0.5)',
        '',
        'From `src.compare_datasets` (`results/dataset_runs/model_comparison_long.csv`).',
        '',
        _md_table(table[list(columns)].rename(columns=columns)),
        '',
    ]


def _ablation_section():
    table = _read_optional(RESULTS_DIR / 'ablation' / 'ablation_summary.csv')
    if table is None:
        return []
    view = pd.DataFrame({
        'Dataset': table['dataset'],
        'Exp.': table['experiment'],
        'Variant': table['variant'],
        'F1 (churn)': [_mean_std(m, s) for m, s in zip(table['f1_churn_mean'], table['f1_churn_std'])],
        'Recall (churn)': table['recall_churn_mean'],
        'Precision (churn)': table['precision_churn_mean'],
        'PR-AUC (AP)': [
            _mean_std(m, s)
            for m, s in zip(table['average_precision_mean'], table['average_precision_std'])
        ],
        'Seeds': table['n_seeds'],
    })
    return [
        '## 6. Ablation A2 (imbalance handling) and A3 (scaling) — RF, default parameters',
        '',
        'Mean ± std over seeds, test set at threshold 0.5 (`src.ablation`).',
        '',
        _md_table(view),
        '',
    ]


def _stability_models_section():
    summary = _read_optional(RESULTS_DIR / 'model_stability' / 'stability_summary.csv')
    if summary is None:
        return []
    metrics = {
        'accuracy': 'Accuracy',
        'precision_churn': 'Precision',
        'recall_churn': 'Recall',
        'f1_churn': 'F1',
        'roc_auc': 'ROC-AUC',
        'average_precision': 'AP',
    }
    summary = summary[summary['metric'].isin(metrics)].copy()
    summary['cell'] = [_mean_std(m, s) for m, s in zip(summary['mean'], summary['std'])]
    wide = summary.pivot_table(
        index=['dataset', 'model'], columns='metric', values='cell', aggfunc='first', sort=False,
    ).reset_index()
    wide = wide[['dataset', 'model', *metrics]].rename(
        columns={'dataset': 'Dataset', 'model': 'Model', **metrics},
    )
    parts = [
        '## 7. Stability over 10 seeds — all models (test set, threshold 0.5)',
        '',
        'Mean ± std over repeated 70/15/15 splits with fixed tuned parameters '
        '(`src.model_stability`).',
        '',
        _md_table(wide),
        '',
    ]
    tests = _read_optional(RESULTS_DIR / 'model_stability' / 'stat_tests.csv')
    if tests is not None and not tests.empty:
        view = pd.DataFrame({
            'Dataset': tests['dataset'],
            'Test': tests['test'],
            'RF': tests['rf_mean'],
            'Rival': tests['rival'],
            'Rival score': tests['rival_mean'],
            'Statistic': tests['statistic'].astype(str),
            'p-value': tests['p_value'].map(lambda p: f'{p:.3g}'),
            'n': tests['n_pairs'],
        })
        parts += [
            'RandomForest vs the strongest other model (two-sided, not corrected for '
            'multiple comparisons; Wilcoxon over seeds, McNemar on the seed-42 test set):',
            '',
            _md_table(view.astype(object).where(view.notna(), None)),
            '',
        ]
    return parts


def _calibration_section():
    table = _read_optional(RESULTS_DIR / 'calibration' / 'calibration_comparison.csv')
    if table is None:
        return []
    table = table.copy()
    if 'deployed' in table:
        table['deployed'] = table['deployed'].map(lambda value: 'yes' if bool(value) else '')
    else:
        table['deployed'] = ''
    for column in ['size_mb', 'size_mb_compressed']:
        table[column] = table[column].map(lambda value: f'{value:.1f}')
    columns = {
        'dataset': 'Dataset',
        'model': 'Model',
        'deployed': 'Deployed',
        'roc_auc': 'ROC-AUC',
        'average_precision': 'AP',
        'brier_score': 'Brier',
        'ece': 'ECE',
        'size_mb': 'Size MB',
        'size_mb_compressed': 'Size MB (compressed)',
    }
    return [
        '## 8. Calibration of the deployed RF — quality vs size (test set)',
        '',
        'From `src.calibration_analysis`; reliability diagrams in `figures/rf/`.',
        '',
        _md_table(table[list(columns)].rename(columns=columns)),
        '',
    ]


def _rf_analysis_section():
    table = _read_optional(RESULTS_DIR / 'rf_analysis' / 'rf_summary.csv')
    if table is None:
        return []

    def as_label(value):
        # CSV round-trips turn 'None' into NaN and 10 into 10.0.
        if pd.isna(value):
            return 'None'
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value)

    scores = pd.DataFrame({
        'Dataset': table['dataset'],
        'n_estimators': table['n_estimators'].map(as_label),
        'max_depth': table['max_depth'].map(as_label),
        'OOB AP': table['oob_ap'],
        'CV AP': [_mean_std(m, s) for m, s in zip(table['cv_ap_mean'], table['cv_ap_std'])],
        'Test AP': table['test_ap'],
        'ρ(MDI, perm.)': table['spearman_mdi_permutation'],
        'ρ(MDI, SHAP)': table['spearman_mdi_shap'],
        'ρ(perm., SHAP)': table['spearman_permutation_shap'],
    })
    tops = pd.DataFrame({
        'Dataset': table['dataset'],
        'Top-5 MDI': table['top5_mdi'],
        'Top-5 permutation': table['top5_permutation'],
        'Top-5 SHAP': table['top5_shap'],
    })
    return [
        '## 9. Random Forest analysis — OOB vs CV vs test, importance agreement',
        '',
        'From `src.rf_analysis` (tuned RF; ρ = Spearman rank correlation between '
        'importance methods). Figures in `figures/rf/`.',
        '',
        _md_table(scores),
        '',
        _md_table(tops),
        '',
    ]


def main(out_path=None):
    log_path = RESULTS_DIR / 'reproduce_log.json'
    log = _load_json(log_path) if log_path.exists() else {}
    if out_path is None:
        out_path = QUICK_OUT_PATH if log.get('mode') == 'quick' else OUT_PATH
    ibm_meta = _load_json(RUNS_DIR / 'ibm' / 'metadata.json')
    split = ibm_meta['split']

    parts = [
        '# Results snapshot',
        '',
        '> Auto-generated by `python -m src.export_results` from `results/`. '
        'Do not edit by hand — rerun `python -m src.reproduce` then this script.',
        '',
        f"- Run: `{log.get('mode', 'unknown')}` mode, status `{log.get('status', 'unknown')}`, "
        f"finished {log.get('finished_utc', 'unknown')} UTC",
        f"- Git commit: `{log.get('git_commit', 'unknown')}`"
        + (' (working tree had uncommitted changes)' if log.get('git_dirty') else ''),
        f"- IBM split: train {split['train']:,} / validation {split['validation']:,} / "
        f"test {split['test']:,} (seed {split['random_state']})",
        f'- Cost placeholders: FP = ${COST_FP}, FN = ${COST_FN} '
        '(NOT business-validated; savings below are illustrative only)',
        '',
        '## 1. IBM Telco — model comparison on the test set (threshold 0.5)',
        '',
        '`RandomForest` is the tuned, uncalibrated forest. `RF_Calibrated@…` rows are '
        'the deployed Platt-calibrated RF at 0.5 and at the validation-selected threshold.',
        '',
        _md_table(_ibm_model_table()),
        '',
        '## 2. IBM Telco — cost-optimal threshold (chosen on validation, scored on test)',
        '',
        _md_table(_policy_section()),
        '',
        '## 3. Three-dataset benchmark (RF vs LogReg, test set)',
        '',
        '`ibm_with_total_charges` is ablation A1 (keeps TotalCharges).',
        '',
        _md_table(_dataset_table()),
        '',
    ]

    stability = _stability_table()
    if stability is not None:
        parts += [
            '## 4. Iranian Churn — 10-seed stability (row-wise vs group-aware split)',
            '',
            _md_table(stability),
            '',
        ]

    parts += _all_models_section()
    parts += _ablation_section()
    parts += _stability_models_section()
    parts += _calibration_section()
    parts += _rf_analysis_section()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(parts), encoding='utf-8')
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
