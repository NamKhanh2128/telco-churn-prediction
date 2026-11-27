"""Export the numbers the report cites into one tracked Markdown file.

Usage:
    python -m src.export_results

Reads only generated artifacts under results/ (which are git-ignored) and writes
reports/results_snapshot.md (which is tracked). README, mustdo and the written
report should quote numbers from this snapshot, never type them by hand.

If the last `src.reproduce` run was `--quick`, the snapshot goes to
results_snapshot_quick.md inside the quick sandbox (results/quick/results/,
git-ignored) instead, so smoke runs on a member's machine never overwrite the
official numbers.

Other members' modules can add a section without editing this file: write a
Markdown fragment to results/<name>/snapshot.md (start it with a "### "
heading). Every fragment found is appended under "10. Additional analyses",
in alphabetical order of <name>.
"""

import json

import pandas as pd

from src.config import COST_FN, COST_FP, PROJECT_ROOT, RESULTS_DIR, SANDBOXED

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


def _provenance_lines(log):
    """Data hashes and library versions recorded by src.reproduce."""
    lines = []
    hashes = log.get('dataset_sha256') or {}
    if hashes:
        lines.append('- Data SHA-256: ' + ', '.join(
            f"{name} `{(digest or 'missing')[:12]}`" for name, digest in hashes.items()
        ) + ' (first 12 hex digits; full values in `results/reproduce_log.json`)')
    versions = log.get('versions') or {}
    if versions:
        lines.append('- Versions: ' + ', '.join(
            f'{name} {version}' for name, version in versions.items() if version
        ))
    return lines


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
    })
    if 'ece_mean' in table:
        view['ECE'] = table['ece_mean']
    if 'val_average_precision_mean' in table:
        view['Val AP'] = table['val_average_precision_mean']
    view['Seeds'] = table['n_seeds']
    return [
        '## 6. Ablation A2 (imbalance handling) and A3 (scaling) — RF, tuned parameters',
        '',
        'Mean ± std over seeds, test set at threshold 0.5 (`src.ablation`); every variant '
        'uses the tuned forest of the main benchmark and changes one factor. '
        '`Val AP` is the validation-split AP that design choices are read from.',
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
    n_seeds = int(summary['count'].max()) if 'count' in summary and not summary.empty else 0
    summary['cell'] = [_mean_std(m, s) for m, s in zip(summary['mean'], summary['std'])]
    wide = summary.pivot_table(
        index=['dataset', 'model'], columns='metric', values='cell', aggfunc='first', sort=False,
    ).reset_index()
    wide = wide[['dataset', 'model', *metrics]].rename(
        columns={'dataset': 'Dataset', 'model': 'Model', **metrics},
    )
    parts = [
        f'## 7. Stability over {n_seeds} seeds — all models (test set, threshold 0.5)',
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
            'Scope': tests['scope'] if 'scope' in tests else '',
            'RF': tests['rf_mean'],
            'Rival': tests['rival'],
            'Rival score': tests['rival_mean'],
            'Statistic': tests['statistic'].astype(str),
            'p-value': tests['p_value'].map(lambda p: f'{p:.3g}'),
            'n': tests['n_pairs'],
        })
        parts += [
            'RandomForest vs the strongest other model, two-sided, not corrected for '
            'multiple comparisons. `corrected_t` = Nadeau–Bengio corrected resampled t-test '
            '(accounts for overlapping training sets; the one to cite), `wilcoxon` = signed-rank '
            'test over the same seeds (ignores the overlap; smallest possible p with 10 pairs '
            'is 0.002), `mcnemar` = exact test on the seed-42 test set. Hyper-parameters were '
            'tuned once on the seed-42 training split and reused for every seed. `Scope`: AP and '
            'ROC-AUC compare how well the models rank customers (model quality); F1 and McNemar '
            'use threshold 0.5, which is a different operating point for the uncalibrated, '
            'class-weighted RF than for the Platt-calibrated rival, so those rows compare '
            'operating points, not models.',
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
        'mean_proba': 'Mean p',
        'test_churn_rate': 'Churn rate',
        'val_average_precision': 'Val AP',
        'val_brier_score': 'Val Brier',
        'size_mb': 'Size MB',
        'size_mb_compressed': 'Size MB (compressed)',
    }
    columns = {key: label for key, label in columns.items() if key in table}
    return [
        '## 8. Calibration of the deployed RF — quality vs size',
        '',
        'From `src.calibration_analysis`; reliability diagrams in `figures/rf/`. '
        'AP / Brier / ECE are on the test set; `Mean p` is the average predicted churn '
        'probability next to the real churn rate of the test set. The choice of the deployed '
        'variant is read from the validation columns (`Val AP`, `Val Brier`). Sizes are the '
        'serialized model without / with joblib compression.',
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
        'CV AP (tuning folds)': [
            _mean_std(m, s) for m, s in zip(table['cv_ap_mean'], table['cv_ap_std'])
        ],
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
        'importance methods). The CV score reuses the folds that picked the parameters, '
        'so it is optimistic; OOB is computed on the same training rows, so it shares part '
        'of that optimism; only the test split is independent of the tuning. '
        'Figures in `figures/rf/`.',
        '',
        _md_table(scores),
        '',
        _md_table(tops),
        '',
    ]


def _fragment_section():
    """Markdown fragments written by other modules (results/<name>/snapshot.md)."""
    fragments = sorted(RESULTS_DIR.glob('*/snapshot.md'))
    if not fragments:
        return []
    parts = ['## 10. Additional analyses', '']
    for path in fragments:
        parts += [
            f'From `results/{path.parent.name}/snapshot.md`.',
            '',
            path.read_text(encoding='utf-8').strip(),
            '',
        ]
    return parts


def main(out_path=None):
    log_path = RESULTS_DIR / 'reproduce_log.json'
    log = _load_json(log_path) if log_path.exists() else {}
    if out_path is None:
        # A quick run, or any run inside a CHURN_OUTPUT_ROOT sandbox, never
        # writes the tracked official snapshot.
        out_path = QUICK_OUT_PATH if (log.get('mode') == 'quick' or SANDBOXED) else OUT_PATH
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
        *_provenance_lines(log),
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
    parts += _fragment_section()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(parts), encoding='utf-8')
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
