"""
Evaluation metrics, visualization, and reporting.

Generates:
- Classification report
- Confusion matrix plot
- ROC curve
- Precision-Recall curve
- Comparison table across models
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from src.config import FIGURES_DIR, RESULTS_DIR


def expected_calibration_error(y_true, y_proba, n_bins=10):
    '''Compute the weighted absolute calibration error for binary scores.'''
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError('y_true and y_proba must have the same length')
    if not np.isfinite(y_proba).all():
        raise ValueError('y_proba contains NaN or infinite values')

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_proba, bin_edges[1:-1], right=False)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        confidence = y_proba[mask].mean()
        frequency = y_true[mask].mean()
        ece += mask.mean() * abs(confidence - frequency)
    return float(ece)


def lift_at_fraction(y_true, y_score, fraction=0.10):
    '''Compute top-decile lift for retention prioritization.'''
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0 or y_true.size != y_score.size:
        return None
    baseline_rate = float(y_true.mean())
    if baseline_rate == 0:
        return None
    n_selected = max(1, int(np.ceil(y_true.size * fraction)))
    top_indices = np.argsort(y_score)[-n_selected:]
    precision_at_k = float(y_true[top_indices].mean())
    return round(precision_at_k / baseline_rate, 4)


def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Compute all evaluation metrics for binary classification.

    Returns
    -------
    dict
        Keys: accuracy, precision_churn, recall_churn, f1_churn, f1_macro, roc_auc
    """
    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)

    metrics = {
        'balanced_accuracy': round(balanced_accuracy_score(y_true, y_pred), 4),
        'precision_macro': round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'recall_macro': round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_churn": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall_churn": round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "f1_churn": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }

    if y_proba is not None:
        metrics['average_precision'] = round(average_precision_score(y_true, y_proba), 4)
        metrics['lift_at_10pct'] = lift_at_fraction(y_true, y_proba, fraction=0.10)
        is_probability = bool(np.all((y_proba >= 0.0) & (y_proba <= 1.0)))
        if is_probability:
            metrics['brier_score'] = round(brier_score_loss(y_true, y_proba), 4)
            metrics['log_loss'] = round(log_loss(y_true, np.column_stack([1 - y_proba, y_proba])), 4)
            metrics['ece'] = round(expected_calibration_error(y_true, y_proba), 4)
        else:
            metrics['brier_score'] = None
            metrics['log_loss'] = None
            metrics['ece'] = None
    else:
        metrics['average_precision'] = None
        metrics['lift_at_10pct'] = None
        metrics['brier_score'] = None
        metrics['log_loss'] = None
        metrics['ece'] = None

    if y_proba is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
    else:
        metrics["roc_auc"] = None

    return metrics


def print_evaluation(name, y_true, y_pred, y_proba=None):
    """
    Print full evaluation for a model.

    Returns
    -------
    dict : metrics
    """
    metrics = compute_metrics(y_true, y_pred, y_proba)

    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")
    print(classification_report(
        y_true, y_pred,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    ))
    if metrics["roc_auc"] is not None:
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"{'=' * 55}")

    return metrics


def _format_metric(value, digits=3):
    if value is None:
        return 'n/a'
    try:
        if not np.isfinite(float(value)):
            return 'n/a'
    except (TypeError, ValueError):
        return 'n/a'
    return f'{float(value):.{digits}f}'


def _metric_footer_lines(results, metric_specs):
    '''Create aligned, monospace metric rows for the bottom of a figure.'''
    header = 'Model'.ljust(18)
    for label, _ in metric_specs:
        header += f' {label:>10}'

    lines = [header]
    for name, res in results.items():
        metrics = res.get('metrics', {})
        line = str(name)[:18].ljust(18)
        for _, key in metric_specs:
            line += f' {_format_metric(metrics.get(key)):>10}'
        lines.append(line)
    return lines


def _add_metric_footer(fig, lines, bottom=0.22):
    '''Place an aligned metric note below the plotting area.'''
    if not lines:
        return
    fig.text(
        0.5,
        0.018,
        '\n'.join(lines),
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
    fig.tight_layout(rect=[0.02, bottom, 0.98, 0.96])


def _annotate_bars(ax, bars, values, digits=3):
    '''Write the exact value above each bar without relying on bar height.'''
    for bar, value in zip(bars, values):
        label = _format_metric(value, digits=digits)
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=9,
            color='#222222',
        )


def plot_confusion_matrices(results, X_test, y_test, save=True):
    """
    Plot confusion matrices for all models side by side.

    Parameters
    ----------
    results : dict[str, dict]
        Keys are model names, values have key "pipeline" (fitted Pipeline)
    """
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.7))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        ConfusionMatrixDisplay.from_estimator(
            res["pipeline"], X_test, y_test,
            display_labels=["No Churn", "Churn"],
            cmap="Blues", ax=ax, values_format='d', colorbar=False,
        )
        ax.set_title(name, fontsize=12, fontweight="bold")

    footer = _metric_footer_lines(
        results,
        [
            ('F1', 'f1_churn'),
            ('Precision', 'precision_churn'),
            ('Recall', 'recall_churn'),
            ('Bal.Acc', 'balanced_accuracy'),
        ],
    )
    _add_metric_footer(fig, footer, bottom=0.25)
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
        print(f"[evaluate] Saved confusion_matrices.png")
    plt.close(fig)


def plot_roc_curves(results, X_test, y_test, save=True):
    """
    Plot ROC curves for all models on one figure.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))

    for name, res in results.items():
        auc_value = res.get('metrics', {}).get('roc_auc')
        curve_name = (
            f'{name} (AUC={auc_value:.3f})'
            if auc_value is not None else name
        )
        pipeline = res["pipeline"]
        if hasattr(pipeline, "predict_proba"):
            RocCurveDisplay.from_estimator(
                pipeline, X_test, y_test, name=curve_name, ax=ax
            )
        elif hasattr(pipeline, "decision_function"):
            RocCurveDisplay.from_estimator(
                pipeline, X_test, y_test, name=curve_name, ax=ax,
                response_method="decision_function"
            )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.5)")
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    footer = _metric_footer_lines(
        results,
        [
            ('AUC', 'roc_auc'),
            ('AP', 'average_precision'),
            ('Brier', 'brier_score'),
            ('ECE', 'ece'),
        ],
    )
    _add_metric_footer(fig, footer, bottom=0.25)

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
        print(f"[evaluate] Saved roc_curves.png")
    plt.close(fig)


def plot_precision_recall_curves(results, X_test, y_test, save=True):
    """
    Plot Precision-Recall curves for all models on one figure.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))

    for name, res in results.items():
        ap_value = res.get('metrics', {}).get('average_precision')
        curve_name = (
            f'{name} (AP={ap_value:.3f})'
            if ap_value is not None else name
        )
        pipeline = res["pipeline"]
        if hasattr(pipeline, "predict_proba"):
            PrecisionRecallDisplay.from_estimator(
                pipeline, X_test, y_test, name=curve_name, ax=ax,
                pos_label=1,
            )
        elif hasattr(pipeline, 'decision_function'):
            PrecisionRecallDisplay.from_estimator(
                pipeline,
                X_test,
                y_test,
                name=curve_name,
                ax=ax,
                pos_label=1,
                response_method='decision_function',
            )

    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.25)
    footer = _metric_footer_lines(
        results,
        [
            ('AP', 'average_precision'),
            ('Lift@10%', 'lift_at_10pct'),
            ('F1', 'f1_churn'),
            ('Recall', 'recall_churn'),
        ],
    )
    _add_metric_footer(fig, footer, bottom=0.25)

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "precision_recall_curves.png", dpi=150, bbox_inches="tight")
        print(f"[evaluate] Saved precision_recall_curves.png")
    plt.close(fig)


def plot_metric_summary(results, save=True):
    '''Plot readable test-set metric bars with values printed above each bar.'''
    if not results:
        return

    metric_specs = [
        ('roc_auc', 'ROC-AUC', 'higher is better'),
        ('average_precision', 'Average Precision', 'higher is better'),
        ('f1_churn', 'F1 (Churn)', 'higher is better'),
        ('recall_churn', 'Recall (Churn)', 'higher is better'),
        ('balanced_accuracy', 'Balanced Accuracy', 'higher is better'),
        ('lift_at_10pct', 'Lift @ top 10%', 'higher is better'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()
    names = list(results.keys())

    for ax, (key, title, direction) in zip(axes, metric_specs):
        values = [
            results[name].get('metrics', {}).get(key)
            for name in names
        ]
        numeric_values = [
            float(value) if value is not None and np.isfinite(float(value))
            else 0.0
            for value in values
        ]
        bars = ax.bar(
            range(len(names)),
            numeric_values,
            color='#4C78A8',
            alpha=0.9,
        )
        _annotate_bars(ax, bars, values)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
        ax.set_ylim(0, max(1.0, max(numeric_values, default=0.0) * 1.28))
        ax.grid(axis='y', alpha=0.25)
        ax.text(
            0.02,
            0.95,
            direction,
            transform=ax.transAxes,
            fontsize=8,
            color='#555555',
            va='top',
        )

    fig.suptitle(
        'Test-set Metric Summary',
        fontsize=15,
        fontweight='bold',
        y=0.99,
    )
    _add_metric_footer(
        fig,
        [
            'Labels above bars are exact test-set values.',
            'All panels use the default probability threshold; the selected business threshold is shown in cost_threshold.png.',
        ],
        bottom=0.14,
    )
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            FIGURES_DIR / 'metric_summary.png',
            dpi=150,
            bbox_inches='tight',
        )
        print('[evaluate] Saved metric_summary.png')
    plt.close(fig)


def build_comparison_table(results):
    """
    Build a DataFrame comparing all models.

    Parameters
    ----------
    results : dict[str, dict]
        Each value has key "metrics" (output of compute_metrics)

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for name, res in results.items():
        row = {"Model": name, **res["metrics"]}
        if "train_time" in res:
            row["train_time_s"] = round(res["train_time"], 2)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    return df


def save_results(results, comparison_df):
    """
    Save evaluation results to JSON and CSV.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Metrics JSON
    metrics_dict = {}
    for name, res in results.items():
        entry = {**res["metrics"]}
        if "best_params" in res:
            entry["best_params"] = res["best_params"]
        if "train_time" in res:
            entry["train_time_s"] = round(res["train_time"], 2)
        metrics_dict[name] = entry

    with open(RESULTS_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2, default=str)

    # Comparison CSV
    comparison_df.to_csv(RESULTS_DIR / "comparison_table.csv")

    print(f"[evaluate] Saved test_metrics.json")
    print(f"[evaluate] Saved comparison_table.csv")
