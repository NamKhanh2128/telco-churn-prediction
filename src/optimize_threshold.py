"""
Cost-based threshold optimization.

Thay vì dùng ngưỡng mặc định 0.5, tìm ngưỡng tối ưu chi phí business:
- FP cost = $50  (gửi retention offer cho KH không churn)
- FN cost = $500 (bỏ sót KH sắp churn → mất CLV)

Bài Audit chỉ ra: ngưỡng tối ưu chi phí thấp hơn ngưỡng tối ưu F1,
tiết kiệm ~$77,646 trên mỗi 1,000 khách hàng.

NOTE: Threshold KHÔNG ĐƯỢC hard-code.
      Sau khi tìm được, lưu vào models/config.json để API đọc.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from src.config import COST_FP, COST_FN, COST_ACTION, MODELS_DIR, FIGURES_DIR


def find_cost_optimal_threshold(
    y_true,
    y_proba,
    cost_fp=None,
    cost_fn=None,
    cost_mode='classification',
    cost_action=None,
    thresholds=None,
    verbose=True,
):
    """
    Tìm ngưỡng quyết định tối ưu theo chi phí business.

    Parameters
    ----------
    y_true : array-like
        True labels (0/1)
    y_proba : array-like
        Predicted probabilities for class 1 (Churn)
    cost_fp : float
        Cost of False Positive (retention offer wasted)
    cost_fn : float
        Cost of False Negative (customer lost)
    verbose : bool
        Print a short summary (pass False when calling it in a loop).

    Returns
    -------
    dict
        best_threshold, best_cost, default_cost, savings, all thresholds/costs
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError('y_true and y_proba must have the same length')
    if not np.isfinite(y_proba).all():
        raise ValueError('y_proba contains NaN or infinite values')
    if cost_mode not in {'classification', 'action'}:
        raise ValueError('cost_mode must be classification or action')

    cost_fp = COST_FP if cost_fp is None else cost_fp
    cost_fn = COST_FN if cost_fn is None else cost_fn
    cost_action = COST_ACTION if cost_action is None else cost_action

    if thresholds is None:
        thresholds = np.unique(np.r_[0.0, np.clip(y_proba, 0.0, 1.0), 1.0])
    else:
        thresholds = np.unique(np.asarray(thresholds, dtype=float))
    costs = []
    f1_scores = []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        fp = ((y_pred_t == 1) & (y_true == 0)).sum()
        fn = ((y_pred_t == 0) & (y_true == 1)).sum()
        if cost_mode == 'action':
            total_cost = y_pred_t.sum() * cost_action + fn * cost_fn
        else:
            total_cost = fp * cost_fp + fn * cost_fn
        costs.append(total_cost)
        f1_scores.append(f1_score(y_true, y_pred_t, pos_label=1, zero_division=0))

    costs = np.array(costs)
    f1_scores = np.array(f1_scores)

    # Cost-optimal threshold
    best_cost_idx = np.argmin(costs)
    best_threshold = thresholds[best_cost_idx]
    best_cost = costs[best_cost_idx]

    # F1-optimal threshold (for comparison)
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = thresholds[best_f1_idx]

    # Default threshold = 0.5
    y_pred_default = (y_proba >= 0.5).astype(int)
    fp_default = ((y_pred_default == 1) & (y_true == 0)).sum()
    fn_default = ((y_pred_default == 0) & (y_true == 1)).sum()
    if cost_mode == 'action':
        default_cost = y_pred_default.sum() * cost_action + fn_default * cost_fn
    else:
        default_cost = fp_default * cost_fp + fn_default * cost_fn

    savings = default_cost - best_cost

    result = {
        'cost_mode': cost_mode,
        'cost_action': cost_action,
        "best_threshold": round(float(best_threshold), 4),
        "best_cost": float(best_cost),
        "default_cost": float(default_cost),
        "savings": float(savings),
        "best_f1_threshold": round(float(best_f1_threshold), 4),
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "thresholds": thresholds,
        "costs": costs,
        "f1_scores": f1_scores,
    }

    if verbose:
        # ASCII only, so it also prints in consoles without UTF-8 (Windows cp1252).
        rule = '-' * 34
        print(f"\n{'=' * 55}")
        print("  COST-BASED THRESHOLD OPTIMIZATION")
        print(f"{'=' * 55}")
        print(f"  Cost FP (retention offer):  ${cost_fp}")
        print(f"  Cost FN (lost customer):    ${cost_fn}")
        print(f"  {rule}")
        print(f"  Default threshold (0.5):    ${default_cost:,.0f}")
        print(f"  Optimal threshold ({best_threshold:.3f}):  ${best_cost:,.0f}")
        print(f"  F1-optimal threshold:       {best_f1_threshold:.3f}")
        print(f"  {rule}")
        print(f"  SAVINGS: ${savings:,.0f}")
        print(f"{'=' * 55}\n")

    return result


def plot_threshold_analysis(result, save=True):
    """
    Plot cost vs threshold and F1 vs threshold side by side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2))

    thresholds = result["thresholds"]
    costs = result["costs"]
    f1_scores = result["f1_scores"]
    best_t = result["best_threshold"]
    best_f1_t = result["best_f1_threshold"]

    # ─── Cost plot ───────────────────────────────────
    ax1.plot(thresholds, costs, "b-", linewidth=2)
    ax1.axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="Default (0.5)")
    ax1.axvline(x=best_t, color="green", linestyle="--", alpha=0.7, label=f"Cost-optimal ({best_t:.3f})")
    ax1.set_xlabel("Threshold", fontsize=12)
    ax1.set_ylabel("Total Cost ($)", fontsize=12)
    ax1.set_title("Cost vs Decision Threshold", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ─── F1 plot ─────────────────────────────────────
    ax2.plot(thresholds, f1_scores, "orange", linewidth=2)
    ax2.axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="Default (0.5)")
    ax2.axvline(x=best_f1_t, color="green", linestyle="--", alpha=0.7, label=f"F1-optimal ({best_f1_t:.3f})")
    ax2.axvline(x=best_t, color="blue", linestyle=":", alpha=0.7, label=f"Cost-optimal ({best_t:.3f})")
    ax2.set_xlabel("Threshold", fontsize=12)
    ax2.set_ylabel("F1-score (Churn)", fontsize=12)
    ax2.set_title("F1 vs Decision Threshold", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    best_f1_index = int(np.argmax(f1_scores))
    default_index = int(np.argmin(np.abs(thresholds - 0.5)))
    best_f1 = float(f1_scores[best_f1_index])
    best_cost = result['best_cost']
    default_cost = result['default_cost']
    savings = result['savings']
    cost_mode = result['cost_mode']

    ax1.scatter(
        [best_t],
        [best_cost],
        color='green',
        s=42,
        zorder=5,
    )
    ax1.annotate(
        f'Cost-optimal\n t={best_t:.4f}\n USD {best_cost:,.0f}',
        xy=(best_t, best_cost),
        xytext=(12, -42),
        textcoords='offset points',
        fontsize=9,
        bbox={'boxstyle': 'round,pad=0.3', 'facecolor': '#e8f5e9'},
        arrowprops={'arrowstyle': '->', 'color': 'green'},
    )
    ax1.scatter(
        [0.5],
        [default_cost],
        color='red',
        s=30,
        zorder=5,
    )

    ax2.scatter(
        [best_f1_t],
        [f1_scores[best_f1_index]],
        color='green',
        s=42,
        zorder=5,
    )
    ax2.annotate(
        f'F1-optimal\n t={best_f1_t:.4f}\n F1={best_f1:.3f}',
        xy=(best_f1_t, f1_scores[best_f1_index]),
        xytext=(12, 10),
        textcoords='offset points',
        fontsize=9,
        bbox={'boxstyle': 'round,pad=0.3', 'facecolor': '#e8f5e9'},
        arrowprops={'arrowstyle': '->', 'color': 'green'},
    )
    ax2.scatter(
        [0.5],
        [f1_scores[default_index]],
        color='red',
        s=30,
        zorder=5,
    )

    fig.text(
        0.5,
        0.018,
        (
            f'Default: t=0.500 | cost=USD {default_cost:,.0f}    '
            f'Cost-optimal: t={best_t:.4f} | cost=USD {best_cost:,.0f} | '
            f'savings=USD {savings:,.0f}    '
            f'F1-optimal: t={best_f1_t:.4f} | F1={best_f1:.3f}    '
            f'mode={cost_mode}'
        ),
        ha='center',
        va='bottom',
        fontsize=9,
        family='monospace',
        bbox={
            'boxstyle': 'round,pad=0.45',
            'facecolor': '#f7f7f7',
            'edgecolor': '#cccccc',
        },
    )
    plt.tight_layout(rect=[0.02, 0.13, 0.98, 0.96])
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "cost_threshold.png", dpi=150, bbox_inches="tight")
        print(f"[threshold] Saved cost_threshold.png")
    plt.close(fig)


def save_model_config(threshold_result, model_name="random_forest"):
    """
    Save threshold + cost config to models/config.json for API to load.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        'cost_mode': threshold_result.get('cost_mode', 'classification'),
        'cost_action': threshold_result.get('cost_action', COST_ACTION),
        'threshold_source': 'validation',
        "model": model_name,
        "threshold": threshold_result["best_threshold"],
        "cost_fp": threshold_result["cost_fp"],
        "cost_fn": threshold_result["cost_fn"],
        "savings_vs_default": threshold_result["savings"],
    }

    config_path = MODELS_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[threshold] Saved model config to {config_path}")
    return config
