"""Tests for the experiment framework and analysis scripts (no real data needed)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from src.config import CALIBRATE_PROBABILITIES
from src.preprocessing import build_model_pipeline, build_preprocessor

NUMERIC = ['tenure', 'MonthlyCharges']
CATEGORICAL = ['Contract', 'TechSupport']


def _toy(n=240, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        'tenure': rng.integers(0, 72, n),
        'MonthlyCharges': rng.uniform(20, 110, n).round(2),
        'Contract': rng.choice(['Month-to-month', 'One year', 'Two year'], n),
        'TechSupport': rng.choice(['Yes', 'No'], n),
    })
    logit = (
        -1.0 + 0.03 * X['MonthlyCharges'] - 0.05 * X['tenure']
        + 1.5 * (X['Contract'] == 'Month-to-month')
    )
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), name='Churn')
    return X, y


def _rf_pipeline(n_estimators=25):
    return build_model_pipeline(
        build_preprocessor(NUMERIC, CATEGORICAL, scale_numeric=False),
        RandomForestClassifier(n_estimators=n_estimators, random_state=0),
    )


# ─── Benchmark framework (src.compare_datasets) ───────────────

def test_grid_keys_follow_the_pipeline():
    from src.compare_datasets import _build_model, _grid_for_pipeline

    logreg = _build_model('LogReg', NUMERIC, CATEGORICAL, False)
    expected = 'classifier__estimator__C' if CALIBRATE_PROBABILITIES else 'classifier__C'
    assert set(_grid_for_pipeline('LogReg', logreg)) == {expected}

    forest = _build_model('RandomForest', NUMERIC, CATEGORICAL, False)
    rf_grid = _grid_for_pipeline('RandomForest', forest)
    assert rf_grid and all(
        key.startswith('classifier__') and 'estimator__' not in key for key in rf_grid
    )
    assert _grid_for_pipeline('Dummy', forest) is None


def test_unknown_benchmark_model_is_an_error():
    from src.compare_datasets import _build_model

    with pytest.raises(ValueError, match='Unknown model'):
        _build_model('NotAModel', NUMERIC, CATEGORICAL, False)


def test_long_table_replaces_rows_of_rerun_datasets(tmp_path, monkeypatch):
    import src.compare_datasets as compare

    monkeypatch.setattr(compare, 'RESULTS_DIR', tmp_path)
    compare.update_long_table([
        {'dataset': 'a', 'model': 'RandomForest', 'f1_churn': 0.5},
        {'dataset': 'a', 'model': 'Dummy', 'f1_churn': 0.0},
    ])
    compare.update_long_table([{'dataset': 'b', 'model': 'LogReg', 'f1_churn': 0.4}])
    table = compare.update_long_table([{'dataset': 'a', 'model': 'RandomForest', 'f1_churn': 0.6}])

    assert set(table['dataset']) == {'a', 'b'}
    rows_a = table[table['dataset'] == 'a']
    assert list(rows_a['model']) == ['RandomForest']
    assert rows_a['f1_churn'].iloc[0] == 0.6


def test_split_seed_controls_the_split():
    from src.compare_datasets import _split_data

    X, y = _toy()
    first = _split_data(X, y, dataset_name='ibm', seed=1)
    again = _split_data(X, y, dataset_name='ibm', seed=1)
    other = _split_data(X, y, dataset_name='ibm', seed=2)
    assert list(first[2].index) == list(again[2].index)
    assert list(first[2].index) != list(other[2].index)
    assert len(first[0]) + len(first[1]) + len(first[2]) == len(X)


def test_feature_groups_mark_duplicate_rows():
    from src.compare_datasets import feature_groups

    X = pd.DataFrame({'a': [1, 1, 2], 'b': ['x', 'x', 'y']})
    groups = feature_groups(X)
    assert groups[0] == groups[1] != groups[2]


# ─── Ablation (src.ablation) ──────────────────────────────────

def test_smote_variant_resamples_only_during_fit():
    from src.ablation import build_variant

    X, y = _toy()
    pipeline = build_variant(
        NUMERIC, CATEGORICAL, is_ibm=False, class_weight=None,
        use_smote=True, scale_numeric=False, seed=0,
    )
    assert 'smote' in pipeline.named_steps
    pipeline.fit(X.iloc[:180], y.iloc[:180])
    proba = pipeline.predict_proba(X.iloc[180:])[:, 1]
    assert proba.shape == (60,)
    assert np.all((proba >= 0) & (proba <= 1))


# ─── Stability tests (src.model_stability) ────────────────────

def test_mcnemar_counts_discordant_pairs():
    from src.model_stability import mcnemar_exact

    y_true = np.array([1, 1, 0, 0, 1])
    pred_a = np.array([1, 1, 0, 1, 0])
    pred_b = np.array([0, 1, 0, 0, 0])
    b, c, p_value = mcnemar_exact(y_true, pred_a, pred_b)
    assert (b, c) == (1, 1)
    assert p_value == pytest.approx(1.0)


def test_wilcoxon_on_identical_scores_is_not_significant():
    from src.model_stability import paired_wilcoxon

    statistic, p_value = paired_wilcoxon([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
    assert statistic is None
    assert p_value == 1.0


# ─── Explanations (src.explain) ───────────────────────────────

@pytest.mark.parametrize('ensemble', [True, False])
def test_explain_customer_works_on_calibrated_forest(ensemble):
    from src.explain import explain_customer

    X, y = _toy()
    model = CalibratedClassifierCV(
        _rf_pipeline(), method='sigmoid', cv=3, ensemble=ensemble,
    ).fit(X, y)
    factors = explain_customer(model, X.iloc[[0]], top_n=3)

    assert len(factors) == 3
    contributions = [factor['contribution'] for factor in factors]
    assert contributions == sorted(contributions, key=abs, reverse=True)
    for factor in factors:
        assert factor['feature'] in NUMERIC + CATEGORICAL
        expected = 'increases_churn' if factor['contribution'] > 0 else 'decreases_churn'
        assert factor['direction'] == expected
        assert factor['impact'] in {'high', 'medium', 'low'}


def test_input_features_lists_preprocessor_columns():
    from src.explain import input_features

    X, y = _toy()
    pipeline = _rf_pipeline().fit(X, y)
    assert input_features(pipeline) == NUMERIC + CATEGORICAL


# ─── Calibration and RF analysis helpers ──────────────────────

def test_model_size_is_measured():
    from src.calibration_analysis import model_size_mb

    X, y = _toy()
    raw, compressed = model_size_mb(_rf_pipeline(n_estimators=10).fit(X, y))
    assert raw > 0 and compressed > 0


def test_pick_cases_finds_tp_fn_fp():
    from src.rf_analysis import pick_cases

    cases = pick_cases(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1]))
    assert list(cases.values()) == [0, 1, 2]
