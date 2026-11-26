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


# ─── Contract for every benchmark model (covers models added later) ────

def _contract_frame(dataset_name, n=200, seed=0):
    """Small synthetic frame with the columns each dataset's pipeline expects."""
    rng = np.random.default_rng(seed)
    if dataset_name == 'ibm':
        X = pd.DataFrame({
            'gender': rng.choice(['Male', 'Female'], n),
            'SeniorCitizen': rng.integers(0, 2, n),
            'Partner': rng.choice(['Yes', 'No'], n),
            'tenure': rng.integers(0, 72, n),
            'PhoneService': rng.choice(['Yes', 'No'], n),
            'InternetService': rng.choice(['DSL', 'Fiber optic', 'No'], n),
            'OnlineSecurity': rng.choice(['Yes', 'No'], n),
            'TechSupport': rng.choice(['Yes', 'No'], n),
            'StreamingTV': rng.choice(['Yes', 'No'], n),
            'Contract': rng.choice(['Month-to-month', 'One year', 'Two year'], n),
            'PaymentMethod': rng.choice(['Electronic check', 'Mailed check'], n),
            'MonthlyCharges': rng.uniform(20, 110, n).round(2),
        })
        score = 0.04 * X['MonthlyCharges'] - 0.06 * X['tenure'] + 1.5 * (X['Contract'] == 'Month-to-month')
    else:
        X = pd.DataFrame({
            'Calls': rng.integers(0, 10, n),
            'Minutes': rng.uniform(0, 300, n).round(1),
            'Plan': rng.choice(['yes', 'no'], n),
        })
        score = 0.4 * X['Calls'] + 0.01 * X['Minutes'] - 3
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-score))).astype(int), name='Churn')
    return X, y


@pytest.mark.parametrize('dataset_name', ['ibm', 'orange'])
def test_every_benchmark_model_fits_and_returns_probabilities(dataset_name):
    from src.compare_datasets import BENCHMARK_MODELS, _build_model, feature_lists

    X, y = _contract_frame(dataset_name)
    numeric, categorical = feature_lists(X, dataset_name)
    for name in BENCHMARK_MODELS:
        pipeline = _build_model(name, numeric, categorical, is_ibm=(dataset_name == 'ibm'))
        pipeline.fit(X.iloc[:150], y.iloc[:150])
        proba = pipeline.predict_proba(X.iloc[150:])[:, 1]
        assert proba.shape == (50,), name
        assert np.all((proba >= 0) & (proba <= 1)), name


def test_every_tuning_grid_matches_its_pipeline():
    from src.compare_datasets import (
        BENCHMARK_MODELS, MODEL_PARAM_GRIDS, _build_model, _grid_for_pipeline,
    )

    assert set(MODEL_PARAM_GRIDS) <= set(BENCHMARK_MODELS)
    for name in BENCHMARK_MODELS:
        pipeline = _build_model(name, NUMERIC, CATEGORICAL, False)
        grid = _grid_for_pipeline(name, pipeline) or {}
        valid = pipeline.get_params()
        for key in grid:
            assert key in valid, f'{name}: grid key {key} is not a pipeline parameter'


# ─── Same IBM model in src.train and src.compare_datasets ─────

def test_ibm_features_match_between_train_and_compare(tmp_path, monkeypatch):
    """src.train and src.compare_datasets must build the same IBM inputs, in the same order."""
    import src.compare_datasets as compare
    from src.config import ENGINEERED_NUMERIC_FEATURES
    from src.data_loader import detect_feature_types, get_feature_target, load_raw_data

    X, y = _contract_frame('ibm', n=120)
    raw = X.copy()
    raw.insert(0, 'customerID', [f'C{i:04d}' for i in range(len(raw))])
    raw['TotalCharges'] = (raw['MonthlyCharges'] * raw['tenure']).round(2).astype(str)
    raw.loc[raw['tenure'] == 0, 'TotalCharges'] = ' '      # blank strings, as in the source
    raw['Churn'] = np.where(y == 1, 'Yes', 'No')
    path = tmp_path / 'ibm.csv'
    raw.to_csv(path, index=False)

    X_train_script, y_train_script = get_feature_target(load_raw_data(path))   # src.train route
    monkeypatch.setitem(compare.DATASET_PROFILES['ibm'], 'path', path)
    X_compare, y_compare, _, _ = compare.load_dataset('ibm')                  # benchmark route

    assert list(X_train_script.columns) == list(X_compare.columns)
    assert (np.asarray(y_train_script) == np.asarray(y_compare)).all()
    numeric, categorical = compare.feature_lists(X_compare, 'ibm')
    assert (numeric, categorical) == detect_feature_types(X_train_script, verbose=False)
    assert numeric == ['tenure', 'MonthlyCharges', 'SeniorCitizen', *ENGINEERED_NUMERIC_FEATURES]


# ─── Deployed model construction ──────────────────────────────

def test_deployed_model_does_not_keep_a_fitted_copy():
    from sklearn.utils.validation import check_is_fitted
    from sklearn.exceptions import NotFittedError
    from src.compare_datasets import make_deployed_model

    X, y = _toy()
    fitted_rf = _rf_pipeline().fit(X, y)
    model = make_deployed_model(fitted_rf, X, y).fit(X, y)
    if CALIBRATE_PROBABILITIES:
        with pytest.raises(NotFittedError):
            check_is_fitted(model.estimator.named_steps['classifier'])
    proba = model.predict_proba(X)[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))


def test_calibration_cv_keeps_duplicate_groups_together():
    from src.compare_datasets import calibration_cv, feature_groups
    from src.config import CV_FOLDS

    X, y = _toy(n=300)
    X = pd.concat([X, X.iloc[:60]], ignore_index=True)   # 60 exact duplicates
    y = pd.concat([y, y.iloc[:60]], ignore_index=True)
    groups = feature_groups(X)
    assert calibration_cv(X, y, None) == CV_FOLDS
    for train_idx, test_idx in calibration_cv(X, y, groups):
        assert not set(groups[train_idx]) & set(groups[test_idx])


# ─── Tuned parameters (models/tuned_params.json) ──────────────

def test_tuned_params_round_trip(tmp_path, monkeypatch):
    import src.compare_datasets as compare

    monkeypatch.setattr(compare, 'MODELS_DIR', tmp_path)
    monkeypatch.setattr(compare, 'OFFICIAL_TUNED_PARAMS_PATH', tmp_path / 'missing.json')
    assert compare.tuned_params('ibm', 'RandomForest') == {}

    compare.save_tuned_params('ibm', {
        'RandomForest': {'best_params': {'classifier__max_depth': 10, 'classifier__n_estimators': 500}},
        'LogReg': {'best_params': {'classifier__estimator__C': 0.1}},
        'Dummy': {'best_params': None},
    }, {'split_seed': 42})
    compare.save_tuned_params('orange', {
        'RandomForest': {'best_params': {'classifier__max_depth': None}},
    }, {'split_seed': 42})

    assert compare.tuned_params('ibm', 'RandomForest') == {'max_depth': 10, 'n_estimators': 500}
    assert compare.tuned_params('ibm', 'LogReg') == {'C': 0.1}
    assert compare.tuned_params('orange', 'RandomForest') == {'max_depth': None}


def test_partial_tuning_keeps_the_other_models(tmp_path, monkeypatch):
    import src.compare_datasets as compare

    monkeypatch.setattr(compare, 'MODELS_DIR', tmp_path)
    monkeypatch.setattr(compare, 'OFFICIAL_TUNED_PARAMS_PATH', tmp_path / 'missing.json')
    compare.save_tuned_params('ibm', {
        'LogReg': {'best_params': {'classifier__estimator__C': 1.0}},
        'RandomForest': {'best_params': {'classifier__max_depth': 10}},
        'DecisionTree': {'best_params': {'classifier__max_depth': 5}},
    }, {})
    compare.save_tuned_params('ibm', {                      # what `--tune --models KNN` saves
        'LogReg': {'best_params': {'classifier__estimator__C': 0.1}},
        'RandomForest': {'best_params': {'classifier__max_depth': 15}},
        'KNN': {'best_params': {'classifier__n_neighbors': 11}},
    }, {}, partial=True)
    assert compare.tuned_params('ibm', 'DecisionTree') == {'max_depth': 5}
    assert compare.tuned_params('ibm', 'KNN') == {'n_neighbors': 11}
    assert compare.tuned_params('ibm', 'RandomForest') == {'max_depth': 15}

    compare.save_tuned_params('ibm', {                      # full run: replaces the entry
        'RandomForest': {'best_params': {'classifier__max_depth': 5}},
    }, {})
    assert compare.tuned_params('ibm', 'DecisionTree') == {}


def test_sandbox_tuning_does_not_hide_official_params(tmp_path, monkeypatch):
    import json
    import src.compare_datasets as compare

    official = tmp_path / 'official.json'
    official.write_text(json.dumps({'runs': {
        'orange': {'RandomForest': {'max_depth': 15}},
        'ibm': {'RandomForest': {'max_depth': 10}, 'LogReg': {'C': 5.0}},
    }}), encoding='utf-8')
    sandbox = tmp_path / 'sandbox'
    sandbox.mkdir()
    monkeypatch.setattr(compare, 'MODELS_DIR', sandbox)
    monkeypatch.setattr(compare, 'OFFICIAL_TUNED_PARAMS_PATH', official)
    compare.save_tuned_params('ibm', {
        'RandomForest': {'best_params': {'classifier__max_depth': 5}},
    }, {}, partial=True)

    assert compare.tuned_params('orange', 'RandomForest') == {'max_depth': 15}
    assert compare.tuned_params('ibm', 'RandomForest') == {'max_depth': 5}     # sandbox wins
    assert compare.tuned_params('ibm', 'LogReg') == {'C': 5.0}                 # kept from official


def test_benchmark_model_applies_tuned_params(tmp_path, monkeypatch):
    import src.compare_datasets as compare

    monkeypatch.setattr(compare, 'MODELS_DIR', tmp_path)
    monkeypatch.setattr(compare, 'OFFICIAL_TUNED_PARAMS_PATH', tmp_path / 'missing.json')
    compare.save_tuned_params('orange', {
        'RandomForest': {'best_params': {'classifier__max_depth': 7}},
        'LogReg': {'best_params': {'classifier__estimator__C': 5.0}},
    }, {})
    X, _ = _contract_frame('orange')
    forest = compare.build_benchmark_model('RandomForest', X, 'orange')
    assert forest.named_steps['classifier'].max_depth == 7
    logreg = compare.build_benchmark_model('LogReg', X, 'orange')
    estimator = logreg.named_steps['classifier']
    C = estimator.estimator.C if CALIBRATE_PROBABILITIES else estimator.C
    assert C == 5.0


def test_run_names_keep_variants_apart():
    from src.compare_datasets import run_name_for

    assert run_name_for('ibm') == 'ibm'
    assert run_name_for('ibm', keep_total_charges=True) == 'ibm_with_total_charges'
    assert run_name_for('iranian', split_method='row_wise') == 'iranian_row_wise'
    assert run_name_for('iranian', split_method='group_aware') == 'iranian'


def test_long_table_pair_mode_keeps_other_models(tmp_path, monkeypatch):
    import src.compare_datasets as compare

    monkeypatch.setattr(compare, 'RESULTS_DIR', tmp_path)
    compare.update_long_table([
        {'dataset': 'a', 'model': 'RandomForest', 'f1_churn': 0.5},
        {'dataset': 'a', 'model': 'Dummy', 'f1_churn': 0.0},
    ])
    table = compare.update_long_table(
        [{'dataset': 'a', 'model': 'RandomForest', 'f1_churn': 0.7}], replace='pair',
    )
    rows = table.set_index('model')['f1_churn']
    assert rows['RandomForest'] == 0.7 and rows['Dummy'] == 0.0


# ─── Corrected resampled t-test ───────────────────────────────

def test_corrected_ttest_is_more_conservative_than_plain_ttest():
    from scipy.stats import ttest_rel
    from src.model_stability import corrected_resampled_ttest

    a = np.array([0.70, 0.72, 0.69, 0.74, 0.71, 0.73, 0.70, 0.72, 0.71, 0.73])
    b = a - np.array([0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.03, 0.01, 0.02, 0.02])
    statistic, p_value = corrected_resampled_ttest(a, b, test_train_ratio=0.15 / 0.70)
    plain = ttest_rel(a, b)
    assert statistic > 0
    assert p_value > plain.pvalue
    assert corrected_resampled_ttest(a, a, 0.2) == (None, 1.0)


# ─── Explanations of engineered features / legacy API helper ──

def test_explain_customer_names_source_columns_of_engineered_features():
    from src.compare_datasets import _build_model, feature_lists
    from src.explain import explain_customer
    from src.preprocessing import ENGINEERED_FEATURE_SOURCES

    X, y = _contract_frame('ibm')
    numeric, categorical = feature_lists(X, 'ibm')
    model = _build_model('RandomForest', numeric, categorical, is_ibm=True)
    model.set_params(classifier__n_estimators=20).fit(X, y)
    factors = explain_customer(model, X.iloc[[0]], top_n=40)
    by_feature = {factor['feature']: factor for factor in factors}
    for feature, factor in by_feature.items():
        expected = ENGINEERED_FEATURE_SOURCES.get(feature, [feature])
        assert factor['source_features'] == expected


def test_top_factors_helper_works_on_calibrated_model():
    from src.explain import get_top_factors_for_customer

    X, y = _toy()
    model = CalibratedClassifierCV(_rf_pipeline(), method='sigmoid', cv=3, ensemble=False).fit(X, y)
    factors = get_top_factors_for_customer(model, X.iloc[[0]], list(X.columns), top_n=2)
    assert len(factors) == 2
    assert {factor['feature'] for factor in factors} <= set(X.columns)
    assert all(set(factor) == {'feature', 'value', 'impact'} for factor in factors)


# ─── Reproduce / snapshot integration points ──────────────────

def test_optional_steps_run_only_when_the_module_exists(tmp_path, monkeypatch):
    import src.reproduce as reproduce

    monkeypatch.setattr(reproduce, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(reproduce, 'OPTIONAL_STEPS', [('src.extra_analysis', 'test')])
    assert reproduce.optional_steps(quick=True) == []

    (tmp_path / 'src').mkdir()
    module = tmp_path / 'src' / 'extra_analysis.py'
    module.write_text("parser.add_argument('--quick')\n", encoding='utf-8')
    assert reproduce.optional_steps(quick=True) == [['-m', 'src.extra_analysis', '--quick']]
    assert reproduce.optional_steps(quick=False) == [['-m', 'src.extra_analysis']]


def test_quick_run_never_tunes():
    import src.reproduce as reproduce

    quick = [' '.join(step) for step in reproduce.build_steps(quick=True)]
    final = [' '.join(step) for step in reproduce.build_steps(quick=False)]
    assert not any('--tune' in step for step in quick)
    assert any('src.compare_datasets --dataset all --tune' in step for step in final)


def test_sandboxed_export_never_writes_the_official_snapshot(tmp_path, monkeypatch):
    import pathlib
    import src.export_results as export

    written = []
    monkeypatch.setattr(export, 'RESULTS_DIR', tmp_path)
    monkeypatch.setattr(export, 'RUNS_DIR', tmp_path / 'dataset_runs')
    monkeypatch.setattr(export, 'QUICK_OUT_PATH', tmp_path / 'results_snapshot_quick.md')
    monkeypatch.setattr(export, 'SANDBOXED', True)
    monkeypatch.setattr(export, '_load_json', lambda path: {'split': {
        'train': 1, 'validation': 1, 'test': 1, 'random_state': 42}})
    for name in ['_ibm_model_table', '_policy_section', '_dataset_table']:
        monkeypatch.setattr(export, name, lambda: pd.DataFrame({'a': [1]}))
    monkeypatch.setattr(pathlib.Path, 'write_text', lambda self, *a, **k: written.append(self))
    export.main()                     # no reproduce log, but sandboxed -> never reports/
    assert written == [tmp_path / 'results_snapshot_quick.md']


def test_validation_curves_always_include_the_tuned_value():
    from src.rf_analysis import CURVES, _with_tuned

    curves = _with_tuned(
        {'classifier__max_depth': [3, 5, None], 'classifier__n_estimators': [50, 200]},
        {'classifier__max_depth': 10, 'classifier__n_estimators': 200},
    )
    assert curves['classifier__max_depth'] == [3, 5, 10, None]
    assert curves['classifier__n_estimators'] == [50, 200]
    from src.config import RF_PARAM_GRID
    for param, values in RF_PARAM_GRID.items():
        if param in CURVES:
            assert set(values) <= set(CURVES[param]), param


def test_stability_tests_label_threshold_dependent_rows():
    from src.model_stability import TEST_METRICS, _scope

    assert {'average_precision', 'roc_auc'} <= set(TEST_METRICS)
    assert _scope('average_precision') == _scope('roc_auc') == 'ranking (threshold-free)'
    assert _scope('f1_churn') == _scope('accuracy') == 'operating point @0.5'


def test_reproduce_clears_tables_merged_across_runs():
    import src.reproduce as reproduce

    names = {table.name for table in reproduce.MERGED_TABLES}
    assert {'model_comparison_long.csv', 'dataset_comparison.csv', 'rf_summary.csv'} <= names


def test_snapshot_appends_fragments_of_other_modules(tmp_path, monkeypatch):
    import src.export_results as export

    monkeypatch.setattr(export, 'RESULTS_DIR', tmp_path)
    assert export._fragment_section() == []
    (tmp_path / 'policy').mkdir()
    (tmp_path / 'policy' / 'snapshot.md').write_text('### Policy\n\n| a |\n|---|\n| 1 |\n', encoding='utf-8')
    section = '\n'.join(export._fragment_section())
    assert '## 10. Additional analyses' in section
    assert 'results/policy/snapshot.md' in section and '### Policy' in section
