"""Regenerate every reported artifact with one command.

Usage:
    python -m src.reproduce            # full final protocol (tuned, ~30-40 min)
    python -m src.reproduce --quick    # untuned smoke run in a sandbox folder

Steps (in order):
    1. pytest                                    — abort if any test fails
    2. src.train --tune --with-shap              — IBM: 7 models + calibrated RF + SHAP
    3. src.compare_datasets --dataset all --tune — IBM / Orange / Iranian benchmark;
                                                   writes models/tuned_params.json
    4. src.compare_datasets --dataset ibm --keep-total-charges --tune
       --models LogReg RandomForest              — ablation A1 (own run folder)
    5. src.iranian_audit                         — duplicate audit + 10-seed stability
    6. src.ablation                              — A2 (SMOTE/class_weight), A3 (scaling)
    7. src.model_stability                       — all models x 10 seeds + tests vs RF
    8. src.calibration_analysis                  — calibration quality vs model size
    9. src.rf_analysis                           — validation curves, OOB, importance, SHAP
   10. optional modules of the other members, run only when their file exists
       (see OPTIONAL_STEPS); a module receives --quick in quick mode when its
       source mentions "--quick".
Then src.export_results writes the snapshot.

Quick mode sets CHURN_OUTPUT_ROOT=results/quick for every step, so all its
outputs (models, figures, results, results_snapshot_quick.md) stay under the
git-ignored results/quick/ and no tracked file is touched. Its analyses reuse
the official tuned parameters in models/tuned_params.json.

Before the first step, the tables that merge rows across runs (MERGED_TABLES)
are deleted, so the snapshot contains exactly what this run produced.

The log (results/reproduce_log.json) records the git commit, whether the tree
was dirty, library versions, the SHA-256 of the three datasets and every
command with its exit code and duration, so each number in the snapshot can be
traced to one run.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from src.config import (
    DATASET_PATH,
    IRANIAN_DATASET_PATH,
    ORANGE_DATASET_PATH,
    PROJECT_ROOT,
)
from src.data_loader import get_file_sha256

QUICK_OUTPUT_ROOT = PROJECT_ROOT / 'results' / 'quick'

# Analyses owned by other members, hooked in as soon as their module exists:
# (module, short description). Contract: `python -m <module>` runs it; it takes
# its output folders from src.config (RESULTS_DIR / '<name>', FIGURES_DIR /
# '<name>' — never hard-coded 'results/...' paths, so quick runs stay in their
# sandbox) and may write RESULTS_DIR / '<name>' / 'snapshot.md', which
# src.export_results appends to the snapshot.
OPTIONAL_STEPS = [
    ('src.policy_analysis', 'cost / top-k threshold policy (Nhat Linh)'),
]

# Tables that merge rows across runs. A reproduce run starts them empty, so the
# snapshot shows exactly the rows this run produced (no leftovers of manual or
# partial runs such as iranian_row_wise or an older A1 run with more models).
MERGED_TABLES = [
    Path('dataset_runs') / 'dataset_comparison.csv',
    Path('dataset_runs') / 'dataset_comparison.json',
    Path('dataset_runs') / 'model_comparison_long.csv',
    Path('rf_analysis') / 'rf_summary.csv',
]

DATASET_FILES = {
    'ibm': DATASET_PATH,
    'orange': ORANGE_DATASET_PATH,
    'iranian': IRANIAN_DATASET_PATH,
}

LIBRARIES = ['numpy', 'pandas', 'scikit-learn', 'imbalanced-learn', 'shap', 'scipy', 'matplotlib']


def _git(*args):
    try:
        return subprocess.run(
            ['git', *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _module_path(module):
    return PROJECT_ROOT.joinpath(*module.split('.')).with_suffix('.py')


def optional_steps(quick):
    '''Commands for the optional modules that exist in this checkout.'''
    steps = []
    for module, _ in OPTIONAL_STEPS:
        path = _module_path(module)
        if not path.exists():
            continue
        accepts_quick = '--quick' in path.read_text(encoding='utf-8', errors='ignore')
        steps.append(['-m', module, *(['--quick'] if quick and accepts_quick else [])])
    return steps


def build_steps(quick):
    '''Ordered list of `python ...` argument lists for one run.'''
    tune = [] if quick else ['--tune']
    quick_flag = ['--quick'] if quick else []
    return [
        ['-m', 'pytest', '-q'],
        ['-m', 'src.train', *tune, '--with-shap'],
        ['-m', 'src.compare_datasets', '--dataset', 'all', *tune],
        ['-m', 'src.compare_datasets', '--dataset', 'ibm', '--keep-total-charges', *tune,
         '--models', 'LogReg', 'RandomForest'],
        ['-m', 'src.iranian_audit'],
        # Analyses below reuse the tuned parameters written by compare_datasets.
        ['-m', 'src.ablation', *quick_flag],
        ['-m', 'src.model_stability', *quick_flag],
        ['-m', 'src.calibration_analysis'],
        ['-m', 'src.rf_analysis', *quick_flag],
        *optional_steps(quick),
    ]


def _versions():
    versions = {'python': sys.version.split()[0]}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _dataset_hashes():
    return {
        name: (get_file_sha256(path) if Path(path).exists() else None)
        for name, path in DATASET_FILES.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--quick', action='store_true',
        help='Untuned smoke run; outputs go to results/quick/ (nothing tracked changes).',
    )
    args = parser.parse_args()

    env = os.environ.copy()
    if args.quick:
        env['CHURN_OUTPUT_ROOT'] = str(QUICK_OUTPUT_ROOT)
        results_dir = QUICK_OUTPUT_ROOT / 'results'
    else:
        # The final run always writes the official locations, even if the
        # calling shell had a sandbox set.
        env.pop('CHURN_OUTPUT_ROOT', None)
        results_dir = PROJECT_ROOT / 'results'

    for table in MERGED_TABLES:
        (results_dir / table).unlink(missing_ok=True)

    dirty = bool(_git('status', '--porcelain'))
    if dirty and not args.quick:
        print('WARNING: the working tree has uncommitted changes; the snapshot will say so. '
              'Official numbers should come from a clean commit.', flush=True)

    log = {
        'started_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'git_commit': _git('rev-parse', 'HEAD'),
        'git_dirty': dirty,
        'python': sys.version.split()[0],
        'versions': _versions(),
        'dataset_sha256': _dataset_hashes(),
        'mode': 'quick' if args.quick else 'final',
        'output_root': env.get('CHURN_OUTPUT_ROOT', str(PROJECT_ROOT)),
        'steps': [],
    }

    for step in build_steps(args.quick):
        command = [sys.executable, *step]
        print('\n' + '#' * 70 + f'\n# {" ".join(step)}\n' + '#' * 70, flush=True)
        start = time.time()
        code = subprocess.run(command, cwd=PROJECT_ROOT, env=env).returncode
        log['steps'].append({
            'command': 'python ' + ' '.join(step),
            'exit_code': code,
            'seconds': round(time.time() - start, 1),
        })
        if code != 0:
            log['status'] = 'failed'
            break
    else:
        log['status'] = 'completed'

    log['finished_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / 'reproduce_log.json'
    with open(log_path, 'w', encoding='utf-8') as handle:
        json.dump(log, handle, indent=2)
    print(f"\nreproduce: {log['status']} -> {log_path}")
    if log['status'] == 'completed':
        # Same environment as the steps, so the snapshot reads this run's outputs.
        code = subprocess.run(
            [sys.executable, '-m', 'src.export_results'], cwd=PROJECT_ROOT, env=env,
        ).returncode
        if code != 0:
            log['status'] = 'export_failed'
    sys.exit(0 if log['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
