"""Regenerate every reported artifact with one command.

Usage:
    python -m src.reproduce            # full final protocol (tuned, ~15-30 min)
    python -m src.reproduce --quick    # untuned smoke run; snapshot goes to
                                       # results/results_snapshot_quick.md (ignored)

Steps (in order):
    1. pytest                                    — abort if any test fails
    2. src.train --tune --with-shap              — IBM: 7 models + calibrated RF + SHAP
    3. src.compare_datasets --dataset all        — IBM / Orange / Iranian benchmark
    4. src.compare_datasets --dataset ibm --keep-total-charges   — ablation A1
    5. src.iranian_audit                         — duplicate audit + 10-seed stability
    6. src.ablation                              — A2 (SMOTE/class_weight), A3 (scaling)
    7. src.model_stability                       — all models × 10 seeds + tests vs RF
    8. src.calibration_analysis                  — calibration quality vs model size
    9. src.rf_analysis                           — validation curves, OOB, importance, SHAP

Writes results/reproduce_log.json with the git commit, timestamps and the exact
commands, then exports reports/results_snapshot.md (src.export_results), so
every number in the report can be traced to one run.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

from src.config import PROJECT_ROOT, RESULTS_DIR


def _git(*args):
    try:
        return subprocess.run(
            ['git', *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--quick', action='store_true', help='Skip tuning.')
    args = parser.parse_args()
    tune = [] if args.quick else ['--tune']
    quick = ['--quick'] if args.quick else []

    steps = [
        ['-m', 'pytest', '-q'],
        ['-m', 'src.train', *tune, '--with-shap'],
        ['-m', 'src.compare_datasets', '--dataset', 'all', *tune],
        ['-m', 'src.compare_datasets', '--dataset', 'ibm', '--keep-total-charges', *tune],
        ['-m', 'src.iranian_audit'],
        # Analyses below reuse the tuned parameters written by compare_datasets.
        ['-m', 'src.ablation', *quick],
        ['-m', 'src.model_stability', *quick],
        ['-m', 'src.calibration_analysis'],
        ['-m', 'src.rf_analysis', *quick],
    ]

    log = {
        'started_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'git_commit': _git('rev-parse', 'HEAD'),
        'git_dirty': bool(_git('status', '--porcelain')),
        'python': sys.version.split()[0],
        'mode': 'quick' if args.quick else 'final',
        'steps': [],
    }

    for step in steps:
        command = [sys.executable, *step]
        print('\n' + '#' * 70 + f'\n# {" ".join(step)}\n' + '#' * 70, flush=True)
        start = time.time()
        code = subprocess.run(command, cwd=PROJECT_ROOT).returncode
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'reproduce_log.json', 'w', encoding='utf-8') as handle:
        json.dump(log, handle, indent=2)
    print(f"\nreproduce: {log['status']} -> {RESULTS_DIR / 'reproduce_log.json'}")
    if log['status'] == 'completed':
        from src.export_results import main as export_results
        export_results()
    sys.exit(0 if log['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
