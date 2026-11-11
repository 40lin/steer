"""Portable entry points for checks, a tiny smoke run, and the complete study."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
OUT = ROOT / 'experiments' / 'dynamic_threshold'


def run(script, *args):
    command = [sys.executable, str(CODE / script), *map(str, args)]
    print('Running:', ' '.join(command), flush=True)
    environment = os.environ.copy()
    environment['PYTHONIOENCODING'] = 'utf-8'
    with subprocess.Popen(command, cwd=ROOT, env=environment,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace',
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


def check():
    # Existing scientific checks cover physical allocation, payoff accounting,
    # public-only inference, GRU replay, attack selection and paired statistics.
    for script, args in [
        ('validate_micro_model.py', ()),
        ('steering_env.py', ('--self-test',)),
        ('belief_baselines.py', ('--self-test',)),
        ('schedule_baselines.py', ('self-test',)),
        ('adaptive_coalition_attack.py', ('--self-test',)),
        ('analyze_threshold_experiment.py', ('--self-test',)),
    ]:
        run(script, *args)
    print('All model, inference, attack and statistical self-checks passed.', flush=True)


def smoke():
    sys.path.insert(0, str(CODE))
    import numpy as np
    import run_threshold_experiment as experiment
    experiment.OUT = ROOT / 'experiments' / 'smoke'
    experiment.OUT.mkdir(parents=True, exist_ok=True)
    experiment.baseline_split('validation', worlds=2, seed=300017)
    experiment.train('belief', 0, steps=6400, batch=16, epochs=1,
                     validation_worlds=2)
    experiment.evaluate_saved('belief', 0, split='smoke_test', worlds=2,
                              stream_seed=801027)
    path = experiment.OUT / 'data' / 'smoke_test' / 'belief_0.npz'
    with np.load(path) as data:
        assert data['metrics'].shape == (32, 8)
        assert np.isfinite(data['metrics']).all()
        np.testing.assert_allclose(data['phase_totals'].sum(axis=1),
                                   data['metrics'][:, :2], atol=1e-8)
        assert np.all(data['actions'][20:] == data['actions'][19])
    summary = {'status': 'passed', 'training_market_steps': 6400,
               'test_episodes': 32, 'output': str(path.relative_to(ROOT)),
               'scientific_scope': 'Pipeline check only; not a reproduction of reported effects.'}
    (experiment.OUT / 'smoke_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


def full(workers):
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit('experiments/dynamic_threshold already contains a run. '
                         'Archive it or use a fresh checkout before starting a new study; '
                         'this command never overwrites a frozen experiment.')
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ('protocol_audit.md', 'implementation_addendum.json'):
        shutil.copy2(ROOT / 'docs' / name, OUT / name)
    provenance = {'started_utc': datetime.now(timezone.utc).isoformat(),
                  'source': 'Fresh run of the published implementation',
                  'reference_results': 'results/reference are the separate historical run',
                  'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(CODE.glob('*.py'))}}
    (OUT / 'reproduction_run.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    run('validate_micro_model.py')
    run('run_threshold_experiment.py', 'baselines', '--split', 'validation',
        '--worlds', 100, '--stream-seed', 300017)
    run('schedule_baselines.py', 'validation')
    run('adaptive_coalition_attack.py', '--prepare')
    run('analyze_threshold_experiment.py', '--freeze')
    run('run_training_grid.py', '--workers', workers)
    run('run_evaluation_grid.py')
    run('analyze_threshold_experiment.py', '--run')
    run('stress_batch_and_summary.py', 'all', '--workers', min(workers, 3))
    for seed in (0, 4, 9):
        run('adaptive_coalition_attack.py', '--seed', seed)
    run('exact_threshold_stress.py')
    print('Full experiment completed. Results: experiments/dynamic_threshold/analysis/test; '
          'attacks: experiments/adaptive_coalition_attack', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'smoke', 'full'))
    parser.add_argument('--workers', type=int, default=6,
                        help='Training processes for full; stress uses at most three.')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    # Bound BLAS thread pools before importing numerical libraries in any child.
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ.setdefault(variable, '1')
    if args.command == 'check':
        check()
    elif args.command == 'smoke':
        smoke()
    else:
        full(args.workers)


if __name__ == '__main__':
    main()
