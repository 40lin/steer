"""Run the four authorized frozen-policy stress cases and summarize them.

Waits for pretest_freeze.json before launching any evaluation. Three child
processes at most; each case's four fixed baselines precede its learned policies.
This driver never retrains or selects a platform policy from stress outcomes.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from analyze_threshold_experiment import crossed_bootstrap, estimate, write_json, write_csv, sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments' / 'dynamic_threshold'
CASES = ('exact', 'tabular', 'raw', 'longhold')
SEEDS = tuple(range(10))
BASE_METRICS = ('revenue', 'volume', 'completed_jobs', 'completed_value',
                'total_user_utility', 'active_buyers', 'underreport_count_active', 'overreport_count')
METRICS = BASE_METRICS + ('control_revenue', 'control_volume', 'hold_revenue', 'hold_volume')
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 78112301


def wait_for_freeze():
    freeze = OUT / 'pretest_freeze.json'
    start = time.monotonic()
    last_print = -60.
    while not freeze.exists():
        elapsed = time.monotonic() - start
        if elapsed - last_print >= 60:
            print(json.dumps({'status': 'waiting_for_pretest_freeze',
                              'elapsed_seconds': elapsed}), flush=True)
            last_print = elapsed
        time.sleep(2)
    # Parse as well as checking existence, so an incompletely written manifest
    # cannot silently authorize opening checkpoints/results.
    json.loads(freeze.read_text(encoding='utf-8'))
    print(json.dumps({'status': 'freeze_present', 'sha256': sha256(freeze)}), flush=True)
    return freeze


def run_job(case, seed):
    logdir = OUT / 'logs' / 'stress'
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f'{case}_{"fixed" if seed < 0 else "belief_"+str(seed)}.log'
    command = [sys.executable, str(Path(__file__).with_name('run_stress_evaluation.py')),
               case, '--seed', str(seed)]
    begin = time.monotonic()
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with log.open('w', encoding='utf-8') as handle:
        proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                              creationflags=flags, cwd=str(ROOT))
    record = {'case': case, 'seed': seed, 'exit_code': proc.returncode,
              'elapsed_seconds': time.monotonic()-begin, 'log': str(log)}
    print(json.dumps(record), flush=True)
    if proc.returncode:
        tail = log.read_text(encoding='utf-8')[-5000:]
        raise RuntimeError(f'Stress job failed: {record}\n{tail}')
    return record


def run_all(workers=3):
    freeze = wait_for_freeze()
    code_hashes = {name: sha256(Path(__file__).with_name(name)) for name in
                  ('stress_batch_and_summary.py', 'run_stress_evaluation.py',
                   'run_threshold_experiment.py', 'steering_env.py', 'belief_baselines.py')}
    start = datetime.now(timezone.utc).isoformat()
    records = []
    # Fixed baselines finish before any learned case is opened. All jobs use
    # stream1800021 and 100 worlds per original catalogue label in root code.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, case, -1) for case in CASES]
        for future in as_completed(futures):
            records.append(future.result())
        futures = [pool.submit(run_job, case, seed)
                   for seed in SEEDS for case in CASES]
        for future in as_completed(futures):
            records.append(future.result())
    changed = {name: {'before': digest, 'after': sha256(Path(__file__).with_name(name))}
               for name, digest in code_hashes.items()
               if sha256(Path(__file__).with_name(name)) != digest}
    manifest = {'status': 'all_authorized_stress_evaluations_completed',
                'started_utc': start, 'completed_utc': datetime.now(timezone.utc).isoformat(),
                'pretest_freeze_sha256': sha256(freeze), 'workers': workers,
                'stream_seed': 1800021, 'worlds_per_original_model_label': 100,
                'training_seeds': list(SEEDS), 'code_sha256_at_start': code_hashes,
                'code_changes_during_run': changed, 'jobs': records,
                'platform_retraining_or_selection_from_stress': False}
    write_json(OUT / 'stress_run_manifest.json', manifest)
    if changed:
        raise RuntimeError('Source changed during stress run; audit before combining results: '+str(changed))
    return manifest


def load_case_result(path, case, reference_models=None):
    with np.load(path, allow_pickle=False) as data:
        names = tuple(map(str, data['metric_names']))
        models = data['models'].astype(int)
        values = data['metrics'].astype(float)
        phase = data['phase_totals'].astype(float)
        trajectories = data['trajectories'].astype(float)
        actions = data['actions']
    if names != BASE_METRICS:
        raise ValueError(f'{path}: unexpected metrics')
    horizon = 1200 if case == 'longhold' else 400
    expected_labels = np.array([2,3,6,7,10,11,14,15]) if case == 'exact' else np.arange(16)
    expected_models = np.repeat(expected_labels, 100)
    if not np.array_equal(models, expected_models):
        raise ValueError(f'{path}: model/world order differs from frozen request')
    if reference_models is not None and not np.array_equal(models, reference_models):
        raise ValueError(f'{path}: invalid pairing')
    n = len(models)
    if values.shape != (n, 8) or phase.shape != (n, 2, 2):
        raise ValueError(f'{path}: wrong totals shape')
    if actions.shape != (horizon//10, n) or trajectories.shape != (horizon//10, n, 2):
        raise ValueError(f'{path}: wrong horizon')
    if not all(np.isfinite(x).all() for x in (values, phase, trajectories)):
        raise ValueError(f'{path}: nonfinite data')
    if np.any((actions < 0) | (actions > 3)):
        raise ValueError(f'{path}: invalid canonical actions')
    if not np.array_equal(actions[20:], np.broadcast_to(actions[19], actions[20:].shape)):
        raise ValueError(f'{path}: holding action changed')
    np.testing.assert_allclose(phase.sum(axis=1), values[:, :2], rtol=1e-10, atol=1e-7)
    np.testing.assert_allclose(trajectories.sum(axis=0)*10, values[:, :2], rtol=2e-6, atol=.01)
    np.testing.assert_allclose(values[:, 0]+values[:, 4], values[:, 3], rtol=1e-10, atol=1e-7)
    if np.any(values[:, :2] < -1e-9) or np.any(values[:, :2] > 5*horizon+1e-8):
        raise ValueError(f'{path}: impossible cash/volume')
    if np.any(values[:, 5] < 0) or np.any(values[:, 5] > 2*horizon):
        raise ValueError(f'{path}: impossible active count')
    if np.any(values[:, 6]+values[:, 7] > values[:, 5]+1e-8):
        raise ValueError(f'{path}: inconsistent reporting counts')
    augmented = np.column_stack((values, phase[:,0,0], phase[:,0,1], phase[:,1,0], phase[:,1,1]))
    if case == 'exact':
        # Learning-rate labels carry no behavioral distinction under exact
        # response. Their separate RNG rows are 200 independent worlds within
        # each of FOUR fixed markets, not eight kinds of economic evidence.
        grouped = np.stack([augmented[models//4 == market] for market in range(4)])
        assert grouped.shape == (4, 200, len(METRICS))
    else:
        grouped = augmented.reshape(16, 100, len(METRICS))
    return grouped, models


def summarize():
    freeze = OUT / 'pretest_freeze.json'
    if not freeze.exists():
        raise RuntimeError('Missing pretest freeze')
    validation = json.loads((OUT/'validation_selection.json').read_text(encoding='utf-8'))
    reference = f'fixed_{int(validation["best_fixed_action"])}'
    if reference != 'fixed_1':
        raise ValueError('Root requested frozen validation fixed=.5; manifest disagrees')
    methods = ('fixed_0', 'fixed_1', 'fixed_2', 'fixed_3', 'belief')
    summary = {'status': 'frozen_policy_stress_summary', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'pretest_freeze_sha256': sha256(freeze), 'reference': reference,
               'analysis_script_sha256': sha256(__file__), 'bootstrap_replicates': BOOTSTRAP_REPLICATES,
               'bootstrap_seed_base': BOOTSTRAP_SEED,
               'interpretation': 'All stress comparisons exploratory; no policy tuning from these results.',
               'cases': {}}
    rows, seed_rows, type_rows = [], [], []
    for index, case in enumerate(CASES):
        folder = OUT / 'data' / f'stress_{case}'
        arrays, hashes = {}, {}
        reference_models = None
        for method in methods:
            filenames = ([f'belief_{seed}.npz' for seed in SEEDS] if method == 'belief' else [method+'.npz'])
            pieces = []
            for filename in filenames:
                path = folder/filename
                x, ids = load_case_result(path, case, reference_models)
                if reference_models is None:
                    reference_models = ids
                pieces.append(x)
                hashes[filename] = sha256(path)
            arrays[method] = np.stack(pieces)
        boots = crossed_bootstrap(arrays, replicates=BOOTSTRAP_REPLICATES,
                                  seed=BOOTSTRAP_SEED+index)
        # Keep the full fixed catalogue as strata; train seed and world vary.
        if case == 'exact':
            groups = {'all': np.arange(4)}
            unit_note = '4 fixed physical markets, 200 worlds/market; learning-rate labels merged'
        else:
            groups = {'all': np.arange(16),
                      'independent': np.array([i for i in range(16) if (i//2)%2 == 0]),
                      'coalition': np.array([i for i in range(16) if (i//2)%2 == 1])}
            unit_note = '16 fixed catalogue strata, 100 worlds/stratum'
        points = {m: x.mean(axis=(0,2)) for m,x in arrays.items()}
        global_points = {m: p.mean(axis=0) for m,p in points.items()}
        posthoc = max(methods[:4], key=lambda m: global_points[m][0])
        feasible = [m for m in methods[:4] if global_points[m][1] >= .95*global_points['fixed_0'][1]]
        posthoc_feasible = max(feasible, key=lambda m: global_points[m][0])
        case_summary = {'horizon': 1200 if case == 'longhold' else 400,
                        'unit_note': unit_note, 'input_sha256': hashes,
                        'matched_candidate_replay': case in ('raw','tabular'),
                        'posthoc_best_fixed_diagnostic_only': posthoc,
                        'posthoc_best_service_feasible_fixed_diagnostic_only': posthoc_feasible,
                        'groups': {}}
        for group, positions in groups.items():
            gp = {m: points[m][positions].mean(axis=0) for m in methods}
            gb = {m: boots[m][:,positions].mean(axis=1) for m in methods}
            group_record = {'methods': {}, 'belief_comparisons': {}}
            for method in methods:
                statistics = {name: estimate(gp[method][j], gb[method][:,j])
                              for j,name in enumerate(METRICS)}
                statistics['service_slack'] = estimate(
                    gp[method][1]-.95*gp['fixed_0'][1], gb[method][:,1]-.95*gb['fixed_0'][:,1])
                group_record['methods'][method] = statistics
                contrast = estimate(gp[method][0]-gp[reference][0], gb[method][:,0]-gb[reference][:,0])
                rows.append({'case': case, 'group': group, 'method': method,
                             'horizon': case_summary['horizon'], 'unit_note': unit_note,
                             'revenue': statistics['revenue']['estimate'],
                             'revenue_ci95_low': statistics['revenue']['ci95_low'],
                             'revenue_ci95_high': statistics['revenue']['ci95_high'],
                             'volume': statistics['volume']['estimate'],
                             'volume_ci95_low': statistics['volume']['ci95_low'],
                             'volume_ci95_high': statistics['volume']['ci95_high'],
                             'delta_revenue_vs_validation_fixed': contrast['estimate'],
                             'delta_ci95_low': contrast['ci95_low'], 'delta_ci95_high': contrast['ci95_high'],
                             'service_slack': statistics['service_slack']['estimate'],
                             'service_lower95': statistics['service_slack']['one_sided95_low'],
                             'hold_revenue': statistics['hold_revenue']['estimate'],
                             'hold_volume': statistics['hold_volume']['estimate']})
            for comparator in dict.fromkeys((reference, posthoc, posthoc_feasible)):
                group_record['belief_comparisons'][comparator] = {
                    'reference_status': 'preselected_from_main_validation' if comparator==reference else 'stress_posthoc_diagnostic',
                    'revenue_difference': estimate(gp['belief'][0]-gp[comparator][0],
                                                   gb['belief'][:,0]-gb[comparator][:,0]),
                    'volume_difference': estimate(gp['belief'][1]-gp[comparator][1],
                                                  gb['belief'][:,1]-gb[comparator][:,1])}
            group_record['belief_seed_distribution'] = []
            for seed in SEEDS:
                v = arrays['belief'][seed,positions].mean(axis=(0,1))
                seed_row = {'case': case, 'training_seed': seed, 'group': group,
                            **{name: float(v[j]) for j,name in enumerate(METRICS)},
                            'revenue_difference_vs_validation_fixed': float(v[0]-gp[reference][0]),
                            'service_slack': float(v[1]-.95*gp['fixed_0'][1])}
                seed_rows.append(seed_row)
                group_record['belief_seed_distribution'].append(seed_row)
            case_summary['groups'][group] = group_record
        for unit in range(arrays['belief'].shape[1]):
            for method in methods:
                record = {'case':case, 'unit_id': unit,
                          'unit_label': 'ABCD'[unit] if case=='exact' else str(unit),
                          'unit_kind': 'market' if case=='exact' else 'model', 'method':method}
                for column,name in enumerate(('revenue','volume')):
                    stat = estimate(points[method][unit,column], boots[method][:,unit,column])
                    record[name] = stat['estimate']
                    record[name+'_ci95_low'] = stat['ci95_low']
                    record[name+'_ci95_high'] = stat['ci95_high']
                type_rows.append(record)
        summary['cases'][case] = case_summary
        print(json.dumps({'summary_completed':case,
                          'belief_revenue':global_points['belief'][0],
                          'belief_volume':global_points['belief'][1],
                          'selected_fixed_revenue':global_points[reference][0],
                          'posthoc_best_fixed_diagnostic':posthoc}), flush=True)
    write_json(OUT/'stress_summary.json', summary)
    write_csv(OUT/'stress_summary.csv', rows)
    write_csv(OUT/'stress_seed_results.csv', seed_rows)
    write_csv(OUT/'stress_type_results.csv', type_rows)
    print(json.dumps({'status':'stress_summary_written','path':str(OUT/'stress_summary.json')}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['run','summarize','all'])
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.workers<=3:
        raise ValueError('At most three stress processes authorized')
    if args.command in ('run','all'):
        run_all(args.workers)
    if args.command in ('summarize','all'):
        summarize()


if __name__=='__main__':main()
