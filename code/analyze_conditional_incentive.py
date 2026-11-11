"""Prespecified V5 conditional-reward analysis, with withdrawal endpoints and complete costs.

No training/test episodes are statistical replicates inside a world. The paired
crossed bootstrap resamples ten platform seeds and worlds within fixed types;
it never resamples 400 rounds, three buyers, or the finite type catalogue.
"""
# Independently derived from V4 analyzer SHA256 67b94667e784d5018d19f384666f7f6d586190f6f55fe84b674577d510fc36a0.
# Configuration: V5 source paths, 40 public reward actions, streams 850041/850051.
# This file never reads V4 experimental data or mutates the V4 analyzer.
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
for _variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_variable, '1')
import numpy as np
from analyze_threshold_experiment import crossed_bootstrap
from conditional_incentive_market import PHYSICAL_VALUES, BONUS_ACTIONS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments' / 'v5_conditional_incentive'
SEEDS = tuple(range(10))
CASES = ('bounded', 'transferable', 'exact_transferable', 'no_trade')
BUDGETS = (0, 10, 25, 50, 100)
METRICS = ('gross_revenue', 'cost', 'net_revenue', 'jobs', 'value', 'utility',
           'underreport_count', 'severity', 'overreport_count', 'active',
           'join_count', 'formed_members', 'harmful_members',
           'successful_collusion_members', 'trade_attempt', 'trade_success',
           'trade_gains', 'failed_jobs')
INDEX = {name: position for position, name in enumerate(METRICS)}
BUYER_METRICS = ('completed', 'payments', 'bonuses', 'active', 'underreport', 'overreport')
BUYER_ANALYSIS_METRICS = (*BUYER_METRICS, 'completed_value')
PHASES = ('full', 'control', 'withdraw')
ROUNDS = (400, 200, 200)
METHODS = ('none', 'fixed', 'dynamic')
REPLICATES = 10_000
BOOTSTRAP_SEED = 850051
TEST_STREAM = 850041
TOLERANCE = 1e-9


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding='utf-8', newline='\n')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def estimate(point, samples):
    samples = np.asarray(samples, dtype=float)
    invalid = int((~np.isfinite(samples)).sum())
    result = {'estimate': float(point) if np.isfinite(point) else None,
              'ci95_low': None, 'ci95_high': None,
              'one_sided95_low': None, 'one_sided95_high': None,
              'bootstrap_sd': None, 'invalid_bootstrap_draws': invalid}
    # Never silently drop zero-denominator draws and change the estimand.
    if invalid or not np.isfinite(point):
        return result
    q = np.quantile(samples, [.025, .975, .05, .95])
    result.update(ci95_low=float(q[0]), ci95_high=float(q[1]),
                  one_sided95_low=float(q[2]), one_sided95_high=float(q[3]),
                  bootstrap_sd=float(np.std(samples, ddof=1)))
    return result


def ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(np.asarray(numerator, dtype=float),
                                                  np.asarray(denominator, dtype=float))
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan),
                     where=denominator > 0)


def quantities(raw, rounds):
    """Derive rates AFTER aggregation of numerator/denominator counts."""
    result = {name: raw[..., column] for column, name in enumerate(METRICS)}
    slots = 3 * rounds
    result.update(
        underreport_active_rate=ratio(result['underreport_count'], result['active']),
        underreport_all_rate=result['underreport_count'] / slots,
        overreport_active_rate=ratio(result['overreport_count'], result['active']),
        overreport_all_rate=result['overreport_count'] / slots,
        exit_rate=1 - result['active'] / slots,
        join_rate=result['join_count'] / slots,
        formed_member_rate=result['formed_members'] / slots,
        coordinated_underreport_member_rate=result['harmful_members'] / slots,
        successful_collusion_member_rate=result['successful_collusion_members'] / slots,
        trade_attempt_rate=result['trade_attempt'] / rounds,
        trade_success_rate=result['trade_success'] / rounds)
    return result


def buyer_quantities(raw, rounds):
    result = {name: raw[..., column] for column, name in enumerate(BUYER_ANALYSIS_METRICS)}
    result.update(completion_rate=result['completed'] / rounds,
                  net_payment=result['payments'] - result['bonuses'],
                  underreport_active_rate=ratio(result['underreport'], result['active']),
                  underreport_all_rate=result['underreport'] / rounds,
                  overreport_active_rate=ratio(result['overreport'], result['active']),
                  overreport_all_rate=result['overreport'] / rounds,
                  exit_rate=1 - result['active'] / rounds)
    return result


def buyer_weights(types):
    result = {f'buyer_{i}': np.broadcast_to(np.eye(3)[i], (len(types), 3)) for i in range(3)}
    large = np.eye(3)[types // 2 % 3]
    result['large_task_buyer'] = large
    result['small_task_buyer_average'] = (1 - large) / 2
    return result


def freeze():
    paths = [OUT / 'protocol.json', OUT / 'selection.json', Path(__file__),
             ROOT / 'code' / 'run_conditional_incentive.py',
             ROOT / 'code' / 'conditional_incentive_market.py', ROOT / 'code' / 'conditional_incentive_stress.py',
             ROOT / 'code' / 'incentive_market.py', ROOT / 'code' / 'incentive_stress.py',
             ROOT / 'scripts' / 'run_v5_incentives.py',
             ROOT / 'code' / 'validate_micro_model.py',
             ROOT / 'code' / 'analyze_threshold_experiment.py',
             ROOT / 'docs' / 'v5_conditional_protocol.md',
             ROOT / 'docs' / 'v4_technical_design.md',
             ROOT / 'docs' / 'v5_conditional_protocol.freeze.json']
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f'Freeze requires prepared inputs: {path}')
    hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}
    doc_freeze = json.loads((ROOT / 'docs' / 'v5_conditional_protocol.freeze.json').read_text(encoding='utf-8-sig'))
    for name, checksum in doc_freeze['files'].items():
        if digest(ROOT / name) != checksum:
            raise RuntimeError(f'Design document differs from its original freeze: {name}')
        # The V5 protocol inherits frozen V4 definitions; verify those files
        # again at analysis time, not only when constructing this manifest.
        hashes[name] = checksum
    document = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'analysis_frozen_before_v5_test', 'hashes': hashes,
                'replicates': REPLICATES, 'bootstrap_seed_base': BOOTSTRAP_SEED,
                'case_order_and_seed_offsets': {name: i for i, name in enumerate(CASES)},
                'budgets_share_bootstrap_weights_within_case': True,
                'seeds': list(SEEDS), 'cases': list(CASES), 'budgets': list(BUDGETS),
                'worlds_per_fixed_type': 100, 'exact_type_ids': list(range(0, 24, 2)),
                'metric_names': list(METRICS), 'phases': list(PHASES),
                'buyer_metric_names': list(BUYER_METRICS),
                'buyer_reporting': 'Paired counts/rates by account, task-size role and fixed type; completed value derived from frozen private catalogue for evaluation only. No per-buyer utility claim without side-payment records.',
                'primary': 'bounded B50 dynamic versus none; withdrawal U and successful C relative20% reductions, full/withdraw J non-loss, pathwise budget',
                'dynamic_increment': 'versus selected fixed: actual cost upper95<0, U/C upper95<=0, full/withdraw J lower95>=0',
                'rounding_tolerance_only': TOLERANCE,
                'zero_reference_rate': 'absolute zero required; relative reduction not defined',
                'zero_active_denominator': 'undefined, not an artificial zero underreport rate',
                'test_read_before_analysis_freeze': False,
                'scope': 'Finite fixed strata; stress/subgroup/budget-transfer results are explanatory.'}
    path = OUT / 'analysis_freeze.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous['hashes'] != hashes:
            raise RuntimeError('Existing analysis freeze changed; append an explicit amendment.')
        return previous
    if (OUT / 'test').exists():
        raise RuntimeError('Test directory exists: a before-test analysis freeze cannot be created retrospectively.')
    write_json(path, document)
    print('V5 analysis frozen:', hashes[Path(__file__).relative_to(ROOT).as_posix()], flush=True)
    return document


def validate_payload(data, expected_types, expected_budget, path):
    """Check every path and phase; accepts in-memory synthetic data in self-test."""
    models = np.asarray(data['models'], dtype=int)
    names = tuple(str(name) for name in data['metric_names'])
    if names != METRICS:
        raise ValueError(f'{path}: metric schema differs')
    if len(models) % len(expected_types):
        raise ValueError(f'{path}: unbalanced fixed types')
    worlds = len(models) // len(expected_types)
    np.testing.assert_array_equal(models, np.repeat(expected_types, worlds))
    full = np.asarray(data['metrics'], dtype=float)
    phase = np.asarray(data['phases'], dtype=float)
    trajectory = np.asarray(data['trajectories'], dtype=float)
    actions = np.asarray(data['actions'], dtype=int)
    balance = np.asarray(data['remaining_budget'], dtype=float)
    budget = float(np.asarray(data['budget']).item())
    if budget != expected_budget or int(np.asarray(data['stream_seed']).item()) != TEST_STREAM:
        raise ValueError(f'{path}: budget or test stream changed')
    n, k = len(models), len(METRICS)
    assert full.shape == (n, k) and phase.shape == (n, 2, k)
    assert trajectory.shape == (40, n, k) and actions.shape == (40, n)
    assert balance.shape == (n,) and np.isfinite(full).all() and np.isfinite(trajectory).all()
    assert ((actions >= 0) & (actions < len(BONUS_ACTIONS))).all() and (actions[20:] == 0).all()
    np.testing.assert_allclose(phase.sum(1), full, atol=2e-8, rtol=1e-12)
    np.testing.assert_allclose(trajectory.sum(0), full, atol=2e-8, rtol=1e-12)
    np.testing.assert_allclose(trajectory[:20].sum(0), phase[:, 0], atol=2e-8, rtol=1e-12)
    np.testing.assert_allclose(trajectory[20:].sum(0), phase[:, 1], atol=2e-8, rtol=1e-12)
    c = INDEX['cost']
    assert (trajectory[:, :, c] >= -TOLERANCE).all()
    assert (phase[:, 1, c] == 0).all()
    assert (full[:, c] <= budget + TOLERANCE).all() and (balance >= -TOLERANCE).all()
    np.testing.assert_allclose(full[:, c] + balance, budget, atol=2e-8, rtol=1e-12)
    # Check reservations against balances BEFORE each block, not only final cost.
    sums = np.asarray(BONUS_ACTIONS, dtype=float).sum(axis=1)
    before = budget - np.vstack([np.zeros(n), np.cumsum(trajectory[:, :, c], axis=0)[:-1]])
    assert (10 * sums[actions] <= before + TOLERANCE).all(), (path, 'unfunded promise')
    combined = np.concatenate([full[:, None], phase], axis=1)
    buyer_names = tuple(str(name) for name in data['buyer_metric_names'])
    if buyer_names != BUYER_METRICS:
        raise ValueError(f'{path}: buyer metric schema differs')
    buyer_phase = np.asarray(data['buyer_phases'], dtype=float)
    assert buyer_phase.shape == (n, 2, 3, len(BUYER_METRICS))
    assert np.isfinite(buyer_phase).all() and (buyer_phase >= -TOLERANCE).all()
    market_names = ('jobs', 'gross_revenue', 'cost', 'active', 'underreport_count', 'overreport_count')
    for bi, name in enumerate(market_names):
        np.testing.assert_allclose(buyer_phase[:, :, :, bi].sum(2), phase[:, :, INDEX[name]], atol=2e-8)
    assert (buyer_phase[:, :, :, 0] <= 200).all()
    assert (buyer_phase[:, :, :, 3] <= 200).all()
    for bi in (4, 5):
        assert (buyer_phase[:, :, :, bi] <= buyer_phase[:, :, :, 3]).all()
    assert (buyer_phase[:, 1, :, 2] == 0).all()
    buyer_combined = np.concatenate([buyer_phase.sum(1)[:, None], buyer_phase], axis=1)
    buyer_value = buyer_combined[..., 0] * PHYSICAL_VALUES[models // 2][:, None, :]
    np.testing.assert_allclose(buyer_value.sum(2), combined[:, :, INDEX['value']], atol=2e-8)
    buyer_combined = np.concatenate([buyer_combined, buyer_value[..., None]], axis=-1)
    for pi, rounds in enumerate(ROUNDS):
        raw = combined[:, pi]
        np.testing.assert_allclose(raw[:, INDEX['gross_revenue']] - raw[:, c], raw[:, INDEX['net_revenue']], atol=2e-8)
        np.testing.assert_allclose(raw[:, INDEX['net_revenue']] + raw[:, INDEX['utility']], raw[:, INDEX['value']], atol=2e-8)
        assert ((raw[:, INDEX['jobs']] >= 0) & (raw[:, INDEX['jobs']] <= 2 * rounds)).all()
        np.testing.assert_allclose(raw[:, INDEX['failed_jobs']] + raw[:, INDEX['jobs']], 3 * rounds)
        active = raw[:, INDEX['active']]
        assert ((active >= 0) & (active <= 3 * rounds)).all()
        for name in ('underreport_count', 'overreport_count', 'join_count'):
            assert ((raw[:, INDEX[name]] >= 0) & (raw[:, INDEX[name]] <= active)).all()
        assert (raw[:, INDEX['harmful_members']] <= raw[:, INDEX['formed_members']]).all()
        assert (raw[:, INDEX['successful_collusion_members']] <= raw[:, INDEX['harmful_members']]).all()
        assert (raw[:, INDEX['trade_success']] <= raw[:, INDEX['trade_attempt']]).all()
    audit = {'file': str(path), 'worlds': n, 'budget': budget,
             'maximum_actual_cost': float(full[:, c].max()), 'minimum_remaining': float(balance.min()),
             'budget_violations': int((full[:, c] > budget + TOLERANCE).sum()),
             'zero_active_withdraw_worlds': int((phase[:, 1, INDEX['active']] == 0).sum())}
    return (combined.reshape(len(expected_types), worlds, 3, k),
            buyer_combined.reshape(len(expected_types), worlds, 3, 3, len(BUYER_ANALYSIS_METRICS)), audit)


def load_case(case, budget):
    types = np.arange(0, 24, 2) if case == 'exact_transferable' else np.arange(24)
    arrays, buyers, hashes, audits = {}, {}, {}, []
    for method in METHODS:
        names = ([f'dynamic_b{budget}_s{s}.npz' for s in SEEDS] if method == 'dynamic'
                 else ['none.npz'] if method == 'none' else [f'fixed_b{budget}.npz'])
        pieces, buyer_pieces = [], []
        for name in names:
            path = OUT / 'test' / case / name
            with np.load(path, allow_pickle=False) as data:
                values, buyer_values, audit = validate_payload(data, types, 0 if method == 'none' else budget, path)
            assert values.shape[1] == 100, (path, 'expected 100 independent worlds per type')
            if case == 'no_trade':
                for metric in ('trade_attempt', 'trade_success', 'trade_gains', 'successful_collusion_members'):
                    assert (values[..., INDEX[metric]] == 0).all()
            pieces.append(values)
            buyer_pieces.append(buyer_values)
            audit.update(method=method, case=case, analysis_budget=budget)
            audits.append(audit)
            hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        arrays[method] = np.stack(pieces)
        buyers[method] = np.stack(buyer_pieces)
    if budget == 0:
        for method in ('fixed', 'dynamic'):
            np.testing.assert_allclose(arrays[method], np.broadcast_to(arrays['none'], arrays[method].shape), atol=1e-12)
            np.testing.assert_allclose(buyers[method], np.broadcast_to(buyers['none'], buyers[method].shape), atol=1e-12)
    return arrays, buyers, types, hashes, audits


def groups_for(types):
    result = {'all': np.arange(len(types))}
    for family in range(4):
        result[f'family_{"ABCD"[family]}'] = np.flatnonzero(types // 6 == family)
    if len(np.unique(types % 2)) > 1:
        result['eta_0.05'] = np.flatnonzero(types % 2 == 0)
        result['eta_0.2'] = np.flatnonzero(types % 2 == 1)
    return result


def bootstrap(arrays, replicates=REPLICATES, seed=BOOTSTRAP_SEED):
    metrics = next(iter(arrays.values())).shape[-1]
    flat = {name: data.reshape(*data.shape[:3], -1) for name, data in arrays.items()}
    draws = crossed_bootstrap(flat, replicates=replicates, seed=seed, chunk_size=64)
    return {name: data.reshape(data.shape[0], data.shape[1], 3, metrics)
            for name, data in draws.items()}


def upper_pass(record, strict=False):
    value = record['one_sided95_high']
    return value is not None and (value < -TOLERANCE if strict else value <= TOLERANCE)


def lower_pass(record):
    value = record['one_sided95_low']
    return value is not None and value >= -TOLERANCE


def compare(point, draws, reference_point, reference_draws):
    phases = {}
    for pi, (phase, rounds) in enumerate(zip(PHASES, ROUNDS)):
        p, b = quantities(point[pi], rounds), quantities(draws[:, pi], rounds)
        r, rb = quantities(reference_point[pi], rounds), quantities(reference_draws[:, pi], rounds)
        stats = {}
        for metric in p:
            difference = p[metric] - r[metric]
            sample = b[metric] - rb[metric]
            stats[metric] = {'difference': estimate(difference, sample),
                             'relative_percent': estimate(100 * ratio(difference, r[metric]),
                                                          100 * ratio(sample, rb[metric]))}
        phases[phase] = stats
    p, b = quantities(point[2], 200), quantities(draws[:, 2], 200)
    r, rb = quantities(reference_point[2], 200), quantities(reference_draws[:, 2], 200)
    targets = {}
    for name in ('underreport_active_rate', 'successful_collusion_member_rate'):
        targets[name] = estimate(p[name] - .8 * r[name], b[name] - .8 * rb[name])
    return {'phases': phases, 'relative20_target_contrasts': targets,
            'reference_positive': {name: bool(np.isfinite(r[name]) and r[name] > 0) for name in targets}}


def target_criteria(comparison, budget_ok=True):
    phase = comparison['phases']
    criteria = {'withdraw_underreport_relative20_or_absolute_zero': upper_pass(comparison['relative20_target_contrasts']['underreport_active_rate']),
                'withdraw_successful_collusion_relative20_or_absolute_zero': upper_pass(comparison['relative20_target_contrasts']['successful_collusion_member_rate']),
                'full_jobs_nonloss': lower_pass(phase['full']['jobs']['difference']),
                'withdraw_jobs_nonloss': lower_pass(phase['withdraw']['jobs']['difference']),
                'pathwise_budget': bool(budget_ok)}
    return {'criteria': criteria, 'joint_pass': all(criteria.values()),
            'both_relative_reductions_defined': all(comparison['reference_positive'].values())}


def incremental_criteria(comparison, budget_ok=True):
    phase = comparison['phases']
    criteria = {'actual_cost_strictly_lower': upper_pass(phase['full']['cost']['difference'], strict=True),
                'withdraw_underreport_nonincrease': upper_pass(phase['withdraw']['underreport_active_rate']['difference']),
                'withdraw_successful_collusion_nonincrease': upper_pass(phase['withdraw']['successful_collusion_member_rate']['difference']),
                'full_jobs_nonloss': lower_pass(phase['full']['jobs']['difference']),
                'withdraw_jobs_nonloss': lower_pass(phase['withdraw']['jobs']['difference']),
                'pathwise_budget': bool(budget_ok)}
    return {'criteria': criteria, 'joint_pass': all(criteria.values())}


def analyze():
    frozen = json.loads((OUT / 'analysis_freeze.json').read_text(encoding='utf-8'))
    for name, checksum in frozen['hashes'].items():
        if digest(ROOT / name) != checksum:
            raise RuntimeError(f'Frozen source/input changed: {name}')
    pretest = json.loads((OUT / 'pretest_freeze.json').read_text(encoding='utf-8'))
    for name, checksum in pretest['hashes'].items():
        if digest(ROOT / name) != checksum:
            raise RuntimeError(f'Pretest source/checkpoint changed: {name}')
    key = Path(__file__).relative_to(ROOT).as_posix()
    if pretest['hashes'].get(key) != digest(__file__):
        raise RuntimeError('Test was not frozen against this analysis source.')
    selection = json.loads((OUT / 'selection.json').read_text(encoding='utf-8'))
    import torch
    selected_seeds = []
    for seed in SEEDS:
        checkpoint = torch.load(OUT / 'checkpoints' / f'dynamic_{seed}.pt', weights_only=False)
        selected_seeds.append({'seed': seed, 'kind': checkpoint['kind'],
                               'iteration': checkpoint['iteration'], 'controller': checkpoint.get('controller')})
    method_rows, difference_rows, type_rows, seed_rows = [], [], [], []
    buyer_rows, buyer_type_rows, buyer_difference_rows = [], [], []
    frontier, audits, data_hashes, results = [], [], {}, {}
    for ci, case in enumerate(CASES):
        results[case] = {}
        for budget in (BUDGETS if case == 'bounded' else (50,)):
            arrays, buyer_arrays, types, hashes, path_audits = load_case(case, budget)
            data_hashes.update(hashes)
            audits.extend(path_audits)
            # Joint call guarantees identical resampling of market and buyer data.
            joined = {name: np.concatenate([arrays[name], buyer_arrays[name].reshape(*arrays[name].shape[:-1], -1)], axis=-1)
                      for name in METHODS}
            joint_boots = bootstrap(joined, seed=BOOTSTRAP_SEED + ci)
            boots = {name: values[..., :len(METRICS)] for name, values in joint_boots.items()}
            buyer_boots = {name: values[..., len(METRICS):].reshape(*values.shape[:3], 3, len(BUYER_ANALYSIS_METRICS))
                           for name, values in joint_boots.items()}
            points = {name: data.mean(axis=(0, 2)) for name, data in arrays.items()}
            buyer_points = {name: data.mean(axis=(0, 2)) for name, data in buyer_arrays.items()}
            groups = groups_for(types)
            roles = buyer_weights(types)
            case_result = {'fixed_type_ids': types.tolist(), 'worlds_per_type': 100,
                           'selected_fixed': selection['by_budget'][str(budget)]['candidate'],
                           'methods': {}, 'comparisons': {}}
            for method in METHODS:
                case_result['methods'][method] = {}
                for group, indices in groups.items():
                    p = points[method][indices].mean(0)
                    b = boots[method][:, indices].mean(1)
                    phase_results = {}
                    for pi, (phase, rounds) in enumerate(zip(PHASES, ROUNDS)):
                        qp, qb = quantities(p[pi], rounds), quantities(b[:, pi], rounds)
                        phase_results[phase] = {name: estimate(qp[name], qb[name]) for name in qp}
                        for name, stat in phase_results[phase].items():
                            method_rows.append({'case': case, 'budget': budget, 'method': method,
                                                'group': group, 'phase': phase, 'metric': name, **stat})
                    case_result['methods'][method][group] = phase_results
                    for si in range(arrays[method].shape[0]):
                        raw = arrays[method][si, indices].mean(axis=(0, 1))
                        for pi, (phase, rounds) in enumerate(zip(PHASES, ROUNDS)):
                            vals = quantities(raw[pi], rounds)
                            seed_rows.append({'case': case, 'budget': budget, 'method': method,
                                'training_seed': si if method == 'dynamic' else 'not_trained',
                                'group': group, 'phase': phase,
                                **{name: float(value) if np.isfinite(value) else None for name, value in vals.items()}})
                for ti, model_id in enumerate(types):
                    for pi, (phase, rounds) in enumerate(zip(PHASES, ROUNDS)):
                        qp = quantities(points[method][ti, pi], rounds)
                        qb = quantities(boots[method][:, ti, pi], rounds)
                        for name in qp:
                            type_rows.append({'case': case, 'budget': budget, 'method': method,
                                'model_id': int(model_id), 'physical_id': int(model_id // 2),
                                'eta_label_relevant': case != 'exact_transferable',
                                'phase': phase, 'metric': name, **estimate(qp[name], qb[name])})
                    for buyer in range(3):
                        for pi, (phase, rounds) in enumerate(zip(PHASES, ROUNDS)):
                            quantities_at_type = buyer_quantities(buyer_points[method][ti, pi, buyer], rounds)
                            buyer_type_rows.append({'case': case, 'budget': budget, 'method': method,
                                'model_id': int(model_id), 'physical_id': int(model_id // 2), 'buyer': buyer,
                                'task_size_role': 'large' if buyer == model_id // 2 % 3 else 'small',
                                'phase': phase, 'inference': 'descriptive fixed-type mean',
                                **{name: float(value) if np.isfinite(value) else None for name, value in quantities_at_type.items()}})
                for group, indices in groups.items():
                    for role, weights in roles.items():
                        bp = np.einsum('tpbk,tb->pk', buyer_points[method][indices], weights[indices]) / len(indices)
                        bb = np.einsum('rtpbk,tb->rpk', buyer_boots[method][:, indices], weights[indices]) / len(indices)
                        for pi, (phase, rounds) in enumerate(zip(PHASES, ROUNDS)):
                            qp, qb = buyer_quantities(bp[pi], rounds), buyer_quantities(bb[:, pi], rounds)
                            for name in qp:
                                buyer_rows.append({'case': case, 'budget': budget, 'method': method,
                                    'group': group, 'buyer_or_role': role, 'phase': phase,
                                    'metric': name, **estimate(qp[name], qb[name])})
            for method, reference, label in (('dynamic', 'none', 'dynamic_vs_none'),
                                              ('fixed', 'none', 'fixed_vs_none'),
                                              ('dynamic', 'fixed', 'dynamic_vs_fixed')):
                case_result['comparisons'][label] = {}
                for group, indices in groups.items():
                    comparison = compare(points[method][indices].mean(0), boots[method][:, indices].mean(1),
                                         points[reference][indices].mean(0), boots[reference][:, indices].mean(1))
                    comparison['method'], comparison['reference'] = method, reference
                    # The relative20 target is meaningful only against no reward.
                    if reference == 'none':
                        comparison['decision'] = target_criteria(comparison)
                    else:
                        comparison.pop('relative20_target_contrasts')
                        comparison.pop('reference_positive')
                        comparison['decision'] = incremental_criteria(comparison)
                    case_result['comparisons'][label][group] = comparison
                    for phase, metric_stats in comparison['phases'].items():
                        for metric, effects in metric_stats.items():
                            for effect, stat in effects.items():
                                difference_rows.append({'case': case, 'budget': budget, 'comparison': label,
                                    'group': group, 'phase': phase, 'metric': metric, 'effect': effect,
                                    'joint_pass': comparison['decision']['joint_pass'], **stat})
                    for metric, stat in comparison.get('relative20_target_contrasts', {}).items():
                        difference_rows.append({'case': case, 'budget': budget, 'comparison': label,
                            'group': group, 'phase': 'withdraw', 'metric': metric, 'effect': 'method_minus_0.8_none',
                            'joint_pass': comparison['decision']['joint_pass'], **stat})
                    for role, weights in roles.items():
                        mp = np.einsum('tpbk,tb->pk', buyer_points[method][indices], weights[indices]) / len(indices)
                        rp = np.einsum('tpbk,tb->pk', buyer_points[reference][indices], weights[indices]) / len(indices)
                        mb = np.einsum('rtpbk,tb->rpk', buyer_boots[method][:, indices], weights[indices]) / len(indices)
                        rb = np.einsum('rtpbk,tb->rpk', buyer_boots[reference][:, indices], weights[indices]) / len(indices)
                        for pi, (phase, rounds) in enumerate(zip(PHASES, ROUNDS)):
                            mq, rq = buyer_quantities(mp[pi], rounds), buyer_quantities(rp[pi], rounds)
                            mqb, rqb = buyer_quantities(mb[:, pi], rounds), buyer_quantities(rb[:, pi], rounds)
                            for name in ('completed', 'completed_value', 'bonuses', 'net_payment', 'underreport_active_rate', 'exit_rate'):
                                buyer_difference_rows.append({'case': case, 'budget': budget, 'comparison': label,
                                    'group': group, 'buyer_or_role': role, 'phase': phase, 'metric': name,
                                    **estimate(mq[name] - rq[name], mqb[name] - rqb[name])})
            results[case][str(budget)] = case_result
            for method in METHODS:
                stats = case_result['methods'][method]['all']
                decision = (case_result['comparisons'][f'{method}_vs_none']['all']['decision']
                            if method != 'none' else {'joint_pass': False})
                frontier.append({'case': case, 'budget': budget, 'method': method,
                    'cost': stats['full']['cost']['estimate'], 'net_revenue': stats['full']['net_revenue']['estimate'],
                    'jobs': stats['full']['jobs']['estimate'], 'withdraw_jobs': stats['withdraw']['jobs']['estimate'],
                    'withdraw_underreport_rate': stats['withdraw']['underreport_active_rate']['estimate'],
                    'withdraw_successful_collusion_rate': stats['withdraw']['successful_collusion_member_rate']['estimate'],
                    'withdraw_coordination_proxy_rate': stats['withdraw']['coordinated_underreport_member_rate']['estimate'],
                    'joint_target_pass': decision['joint_pass'],
                    'budget_specific_dynamic_retraining': False})
            print(json.dumps({'analyzed_case': case, 'budget': budget,
                'target': case_result['comparisons']['dynamic_vs_none']['all']['decision'],
                'increment': case_result['comparisons']['dynamic_vs_fixed']['all']['decision']}), flush=True)
            del arrays, buyer_arrays, boots, buyer_boots, joint_boots, joined, points, buyer_points
    destination = OUT / 'analysis'
    for filename, rows in (('summary.csv', method_rows), ('comparisons.csv', difference_rows),
                           ('type_results.csv', type_rows), ('seed_results.csv', seed_rows),
                           ('cost_frontier.csv', frontier), ('path_audits.csv', audits),
                           ('buyer_summary.csv', buyer_rows), ('buyer_type_results.csv', buyer_type_rows),
                           ('buyer_comparisons.csv', buyer_difference_rows)):
        write_csv(destination / filename, rows)
    main = results['bounded']['50']
    primary = {'status': 'complete', 'primary_case': 'bounded', 'primary_budget': 50,
        'selected_fixed': main['selected_fixed'], 'selected_platform_seeds': selected_seeds,
        'learned_policy_count': sum(row['kind'] == 'dynamic' for row in selected_seeds),
        'scheduled_fallback_count': sum(row['kind'] == 'scheduled' for row in selected_seeds),
        'comparisons': {name: group['all'] for name, group in main['comparisons'].items()},
        'methods': {name: group['all'] for name, group in main['methods'].items()},
        'all_path_budget_violations': sum(row['budget_violations'] for row in audits),
        'interpretation': 'Only the bounded B50 overall joint comparison is the prespecified primary target. Dynamic cost superiority is separate; stress, type and budget-transfer claims are conditional.',
        'relative_target': '20 percent relative reduction, not 20 percentage points; zero baselines require absolute zero and carry no relative-reduction claim.'}
    write_json(destination / 'primary.json', primary)
    write_json(destination / 'summary.json', {'status': 'complete', 'analysis_freeze': frozen,
        'pretest_freeze_sha256': digest(OUT / 'pretest_freeze.json'), 'data_sha256': data_hashes,
        'selected_platform_seeds': selected_seeds, 'cases': results,
        'scope': 'Synthetic finite-catalogue evidence, conditional on specified learners and transaction rules; no universal truthful-reporting or coalition-proof guarantee.'})
    print('V5 analysis written:', destination, flush=True)


def synthetic_payload(worlds=8):
    """Only self-test fixtures: no experiment output or private results used."""
    types = np.array([0, 2, 6, 8])
    n = worlds * len(types)
    t = np.zeros((40, n, len(METRICS)))
    t[:, :, INDEX['gross_revenue']] = 20
    t[:, :, INDEX['net_revenue']] = 20
    t[:, :, INDEX['jobs']] = 15
    t[:, :, INDEX['value']] = 50
    t[:, :, INDEX['utility']] = 30
    t[:, :, INDEX['failed_jobs']] = 15
    t[:, :, INDEX['active']] = 30
    t[:, :, INDEX['underreport_count']] = 10
    t[:, :, INDEX['join_count']] = 10
    t[:, :, INDEX['formed_members']] = 10
    t[:, :, INDEX['harmful_members']] = 8
    t[:, :, INDEX['successful_collusion_members']] = 6
    t[:, :, INDEX['trade_attempt']] = 5
    t[:, :, INDEX['trade_success']] = 3
    buyer_phase = np.zeros((n, 2, 3, len(BUYER_METRICS)))
    # Give equal completion counts; value differs by the frozen physical type.
    buyer_phase[..., 0] = 100
    buyer_phase[..., 1] = 400 / 3
    buyer_phase[..., 3] = 200
    buyer_phase[..., 4] = 200 / 3
    expected_value = (buyer_phase[:, 0, :, 0] * PHYSICAL_VALUES[np.repeat(types, worlds) // 2]).sum(1)
    t[:, :, INDEX['value']] = expected_value[None] / 20
    t[:, :, INDEX['utility']] = t[:, :, INDEX['value']] - 20
    return types, {'models': np.repeat(types, worlds), 'metric_names': np.array(METRICS),
        'metrics': t.sum(0), 'phases': np.stack([t[:20].sum(0), t[20:].sum(0)], axis=1),
        'trajectories': t, 'actions': np.zeros((40, n), dtype=int),
        'remaining_budget': np.full(n, 50.), 'budget': np.array(50), 'stream_seed': np.array(TEST_STREAM),
        'buyer_metric_names': np.array(BUYER_METRICS), 'buyer_phases': buyer_phase}


def self_test():
    types, data = synthetic_payload()
    fixture, buyer_fixture, audit = validate_payload(data, types, 50, 'synthetic')
    assert audit['budget_violations'] == 0
    np.testing.assert_allclose(buyer_fixture[..., 0].sum(-1), fixture[..., INDEX['jobs']])
    weights = buyer_weights(types)
    np.testing.assert_array_equal(weights['large_task_buyer'].sum(1), 1)
    np.testing.assert_array_equal(weights['small_task_buyer_average'].sum(1), 1)
    a = fixture[None]
    b = np.repeat(a, 10, axis=0)
    b[..., INDEX['underreport_count']] *= .7
    b[..., INDEX['successful_collusion_members']] *= .7
    arrays = {'none': a, 'dynamic': b}
    draws = bootstrap(arrays, replicates=501, seed=19)
    points = {name: values.mean(axis=(0, 1, 2)) for name, values in arrays.items()}
    result = compare(points['dynamic'], draws['dynamic'].mean(1), points['none'], draws['none'].mean(1))
    assert target_criteria(result)['joint_pass']
    identity = compare(points['none'], draws['none'].mean(1), points['none'], draws['none'].mean(1))
    assert not target_criteria(identity)['joint_pass']
    assert not incremental_criteria(identity)['joint_pass']
    paid_reference = points['none'].copy()
    paid_reference[0, INDEX['cost']] = 10
    paid_reference_draws = draws['none'].mean(1).copy()
    paid_reference_draws[:, 0, INDEX['cost']] = 10
    cheaper = compare(points['dynamic'], draws['dynamic'].mean(1), paid_reference, paid_reference_draws)
    assert incremental_criteria(cheaper)['joint_pass']
    zero_reference = points['none'].copy()
    zero_reference[2, INDEX['successful_collusion_members']] = 0
    zero_draws = draws['none'].mean(1).copy()
    zero_draws[:, 2, INDEX['successful_collusion_members']] = 0
    zero_check = compare(zero_reference, zero_draws, zero_reference, zero_draws)
    assert not zero_check['reference_positive']['successful_collusion_member_rate']
    assert upper_pass(zero_check['relative20_target_contrasts']['successful_collusion_member_rate'])
    assert zero_check['phases']['withdraw']['successful_collusion_member_rate']['relative_percent']['estimate'] is None
    damaged = points['dynamic'].copy()
    damaged[0, INDEX['jobs']] -= 1
    bad_draws = draws['dynamic'].mean(1).copy()
    bad_draws[:, 0, INDEX['jobs']] -= 1
    assert not target_criteria(compare(damaged, bad_draws, points['none'], draws['none'].mean(1)))['joint_pass']
    # Ratio of pooled counts differs from the average of per-world ratios.
    raw = np.zeros((2, len(METRICS)))
    raw[:, INDEX['underreport_count']] = [1, 9]
    raw[:, INDEX['active']] = [1, 99]
    np.testing.assert_allclose(quantities(raw.mean(0), 200)['underreport_active_rate'], .1)
    assert np.isnan(quantities(np.zeros(len(METRICS)), 200)['underreport_active_rate'])
    assert estimate(0., np.zeros(501))['one_sided95_high'] == 0
    # Exact pairing: a common noisy baseline plus a constant must have zero-width delta.
    rng = np.random.default_rng(71)
    noisy = rng.normal(size=(1, 4, 8, 3, len(METRICS)))
    paired = {'left': noisy, 'right': np.repeat(noisy, 10, axis=0) + 3}
    bs = bootstrap(paired, replicates=251, seed=9)
    np.testing.assert_allclose(bs['right'] - bs['left'], 3, atol=1e-12)
    # Budget violation is checked per path, even when its mean is below the cap.
    broken = {key: np.array(value, copy=True) for key, value in data.items()}
    broken['remaining_budget'][0] = -1
    try:
        validate_payload(broken, types, 50, 'invalid synthetic path')
    except AssertionError:
        pass
    else:
        raise AssertionError('Pathwise budget validation failed')
    unfunded = {key: np.array(value, copy=True) for key, value in data.items()}
    unfunded['budget'] = np.array(0)
    unfunded['remaining_budget'][:] = 0
    unfunded['actions'][0, 0] = 1
    try:
        validate_payload(unfunded, types, 0, 'unfunded synthetic offer')
    except AssertionError:
        pass
    else:
        raise AssertionError('An unpaid but unfunded offer escaped the reservation check')
    assert 'eta_0.2' not in groups_for(np.arange(0, 24, 2))
    print('PASS: phase/cash/budget validation, paired bootstrap, pooled rates, zero-denominator handling, relative20 targets, job nonloss, strict incremental cost and exact strata. No formal V5 test data read.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.freeze:
        freeze()
    if args.run:
        analyze()
    if not (args.self_test or args.freeze or args.run):
        parser.print_help()


if __name__ == '__main__':
    main()
