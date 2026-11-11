"""Exploratory V3: public-posterior matching priority at committed capacity price.

This is a new hypothesis after V2, not a replacement for its frozen analysis.
The platform chooses only among the original maximum-capacity matchings. It
never receives a true market, learner weight, utility or completion outcome.
Current reports update a temporary posterior before matching; observe() then
assimilates the public allocation exactly once, including candidate Q updates.
No policy training or artificial training-seed replication is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

for _variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_variable] = '1'
import numpy as np

from capacity_billing_env import CapacityMarket, CapacityBeliefTracker, pricing_tables
from steering_env import (TABLES, JOINT_REPORTS, MODEL_COUNT, VALUES,
                          PRIMARY_ACTION_INDICES, RHO_VALUES, _softmax, _categorical)
from analyze_threshold_experiment import write_json, write_csv, estimate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments' / 'v3_capacity_allocation'
PRICE = .78
HORIZON = 400
WORLDS = 100
VALIDATION_SEED = 720031
TEST_SEED = 720041
BOOTSTRAP_SEED = 720051
REPLICATES = 10_000
TOL = 1e-9
METRICS = ('revenue', 'volume', 'completed_jobs', 'completed_value',
           'total_user_utility', 'active_buyers', 'underreport_count_active',
           'overreport_count')
CASES = ('canonical', 'tabular', 'exact')
PRIORITIES = ('uniform', 'priority0', 'priority1')
STATIC_METHODS = tuple(f'{priority}_r{local}' for priority in PRIORITIES for local in range(4))
SOURCE_NAMES = ('capacity_allocation_steering.py', 'capacity_billing_env.py',
                'steering_env.py', 'belief_baselines.py', 'analyze_threshold_experiment.py')

# Hypothetical catalogue outcomes are public model predictions, not the true
# environment's completed service. Axes below are rho, report, matching, model.
_models = np.arange(MODEL_COUNT)
_hypothetical_served = TABLES['served'][_models // 4, (_models // 2) % 2]
HYPOTHETICAL_JOBS = _hypothetical_served.sum(-1).transpose(1, 2, 3, 0)
HYPOTHETICAL_VALUE = (
    _hypothetical_served * VALUES[_models // 4, None, None, None, :]
).sum(-1).transpose(1, 2, 3, 0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_hashes():
    return {f'code/{name}': digest(ROOT / 'code' / name) for name in SOURCE_NAMES}


def check_protocol():
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    if protocol['source_hashes'] != current_hashes():
        raise RuntimeError('Source changed after the exploratory protocol freeze')
    return protocol


def random_tie(eligible, random_u):
    """Uniform among eligible outcomes, using the shared matching random draw."""
    count = eligible.sum(-1)
    if np.any(count == 0):
        raise RuntimeError('No eligible maximum-capacity matching')
    rank = np.floor(np.asarray(random_u) * count).astype(int)
    return (np.cumsum(eligible, axis=-1) > rank[:, None]).argmax(-1)


class AllocationActor:
    """Public-only matching API: actions, current reports, public random draw.

    The bank's private fields below are HYPOTHETICAL models, each advanced on
    public reports/owners. There is no reference to a real environment here.
    """
    def __init__(self, batch_size, priority='uniform', learner='canonical'):
        if priority not in (*PRIORITIES, 'posterior'):
            raise ValueError(priority)
        self.priority = priority
        self.tracker = (CapacityBeliefTracker(batch_size, learner_mode=learner,
                        unit_price=PRICE, capacity_weight=1.)
                        if priority == 'posterior' else None)
        self.temporary_posterior = None

    def choose(self, actions, reports, random_u):
        actions = np.asarray(actions, dtype=int)
        reports = np.asarray(reports, dtype=int)
        joint = 4 * reports[:, 0] + reports[:, 1]
        count = TABLES['counts'][actions, joint]
        valid = np.arange(TABLES['owners'].shape[2])[None, :] < count[:, None]
        if self.priority == 'uniform':
            return random_tie(valid, random_u)
        if self.priority in ('priority0', 'priority1'):
            # Prefer the designated buyer as owner of the 3-unit VM whenever
            # this remains a maximum-capacity feasible allocation.
            buyer = int(self.priority[-1])
            score = (TABLES['owners'][actions, joint, :, 0] == buyer).astype(float)
            best = np.where(valid, score, -np.inf).max(-1, keepdims=True)
            return random_tie(valid & (score == best), random_u)
        # Current bids are observed before ownership is selected. Bayes update
        # without changing posterior or candidate learning states yet.
        candidate_probabilities = self.tracker._candidates.report_probabilities(
            np.repeat(actions, MODEL_COUNT)).reshape(-1, MODEL_COUNT, 16)
        likelihood = np.take_along_axis(candidate_probabilities,
                    joint[:, None, None], axis=2)[:, :, 0]
        logpost = self.tracker._log_posterior + np.log(np.maximum(
            likelihood, np.finfo(np.float64).tiny))
        logpost -= logpost.max(-1, keepdims=True)
        posterior = np.exp(logpost)
        posterior /= posterior.sum(-1, keepdims=True)
        self.temporary_posterior = posterior
        jobs = np.einsum('bm,bkm->bk', posterior, HYPOTHETICAL_JOBS[actions, joint])
        value = np.einsum('bm,bkm->bk', posterior, HYPOTHETICAL_VALUE[actions, joint])
        best_jobs = np.where(valid, jobs, -np.inf).max(-1, keepdims=True)
        eligible = valid & (jobs >= best_jobs - 1e-12)
        best_value = np.where(eligible, value, -np.inf).max(-1, keepdims=True)
        return random_tie(eligible & (value >= best_value - 1e-12), random_u)

    def observe(self, actions, reports, owners):
        if self.tracker is not None:
            self.tracker.observe(actions, reports, owners)


class AllocationMarket(CapacityMarket):
    """Actual environment samples reports privately, then calls the public actor.

    True learner predictions and true coalition membership below are used ONLY
    for simulating the buyers. The selector receives just public actions/reports.
    """
    def step_with_selector(self, actions, random_u, selector, response='learning'):
        a = self._actions(actions)
        u = np.asarray(random_u, dtype=float)
        if u.shape != (self.batch_size, 3) or np.any((u < 0) | (u >= 1)):
            raise ValueError('Expected shared uniforms of shape (B,3)')
        phi, qi, qg = self._predictions(a)
        if response == 'learning':
            ri = _categorical(_softmax(qi), u[:, :2])
            ji = 4 * ri[:, 0] + ri[:, 1]
            jg = _categorical(_softmax(qg), u[:, 0])
            joint = np.where(self._coalition.astype(bool), jg, ji)
        elif response == 'exact':
            # Privileged attacker response; pooling utility is invariant to
            # ownership among each report's maximum-capacity matchings.
            slots = np.arange(TABLES['owners'].shape[2])[None, None, :]
            count = TABLES['counts'][a]
            valid = slots < count[:, :, None]
            utility = self.tables['utility'][self._market[:, None, None], 1,
                a[:, None, None], np.arange(16)[None, :, None], slots].sum(-1)
            eu = (utility * valid).sum(-1) / count
            er = (self.tables['revenue'][a] * valid).sum(-1) / count
            best = eu >= eu.max(-1, keepdims=True) - 1e-10
            worst_revenue = np.where(best, er, np.inf).min(-1, keepdims=True)
            joint = (best & (er <= worst_revenue + 1e-10)).argmax(-1)
        else:
            raise ValueError(response)
        outcome = selector(a, JOINT_REPORTS[joint].copy(), u[:, 2].copy())
        return self._realize_and_update(a, joint, outcome, phi, qi, qg), outcome


def specification(method):
    if method == 'old_fixed':
        return 'uniform', 1, 1., .2
    if method == 'dynamic':
        return 'posterior', 0, PRICE, 1.
    priority, local = method.rsplit('_r', 1)
    return priority, int(PRIMARY_ACTION_INDICES[int(local)]), PRICE, 1.


def ids_for(case, worlds=WORLDS):
    return np.repeat([2, 6, 10, 14] if case == 'exact' else np.arange(16), worlds)


def run(method, case, stream, worlds=WORLDS, horizon=HORIZON):
    ids = ids_for(case, worlds)
    learner = 'tabular' if case == 'tabular' else 'canonical'
    priority, action, price, alpha = specification(method)
    actor = AllocationActor(len(ids), priority, learner)
    env = AllocationMarket(ids, learner_mode=learner,
                           unit_price=price, capacity_weight=alpha)
    a = np.full(len(ids), action, dtype=int)
    rng = np.random.default_rng(stream)
    metrics = np.zeros((len(ids), len(METRICS)))
    trajectory = np.zeros((horizon // 10, len(ids), 3))
    owners = np.zeros((horizon, len(ids), 2), dtype=np.int8)
    matching_indices = np.zeros((horizon, len(ids)), dtype=np.int8)
    for t in range(horizon):
        (observation, hidden_metrics), chosen = env.step_with_selector(
            a, rng.random((len(ids), 3)), actor.choose,
            response='exact' if case == 'exact' else 'learning')
        actor.observe(a, observation['reports'], observation['owners'])
        owners[t] = observation['owners']
        matching_indices[t] = chosen
        for column, key in enumerate(METRICS):
            metrics[:, column] += observation[key] if key in observation else hidden_metrics[key]
        trajectory[t // 10] += np.column_stack((observation['revenue'],
            hidden_metrics['completed_jobs'], hidden_metrics['completed_value']))
    result = {'models': ids, 'metric_names': np.array(METRICS), 'metrics': metrics,
        'trajectory': trajectory, 'owners': owners, 'matching_indices': matching_indices,
        'env_action': np.array(action), 'stream_seed': np.array(stream),
        'horizon': np.array(horizon), 'method': np.array(method)}
    if actor.tracker is not None:
        result['posterior'] = actor.tracker.posterior
    return result


def means(result):
    return dict(zip(METRICS, map(float, result['metrics'].mean(0))))


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f'Refusing to overwrite {path}')
    np.savez_compressed(path, **result)


def self_test():
    rng = np.random.default_rng(729001)
    for learner in ('canonical', 'tabular'):
        env = AllocationMarket(np.arange(16), learner_mode=learner)
        uniform = CapacityMarket(np.arange(16), learner_mode=learner)
        actor = AllocationActor(16)
        for _ in range(30):
            actions = rng.choice(PRIMARY_ACTION_INDICES, 16)
            u = rng.random((16, 3))
            (a, am), _ = env.step_with_selector(actions, u, actor.choose)
            b, bm = uniform.step(actions, u)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            for key in am:
                np.testing.assert_array_equal(am[key], bm[key])
        env = AllocationMarket(np.arange(16), learner_mode=learner)
        actor = AllocationActor(16, 'posterior', learner)
        for _ in range(30):
            actions = np.zeros(16, dtype=int)
            (obs, _), _ = env.step_with_selector(actions, rng.random((16, 3)), actor.choose)
            temporary = actor.temporary_posterior.copy()
            actor.observe(actions, obs['reports'], obs['owners'])
            np.testing.assert_allclose(temporary, actor.tracker.posterior, atol=1e-14)
        bank = actor.tracker._candidates
        for model in range(16):
            np.testing.assert_allclose(bank._w_independent[model * 16 + model],
                                       env._w_independent[model], atol=1e-12)
            np.testing.assert_allclose(bank._w_grand[model * 16 + model],
                                       env._w_grand[model], atol=1e-12)
    for price, alpha in ((PRICE, 1.), (1., .2)):
        tables = pricing_tables(price, alpha)
        for market in range(4):
            for action in PRIMARY_ACTION_INDICES:
                for report in range(16):
                    count = tables['counts'][action, report]
                    utility = tables['utility'][market, 1, action, report, :count].sum(-1)
                    np.testing.assert_allclose(utility, utility[0], atol=1e-12)
    a = np.zeros(16, int)
    reports = JOINT_REPORTS.copy()
    for buyer in (0, 1):
        actor = AllocationActor(16, f'priority{buyer}')
        chosen = actor.choose(a, reports, rng.random(16))
        for report in range(16):
            own = TABLES['owners'][0, report, :TABLES['counts'][0, report], 0]
            if buyer in own:
                assert TABLES['owners'][0, report, chosen[report], 0] == buyer
    # Check the resampling estimator against a constant paired difference.
    base = np.zeros((2, 4, 3, len(METRICS)))
    base[1] = 7.
    draws = bootstrap(base, 73, 123)
    np.testing.assert_allclose(draws[1] - draws[0], 7.)
    print('PASS: uniform replay, candidate-state replay, once-only Bayes update, '
          'grand-ownership invariance, static priorities, paired stratified bootstrap.', flush=True)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        raise RuntimeError('Exploratory protocol already exists; do not overwrite')
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'exploratory_new_hypothesis_after_v2_results',
        'relation_to_v2': 'Separate exploratory evidence; V2 primary p=.78 judgment remains unchanged.',
        'source_hashes': current_hashes(), 'unit_price': PRICE, 'capacity_weight': 1.,
        'dynamic_rho': 0., 'dynamic_horizon': HORIZON,
        'matching_rule': 'Among maximum-capacity matchings, maximize current-report-updated posterior expected completed jobs; break ties by expected completed value; remaining ties use shared u2.',
        'posterior': 'Uniform prior on 16 fixed models; CapacityBeliefTracker public report/owner replay; no true service/private utilities in actor.',
        'update_order': 'Sample reports privately; temporary report likelihood posterior; choose matching; realize; observe once to update posterior and all hypothetical learner states.',
        'static_candidates': STATIC_METHODS,
        'static_priority_definition': 'Prefer buyer 0 or buyer 1 as owner of 3-unit VM when feasible among maximum-capacity outcomes; otherwise random among remaining ties.',
        'validation_seed': VALIDATION_SEED, 'test_seed': TEST_SEED,
        'worlds_per_fixed_type': WORLDS, 'test_cases': CASES,
        'validation_case': 'canonical only; one selected global static used for all test cases',
        'static_selection': 'Max revenue among 12 candidates whose validation mean R/J/V all >= old fixed .5; deterministic candidate order breaks ties. If none feasible, retain the candidate with largest minimum normalized R/J/V slack and flag infeasible; never reselect on test.',
        'test_methods': ['dynamic', 'old_fixed', 'validation-selected best static',
                         'uniform_r0', 'priority0_r0', 'priority1_r0'],
        'sampling': 'All methods share within-case exogenous (report, matching) uniforms; 16 equally weighted fixed types, or 4 fixed markets for exact grand.',
        'training': 'No training; no training seeds or artificial seed replication.',
        'primary_exploratory_comparisons': ['dynamic minus selected best static', 'dynamic minus old fixed'],
        'endpoints': ['revenue', 'completed_jobs', 'completed_value'],
        'joint_criterion': 'Each contrast requires R one-sided 95% lower > 0 and J/V one-sided 95% lower >= 0; numerical tolerance 1e-9 only.',
        'bootstrap': '10,000 paired resamples of worlds within every fixed type; same draws shared across methods; no type or training-seed resampling.',
        'bootstrap_seed_base': BOOTSTRAP_SEED, 'bootstrap_replicates': REPLICATES,
        'groups': 'all, independent, coalition, each market, each fixed type; subgroup claims exploratory',
        'limits': ['No reproduction of a trained 2025 steering algorithm.',
                   'Finite and correctly specified learning-model catalogue; exact-grand filter is deliberately misspecified.',
                   'Because each tested rule uses a constant rho, canonical and tabular learners are mathematically equivalent on the visited single eligibility state; these are reproducibility checks, not independent generalization evidence.',
                   'For grand pooling, ownership cannot alter one-stage utility; any added matching benefit is primarily independent-buyer coordination.',
                   'No truthfulness, model-free collusion resistance, or out-of-catalogue guarantee.'],
        'test_read_before_freeze': False}
    write_json(path, protocol)
    print('Exploratory protocol frozen', flush=True)


def validate():
    check_protocol()
    selection_path = OUT / 'selection.json'
    if selection_path.exists() or (OUT / 'pretest_freeze.json').exists():
        raise RuntimeError('Validation selection already complete')
    records = []
    start = time.perf_counter()
    for method in ('old_fixed', *STATIC_METHODS):
        result = run(method, 'canonical', VALIDATION_SEED)
        save(OUT / 'data' / 'validation' / f'{method}.npz', result)
        row = {'method': method, **means(result)}
        records.append(row)
        print(json.dumps({'validation': row, 'seconds': time.perf_counter() - start}), flush=True)
    old = records[0]
    feasible = [r for r in records[1:] if all(r[key] >= old[key] - TOL
                for key in ('revenue', 'completed_jobs', 'completed_value'))]
    if feasible:
        best = max(feasible, key=lambda r: r['revenue'])
    else:
        best = max(records[1:], key=lambda r: (min((r[key] - old[key]) / scale for key, scale
                   in (('revenue', 2000), ('completed_jobs', 800), ('completed_value', 5600))), r['revenue']))
    document = {'validation_seed': VALIDATION_SEED, 'case': 'canonical',
        'records': records, 'selected_method': best['method'],
        'selected_validation_feasible': bool(feasible),
        'feasible_methods': [r['method'] for r in feasible]}
    write_json(selection_path, document)
    write_csv(OUT / 'validation.csv', records)
    print(json.dumps(document), flush=True)


def test():
    check_protocol()
    selected = json.loads((OUT / 'selection.json').read_text(encoding='utf-8'))['selected_method']
    path = OUT / 'pretest_freeze.json'
    if path.exists():
        raise RuntimeError('Test already frozen; do not overwrite')
    files = [OUT / 'protocol.json', OUT / 'selection.json',
             *sorted((OUT / 'data' / 'validation').glob('*.npz'))]
    write_json(path, {'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_hashes': current_hashes(),
        'input_hashes': {p.relative_to(ROOT).as_posix(): digest(p) for p in files},
        'test_data_read': False, 'selected_method': selected})
    methods = tuple(dict.fromkeys(('dynamic', 'old_fixed', selected,
                                 'uniform_r0', 'priority0_r0', 'priority1_r0')))
    start = time.perf_counter()
    for case in CASES:
        for method in methods:
            result = run(method, case, TEST_SEED)
            save(OUT / 'data' / 'test' / case / f'{method}.npz', result)
            print(json.dumps({'test_case': case, 'method': method, **means(result),
                              'seconds': time.perf_counter() - start}), flush=True)


def bootstrap(data, replicates=REPLICATES, seed=BOOTSTRAP_SEED):
    """data: methods x fixed types x paired worlds x metrics; no trained seeds."""
    methods, types, worlds, metrics = data.shape
    rng = np.random.default_rng(seed)
    result = np.empty((methods, replicates, types, metrics))
    for t in range(types):
        counts = rng.multinomial(worlds, np.full(worlds, 1 / worlds), size=replicates)
        values = data[:, t].transpose(1, 0, 2).reshape(worlds, methods * metrics)
        result[:, :, t] = (counts @ values / worlds).reshape(replicates, methods, metrics).transpose(1, 0, 2)
    return result


def analyze():
    check_protocol()
    frozen = json.loads((OUT / 'pretest_freeze.json').read_text(encoding='utf-8'))
    if frozen['source_hashes'] != current_hashes():
        raise RuntimeError('Source hashes do not match pretest freeze')
    for name, value in frozen['input_hashes'].items():
        if digest(ROOT / name) != value:
            raise RuntimeError(f'Frozen input changed: {name}')
    selected = frozen['selected_method']
    methods = tuple(dict.fromkeys(('dynamic', 'old_fixed', selected,
                                 'uniform_r0', 'priority0_r0', 'priority1_r0')))
    summary = {'status': 'exploratory_completed', 'selected_static': selected,
        'trained_policy_count': 0, 'bootstrap_replicates': REPLICATES, 'cases': {},
        'interpretation': 'Primary V2 remains unchanged. This is posterior-guided matching, not trained threshold steering.'}
    metric_rows = []; comparison_rows = []; type_rows = []; hashes = {}
    for case_number, case in enumerate(CASES):
        ids = ids_for(case)
        types = np.unique(ids)
        data = []
        for method in methods:
            path = OUT / 'data' / 'test' / case / f'{method}.npz'
            with np.load(path, allow_pickle=False) as file:
                np.testing.assert_array_equal(file['models'], ids)
                np.testing.assert_array_equal(file['metric_names'], METRICS)
                metrics = file['metrics']
                assert metrics.shape == (len(ids), len(METRICS))
                assert int(file['stream_seed']) == TEST_SEED
                assert np.isfinite(metrics).all()
                np.testing.assert_allclose(metrics[:, 0] + metrics[:, 4], metrics[:, 3], atol=2e-8)
                data.append(metrics.reshape(len(types), WORLDS, len(METRICS)))
            hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        data = np.stack(data)
        draws = bootstrap(data, seed=BOOTSTRAP_SEED + case_number)
        points = data.mean(2)
        groups = {'all': np.arange(len(types)),
                  'coalition': np.flatnonzero((types // 2) % 2 == 1),
                  'independent': np.flatnonzero((types // 2) % 2 == 0)}
        groups.update({f'market_{"ABCD"[m]}': np.flatnonzero(types // 4 == m) for m in range(4)})
        groups = {g: ix for g, ix in groups.items() if len(ix)}
        result = {'fixed_types': types.tolist(), 'methods': {}, 'comparisons': {}}
        for m, method in enumerate(methods):
            result['methods'][method] = {}
            for group, ix in groups.items():
                stats = {key: estimate(points[m, ix, col].mean(), draws[m, :, ix, col].mean(0))
                         for col, key in enumerate(METRICS)}
                result['methods'][method][group] = stats
                for key, record in stats.items():
                    metric_rows.append({'case': case, 'method': method, 'group': group, 'metric': key, **record})
            for t, model in enumerate(types):
                type_rows.append({'case': case, 'method': method, 'model_id': int(model),
                                  **dict(zip(METRICS, map(float, points[m, t])))})
        for reference in tuple(dict.fromkeys((selected, 'old_fixed', 'uniform_r0'))):
            ref = methods.index(reference)
            result['comparisons'][reference] = {}
            for group, ix in groups.items():
                records = {}
                for col, key in enumerate(METRICS):
                    point = (points[0, ix, col] - points[ref, ix, col]).mean()
                    distribution = (draws[0, :, ix, col] - draws[ref, :, ix, col]).mean(0)
                    records[key] = estimate(point, distribution)
                passed = {'revenue': records['revenue']['one_sided95_low'] > TOL,
                    'completed_jobs': records['completed_jobs']['one_sided95_low'] >= -TOL,
                    'completed_value': records['completed_value']['one_sided95_low'] >= -TOL}
                result['comparisons'][reference][group] = {
                    'endpoints': records, 'criteria': passed, 'joint_pass': all(passed.values())}
                for key, record in records.items():
                    comparison_rows.append({'case': case, 'method': 'dynamic', 'reference': reference,
                        'group': group, 'metric': key, **record, 'joint_pass': all(passed.values())})
        summary['cases'][case] = result
    write_json(OUT / 'summary.json', summary)
    write_csv(OUT / 'metrics.csv', metric_rows)
    write_csv(OUT / 'comparisons.csv', comparison_rows)
    write_csv(OUT / 'type_results.csv', type_rows)
    write_json(OUT / 'completion_audit.json', {'status': 'complete',
        'source_hashes_match': True, 'frozen_validation_hashes_match': True,
        'test_file_hashes': hashes, 'bootstrap_type_resampling': False,
        'training_seed_resampling': False, 'test_reselection': False})
    print(json.dumps({case: {ref: summary['cases'][case]['comparisons'][ref]['all']
        for ref in summary['cases'][case]['comparisons']} for case in CASES}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('self-test', 'prepare', 'validation', 'test', 'analyze', 'all'))
    args = parser.parse_args()
    if args.command == 'all':
        self_test(); prepare(); validate(); test(); analyze()
    elif args.command == 'self-test':
        self_test()
    elif args.command == 'prepare':
        prepare()
    elif args.command == 'validation':
        validate()
    elif args.command == 'test':
        test()
    else:
        analyze()


if __name__ == '__main__':
    main()
