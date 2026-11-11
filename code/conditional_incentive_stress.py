"""V5 independent conditional-reward and full-pool transferable-utility audits.

All interfaces here are private evaluator/oracle interfaces, never actor input.
V4 files and outputs remain read-only. Same-report-group TU and the stronger
cross-report full-pool coalition are distinguished in every result.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from functools import lru_cache
from itertools import permutations, product
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import conditional_incentive_market as market
import incentive_stress as reference
from analyze_threshold_experiment import write_json, write_csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/v5_conditional_incentive/exact_audit'
TOL = 1e-10
MODES = ('bounded', 'transferable', 'none')
REPORTS = np.array(list(product(range(4), repeat=3)), dtype=int)
SOURCE_NAMES = ('conditional_incentive_stress.py', 'conditional_incentive_market.py',
                'incentive_stress.py', 'incentive_market.py', 'validate_micro_model.py',
                'analyze_threshold_experiment.py')
FULLPOOL_METRICS = ('gross_revenue', 'cost', 'net_revenue', 'jobs', 'value', 'utility',
                   'underreport_count', 'severity', 'overreport_count', 'active',
                   'external_coalition_members', 'reallocated', 'postallocation_joint_gain')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes():
    return {f'code/{name}': digest(ROOT / 'code' / name) for name in SOURCE_NAMES}


def ref_effective(advertised, reports):
    """Independent eligibility; only public reports, never true demands."""
    if advertised < 16:
        return advertised
    threshold = 2 if advertised < 28 else 3
    offset = (advertised - 16) % 12
    target = offset // 4
    return offset + 1 if reports[target] >= threshold else 0


@lru_cache(maxsize=1440)
def stage_table(physical_id, advertised_action_id, trade_mode='transferable'):
    return market.expected_stage(int(physical_id), int(advertised_action_id), trade_mode)


@lru_cache(maxsize=3)
def exact_action_cache(trade_mode='transferable'):
    result = np.empty((12, 40, 3), dtype=int)
    for physical in range(12):
        for advertised in range(40):
            table = stage_table(physical, advertised, trade_mode)
            result[physical, advertised] = table['action_profiles'][reference.best_profile(table)]
    result.flags.writeable = False
    return result


def _ids(physical_ids, advertised_action_ids):
    physical = np.asarray(physical_ids)
    advertised = np.asarray(advertised_action_ids)
    if physical.ndim != 1 or not np.issubdtype(physical.dtype, np.integer):
        raise ValueError('physical_ids must be a one-dimensional integer array')
    if advertised.ndim == 0:
        advertised = np.full(physical.shape, advertised.item())
    if (advertised.shape != physical.shape or not np.issubdtype(advertised.dtype, np.integer)
            or np.any((physical < 0) | (physical >= 12))
            or np.any((advertised < 0) | (advertised >= 40))):
        raise ValueError('Invalid physical or advertised reward IDs')
    return physical.astype(int), advertised.astype(int)


def exact_best_actions(physical_ids, advertised_action_ids, trade_mode='transferable'):
    """Same-report-group oracle; do not label this arbitrary full-pool TU."""
    physical, advertised = _ids(physical_ids, advertised_action_ids)
    return exact_action_cache(trade_mode)[physical, advertised].copy()


def stage_regret(physical_ids, advertised_action_ids, probabilities, trade_mode='transferable'):
    physical, advertised = _ids(physical_ids, advertised_action_ids)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(physical), 3, 7) or np.any(probabilities < 0):
        raise ValueError('Expected nonnegative probabilities (B,3,7)')
    np.testing.assert_allclose(probabilities.sum(-1), 1., atol=1e-10)
    joint = np.einsum('bi,bj,bk->bijk', probabilities[:, 0], probabilities[:, 1],
                      probabilities[:, 2]).reshape(-1, 343)
    expected = np.empty(len(physical)); optimum = np.empty(len(physical))
    for p, a in sorted(set(zip(physical.tolist(), advertised.tolist()))):
        rows = np.flatnonzero((physical == p) & (advertised == a))
        utilities = stage_table(p, a, trade_mode)['utilities'].sum(1)
        expected[rows] = joint[rows] @ utilities
        optimum[rows] = utilities.max()
    return {'learned_joint_utility': expected, 'best_joint_utility': optimum,
            'profitable_joint_deviation': np.maximum(0., optimum - expected)}


@lru_cache(maxsize=64)
def fullpool_initial(report_id):
    reports = tuple(int(x) for x in REPORTS[report_id])
    owners = reference.ref_matchings(reports)
    capacities = []; payments = []
    for allocation in owners:
        initial = [0, 0, 0]
        for capacity, owner in zip((3, 2), allocation):
            if owner >= 0:
                initial[owner] = capacity
        capacities.append(initial)
        payments.append([4 * c + 16 * b if c else 0 for c, b in zip(initial, reports)])
    return owners, np.array(capacities, int), np.array(payments, int)


@lru_cache(maxsize=480)
def fullpool_table(physical, advertised):
    """All 64 report profiles, with post-allocation full-coalition optimization.

    Membership is exogenous: all three buyers may share across reports, including
    a member reporting zero. This explicitly relaxes the same-k coalition model.
    Each task still requires one whole VM; no fractional jobs or extra supply.
    The coalition permits arbitrary balanced transfers and weak participation.
    """
    demands, value_units = reference.ref_type(physical)
    demands = np.asarray(demands)
    values = np.asarray(value_units, dtype=float) / 20
    shape = (64, 6)
    counts = np.empty(64, int)
    initial_all = np.zeros(shape + (3,), int)
    payments_all = np.zeros(shape + (3,), float)
    final_all = np.zeros(shape + (3,), int)
    bonuses_all = np.zeros(shape + (3,), float)
    completed_all = np.zeros(shape + (3,), bool)
    outcomes = {name: np.zeros(shape) for name in FULLPOOL_METRICS}
    for rid, reports in enumerate(REPORTS):
        _, initial_list, payment_units = fullpool_initial(rid)
        counts[rid] = len(initial_list)
        effective = ref_effective(advertised, reports)
        bonus_units = np.array(reference.REF_BONUSES[effective], int)
        effective_value = np.asarray(value_units) + bonus_units
        for match, (initial, bills) in enumerate(zip(initial_list, payment_units)):
            best = None
            for candidate in sorted(set(permutations(tuple(int(x) for x in initial)))):
                final = np.asarray(candidate)
                completed = final >= demands
                cost_units = int(np.dot(completed, bonus_units))
                total_utility = int(np.dot(completed, effective_value) - bills.sum())
                # The binding coalition can choose weak-IR zero-surplus ties.
                rank = (total_utility, cost_units)
                if best is None or rank > best[0]:
                    best = (rank, final.copy(), completed.copy())
            (_, _), final, completed = best
            gross = bills.sum() / 20
            bonus = bonus_units * completed / 20
            cost = float(bonus.sum())
            value = float(np.dot(values, completed))
            baseline = float(np.dot(initial >= demands, effective_value) - bills.sum()) / 20
            utility = value - gross + cost
            assert utility >= baseline - TOL
            initial_all[rid, match] = initial
            payments_all[rid, match] = bills / 20
            final_all[rid, match] = final
            completed_all[rid, match] = completed
            bonuses_all[rid, match] = bonus
            active = reports > 0
            record = dict(zip(FULLPOOL_METRICS, (
                gross, cost, gross - cost, int(completed.sum()), value, utility,
                int(np.sum(active & (reports < demands))),
                int(np.sum(np.maximum(demands - reports, 0) * active)),
                int(np.sum(active & (reports > demands))), int(active.sum()),
                3, int(np.any(initial != final)), utility - baseline)))
            for key, number in record.items():
                outcomes[key][rid, match] = number
    expected = {name: values.sum(1) / counts for name, values in outcomes.items()}
    candidates = np.flatnonzero(expected['utility'] >= expected['utility'].max() - TOL)
    net = expected['net_revenue']
    candidates = candidates[net[candidates] <= net[candidates].min() + TOL]
    under = expected['underreport_count']
    candidates = candidates[under[candidates] >= under[candidates].max() - TOL]
    chosen = int(candidates[0])
    return {'reports': REPORTS, 'matching_counts': counts, 'initial_capacities': initial_all,
        'payments': payments_all, 'final_capacities': final_all, 'completed': completed_all,
        'bonuses': bonuses_all, 'outcomes': outcomes, 'expected': expected,
        'chosen_report_id': chosen}


def fullpool_stage_step(physical_ids, advertised_action_ids, u_match):
    """Stateless full-pool strong-BR replay: returns ONLY public data and metrics.

    Choose reports before the matching random draw, then share optimally after
    matching. No Q learning occurs. The actor must not receive the physical IDs
    or the private metrics, and the parent enforces its ordinary budget mask.
    """
    physical, advertised = _ids(physical_ids, advertised_action_ids)
    u = np.asarray(u_match, dtype=float)
    if u.shape != physical.shape or not np.isfinite(u).all() or np.any((u < 0) | (u >= 1)):
        raise ValueError('u_match must be a uniform vector in [0,1)')
    size = len(physical)
    public = {'reports': np.zeros((size, 3), int), 'allocated_capacities': np.zeros((size, 3), int),
              'payments': np.zeros((size, 3)), 'completed': np.zeros((size, 3), bool),
              'bonuses': np.zeros((size, 3))}
    metrics = {name: np.empty(size) for name in FULLPOOL_METRICS}
    for p, a in sorted(set(zip(physical.tolist(), advertised.tolist()))):
        rows = np.flatnonzero((physical == p) & (advertised == a))
        table = fullpool_table(p, a)
        rid = table['chosen_report_id']
        match = np.floor(u[rows] * table['matching_counts'][rid]).astype(int)
        public['reports'][rows] = REPORTS[rid]
        for destination, source in (('allocated_capacities', 'initial_capacities'),
                                    ('payments', 'payments'), ('completed', 'completed'), ('bonuses', 'bonuses')):
            public[destination][rows] = table[source][rid, match]
        for key in metrics:
            metrics[key][rows] = table['outcomes'][key][rid, match]
    return public, metrics


def self_test():
    qualification_checks = 0
    for advertised in range(40):
        actual = market.effective_bonus_action_ids(advertised, market.ACTION_PROFILES)
        expected = [ref_effective(advertised, [reference.REF_ACTIONS[a][0] for a in profile])
                    for profile in market.ACTION_PROFILES]
        np.testing.assert_array_equal(actual, expected)
        qualification_checks += len(actual)
    rng = np.random.default_rng(859001)
    # Independent scalar outcome/averaging checks under the conditional rules.
    checked = 0
    for mode in MODES:
        for _ in range(512):
            physical, advertised, profile = int(rng.integers(12)), int(rng.integers(40)), int(rng.integers(343))
            actions = market.ACTION_PROFILES[profile]
            reports = [reference.REF_ACTIONS[a][0] for a in actions]
            effective = ref_effective(advertised, reports)
            owners = reference.ref_matchings(tuple(reports))
            results = [reference.ref_resolve(physical, actions, effective, owner, mode) for owner in owners]
            expected = np.mean([x['utilities'] for x in results], axis=0)
            np.testing.assert_allclose(stage_table(physical, advertised, mode)['utilities'][profile], expected, atol=1e-12)
            checked += 1
        ids = np.arange(120) % 24
        env = market.Market(ids, mode)
        ref = market.base.Market(ids, mode, use_disk_cache=False)
        env.q[:] = rng.normal(0, .4, env.q.shape)
        ref.q[:] = env.q
        u = rng.random((len(ids), 4)); advertised = np.arange(len(ids)) % 40
        q = env.q.copy()
        probability = np.exp((q - q.max(2, keepdims=True)) / .2)
        probability /= probability.sum(2, keepdims=True)
        cdf = np.cumsum(probability, 2); cdf[:, :, -1] = 1
        actions = (u[:, :3, None] >= cdf).sum(2)
        effective = np.array([ref_effective(a, [reference.REF_ACTIONS[b][0] for b in profile])
                              for a, profile in zip(advertised, actions)])
        calls = [0]
        original = env.probabilities
        def counted():
            calls[0] += 1
            return original()
        env.probabilities = counted
        outcome = env.step(advertised, u)
        comparator = ref.step(effective, u, buyer_actions=actions)
        assert calls[0] == 1 and env.step_count == 1
        np.testing.assert_array_equal(env.q, ref.q)
        for left, right in zip(outcome, comparator):
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])
    # All report-qualification decisions are independent of physical type.
    for physical in range(12):
        table = fullpool_table(physical, 0)
        rid = table['chosen_report_id']
        np.testing.assert_allclose(table['expected']['gross_revenue'][rid], 2.6, atol=1e-12)
        assert table['expected']['jobs'][rid] == 2
    public, metrics = fullpool_stage_step(np.arange(12), 0, rng.random(12))
    assert set(public) == market.PUBLIC_KEYS
    np.testing.assert_allclose(metrics['utility'] + metrics['net_revenue'], metrics['value'], atol=1e-12)
    # Full pooling contains every same-report group allocation, so its maximum
    # total utility cannot be smaller. Reports are chosen before matching.
    for physical, advertised in ((0, 16), (3, 31), (6, 35), (9, 39)):
        strongest = fullpool_table(physical, advertised)['expected']['utility'].max()
        restricted = stage_table(physical, advertised, 'transferable')['utilities'].sum(1).max()
        assert strongest >= restricted - TOL
    print(f'PASS: {qualification_checks} independent public eligibility cases, {checked} '
          'independent payoff averages; nonzero-Q single sample/update; fullpool cash/response bounds.', flush=True)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / 'protocol.json'
    if destination.exists():
        raise RuntimeError('V5 stage protocol already exists')
    write_json(destination, {'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_hashes': hashes(), 'status': 'stage_audit_frozen_before_enumeration',
        'advertised_actions': 40, 'physical_types': 12, 'same_report_profiles': 343,
        'fullpool_report_profiles': 64, 'same_report_modes': MODES,
        'qualification': 'Original public report >= advertised threshold, then own task completed; no true demand in eligibility.',
        'nash': 'All pure 7x7x7 stage equilibria; not mixed or repeated equilibria.',
        'restricted_ties': 'Same as V4: max U, min net, min gross, max successful-underreport members, max underreport, lexicographic.',
        'fullpool': 'All three buyers form an external binding coalition, share across all reports (including opt-out), choose reports before random A0 allocation, then reassign whole VMs and balanced transfers; weak-IR ties allowed.',
        'fullpool_ties': 'Post-allocation: maximum U, maximum paid bonus (minimum net), lexicographic capacities. Report choice: maximum expected U, minimum expected net, maximum underreport count, lexicographic reports.',
        'fullpool_metrics': FULLPOOL_METRICS,
        'no_formation_claim_for_fullpool': True,
        'reward_threshold': 'Minimum among the declared .25/.5/1/2 amounts that removes all-join-report1 from pure equilibria, reported separately by type, target and public threshold.',
        'platform_policies_evaluated': False, 'v4_outputs_modified': False,
        'known_limits': ['One-stage best responses do not cover all history manipulation.',
            'Breaking one all-low equilibrium does not eliminate all low-report/collusive equilibria.',
            'Higher reports may be overreports rather than truthful demand.',
            'Minimum menu amount is not globally minimum incentive cost.']})
    print('V5 exact-stage audit frozen', flush=True)


def audit():
    frozen = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    if hashes() != frozen['source_hashes']:
        raise RuntimeError('Frozen V5 audit sources changed')
    if (OUT / 'summary.json').exists():
        raise RuntimeError('V5 stage results already exist')
    start = time.perf_counter()
    rows = []; details = []; equilibria_rows = []; full_rows = []; minima = []
    for mode in MODES:
        for physical in range(12):
            m, _ = reference.ref_type(physical)
            truth = reference.profile_id([5 if x == 3 else 3 for x in m])
            for advertised in range(40):
                table = stage_table(physical, advertised, mode)
                best = reference.best_profile(table)
                nash, regret = reference.pure_nash(table)
                optimal = reference.row(table, best)
                low = reference.row(table, 114)
                honest = reference.row(table, truth)
                max_indices = np.flatnonzero(table['utilities'].sum(1) >= optimal['utility'] - TOL)
                item = {'mode': mode, 'physical_id': physical, 'advertised_action': advertised,
                    'threshold': int(market.REWARD_THRESHOLDS[advertised]),
                    'target': int(market.TARGET_BUYERS[advertised]),
                    'promised_sum': float(market.BONUS_ACTIONS[advertised].sum()),
                    'all1_is_pure_nash': bool(114 in nash), 'all1_unilateral_regret': float(regret[114].max()),
                    'pure_nash_count': len(nash), 'truth_unilateral_regret': float(regret[truth].max()),
                    'best_profile_id': best,
                    **{f'best_{key}': optimal[key] for key in market.METRIC_NAMES},
                    'joint_gain_over_truthful': optimal['utility'] - honest['utility'],
                    'all1_joint_utility': low['utility'],
                    'maximizer_min_net_revenue': float(table['metrics']['net_revenue'][max_indices].min()),
                    'maximizer_max_overreport_count': float(table['metrics']['overreport_count'][max_indices].max())}
                rows.append(item)
                details.append({'mode': mode, 'physical_id': physical, 'advertised_action': advertised,
                                'best': optimal, 'all1': low, 'truthful': honest,
                                'pure_nash_ids': nash.tolist()})
                for profile in nash:
                    record = reference.row(table, profile)
                    equilibria_rows.append({'mode': mode, 'physical_id': physical,
                        'advertised_action': advertised, 'profile_id': int(profile),
                        'reports': str(record['reports']), 'joins': str(record['joins']),
                        **{key: record[key] for key in market.METRIC_NAMES}})
                if mode == 'transferable':
                    full = fullpool_table(physical, advertised)
                    rid = full['chosen_report_id']; expected = full['expected']
                    assert expected['utility'][rid] >= optimal['utility'] - TOL
                    winners = np.flatnonzero(expected['utility'] >= expected['utility'][rid] - TOL)
                    full_rows.append({'physical_id': physical, 'advertised_action': advertised,
                        'threshold': int(market.REWARD_THRESHOLDS[advertised]),
                        'target': int(market.TARGET_BUYERS[advertised]),
                        'chosen_reports': str(REPORTS[rid].tolist()),
                        **{key: float(expected[key][rid]) for key in FULLPOOL_METRICS},
                        'worst_optimal_net_revenue': float(expected['net_revenue'][winners].min()),
                        'max_optimal_overreport_count': float(expected['overreport_count'][winners].max()),
                        'advantage_over_restricted_TU': float(expected['utility'][rid] - optimal['utility'])})
        print(f'V5 stage audit completed: {mode}', flush=True)
    for mode in ('bounded', 'transferable'):
        for physical in range(12):
            for threshold in (2, 3):
                for target in range(3):
                    candidates = [r for r in rows if r['mode'] == mode and r['physical_id'] == physical
                                  and r['threshold'] == threshold and r['target'] == target]
                    broken = [r for r in candidates if not r['all1_is_pure_nash']]
                    selected = min(broken, key=lambda r: r['promised_sum']) if broken else None
                    minima.append({'mode': mode, 'physical_id': physical, 'threshold': threshold,
                        'target': target, 'minimum_menu_amount': selected['promised_sum'] if selected else None,
                        'action': selected['advertised_action'] if selected else None,
                        'restricted_oracle_net_revenue': selected['best_net_revenue'] if selected else None,
                        'restricted_oracle_cost': selected['best_cost'] if selected else None,
                        'qualification_is_truthfulness_test': False})
    # Specific public-qualification, overreport, resource-swap and reward witness.
    # Physical 0: buyer1 needs2 but reports3; buyer0 needs3 but reports1.
    physical, advertised = 0, 34  # buyer1, threshold3, completion reward1
    rid = int(np.flatnonzero(np.all(REPORTS == [1, 3, 1], axis=1))[0])
    table = fullpool_table(physical, advertised)
    outcomes = []
    owners = fullpool_initial(rid)[0]
    for k in range(table['matching_counts'][rid]):
        outcomes.append({'original_owners': list(owners[k]),
            'initial_capacities': table['initial_capacities'][rid, k].tolist(),
            'final_capacities': table['final_capacities'][rid, k].tolist(),
            'payments': table['payments'][rid, k].tolist(),
            'completed': table['completed'][rid, k].tolist(),
            'bonuses': table['bonuses'][rid, k].tolist(),
            **{key: float(value[rid, k]) for key, value in table['outcomes'].items()}})
    witness = {'physical_id': physical, 'advertised_action': advertised,
        'reports': [1, 3, 1], 'true_demands_for_audit_only': [3, 2, 2],
        'statement': 'Buyer1 overreports3, may receive a 2-VM after cross-report pooling, completes, and receives the public-threshold reward. No eligibility check reads true demand.',
        'outcomes': outcomes, 'expected': {key: float(value[rid]) for key, value in table['expected'].items()}}
    write_json(OUT / 'summary.json', {'status': 'complete', 'source_hashes': hashes(),
        'records': rows, 'fullpool': full_rows, 'minimum_menu_thresholds': minima,
        'elapsed_seconds': time.perf_counter() - start, 'platform_policies_evaluated': False})
    write_json(OUT / 'stage_details.json', details)
    write_json(OUT / 'overreport_reward_swap_witness.json', witness)
    write_csv(OUT / 'stage_summary.csv', rows)
    write_csv(OUT / 'pure_nash.csv', equilibria_rows)
    write_csv(OUT / 'fullpool_summary.csv', full_rows)
    write_csv(OUT / 'minimum_menu_thresholds.csv', minima)
    write_json(OUT / 'completion_audit.json', {'status': 'complete',
        'source_hashes_match': hashes() == frozen['source_hashes'], 'restricted_configurations': len(rows),
        'fullpool_configurations': len(full_rows), 'platform_policies_evaluated': False,
        'v4_outputs_modified': False})
    print(json.dumps({'restricted_configurations': len(rows), 'fullpool_configurations': len(full_rows),
                      'elapsed_seconds': time.perf_counter() - start}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('self-test', 'freeze', 'audit', 'all'))
    args = parser.parse_args()
    if args.command == 'all':
        self_test(); freeze(); audit()
    elif args.command == 'self-test':
        self_test()
    elif args.command == 'freeze':
        freeze()
    else:
        audit()


if __name__ == '__main__':
    main()
