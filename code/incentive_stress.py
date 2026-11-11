"""Independent finite-stage audits for completion-only V4 incentives.

The base allocation and invoice are fixed. This file owns no platform policy.
The exact-response and regret functions are privileged EVALUATION interfaces;
they must never be called by a deployed actor. Platform stress rollouts require
the parent's separate pretest freeze; the standalone audit only enumerates the
finite one-stage game and does not evaluate or select a trained platform.
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
import incentive_market as market
from analyze_threshold_experiment import write_json, write_csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments' / 'v4_incentive' / 'exact_audit'
TOL = 1e-10
MODES = ('bounded', 'transferable', 'none')
AUDIT_SEED = 930001
# Independently declared constants, not aliases of simulator arrays.
REF_ACTIONS = tuple([(0, 0)] + [(b, j) for b in (1, 2, 3) for j in (0, 1)])
REF_BONUSES = tuple([(0, 0, 0)] + [tuple(v if i == j else 0 for i in range(3))
    for j in range(3) for v in (5, 10, 20, 40)] + [(v, v, v) for v in (5, 10, 20)])
REF_VALUES = ((120, 40, 40), (80, 60, 60), (180, 100, 100), (48, 32, 32))
REF_Q = (4, 8, 12, 16, 19)
SOURCE_NAMES = ('incentive_stress.py', 'incentive_market.py',
                'validate_micro_model.py', 'analyze_threshold_experiment.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {f'code/{name}': digest(ROOT / 'code' / name) for name in SOURCE_NAMES}


def ref_type(physical):
    rotation = physical % 3
    demands = tuple([3 if i == rotation else 2 for i in range(3)])
    base = REF_VALUES[physical // 3]
    values = tuple(base[(i - rotation) % 3] for i in range(3))
    return demands, values


def ref_matchings(reports, supply=(3, 2)):
    """Independent enumeration; deliberately does not call the allocation code."""
    possible = []
    for owners in product(range(-1, 3), repeat=len(supply)):
        used = [i for i in owners if i >= 0]
        if len(used) != len(set(used)):
            continue
        if any(i >= 0 and not 0 < reports[i] <= c for c, i in zip(supply, owners)):
            continue
        possible.append((sum(c for c, i in zip(supply, owners) if i >= 0), owners))
    best = max(volume for volume, _ in possible)
    return tuple(owners for volume, owners in possible if volume == best)


def ref_resolve(physical, action_ids, bonus, owners, mode='bounded', supply=(3, 2)):
    """Scalar integer-twentieth reference, independently written from the paper rules."""
    reports = tuple(REF_ACTIONS[a][0] for a in action_ids)
    joins = tuple(REF_ACTIONS[a][1] for a in action_ids)
    if tuple(owners) not in ref_matchings(reports, supply):
        raise ValueError('Not an original maximum-capacity allocation')
    demands, values = ref_type(physical)
    initial = [0, 0, 0]
    for capacity, owner in zip(supply, owners):
        if owner >= 0:
            initial[owner] = capacity
    bills = [4 * c + 16 * b if c > 0 else 0 for c, b in zip(initial, reports)]
    effective = [v + b for v, b in zip(values, REF_BONUSES[bonus])]
    before = [effective[i] * (initial[i] >= demands[i]) - bills[i] for i in range(3)]
    group = ()
    for bid in (1, 2, 3):
        members = tuple(i for i in range(3) if reports[i] == bid and joins[i])
        if len(members) >= 2:
            group = members
            break
    final = initial[:]
    transfers = [0., 0., 0.]
    gain = 0
    price = None
    if group and mode != 'none':
        order = tuple(sorted(group, key=lambda i: (initial[i], i)))
        for assignment in sorted(set(permutations(tuple(initial[i] for i in order)))):
            proposed = initial[:]
            for i, capacity in zip(order, assignment):
                proposed[i] = capacity
            if proposed == initial:
                continue
            delta = [effective[i] * (int(proposed[i] >= demands[i]) -
                     int(initial[i] >= demands[i])) for i in range(3)]
            total = sum(delta[i] for i in group)
            if mode == 'transferable':
                if total > gain:
                    transfers = [total / len(group) - delta[i] if i in group else 0.
                                 for i in range(3)]
                    final, gain = proposed[:], total
            elif mode == 'bounded':
                for q in REF_Q:
                    side = [(initial[i] - proposed[i]) * q for i in range(3)]
                    gains = [delta[i] + side[i] for i in group]
                    if min(gains) >= 0 and max(gains) > 0 and total > gain:
                        final, transfers, gain, price = proposed[:], side[:], total, q / 20
            else:
                raise ValueError(mode)
    completed = [final[i] >= demands[i] for i in range(3)]
    bonuses = [REF_BONUSES[bonus][i] * completed[i] / 20 for i in range(3)]
    utilities = [(effective[i] * completed[i] - bills[i] + transfers[i]) / 20
                 for i in range(3)]
    return {'initial_capacities': initial, 'final_capacities': final,
            'payments': [v / 20 for v in bills], 'completed': completed,
            'bonuses': bonuses, 'transfers': [v / 20 for v in transfers],
            'utilities': utilities, 'utilities_before_trade': [v / 20 for v in before],
            'trade_gain': gain / 20, 'trade_success': bool(gain > 0),
            'trade_price': price, 'group_members': group}


@lru_cache(maxsize=576)
def stage_table(physical_id, bonus_action_id, trade_mode='bounded'):
    """Cached private exact expectation over original allocation ties."""
    result = market.expected_stage(int(physical_id), int(bonus_action_id), trade_mode)
    for value in result.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
        elif isinstance(value, dict):
            for array in value.values():
                array.flags.writeable = False
    return result


def best_profile(table):
    utility = table['utilities'].sum(axis=1)
    candidates = np.flatnonzero(utility >= utility.max() - TOL)
    # Adverse tie-breaking is applied only to utility-maximizing reports.
    net = table['metrics']['net_revenue']
    candidates = candidates[net[candidates] <= net[candidates].min() + TOL]
    gross = table['metrics']['gross_revenue']
    candidates = candidates[gross[candidates] <= gross[candidates].min() + TOL]
    collusion = table['metrics']['successful_collusion_members']
    candidates = candidates[collusion[candidates] >= collusion[candidates].max() - TOL]
    underreport = table['metrics']['underreport_count']
    candidates = candidates[underreport[candidates] >= underreport[candidates].max() - TOL]
    return int(candidates[0])


@lru_cache(maxsize=3)
def exact_action_cache(trade_mode='bounded'):
    """Shape (12 physical types, 16 bonus choices, 3 buyer actions)."""
    actions = np.empty((12, 16, 3), dtype=np.int64)
    for physical in range(12):
        for bonus in range(16):
            table = stage_table(physical, bonus, trade_mode)
            actions[physical, bonus] = table['action_profiles'][best_profile(table)]
    actions.flags.writeable = False
    return actions


def exact_best_actions(physical_ids, bonus_action_ids, trade_mode='bounded'):
    """Offline oracle: maximize all buyers' joint utility, including paid bonuses.

    Reports/membership are jointly selected, but trading still occurs only within
    same-report joined groups. This is not an equilibrium certificate or proof
    against arbitrary cross-group transfers, binding contracts or future attacks.
    """
    physical = np.asarray(physical_ids)
    bonus = np.asarray(bonus_action_ids)
    if physical.ndim != 1 or not np.issubdtype(physical.dtype, np.integer):
        raise ValueError('physical_ids must be a one-dimensional integer array')
    if bonus.ndim == 0:
        bonus = np.full(physical.shape, bonus.item())
    if (bonus.shape != physical.shape or not np.issubdtype(bonus.dtype, np.integer)
            or np.any((physical < 0) | (physical >= 12))
            or np.any((bonus < 0) | (bonus >= 16))):
        raise ValueError('Invalid physical or bonus action IDs')
    return exact_action_cache(trade_mode)[physical, bonus].copy()


def stage_regret(physical_ids, bonus_action_ids, probabilities, trade_mode='transferable'):
    """Offline joint-deviation gap for learned INDEPENDENT action probabilities.

    The learned joint distribution is the product of the three seven-action
    probabilities. No hard argmax conversion is made. This one-stage metric
    differs from a finite-horizon strategic attack on the platform's history.
    """
    physical = np.asarray(physical_ids, dtype=int)
    bonus = np.broadcast_to(np.asarray(bonus_action_ids, dtype=int), physical.shape)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(physical), 3, 7):
        raise ValueError('probabilities must have shape (B,3,7)')
    if np.any(probabilities < 0) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid probabilities')
    np.testing.assert_allclose(probabilities.sum(-1), 1., atol=1e-10)
    joint = np.einsum('bi,bj,bk->bijk', probabilities[:, 0], probabilities[:, 1],
                      probabilities[:, 2]).reshape(-1, 343)
    result = {name: np.empty(len(physical)) for name in (
        'learned_joint_utility', 'best_joint_utility', 'profitable_joint_deviation',
        'learned_net_revenue', 'best_response_net_revenue', 'learned_jobs', 'best_response_jobs')}
    for p, b in sorted(set(zip(physical.tolist(), bonus.tolist()))):
        rows = np.flatnonzero((physical == p) & (bonus == b))
        table = stage_table(p, b, trade_mode)
        best = best_profile(table)
        utilities = table['utilities'].sum(-1)
        result['learned_joint_utility'][rows] = joint[rows] @ utilities
        result['best_joint_utility'][rows] = utilities[best]
        for name, key in (('net_revenue', 'net_revenue'), ('jobs', 'jobs')):
            result[f'learned_{name}'][rows] = joint[rows] @ table['metrics'][key]
            result[f'best_response_{name}'][rows] = table['metrics'][key][best]
    result['profitable_joint_deviation'] = np.maximum(0., result['best_joint_utility'] -
                                                    result['learned_joint_utility'])
    return result


def verify_platform_freeze(path):
    """Parent runner may call this before a trained-platform stress evaluation."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError('Formal platform stress requires a pretest freeze')
    document = json.loads(path.read_text(encoding='utf-8'))
    dictionaries = [document.get(key, {}) for key in ('hashes', 'source_hashes', 'input_hashes')]
    recorded = {key: value for mapping in dictionaries for key, value in mapping.items()}
    for name in ('incentive_stress.py', 'incentive_market.py'):
        relative = f'code/{name}'
        if recorded.get(relative) != digest(ROOT / relative):
            raise RuntimeError(f'Pretest freeze missing or mismatched: {relative}')
    return document


def profile_id(actions):
    return int(actions[0] * 49 + actions[1] * 7 + actions[2])


def row(table, profile):
    return {'profile_id': int(profile),
        'actions': table['action_profiles'][profile].tolist(),
        'reports': table['reports'][profile].tolist(), 'joins': table['joins'][profile].tolist(),
        'utilities': table['utilities'][profile].tolist(),
        **{k: float(v[profile]) for k, v in table['metrics'].items()}}


def trade_decomposition(outcome, physical, bonus):
    before = np.asarray(outcome['initial_capacities']) >= market.PHYSICAL_DEMANDS[physical]
    after = np.asarray(outcome['completed'])
    delta = after.astype(int) - before.astype(int)
    original_value_gain = float(np.dot(market.PHYSICAL_VALUES[physical], delta))
    bonus_contribution = float(np.dot(market.BONUS_ACTIONS[bonus], delta))
    transfers = float(np.sum(outcome['transfers']))
    np.testing.assert_allclose(original_value_gain + bonus_contribution,
                               outcome['trade_gain'], atol=1e-12)
    return {'original_task_value_gain': original_value_gain,
            'bonus_contribution_to_trade_gain': bonus_contribution,
            'net_internal_transfers': transfers,
            'total_private_trade_gain': float(outcome['trade_gain'])}


def pure_nash(table):
    utilities = table['utilities'].reshape(7, 7, 7, 3)
    gains = np.zeros((343, 3))
    for player in range(3):
        payoff = utilities[..., player]
        gains[:, player] = (payoff.max(axis=player, keepdims=True) - payoff).reshape(-1)
    return np.flatnonzero(gains.max(-1) <= TOL), gains


def self_test():
    np.testing.assert_array_equal(market.BUYER_ACTIONS, REF_ACTIONS)
    np.testing.assert_array_equal(np.rint(market.BONUS_ACTIONS * 20), REF_BONUSES)
    for physical in range(12):
        m, v = ref_type(physical)
        np.testing.assert_array_equal(market.PHYSICAL_DEMANDS[physical], m)
        np.testing.assert_array_equal(market.PHYSICAL_VALUES[physical] * 20, v)
    for reports in product(range(4), repeat=3):
        assert ref_matchings(reports) == market.matchings((3, 2), reports, 0)
    rng = np.random.default_rng(AUDIT_SEED)
    checked = 0
    for mode in MODES:
        tables = market.get_tables(mode)
        for _ in range(512):
            physical, bonus, profile = int(rng.integers(12)), int(rng.integers(16)), int(rng.integers(343))
            actions = tuple(int(x) for x in market.ACTION_PROFILES[profile])
            reports = tuple(REF_ACTIONS[a][0] for a in actions)
            owners_list = ref_matchings(reports)
            matching = int(rng.integers(len(owners_list)))
            owners = owners_list[matching]
            reference = ref_resolve(physical, actions, bonus, owners, mode)
            actual = market.resolve_outcome(physical, actions, bonus, owners, mode)
            for key in reference:
                if key in ('trade_price', 'group_members'):
                    assert reference[key] == actual[key], (mode, physical, bonus, actions, owners, key)
                else:
                    np.testing.assert_allclose(actual[key], reference[key], atol=1e-12,
                        err_msg=str((mode, physical, bonus, actions, owners, key)))
            public, metrics, utilities = market._lookup(tables, np.array([physical]),
                np.array([bonus]), np.array([profile]), np.array([matching]))
            np.testing.assert_allclose(utilities[0], reference['utilities'], atol=1e-12)
            np.testing.assert_allclose(metrics['gross_revenue'][0], sum(reference['payments']), atol=1e-12)
            np.testing.assert_allclose(metrics['cost'][0], sum(reference['bonuses']), atol=1e-12)
            checked += 1
    # Verify averaging across every profile for two prespecified games, rather
    # than trusting the simulator's expected_stage aggregation alone.
    for physical, bonus, mode in ((0, 0, 'bounded'), (9, 15, 'transferable')):
        actual = stage_table(physical, bonus, mode)
        for profile, actions in enumerate(market.ACTION_PROFILES):
            reports = tuple(REF_ACTIONS[a][0] for a in actions)
            expected = [ref_resolve(physical, actions, bonus, owner, mode)
                        for owner in ref_matchings(reports)]
            np.testing.assert_allclose(actual['utilities'][profile],
                np.mean([e['utilities'] for e in expected], axis=0), atol=1e-12)
    forced = exact_best_actions(np.arange(12), np.zeros(12, int), 'transferable')
    env = market.Market(np.arange(12) * 2, trade_mode='transferable')
    obs, metrics = env.step(0, rng.random((12, 4)), buyer_actions=forced)
    np.testing.assert_allclose(metrics['gross_revenue'], 2.6, atol=1e-12)
    np.testing.assert_array_equal(metrics['jobs'], 2)
    # A point mass on the oracle must have exactly zero one-step regret.
    probability = np.zeros((12, 3, 7))
    probability[np.arange(12)[:, None], np.arange(3)[None, :], forced] = 1
    np.testing.assert_allclose(stage_regret(np.arange(12), 0, probability)[
                              'profitable_joint_deviation'], 0, atol=1e-12)
    print(f'PASS: independent allocation enumeration; {checked} independent integer '
          'scalar/vector checks; 686 full profile averages; forced oracle replay; regret.', flush=True)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        raise RuntimeError('Exact audit already frozen; do not overwrite')
    write_json(path, {'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'finite_stage_audit_frozen_before_enumeration', 'source_hashes': source_hashes(),
        'physical_types': list(range(12)), 'bonus_actions': list(range(16)),
        'buyer_joint_actions': 343, 'modes': MODES, 'supply': [3, 2],
        'independent_checks': '64 report allocation enumerations; 512 seeded scalar outcomes per mode with independent integer-twentieth resolver; two complete 343-profile averaging checks.',
        'independent_check_seed': AUDIT_SEED,
        'oracle': 'Maximum total user utility including collected completion bonus; utility ties minimize net receipts, then gross receipts, then maximize successful-coordinated-underreport members, then underreport count, then joint action index. Separate extrema across all utility maximizers are also reported.',
        'trading_scope': 'Only same-positive-report joined groups; no arbitrary cross-group resource transfers. Exact joint report choice itself does not enforce ex-ante coalition participation or a Nash equilibrium.',
        'trade_tie_order': 'Sort group positions by (initial capacity, buyer ID); lexicographic assignment in that order, then ascending q. Duplicate initial capacities still use the declared account-label final tie.',
        'nash': 'All pure unilateral equilibria in the fixed 7x7x7 stage game; mixed equilibria are not enumerated.',
        'bonus_effect': 'Exact trade changes holding reports/membership fixed; catalogue counts are not an empirical policy distribution.',
        'witnesses': ['A completion award can prevent a resource surrender with limited q compensation.',
                      'Lossless VM swaps leave the small task completed and help the large task; completion-only bonuses cannot prevent this feasible voluntary exchange.',
                      'All three buyers join at report 1 and can collect rewards while coordinating underreporting.'],
        'withdrawal': 'Zero-bonus exact-stage oracle is independent of steering history; learned-policy one-step joint regret is a different, possibly positive quantity.',
        'platform_policy_evaluated': False,
        'limits': ['No arbitrary-horizon adversary or universal collusion-proofness guarantee.',
                   'A numerical gap versus truthful reports is not a voluntarily stable coalition certificate.',
                   'An endogenous trade refusal is determined by participation inequalities, never a random externally imposed failure rate.',
                   '3 buyers and 2 indivisible VMs retain scarcity; unlike the earlier 2-buyer fully pooled experiments, all 3 jobs cannot simultaneously finish.']})
    print('Exact audit protocol frozen', flush=True)


def audit():
    frozen = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    if frozen['source_hashes'] != source_hashes():
        raise RuntimeError('Exact-audit source changed after freeze')
    if (OUT / 'summary.json').exists():
        raise RuntimeError('Exact audit results already exist')
    start = time.perf_counter()
    records = []; equilibrium_rows = []; effects = []; withdrawal = []; full_detail = []
    for mode in MODES:
        for physical in range(12):
            demands, _ = ref_type(physical)
            truthful = profile_id([5 if m == 3 else 3 for m in demands])
            zero = stage_table(physical, 0, mode)
            for bonus in range(16):
                table = stage_table(physical, bonus, mode)
                best = best_profile(table)
                equilibria, regret = pure_nash(table)
                truth = row(table, truthful)
                optimum = row(table, best)
                all_join = row(table, profile_id((2, 2, 2)))
                maximizers = np.flatnonzero(table['utilities'].sum(1) >= optimum['utility'] - TOL)
                item = {'trade_mode': mode, 'physical_id': physical, 'bonus_action': bonus,
                    'best_profile_id': best, 'best_joint_utility': optimum['utility'],
                    'best_gross_revenue': optimum['gross_revenue'], 'best_cost': optimum['cost'],
                    'best_net_revenue': optimum['net_revenue'], 'best_jobs': optimum['jobs'],
                    'best_value': optimum['value'], 'best_join_count': optimum['join_count'],
                    'best_underreport_count': optimum['underreport_count'],
                    'truth_joint_utility': truth['utility'], 'truth_gross_revenue': truth['gross_revenue'],
                    'truth_net_revenue': truth['net_revenue'], 'truth_jobs': truth['jobs'],
                    'truth_best_unilateral_gain': float(regret[truthful].max()),
                    'joint_gain_over_truthful': optimum['utility'] - truth['utility'],
                    'pure_nash_count': len(equilibria),
                    'joint_utility_maximizer_count': len(maximizers),
                    'maximizer_min_net_revenue': float(table['metrics']['net_revenue'][maximizers].min()),
                    'maximizer_max_successful_collusion_members': float(table['metrics']['successful_collusion_members'][maximizers].max()),
                    'maximizer_max_underreport_count': float(table['metrics']['underreport_count'][maximizers].max()),
                    'all_join_report_one_is_joint_optimal': all_join['utility'] >= optimum['utility'] - TOL}
                records.append(item)
                full_detail.append({'trade_mode': mode, 'physical_id': physical, 'bonus_action': bonus,
                    'best': optimum, 'truthful': truth, 'all_join_report_one': all_join,
                    'pure_nash_profile_ids': equilibria.tolist()})
                for profile in equilibria:
                    eq = row(table, profile)
                    equilibrium_rows.append({'trade_mode': mode, 'physical_id': physical,
                        'bonus_action': bonus, 'profile_id': int(profile),
                        'reports': str(eq['reports']), 'joins': str(eq['joins']),
                        **{key: eq[key] for key in market.METRIC_NAMES}})
                if mode == 'bounded' and bonus:
                    difference = table['metrics']['trade_success'] - zero['metrics']['trade_success']
                    reduced = np.flatnonzero(difference < -TOL)
                    increased = np.flatnonzero(difference > TOL)
                    effects.append({'physical_id': physical, 'bonus_action': bonus,
                        'profiles_with_less_trade': len(reduced), 'profiles_with_more_trade': len(increased),
                        'profiles_unchanged': int(np.sum(np.abs(difference) <= TOL)),
                        'most_reduced_profile': int(np.argmin(difference)),
                        'max_trade_probability_reduction': float(-difference.min()),
                        'most_increased_profile': int(np.argmax(difference)),
                        'max_trade_probability_increase': float(difference.max())})
                if bonus == 0:
                    withdrawal.append({'trade_mode': mode, 'physical_id': physical,
                        'stationary_stage_oracle': optimum,
                        'history_independence_condition': 'Fixed A0/invoice, reset physical supply each round, no binding cross-period contract; current exact response with bonus zero.',
                        'learned_policy_regret_not_assumed_zero': True})
        print(f'Enumerated {mode}: 12 types x 16 rewards x 343 profiles', flush=True)
    # A resource-surrender trade can be deterred by a completion reward.
    blocked = {'configuration': {'physical_id': 0, 'action_ids': [2, 2, 3], 'owners': [1, 2]},
        'without_reward': market.resolve_outcome(0, (2, 2, 3), 0, (1, 2)),
        'completion_reward_buyer1_1': market.resolve_outcome(0, (2, 2, 3), 7, (1, 2)),
        'transferable_with_same_reward': market.resolve_outcome(0, (2, 2, 3), 7, (1, 2), 'transferable')}
    assert blocked['without_reward']['trade_success']
    assert not blocked['completion_reward_buyer1_1']['trade_success']
    assert blocked['transferable_with_same_reward']['trade_success']
    blocked['value_and_bonus_decomposition'] = {
        name: trade_decomposition(blocked[name], 0, bonus) for name, bonus in (
            ('without_reward', 0), ('completion_reward_buyer1_1', 7),
            ('transferable_with_same_reward', 7))}
    harmless = []
    for physical in range(12):
        big = physical % 3
        small = (big + 1) % 3
        other = (big + 2) % 3
        actions = [1, 1, 1]
        actions[big] = actions[small] = 2
        for bonus in range(16):
            outcome = market.resolve_outcome(physical, actions, bonus, (small, big))
            demands, values = ref_type(physical)
            q = .2
            # Explicit feasible swap: small keeps completion and receives q;
            # big newly finishes and pays q. Other's state remains unchanged.
            gain = np.zeros(3)
            gain[small] = q
            gain[big] = values[big] / 20 + REF_BONUSES[bonus][big] / 20 - q
            assert np.all(gain >= 0) and np.max(gain) > 0
            assert outcome['trade_success'] and outcome['metrics']['jobs'] == 2
            harmless.append({'physical_id': physical, 'bonus_action': bonus,
                'action_ids': actions, 'initial_owners': [small, big],
                'feasible_swap_price': q, 'voluntary_swap_utility_gains': gain.tolist(),
                'original_gross_revenue': outcome['metrics']['gross_revenue'],
                'post_trade_cost': outcome['metrics']['cost'], 'post_trade_jobs': 2,
                'chosen_trade_decomposition': trade_decomposition(outcome, physical, bonus)})
    summary = {'status': 'complete', 'source_hashes': source_hashes(),
        'stage_configurations': len(records), 'joint_profiles_per_configuration': 343,
        'elapsed_seconds': time.perf_counter() - start, 'records': records,
        'bounded_bonus_effects': effects, 'withdrawal_oracles': withdrawal,
        'inference': 'Completion rewards sometimes deter sacrifice-type bounded-price transfers, but cannot block lossless VM swaps. Strong joint deviation remains possible; exact postreward endpoints do not measure coalition dissolution.',
        'platform_policies_evaluated': False}
    write_json(OUT / 'summary.json', summary)
    write_json(OUT / 'stage_details.json', full_detail)
    write_json(OUT / 'witnesses.json', {'surrender_trade': blocked,
                                     'lossless_swaps_all_rewards': harmless})
    write_json(OUT / 'withdrawal_oracles.json', withdrawal)
    write_csv(OUT / 'stage_summary.csv', records)
    write_csv(OUT / 'pure_nash.csv', equilibrium_rows)
    write_csv(OUT / 'bonus_trade_effects.csv', effects)
    write_json(OUT / 'completion_audit.json', {'status': 'complete',
        'sources_match_freeze': source_hashes() == frozen['source_hashes'],
        'platform_policies_evaluated': False, 'all_576_stage_configurations_completed': len(records) == 576,
        'all_192_lossless_swap_witnesses_passed': len(harmless) == 192,
        'no_exogenous_trade_failure_probability': True,
        'results': {p.name: digest(p) for p in OUT.glob('*') if p.is_file()}})
    print(json.dumps({'stage_configurations': len(records), 'pure_equilibria_rows': len(equilibrium_rows),
                      'elapsed_seconds': summary['elapsed_seconds']}), flush=True)


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
