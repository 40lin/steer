"""Fixed-allocation market with completion-only incentives and voluntary trade.

This is a new, explicit economic model, not a modification of the frozen V1/V2
environments or a claim to reproduce a paper's complete coalition algorithm.
The platform observes reports, its ORIGINAL allocations/bills, and verifiable
completion/bonus receipts. Join choices, workloads, values, learning state and
private transfers are never included in ``Market.step``'s public dictionary.

Money is evaluated in integer twentieths before voluntary-trade comparisons.
Only unrestricted, equal-gain side payments may require fractional twentieths.
Cached tables depend on the complete configuration and both source hashes.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from itertools import permutations, product
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from validate_micro_model import matchings


N_BUYERS = 3
TEMPERATURE = 0.2
MONEY_SCALE = 20
ETA_VALUES = np.array([0.05, 0.2], dtype=np.float64)
VALUE_FAMILIES = np.array([[6, 2, 2], [4, 3, 3], [9, 5, 5],
                           [2.4, 1.6, 1.6]], dtype=np.float64)
BUYER_ACTIONS = np.array([(0, 0)] + [(b, j) for b in (1, 2, 3)
                                   for j in (0, 1)], dtype=np.int8)
ACTION_PROFILES = np.array(list(product(range(7), repeat=3)), dtype=np.int16)
BONUS_ACTIONS = np.array([[0., 0., 0.]] +
                         [[amount if i == buyer else 0. for i in range(3)]
                          for buyer in range(3) for amount in (.25, .5, 1., 2.)] +
                         [[amount] * 3 for amount in (.25, .5, 1.)], dtype=np.float64)
BONUS_MAX_COST = BONUS_ACTIONS.sum(axis=1)
BONUS_UNITS = np.rint(BONUS_ACTIONS * MONEY_SCALE).astype(np.int16)
TRADE_PRICES = np.array([.2, .4, .6, .8, .95], dtype=np.float64)
TRADE_PRICE_UNITS = np.array([4, 8, 12, 16, 19], dtype=np.int16)
PHYSICAL_DEMANDS = np.array([np.roll([3, 2, 2], position)
                            for family in range(4) for position in range(3)], dtype=np.int8)
PHYSICAL_VALUES = np.array([np.roll(VALUE_FAMILIES[family], position)
                           for family in range(4) for position in range(3)])
VALUE_UNITS = np.rint(PHYSICAL_VALUES * MONEY_SCALE).astype(np.int16)
MODEL_PHYSICAL_IDS = np.repeat(np.arange(12), 2)
MODEL_ETAS = np.tile(ETA_VALUES, 12)
DEMANDS = PHYSICAL_DEMANDS[MODEL_PHYSICAL_IDS]
VALUES = PHYSICAL_VALUES[MODEL_PHYSICAL_IDS]
MODELS = tuple({'model_id': i, 'physical_id': int(MODEL_PHYSICAL_IDS[i]),
                'family': int(MODEL_PHYSICAL_IDS[i] // 3),
                'big_buyer': int(MODEL_PHYSICAL_IDS[i] % 3),
                'demands': DEMANDS[i].tolist(), 'values': VALUES[i].tolist(),
                'eta': float(MODEL_ETAS[i])} for i in range(24))
METRIC_NAMES = ('gross_revenue', 'cost', 'net_revenue', 'jobs', 'value', 'utility',
                'underreport_count', 'severity', 'overreport_count', 'active',
                'join_count', 'formed_members', 'harmful_members',
                'successful_collusion_members', 'trade_attempt', 'trade_success',
                'trade_gains', 'failed_jobs')
PUBLIC_KEYS = frozenset(('reports', 'allocated_capacities', 'payments',
                         'completed', 'bonuses'))
CACHE_DIR = Path(__file__).resolve().parents[1] / 'experiments/v4_incentive/cache'
TRADE_MODES = ('bounded', 'transferable', 'none')


def _validate_supply(supply):
    supply = tuple(supply)
    if not supply or any(type(c) not in (int, np.int32, np.int64) or c <= 0
                         for c in supply):
        raise ValueError('supply must contain positive integer whole-VM capacities')
    if len(supply) > N_BUYERS:
        raise ValueError('at most three whole VMs are supported')
    return tuple(int(c) for c in supply)


def _validate_mode(trade_mode):
    if trade_mode not in TRADE_MODES:
        raise ValueError(f'trade_mode must be one of {TRADE_MODES}')


def _group(reports, joins):
    """With three buyers, at most one same-positive-report group can form."""
    for b in (1, 2, 3):
        members = tuple(i for i in range(3) if reports[i] == b and joins[i])
        if len(members) >= 2:
            return members
    return ()


def _initial(owners, supply, reports):
    capacities = np.zeros(3, dtype=np.int8)
    for capacity, owner in zip(supply, owners):
        if owner >= 0:
            capacities[owner] = capacity
    payments = np.where(capacities > 0, 4 * capacities + 16 * reports, 0).astype(np.int16)
    return capacities, payments


def _candidate_capacities(initial, group):
    """Initial-capacity order, then candidate tuple and price, break gain ties.

    Members are ordered by (initial capacity, account ID). With leased capacity
    (3,2,0), this is equivariant to relabelling. IDs only break indistinguishable
    initial-capacity ties, e.g. multiple unallocated members. Feasibility and the
    maximum total gain do not depend on this selection among equal-gain trades.
    """
    ordered = sorted(group, key=lambda i: (int(initial[i]), i))
    for assigned in sorted(set(permutations(tuple(int(initial[i]) for i in ordered)))):
        final = initial.copy()
        final[ordered] = assigned
        if not np.array_equal(final, initial):
            yield final


def _resolve_grid(reports, joins, initial, payments, trade_mode):
    """Resolve all 12 physical types x 16 incentive actions at once."""
    shape = (12, 16)
    final = np.broadcast_to(initial, shape + (3,)).copy()
    transfers = np.zeros(shape + (3,), dtype=np.float64)
    gains = np.zeros(shape, dtype=np.int16)
    q_index = np.full(shape, -1, dtype=np.int8)
    group = _group(reports, joins)
    if trade_mode == 'none' or not group:
        return final, transfers, gains, q_index
    g = list(group)
    completion0 = initial >= PHYSICAL_DEMANDS
    effective_values = VALUE_UNITS[:, None, :] + BONUS_UNITS[None, :, :]
    for proposed in _candidate_capacities(initial, group):
        completion1 = proposed >= PHYSICAL_DEMANDS
        delta = effective_values * (completion1.astype(np.int16) -
                                    completion0.astype(np.int16))[:, None, :]
        total = delta[:, :, g].sum(axis=2)
        if trade_mode == 'transferable':
            improve = total > gains  # Strictly positive, and fixed first tie.
            split = total.astype(np.float64) / len(group)
            side = np.zeros(shape + (3,), dtype=np.float64)
            side[:, :, g] = split[:, :, None] - delta[:, :, g]
            final[improve] = proposed
            transfers[improve] = side[improve]
            gains[improve] = total[improve]
        else:
            for qi, q in enumerate(TRADE_PRICE_UNITS):
                side = (initial.astype(np.int16) - proposed) * int(q)
                private_delta = delta[:, :, g] + side[g]
                acceptable = np.all(private_delta >= 0, axis=2) & np.any(private_delta > 0, axis=2)
                improve = acceptable & (total > gains)
                final[improve] = proposed
                transfers[improve] = side
                gains[improve] = total[improve]
                q_index[improve] = qi
    return final, transfers, gains, q_index


def _configuration(trade_mode, supply):
    source = Path(__file__).resolve()
    dependency = source.with_name('validate_micro_model.py')
    return {'schema': 1, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'allocation_source_sha256': hashlib.sha256(dependency.read_bytes()).hexdigest(),
            'trade_mode': trade_mode, 'supply': list(supply),
            'physical_demands': PHYSICAL_DEMANDS.tolist(), 'physical_values': PHYSICAL_VALUES.tolist(),
            'buyer_actions': BUYER_ACTIONS.tolist(), 'bonus_actions': BONUS_ACTIONS.tolist(),
            'trade_prices': TRADE_PRICES.tolist(), 'base_capacity_coefficient': .2,
            'base_report_coefficient': .8, 'rho': 0., 'join_cost': 0.,
            'matching': 'validate_micro_model.matchings; uniform optimal matching',
            'trade_tie_order': 'members by (initial capacity, ID); lexicographic final capacity tuple in that order; ascending q',
            'completion': 'deterministic final capacity >= indivisible task demand'}


class OutcomeTables:
    """Read-only finite outcome tables. Axes: physical, bonus, profile, match."""
    def __init__(self, arrays, configuration, cache_path=None):
        self.configuration = configuration
        self.cache_path = cache_path
        self.arrays = arrays
        for key, value in arrays.items():
            value.flags.writeable = False
            setattr(self, key, value)
        self.nbytes = sum(x.nbytes for x in arrays.values())


def _build_tables(trade_mode, supply):
    reports = BUYER_ACTIONS[ACTION_PROFILES, 0]
    joins = BUYER_ACTIONS[ACTION_PROFILES, 1]
    owner_lists = [matchings(supply, tuple(int(x) for x in b), 0) for b in reports]
    counts = np.array([len(ms) for ms in owner_lists], dtype=np.int16)
    kmax = int(counts.max())
    initial = np.zeros((343, kmax, 3), dtype=np.int8)
    payments = np.zeros((343, kmax, 3), dtype=np.int16)
    owners = np.full((343, kmax, len(supply)), -1, dtype=np.int8)
    formed = np.zeros((343, 3), dtype=bool)
    final = np.zeros((12, 16, 343, kmax, 3), dtype=np.int8)
    transfers = np.zeros((12, 16, 343, kmax, 3), dtype=np.float64)
    gains = np.zeros((12, 16, 343, kmax), dtype=np.int16)
    q_index = np.full((12, 16, 343, kmax), -1, dtype=np.int8)
    for ai, ms in enumerate(owner_lists):
        formed[ai, list(_group(reports[ai], joins[ai]))] = True
        for mi, owner in enumerate(ms):
            owners[ai, mi] = owner
            c, p = _initial(owner, supply, reports[ai])
            initial[ai, mi], payments[ai, mi] = c, p
            outcome = _resolve_grid(reports[ai], joins[ai], c, p, trade_mode)
            final[:, :, ai, mi], transfers[:, :, ai, mi], gains[:, :, ai, mi], q_index[:, :, ai, mi] = outcome
    return {'reports': reports, 'joins': joins, 'matching_counts': counts,
            'owners': owners, 'initial_capacities': initial, 'payment_units': payments,
            'formed': formed, 'final_capacities': final, 'transfer_units': transfers,
            'trade_gain_units': gains, 'trade_q_index': q_index}


@lru_cache(maxsize=6)
def get_tables(trade_mode='bounded', supply=(3, 2), use_disk_cache=True):
    _validate_mode(trade_mode)
    supply = _validate_supply(supply)
    config = _configuration(trade_mode, supply)
    config_json = json.dumps(config, sort_keys=True, separators=(',', ':'))
    key = hashlib.sha256(config_json.encode()).hexdigest()
    cache_path = CACHE_DIR / f'{trade_mode}_{key}.npz'
    if use_disk_cache and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as loaded:
            if str(loaded['configuration_json'].item()) != config_json:
                raise ValueError('cached economic configuration mismatch')
            arrays = {k: loaded[k] for k in loaded.files if k != 'configuration_json'}
        return OutcomeTables(arrays, config, cache_path)
    arrays = _build_tables(trade_mode, supply)
    if use_disk_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=trade_mode + '_', suffix='.npz', dir=CACHE_DIR)
        try:
            with os.fdopen(fd, 'wb') as handle:
                np.savez_compressed(handle, configuration_json=np.array(config_json), **arrays)
            os.replace(temporary, cache_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return OutcomeTables(arrays, config, cache_path if use_disk_cache else None)


def _metrics(reports, joins, formed, initial, payments, final, completed,
             bonuses, utilities, demands, values, trade_gains, trade_mode):
    active = reports > 0
    under = active & (reports < demands)
    member_count = formed.sum(axis=1)
    coordinated_under = member_count * np.any(formed & under, axis=1)
    success = trade_gains > 0
    gross = payments.sum(axis=1)
    cost = bonuses.sum(axis=1)
    jobs = completed.sum(axis=1)
    return dict(zip(METRIC_NAMES, (
        gross, cost, gross - cost, jobs, (values * completed).sum(axis=1),
        utilities.sum(axis=1), under.sum(axis=1),
        (np.maximum(demands - reports, 0) * active).sum(axis=1),
        (active & (reports > demands)).sum(axis=1), active.sum(axis=1),
        joins.sum(axis=1), member_count, coordinated_under,
        coordinated_under * success,
        ((member_count >= 2) & (trade_mode != 'none')).astype(np.int8),
        success.astype(np.int8), trade_gains, 3 - jobs)))


def _lookup(tables, physical_ids, bonus_ids, profile_ids, matching_ids):
    b, a, j, k = physical_ids, bonus_ids, profile_ids, matching_ids
    reports = tables.reports[j]
    joins = tables.joins[j]
    initial = tables.initial_capacities[j, k]
    payments = tables.payment_units[j, k].astype(np.float64) / MONEY_SCALE
    final = tables.final_capacities[b, a, j, k]
    transfers = tables.transfer_units[b, a, j, k] / MONEY_SCALE
    gains = tables.trade_gain_units[b, a, j, k].astype(np.float64) / MONEY_SCALE
    completed = final >= PHYSICAL_DEMANDS[b]
    bonuses = BONUS_ACTIONS[a] * completed
    utilities = PHYSICAL_VALUES[b] * completed - payments + transfers + bonuses
    metrics = _metrics(reports, joins, tables.formed[j], initial, payments, final,
                       completed, bonuses, utilities, PHYSICAL_DEMANDS[b],
                       PHYSICAL_VALUES[b], gains, tables.configuration['trade_mode'])
    public = {'reports': reports.copy(), 'allocated_capacities': initial.copy(),
              'payments': payments, 'completed': completed.copy(), 'bonuses': bonuses}
    return public, metrics, utilities


class Market:
    """Vectorized independent learners; the outer controller enforces budget.

    model_id = (family * 3 + position_of_large_task) * 2 + eta_index.
    Underreport count/severity exclude opt-out. ``harmful_members`` is a legacy
    metric name for coordinated-underreport group membership, NOT causal harm.
    There is no membership cost and labels may drift when joining has no effect.
    """
    def __init__(self, model_ids, trade_mode='bounded', supply=(3, 2), use_disk_cache=True):
        ids = np.asarray(model_ids)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= 24)):
            raise ValueError('model_ids must be a one-dimensional integer array in 0..23')
        self.model_ids = ids.astype(np.int64, copy=True)
        self.physical_ids = MODEL_PHYSICAL_IDS[self.model_ids]
        self.eta = MODEL_ETAS[self.model_ids]
        self.trade_mode = trade_mode
        self.supply = _validate_supply(supply)
        self.tables = get_tables(trade_mode, self.supply, use_disk_cache)
        self.q = np.zeros((len(ids), 3, 7), dtype=np.float64)
        self.step_count = 0

    @property
    def batch_size(self):
        return len(self.model_ids)

    def probabilities(self):
        z = self.q / TEMPERATURE
        z -= z.max(axis=2, keepdims=True)
        exp = np.exp(z)
        return exp / exp.sum(axis=2, keepdims=True)

    def step(self, bonus_action_ids, random_u, buyer_actions=None):
        """Advance one round; optional forced actions are an OFFLINE stress API.

        Forced actions still receive the same economic outcome and Q update.
        They are never added to the public observation or a deployment policy.
        """
        ids = np.asarray(bonus_action_ids)
        if ids.ndim == 0:
            ids = np.full(self.batch_size, ids.item())
        if ids.shape != (self.batch_size,) or not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= 16)):
            raise ValueError('bonus_action_ids must be integer IDs in 0..15, one per world')
        u = np.asarray(random_u, dtype=np.float64)
        if u.shape != (self.batch_size, 4) or not np.all(np.isfinite(u)) or np.any((u < 0) | (u >= 1)):
            raise ValueError('random_u must have shape (batch,4), finite values in [0,1)')
        if buyer_actions is None:
            cdf = np.cumsum(self.probabilities(), axis=2)
            cdf[:, :, -1] = 1.
            actions = (u[:, :3, None] >= cdf).sum(axis=2)
        else:
            actions = np.asarray(buyer_actions)
            if (actions.shape != (self.batch_size, 3) or
                    not np.issubdtype(actions.dtype, np.integer) or
                    np.any((actions < 0) | (actions >= 7))):
                raise ValueError('forced buyer_actions must have shape (batch,3), integer IDs 0..6')
        profiles = actions[:, 0] * 49 + actions[:, 1] * 7 + actions[:, 2]
        matching_ids = (u[:, 3] * self.tables.matching_counts[profiles]).astype(np.int64)
        public, metrics, utilities = _lookup(self.tables, self.physical_ids,
                                            ids.astype(np.int64), profiles, matching_ids)
        rows, buyers = np.arange(self.batch_size)[:, None], np.arange(3)[None, :]
        old = self.q[rows, buyers, actions]
        self.q[rows, buyers, actions] = old + self.eta[:, None] * (utilities / 6. - old)
        self.step_count += 1
        return public, metrics

    def get_state(self):
        """Private replay state, never an actor observation."""
        return {'model_ids': self.model_ids.copy(), 'q': self.q.copy(),
                'step_count': self.step_count, 'trade_mode': self.trade_mode,
                'supply': self.supply}

    def set_state(self, state):
        if (state['trade_mode'] != self.trade_mode or tuple(state['supply']) != self.supply or
                not np.array_equal(state['model_ids'], self.model_ids)):
            raise ValueError('state configuration and model IDs must match this market')
        q = np.asarray(state['q'], dtype=np.float64)
        if q.shape != self.q.shape or not np.isfinite(q).all():
            raise ValueError('invalid Q state')
        self.q = q.copy()
        self.step_count = int(state['step_count'])

    def clone(self):
        clone = Market(self.model_ids, self.trade_mode, self.supply)
        clone.set_state(self.get_state())
        return clone


def resolve_outcome(physical_id, action_ids, bonus_action_id, owners,
                    trade_mode='bounded', supply=(3, 2)):
    """Scalar private audit API, independent of the vectorized table builder.

    ``owners`` is an original A0 matching tuple, one owner ID per VM. Fees stay
    with these original owners. Rewards go only to the final completed users.
    The returned hidden fields MUST NOT be used as controller observations.
    """
    _validate_mode(trade_mode)
    supply = _validate_supply(supply)
    if not 0 <= physical_id < 12 or not 0 <= bonus_action_id < 16:
        raise ValueError('physical or bonus ID outside catalogue')
    actions = np.asarray(action_ids)
    if actions.shape != (3,) or not np.issubdtype(actions.dtype, np.integer) or np.any((actions < 0) | (actions >= 7)):
        raise ValueError('action_ids must be three integers in 0..6')
    reports = BUYER_ACTIONS[actions, 0]
    joins = BUYER_ACTIONS[actions, 1]
    owners = tuple(int(i) for i in owners)
    if owners not in matchings(supply, tuple(int(b) for b in reports), 0):
        raise ValueError('owners must be a feasible maximum-capacity A0 matching')
    initial, payment_units = _initial(owners, supply, reports)
    m = PHYSICAL_DEMANDS[physical_id]
    effective = VALUE_UNITS[physical_id] + BONUS_UNITS[bonus_action_id]
    baseline = effective * (initial >= m) - payment_units
    group = _group(reports, joins)
    final, transfers, gain, price = initial.copy(), np.zeros(3), 0, None
    if group and trade_mode != 'none':
        for proposed in _candidate_capacities(initial, group):
            delta = effective * ((proposed >= m).astype(np.int16) - (initial >= m).astype(np.int16))
            total = int(sum(delta[i] for i in group))
            if trade_mode == 'transferable':
                if total > gain:
                    side = np.zeros(3)
                    side[list(group)] = total / len(group) - delta[list(group)]
                    final, transfers, gain = proposed.copy(), side, total
            else:
                for q in TRADE_PRICE_UNITS:
                    side = (initial.astype(np.int16) - proposed) * int(q)
                    changes = delta[list(group)] + side[list(group)]
                    if np.all(changes >= 0) and np.any(changes > 0) and total > gain:
                        final, transfers, gain = proposed.copy(), side.astype(np.float64), total
                        price = float(q) / MONEY_SCALE
    completed = final >= m
    bonuses = BONUS_ACTIONS[bonus_action_id] * completed
    payments = payment_units.astype(np.float64) / MONEY_SCALE
    utility = (effective * completed - payment_units + transfers) / MONEY_SCALE
    formed = np.zeros((1, 3), dtype=bool)
    formed[0, list(group)] = True
    metric_arrays = _metrics(reports[None], joins[None], formed, initial[None],
                             payments[None], final[None], completed[None], bonuses[None],
                             utility[None], m[None], PHYSICAL_VALUES[physical_id][None],
                             np.array([gain / MONEY_SCALE]), trade_mode)
    return {'action_ids': actions.copy(), 'reports': reports.copy(), 'joins': joins.copy(),
            'group_members': group, 'initial_capacities': initial, 'final_capacities': final,
            'payments': payments, 'completed': completed, 'bonuses': bonuses,
            'transfers': transfers / MONEY_SCALE, 'utilities': utility,
            'utilities_before_trade': baseline.astype(np.float64) / MONEY_SCALE,
            'trade_success': bool(gain > 0), 'trade_gain': gain / MONEY_SCALE,
            'trade_price': price, 'metrics': {k: v[0].item() for k, v in metric_arrays.items()}}


def expected_stage(physical_id, bonus_action_id, trade_mode='bounded', supply=(3, 2)):
    """Private one-stage oracle: all 343 profiles, averaged over A0 matchings."""
    if not 0 <= physical_id < 12 or not 0 <= bonus_action_id < 16:
        raise ValueError('physical or bonus ID outside catalogue')
    tables = get_tables(trade_mode, tuple(supply))
    profiles = np.repeat(np.arange(343), tables.matching_counts)
    matches = np.concatenate([np.arange(k) for k in tables.matching_counts])
    size = len(profiles)
    public, metrics, utilities = _lookup(tables, np.full(size, physical_id),
                                         np.full(size, bonus_action_id), profiles, matches)
    expected = {name: np.bincount(profiles, weights=v, minlength=343) / tables.matching_counts
                for name, v in metrics.items()}
    private = np.stack([np.bincount(profiles, weights=utilities[:, i], minlength=343) /
                        tables.matching_counts for i in range(3)], axis=1)
    return {'action_profiles': ACTION_PROFILES.copy(), 'reports': tables.reports.copy(),
            'joins': tables.joins.copy(), 'utilities': private, 'metrics': expected}


def self_test():
    """Design/invariant tests only; no policy training or outcome selection."""
    rng = np.random.default_rng(80731)
    checked = 0
    for mode in TRADE_MODES:
        tables = get_tables(mode)
        for _ in range(400):
            physical, bonus, profile = int(rng.integers(12)), int(rng.integers(16)), int(rng.integers(343))
            matching = int(rng.integers(tables.matching_counts[profile]))
            outcome = resolve_outcome(physical, ACTION_PROFILES[profile], bonus,
                                      tables.owners[profile, matching], mode)
            public, metrics, utility = _lookup(tables, np.array([physical]), np.array([bonus]),
                                               np.array([profile]), np.array([matching]))
            assert set(public) == PUBLIC_KEYS
            np.testing.assert_allclose(utility[0], outcome['utilities'], atol=1e-12)
            for name in METRIC_NAMES:
                np.testing.assert_allclose(metrics[name][0], outcome['metrics'][name], atol=1e-12)
            assert abs(outcome['transfers'].sum()) < 1e-12
            assert abs(sum(outcome['utilities']) + outcome['metrics']['net_revenue'] -
                       outcome['metrics']['value']) < 1e-12
            np.testing.assert_array_equal(outcome['completed'],
                                          outcome['final_capacities'] >= PHYSICAL_DEMANDS[physical])
            np.testing.assert_allclose(outcome['bonuses'], BONUS_ACTIONS[bonus] * outcome['completed'])
            expected_bill = np.where(outcome['initial_capacities'] > 0,
                                     .2 * outcome['initial_capacities'] + .8 * outcome['reports'], 0)
            np.testing.assert_allclose(outcome['payments'], expected_bill)
            if outcome['trade_success']:
                delta = outcome['utilities'] - outcome['utilities_before_trade']
                assert np.min(delta) >= -1e-12 and np.max(delta) > 0
            checked += 1
    # Bonus can prevent a trade requiring more compensation than the q menu.
    # Large-task buyer 0 is unallocated; small-task buyer 1 owns the 3-VM;
    # buyer 2 is outside their group and keeps the 2-VM.
    no_bonus = resolve_outcome(0, (2, 2, 3), 0, (1, 2))
    protected = resolve_outcome(0, (2, 2, 3), 7, (1, 2))  # s_1 = 1
    assert no_bonus['trade_success'] and not protected['trade_success']
    assert no_bonus['metrics']['successful_collusion_members'] == 2
    assert protected['metrics']['successful_collusion_members'] == 0
    # Useful mutual exchange survives every nonnegative bonus menu: the small
    # task stays complete and is compensated; the large task becomes complete.
    for bonus in range(16):
        benign = resolve_outcome(0, (2, 2, 1), bonus, (1, 0))
        assert benign['trade_success'] and benign['completed'].tolist() == [True, True, False]
        assert np.all(benign['utilities'] - benign['utilities_before_trade'] >= -1e-12)
    # Fixed A0 eligibility and bills cannot depend on type, bonus or joining.
    for reports in product(range(4), repeat=3):
        for owners in matchings((3, 2), reports, 0):
            capacities, payments = _initial(owners, (3, 2), np.array(reports))
            assert sorted(capacities[capacities > 0]) == sorted(c for c, i in zip((3, 2), owners) if i >= 0)
            for position in range(3):
                perm = np.roll(np.arange(3), position)
                inv = np.argsort(perm)
                permuted_reports = np.array(reports)[perm]
                permuted_owners = tuple(int(inv[i]) if i >= 0 else -1 for i in owners)
                assert permuted_owners in matchings((3, 2), tuple(permuted_reports), 0)
                c2, p2 = _initial(permuted_owners, (3, 2), permuted_reports)
                np.testing.assert_array_equal(c2, capacities[perm])
                np.testing.assert_array_equal(p2, payments[perm])
    # Rotate a unique two-member exchange, including its type and incentive.
    for position in range(3):
        actions = np.roll([2, 2, 1], position)
        owners = ((1 + position) % 3, position)
        result = resolve_outcome(position, actions, 13, owners)
        base = resolve_outcome(0, (2, 2, 1), 13, (1, 0))
        np.testing.assert_allclose(result['utilities'], np.roll(base['utilities'], position))
        np.testing.assert_array_equal(result['completed'], np.roll(base['completed'], position))
    market = Market(np.arange(24))
    before = market.get_state()
    u = rng.random((24, 4))
    public, metrics = market.step(np.arange(24) % 16, u)
    assert set(public) == PUBLIC_KEYS and all(v.shape == (24, 3) for v in public.values())
    np.testing.assert_allclose(metrics['utility'] + metrics['net_revenue'], metrics['value'], atol=1e-12)
    clone = market.clone()
    u2 = rng.random((24, 4))
    for left, right in zip(market.step(0, u2), clone.step(0, u2)):
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
    clone.set_state(before)
    for left, right in zip(clone.step(np.arange(24) % 16, u), (public, metrics)):
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
    assert not np.shares_memory(clone.q, market.q)
    forced_market = Market(np.array([0]))
    forced = np.array([[2, 2, 3]])
    forced_u = np.zeros((1, 4))
    original_q = forced_market.q.copy()
    pub, met = forced_market.step(0, forced_u, buyer_actions=forced)
    profile = 2 * 49 + 2 * 7 + 3
    exact = resolve_outcome(0, forced[0], 0, forced_market.tables.owners[profile, 0])
    np.testing.assert_allclose(forced_market.q[0, np.arange(3), forced[0]],
                               .05 * exact['utilities'] / 6.)
    assert set(pub) == PUBLIC_KEYS
    print(f'PASS: {checked} scalar/vector outcome checks, cash/transfer conservation, IR, '
          'original allocation/bills, completion rewards, blocked and benign exchanges, '
          'type permutation, observation keys, deterministic private replay.')


def benchmark(batch_size=1024, rounds=1000):
    """Throughput only: fixed random incentives, no result-based selection."""
    start = time.perf_counter()
    market = Market(np.arange(batch_size) % 24)
    preparation = time.perf_counter() - start
    rng = np.random.default_rng(80732)
    start = time.perf_counter()
    for _ in range(rounds):
        market.step(rng.integers(16, size=batch_size), rng.random((batch_size, 4)))
    elapsed = time.perf_counter() - start
    print(json.dumps({'purpose': 'implementation throughput, not experiment evidence',
                      'batch_size': batch_size, 'rounds': rounds,
                      'market_steps': batch_size * rounds, 'seconds': elapsed,
                      'market_steps_per_second': batch_size * rounds / elapsed,
                      'table_preparation_seconds': preparation,
                      'table_memory_bytes': market.tables.nbytes,
                      'table_axes': ['12 physical types', '16 bonus actions', '343 joint actions',
                                     f"{market.tables.initial_capacities.shape[1]} maximum original matchings"],
                      'cache_file': market.tables.cache_path.name}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.benchmark:
        benchmark()
    if not args.self_test and not args.benchmark:
        parser.print_help()
