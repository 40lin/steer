"""Vectorized, no-rebate Harvesting Market used for actual learning experiments.

The policy-facing contract is ``step(actions, random_u) -> (observation, metrics)``.
Only ``observation`` may be sent to a platform actor; metrics and get_state() are
privileged evaluation interfaces. Payments/revenue are REALIZED cash flows for
the sampled matching, not their expectations. There are no operating costs here.

Model IDs are (market_index * 2 + grand_coalition) * 2 + learning_rate_index.
The 16-model catalogue, prices and capacity are fixed. Buyers have 4 reports
(0 means opt out), coalitions have the 16 lexicographic joint reports.

``canonical`` is the default learner: economically equivalent eligibility rules
have identical features. ``raw`` reproduces the original rho-labelled proposal
and is only a sensitivity experiment. No true type appears in observations.

Run ``python steering_env.py --self-test`` for exact-validator comparisons,
observation isolation, common-random-stream replay and learner-update checks.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
from itertools import product
from pathlib import Path
import importlib.util
import time

import numpy as np

CAPACITIES = np.array([3, 2], dtype=np.int64)
RHO_FRACTIONS = (Fraction(0), Fraction(1, 2), Fraction(2, 3),
                 Fraction(4, 5), Fraction(1))
RHO_VALUES = np.array([float(x) for x in RHO_FRACTIONS], dtype=np.float64)
PRIMARY_ACTION_INDICES = np.array([0, 1, 2, 4], dtype=np.int64)
DEMANDS = np.array([[3, 2], [3, 2], [2, 2], [1, 3]], dtype=np.int64)
VALUES = np.array([[9., 5.], [2.4, 1.6], [5., 5.], [2., 8.]])
LEARNING_RATES = np.array([0.05, 0.20])
JOINT_REPORTS = np.array(list(product(range(4), repeat=2)), dtype=np.int64)
TEMPERATURE = 0.20
MODEL_COUNT = 16

REQUIRED_BIDS = np.array([
    [max(1, (rho.numerator * int(c) + rho.denominator - 1)
         // rho.denominator) for c in CAPACITIES]
    for rho in RHO_FRACTIONS], dtype=np.int64)

_raw = np.column_stack((np.ones(5), RHO_VALUES, np.zeros(5),
                        np.ones(5), np.full(5, 2/3)))
_canonical = np.column_stack((np.ones(5), REQUIRED_BIDS[:, 0] / 3,
                              REQUIRED_BIDS[:, 1] / 2,
                              np.ones(5), np.full(5, 2/3)))
FEATURE_TABLES = {
    'raw': _raw / np.linalg.norm(_raw, axis=1, keepdims=True),
    'canonical': _canonical / np.linalg.norm(_canonical, axis=1, keepdims=True),
    # Independent Q values per economic eligibility class, with duplicate
    # numerical labels 0.8 and 1 deliberately sharing exactly one state.
    'tabular': np.eye(5)[np.array([0, 1, 2, 3, 3])],
}

PUBLIC_OBSERVATION_KEYS = frozenset({
    'time', 'capacities', 'rho_action', 'rho', 'required_bids', 'reports',
    'owners', 'allocated_capacities', 'payments', 'rebates', 'revenue', 'volume',
})


def decode_model_ids(model_ids):
    """Catalogue description, not an actor observation."""
    ids = np.asarray(model_ids, dtype=np.int64)
    if ids.ndim != 1 or np.any((ids < 0) | (ids >= MODEL_COUNT)):
        raise ValueError('model_ids must be a one-dimensional array in 0..15')
    return ids // 4, (ids // 2) % 2, ids % 2


def _max_capacity_matchings(reports, rho):
    candidates = []
    for owners in product(range(-1, 2), repeat=2):
        occupied = [i for i in owners if i >= 0]
        if len(occupied) != len(set(occupied)):
            continue
        if any(i >= 0 and not (0 < reports[i] <= int(c)
                              and reports[i] >= rho * int(c))
               for c, i in zip(CAPACITIES, owners)):
            continue
        capacity = sum(int(c) for c, i in zip(CAPACITIES, owners) if i >= 0)
        candidates.append((capacity, owners))
    best = max(x[0] for x in candidates)
    return [x[1] for x in candidates if x[0] == best]


def _packing(demands, values, owners):
    """Exact indivisible-job packing, with lexicographic tie-breaking.

    Jobs can use any leased VM, including multiple jobs on one VM. One job
    cannot span VMs. Job-placement ties do not change coalition learning reward.
    """
    best_value = -1.
    best_served = None
    for placements in product(range(-1, 2), repeat=2):
        feasible = True
        for j in range(2):
            load = sum(int(demands[i]) for i in range(2) if placements[i] == j)
            if load > (int(CAPACITIES[j]) if owners[j] >= 0 else 0):
                feasible = False
                break
        if not feasible:
            continue
        served = np.array([j >= 0 for j in placements], dtype=np.int64)
        value = float(np.dot(values, served))
        if value > best_value:
            best_value, best_served = value, served
    return best_served


def _build_tables():
    matches = [[_max_capacity_matchings(tuple(map(int, b)), rho)
                for b in JOINT_REPORTS] for rho in RHO_FRACTIONS]
    max_outcomes = max(len(x) for row in matches for x in row)
    shape = (5, 16, max_outcomes)
    counts = np.empty((5, 16), dtype=np.int64)
    owners_table = np.full(shape + (2,), -1, dtype=np.int64)
    allocation = np.zeros(shape + (2,), dtype=np.int64)
    payment = np.zeros(shape + (2,), dtype=np.float64)
    served = np.zeros((4, 2) + shape + (2,), dtype=np.int64)
    utility = np.zeros((4, 2) + shape + (2,), dtype=np.float64)
    for r, row in enumerate(matches):
        for a, outcomes in enumerate(row):
            counts[r, a] = len(outcomes)
            reports = JOINT_REPORTS[a]
            for k, owners in enumerate(outcomes):
                owners_table[r, a, k] = owners
                for c, i in zip(CAPACITIES, owners):
                    if i >= 0:
                        allocation[r, a, k, i] = c
                        payment[r, a, k, i] = (int(c) + 4 * int(reports[i])) / 5
                for market in range(4):
                    independent = (allocation[r, a, k] >= DEMANDS[market]).astype(np.int64)
                    grand = _packing(DEMANDS[market], VALUES[market], owners)
                    for coalition, complete in enumerate((independent, grand)):
                        served[market, coalition, r, a, k] = complete
                        utility[market, coalition, r, a, k] = (
                            complete * VALUES[market] - payment[r, a, k])
    return {
        'counts': counts, 'owners': owners_table, 'allocation': allocation,
        'payment': payment, 'revenue': payment.sum(axis=-1),
        'volume': allocation.sum(axis=-1), 'served': served, 'utility': utility,
    }


TABLES = _build_tables()
for _array in (CAPACITIES, RHO_VALUES, PRIMARY_ACTION_INDICES, DEMANDS, VALUES,
               LEARNING_RATES, JOINT_REPORTS, REQUIRED_BIDS,
               *FEATURE_TABLES.values(), *TABLES.values()):
    _array.setflags(write=False)


def _softmax(values):
    logits = values / TEMPERATURE
    logits = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def _categorical(probabilities, u):
    # Counting crossed CDF boundaries handles u=0 and tiny rounding errors.
    selected = (u[..., None] >= np.cumsum(probabilities, axis=-1)).sum(axis=-1)
    return np.minimum(selected, probabilities.shape[-1] - 1)


class BatchMarket:
    """Synchronous vector environments with explicitly separable observations.

    Args:
        model_ids: One hidden catalogue ID per environment. Actor must not see it.
        seed: Seed for the fallback RNG. Prefer explicit common random streams.
        learner_mode: 'canonical' (primary) or 'raw' (sensitivity).

    ``random_u`` has shape (B,3), all entries in [0,1). Columns are first
    independent report OR grand-coalition joint report, second independent
    report, and matching. The coalition intentionally does not use column 1.
    All policies using the same streams obtain paired exogenous random numbers.
    Model weights, RNG identifiers and hidden utilities are never observations.
    """
    def __init__(self, model_ids, seed=0, learner_mode='canonical'):
        if learner_mode not in FEATURE_TABLES:
            raise ValueError("learner_mode must be 'canonical', 'raw', or 'tabular'")
        self.learner_mode = learner_mode
        self.reset(model_ids=model_ids, seed=seed)

    @property
    def batch_size(self):
        return len(self._model_ids)

    def reset(self, model_ids=None, seed=None):
        if model_ids is None:
            if not hasattr(self, '_model_ids'):
                raise ValueError('model_ids are required for the first reset')
            ids = self._model_ids.copy()
        else:
            ids = np.atleast_1d(np.asarray(model_ids, dtype=np.int64)).copy()
        self._market, self._coalition, eta_index = decode_model_ids(ids)
        if len(ids) == 0:
            raise ValueError('batch cannot be empty')
        self._model_ids = ids
        self._eta = LEARNING_RATES[eta_index]
        self._w_independent = np.zeros((len(ids), 2, 4, 5), dtype=np.float64)
        self._w_grand = np.zeros((len(ids), 16, 5), dtype=np.float64)
        self._rng = np.random.default_rng(seed)
        self.time = 0
        self._last_obs = None
        return {
            'time': 0,
            'capacities': np.broadcast_to(CAPACITIES, (len(ids), 2)).copy(),
        }

    def _actions(self, actions):
        a = np.asarray(actions)
        if a.ndim == 0:
            a = np.full(self.batch_size, a.item())
        if a.shape != (self.batch_size,) or not np.issubdtype(a.dtype, np.integer):
            raise ValueError('actions must be integer rho indices of shape (B,)')
        a = a.astype(np.int64, copy=False)
        if np.any((a < 0) | (a >= len(RHO_VALUES))):
            raise ValueError('rho action index outside 0..4')
        return a

    def _predictions(self, actions):
        phi = FEATURE_TABLES[self.learner_mode][actions]
        qi = np.einsum('biaf,bf->bia', self._w_independent, phi)
        qg = np.einsum('baf,bf->ba', self._w_grand, phi)
        return phi, qi, qg

    def report_probabilities(self, actions):
        """PRIVILEGED evaluation: actual hidden learner's joint report law.

        Do not send this to the actor. A belief controller can call this on its
        own hypothetical candidate environments, never on the true environment.
        """
        a = self._actions(actions)
        _, qi, qg = self._predictions(a)
        pi = _softmax(qi)
        independent = (pi[:, 0, :, None] * pi[:, 1, None, :]).reshape(-1, 16)
        return np.where(self._coalition[:, None].astype(bool), _softmax(qg), independent)

    def action_probabilities(self, rho_indices):
        """Alias for report_probabilities; privileged unless used on hypotheses."""
        return self.report_probabilities(rho_indices)

    def step(self, actions, random_u=None):
        a = self._actions(actions)
        if random_u is None:
            random_u = self._rng.random((self.batch_size, 3))
        u = np.asarray(random_u, dtype=np.float64)
        if u.shape != (self.batch_size, 3) or not np.all(np.isfinite(u)):
            raise ValueError('random_u must be finite with shape (B,3)')
        if np.any((u < 0) | (u >= 1)):
            raise ValueError('random_u entries must be in [0,1)')
        phi, qi, qg = self._predictions(a)
        pi = _softmax(qi)
        ri = _categorical(pi, u[:, :2])
        joint_i = 4 * ri[:, 0] + ri[:, 1]
        joint_g = _categorical(_softmax(qg), u[:, 0])
        joint = np.where(self._coalition.astype(bool), joint_g, joint_i)
        outcome = np.floor(u[:, 2] * TABLES['counts'][a, joint]).astype(np.int64)
        return self._realize_and_update(a, joint, outcome, phi, qi, qg)

    def condition_on_public(self, actions, reports, owners):
        """Advance hypothetical candidate learners on an observed public path.

        Returns each candidate's likelihood of the reports BEFORE updating.
        Matching likelihood is common across candidates and cancels in Bayes'
        rule. This method intentionally does not use any actual hidden reward.
        Use only for candidate models; it forces observed actions off-policy.
        """
        a = self._actions(actions)
        reports = np.asarray(reports, dtype=np.int64)
        owners = np.asarray(owners, dtype=np.int64)
        if reports.shape != (self.batch_size, 2) or owners.shape != reports.shape:
            raise ValueError('reports and owners must have shape (B,2)')
        if np.any((reports < 0) | (reports > 3)):
            raise ValueError('invalid report')
        joint = 4 * reports[:, 0] + reports[:, 1]
        likelihood = self.report_probabilities(a)[np.arange(self.batch_size), joint]
        candidate_owners = TABLES['owners'][a, joint]
        valid = np.arange(candidate_owners.shape[1])[None, :] < TABLES['counts'][a, joint, None]
        equal = np.all(candidate_owners == owners[:, None, :], axis=-1) & valid
        if not np.all(equal.any(axis=-1)):
            raise ValueError('observed owners are not an optimal feasible matching')
        outcome = equal.argmax(axis=-1)
        phi, qi, qg = self._predictions(a)
        self._realize_and_update(a, joint, outcome, phi, qi, qg)
        return likelihood

    def assimilate(self, rho_indices, reports, owners):
        """Alias: force the public trajectory on hypothetical model states."""
        return self.condition_on_public(rho_indices, reports, owners)

    def _realize_and_update(self, a, joint, outcome, phi, qi, qg):
        bsize = self.batch_size
        rows = np.arange(bsize)
        reports = JOINT_REPORTS[joint].copy()
        payment = TABLES['payment'][a, joint, outcome].copy()
        allocation = TABLES['allocation'][a, joint, outcome].copy()
        served = TABLES['served'][self._market, self._coalition, a, joint, outcome]
        utilities = TABLES['utility'][self._market, self._coalition, a, joint, outcome]
        individual_rows = rows[self._coalition == 0]
        if len(individual_rows):
            rr = reports[individual_rows]
            current = qi[individual_rows[:, None], np.arange(2)[None, :], rr]
            delta = (self._eta[individual_rows, None]
                     * (utilities[individual_rows] / 3. - current))
            self._w_independent[individual_rows[:, None], np.arange(2)[None, :], rr] += (
                delta[..., None] * phi[individual_rows, None, :])
        grand_rows = rows[self._coalition == 1]
        if len(grand_rows):
            jj = joint[grand_rows]
            delta = self._eta[grand_rows] * (
                utilities[grand_rows].sum(axis=-1) / 6. - qg[grand_rows, jj])
            self._w_grand[grand_rows, jj] += delta[:, None] * phi[grand_rows]
        self.time += 1
        obs = {
            'time': self.time,
            'capacities': np.broadcast_to(CAPACITIES, (bsize, 2)).copy(),
            'rho_action': a.copy(), 'rho': RHO_VALUES[a].copy(),
            'required_bids': REQUIRED_BIDS[a].copy(), 'reports': reports,
            'owners': TABLES['owners'][a, joint, outcome].copy(),
            'allocated_capacities': allocation,
            'payments': payment, 'rebates': np.zeros_like(payment),
            'revenue': payment.sum(axis=-1),
            'volume': allocation.sum(axis=-1).astype(np.float64),
        }
        active = reports > 0
        true_demands = DEMANDS[self._market]
        values = VALUES[self._market]
        metrics = {
            'completed_jobs': served.sum(axis=-1).astype(np.float64),
            'completed_value': (served * values).sum(axis=-1),
            # Before internal transfers; not a voluntary-alliance IR certificate.
            'utilities_before_transfers': utilities.copy(),
            'total_user_utility': utilities.sum(axis=-1),
            'active_buyers': active.sum(axis=-1).astype(np.float64),
            'underreport_amount_active': (np.maximum(true_demands - reports, 0) * active).sum(axis=-1),
            'underreport_count_active': ((reports < true_demands) & active).sum(axis=-1),
            'overreport_amount': np.maximum(reports - true_demands, 0).sum(axis=-1),
            'overreport_count': (reports > true_demands).sum(axis=-1),
        }
        self._last_obs = {key: value.copy() if isinstance(value, np.ndarray) else value
                          for key, value in obs.items()}
        return obs, metrics

    def get_state(self):
        """PRIVILEGED snapshot for exact replay/evaluator planning, never actor input."""
        return {
            'model_ids': self._model_ids.copy(), 'learner_mode': self.learner_mode,
            'w_independent': self._w_independent.copy(), 'w_grand': self._w_grand.copy(),
            'time': self.time, 'rng_state': copy.deepcopy(self._rng.bit_generator.state),
            'last_obs': copy.deepcopy(self._last_obs),
        }

    def set_state(self, state):
        self.learner_mode = state['learner_mode']
        if self.learner_mode not in FEATURE_TABLES:
            raise ValueError('invalid learner mode in state')
        self.reset(state['model_ids'])
        if (state['w_independent'].shape != self._w_independent.shape
                or state['w_grand'].shape != self._w_grand.shape):
            raise ValueError('state weight shape mismatch')
        self._w_independent[:] = state['w_independent']
        self._w_grand[:] = state['w_grand']
        self.time = int(state['time'])
        self._rng.bit_generator.state = copy.deepcopy(state['rng_state'])
        self._last_obs = copy.deepcopy(state['last_obs'])
        return self

    def clone(self):
        """PRIVILEGED independent copy, preserving fallback RNG state."""
        return BatchMarket(self._model_ids, learner_mode=self.learner_mode).set_state(self.get_state())


def self_test():
    validator_path = Path(__file__).with_name('validate_micro_model.py')
    spec = importlib.util.spec_from_file_location('_fraction_micro_validator', validator_path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    comparisons = 0
    for market in range(4):
        demands = tuple(map(int, DEMANDS[market]))
        values = tuple(Fraction(str(x)) for x in VALUES[market])
        for r, rho in enumerate(RHO_FRACTIONS):
            for a, reports in enumerate(JOINT_REPORTS):
                b = tuple(map(int, reports))
                exact = validator.evaluate(demands, values, (3, 2), b, rho)
                k = TABLES['counts'][r, a]
                assert tuple(map(tuple, TABLES['owners'][r, a, :k])) == validator.matchings((3, 2), b, rho)
                checks = (
                    (TABLES['revenue'][r, a, :k].mean(), float(exact['revenue'])),
                    (TABLES['volume'][r, a, :k].mean(), float(exact['volume'])),
                    (TABLES['utility'][market, 0, r, a, :k].mean(axis=0),
                     np.array([float(x) for x in exact['private_utilities']])),
                    (TABLES['utility'][market, 1, r, a, :k].sum(axis=-1).mean(),
                     float(exact['coalition_utility'])),
                    ((TABLES['served'][market, 1, r, a, :k] * VALUES[market]).sum(axis=-1).mean(),
                     float(exact['coalition_completed_value'])),
                )
                for actual, expected in checks:
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
                    comparisons += 1
    ids = np.arange(16)
    rng = np.random.default_rng(98765)
    for mode in FEATURE_TABLES:
        env = BatchMarket(ids, seed=5, learner_mode=mode)
        candidates = BatchMarket(ids, seed=999, learner_mode=mode)
        np.testing.assert_allclose(env.report_probabilities(np.zeros(16, dtype=int)), 1/16)
        for t in range(40):
            a = rng.integers(0, 5, 16)
            u = rng.random((16, 3))
            obs, metrics = env.step(a, u)
            assert frozenset(obs) == PUBLIC_OBSERVATION_KEYS
            assert not any(key in obs for key in ('model_ids', 'demands', 'values', 'coalition',
                                                  'utilities', 'seed', 'rng_state', 'w_independent', 'w_grand'))
            likelihood = candidates.condition_on_public(a, obs['reports'], obs['owners'])
            assert np.all((likelihood > 0) & (likelihood <= 1))
            np.testing.assert_array_equal(candidates._w_independent, env._w_independent)
            np.testing.assert_array_equal(candidates._w_grand, env._w_grand)
            np.testing.assert_allclose(metrics['total_user_utility'] + obs['revenue'],
                                       metrics['completed_value'], rtol=0, atol=3e-14)
            np.testing.assert_allclose(obs['payments'].sum(axis=-1), obs['revenue'])
        clone = env.clone()
        out_a = env.step(np.full(16, 2, dtype=int))
        out_b = clone.step(np.full(16, 2, dtype=int))
        for left, right in zip(out_a, out_b):
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])
        # Returned public buffers do not alias the hidden environment state.
        snapshot = env.get_state()
        out_a[0]['reports'][:] = 999
        np.testing.assert_array_equal(snapshot['w_independent'], env._w_independent)
        assert np.max(env._last_obs['reports']) <= 3
    lo, hi = BatchMarket(ids), BatchMarket(ids)
    for _ in range(80):
        u = rng.random((16, 3))
        x, _ = lo.step(np.full(16, 3, dtype=int), u)
        y, _ = hi.step(np.full(16, 4, dtype=int), u)
        for key in ('reports', 'owners', 'payments', 'revenue', 'volume'):
            np.testing.assert_array_equal(x[key], y[key])
        np.testing.assert_array_equal(lo._w_independent, hi._w_independent)
        np.testing.assert_array_equal(lo._w_grand, hi._w_grand)
    # Initial update is exactly eta*normalized_reward*unit_feature.
    env = BatchMarket([0, 2])
    obs, metrics = env.step(np.array([0, 0]), np.array([[.5, .5, .25], [.7, .2, .75]]))
    for row in range(2):
        phi = FEATURE_TABLES['canonical'][0]
        if row == 0:
            for i in range(2):
                np.testing.assert_allclose(env._w_independent[row, i, obs['reports'][row, i]],
                    .05 * metrics['utilities_before_transfers'][row, i] / 3 * phi)
        else:
            joint = 4 * obs['reports'][row, 0] + obs['reports'][row, 1]
            np.testing.assert_allclose(env._w_grand[row, joint],
                .05 * metrics['total_user_utility'][row] / 6 * phi)
    print(f'PASS: {comparisons} exact Fraction comparisons; public-observation contract; '
          'candidate replay; cash-flow identity; clone replay; canonical .8/1 equivalence; '
          'initial gradient update.')


def benchmark(batch_size=4096, steps=400):
    env = BatchMarket(np.arange(batch_size) % 16)
    rng = np.random.default_rng(456)
    action = np.full(batch_size, 2, dtype=np.int64)
    start = time.perf_counter()
    revenue = 0.
    for _ in range(steps):
        obs, _ = env.step(action, rng.random((batch_size, 3)))
        revenue += float(obs['revenue'].sum())
    elapsed = time.perf_counter() - start
    print({'batch_size': batch_size, 'steps': steps, 'seconds': elapsed,
           'environment_steps_per_second': batch_size * steps / elapsed,
           'mean_total_revenue': revenue / batch_size})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    if args.self_test or not args.benchmark:
        self_test()
    if args.benchmark:
        benchmark()
