"""V5 public-report-conditioned completion rewards, with fixed V4 economics.

This extension never changes the original allocation mapping, bill, learner,
trade menu or voluntary acceptance test. An advertised reward is eligible only
if its public report threshold is met; it is then paid only on completion.
Sampling is performed once, and V4's forced-action API applies the resulting
effective reward and exactly one Q update. No private type enters eligibility.

V4 source is imported read-only. Its immutable tables are shared in memory;
this module writes no V4 data, cache, checkpoint, protocol or result files.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import numpy as np
import incentive_market as base


BUYER_ACTIONS = base.BUYER_ACTIONS
ACTION_PROFILES = base.ACTION_PROFILES
PHYSICAL_DEMANDS = base.PHYSICAL_DEMANDS
PHYSICAL_VALUES = base.PHYSICAL_VALUES
MODEL_PHYSICAL_IDS = base.MODEL_PHYSICAL_IDS
MODEL_ETAS = base.MODEL_ETAS
DEMANDS = base.DEMANDS
VALUES = base.VALUES
MODELS = base.MODELS
METRIC_NAMES = base.METRIC_NAMES
PUBLIC_KEYS = base.PUBLIC_KEYS
TRADE_MODES = base.TRADE_MODES
BASE_BONUS_IDS = np.r_[np.arange(16), np.arange(1, 13), np.arange(1, 13)].astype(np.int64)
REWARD_THRESHOLDS = np.r_[np.zeros(16), np.full(12, 2), np.full(12, 3)].astype(np.int8)
BONUS_ACTIONS = base.BONUS_ACTIONS[BASE_BONUS_IDS].copy()
BONUS_MAX_COST = BONUS_ACTIONS.sum(axis=1)
TARGET_BUYERS = np.r_[np.full(16, -1), np.repeat(np.arange(3), 4),
                      np.repeat(np.arange(3), 4)].astype(np.int8)
REWARD_RULE = 'V5: completion and advertised public-report threshold'


def effective_bonus_action_ids(advertised_action_ids, buyer_action_ids):
    """Vectorized public eligibility; no model ID, demand, value or join input."""
    actions = np.asarray(buyer_action_ids)
    if (actions.ndim != 2 or actions.shape[1] != 3 or
            not np.issubdtype(actions.dtype, np.integer) or
            np.any((actions < 0) | (actions >= 7))):
        raise ValueError('buyer actions must have shape (batch,3), integer IDs 0..6')
    advertised = np.asarray(advertised_action_ids)
    if advertised.ndim == 0:
        advertised = np.full(len(actions), advertised.item())
    if (advertised.shape != (len(actions),) or
            not np.issubdtype(advertised.dtype, np.integer) or
            np.any((advertised < 0) | (advertised >= 40))):
        raise ValueError('advertised action IDs must be integers in 0..39')
    effective = BASE_BONUS_IDS[advertised].copy()
    conditional = advertised >= 16
    rows = np.flatnonzero(conditional)
    if len(rows):
        targets = TARGET_BUYERS[advertised[rows]]
        reports = BUYER_ACTIONS[actions[rows, targets], 0]
        eligible = reports >= REWARD_THRESHOLDS[advertised[rows]]
        effective[rows[~eligible]] = 0
    return effective


class Market(base.Market):
    """Same economic state and learner as V4; 40 publicly announced rewards."""
    def __init__(self, model_ids, trade_mode='bounded', supply=(3, 2)):
        # Avoid modifying any V4 path while reusing its read-only mathematics.
        super().__init__(model_ids, trade_mode, supply, use_disk_cache=False)

    def step(self, bonus_action_ids, random_u, buyer_actions=None):
        u = np.asarray(random_u, dtype=np.float64)
        if u.shape != (self.batch_size, 4) or not np.isfinite(u).all() or np.any((u < 0) | (u >= 1)):
            raise ValueError('random_u must have shape (batch,4), finite values in [0,1)')
        if buyer_actions is None:
            cdf = np.cumsum(self.probabilities(), axis=2)
            cdf[:, :, -1] = 1.
            sampled = (u[:, :3, None] >= cdf).sum(axis=2)
        else:
            sampled = np.asarray(buyer_actions)
        if sampled.shape != (self.batch_size, 3):
            raise ValueError('buyer_actions must have shape (batch,3)')
        effective = effective_bonus_action_ids(bonus_action_ids, sampled)
        # Reusing the same u is intentional: only u[:,3] now chooses the
        # original matching. The forced API neither resamples nor double-updates.
        return super().step(effective, u, buyer_actions=sampled)

    def get_state(self):
        state = super().get_state()
        state['reward_rule'] = REWARD_RULE
        return state

    def set_state(self, state):
        if state.get('reward_rule') != REWARD_RULE:
            raise ValueError('restore requires a V5 conditional-reward market state')
        super().set_state(state)

    def clone(self):
        clone = Market(self.model_ids, self.trade_mode, self.supply)
        clone.set_state(self.get_state())
        return clone


def resolve_outcome(physical_id, action_ids, bonus_action_id, owners,
                    trade_mode='bounded', supply=(3, 2)):
    """Private scalar audit API; reward eligibility still uses only reports."""
    actions = np.asarray(action_ids)
    effective = int(effective_bonus_action_ids(bonus_action_id, actions[None])[0])
    outcome = base.resolve_outcome(physical_id, action_ids, effective, owners, trade_mode, supply)
    outcome['advertised_bonus_action'] = int(bonus_action_id)
    outcome['effective_base_bonus_action'] = effective
    return outcome


@lru_cache(maxsize=576)
def _expected_base_stage(physical_id, effective_action_id, trade_mode, supply):
    """Read the immutable shared table in memory, without a V4 disk write."""
    if not 0 <= physical_id < 12:
        raise ValueError('physical ID outside catalogue')
    tables = base.get_tables(trade_mode, supply, use_disk_cache=False)
    profiles = np.repeat(np.arange(343), tables.matching_counts)
    matches = np.concatenate([np.arange(k) for k in tables.matching_counts])
    n = len(profiles)
    _, metrics, utilities = base._lookup(tables, np.full(n, physical_id),
                                         np.full(n, effective_action_id), profiles, matches)
    expected = {name: np.bincount(profiles, weights=v, minlength=343) / tables.matching_counts
                for name, v in metrics.items()}
    private = np.stack([np.bincount(profiles, weights=utilities[:, i], minlength=343) /
                        tables.matching_counts for i in range(3)], axis=1)
    return {'reports': tables.reports, 'joins': tables.joins,
            'utilities': private, 'metrics': expected}


def expected_stage(physical_id, bonus_action_id, trade_mode='bounded', supply=(3, 2)):
    """All 343 profile expectations, gating each profile before trade/payoffs.

    This is a private evaluation helper, not an information source for actors.
    It deliberately does not average eligibility before the nonlinear trade
    acceptance calculation: eligible and ineligible profiles use different
    original V4 payoff tables.
    """
    effective = effective_bonus_action_ids(bonus_action_id, ACTION_PROFILES)
    parts = {int(action): _expected_base_stage(physical_id, int(action), trade_mode, tuple(supply))
             for action in np.unique(effective)}
    reference = parts[int(effective[0])]
    utilities = np.empty((343, 3), dtype=np.float64)
    metrics = {name: np.empty(343) for name in METRIC_NAMES}
    for action, part in parts.items():
        selected = effective == action
        utilities[selected] = part['utilities'][selected]
        for name in METRIC_NAMES:
            metrics[name][selected] = part['metrics'][name][selected]
    return {'action_profiles': ACTION_PROFILES.copy(), 'reports': reference['reports'].copy(),
            'joins': reference['joins'].copy(), 'utilities': utilities, 'metrics': metrics,
            'effective_base_bonus_actions': effective}


def self_test():
    """Design checks only: no validation selection or formal policy evaluation."""
    assert BONUS_ACTIONS.shape == (40, 3)
    np.testing.assert_array_equal(BONUS_ACTIONS[:16], base.BONUS_ACTIONS)
    eligibility_checks = 0
    for advertised in range(40):
        got = effective_bonus_action_ids(advertised, ACTION_PROFILES)
        for i, profile in enumerate(ACTION_PROFILES):
            expected = advertised if advertised < 16 else int(BASE_BONUS_IDS[advertised])
            if advertised >= 16:
                target = int(TARGET_BUYERS[advertised])
                if BUYER_ACTIONS[profile[target], 0] < REWARD_THRESHOLDS[advertised]:
                    expected = 0
            assert got[i] == expected
            eligibility_checks += 1
    rng = np.random.default_rng(851731)
    size = 960
    model_ids = np.arange(size) % 24
    advertised = np.arange(size) % 40
    checks = 0
    for mode in TRADE_MODES:
        for forced in (False, True):
            env = Market(model_ids, mode)
            ref = base.Market(model_ids, mode, use_disk_cache=False)
            initial_q = rng.normal(0, .3, env.q.shape)
            env.q[:] = initial_q
            ref.q[:] = initial_q
            u = rng.random((size, 4))
            if forced:
                actions = rng.integers(0, 7, (size, 3))
            else:
                cdf = np.cumsum(env.probabilities(), axis=2)
                cdf[:, :, -1] = 1.
                actions = (u[:, :3, None] >= cdf).sum(axis=2)
            effective = effective_bonus_action_ids(advertised, actions)
            result = env.step(advertised, u, buyer_actions=actions if forced else None)
            reference = ref.step(effective, u, buyer_actions=actions)
            for left, right in zip(result, reference):
                for key in left:
                    np.testing.assert_array_equal(left[key], right[key])
            np.testing.assert_array_equal(env.q, ref.q)
            assert env.step_count == 1 and set(result[0]) == PUBLIC_KEYS
            rewards = result[0]['bonuses']
            reports = result[0]['reports']
            active_condition = advertised >= 16
            rows = np.flatnonzero(active_condition)
            targets = TARGET_BUYERS[advertised[rows]]
            assert np.all((rewards[rows, targets] == 0) |
                          (reports[rows, targets] >= REWARD_THRESHOLDS[advertised[rows]]))
            np.testing.assert_allclose(result[1]['utility'] + result[1]['net_revenue'],
                                       result[1]['value'], atol=1e-12)
            clone = env.clone()
            for left, right in zip(env.step(39, u), clone.step(39, u)):
                for key in left:
                    np.testing.assert_array_equal(left[key], right[key])
            assert isinstance(clone, Market)
            checks += size
    # Meeting a public threshold is not a proof of truthful demand: a small
    # buyer may report 3, win the 3-VM, complete, and legitimately receive it.
    witness = resolve_outcome(0, (0, 5, 0), 34, (1, -1))
    assert witness['reports'][1] == 3 > PHYSICAL_DEMANDS[0, 1]
    assert witness['bonuses'][1] == 1.
    assert witness['metrics']['gross_revenue'] == 3.
    # Every original reward remains exactly the same economic experiment.
    env = Market(model_ids)
    ref = base.Market(model_ids, use_disk_cache=False)
    u = rng.random((size, 4))
    for left, right in zip(env.step(np.arange(size) % 16, u),
                           ref.step(np.arange(size) % 16, u)):
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
    np.testing.assert_array_equal(env.q, ref.q)
    print(f'PASS: {eligibility_checks} eligibility profiles, {checks} exact effective-reward '
          'outcomes/Q updates, single sampling/update, 16-action identity, private clone, '
          'cash conservation, public observation whitelist, and overreporting boundary.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.print_help()
