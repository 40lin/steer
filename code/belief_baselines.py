"""Public-history Bayesian baselines for the 16-model no-rebate experiment.

These are deliberately simple controls, NOT a reproduction of the trained
ICLR 2025 method: posterior immediate-revenue greedy, and a fixed 40-round probe
followed by a single posterior immediate-revenue choice. The latter is called
``fete_fixed`` to distinguish it from a learned FETE exploration/exploitation
policy. Neither controller is trained or claims an optimal long-run objective.

Only reported bids, public VM owners and the platform's own rule enter observe().
All payoff/learner states used in inference belong to hypothetical catalogue
models. No method accepts a true environment, true model ID or private reward.
"""
from __future__ import annotations

import argparse
import numpy as np

from steering_env import (BatchMarket, MODEL_COUNT, PRIMARY_ACTION_INDICES,
                          RHO_VALUES, TABLES)


_valid_outcomes = (np.arange(TABLES['revenue'].shape[-1])[None, None, :]
                   < TABLES['counts'][..., None])
EXPECTED_REVENUE_BY_REPORT = (
    (TABLES['revenue'] * _valid_outcomes).sum(axis=-1) / TABLES['counts'])
EXPECTED_VOLUME_BY_REPORT = (
    (TABLES['volume'] * _valid_outcomes).sum(axis=-1) / TABLES['counts'])
EXPECTED_REVENUE_BY_REPORT.setflags(write=False)
EXPECTED_VOLUME_BY_REPORT.setflags(write=False)


class BeliefTracker:
    """Exact finite-catalogue filter plus two simple action recommendations.

    Args:
        batch_size: Independent worlds. Every world begins with 16 hypotheses.
        learner_mode: Same declared learner representation as the experiment.
        block_size: Rules may change once per block (default 10 rounds).
        control_rounds: Greedy control ends here (default 200); retain its last
            selected rule afterwards, matching the main no-rebate holding phase.
        exploration_rounds: FETE probe length, default 40, then freeze forever.

    Main API:
        posterior -> (B,16) copy
        observe(actions, reports, owners) -> (B,) predictive report likelihood
        predict_revenue() -> (B,4), columns env indices [0,1,2,4]
        recommend('bayes_greedy'|'fete_fixed') -> (B,) environment rho indices

    Call recommend before executing a round, then observe the realized report
    and VM-owner tuple after that round. Calling recommend each round is fine:
    the block cache enforces the chosen cadence. No observation contains types.
    """
    def __init__(self, batch_size, learner_mode='canonical', block_size=10,
                 control_rounds=200, exploration_rounds=40):
        if int(batch_size) != batch_size or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        if block_size <= 0 or control_rounds <= 0 or exploration_rounds <= 0:
            raise ValueError('all round counts must be positive')
        if control_rounds % block_size or exploration_rounds % block_size:
            raise ValueError('control/exploration lengths must be whole blocks')
        if exploration_rounds >= control_rounds:
            raise ValueError('exploration must finish before control ends')
        self.batch_size = int(batch_size)
        self.learner_mode = learner_mode
        self.block_size = int(block_size)
        self.control_rounds = int(control_rounds)
        self.exploration_rounds = int(exploration_rounds)
        self.reset()

    def reset(self):
        # Same full prior for every world; there is no true-type argument.
        hypotheses = np.tile(np.arange(MODEL_COUNT, dtype=np.int64), self.batch_size)
        self._candidates = BatchMarket(hypotheses, seed=0, learner_mode=self.learner_mode)
        self._log_posterior = np.full((self.batch_size, MODEL_COUNT), -np.log(MODEL_COUNT))
        self.round = 0
        self._cached_actions = {}
        self._cached_blocks = {}
        self._fete_frozen = None
        return self

    @property
    def posterior(self):
        return np.exp(self._log_posterior).copy()

    def _world_actions(self, actions):
        a = np.asarray(actions)
        if a.ndim == 0:
            a = np.full(self.batch_size, a.item())
        if a.shape != (self.batch_size,) or not np.issubdtype(a.dtype, np.integer):
            raise ValueError('actions must have integer shape (B,)')
        if np.any((a < 0) | (a >= len(RHO_VALUES))):
            raise ValueError('rho index must be in 0..4')
        return a.astype(np.int64, copy=False)

    def observe(self, actions, reports, owners):
        """Bayes-update and advance every hypothetical learner on actual actions.

        The observed matching is conditioned on, not re-sampled. Its conditional
        probability is identical for all private-type hypotheses and cancels in
        the posterior. Its realized allocation still affects each model's reward
        and its next learner state. Returned likelihood concerns the bids only.
        """
        a = self._world_actions(actions)
        r = np.asarray(reports)
        o = np.asarray(owners)
        if (r.shape != (self.batch_size, 2) or o.shape != r.shape
                or not np.issubdtype(r.dtype, np.integer)
                or not np.issubdtype(o.dtype, np.integer)):
            raise ValueError('reports and owners must be integer arrays (B,2)')
        likelihood = self._candidates.assimilate(
            np.repeat(a, MODEL_COUNT),
            np.repeat(r, MODEL_COUNT, axis=0),
            np.repeat(o, MODEL_COUNT, axis=0)).reshape(self.batch_size, MODEL_COUNT)
        # Clipping only protects a numeric log(0), not semantic model recovery.
        log_likelihood = np.log(np.maximum(likelihood, np.finfo(np.float64).tiny))
        unnormalized = self._log_posterior + log_likelihood
        max_log = unnormalized.max(axis=1, keepdims=True)
        log_evidence = max_log[:, 0] + np.log(np.exp(unnormalized - max_log).sum(axis=1))
        self._log_posterior = unnormalized - log_evidence[:, None]
        self.round += 1
        return np.exp(log_evidence)

    def _predict_table(self, table):
        posterior = np.exp(self._log_posterior)
        result = np.empty((self.batch_size, len(PRIMARY_ACTION_INDICES)))
        candidate_count = self.batch_size * MODEL_COUNT
        for col, action in enumerate(PRIMARY_ACTION_INDICES):
            probabilities = self._candidates.action_probabilities(
                np.full(candidate_count, action, dtype=np.int64))
            expected = (probabilities @ table[action]).reshape(self.batch_size, MODEL_COUNT)
            result[:, col] = (posterior * expected).sum(axis=1)
        return result

    def predict_revenue(self):
        """Posterior expected NEXT-ROUND revenue for the 4 distinct rules.

        This does not predict a 10-round return or solve a dynamic program.
        Actual environment cash-flow rewards remain sampled realized payments.
        """
        return self._predict_table(EXPECTED_REVENUE_BY_REPORT)

    def predict_volume(self):
        """Analogous next-round volume prediction, useful for diagnostic plots."""
        return self._predict_table(EXPECTED_VOLUME_BY_REPORT)

    def recommend(self, policy='bayes_greedy'):
        """Return environment rho indices, not local 0..3 policy indices."""
        if policy not in ('bayes_greedy', 'fete_fixed'):
            raise ValueError("policy must be 'bayes_greedy' or 'fete_fixed'")
        if policy == 'fete_fixed':
            if self.round < self.exploration_rounds:
                position = (self.round // self.block_size) % len(PRIMARY_ACTION_INDICES)
                return np.full(self.batch_size, PRIMARY_ACTION_INDICES[position], dtype=np.int64)
            if self._fete_frozen is None:
                # Must be called at the switch, not retrospectively after more data.
                if self.round != self.exploration_rounds:
                    raise RuntimeError('call fete_fixed.recommend at the end of exploration')
                self._fete_frozen = PRIMARY_ACTION_INDICES[self.predict_revenue().argmax(axis=1)].copy()
            return self._fete_frozen.copy()
        if self.round >= self.control_rounds:
            if policy not in self._cached_actions:
                raise RuntimeError('no preceding greedy action to freeze for the holding phase')
            return self._cached_actions[policy].copy()
        block = self.round // self.block_size
        if self._cached_blocks.get(policy) != block:
            self._cached_actions[policy] = PRIMARY_ACTION_INDICES[
                self.predict_revenue().argmax(axis=1)].copy()
            self._cached_blocks[policy] = block
        return self._cached_actions[policy].copy()


def bayes_greedy(tracker):
    """Convenience policy, block-cached and frozen after control_rounds."""
    return tracker.recommend('bayes_greedy')


def fete_fixed(tracker):
    """Convenience policy: fixed four-block probe, then one permanent choice."""
    return tracker.recommend('fete_fixed')


def self_test():
    ids = np.arange(16, dtype=np.int64)
    rng = np.random.default_rng(61234)
    for mode in ('canonical', 'raw'):
        true_env = BatchMarket(ids, learner_mode=mode)
        tracker = BeliefTracker(len(ids), learner_mode=mode)
        np.testing.assert_allclose(tracker.posterior, 1/16)
        # At zero weights every model has uniform joint reports, so this can be
        # computed directly without consulting the true simulation at all.
        expected = EXPECTED_REVENUE_BY_REPORT[PRIMARY_ACTION_INDICES].mean(axis=1)
        np.testing.assert_allclose(tracker.predict_revenue(), np.tile(expected, (len(ids), 1)))
        reference_log_weights = np.full((len(ids), 16), -np.log(16.))
        for t in range(50):
            actions = rng.choice(PRIMARY_ACTION_INDICES, size=len(ids))
            before = tracker._candidates.action_probabilities(np.repeat(actions, 16)).reshape(16, 16, 16)
            obs, _ = true_env.step(actions, rng.random((16, 3)))
            joint = 4 * obs['reports'][:, 0] + obs['reports'][:, 1]
            observed_like = np.take_along_axis(before, joint[:, None, None], axis=2)[..., 0]
            reference_log_weights += np.log(observed_like)
            ref_max = reference_log_weights.max(axis=1, keepdims=True)
            ref_posterior = np.exp(reference_log_weights - ref_max)
            ref_posterior /= ref_posterior.sum(axis=1, keepdims=True)
            tracker.observe(actions, obs['reports'], obs['owners'])
            np.testing.assert_allclose(tracker.posterior, ref_posterior, rtol=2e-13, atol=2e-15)
            # Only the self-test accesses the actual hidden state to independently
            # verify that conditioning reproduces the true candidate exactly.
            true_state = true_env.get_state()
            candidate_state = tracker._candidates.get_state()
            true_candidate_rows = 16 * np.arange(16) + ids
            np.testing.assert_array_equal(candidate_state['w_independent'][true_candidate_rows],
                                          true_state['w_independent'])
            np.testing.assert_array_equal(candidate_state['w_grand'][true_candidate_rows],
                                          true_state['w_grand'])
        state_before_prediction = tracker._candidates.get_state()
        predicted = tracker.predict_revenue()
        assert predicted.shape == (16, 4) and np.all((predicted >= 0) & (predicted <= 5))
        np.testing.assert_array_equal(state_before_prediction['w_independent'],
                                      tracker._candidates.get_state()['w_independent'])
        # Self-test truth ids were never passed to the tracker: its constructor
        # and all update calls accepted only batch size and public arrays.
        assert not hasattr(tracker, 'true_model_ids')
    for name in ('bayes_greedy', 'fete_fixed'):
        tracker = BeliefTracker(4)
        env = BatchMarket([0, 3, 8, 15])
        chosen = []
        for t in range(230):
            actions = tracker.recommend(name)
            chosen.append(actions.copy())
            obs, _ = env.step(actions, rng.random((4, 3)))
            tracker.observe(actions, obs['reports'], obs['owners'])
        chosen = np.stack(chosen)
        if name == 'fete_fixed':
            for block, action in enumerate(PRIMARY_ACTION_INDICES):
                np.testing.assert_array_equal(chosen[10*block:10*(block+1)],
                                              np.full((10, 4), action))
            np.testing.assert_array_equal(chosen[40:], np.tile(chosen[40], (190, 1)))
        else:
            for start in range(0, 200, 10):
                np.testing.assert_array_equal(chosen[start:start+10], np.tile(chosen[start], (10, 1)))
            np.testing.assert_array_equal(chosen[200:], np.tile(chosen[199], (30, 1)))
    print('PASS: exact finite-catalogue posterior replay; true-candidate state consistency; '
          'initial cash-flow prediction; read-only hypothetical prediction; '
          'public-only tracker API; 10-round control; 40-round FETE switch; 200-round hold.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.parse_args()
    self_test()
