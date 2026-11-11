"""Validation-selected, type-blind open-loop threshold schedules.

Uses the SAME BatchMarket and time-indexed (world,3) public random streams as
run_threshold_experiment.run_episodes. This module does not change the root
training script and its CLI only runs validation, never the sealed test split.

Schedules use canonical action IDs 0,1,2,3 (rho=0,.5,2/3,1). Actions are held
for ten market rounds. The last control action is held for 200 further rounds.
Full alternating periods 20/40/100 each have two equal phases. A full period of
10 cannot contain two different actions held for >=10 rounds, and is therefore
economically a fixed schedule already included in the candidate set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from steering_env import BatchMarket, PRIMARY_ACTION_INDICES, RHO_VALUES
from run_threshold_experiment import (
    BLOCK, CONTROL, HOLD, HORIZON, MACROS, CONTROL_MACROS, METRICS, OUT,
    run_episodes, save_result,
)

VALIDATION_STREAM = 300017
SWITCH_TIMES = (0, 20, 40, 80, 120, 160, 200)
FULL_PERIODS = (10, 20, 40, 100)


def candidate_schedules():
    """Deduplicate identical 200-round control paths; preserve all origins."""
    unique = {}

    def add(actions, origin):
        key = tuple(map(int, actions))
        if len(key) != CONTROL_MACROS or any(a not in range(4) for a in key):
            raise ValueError('invalid macro-action schedule')
        if key not in unique:
            unique[key] = {'actions': list(key), 'origins': []}
        unique[key]['origins'].append(origin)

    for a in range(4):
        add([a] * CONTROL_MACROS, {'kind': 'fixed', 'action': a})
    for a in range(4):
        for b in range(4):
            for switch in SWITCH_TIMES:
                add([a if t * BLOCK < switch else b
                     for t in range(CONTROL_MACROS)],
                    {'kind': 'single_switch', 'first': a, 'second': b,
                     'switch_market_round': switch})
    for a in range(4):
        for b in range(4):
            for period in FULL_PERIODS:
                if a == b:
                    add([a] * CONTROL_MACROS,
                        {'kind': 'cycle', 'first': a, 'second': b,
                         'full_period_market_rounds': period})
                elif period >= 2 * BLOCK:
                    half = period // 2
                    assert period % (2 * BLOCK) == 0
                    add([a if ((t * BLOCK) // half) % 2 == 0 else b
                         for t in range(CONTROL_MACROS)],
                        {'kind': 'cycle', 'first': a, 'second': b,
                         'full_period_market_rounds': period,
                         'phase_duration_market_rounds': half})
    candidates = []
    for i, (_, row) in enumerate(unique.items()):
        row['id'] = f'schedule_{i:03d}'
        row['rho_actions'] = [float(RHO_VALUES[PRIMARY_ACTION_INDICES[a]])
                              for a in row['actions']]
        candidates.append(row)
    return candidates


def run_schedule_batch(schedules, model_ids, stream_seed,
                       learner_mode='canonical', collect_paths=False):
    """Evaluate schedules in one vector batch, reusing each world's RNG path.

    This function receives model IDs only to construct evaluation environments.
    The action sequence is globally precommitted and never branches on IDs or
    observations. The schedule dimension is outside the model/world dimension.
    """
    schedule_array = np.asarray(schedules, dtype=np.int64)
    if schedule_array.ndim != 2 or schedule_array.shape[1] != CONTROL_MACROS:
        raise ValueError('schedules need shape (S,20)')
    if np.any((schedule_array < 0) | (schedule_array > 3)):
        raise ValueError('schedule actions must use canonical IDs 0..3')
    model_ids = np.asarray(model_ids, dtype=np.int64)
    count, worlds = len(schedules), len(model_ids)
    env = BatchMarket(np.tile(model_ids, count), learner_mode=learner_mode)
    rng = np.random.default_rng(stream_seed)
    totals = np.zeros((count, worlds, len(METRICS)), dtype=np.float64)
    phase = np.zeros((count, worlds, 2, 2), dtype=np.float64)
    paths = (np.zeros((count, MACROS, worlds, 2), dtype=np.float32)
             if collect_paths else None)
    for macro in range(MACROS):
        a = schedule_array[:, min(macro, CONTROL_MACROS - 1)]
        env_a = np.repeat(PRIMARY_ACTION_INDICES[a], worlds)
        block_r = np.zeros((count, worlds), dtype=np.float64)
        block_q = np.zeros((count, worlds), dtype=np.float64)
        for _ in range(BLOCK):
            # Generate exactly (worlds,3), as the root runner does, then repeat
            # it for schedules. Generating (count*worlds,3) would break pairing.
            u = np.tile(rng.random((worlds, 3)), (count, 1))
            observation, metrics = env.step(env_a, u)
            for k, name in enumerate(METRICS):
                value = observation[name] if name in observation else metrics[name]
                totals[:, :, k] += value.reshape(count, worlds)
            block_r += observation['revenue'].reshape(count, worlds)
            block_q += observation['volume'].reshape(count, worlds)
        phase[:, :, int(macro >= CONTROL_MACROS), 0] += block_r
        phase[:, :, int(macro >= CONTROL_MACROS), 1] += block_q
        if paths is not None:
            paths[:, macro, :, 0] = block_r / BLOCK
            paths[:, macro, :, 1] = block_q / BLOCK
    return {'metrics': totals, 'phase_totals': phase, 'trajectories': paths}


def run_schedule(actions, model_ids, stream_seed, learner_mode='canonical'):
    """Root-compatible result for a frozen schedule, usable in later audits."""
    rows = run_schedule_batch([actions], model_ids, stream_seed,
                              learner_mode, collect_paths=True)
    macro_actions = np.asarray(list(actions) + [int(actions[-1])] * (HOLD // BLOCK))
    return {
        'models': np.asarray(model_ids, dtype=int),
        'totals': {k: rows['metrics'][0, :, j] for j, k in enumerate(METRICS)},
        'phase_totals': rows['phase_totals'][0],
        'trajectories': rows['trajectories'][0],
        'actions': np.repeat(macro_actions[:, None], len(model_ids), axis=1).astype(np.int8),
    }


def self_test():
    ids = np.repeat(np.arange(16), 2)
    cases = [[a] * CONTROL_MACROS for a in range(4)]
    all_batched = run_schedule_batch(cases, ids, 19781, collect_paths=True)
    comparisons = 0
    for a in range(4):
        reference = run_episodes('fixed', ids, 19781, fixed_action=a)
        expected = np.stack([reference['totals'][k] for k in METRICS], axis=-1)
        np.testing.assert_array_equal(all_batched['metrics'][a], expected)
        np.testing.assert_array_equal(all_batched['phase_totals'][a], reference['phase_totals'])
        np.testing.assert_array_equal(all_batched['trajectories'][a], reference['trajectories'])
        comparisons += expected.size
    cycling = ([0, 3] * 10)
    switched = ([1] * 8 + [2] * 12)
    pair = run_schedule_batch([cycling, switched], ids, 91571, collect_paths=True)
    for i, schedule in enumerate((cycling, switched)):
        solo = run_schedule(schedule, ids, 91571)
        np.testing.assert_array_equal(pair['metrics'][i],
            np.stack([solo['totals'][k] for k in METRICS], axis=-1))
        np.testing.assert_array_equal(pair['trajectories'][i], solo['trajectories'])
        assert np.all(solo['actions'][CONTROL_MACROS:] == schedule[-1])
    rows = candidate_schedules()
    assert len({tuple(r['actions']) for r in rows}) == len(rows)
    assert all(r['actions'] in [s['actions'] for s in rows] for r in rows[:4])
    print(json.dumps({'self_test': 'passed', 'fixed_metric_values_compared': comparisons,
                      'candidate_count': len(rows),
                      'checks': ['root fixed replay', 'batch chunk invariance',
                                 'common streams', 'hold phase', 'candidate dedup']}), flush=True)


def select_validation(worlds=100, stream_seed=VALIDATION_STREAM, chunk=8):
    if worlds != 100 or stream_seed != VALIDATION_STREAM:
        raise ValueError('Frozen validation requires 100 worlds/model and stream 300017')
    rows = candidate_schedules()
    ids = np.repeat(np.arange(16), worlds)
    all_metrics = np.empty((len(rows), len(ids), len(METRICS)), dtype=np.float64)
    all_phase = np.empty((len(rows), len(ids), 2, 2), dtype=np.float64)
    start = time.perf_counter()
    for lo in range(0, len(rows), chunk):
        part = rows[lo:lo + chunk]
        output = run_schedule_batch([r['actions'] for r in part], ids, stream_seed)
        all_metrics[lo:lo + len(part)] = output['metrics']
        all_phase[lo:lo + len(part)] = output['phase_totals']
        print(json.dumps({'validation_schedules_completed': lo + len(part),
                          'total': len(rows), 'elapsed_seconds': time.perf_counter() - start}),
              flush=True)
    revenue_index, volume_index = METRICS.index('revenue'), METRICS.index('volume')
    means = all_metrics.mean(axis=1)
    original_index = next(i for i, row in enumerate(rows) if row['actions'] == [0] * CONTROL_MACROS)
    q0 = means[original_index, volume_index]
    target = .95 * q0
    feasible = means[:, volume_index] >= target
    eligible = np.flatnonzero(feasible)
    # Revenue first, then volume, then deterministic enumeration order.
    best = max(eligible, key=lambda i: (means[i, revenue_index], means[i, volume_index], -int(i)))
    unrestricted = max(range(len(rows)), key=lambda i: (means[i, revenue_index], means[i, volume_index], -i))
    manifest = []
    for i, row in enumerate(rows):
        manifest.append({**row, 'validation_revenue': float(means[i, revenue_index]),
                         'validation_volume': float(means[i, volume_index]),
                         'validation_feasible': bool(feasible[i])})
    data_dir = OUT / 'data' / 'validation'
    data_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = OUT / 'schedule_candidates_validation.npz'
    np.savez_compressed(candidates_path, ids=np.array([r['id'] for r in rows]), models=ids,
                        metric_names=np.array(METRICS), metrics=all_metrics,
                        phase_totals=all_phase,
                        control_actions=np.array([r['actions'] for r in rows], dtype=np.int8))
    (OUT / 'schedule_candidates_validation.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    selected = run_schedule(rows[best]['actions'], ids, stream_seed)
    np.testing.assert_array_equal(all_metrics[best],
        np.stack([selected['totals'][k] for k in METRICS], axis=-1))
    save_result(data_dir / 'schedule.npz', selected)
    source_hashes = {}
    for filename in ('schedule_baselines.py', 'steering_env.py', 'run_threshold_experiment.py'):
        source_hashes[filename] = hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
    selection = {
        'status': 'validation_selected_test_not_run', 'split': 'validation',
        'worlds_per_model': worlds, 'stream_seed': stream_seed,
        'model_prior': 'uniform_over_fixed_16_catalogue',
        'observations_used_by_schedule': 'time_only_no_type_or_response',
        'learner_mode': 'canonical', 'control_rounds': CONTROL, 'hold_rounds': HOLD,
        'macro_rounds': BLOCK, 'total_unique_candidates': len(rows),
        'service_selection': 'mean_Q >= .95 * mean_Q_rho0_on_same_validation_worlds',
        'original_volume': float(q0), 'target_volume': float(target),
        'selected_index': int(best), 'selected': manifest[best],
        'unconstrained_best_index': int(unrestricted), 'unconstrained_best': manifest[unrestricted],
        'fixed_schedule_indices': list(range(4)),
        'cycle_convention': 'full periods 20/40/100, equal phases >=10; period10 collapses to fixed',
        'validation_candidate_market_steps': len(rows) * len(ids) * HORIZON,
        'elapsed_seconds': time.perf_counter() - start,
        'code_sha256': source_hashes,
    }
    selection_path = OUT / 'schedule_selection.json'
    selection_path.write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print(json.dumps({'selection_path': str(selection_path),
                      'selected': selection['selected'],
                      'target_volume': float(target), 'test_run': False}), flush=True)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['self-test', 'validation'])
    parser.add_argument('--chunk', type=int, default=8)
    args = parser.parse_args()
    if args.chunk < 1:
        raise ValueError('chunk must be positive')
    if args.command == 'self-test':
        self_test()
    else:
        select_validation(chunk=args.chunk)


if __name__ == '__main__':
    main()
