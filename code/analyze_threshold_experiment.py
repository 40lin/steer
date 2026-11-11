"""Frozen, paired crossed-bootstrap analysis for the canonical threshold test.

No test data are read by --self-test or --freeze. The 16 catalogue strata are a
fixed, equally weighted population; they are NEVER resampled as random types.
Training seeds and within-type test worlds are crossed: one world resample per
type is shared by ALL resampled training seeds and ALL methods. Fixed baselines
have no training-seed uncertainty. All ten trained seeds are mandatory.

Primary endpoint: belief minus validation-selected feasible global fixed rule.
Primary service condition: belief volume minus .95 * original-rule volume.
Success requires a strictly positive one-sided 95% revenue lower bound and a
nonnegative one-sided 95% service-slack lower bound. Other comparisons, groups,
16-cell results and holding-period effects are explicitly explanatory.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments' / 'dynamic_threshold'
DEFAULT_SEED = 48151623
DEFAULT_REPLICATES = 10_000
TRAINING_SEEDS = tuple(range(10))
TYPE_COUNT = 16
HORIZON = 400
BLOCK = 10
SERVICE_FRACTION = .95
BASE_METRICS = (
    'revenue', 'volume', 'completed_jobs', 'completed_value', 'total_user_utility',
    'active_buyers', 'underreport_count_active', 'overreport_count',
)
METRICS = BASE_METRICS + (
    'control_revenue', 'control_volume', 'hold_revenue', 'hold_volume', 'exit_count',
)
TRAINED_METHODS = ('belief', 'history', 'openloop')
FIXED_METHODS = tuple(f'fixed_{i}' for i in range(4))
OTHER_METHODS = ('ucb', 'bayes_greedy', 'fete_fixed', 'oracle_fixed',
                 'oracle_unconstrained', 'schedule_best')
METHODS = FIXED_METHODS + TRAINED_METHODS + OTHER_METHODS
GROUPS = {
    'all': np.arange(16),
    'independent': np.array([i for i in range(16) if (i // 2) % 2 == 0]),
    'coalition': np.array([i for i in range(16) if (i // 2) % 2 == 1]),
}
METHOD_NOTES = {
    'belief': 'Primary: recurrent PPO with public-history model posterior; all 10 seeds.',
    'history': 'Explanatory: direct public-history recurrent PPO; all 10 seeds.',
    'openloop': 'Explanatory: time/own-action control without buyer-response observations; all 10 seeds.',
    'oracle_fixed': 'Privileged fixed rule selected per type subject to per-type service; not a dynamic upper bound.',
    'oracle_unconstrained': 'Privileged per-type validation revenue maximum, without service constraint; diagnostic, not deployable.',
    'fete_fixed': 'Simple fixed probing/posterior choice baseline; not trained ICLR FETE.',
    'schedule_best': 'Strong predetermined schedule selected on validation, without test tuning.',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, ensure_ascii=False,
                               allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0])
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean_json(rows))


def expected_files(data_dir):
    files = {}
    for method in METHODS:
        names = ([f'{method}_{seed}.npz' for seed in TRAINING_SEEDS]
                 if method in TRAINED_METHODS else [f'{method}.npz'])
        files[method] = [Path(data_dir) / name for name in names]
    return files


def freeze(experiment=EXPERIMENT, replicates=DEFAULT_REPLICATES, seed=DEFAULT_SEED):
    """Record analysis choices without reading or generating ANY test result."""
    experiment = Path(experiment)
    document = {
        'status': 'analysis_script_frozen_before_test_analysis',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_script_sha256': sha256(__file__),
        'bootstrap_replicates': replicates, 'analysis_rng_seed': seed,
        'mandatory_training_seeds': TRAINING_SEEDS, 'mandatory_methods': METHODS,
        'strata': '16 fixed catalogue types, equally weighted; no type resampling',
        'bootstrap': 'paired crossed seed x within-type world resampling, shared across all methods',
        'primary_comparison': 'belief minus validation-selected feasible global fixed',
        'primary_service': 'belief volume minus .95 original fixed_0 volume',
        'success_rule': 'revenue one-sided 95% lower > 0 AND service lower >= 0',
        'other_comparisons': 'explanatory effect estimates; no multiplicity-unadjusted confirmatory claims',
        'holding_period': 'rounds 201..400, included in whole-episode objective',
        'test_results_read_by_freeze': False,
    }
    for key, name in [('protocol_sha256', 'protocol_audit.md'),
                      ('validation_selection_sha256', 'validation_selection.json')]:
        file = experiment / name
        document[key] = sha256(file) if file.exists() else None
    destination = experiment / 'analysis_script_freeze.json'
    if destination.exists():
        previous = json.loads(destination.read_text(encoding='utf-8'))
        if previous['analysis_script_sha256'] != document['analysis_script_sha256']:
            raise RuntimeError('Existing freeze has another script hash; append a documented amendment instead of overwriting.')
        print(f'Already frozen: {destination}')
        return previous
    write_json(destination, document)
    print(f'Frozen analysis script: {document["analysis_script_sha256"]}')
    return document


def load_npz(path, reference_models=None):
    with np.load(path, allow_pickle=False) as data:
        names = tuple(str(x) for x in data['metric_names'])
        if names != BASE_METRICS:
            raise ValueError(f'{path}: unexpected metric names/order: {names}')
        models = data['models'].astype(np.int64)
        metrics = data['metrics'].astype(np.float64)
        phase = data['phase_totals'].astype(np.float64)
        trajectories = data['trajectories'].astype(np.float64)
        actions = data['actions']
    if reference_models is not None and not np.array_equal(models, reference_models):
        raise ValueError(f'{path}: worlds/type ordering differs; paired analysis is invalid')
    if set(models.tolist()) != set(range(16)):
        raise ValueError(f'{path}: primary analysis requires all 16 types; use a separate stress-test analysis')
    counts = np.bincount(models, minlength=16)
    if len(set(counts.tolist())) != 1 or counts[0] == 0:
        raise ValueError(f'{path}: unequal or empty within-type world counts')
    worlds = int(counts[0])
    if not np.array_equal(models, np.repeat(np.arange(16), worlds)):
        raise ValueError(f'{path}: expected canonical sorted type/world order')
    size = len(models)
    if metrics.shape != (size, len(BASE_METRICS)) or phase.shape != (size, 2, 2):
        raise ValueError(f'{path}: wrong metric/phase shape')
    if trajectories.shape != (40, size, 2) or actions.shape != (40, size):
        raise ValueError(f'{path}: wrong horizon; primary test is 20 control + 20 holding blocks')
    if not all(np.isfinite(x).all() for x in (metrics, phase, trajectories)):
        raise ValueError(f'{path}: nonfinite metric')
    if not np.all((actions >= 0) & (actions <= 3)):
        raise ValueError(f'{path}: invalid local policy actions')
    if not np.array_equal(actions[20:], np.broadcast_to(actions[19], actions[20:].shape)):
        raise ValueError(f'{path}: holding period did not freeze the last control action')
    np.testing.assert_allclose(phase.sum(axis=1), metrics[:, :2], rtol=1e-10, atol=2e-8,
                               err_msg=f'{path}: phase cash-flow ledger mismatch')
    # Trajectories are stored in float32; allow their known quantization error.
    np.testing.assert_allclose(trajectories.sum(axis=0)*BLOCK, metrics[:, :2],
                               rtol=2e-6, atol=2e-3,
                               err_msg=f'{path}: block cash-flow ledger mismatch')
    if np.any(metrics[:, 0] < -1e-9) or np.any(metrics[:, 0] > 2000+1e-8):
        raise ValueError(f'{path}: impossible no-rebate revenue')
    if np.any(metrics[:, 1] < -1e-9) or np.any(metrics[:, 1] > 2000+1e-8):
        raise ValueError(f'{path}: impossible sold volume')
    if np.any(metrics[:, 5] < 0) or np.any(metrics[:, 5] > 800):
        raise ValueError(f'{path}: impossible active buyer count')
    if np.any(metrics[:, 6] + metrics[:, 7] > metrics[:, 5]+1e-8):
        raise ValueError(f'{path}: reporting counts exceed active buyers')
    np.testing.assert_allclose(metrics[:, 0]+metrics[:, 4], metrics[:, 3],
                               rtol=1e-10, atol=2e-8,
                               err_msg=f'{path}: buyer utility plus payment != completed value')
    augmented = np.column_stack((metrics, phase[:, 0, 0], phase[:, 0, 1],
                                  phase[:, 1, 0], phase[:, 1, 1], 800-metrics[:, 5]))
    return augmented.reshape(16, worlds, len(METRICS)), models


def load_all(data_dir):
    paths = expected_files(data_dir)
    missing = [str(p) for files in paths.values() for p in files if not p.exists()]
    if missing:
        return None, {'status': 'pending_complete_test_data', 'missing_files': missing}
    arrays, hashes = {}, {}
    reference_models = None
    for method, files in paths.items():
        samples = []
        for path in files:
            sample, models = load_npz(path, reference_models)
            if reference_models is None:
                reference_models = models
            samples.append(sample)
            hashes[path.name] = sha256(path)
        arrays[method] = np.stack(samples)  # [training seed or 1, type, world, metric]
    return arrays, {
        'status': 'complete', 'worlds_per_type': arrays['fixed_0'].shape[2],
        'input_file_sha256': hashes,
        'pairing_limit': 'NPZ world ordering checked; external frozen run manifest must attest common random streams.',
    }


def crossed_bootstrap(arrays, replicates=DEFAULT_REPLICATES, seed=DEFAULT_SEED,
                      chunk_size=128):
    """Return [replicate,type,metric] means for each method.

    Separate multinomial counts encode bootstrap samples. World counts are drawn
    ONCE per replicate/type and shared across selected seeds/methods. A fixed
    method's single seed row remains single; only its test worlds are resampled.
    """
    if not arrays:
        raise ValueError('no arrays')
    dimensions = next(iter(arrays.values())).shape
    _, types, worlds, metrics = dimensions
    seeds = max(x.shape[0] for x in arrays.values())
    for method, x in arrays.items():
        if x.ndim != 4 or x.shape[1:] != (types, worlds, metrics):
            raise ValueError(f'{method}: inconsistent crossed dimensions')
        if x.shape[0] not in (1, seeds):
            raise ValueError(f'{method}: do not silently discard unmatched training seeds')
    rng = np.random.default_rng(seed)
    result = {name: np.empty((replicates, types, metrics), dtype=np.float64)
              for name in arrays}
    p_seed = np.full(seeds, 1/seeds)
    p_world = np.full(worlds, 1/worlds)
    for start in range(0, replicates, chunk_size):
        stop = min(start+chunk_size, replicates)
        length = stop-start
        seed_weights = rng.multinomial(seeds, p_seed, size=length) / seeds
        world_weights = rng.multinomial(worlds, p_world, size=(length, types)) / worlds
        for name, x in arrays.items():
            if x.shape[0] == 1:
                means = np.einsum('twm,ktw->ktm', x[0], world_weights, optimize=True)
            else:
                seed_mean = np.tensordot(seed_weights, x, axes=(1, 0))
                means = np.einsum('ktwm,ktw->ktm', seed_mean, world_weights, optimize=True)
            result[name][start:stop] = means
    return result


def estimate(point, samples):
    samples = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(samples).all() or not np.isfinite(point):
        return {'estimate': None, 'ci95_low': None, 'ci95_high': None,
                'one_sided95_low': None, 'bootstrap_sd': None}
    low, high = np.quantile(samples, [.025, .975])
    return {'estimate': float(point), 'ci95_low': float(low), 'ci95_high': float(high),
            'one_sided95_low': float(np.quantile(samples, .05)),
            'bootstrap_sd': float(np.std(samples, ddof=1))}


def group_statistics(arrays, boots):
    points = {name: x.mean(axis=(0, 2)) for name, x in arrays.items()}  # [type,metric]
    group_points = {name: {group: point[indices].mean(axis=0)
                          for group, indices in GROUPS.items()} for name, point in points.items()}
    group_boots = {name: {group: sample[:, indices].mean(axis=1)
                         for group, indices in GROUPS.items()} for name, sample in boots.items()}
    return points, group_points, group_boots


def compare(method, reference, group, group_points, group_boots):
    p, q = group_points[method][group], group_points[reference][group]
    b, c = group_boots[method][group], group_boots[reference][group]
    results = {
        'revenue_difference': estimate(p[0]-q[0], b[:, 0]-c[:, 0]),
        'revenue_gain_percent': estimate(100*(p[0]-q[0])/q[0],
                                         100*(b[:, 0]-c[:, 0])/c[:, 0]),
        'volume_difference': estimate(p[1]-q[1], b[:, 1]-c[:, 1]),
        'hold_revenue_difference': estimate(p[10]-q[10], b[:, 10]-c[:, 10]),
    }
    return results


def analyze(experiment=EXPERIMENT, split='test', replicates=DEFAULT_REPLICATES,
            seed=DEFAULT_SEED):
    experiment = Path(experiment)
    out = experiment / 'analysis' / split
    out.mkdir(parents=True, exist_ok=True)
    frozen_path = experiment / 'analysis_script_freeze.json'
    if not frozen_path.exists():
        raise RuntimeError('Run --freeze before generating/opening main test results.')
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    if frozen['analysis_script_sha256'] != sha256(__file__):
        raise RuntimeError('Analysis script changed after freeze; document amendment before analysis.')
    if frozen['bootstrap_replicates'] != replicates or frozen['analysis_rng_seed'] != seed:
        raise RuntimeError('Bootstrap settings differ from the frozen analysis.')
    arrays, integrity = load_all(experiment / 'data' / split)
    if arrays is None:
        write_json(out/'status.json', integrity)
        print(json.dumps({'status': integrity['status'], 'missing_count': len(integrity['missing_files'])}))
        return 2
    selection_path = experiment/'validation_selection.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    # Baseline choice comes from validation; never choose the largest test mean.
    fixed = f'fixed_{int(selection["best_fixed_action"])}'
    if fixed not in FIXED_METHODS:
        raise ValueError('invalid validation-selected fixed action')
    if frozen.get('validation_selection_sha256') != sha256(selection_path):
        raise RuntimeError('Validation selection changed after analysis freeze.')
    boots = crossed_bootstrap(arrays, replicates, seed)
    points, group_points, group_boots = group_statistics(arrays, boots)
    method_rows, type_rows, seed_rows, comparison_rows, paper_rows = [], [], [], [], []
    summaries = {}
    for method in METHODS:
        summaries[method] = {'note': METHOD_NOTES.get(method, ''), 'groups': {}}
        for group in GROUPS:
            group_summary = {}
            for index, metric in enumerate(METRICS):
                stat = estimate(group_points[method][group][index], group_boots[method][group][:, index])
                group_summary[metric] = stat
                method_rows.append({'method': method, 'group': group, 'metric': metric, **stat})
            p = group_points[method][group]
            b = group_boots[method][group]
            # Rates are ratios of equal-weight aggregate counts, not averages of
            # arbitrary per-world ratios. Exits have a fixed 800-arrival divisor.
            for name, numerator, denominator in (
                ('underreport_rate_among_active', p[6], p[5]),
                ('overreport_rate_among_active', p[7], p[5]),
                ('exit_rate_among_arrivals', p[12], 800.),
            ):
                column = {'underreport_rate_among_active': 6,
                          'overreport_rate_among_active': 7,
                          'exit_rate_among_arrivals': 12}[name]
                divisor = b[:, 5] if column != 12 else 800.
                with np.errstate(divide='ignore', invalid='ignore'):
                    stat = estimate(numerator/denominator, b[:, column]/divisor)
                group_summary[name] = stat
                method_rows.append({'method': method, 'group': group, 'metric': name, **stat})
            service = estimate(p[1]-SERVICE_FRACTION*group_points['fixed_0'][group][1],
                               b[:, 1]-SERVICE_FRACTION*group_boots['fixed_0'][group][:, 1])
            group_summary['service_slack_vs_original'] = service
            method_rows.append({'method': method, 'group': group,
                                'metric': 'service_slack_vs_original', **service})
            summaries[method]['groups'][group] = group_summary
        for model in range(16):
            for index, metric in enumerate(METRICS):
                stat = estimate(points[method][model, index], boots[method][:, model, index])
                type_rows.append({'method': method, 'model_id': model,
                    'market': 'ABCD'[model//4], 'coalition': (model//2)%2,
                    'learning_rate': [.05, .2][model%2], 'metric': metric, **stat})
        seed_ids = TRAINING_SEEDS if method in TRAINED_METHODS else ('not_trained',)
        for index, seed_id in enumerate(seed_ids):
            seed_type_means = arrays[method][index].mean(axis=1)
            for group, indices in GROUPS.items():
                means = seed_type_means[indices].mean(axis=0)
                row = {'method': method, 'training_seed': seed_id, 'group': group,
                       **{metric: float(means[i]) for i, metric in enumerate(METRICS)}}
                row['revenue_difference_vs_selected_fixed'] = float(means[0]-group_points[fixed][group][0])
                row['service_slack_vs_original'] = float(means[1]-.95*group_points['fixed_0'][group][1])
                seed_rows.append(row)
        for group in GROUPS:
            for metric, stat in compare(method, fixed, group, group_points, group_boots).items():
                comparison_rows.append({'method': method, 'reference': fixed, 'group': group,
                                        'metric': metric, 'status': ('primary' if method == 'belief' and group == 'all'
                                                                    and metric == 'revenue_difference' else 'explanatory'), **stat})
        summary = summaries[method]['groups']['all']
        contrast = compare(method, fixed, 'all', group_points, group_boots)
        paper_rows.append({'method': method,
            'revenue_mean': summary['revenue']['estimate'], 'revenue_ci_low': summary['revenue']['ci95_low'],
            'revenue_ci_high': summary['revenue']['ci95_high'],
            'volume_mean': summary['volume']['estimate'], 'volume_ci_low': summary['volume']['ci95_low'],
            'volume_ci_high': summary['volume']['ci95_high'],
            'delta_revenue_vs_fixed': contrast['revenue_difference']['estimate'],
            'delta_ci_low': contrast['revenue_difference']['ci95_low'],
            'delta_ci_high': contrast['revenue_difference']['ci95_high'],
            'service_slack': summary['service_slack_vs_original']['estimate'],
            'service_slack_lower95': summary['service_slack_vs_original']['one_sided95_low'],
            'hold_revenue': summary['hold_revenue']['estimate'], 'note': METHOD_NOTES.get(method, '')})
    # Additional paired contrasts are explanatory; no unadjusted significance claims.
    for reference in ('history', 'openloop', 'schedule_best', 'bayes_greedy', 'fete_fixed',
                      'oracle_fixed', 'oracle_unconstrained'):
        for group in GROUPS:
            for metric, stat in compare('belief', reference, group, group_points, group_boots).items():
                comparison_rows.append({'method': 'belief', 'reference': reference, 'group': group,
                                       'metric': metric, 'status': 'explanatory', **stat})
    primary = {
        'method': 'belief', 'reference': fixed,
        'revenue': compare('belief', fixed, 'all', group_points, group_boots)['revenue_difference'],
        'revenue_gain_percent': compare('belief', fixed, 'all', group_points, group_boots)['revenue_gain_percent'],
        'service_slack': summaries['belief']['groups']['all']['service_slack_vs_original'],
        'all_ten_seed_gains': [row['revenue_difference_vs_selected_fixed'] for row in seed_rows
                               if row['method'] == 'belief' and row['group'] == 'all'],
        'coalition_subgroup_is_explanatory': True,
    }
    primary['revenue_criterion_passes'] = primary['revenue']['one_sided95_low'] > 0
    primary['service_criterion_passes'] = primary['service_slack']['one_sided95_low'] >= 0
    primary['joint_confirmatory_success'] = primary['revenue_criterion_passes'] and primary['service_criterion_passes']
    metadata = {
        'status': 'analyzed_all_frozen_methods_and_seeds', 'split': split,
        'analysis_script_sha256': sha256(__file__), 'freeze': frozen,
        'bootstrap_replicates': replicates, 'analysis_rng_seed': seed,
        'training_seed_count': 10, 'fixed_type_count': 16, **integrity,
        'interpretation': 'Model-in-catalogue no-rebate revenue experiment; not arbitrary-collusion robustness.',
        'oracle_warning': METHOD_NOTES['oracle_fixed']+' '+METHOD_NOTES['oracle_unconstrained'],
    }
    for name, rows in [('method_summary.csv', method_rows), ('type_results.csv', type_rows),
                       ('seed_results.csv', seed_rows), ('comparisons.csv', comparison_rows),
                       ('paper_table.csv', paper_rows)]:
        write_csv(out/name, rows)
    write_json(out/'primary_summary.json', primary)
    write_json(out/'results_summary.json', {'metadata': metadata, 'primary': primary, 'methods': summaries})
    write_json(out/'status.json', metadata)
    np.savez_compressed(out/'primary_bootstrap.npz',
        revenue_difference=group_boots['belief']['all'][:, 0]-group_boots[fixed]['all'][:, 0],
        service_slack=group_boots['belief']['all'][:, 1]-.95*group_boots['fixed_0']['all'][:, 1],
        analysis_rng_seed=np.array(seed), validation_selected_fixed=np.array(fixed))
    print(json.dumps(clean_json(primary), ensure_ascii=False, indent=2))
    return 0


def self_test():
    """Synthetic checks only: never reads main validation/test values."""
    seeds, types, worlds = 10, 16, 20
    rng = np.random.default_rng(661)
    world = rng.normal(size=(types, worlds))
    offsets = np.linspace(-2, 3, seeds)
    fixed = world[None, :, :, None]
    learned = fixed + offsets[:, None, None, None]
    arrays = {'fixed': fixed, 'learned': learned, 'shifted': fixed+2,
              'type_only': np.broadcast_to(np.arange(types)[None, :, None, None], (1, types, worlds, 1))}
    bootstrap = crossed_bootstrap(arrays, replicates=10_000, seed=13)
    diff = (bootstrap['shifted']-bootstrap['fixed']).mean(axis=1)[:, 0]
    np.testing.assert_allclose(diff, 2., rtol=0, atol=1e-14)
    # Fixed types stay fixed: deterministic type heterogeneity adds no sampling noise.
    np.testing.assert_allclose(bootstrap['type_only'].mean(axis=1)[:, 0], 7.5, atol=1e-13)
    learned_diff = (bootstrap['learned']-bootstrap['fixed']).mean(axis=1)[:, 0]
    expected_seed_variance = offsets.var(ddof=0)/seeds
    assert abs(learned_diff.var(ddof=1)/expected_seed_variance-1) < .08
    fixed_global = bootstrap['fixed'].mean(axis=1)[:, 0]
    expected_world_variance = world.var(axis=1, ddof=0).sum()/(types**2*worlds)
    assert abs(fixed_global.var(ddof=1)/expected_world_variance-1) < .08
    learned_global = bootstrap['learned'].mean(axis=1)[:, 0]
    expected_both = expected_seed_variance + expected_world_variance
    assert abs(learned_global.var(ddof=1)/expected_both-1) < .08
    # Training uncertainty is shared across types, not averaged away 16 times.
    np.testing.assert_allclose((bootstrap['learned']-bootstrap['fixed'])[:, :, 0],
                               np.broadcast_to(learned_diff[:, None], (10_000, 16)), atol=1e-13)
    reference = crossed_bootstrap(arrays, replicates=128, seed=13)
    for key in arrays:
        np.testing.assert_array_equal(bootstrap[key][:128], reference[key])
    check = estimate(2., np.full(100, 2.))
    assert check['ci95_low'] == check['ci95_high'] == check['one_sided95_low'] == 2.
    assert len(expected_files(Path('/not_read'))) == len(METHODS)
    print('PASS: paired world cancellation; fixed-stratum weighting; seed uncertainty; '
          'world uncertainty; crossed variance; cross-type seed sharing; deterministic bootstrap; '
          'interval calculation. No real test results were opened.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--experiment-dir', type=Path, default=EXPERIMENT)
    parser.add_argument('--split', default='test')
    parser.add_argument('--replicates', type=int, default=DEFAULT_REPLICATES)
    parser.add_argument('--analysis-seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.freeze:
        freeze(args.experiment_dir, args.replicates, args.analysis_seed)
    if args.run:
        return analyze(args.experiment_dir, args.split, args.replicates, args.analysis_seed)
    if not (args.self_test or args.freeze):
        parser.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
