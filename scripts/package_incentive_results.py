"""Publish compact, outcome-independent V4/V5 reference evidence.

Usage: python scripts/package_incentive_results.py --study v4
       python scripts/package_incentive_results.py --study all
       python scripts/package_incentive_results.py --verify --verify-sources

The full local analyses remain the source of record. Selection is by case,
budget and prespecified group identity, never by effect size or significance.
All three phases and every metric of retained groups are kept. Source hashes
identify unmodified local originals; published hashes identify filtered,
path-sanitized, UTF-8/LF artifacts. They are intentionally distinct.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'results/reference/incentives'
STUDIES = {'v4': 'v4_incentive', 'v5': 'v5_conditional_incentive'}
GROUP_FILTER = ('Keep every case/budget group=all, plus bounded budget=50 '
                'family_A..D and eta_0.05/eta_0.2. Keep every method/comparison, metric, buyer/role '
                'and full/control/withdraw phase. No outcome-based filtering.')
TYPE_FILTER = ('Keep case=bounded and budget=50, all 24 model types, all '
               'methods/buyers/metrics and full/control/withdraw phases.')
COPY_ANALYSIS = ('primary.json', 'cost_frontier.csv', 'path_audits.csv')
GROUP_CSV = ('summary.csv', 'comparisons.csv', 'buyer_summary.csv',
             'buyer_comparisons.csv', 'seed_results.csv')
TYPE_CSV = ('type_results.csv', 'buyer_type_results.csv')
METADATA = ('protocol.json', 'selection.json', 'pretest_freeze.json', 'analysis_freeze.json')
COMMON_STAGE = ('protocol.json', 'summary.json', 'stage_summary.csv',
                'pure_nash.csv', 'completion_audit.json')
EXTRA_STAGE = {
    'v4': ('bonus_trade_effects.csv', 'withdrawal_oracles.json',
           'witnesses.json'),
    'v5': ('fullpool_summary.csv', 'minimum_menu_thresholds.csv',
           'overreport_reward_swap_witness.json'),
}
WINDOWS_ABSOLUTE = re.compile(r'[A-Za-z]:[\\/]')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value, *, pretty=False):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False,
                       indent=2 if pretty else None,
                       separators=None if pretty else (',', ':')) + '\n').encode('utf-8')


def keep_group(case, budget, group):
    return group == 'all' or (case == 'bounded' and str(budget) == '50' and
                               group in ('family_A', 'family_B', 'family_C', 'family_D',
                                         'eta_0.05', 'eta_0.2'))


def keep_type(row):
    return row['case'] == 'bounded' and row['budget'] == '50'


def clean_string(value, changes, field):
    """Replace local paths, including historical mojibake workspace prefixes.

    Repository suffixes are recognized independently of the local parent path.
    Unknown standalone local paths retain only a marked basename. No original
    absolute path is copied into the public sanitization log.
    """
    normalized = value.replace('\\', '/')
    absolute = bool(WINDOWS_ABSOLUTE.match(value)) or value.startswith('/') or value.startswith('\\\\')
    if not absolute:
        if WINDOWS_ABSOLUTE.search(value):
            raise ValueError(f'Embedded local path requires explicit sanitization at {field}')
        return value
    normalized = re.sub('/+', '/', normalized)
    local_root = ROOT.as_posix().rstrip('/')
    if normalized.startswith(local_root + '/'):
        cleaned = normalized[len(local_root) + 1:]
        rule = 'replace local checkout prefix with repository-relative suffix'
    elif '/steer/' in normalized:
        cleaned = normalized.split('/steer/', 1)[1]
        rule = 'strip historical local checkout prefix at /steer/; normalize separators'
    else:
        cleaned = '<LOCAL_PATH>/' + normalized.rstrip('/').rsplit('/', 1)[-1]
        rule = 'redact unknown local directory; retain marked basename'
    changes.append({'field': field, 'rule': rule})
    return cleaned


def sanitize(value, changes, field='$'):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            new_key = clean_string(key, changes, field + '/<key>')
            if new_key in result:
                raise ValueError('Path cleaning produced a duplicate JSON key')
            result[new_key] = sanitize(item, changes, field + '/' + new_key)
        return result
    if isinstance(value, list):
        return [sanitize(item, changes, field + '/' + str(i)) for i, item in enumerate(value)]
    if isinstance(value, str):
        return clean_string(value, changes, field)
    return value  # Preserve every number, boolean and null exactly as parsed.


def selected_summary(summary):
    result = {key: value for key, value in summary.items() if key != 'cases'}
    result['cases'] = {}
    before = after = 0
    for case, budgets in summary['cases'].items():
        result['cases'][case] = {}
        for budget, block in budgets.items():
            kept = {key: value for key, value in block.items()
                    if key not in ('methods', 'comparisons')}
            for category in ('methods', 'comparisons'):
                kept[category] = {}
                for method, groups in block[category].items():
                    before += len(groups)
                    chosen = {group: data for group, data in groups.items()
                              if keep_group(case, budget, group)}
                    assert 'all' in chosen
                    if category == 'methods':
                        for phases in chosen.values():
                            assert set(phases) == {'full', 'control', 'withdraw'}
                    else:
                        for data in chosen.values():
                            assert set(data['phases']) == {'full', 'control', 'withdraw'}
                    after += len(chosen)
                    kept[category][method] = chosen
            result['cases'][case][budget] = kept
    return result, {'source_group_records': before, 'published_group_records': after}


def prepare_artifact(source, published_relative, selector='copy'):
    raw = source.read_bytes()
    changes = []
    counts = {}
    if source.suffix == '.json':
        original = json.loads(raw.decode('utf-8-sig'))
        if selector == 'summary':
            selected, counts = selected_summary(original)
            description = GROUP_FILTER
        else:
            selected = original
            description = 'Complete JSON object; only path/encoding/line-ending/whitespace normalization.'
        cleaned = sanitize(selected, changes)
        published = json_bytes(cleaned)
        assert json.loads(published) == cleaned
    elif source.suffix == '.csv':
        reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f'CSV has no header: {source.relative_to(ROOT)}')
        rows = list(reader)
        if selector == 'group':
            retained = [(i, row) for i, row in enumerate(rows)
                        if keep_group(row['case'], row['budget'], row['group'])]
            description = GROUP_FILTER
        elif selector == 'type':
            retained = [(i, row) for i, row in enumerate(rows) if keep_type(row)]
            description = TYPE_FILTER
        else:
            retained = list(enumerate(rows))
            description = 'All CSV rows and columns; only local paths and text formatting normalized.'
        if not retained:
            raise ValueError(f'Publication filter unexpectedly kept zero rows: {source.relative_to(ROOT)}')
        if 'phase' in fieldnames:
            assert {row['phase'] for _, row in retained} == {'full', 'control', 'withdraw'}
        output = io.StringIO(newline='')
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        for i, row in retained:
            # Numeric CSV cells remain original strings; no rounding/parsing.
            writer.writerow({key: clean_string(value, changes, f'row[{i}]/{key}')
                             for key, value in row.items()})
        published = output.getvalue().encode('utf-8')
        counts = {'source_rows': len(rows), 'published_rows': len(retained)}
    else:
        raise ValueError('Only structured JSON/CSV evidence is packaged')
    if WINDOWS_ABSOLUTE.search(published.decode('utf-8')):
        raise ValueError('Publication still contains an absolute Windows path')
    assert b'\r\n' not in published and not published.startswith(b'\xef\xbb\xbf')
    entry = {
        'source_file': source.relative_to(ROOT).as_posix(),
        'published_file': published_relative.as_posix(),
        'source_sha256': digest(raw), 'published_sha256': digest(published),
        'source_bytes': len(raw), 'published_bytes': len(published),
        'filter': description, **counts,
        'numeric_values_changed': False,
        'formatting': {'published_encoding': 'UTF-8 without BOM', 'published_line_endings': 'LF',
                       'source_has_UTF8_BOM': raw.startswith(b'\xef\xbb\xbf'),
                       'source_CRLF_sequences': raw.count(b'\r\n'),
                       'JSON_whitespace': 'compact; object content otherwise retained under declared filter'},
        'path_sanitization_count': len(changes), 'path_sanitization': changes,
    }
    return entry, published


def prepare_study(study):
    local = ROOT / 'experiments' / STUDIES[study]
    completed_files = ['analysis/primary.json', 'analysis/summary.json', 'exact_audit/completion_audit.json']
    if study == 'v5':
        completed_files.append('fullpool_replay/summary.json')
    for relative in completed_files:
        source = local / relative
        if not source.exists() or json.loads(source.read_text(encoding='utf-8-sig')).get('status') != 'complete':
            raise RuntimeError(f'{study} is not complete: {source.relative_to(ROOT)}')
    plan = []

    def add(relative, *, name=None, selector='copy', required=True):
        source = local / relative
        if not source.exists():
            if required:
                raise FileNotFoundError(source)
            return
        published = Path(study) / (name or relative)
        plan.append(prepare_artifact(source, published, selector))

    for name in COPY_ANALYSIS:
        add('analysis/' + name)
    add('analysis/summary.json', name='analysis/summary_selected.json', selector='summary')
    for name in GROUP_CSV:
        add('analysis/' + name, name='analysis/' + name.replace('.csv', '_selected.csv'), selector='group')
    for name in TYPE_CSV:
        add('analysis/' + name, name='analysis/' + name.replace('.csv', '_bounded_b50.csv'), selector='type')
    for name in METADATA:
        add(name)
    for name in COMMON_STAGE + EXTRA_STAGE[study]:
        add('exact_audit/' + name)
    # This small supplementary interpretation may be generated after the
    # frozen stage enumerator; preserve it when present without requiring a
    # manually authored supplement for a clean numerical reproduction.
    add('exact_audit/economic_findings.json', required=False)
    add('environment_design_audit.json', required=False)
    # Future executed-policy full-pooling diagnostics are copied in full when
    # present. The stage full-pooling tables above are already mandatory for V5.
    for directory in ('fullpool_audit', 'fullpool_analysis', 'fullpool_replay'):
        folder = local / directory
        if folder.exists():
            if directory == 'fullpool_replay':
                for name in ('protocol.json', 'summary.json'):
                    add(directory + '/' + name)
                continue
            for source in sorted(folder.glob('*')):
                if source.suffix in ('.json', '.csv') and 'details' not in source.stem:
                    add(source.relative_to(local).as_posix())
    excluded = []
    details = local / 'exact_audit/stage_details.json'
    if details.exists():
        raw = details.read_bytes()
        excluded.append({'source_file': details.relative_to(ROOT).as_posix(),
                         'source_sha256': digest(raw), 'source_bytes': len(raw),
                         'reason': 'Large per-configuration detailed arrays; full summaries, NE tables and witnesses retained.'})
    record = {'status': 'complete_compact_publication', 'source_directory': local.relative_to(ROOT).as_posix(),
              'group_filter': GROUP_FILTER, 'type_filter': TYPE_FILTER,
              'phase_policy': 'Retain full, control and withdraw; do not filter metrics or signs.',
              'artifacts': [entry for entry, _ in plan], 'omitted_large_artifacts': excluded,
              'raw_trajectory_policy': 'NPZ test/validation trajectories and checkpoints remain local and reproducible; '
                                       'their source hashes remain in original analysis/pretest manifests.'}
    return plan, record


def verify(manifest, *, sources=False):
    files = total = 0
    for record in manifest['studies'].values():
        for entry in record['artifacts']:
            target = (DESTINATION / entry['published_file']).resolve()
            if not target.is_relative_to(DESTINATION.resolve()):
                raise ValueError('Manifest publication path leaves its directory')
            raw = target.read_bytes()
            assert digest(raw) == entry['published_sha256'] and len(raw) == entry['published_bytes']
            assert b'\r\n' not in raw and not raw.startswith(b'\xef\xbb\xbf')
            assert not WINDOWS_ABSOLUTE.search(raw.decode('utf8'))
            if sources:
                original = (ROOT / entry['source_file']).read_bytes()
                assert digest(original) == entry['source_sha256'] and len(original) == entry['source_bytes']
            files += 1
            total += len(raw)
    return {'verified_published_files': files, 'published_bytes': total,
            'original_sources_verified': sources}


def package(studies):
    # Prepare every requested study before writing anything; incomplete V5
    # cannot leave a misleading half-published study in the public directory.
    prepared = {study: prepare_study(study) for study in studies}
    manifest_path = DESTINATION / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8')) if manifest_path.exists() else {'studies': {}}
    for study, (plan, record) in prepared.items():
        for entry, raw in plan:
            target = DESTINATION / entry['published_file']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        manifest['studies'][study] = record
    manifest.update({
        'schema': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'compact_reference_publication',
        'source_vs_publication': 'source_sha256/source_bytes describe unmodified local originals; '
                                 'published_sha256/published_bytes describe the selected, sanitized UTF-8/LF copies. '
                                 'These hashes must not be conflated.',
        'selection_policy': 'Identity-based filters fixed in the packaging program; negative and positive results retained equally.',
        'numeric_policy': 'JSON numeric values remain parsed numbers without arithmetic/rounding; CSV numeric cells remain source strings.',
        'packaging_script': Path(__file__).relative_to(ROOT).as_posix(),
        'published_code_sha256': {p.relative_to(ROOT).as_posix(): digest(p.read_bytes())
                                  for folder in ('code', 'scripts')
                                  for p in sorted((ROOT / folder).glob('*.py'))},
        'published_code_scope': 'Current publication-time Python source provenance, including shared original logic. '
                                 'This is not a retrospective pretest dependency lock; use each original freeze for that claim.',
    })
    manifest_path.write_bytes(json_bytes(manifest, pretty=True))
    print(json.dumps(verify(manifest, sources=True), ensure_ascii=False))


def self_test():
    assert keep_group('bounded', 50, 'family_D')
    assert not keep_group('bounded', 25, 'family_D')
    assert keep_group('exact_transferable', 50, 'all')
    assert keep_group('bounded', 50, 'eta_0.05')
    assert keep_type({'case': 'bounded', 'budget': '50'})
    changes = []
    assert clean_string(r'D:\old_workspace\steer\experiments\v4_incentive\test\a.npz', changes, 'file') == 'experiments/v4_incentive/test/a.npz'
    assert clean_string(r'D:\\old_workspace\\steer\\experiments\\a.npz', changes, 'file') == 'experiments/a.npz'
    obj = {'x': -0.12345678901234567, 'y': 1e-12, 'z': 0, 'b': False, 'n': None}
    assert json.loads(json_bytes(sanitize(obj, [], '$'))) == obj
    print('PASS: sign-independent case/budget/group filters, all-case totals, local-path cleanup and exact JSON numeric round trip.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('v4', 'v5', 'all'), default='all')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--verify-sources', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.verify:
        loaded = json.loads((DESTINATION / 'manifest.json').read_text(encoding='utf8'))
        print(json.dumps(verify(loaded, sources=args.verify_sources)))
    else:
        package(tuple(STUDIES) if args.study == 'all' else (args.study,))
