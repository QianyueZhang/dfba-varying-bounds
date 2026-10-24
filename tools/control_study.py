"""Read-only record checks and isolated workspace preparation for control studies."""
from pathlib import Path
from collections import defaultdict
import argparse
import ast
import hashlib
import json
import math
import os
import shutil
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def integrity(root):
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    errors = []
    for item in manifest['files']:
        path = root / item['release']
        if not path.is_file() or sha(path) != item['sha256']:
            errors.append('Source/result identity differs: ' + item['release'])
        elif path.suffix == '.py':
            ast.parse(path.read_text())
    lock_path = root / ('audit/design_lock.json' if root.name == 'terminal_recovery' else 'LOCK.json')
    locks = json.loads(lock_path.read_text())
    for name, expected in locks.get('hashes', locks).items():
        if not (root / name).is_file() or sha(root / name) != expected:
            errors.append('Executed numerical source differs: ' + name)
    return {'files_checked': len(manifest['files']), 'errors': errors}


def recovery_counts(root):
    deployment = json.loads((root / 'audit/deployment_results.json').read_text())
    rows = deployment['primary_episodes']
    groups = defaultdict(list)
    for row in rows:
        groups[row['method'], row['root'], row['target']].append(row)
    counts = defaultdict(lambda: {'problems': 0, 'episodes': 0, 'inner': 0, 'actual': 0})
    errors = []
    for (method, _, _), cases in groups.items():
        counts[method]['problems'] += 1
        counts[method]['episodes'] += len(cases)
        if len(cases) != 9 or {tuple(r['path']) for r in cases} != {(a, b) for a in range(3) for b in range(3)}:
            errors.append('Incomplete uncertainty paths: ' + method)
        for band in ('inner', 'actual'):
            counts[method][band] += all(r['complete'] and r['physical_valid'] and r['terminal_band_success'][band] for r in cases)
    expected_actual = {m: (0 if m == 'nominal_replan' else 2) for m in counts}
    if len(rows) != 4860 or len(groups) != 540:
        errors.append('Primary recovery denominator differs')
    for method, values in counts.items():
        if values != {'problems': 108, 'episodes': 972, 'inner': 0, 'actual': expected_actual[method]}:
            errors.append('Primary recovery counts differ: ' + method)
    if deployment['passed'] is not False or len(deployment['failures']) != 10:
        errors.append('Original deployment audit qualifications differ')
    refined = json.loads((root / 'audit/refined_target_results.json').read_text())
    cohorts = {}
    for name in ('component_midpoints', 'fixed_targets'):
        problems = [p for p in refined['problems'] if p['cohort'] == name]
        episodes = [p for p in refined['episodes'] if p['cohort'] == name]
        cohorts[name] = {'problems': len(problems), 'episodes': len(episodes),
                         'inner_all_paths': sum(p['all_paths_inner_feasible'] for p in problems),
                         'actual_all_paths': sum(p['all_paths_actual_feasible'] for p in problems)}
        if any(not p['complete'] for p in episodes):
            errors.append('Incomplete refined episode')
    if cohorts['component_midpoints'] != {'problems': 161, 'episodes': 1449, 'inner_all_paths': 161, 'actual_all_paths': 161}:
        errors.append('Exploratory midpoint counts differ')
    if cohorts['fixed_targets'] != {'problems': 36, 'episodes': 324, 'inner_all_paths': 1, 'actual_all_paths': 5}:
        errors.append('Separate fixed-target sensitivity counts differ')
    if refined['passed'] is not True or refined['failures']:
        errors.append('Refined verification status differs')
    return {'primary': dict(counts), 'refined': cohorts,
            'original_strict_audit_passed': False, 'retained_original_flags': 10,
            'errors': errors}


def scenario_counts(root):
    folder = root / 'results/locked'
    execution = json.loads((folder / 'execution.json').read_text())
    records = {}
    counts = defaultdict(int)
    errors = []
    for case in execution['cases']:
        for method in execution['methods']:
            row = json.loads((folder / f'{case["id"]}__{method}.json').read_text())
            records[case['id'], method] = row
            counts[method] += bool(row['result']['terminal_feasible'])
            if row['status'] != 'completed' or row['case'] != case or row['method'] != method:
                errors.append('Episode identity/completion: ' + case['id'] + '/' + method)
            if not math.isclose(row['controller_seconds'], math.fsum(s['controller_seconds'] for s in row['stages']), abs_tol=1e-8):
                errors.append('Controller time sum: ' + case['id'] + '/' + method)
    expected = {'ce_direct': 54, 'ce_limited': 53, 'scenario_feedback': 61, 'scenario_managed': 69, 'scenario_open': 68}
    if len(records) != 440 or dict(counts) != expected:
        errors.append('Primary scenario counts differ')
    audit = json.loads((folder / 'independent_amount_audit.json').read_text())
    if audit['passed'] is not False or len(audit['failures']) != 18:
        errors.append('Original strict scenario audit status differs')
    for row in audit['episodes']:
        original = records[row['case'], row['method']]['result']
        if any(row[k] != original[k] for k in ('physical_valid', 'terminal_feasible', 'operational_success')):
            errors.append('Independent outcome classification: ' + row['case'] + '/' + row['method'])
    timing = json.loads((root / 'results/timing_serial/execution.json').read_text())
    serial = defaultdict(list)
    for case in timing['cases']:
        for method in timing['methods']:
            row = json.loads((root / 'results/timing_serial' / f'{case["id"]}__{method}.json').read_text())
            serial[method].append(row)
            if row['result'] != records[case['id'], method]['result']:
                errors.append('Serial result identity differs')
    summary = json.loads((root / 'results/timing_summary.json').read_text())['summary']
    for method, rows in serial.items():
        total = math.fsum(r['controller_seconds'] for r in rows)
        if len(rows) != 8 or not math.isclose(total, summary[method]['total_seconds'], abs_tol=1e-8):
            errors.append('Serial complete-cost total differs: ' + method)
    return {'primary_episodes': len(records), 'successes_out_of_88': dict(counts),
            'serial_episodes': sum(map(len, serial.values())),
            'original_strict_audit_passed': False, 'retained_comparison_flags': 18,
            'errors': errors}


def verify(root):
    record = integrity(root)
    scientific = recovery_counts(root) if root.name == 'terminal_recovery' else scenario_counts(root)
    scientific_errors = scientific.pop('errors')
    errors = record['errors'] + scientific_errors
    report = {'passed': not errors, 'study': root.name, 'files_checked': record['files_checked'],
              'checks_total': 2, 'checks_passed': int(not record['errors']) + int(not scientific_errors),
              'scientific_record_checks': scientific, 'errors': errors,
              'scope': 'Release hashes, executed-source locks, complete numerical denominators and saved outcome arithmetic. Scientific audit failures are retained; packaging success does not turn those failures into passes. No optimisation campaign is rerun.'}
    save(root / 'local_results/package_verification.json', report)
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


def prepare(root, destination):
    destination = destination.expanduser().resolve()
    if destination.exists() or root == destination or root in destination.parents:
        raise ValueError('Choose a new directory outside the supplied study; existing files are never replaced.')
    checked = integrity(root)
    if checked['errors']:
        raise ValueError('\n'.join(checked['errors']))
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    selected = []
    for item in manifest['files']:
        rel = Path(item['release'])
        if rel.parts[0] == 'data' or rel.parts[0] == 'vendor' or rel.suffix == '.py' and 'public_validation' not in rel.parts and 'results' not in rel.parts:
            selected.append(rel)
        elif len(rel.parts) == 1 and rel.name in ('PROTOCOL.json', 'protocol.json', 'LOCK.json', 'SOURCE.json', 'ENVIRONMENT.json', 'requirements.txt', 'EXPLORATORY_TARGET_SET_PROTOCOL.md', 'EXPLORATORY_REFINEMENT_AMENDMENT.md'):
            selected.append(rel)
        elif rel == Path('audit/design_lock.json'):
            selected.append(rel)
        elif root.name == 'scenario_feedback' and rel.parts[0] == 'audit' and 'protocol' in rel.name and rel.name != 'replay_discrepancy_protocol.json':
            selected.append(rel)
        elif 'core_snapshot_01' in rel.parts:
            selected.append(rel)
    destination.mkdir(parents=True)
    copied = {}
    for rel in sorted(set(selected)):
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, target)
        copied[str(rel)] = sha(target)
    for name in ('results', 'figures', 'audit'):
        (destination / name).mkdir(exist_ok=True)
    commands = json.loads((root / 'COMMANDS.json').read_text())
    save(destination / 'RUN_COMMANDS.json', commands)
    save(destination / 'PREPARED_INPUTS.json', {'source_study': root.name, 'copied_inputs': copied,
         'initial_results': 0, 'execution': 'Not started. Use RUN_COMMANDS.json and the study README.'})
    shutil.copy2(root / 'README.md', destination / 'STUDY_GUIDE.md')
    print(json.dumps({'prepared': str(destination), 'copied_inputs': len(copied), 'initial_results': 0}, indent=2))


def numerical_smoke(root):
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    import numpy as np
    sys.path.insert(0, str(root / 'scripts'))
    sys.path.insert(0, str(root / 'audit'))
    from independent_amount_model import AmountReactor, from_log, to_log
    rows = []
    if root.name == 'terminal_recovery':
        import engine
        for family in ('table', 'analytic'):
            initial = engine.initial()
            terminal = engine.flow(initial, .07, np.array([.975, 1.025]), engine.arguments(family))[0]
            reference = AmountReactor(family).interval(from_log(initial), .07, [.975, 1.025])
            error = float(np.max(abs(terminal - reference['state'])))
            rows.append({'family': family, 'maximum_state_error': error, 'passed': error < 2e-7})
    else:
        from model import Reactor, initial_state
        for family in ('table', 'analytic'):
            initial = initial_state(1., 20.)
            native, ordinary = Reactor(family, None, True), Reactor(family, None, False)
            a = native.predict(initial, [.07] * 7, [.975, 1.025], gradients=True, tight=True)
            b = ordinary.predict(initial, [.07] * 7, [.975, 1.025], gradients=True, tight=True)
            error = max(float(np.max(abs(np.asarray(a[k]) - np.asarray(b[k])))) for k in ('state', 'jz', 'jlogB'))
            rows.append({'family': family, 'native_available': native.fast, 'maximum_state_or_gradient_error': error,
                         'passed': bool(native.fast and error < 2e-6)})
    report = {'passed': all(r['passed'] for r in rows), 'rows': rows,
              'scope': 'Two model-family numerical installation checks. No full campaign or held-out efficacy evaluation.'}
    save(root / 'local_results/numerical_smoke.json', report)
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


def main(root):
    root = Path(root).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify', 'prepare', 'numerical-smoke'))
    parser.add_argument('destination', nargs='?', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.destination is None:
            parser.error('prepare requires a new destination directory')
        prepare(root, args.destination)
    elif args.command == 'verify':
        verify(root)
    else:
        numerical_smoke(root)
