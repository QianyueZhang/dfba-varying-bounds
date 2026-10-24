"""Post hoc independent replay of the retained minimum-error open-loop plans.

All 36 completed problems are included. The original economic/selected plans,
audits and strict-inner label disagreement remain unchanged.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'audit'
PROTOCOL = HERE / 'continuous_minima_protocol.json'
OUTPUT = HERE / 'continuous_minima_results.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def declare():
    if PROTOCOL.exists() or OUTPUT.exists():
        raise RuntimeError('Supplementary minima replay already declared or run')
    lock = json.loads((HERE / 'design_lock.json').read_text())
    for relative, expected in lock.items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('Frozen input changed: ' + relative)
    primary = json.loads((ROOT / 'results/continuous_baseline/summary.json').read_text())
    if not primary['complete'] or not primary['source_input_hashes_preserved'] or primary['completed'] != 36:
        raise RuntimeError('All 36 original continuous records must be complete')
    refined = json.loads((HERE / 'refined_target_results.json').read_text())
    if not refined['complete']:
        raise RuntimeError('Complete the separate refined witness audit before this supplementary replay')
    expected = {(r['root'], float(r['target'])) for r in primary['records']}
    grid41 = [r for r in json.loads((ROOT / 'results/grid_sensitivity.json').read_text()) if r['grid'] == 41]
    if len(expected) != 36 or len(grid41) != 36 or {(r['root'], float(r['target'])) for r in grid41} != expected:
        raise RuntimeError('The 36 grid-41 fixed targets must exactly match the continuous subset')
    selections = []
    inputs = list(lock) + ['audit/design_lock.json', 'audit/verify_continuous_minima.py',
                          'results/roots.json', 'results/continuous_baseline/summary.json',
                          'results/continuous_baseline/execution.json', 'results/grid_sensitivity.json',
                          'audit/refined_target_protocol.json', 'audit/refined_target_results.json',
                          'EXPLORATORY_REFINEMENT_AMENDMENT.md']
    for item in primary['records']:
        relative = 'results/continuous_baseline/' + item['file']
        path = ROOT / relative
        if sha(path) != item['sha256']:
            raise RuntimeError('Original continuous record changed: ' + relative)
        original = json.loads(path.read_text())
        choices = [c for c in original['solve']['candidates'] if c['numerical_valid'] and c['physical_valid']]
        if not choices:
            raise RuntimeError('No valid candidate in assigned problem: ' + relative)
        selected = min(choices, key=lambda c: (c['max_error'], -c['worst_logB'], c['index']))
        selections.append(dict(root=item['root'], target=item['target'], source=relative,
                               candidate_index=selected['index'], plan=selected['plan'],
                               engine_max_error=selected['max_error'], worstB=selected['worstB']))
        inputs.append(relative)
    protocol = dict(status='Post hoc retained-minimum selection declared before its independent replay',
                    declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    selection='ALL 36 original problems; select numerical_valid AND physical_valid retained candidate by minimum max_error, then maximum worst_logB, then ascending candidate index',
                    cases=selections, expected_cases=36, expected_paths=324,
                    source_input_sha256={p: sha(ROOT / p) for p in inputs},
                    numerics=json.loads((ROOT / 'PROTOCOL.json').read_text())['audit'],
                    classification='Original inner 0.00748 plus 1e-10 comparison tolerance; outer 0.0075 and actual log(1.01) without added tolerance; retain all numerical and label discrepancies',
                    comparison='Compare cached grid-41 DP minimax J and deployed economic-policy maximum errors separately against this retained local-search minimum. No global continuous optimum claim.',
                    preservation='Supplementary replay only; do not replace original selected/economic plan, audit, classification flip or costs',
                    timing='Offline replay; no speed inference')
    save(PROTOCOL, protocol)
    print(json.dumps(dict(declared=True, cases=36, paths=324, numerical_execution=False), indent=2))


def execute():
    if not PROTOCOL.exists() or OUTPUT.exists():
        raise RuntimeError('A new supplementary protocol without existing output is required')
    protocol = json.loads(PROTOCOL.read_text())
    for relative, expected in protocol['source_input_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('Declared source/input changed: ' + relative)
    sys.path.insert(0, str(HERE))
    import numpy as np
    import scipy
    import independent_amount_model as independent
    design = json.loads((ROOT / 'PROTOCOL.json').read_text())
    n = protocol['numerics']
    if (independent.RTOL, independent.ATOL, independent.MAX_STEP) != (n['independent_rtol'], n['independent_atol'], n['independent_maxstep_h']):
        raise RuntimeError('Independent integrator differs from declared settings')
    roots = {r['id']: r for r in json.loads((ROOT / 'results/roots.json').read_text())}
    grid41 = {(r['root'], float(r['target'])): r for r in json.loads((ROOT / 'results/grid_sensitivity.json').read_text()) if r['grid'] == 41}
    refined = {(r['root'], float(r['target'])): r for r in json.loads((HERE / 'refined_target_results.json').read_text())['problems'] if r['cohort'] == 'fixed_targets'}
    report = dict(complete=False, protocol_sha256=sha(PROTOCOL), protocol=protocol, cases=[],
                  runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__))
    started = time.perf_counter()
    for item in protocol['cases']:
        tick = time.perf_counter()
        original = json.loads((ROOT / item['source']).read_text())
        selected = original['solve']['candidates'][item['candidate_index']]
        root = roots[item['root']]
        target = float(item['target'])
        initial = independent.from_log(root['state'])
        plant = independent.AmountReactor(root['family'])
        row = dict(item, completed=False, paths=[], comparisons=[], failures=[], label_disagreements=[],
                   original_selected_candidate_index=original['solve']['selected_candidate_index'],
                   original_selected_engine_error=original['solve']['best_error'],
                   original_selected_independent_error=original['audit'].get('independent_max_error'))
        try:
            def compare(actual, saved, context):
                old = independent.from_log(saved)
                errors = np.abs(actual-old)
                allowance = n['amount_absolute_g'] + n['amount_relative']*np.abs(actual[1:])
                z_error = abs(float(np.log(actual[1]/actual[2])) - float(saved[1]-saved[2]))
                tests = dict(amounts=bool(np.all(errors[1:] <= allowance)), volume=bool(errors[0] <= n['volume_absolute_L']),
                             log_ratio=bool(z_error <= n['log_ratio_absolute']))
                record = dict(context=context, absolute_component_errors=errors.tolist(),
                              amount_allowance=allowance.tolist(), log_ratio_error=z_error, tests=tests)
                row['comparisons'].append(record)
                if not all(tests.values()):
                    row['failures'].append(record)

            u0, u1 = item['plan']
            plan_valid = bool(0 <= u0 <= .2 and 0 <= u1 <= .2 and
                              2*(u0+u1) <= design['volume_cap_L']-initial[0]-design['volume_guard_L'])
            for i, k0 in enumerate(design['K']):
                first = plant.interval(initial, u0, k0)
                compare(first['amount_state'], selected['first_states'][i], dict(stage=0, scenario=i))
                for j, k1 in enumerate(design['K']):
                    terminal = plant.interval(first['amount_state'], u1, k1)
                    y = terminal['amount_state']
                    compare(y, selected['paths'][i*3+j]['state'], dict(stage=1, path=[i, j]))
                    error = abs(float(np.log(y[1]/y[2]))-math.log(target))
                    minimum = min(float(initial[3]/initial[0]), first['min_S'], terminal['min_S'])
                    physical = bool(plan_valid and first['amount_state'][0] <= design['volume_cap_L'] and y[0] <= design['volume_cap_L']
                                    and minimum >= -design['physical_substrate_tolerance_g_L'])
                    row['paths'].append(dict(path=[i, j], state=y.tolist(), error=error, min_S=minimum,
                                             biomass_gDW=float(y[1]+y[2]), physical_valid=physical,
                                             inner=bool(physical and error <= design['inner_design_band']+design['comparison_tolerance']),
                                             outer=bool(physical and error <= design['outer_design_band']),
                                             actual=bool(physical and error <= math.log(1.01))))
            labels = {band: all(r[band] for r in row['paths']) for band in ('inner', 'outer', 'actual')}
            expected_labels = dict(inner=selected['feasible'], outer=selected['outer_band_feasible'],
                                   actual=bool(selected['physical_valid'] and selected['max_error'] <= math.log(1.01)))
            for band in labels:
                if labels[band] != expected_labels[band]:
                    row['label_disagreements'].append(dict(band=band, independent=labels[band], engine=expected_labels[band]))
            maximum = max(r['error'] for r in row['paths'])
            mapping = grid41[(item['root'], target)]
            deployment = refined[(item['root'], target)]
            row.update(completed=True, independent_max_error=maximum, independent_labels=labels,
                       independent_worstB=min(r['biomass_gDW'] for r in row['paths']),
                       numerical_checks_passed=not row['failures'],
                       passed=not row['failures'] and not row['label_disagreements'],
                       grid41_minimax_J=mapping['J'], grid41_minimax_minus_retained_minimum=mapping['J']-selected['max_error'],
                       difference_within_two_audit_log_ratio_tolerances=bool(abs(mapping['J']-selected['max_error']) <= 2*n['log_ratio_absolute']),
                       grid41_economic_deployed_max_error=deployment['maximum_error'],
                       grid41_economic_all_paths_inner=deployment['all_paths_inner_feasible'],
                       grid41_economic_all_paths_outer=deployment['all_paths_outer_feasible'],
                       comparison_note='Finite-grid causal minimax value vs best retained continuous local open candidate. Raw differences are retained; differences within twice the audit log-ratio threshold are flagged as numerically close, not proven superiority or a rigorous uncertainty interval. No global continuous optimality claim.')
        except Exception as exc:
            row['exception'] = repr(exc)
        row['seconds'] = time.perf_counter()-tick
        report['cases'].append(row)
        save(OUTPUT, report)
        print('%d/36 %s target=%g completed=%s' % (len(report['cases']), item['root'], target, row['completed']), flush=True)
    preserved = all(sha(ROOT / p) == h for p, h in protocol['source_input_sha256'].items())
    completed = [r for r in report['cases'] if r['completed']]
    report['source_input_hashes_preserved'] = preserved
    report['complete'] = bool(preserved and len(report['cases']) == 36)
    report['all_cases_completed'] = len(completed) == 36
    report['passed'] = bool(report['complete'] and report['all_cases_completed'] and all(r['passed'] for r in completed))
    report['summary'] = dict(completed=len(completed), numerical_failed_cases=sum(not r['numerical_checks_passed'] for r in completed),
                           label_disagreement_cases=sum(bool(r['label_disagreements']) for r in completed),
                           inner_feasible=sum(r['independent_labels']['inner'] for r in completed),
                           outer_feasible=sum(r['independent_labels']['outer'] for r in completed),
                           actual_all_paths_feasible=sum(r['independent_labels']['actual'] for r in completed),
                           grid41_J_less_than_retained_minimum=sum(r['grid41_minimax_minus_retained_minimum'] < -design['comparison_tolerance'] for r in completed),
                           grid41_J_greater_than_retained_minimum=sum(r['grid41_minimax_minus_retained_minimum'] > design['comparison_tolerance'] for r in completed))
    report['summary']['numerically_close_comparisons_at_twice_audit_threshold'] = sum(r['difference_within_two_audit_log_ratio_tolerances'] for r in completed)
    report['elapsed_seconds'] = time.perf_counter()-started
    save(OUTPUT, report)
    print(json.dumps(dict(complete=report['complete'], passed=report['passed'], summary=report['summary'],
                         elapsed_seconds=report['elapsed_seconds']), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--declare', action='store_true')
    group.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    declare() if args.declare else execute()


if __name__ == '__main__':
    main()
