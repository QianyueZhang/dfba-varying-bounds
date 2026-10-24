"""Run the prespecified continuous-open subset after a complete locked design.

This runner implements PROTOCOL.json without changing its frozen eight sources.
It records its own source/input hashes before the first solve. Use --execute;
--check-only verifies readiness without numerical computation.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/continuous_baseline'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def preflight():
    lock_path = ROOT / 'audit/design_lock.json'
    protocol_path = ROOT / 'PROTOCOL.json'
    if not lock_path.exists() or not protocol_path.exists():
        raise RuntimeError('Frozen protocol and design lock are required before execution')
    lock = json.loads(lock_path.read_text())
    for relative, expected in lock.items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('Frozen source identity mismatch: ' + relative)
    protocol = json.loads(protocol_path.read_text())
    if protocol['status'] != 'frozen_before_evaluation':
        raise RuntimeError('Protocol is not frozen for evaluation')
    roots_path = ROOT / 'results/roots.json'
    primary_path = ROOT / 'results/primary_design.json'
    if not roots_path.exists() or not primary_path.exists():
        raise RuntimeError('Await complete roots and primary design; no baseline was started')
    roots = json.loads(roots_path.read_text())
    primary = json.loads(primary_path.read_text())
    if len(roots) != protocol['root_count'] or len(primary) != protocol['primary_root_target_count']:
        raise RuntimeError('Primary design incomplete: %d roots and %d root-target rows' % (len(roots), len(primary)))
    root_lookup = {r['id']: r for r in roots}
    expected = {(r['id'], float(t)) for r in roots for t in protocol['fixed_targets']}
    actual = {(r['root'], float(r['target'])) for r in primary}
    if len(root_lookup) != len(roots) or len(actual) != len(primary) or actual != expected:
        raise RuntimeError('Primary root-target identities are missing, duplicated or unexpected')
    subset = [r for r in primary if root_lookup[r['root']]['timing'] == 'uniform']
    if len(subset) != 36 or len({r['root'] for r in subset}) != 6:
        raise RuntimeError('Expected all six uniform roots by six fixed targets')
    expected_uniform = {(family, float(volume)) for family in protocol['families']
                        for volume in protocol['root_volumes_L']}
    actual_uniform = {(root_lookup[r['root']]['family'], float(root_lookup[r['root']]['volume'])) for r in subset}
    if actual_uniform != expected_uniform:
        raise RuntimeError('Uniform subset family-volume coverage differs from the frozen design')
    source_identity = dict(lock)
    for relative in ('audit/design_lock.json', 'results/roots.json', 'results/primary_design.json',
                     'scripts/run_continuous.py'):
        source_identity[relative] = sha(ROOT / relative)
    return protocol, root_lookup, subset, source_identity


def execute():
    protocol, root_lookup, subset, identities = preflight()
    manifest_path = OUT / 'execution.json'
    if manifest_path.exists():
        raise RuntimeError('A continuous execution already exists; this runner will not overwrite or rerun it')
    import numpy as np
    import scipy
    import numba
    import llvmlite
    from engine import arguments, K, CAP, GUARD, BAND, DT
    from continuous_baseline import solve_open, INNER_MARGIN, COMPARISON_TOLERANCE, FLOW_NUMERICS
    sys.path.insert(0, str(ROOT / 'audit'))
    import independent_amount_model as independent
    audit_settings = protocol['audit']
    if not np.array_equal(np.asarray(protocol['K']), K):
        raise RuntimeError('Runtime K differs from the protocol')
    if not (float(CAP) == protocol['volume_cap_L'] and float(GUARD) == protocol['volume_guard_L']
            and float(BAND) == protocol['outer_design_band'] and abs(BAND-INNER_MARGIN-protocol['inner_design_band']) < 1e-15
            and COMPARISON_TOLERANCE == protocol['comparison_tolerance'] and DT == protocol['interval_hours']):
        raise RuntimeError('Baseline constants differ from the frozen protocol')
    if (independent.RTOL, independent.ATOL, independent.MAX_STEP) != (
            audit_settings['independent_rtol'], audit_settings['independent_atol'], audit_settings['independent_maxstep_h']):
        raise RuntimeError('Independent replay settings differ from the protocol')
    if FLOW_NUMERICS != dict(rtol=audit_settings['engine_rtol'], atol=audit_settings['engine_atol'],
                            max_step_h=audit_settings['engine_maxstep_h']):
        raise RuntimeError('Continuous prediction numerics differ from the protocol')
    manifest = dict(status='declared_before_continuous_execution', declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    source_input_sha256=identities, original_protocol=protocol,
                    assigned=[dict(root=r['root'], target=r['target']) for r in subset], expected=36,
                    runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                 numba=numba.__version__, llvmlite=llvmlite.__version__, platform=platform.platform()),
                    audit=dict(checkpoints='Three shared first-step states and all nine final paths, chained from the declared root state',
                               numerical_comparisons=audit_settings, relative_amount_reference='independent physical amount',
                               label_checks='Report inner, outer and actual-acceptance labels separately; any disagreement remains visible',
                               classifications='Solver convergence status is separate from validity of the retained candidate'),
                    scope='Prespecified continuous local baseline on 36 uniform-root/target problems; no global optimum or controller-speed inference')
    save(manifest_path, manifest)
    started = time.perf_counter()
    summary = dict(complete=False, expected=36, completed=0, failed=0, records=[],
                   source_input_sha256=identities, execution_sha256=sha(manifest_path))

    def independent_replay(root, target, solve):
        tick = time.perf_counter()
        selected_index = solve['selected_candidate_index']
        if selected_index is None:
            return dict(completed=False, numerical_checks_passed=False, passed=False,
                        failures=[dict(code='no_valid_selected_candidate', detail=solve.get('failure'))],
                        label_disagreements=[], elapsed_seconds=time.perf_counter()-tick)
        selected = solve['candidates'][selected_index]
        plant = independent.AmountReactor(root['family'])
        initial = independent.from_log(root['state'])
        plan = np.asarray(solve['plan'], float)
        failures = []
        label_disagreements = []
        comparisons = []
        terminal_rows = []
        component_max = np.zeros(4)
        max_z_error = 0.
        calls = 0

        def compare(amounts, saved_state, context):
            nonlocal max_z_error, component_max
            saved_amounts = independent.from_log(saved_state)
            errors = np.abs(amounts - saved_amounts)
            component_max = np.maximum(component_max, errors)
            z = float(np.log(amounts[1] / amounts[2]))
            saved_z = float(saved_state[1] - saved_state[2])
            z_error = abs(z - saved_z)
            max_z_error = max(max_z_error, z_error)
            allowed = audit_settings['amount_absolute_g'] + audit_settings['amount_relative'] * np.abs(amounts[1:])
            checks = dict(volume=bool(errors[0] <= audit_settings['volume_absolute_L']),
                          physical_amounts=bool(np.all(errors[1:] <= allowed)),
                          log_ratio=bool(z_error <= audit_settings['log_ratio_absolute']))
            for code, passed in checks.items():
                if not passed:
                    failures.append(dict(code=code, context=context, component_errors=errors.tolist(),
                                         allowed_amount_errors=allowed.tolist(), log_ratio_error=z_error))
            comparisons.append(dict(context=context, checks=checks, amount_state=amounts.tolist(),
                                    saved_amount_state=saved_amounts.tolist(), absolute_errors=errors.tolist(),
                                    z=z, saved_z=saved_z, log_ratio_error=z_error))

        plan_physical = bool(np.all(np.isfinite(plan)) and np.all((0 <= plan) & (plan <= .2))
                             and 2 * plan.sum() <= CAP - initial[0] - GUARD)
        physical_all = plan_physical
        for i, k0 in enumerate(K):
            first = plant.interval(initial, plan[0], k0)
            calls += 1
            compare(first['amount_state'], selected['first_states'][i], dict(stage=0, scenario=i))
            for j, k1 in enumerate(K):
                terminal = plant.interval(first['amount_state'], plan[1], k1)
                calls += 1
                expected_path = selected['paths'][i * len(K) + j]
                if expected_path['path'] != [i, j]:
                    raise RuntimeError('Saved continuous scenario path ordering differs')
                compare(terminal['amount_state'], expected_path['state'], dict(stage=1, path=[i, j]))
                y = terminal['amount_state']
                z = float(np.log(y[1] / y[2]))
                error = float(abs(z - np.log(target)))
                minimum = min(float(initial[3]/initial[0]), first['min_S'], terminal['min_S'])
                physical = bool(plan_physical and first['amount_state'][0] <= CAP and y[0] <= CAP
                                and minimum >= -protocol['physical_substrate_tolerance_g_L'])
                physical_all = physical_all and physical
                terminal_rows.append(dict(path=[i, j], amount_state=y.tolist(), z=z,
                                          logB=float(np.log(y[1]+y[2])), biomass_gDW=float(y[1]+y[2]),
                                          error=error, min_S=minimum, physical_valid=physical,
                                          inner_feasible=bool(physical and error <= protocol['inner_design_band'] + COMPARISON_TOLERANCE),
                                          outer_feasible=bool(physical and error <= BAND + COMPARISON_TOLERANCE),
                                          actual_terminal_feasible=bool(physical and error <= np.log(1.01))))
        inner = bool(all(r['inner_feasible'] for r in terminal_rows))
        outer = bool(all(r['outer_feasible'] for r in terminal_rows))
        actual = bool(all(r['actual_terminal_feasible'] for r in terminal_rows))
        for field, expected, old in (('inner_feasible', inner, solve['feasible']),
                                     ('outer_feasible', outer, solve['outer_band_feasible']),
                                     ('physical_valid', bool(physical_all), selected['physical_valid'])):
            if bool(expected) != bool(old):
                label_disagreements.append(dict(field=field, independent=bool(expected), engine=bool(old)))
        return dict(completed=True, numerical_checks_passed=not failures,
                    passed=not failures and not label_disagreements,
                    failures=failures, label_disagreements=label_disagreements,
                    comparisons=comparisons, paths=terminal_rows,
                    max_component_absolute_error=component_max.tolist(), max_log_ratio_absolute_error=max_z_error,
                    independent_inner_feasible=inner, independent_outer_feasible=outer,
                    independent_actual_terminal_feasible=actual, independent_physical_valid=bool(physical_all),
                    independent_max_error=max(r['error'] for r in terminal_rows),
                    independent_worst_biomass_gDW=min(r['biomass_gDW'] for r in terminal_rows),
                    interval_integrations=calls, elapsed_seconds=time.perf_counter()-tick)

    for ordinal, row in enumerate(subset):
        root = root_lookup[row['root']]
        target = float(row['target'])
        filename = '%s__target_%g.json' % (root['id'], target)
        case_started = time.perf_counter()
        record = dict(ordinal=ordinal, root=root['id'], family=root['family'], target=target,
                      status='started', execution_sha256=sha(manifest_path),
                      initial_state=root['state'], grid_comparator=row['values']['open'])
        try:
            base = row['values']['open']
            result = solve_open(np.asarray(root['state'], float), arguments(root['family']), target,
                                base['minimax_plan'], base['plan'] if base['safe_count'] > 0 else None)
            record['solve'] = result
            record['audit'] = independent_replay(root, target, result)
            record['status'] = 'completed'
            summary['completed'] += 1
            summary['records'].append(dict(file=filename, root=root['id'], family=root['family'], target=target,
                                           status='completed', feasible=result['feasible'], best_error=result['best_error'],
                                           worstB=result['worstB'], seconds=result['seconds'], count=result['count'],
                                           valid_retained_candidate=result['selected_candidate_index'] is not None,
                                           optimizer_statuses=result['solver_statuses'],
                                           audit_completed=record['audit']['completed'],
                                           numerical_checks_passed=record['audit']['numerical_checks_passed'],
                                           audit_passed=record['audit']['passed'],
                                           audit_failures=record['audit']['failures'],
                                           label_disagreements=record['audit']['label_disagreements'],
                                           independent_inner_feasible=record['audit'].get('independent_inner_feasible'),
                                           independent_outer_feasible=record['audit'].get('independent_outer_feasible'),
                                           independent_actual_terminal_feasible=record['audit'].get('independent_actual_terminal_feasible'),
                                           independent_max_error=record['audit'].get('independent_max_error'),
                                           max_log_ratio_absolute_error=record['audit'].get('max_log_ratio_absolute_error'),
                                           max_component_absolute_error=record['audit'].get('max_component_absolute_error')))
        except Exception as exc:
            record.update(status='failed', error=repr(exc))
            summary['failed'] += 1
            summary['records'].append(dict(file=filename, root=root['id'], family=root['family'], target=target,
                                           status='failed', error=repr(exc)))
        record['total_seconds'] = time.perf_counter()-case_started
        save(OUT / filename, record)
        summary['records'][-1]['sha256'] = sha(OUT / filename)
        save(OUT / 'summary.json', summary)
        print('%d/36 %s target=%g status=%s' % (ordinal+1, root['id'], target, record['status']), flush=True)
    preserved = all(sha(ROOT / relative) == expected for relative, expected in identities.items())
    summary['source_input_hashes_preserved'] = preserved
    summary['complete'] = bool(summary['completed'] + summary['failed'] == 36 and preserved)
    completed = [r for r in summary['records'] if r['status'] == 'completed']
    summary['all_assigned_solved'] = len(completed) == 36
    summary['valid_retained_candidates'] = sum(r['valid_retained_candidate'] for r in completed)
    summary['engine_inner_feasible_count'] = sum(r['feasible'] for r in completed)
    summary['independent_inner_feasible_count'] = sum(r['independent_inner_feasible'] is True for r in completed)
    summary['independent_actual_all_paths_feasible_count'] = sum(r['independent_actual_terminal_feasible'] is True for r in completed)
    summary['numerical_failed_records'] = sum(not r['numerical_checks_passed'] for r in completed)
    summary['label_disagreement_records'] = sum(bool(r['label_disagreements']) for r in completed)
    summary['optimizer_calls'] = sum(len(r['optimizer_statuses']) for r in completed)
    summary['optimizer_nonconvergence_calls'] = sum(not s['success'] for r in completed for s in r['optimizer_statuses'])
    summary['elapsed_seconds'] = time.perf_counter()-started
    summary['timing_scope'] = 'Offline solve and independent-audit elapsed times; no isolated or real-time acceleration claim'
    save(OUT / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('records', 'source_input_sha256')}, indent=2))
    if not preserved:
        raise RuntimeError('Sources or complete input panel changed during execution; results are retained for inspection')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--execute', action='store_true')
    group.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        protocol, roots, subset, identities = preflight()
        print(json.dumps(dict(ready=True, roots=len(roots), assigned=len(subset), source_input_hashes=identities,
                              numerical_execution=False), indent=2))
    else:
        execute()


if __name__ == '__main__':
    main()
