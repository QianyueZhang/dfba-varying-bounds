"""Independent grid-41 causal witnesses, with all selected targets retained.

Declare before execution. Component midpoints and fixed targets are separate
post hoc cohorts. No hidden scenario label is supplied to either decision.
"""
from pathlib import Path
import argparse
import collections
import datetime
import hashlib
import itertools
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'audit'
PROTOCOL = HERE / 'refined_target_protocol.json'
RESULTS = HERE / 'refined_target_results.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if hasattr(value, 'tolist'):
        return finite(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(finite(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def declare():
    if PROTOCOL.exists() or RESULTS.exists():
        raise RuntimeError('Refinement verification already declared or executed')
    design = json.loads((ROOT / 'PROTOCOL.json').read_text())
    lock = json.loads((HERE / 'design_lock.json').read_text())
    for relative, expected in lock.items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('Frozen source changed: ' + relative)
    roots = json.loads((ROOT / 'results/roots.json').read_text())
    uniform = {r['id']: r for r in roots if r['timing'] == 'uniform'}
    sets = [r for r in json.loads((ROOT / 'results/target_set_sensitivity.json').read_text()) if r['grid'] == 41]
    if len(uniform) != 6 or {r['root'] for r in sets} != set(uniform) or len(sets) != 6:
        raise RuntimeError('Expected six uniform grid-41 target sets')
    cases = []
    for row in sets:
        for component, (lo, hi) in enumerate(row['feedback']):
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                raise RuntimeError('Invalid stored target component')
            cases.append(dict(id=row['root'] + '__component_' + str(component), root=row['root'],
                              cohort='component_midpoints', component=component, log_interval=[lo, hi],
                              target=math.exp((lo + hi) / 2)))
    if len(cases) != 161:
        raise RuntimeError('Expected the observed complete set of 161 connected components')
    for root in uniform:
        for target in design['fixed_targets']:
            cases.append(dict(id=root + '__fixed_' + str(target), root=root,
                              cohort='fixed_targets', target=float(target)))
    inputs = list(lock) + ['audit/design_lock.json', 'audit/verify_refined_targets.py',
                          'results/roots.json', 'results/primary_design.json', 'results/target_set_sensitivity.json',
                          'EXPLORATORY_TARGET_SET_PROTOCOL.md', 'EXPLORATORY_REFINEMENT_AMENDMENT.md']
    protocol = dict(status='Post hoc verification declared before its integrations',
                    declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    source_input_sha256={p: sha(ROOT / p) for p in inputs}, cases=cases,
                    expected=dict(component_midpoints=161, midpoint_episodes=1449, fixed_targets=36,
                                  fixed_target_episodes=324, total_problems=197, total_episodes=1773),
                    paths=[list(p) for p in itertools.product(range(3), repeat=2)],
                    grid=dict(count=41, lower=0., upper=.2), K=design['K'],
                    method='recovery_economic',
                    construction='Rebuild every uniform-root grid-41 tree and save it; first action uses that tree. At the independent actual next state, recompute the grid-41 one-step economic/minimax decision from exact full-state observation.',
                    cache='Exact state bytes, family, feed and kappa for matched plant transitions; family and exact observed state bytes for one-step controller maps. No rounding, disturbance labels or hidden parameters enter controller decisions.',
                    thresholds=dict(inner=design['inner_design_band'], outer=design['outer_design_band'],
                                    actual=math.log(1.01), comparison_tolerance=design['comparison_tolerance'],
                                    physical_substrate_min=-design['physical_substrate_tolerance_g_L'],
                                    volume_cap=design['volume_cap_L'], feed_max=.2),
                    numerics=design['audit'],
                    classification='Inner includes only declared 1e-10 comparison tolerance; outer and actual bands have no added comparison tolerance. Numerical and label mismatches remain visible; optimizer/map values do not replace deployed outcomes.',
                    midpoint_rule='Retain inner-band witness failures as qualifications and outer-band witness failures as hard failures. Fixed-target nonrecovery is an outcome, not an audit implementation error.',
                    scope='All component midpoints are outcome-selected consistency witnesses, not interval-edge verification or a prospective efficacy denominator. Grid-41 fixed targets are a separate post hoc sensitivity comparison; primary grid-21 outcomes unchanged.',
                    timing='Offline cached verification only, no online speed inference')
    save(PROTOCOL, protocol)
    print(json.dumps(dict(declared=True, expected=protocol['expected'], numerical_execution=False), indent=2))


def execute():
    if not PROTOCOL.exists() or RESULTS.exists():
        raise RuntimeError('A new declared refinement protocol without existing results is required')
    protocol = json.loads(PROTOCOL.read_text())
    for relative, expected in protocol['source_input_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('Declared input changed: ' + relative)
    sys.path.insert(0, str(ROOT / 'scripts'))
    sys.path.insert(0, str(HERE))
    import numpy as np
    import scipy
    import numba
    import engine as e
    import policies
    from campaign import save_tree
    import independent_amount_model as independent
    n = protocol['numerics']
    thresholds = protocol['thresholds']
    if (independent.RTOL, independent.ATOL, independent.MAX_STEP) != (n['independent_rtol'], n['independent_atol'], n['independent_maxstep_h']):
        raise RuntimeError('Independent integrator differs from protocol')
    if not np.array_equal(e.K, protocol['K']) or policies.INNER != thresholds['inner'] or policies.TOL != thresholds['comparison_tolerance']:
        raise RuntimeError('Decision constants differ from protocol')
    e.GRID = np.linspace(0., .2, 41)
    roots = {r['id']: r for r in json.loads((ROOT / 'results/roots.json').read_text()) if r['timing'] == 'uniform'}
    arguments = {family: e.arguments(family) for family in ('analytic', 'table')}
    plants = {family: independent.AmountReactor(family) for family in ('analytic', 'table')}
    report = dict(complete=False, protocol_sha256=sha(PROTOCOL), protocol=protocol,
                  trees=[], transitions=[], second_decisions=[], problems=[], episodes=[], failures=[], witness_qualifications=[],
                  runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__))
    counts = collections.Counter()
    cached_transitions = {}
    cached_maps = {}
    cached_decisions = {}
    started = time.perf_counter()

    def failure(code, **detail):
        report['failures'].append(finite(dict(code=code, **detail)))

    def key_state(x):
        return np.asarray(x, dtype=np.float64).tobytes()

    def physical(amount, minimum, feed):
        return bool(np.all(np.isfinite(amount)) and amount[0] > 0 and np.all(amount[1:3] > 0)
                    and amount[0] <= thresholds['volume_cap'] and minimum >= thresholds['physical_substrate_min']
                    and 0 <= feed <= thresholds['feed_max'])

    def transition(initial, feed, kappa, family):
        counts['logical_interval_requests'] += 1
        key = (family, key_state(initial), float(feed), tuple(kappa))
        if key in cached_transitions:
            counts['transition_cache_hits'] += 1
            return report['transitions'][cached_transitions[key]]
        index = len(report['transitions'])
        cached_transitions[key] = index
        row = dict(id=index, family=family, initial_amount_state=np.asarray(initial).tolist(), feed=float(feed),
                   kappa=list(kappa), completed=False)
        report['transitions'].append(row)
        tick = time.perf_counter()
        try:
            counts['independent_interval_calls'] += 1
            actual = plants[family].interval(initial, feed, kappa)
            a = actual['amount_state']
            row['independent'] = dict(amount_state=a.tolist(), min_S=actual['min_S'], max_S=actual['max_S'],
                                      rhs_calls=actual['rhs_calls'], physical_valid=physical(a, actual['min_S'], feed))
            counts['matched_engine_interval_calls'] += 1
            prediction, minimum, accepted, rejected = e.flow(independent.to_log(initial), feed, np.asarray(kappa),
                                                            arguments[family], n['engine_rtol'], n['engine_atol'], n['engine_maxstep_h'])
            b = independent.from_log(prediction)
            row['engine'] = dict(amount_state=b.tolist(), min_S=float(minimum), accepted_steps=int(accepted),
                                 rejected_steps=int(rejected), physical_valid=physical(b, minimum, feed))
            errors = np.abs(a-b)
            allowance = n['amount_absolute_g'] + n['amount_relative'] * np.abs(a[1:])
            z_error = abs(float(np.log(a[1]/a[2]) - np.log(b[1]/b[2])))
            tests = dict(amounts=bool(np.all(errors[1:] <= allowance)), volume=bool(errors[0] <= n['volume_absolute_L']),
                         log_ratio=bool(z_error <= n['log_ratio_absolute']))
            row['comparison'] = dict(passed=all(tests.values()), tests=tests,
                                      absolute_component_errors=errors.tolist(), amount_allowance=allowance.tolist(),
                                      log_ratio_error=z_error)
            for test, passed in tests.items():
                if not passed:
                    failure('matched_interval_' + test, transition_id=index, comparison=row['comparison'])
            if row['independent']['physical_valid'] != row['engine']['physical_valid']:
                failure('matched_physical_label_disagreement', transition_id=index)
            if not row['independent']['physical_valid']:
                failure('physical_invalid', transition_id=index)
            row['completed'] = True
        except Exception as exc:
            row['exception'] = repr(exc)
            failure('interval_exception', transition_id=index, error=repr(exc))
        row['seconds'] = time.perf_counter()-tick
        return row

    def second_decision(observed, family, target):
        counts['second_decision_requests'] += 1
        key = (family, key_state(observed), float(target))
        if key in cached_decisions:
            counts['second_decision_cache_hits'] += 1
            return report['second_decisions'][cached_decisions[key]]
        map_key = (family, key_state(observed))
        if map_key not in cached_maps:
            cached_maps[map_key] = e.one_step(observed, arguments[family])
            counts['controller_one_step_maps'] += 1
            counts['controller_map_transitions'] += cached_maps[map_key]['transitions']
        else:
            counts['controller_map_cache_hits'] += 1
        mapping = cached_maps[map_key]
        # Only observed state, model family, target and remaining horizon enter.
        choice = policies.decision(observed, arguments[family], target, 1, 'recovery_economic', mapping)
        values = policies.one_values(mapping, target)
        selected = choice['index']
        index = len(report['second_decisions'])
        row = finite(dict(id=index, family=family, target=target, observed_log_state=observed,
                          choice=choice, selected_worst_error=values['errors'][selected],
                          selected_worst_biomass=values['biomass'][selected], selected_physical=values['valid'][selected],
                          robust_state_recoverable=bool(len(values['safe']))))
        cached_decisions[key] = index
        report['second_decisions'].append(row)
        return row

    for root_id, root in roots.items():
        tick = time.perf_counter()
        tree = e.tree(np.asarray(root['state']), arguments[root['family']])
        tree_path = ROOT / 'results/refined_target_verification' / ('tree41_' + root_id + '.npz')
        tree_path.parent.mkdir(parents=True, exist_ok=True)
        save_tree(tree_path, tree)
        report['trees'].append(dict(root=root_id, path=str(tree_path.relative_to(ROOT)), sha256=sha(tree_path),
                                    transitions=tree['transitions'], seconds=time.perf_counter()-tick))
        for case in [c for c in protocol['cases'] if c['root'] == root_id]:
            problem = dict(case, completed=False)
            problem_episodes = []
            try:
                choice = policies.decision(np.asarray(root['state']), arguments[root['family']], case['target'],
                                           2, 'recovery_economic', tree)
                problem['first_choice'] = finite(choice)
                if case['cohort'] == 'component_midpoints' and not choice['recoverable']:
                    failure('declared_component_midpoint_not_recoverable', case=case['id'])
            except Exception as exc:
                choice = None
                problem['exception'] = repr(exc)
                failure('first_decision_exception', case=case['id'], error=repr(exc))
            for i, j in protocol['paths']:
                episode = dict(case=case['id'], cohort=case['cohort'], root=root_id, target=case['target'], path=[i, j],
                               complete=False, physical_valid=False, inner_feasible=False, outer_feasible=False,
                               actual_terminal_feasible=False, operational_success=False)
                try:
                    if choice is None:
                        raise RuntimeError('First decision unavailable')
                    first = transition(independent.from_log(root['state']), float(choice['feed']), e.K[i], root['family'])
                    episode['first_transition_id'] = first['id']
                    if not first['completed']:
                        raise RuntimeError('First matched transition incomplete')
                    current = np.asarray(first['independent']['amount_state'])
                    decision = second_decision(independent.to_log(current), root['family'], case['target'])
                    episode['second_decision_id'] = decision['id']
                    second = transition(current, float(decision['choice']['feed']), e.K[j], root['family'])
                    episode['second_transition_id'] = second['id']
                    if not second['completed']:
                        raise RuntimeError('Second matched transition incomplete')
                    y = np.asarray(second['independent']['amount_state'])
                    py = np.asarray(second['engine']['amount_state'])
                    z = float(np.log(y[1]/y[2]))
                    error = abs(z - np.log(case['target']))
                    predicted_error = abs(float(np.log(py[1]/py[2])) - np.log(case['target']))
                    valid = bool(first['independent']['physical_valid'] and second['independent']['physical_valid'])
                    predicted_valid = bool(first['engine']['physical_valid'] and second['engine']['physical_valid'])
                    labels = dict(inner_feasible=bool(valid and error <= thresholds['inner'] + thresholds['comparison_tolerance']),
                                  outer_feasible=bool(valid and error <= thresholds['outer']),
                                  actual_terminal_feasible=bool(valid and error <= thresholds['actual']))
                    predicted_labels = dict(inner_feasible=bool(predicted_valid and predicted_error <= thresholds['inner'] + thresholds['comparison_tolerance']),
                                            outer_feasible=bool(predicted_valid and predicted_error <= thresholds['outer']),
                                            actual_terminal_feasible=bool(predicted_valid and predicted_error <= thresholds['actual']))
                    for field in labels:
                        if labels[field] != predicted_labels[field]:
                            failure('matched_' + field + '_label_disagreement', case=case['id'], path=[i, j],
                                    independent_error=error, matched_engine_error=predicted_error)
                    episode.update(complete=True, physical_valid=valid, z=z, error=error,
                                   biomass_gDW=float(y[1]+y[2]), terminal_amount_state=y.tolist(),
                                   min_S=min(first['independent']['min_S'], second['independent']['min_S']),
                                   feeds=[float(choice['feed']), float(decision['choice']['feed'])],
                                   matched_engine_error=predicted_error, matched_engine_labels=predicted_labels,
                                   operational_success=labels['actual_terminal_feasible'], **labels)
                    if case['cohort'] == 'component_midpoints' and not labels['inner_feasible']:
                        report['witness_qualifications'].append(dict(case=case['id'], path=[i, j],
                                                                      code='midpoint_inner_band_failure', error=float(error)))
                    if case['cohort'] == 'component_midpoints' and not labels['outer_feasible']:
                        failure('midpoint_outer_band_failure', case=case['id'], path=[i, j], error=error)
                except Exception as exc:
                    episode['exception'] = repr(exc)
                    failure('episode_exception', case=case['id'], path=[i, j], error=repr(exc))
                report['episodes'].append(episode)
                problem_episodes.append(episode)
            problem.update(completed=all(r['complete'] for r in problem_episodes),
                           all_paths_inner_feasible=all(r['inner_feasible'] for r in problem_episodes),
                           all_paths_outer_feasible=all(r['outer_feasible'] for r in problem_episodes),
                           all_paths_actual_feasible=all(r['actual_terminal_feasible'] for r in problem_episodes),
                           maximum_error=max((r['error'] for r in problem_episodes if r['complete']), default=None),
                           minimum_biomass=min((r['biomass_gDW'] for r in problem_episodes if r['complete']), default=None))
            report['problems'].append(problem)
            if len(report['problems']) % 10 == 0:
                report['counts'] = dict(counts)
                save(RESULTS, report)
                print('Verified %d/197 problems (%d episodes)' % (len(report['problems']), len(report['episodes'])), flush=True)
        save(RESULTS, report)
        print('Completed grid-41 root', root_id, flush=True)
    preserved = all(sha(ROOT / p) == h for p, h in protocol['source_input_sha256'].items())
    report['source_input_hashes_preserved'] = preserved
    report['complete'] = bool(preserved and len(report['problems']) == 197 and len(report['episodes']) == 1773)
    report['all_episodes_completed'] = all(r['complete'] for r in report['episodes'])
    report['passed'] = report['complete'] and report['all_episodes_completed'] and not report['failures'] and not report['witness_qualifications']
    report['counts'] = dict(counts)
    report['elapsed_seconds'] = time.perf_counter()-started
    report['cohort_summary'] = {}
    for cohort in ('component_midpoints', 'fixed_targets'):
        problems = [r for r in report['problems'] if r['cohort'] == cohort]
        episodes = [r for r in report['episodes'] if r['cohort'] == cohort]
        report['cohort_summary'][cohort] = dict(problems=len(problems), episodes=len(episodes),
                                              completed_episodes=sum(r['complete'] for r in episodes),
                                              physical_valid_episodes=sum(r['physical_valid'] for r in episodes),
                                              inner_feasible_episodes=sum(r['inner_feasible'] for r in episodes),
                                              outer_feasible_episodes=sum(r['outer_feasible'] for r in episodes),
                                              actual_feasible_episodes=sum(r['actual_terminal_feasible'] for r in episodes),
                                              all_paths_inner_feasible_problems=sum(r['all_paths_inner_feasible'] for r in problems),
                                              all_paths_outer_feasible_problems=sum(r['all_paths_outer_feasible'] for r in problems),
                                              all_paths_actual_feasible_problems=sum(r['all_paths_actual_feasible'] for r in problems))
    save(RESULTS, report)
    print(json.dumps(dict(complete=report['complete'], passed=report['passed'],
                         failures=len(report['failures']), witness_qualifications=len(report['witness_qualifications']), cohort_summary=report['cohort_summary'],
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
